# Memory-aware MLIR fusion on a GTX 1080

A working prototype for this research question:

> Can a compiler choose fusion boundaries and thread tiles that reduce inference
> latency without excessive live storage or repeated memory access?

The initial workload is **residual addition followed by RMS normalization**:

```text
s = float32(x + residual)
y[row, col] = s[row, col] * rsqrt(mean(s[row, :]²) + epsilon) * gamma[col]
```

A C++ MLIR pass verifies the graph, enumerates schedules, estimates their cost,
and rewrites the graph to `fusion.execute`. A small Python backend emits CUDA C++
for those schedules. NVRTC 12.9 compiles it to real GPU machine code, and the CUDA
Driver API executes and times it on the GPU. **This is not a timing simulator.**

The backend uses explicit templates for this one graph. It does not implement a
general `linalg` → `gpu` → NVVM lowering, a PyTorch importer, or Tensor Core codegen.
The separate [dataflow project](../mlir-dataflow/) explores spatial scheduling in
a simulator; this project supplies complementary measurements on actual hardware.

## Objectives and acceptance criteria

| Objective | Acceptance evidence |
|---|---|
| Preserve the supported graph's semantics | Every emitted schedule agrees with an independent NumPy reference at `rtol=2e-5`, `atol=2e-5`; test zero inputs, two random seeds, partial tiles, and output guards |
| Make fusion a compiler decision | Native MLIR pass enumerates four execution strategies and multiple thread tiles, emits a ranked JSON plan, and rewrites the IR |
| Respect explicit resource budgets | Reject schedules exceeding the global-scratch limit or estimated register budget; diagnose an empty search space |
| Limit online measurement effort | Choose among at most `--budget` analytically ranked candidates using separate tuning samples |
| Evaluate prediction quality honestly | Compare selected latency with the best observed enumerated candidate; publish regret, failures, compile cost, and raw samples |
| Reproduce hardware results | Document toolchain, seeds, shapes, timing method, generated source hashes, and compiled kernel resources |

The aspirational performance objective is **within 10% of the best enumerated
candidate with a three-candidate tuning budget**. The first measured run achieves
this in **6 of 8 cases**. It is not yet a generally accurate predictor. Performance
thresholds are research metrics, not flaky unit-test assertions.

## Measured results

See [the full report](results/report.md), [raw JSON](results/metrics.json), and
[per-candidate CSV](results/candidates.csv). These measurements were collected on
an NVIDIA GeForce GTX 1080 (8 GiB, `sm_61`) using NVRTC 12.9.

| Shape | Selected median | Tuned split median | Speedup |
|---|---:|---:|---:|
| 256 × 256 | 3.296 µs | 8.400 µs | 2.55× |
| 128 × 1024 | 4.144 µs | 11.840 µs | 2.86× |
| 32 × 4096 | 4.640 µs | 12.944 µs | 2.79× |
| 1 × 65536 | 41.744 µs | 41.744 µs | 1.00× |

All **90 benchmark candidates** passed numerical and output-guard checks; maximum
absolute error was **9.54e-7**. Worst selection regret was **88.6%**, at 8 × 32771.
The report includes both large misses. The 1 × 65536 case favors a materialized
split within the tested candidate set, demonstrating that fusion is not uniformly
beneficial. These are workload-block timings, not whole-model speedups.

## Build and run

Requirements: Python 3.10+, NumPy, CMake, Ninja, a C++17 compiler, and a matched
LLVM/MLIR development installation. GPU execution additionally requires an NVIDIA
driver and NVRTC. The optional dependency below supplies NVRTC without installing
the full CUDA toolkit or changing your driver.

```sh
python -m venv .venv
# Linux: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e '.[cuda]'
```

For a GTX 1080, keep NVRTC at **12.9 or earlier**; the CUDA 13 compiler no longer
targets Pascal. The `[cuda]` extra pins `nvidia-cuda-nvrtc-cu12==12.9.86`. You can
instead set `NVRTC_LIBRARY` to an existing compatible library's absolute path.
Current Nsight Compute and Triton are not prerequisites.

On Ubuntu 24.04, the CI configuration uses packaged LLVM/MLIR 19:

```sh
sudo apt-get update
sudo apt-get install -y ninja-build llvm-19-dev libmlir-19-dev mlir-19-tools
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=/usr/lib/llvm-19/lib/cmake/mlir
cmake --build build -j 2
ctest --test-dir build --output-on-failure
```

The local Windows build used MSYS2 UCRT64 Clang and an existing LLVM/MLIR build.
Use a compiler with the same ABI as the LLVM libraries:

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DMLIR_DIR=C:/path/to/llvm-build/lib/cmake/mlir `
  -DCMAKE_PREFIX_PATH=C:/msys64/ucrt64 `
  -DCMAKE_C_COMPILER=C:/msys64/ucrt64/bin/clang.exe `
  -DCMAKE_CXX_COMPILER=C:/msys64/ucrt64/bin/clang++.exe `
  -DPython3_EXECUTABLE=C:/path/to/venv/Scripts/python.exe
cmake --build build -j 2
$env:FUSION_REQUIRE_CUDA = '1'
ctest --test-dir build --output-on-failure
```

`FUSION_REQUIRE_CUDA=1` makes an unavailable CUDA environment a test failure.
Without it, the two GPU integration tests explicitly skip when CUDA/NVRTC is
unavailable. CTest sets `FUSION_OPT` and `PYTHONPATH`; a direct unittest run needs
those variables. Seven test methods cover compiler, backend, and GPU behavior.

Run a quick experiment (use `build/fusion-opt.exe` on Windows):

```sh
python -m cuda_fusion.benchmark --compiler build/fusion-opt \
  --shapes 7x257 --samples 5 --repetitions 16 --output out/quick
```

Reproduce the checked-in eight-shape sweep:

```sh
python -m cuda_fusion.benchmark --compiler build/fusion-opt \
  --samples 11 --repetitions 64 --budget 3 --seed 2026 --output out/benchmark
```

Each shape directory contains the input MLIR, rewritten MLIR, candidate plan, and
generated CUDA sources. The top-level directory contains the report and raw data.
GPU errors or numerical mismatches fail the command; it never substitutes CPU
timings or estimates for GPU results.

To inspect just the compiler, create `out/` and run:

```sh
build/fusion-opt examples/tail.mlir \
  '--fusion-plan=output=out/plan.json threads=64,128,256 scratch-limit=134217728 register-limit=255' \
  -o out/planned.mlir
```

## Implementation and scope

| Component | Location |
|---|---|
| Registered MLIR operations, verification, analytical planner, IR rewrite | `src/fusion-opt.cpp` |
| Input generation and native compiler invocation | `python/cuda_fusion/frontend.py` |
| CUDA template backend for each candidate | `python/cuda_fusion/codegen.py` |
| NVRTC and CUDA Driver bindings, resource queries, graph timing | `python/cuda_fusion/cuda.py` |
| Correctness, bounded selection, independent evaluation, JSON/CSV/report | `python/cuda_fusion/benchmark.py` |
| Focused native and GPU tests | `tests/test_fusion.py` |

The supported input is one function with exactly `fusion.add` → `fusion.rms_norm`
→ return, three arguments, positive static FP32 dimensions, and positive finite
FP32 epsilon. `gamma` has one value per column. Maximum shape: 65535 rows, 131072
columns, and 16777216 elements. Extra consumers, dynamic shapes, other operations,
and alternate wiring are rejected rather than silently miscompiled.

Inputs must have finite, non-overflowing FP32 additions and squared reductions.
Floating-point reduction reassociation is allowed within the stated tolerance.
The reference rounds the residual addition to FP32, then accumulates the RMS
reduction in FP64. `--fmad=false` is used and fast math is not enabled.

See [design and measurement details](design.md) and [validation record](validation.md).

## Next research experiments

1. Add a shortlist-diversity policy and evaluate it on a new, predeclared shape set.
   The current model can exclude a good register-cached candidate on wide rows.
2. Calibrate bandwidth/parallelism estimates on separate microbenchmarks and test
   generalization on held-out shapes. Keep the current results as a baseline.
3. Use compiled resource feedback to refine ranking before timing; account for the
   additional compilation cost explicitly.
4. Import equivalent standard `linalg` graphs and replace template emission with
   GPU/NVVM lowering. Add a second workload only after that path is validated.
5. Evaluate on a newer NVIDIA GPU or map the scheduling analysis to the existing
   spatial simulator. Retain separate evidence for each target.
