# MLIR compiler for a simulated dataflow accelerator

An executable backend starter for investigating **when fusion is profitable or legal**.
The C++ MLIR pass accepts `linalg`/`tensor` IR and emits a versioned JSON hardware
execution plan. A separate Python simulator schedules that plan, executes its
transfers and kernels through physical SRAM offsets, and checks its output against
a whole-tensor NumPy reference.

The initial three milestones work end to end for `matmul → bias → tanh GELU`.
A second matmul also runs, with a materialized boundary before it; fusing across
that projection is an extension exercise. This is a deliberately small simulator
target, not a model of SambaNova hardware.

## What is here

| Component | Implementation |
|---|---|
| MLIR recognition | C++ module pass; validates static f32 shapes, SSA connections, indexing maps, zero initialization, and the exact scalar bias/GELU bodies |
| Fusion decision | Enumerates the four partitions of the first three stages and ranks legal plans by estimated resource load plus pipeline fill/drain |
| Tiling | Four default M/N/K tile triples, with clipped M, N, and reduction edges |
| Memory planning | Per-bank 16-byte-aligned linear-scan allocation, lifetime reuse, explicit streaming slots, capacity rejection |
| Placement | Compact regions or spread stages on a configurable grid; local SRAM banks and deterministic XY routes |
| Simulation | Dependency scheduling, CU/bank/DRAM/link contention, physical buffer reads/writes, numerical comparison, stall attribution, Perfetto trace export |
| Experiments | Shape/SRAM sweep against always-split, always-fuse when legal, and exhaustive simulation over the same candidates |

## Try the simulator without building LLVM

Use Python 3.10+ and a virtual environment. From this directory:

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m dataflow.simulator examples/edge.split.plan.json --explain
python -m unittest discover -s tests -v
```

The checked-in plan was emitted by the C++ compiler. It uses fixed 8×8×8 tiles on
`A[17,19] @ B[19,13] + bias[13]`, including partial tiles in all three dimensions.
The simulator-only run executes seven tests and explicitly skips native compiler
tests until `DATAFLOW_OPT` is set. This route does not parse or compile MLIR.

## Build the C++ compiler

The CI toolchain is LLVM/MLIR 19 on Ubuntu 24.04. Headers, libraries, CMake metadata,
and the compiler ABI must come from the same LLVM build. The project was also
built locally with LLVM 24.0.0git; see [validation details](docs/validation.md).

```sh
sudo apt-get update
sudo apt-get install -y cmake ninja-build llvm-19-dev libmlir-19-dev mlir-19-tools
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=/usr/lib/llvm-19/lib/cmake/mlir
cmake --build build -j 2
ctest --test-dir build --output-on-failure
```

Alternatively point `MLIR_DIR` at your existing build's `lib/cmake/mlir`. For an
MSYS2 UCRT64 build on Windows, use matching UCRT64 compilers and dependencies:

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DMLIR_DIR=C:/path/to/llvm-build/lib/cmake/mlir `
  -DCMAKE_PREFIX_PATH=C:/msys64/ucrt64 `
  -DCMAKE_C_COMPILER=C:/msys64/ucrt64/bin/clang.exe `
  -DCMAKE_CXX_COMPILER=C:/msys64/ucrt64/bin/clang++.exe
cmake --build build -j 2
ctest --test-dir build --output-on-failure
```

NumPy must be installed in the Python selected by CMake; pass
`-DPython3_EXECUTABLE=/path/to/python` if needed. CTest sets `DATAFLOW_OPT` and
`PYTHONPATH` and runs the full native integration suite.

## Milestone 1: a fixed split plan

Create `out/` first (`mkdir -p out` in Bash or `New-Item -ItemType Directory -Force out`
in PowerShell). Use `build/dataflow-opt.exe` on Windows.

```sh
build/dataflow-opt examples/edge.mlir \
  '--dataflow-plan=output=out/split.json strategy=split tiles=8x8x8 placement=compact' \
  -o out/planned.mlir
python -m dataflow.simulator out/split.json --metrics out/split.metrics.json --explain
```

The MLIR output retains the source graph and records `dataflow.plan_file`; the
lowered execution representation is the JSON file. Inspect `regions`, `buffers`,
and `events` to follow individual compute and transfer commands.

## Milestone 2: let the compiler choose

```sh
build/dataflow-opt examples/edge.mlir \
  '--dataflow-plan=output=out/selected.json hardware=configs/default.json emit-all=true' \
  -o out/planned.mlir
python -m dataflow.simulator out/selected.json --explain --trace out/trace.json
```

`--explain` prints rejected candidates and resource stalls. Open `out/trace.json`
in Perfetto or a Chrome trace viewer. Trace timestamps represent modeled cycles;
they are not calibrated nanoseconds. `emit-all=true` includes executable bodies
for every legal candidate, which enables exhaustive comparison.

For the second matmul, substitute `examples/ffn.mlir`. Generate other shapes with:

```sh
python -m dataflow.generate --m 33 --k 31 --n 29 --p 17 -o out/ffn.mlir
```

## Milestone 3: reproduce the decision and measure misses

```sh
python -m dataflow.experiments --compiler build/dataflow-opt --output out/sweep
python -m dataflow.experiments --compiler build/dataflow-opt \
  --shapes 17x19x13 --sram 512,16384 --save-plans --output out/demo
```

Both demo runs use **exactly the same MLIR graph**. Only per-bank SRAM capacity
changes. The checked-in [measured report](results/report.md) shows:

| Configuration | Selected plan | Simulated cycles | Off-chip bytes | Peak reserved SRAM |
|---|---|---:|---:|---:|
| 16,384 bytes/bank | fused, compact, 32×16×16 | 458 | 3,216 | 5,632 |
| 512 bytes/bank | split, compact, 8×8×8 | 1,684 | 10,124 | 768 |

At the generous budget, always-split takes 754 cycles and transfers 6,752 bytes
using the same 32×16×16 tile. At the tight budget, even the smallest fused plan
requires 1,024 bytes in bank 0 of CU 0 for two streaming slots, exceeding 512.
The compiler prints that exact rejection and selects a legal split candidate.

Across the default 16 cases, the heuristic matches exhaustive simulation in
13 cases, with 12.9% worst regret. The report keeps those misses visible. All legal
candidates are numerically verified, including the candidates the heuristic loses to.
`sweep.csv` and `sweep.json` contain the raw metrics.

## Scope and next work

The search is intentionally bounded: four tiles × four partitions × two placement
policies, with fixed bank assignment. Fused regions reserve two ping-pong slots;
split stages use one slot and a full region barrier. Even a one-tile fused region
reserves both slots. Thus “fusion is illegal” means illegal **within this candidate
grammar**; it does not prove that a serialized one-slot fused implementation cannot
fit. Adding slot count as a search dimension is a useful next experiment.

The target has no real ISA, cache, bank interleaving, multicast, hardware DMA queue,
or calibrated power/timing model. Input tiles are reloaded for each output tile.
All routes reserve their links for the whole transfer. GELU is the tanh approximation,
not erf GELU. The second matmul always starts a new region. There is no general graph
frontend, dynamic-shape support, or MLIR-to-native machine-code lowering.

Start with [the milestone checklist](docs/milestones.md),
[the input contract](docs/input-contract.md), and [the plan/model design](docs/design.md).

## Upstream references

The compiler uses MLIR's [pass infrastructure](https://mlir.llvm.org/docs/PassManagement/),
[Linalg dialect](https://mlir.llvm.org/docs/Dialects/Linalg/), and
[standalone build pattern](https://github.com/llvm/llvm-project/tree/main/mlir/examples/standalone).
Ubuntu's [MLIR 19 development package](https://packages.ubuntu.com/noble/libmlir-19-dev)
provides the CI headers and libraries.
