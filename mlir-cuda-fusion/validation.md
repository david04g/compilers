# Validation record

Local validation on 2026-11-13:

- NVIDIA GeForce GTX 1080, compute capability 6.1, 8192 MiB; driver 582.66.
- Windows; MSYS2 UCRT64 Clang 22.1.4; CMake/Ninja.
- LLVM/MLIR 24.0.0git, source revision `ed01afe02dc4feaa72e34e462a874d4067b58f41`.
- Python 3.13.9, NumPy 2.5.4, `nvidia-cuda-nvrtc-cu12==12.9.86`.
- C++ compiler configured, compiled, linked, and invoked for every experiment.
- Seven unittest methods, including native compiler rejection tests and real CUDA
  integration tests. GPU checks are mandatory when `FUSION_REQUIRE_CUDA=1`.
- GPU correctness cases: 1×1, 7×257, 3×1003, and 128×1024; every default schedule,
  random seeds 11 and 23, plus all-zero activations. Output tails are guarded.
- Full benchmark: eight shapes, 90 admissible candidates, seed 2026 + shape index,
  three tuning samples, eleven evaluation samples, 64 invocations per graph.

The raw measured report is committed under `results/`. Its timestamps are UTC.
Source hashes and compile options identify each generated kernel; machine-local
paths are not needed to reproduce it. CI builds against LLVM/MLIR 19 and runs
native tests; hosted runners without a GPU explicitly skip GPU tests.

The 8×32771 and 4×16384 cases expose inaccurate ranking. They are retained as
research findings. Passing correctness tests does not imply the analytical model
has met its performance objective on every shape.
