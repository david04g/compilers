# Validation record

Initial local validation on 2026-11-10:

- Windows, MSYS2 UCRT64 Clang 22.1.4, CMake/Ninja.
- LLVM/MLIR 24.0.0git, source commit `bdee1ad45596b867c18c39bfdcb299f7040296a3`.
- Python 3.13.14, NumPy 2.3.5.
- C++ `dataflow-opt` configured, compiled, linked, and used for all reported plans.
- CTest: 15 tests passed, including native compilation, all four fusion partitions,
  both placement policies, partial M/N/K tiles, the second projection, one-bank/
  one-CU hardware, negative MLIR cases, SRAM overflow/alias detection, and link contention.
- The default 16-case sweep executed every legal candidate against the independent
  NumPy reference at seed 0. The test suite additionally uses seeds 11 and 23.
- The checked-in `examples/edge.split.plan.json` is generated from `examples/edge.mlir`
  with `strategy=split tiles=8x8x8 placement=compact` and default hardware.

Reproduce:

```sh
cmake --build build
ctest --test-dir build --output-on-failure
python -m dataflow.experiments --compiler build/dataflow-opt --output out/sweep
```

The GitHub workflow separately builds the C++ executable with packaged LLVM/MLIR 19
on Ubuntu 24.04 and runs the same integration suite and sweep. Simulator tests also
run on Windows. The full raw report is in `results/sweep.json` and `results/sweep.csv`.
These are simulated-cycle results, not timings of compiler or simulator execution.
