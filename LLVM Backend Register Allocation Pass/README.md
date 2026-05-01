# Custom LLVM Register Allocator

This is an out-of-tree LLVM backend register allocation prototype. It registers a
MachineFunctionPass as `custom-ra` and is intended to be loaded into `llc`.

## Build

```powershell
cmake -S . -B build -DLLVM_DIR=<path-to-llvm-lib-cmake-llvm>
cmake --build build
```

LLVM, CMake, Clang, and lit are external prerequisites and are not vendored in
this repository.

## Run

```powershell
llc -load .\build\CustomRegAlloc.dll -regalloc=custom-ra input.mir
```

Useful diagnostics:

```powershell
llc -load .\build\CustomRegAlloc.dll -regalloc=custom-ra `
  -custom-ra-debug `
  -custom-ra-dot=graph.dot `
  -custom-ra-trace=trace.txt `
  input.mir
```

## Current Scope

The first milestone is a deterministic X86-oriented allocator prototype. It
uses LLVM CodeGen analyses for live intervals, register classes, reserved
registers, virtual register mapping, and `LiveRegMatrix` legality checks.

This version is correctness-first: it assigns only virtual registers that have a
legal physical register according to LLVM's live register matrix. If allocation
would require spilling or live-range splitting, it emits a trace entry and stops
with a fatal diagnostic instead of producing invalid MIR. Full spill insertion,
eviction, and live-range splitting remain future work that should be built
against a concrete LLVM tree and validated with `-verify-machineinstrs`.
