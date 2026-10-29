# Custom LLVM Register Allocator

This is an out-of-tree LLVM backend register allocation prototype. It registers a
MachineFunctionPass as `custom-ra` and is intended to be loaded into `llc`.

## Build

### Windows prerequisites

Use the MSYS2 UCRT64 LLVM packages so `llc`, `FileCheck`, LLVM headers, and
`LLVMConfig.cmake` all come from the same toolchain:

```powershell
winget install -e --id MSYS2.MSYS2
```

Then install the build dependencies from an MSYS2 shell or through PowerShell:

```powershell
& 'C:\msys64\usr\bin\bash.exe' -lc 'pacman -Syu --noconfirm'
& 'C:\msys64\usr\bin\bash.exe' -lc 'pacman -S --needed --noconfirm mingw-w64-ucrt-x86_64-llvm mingw-w64-ucrt-x86_64-llvm-tools mingw-w64-ucrt-x86_64-clang mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja mingw-w64-ucrt-x86_64-python-pip'
```

`lit` is not packaged with the UCRT64 LLVM tools, so create a local virtual
environment for it:

```powershell
& 'C:\msys64\usr\bin\bash.exe' -lc 'export PATH=/ucrt64/bin:/usr/bin:$PATH; python -m venv .venv-lit; .venv-lit/bin/python -m pip install lit'
```

### Configure and compile

```powershell
& 'C:\msys64\usr\bin\bash.exe' -lc 'export PATH=/ucrt64/bin:/usr/bin:$PATH; cmake -S . -B build -G Ninja -DLLVM_DIR=C:/msys64/ucrt64/lib/cmake/llvm -DCMAKE_C_COMPILER=C:/msys64/ucrt64/bin/clang.exe -DCMAKE_CXX_COMPILER=C:/msys64/ucrt64/bin/clang++.exe -DLLVM_EXTERNAL_LIT=$PWD/.venv-lit/bin/lit'
& 'C:\msys64\usr\bin\bash.exe' -lc 'export PATH=/ucrt64/bin:/usr/bin:$PATH; cmake --build build'
```

LLVM, CMake, Clang, and lit are external prerequisites and are not vendored in
this repository.

## Test

```powershell
& 'C:\msys64\usr\bin\bash.exe' -lc 'export PATH=/ucrt64/bin:/usr/bin:$PATH; cmake --build build --target check-custom-ra'
```

## Run

```powershell
& 'C:\msys64\usr\bin\bash.exe' -lc 'export PATH=/ucrt64/bin:/usr/bin:$PATH; llc -load build/CustomRegAlloc.dll -regalloc=custom-ra input.mir'
```

Useful diagnostics:

```powershell
& 'C:\msys64\usr\bin\bash.exe' -lc 'export PATH=/ucrt64/bin:/usr/bin:$PATH; llc -load build/CustomRegAlloc.dll -regalloc=custom-ra -custom-ra-debug -custom-ra-dot=graph.dot -custom-ra-trace=trace.txt input.mir'
```

## Current Scope

The first milestone is a deterministic X86-oriented allocator prototype. It
uses LLVM CodeGen analyses for live intervals, register classes, reserved
registers, virtual register mapping, and `LiveRegMatrix` legality checks.

This version supports a simple, traceable stack-spilling policy: allocate until
a value does not fit, spill that value with LLVM's InlineSpiller, recompute
liveness and interference, and retry. Spill-generated unspillable temporaries
receive highest priority. The trace records each value, stack slot, replacement
temporaries, and the final number of rounds. LLVM handles target-specific spill
instructions, memory operand folding, and optional rematerialization.

This policy favors clarity over performance. It does not implement custom
eviction or region-based splitting and can fail on unspillable intervals; a
10,000-round limit prevents endless retries. Tested with LLVM 23.1.4 on Windows
x86-64. See `docs/design.md` and `docs/benchmark-report.md` for details.

## Executable correctness and performance tests

In addition to the MIR/lit suite, run:

```text
python scripts/test_spilling.py --llc /path/to/llc --clang /path/to/clang --plugin /path/to/CustomRegAlloc.dll --output spill-results --repeats 5
```

This generates 11 workloads, compares Greedy and custom-ra against independent
expected results, verifies actual stack accesses, and saves traces, rewritten
MIR, assembly, raw measurements, and native execution output. Coverage includes
register pressure, calls, branches, loops, 32-bit wraparound, and floating-point
values. Compile timing disables trace output. The synthetic benchmarks show a
correctness improvement over the previous rejection behavior, not a speedup
over LLVM Greedy.
