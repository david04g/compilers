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

This version is correctness-first: it assigns only virtual registers that have a
legal physical register according to LLVM's live register matrix. If allocation
would require spilling or live-range splitting, it emits a trace entry and stops
with a fatal diagnostic instead of producing invalid MIR. Full spill insertion,
eviction, and live-range splitting remain future work that should be built
against a concrete LLVM tree and validated with `-verify-machineinstrs`.
