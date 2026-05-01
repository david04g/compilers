# Custom Register Allocator Design

## Pipeline Position

`custom-ra` is a legacy CodeGen `MachineFunctionPass` registered through
`RegisterRegAlloc`, so `llc -regalloc=custom-ra` can select it in the normal
backend allocation slot.

## Analysis Flow

1. Compute diagnostic block live-in/live-out sets from MachineOperand virtual
   register uses and defs.
2. Collect virtual register statistics from `LiveIntervalsWrapperPass`,
   `MachineLoopInfoWrapperPass`, and `MachineRegisterInfo`.
3. Build an interference graph by checking overlap between virtual register
   live intervals.
4. Sort virtual registers by a deterministic priority derived from loop-weighted
   uses, defs, live-range length, spill cost, and split candidacy.
5. Query `LiveRegMatrix` for each target-allocatable physical register in the
   virtual register's class, rejecting candidates that conflict with assigned
   virtual registers, fixed physical live ranges, or call regmasks.
6. Assign legal registers through `LiveRegMatrix::assign`, which updates
   `VirtRegMap` and the matrix consistently for `virtregrewriter`.
7. If no physical register is available, report a fatal unsupported-spill
   diagnostic after writing trace output; v1 intentionally does not emit
   incomplete stack-slot-only spill mappings.

## Debug Outputs

- `-custom-ra-debug` writes allocation traces to stderr.
- `-custom-ra-trace=<path>` appends block liveness, virtual register statistics,
  graph size, and allocation decisions using target register names.
- `-custom-ra-dot=<path>` writes a Graphviz DOT interference graph.
- `-custom-ra-annotate-mir` adds MIR-oriented allocation annotations to the trace
  without mutating MIR text.

## Prototype Boundaries

The pass performs deterministic physical-register selection and records
spill-required and split-candidate events. It does not claim spill reduction
until real spill insertion/splitting is implemented and benchmarked. Full
production-quality rematerialization, eviction, aggressive interval splitting,
and iterative spill-code repair should be added after the lit tests run against
a concrete LLVM installation.
