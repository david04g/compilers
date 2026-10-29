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
7. If no physical register is available, unassign the partial plan from the
   live register matrix and use `LiveRangeEdit` with LLVM's InlineSpiller to
   spill that interval. This updates instructions, stack slots, and liveness.
8. Record the original value, frame index (or rematerialization), and newly
   created temporaries. Rebuild the analysis and allocation plan from scratch.
   Unspillable short temporaries get highest priority on the next round.
9. Once allocation succeeds, run the spiller's cleanup and remove dead
   rematerialized definitions before LLVM's normal register rewriting.

## Debug Outputs

- `-custom-ra-debug` writes allocation traces to stderr.
- `-custom-ra-trace=<path>` appends block liveness, virtual register statistics,
  graph size, and allocation decisions using target register names.
- `-custom-ra-dot=<path>` writes a Graphviz DOT interference graph.
- `-custom-ra-annotate-mir` adds MIR-oriented allocation annotations to the trace
  without mutating MIR text.

## Prototype Boundaries

The pass uses a deterministic spill-current-value policy, not a custom eviction
or optimal splitting algorithm. The graph is rebuilt after every spill instead
of incrementally repaired; this is easy to follow but adds compile work. LLVM's
spiller may fold memory operands and rematerialize values, so not every spill
event must create a new stack slot. Traces distinguish these cases.

An unspillable interval that cannot fit produces a fatal diagnostic; the pass
does not silently discard values. A 10,000-round limit bounds retrying. The
initial priority formula still includes a split-candidate bonus, which is only
a heuristic; no custom region splitting is implemented. Local microbenchmarks
show more stack traffic and slower generated code than LLVM Greedy. The current
milestone is correct spill insertion with inspectable output.
