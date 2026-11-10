# Milestones and extension checklist

## 1. Make a split plan run — implemented

- [x] Parse and validate a fixed-shape linalg/tensor graph in C++.
- [x] Emit fixed tiles and placements with explicit load/store dependencies.
- [x] Execute SRAM buffers and tiled reductions in the NumPy-backed simulator.
- [x] Check M/N/K partial edges against an independent whole-tensor reference.
- [x] Reject malformed/nonmatching MLIR instead of guessing its semantics.

Entry point: `examples/edge.mlir`, `strategy=split tiles=8x8x8 placement=compact`.
Explain one edge tile's clipped load, partial K accumulation, and output store.

## 2. Make fusion a compiler decision — implemented for the first three stages

- [x] Enumerate all two-boundary partitions and a small tile set.
- [x] Allocate/reuse SRAM by lifetime and reject per-bank capacity violations.
- [x] Search compact/spread placement and emit on-chip transfer routes.
- [x] Rank using estimated traffic/resource pressure; retain every diagnostic.
- [x] Execute the optional second matmul through a materialized boundary.

Next compiler exercises:

- [ ] Search one versus two streaming slots instead of requiring two for fusion.
- [ ] Search bank assignment and placement beyond the two fixed policies.
- [ ] Add subgraph matching/canonicalization for equivalent GELU expressions.
- [ ] Fuse into the second matmul with partial-reduction buffers and explicit
  dependence on all contributing hidden-dimension tiles. Do not apply GELU before
  the first matmul's reduction is complete.
- [ ] Replace explicit event expansion with reusable tile-loop regions or a small
  custom plan dialect, while retaining the JSON emitter for debugging.

## 3. Measure where the decision works — implemented baseline

- [x] Sweep shapes and per-bank SRAM budgets with fixed seed and hardware rates.
- [x] Compare heuristic, always-split, always-fuse when legal, and exhaustive search.
- [x] Numerically verify every legal candidate before accepting its metrics.
- [x] Report latency, off-chip bytes, peak SRAM, legality, and heuristic regret.
- [x] Include the same-graph budget-switch demo and a counterexample to the heuristic.
- [x] Export resource timelines and explain candidate rejection/stalls.

Next experiments:

- [ ] Sweep DRAM/link bandwidth and GELU throughput independently of SRAM.
- [ ] Compare the current estimate to a slot-aware recurrence or sampled scheduler.
- [ ] Add hardware-calibrated rates only after separating simulator claims from
  measurements on a real target.
- [ ] Evaluate cache/reuse, multicast, and split regions with per-tile overlap as
  additional candidate families; rerun all baselines over the enlarged set.

Before claiming an improvement, preserve the candidate set, scheduler, random seed,
and numerical tolerance in the experiment output. Report cases where the estimate
loses as well as cases where it wins.
