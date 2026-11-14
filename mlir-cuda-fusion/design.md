# Design and measurement protocol

## Compiler boundary

The registered `fusion` dialect has three operations. `fusion.add` and
`fusion.rms_norm` describe the input graph; `fusion.execute` records a selected
schedule and preserves the original tensor result type. Verifiers enforce the
shape/dtype/epsilon contract. The pass additionally checks all SSA connections,
the return value, and single-use intermediates before erasing the input operations.

The JSON plan is the executable boundary between the native planner and the CUDA
template emitter. Both the plan and rewritten MLIR select the same initial
analytical candidate. The benchmark can subsequently select a different candidate
from the ranked shortlist using measurements; that choice is recorded in the
metrics, not retroactively written into the original analytical plan.

## Candidate space

Each candidate uses 64, 128, or 256 threads by default. One reduction block handles
one row. Column tiles are strided by the block width; tail columns contribute zero
to the reduction and are excluded from stores. The reduction uses shared memory
and unconditional block barriers. It has no warp-synchronous assumptions.

| Mode | Execution | Kernels | Intermediate global scratch | Logical FP32 bytes, S = rows × width |
|---|---|---:|---:|---:|
| cached | Add, retain a thread's values across reduction, normalize | 1 | 0 | 16S |
| stream | Add/reduce, then reread x and residual to normalize | 1 | 0 | 24S |
| partial | Add/reduce while writing the sum and row inverse RMS, then normalize in a flat grid | 2 | 4S + 4rows | 28S + 4rows |
| split | Flat addition, row reduction, flat normalization | 3 | 4S + 4rows | 32S + 4rows |

Logical bytes count source-level loads/stores, including a gamma load per element
and an inverse-RMS load per element for materialized normalization. Cache reuse,
compiler common-subexpression elimination, transactions, and spills make physical
DRAM traffic different. The values are explicitly **estimates**, never counters.

The driver allocates shared scratch for all candidates during the offline sweep.
The per-candidate scratch metric reports the storage that its schedule requires,
not the experiment process's total allocation. All schedules share input/output
buffers. Persistent device input/output storage is excluded from scratch bytes.

## Analytical model

The initial model assumes 20 SMs, 180 GB/s effective bandwidth, 3 µs per launch,
65536 registers/SM, and 96 KiB shared memory/SM. These are an uncalibrated
GTX-1080-oriented prior, not a fitted latency predictor or a portable hardware
description. Cached register demand is estimated as 24 + ceil(width / threads);
other modes use an estimate of 24 registers/thread. The model approximates
resident blocks, available row parallelism, and transfer cost.

The global scratch limit is exact for this candidate grammar. The register limit
is an estimate-based screening rule, not a proof that the machine-code allocator
will use that many registers or avoid spills. Rejected candidates remain in the
plan with explanations. An empty admissible search space is a compilation error.

The emitter compiles directly to cubin with NVRTC 12.9 and the actual device's
`sm_XX`, with FP32 and FMA contraction disabled. Runtime function attributes report
compiled registers and local storage. The CUDA occupancy API reports the
theoretical active-block limit. None is an achieved-occupancy or dynamic-memory
counter. A future GPU can execute the templates, but needs its own validated model.

## Timing and selection

1. Generate deterministic random inputs; compute an independent reference.
2. Compile each admissible candidate, launch it, and check every output plus a
   trailing allocation guard. Fail immediately on errors.
3. Capture a CUDA graph containing 64 consecutive invocations of the block for
   the published run. Warm each graph three times.
4. Collect three tuning samples per candidate in independently shuffled order.
5. Choose the lowest tuning median among the analytical top three. Choose the
   split baseline's thread tile using tuning samples as well.
6. Collect eleven separate evaluation samples, shuffled per round. Report the
   median per invocation, p25/p75, all raw samples, and the best observed candidate.

Repeated invocation uses the same resident data: this is a hot-buffer, graph-replay
experiment. CUDA events surround each graph replay; dividing by 64 amortizes graph
submission overhead. Python per-kernel dispatch does not appear inside the timed
graph. Host/device transfers, compilation, and graph construction are excluded.
Synchronous uploads are explicitly completed before using the nonblocking compute
stream; omitting that dependency caused an intermittent poisoning race caught
during development and covered by the larger GPU regression case.

The offline evaluator measures all candidates to estimate regret. A budget of
three describes what the selector inspects, not the total cost of this exhaustive
experiment. `shortlist_compile_ms` sums measured compilation times for those
candidates; `full_sweep_compile_ms` includes all candidates. Both exclude native
planner process time, which is separately recorded. Regret is
`100 * (selected evaluation median / best observed evaluation median - 1)`.

The broad hypothesis is only partially supported: the first run is near the
finite-set oracle in six cases and misses badly in two. No thresholds were adjusted
to hide misses. The fixed shape suite is exploratory, not a held-out generalization
study. Quantization, layout conversion, matmul, power, multiple GPUs, and full-model
latency remain outside this prototype.
