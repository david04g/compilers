# GTX 1080 fusion experiment

Median of 11 separately timed graph replays, each containing 64 block invocations. Three independent tuning samples select among the model's top 3 candidates. All legal candidates are evaluated for comparison.

Device: **NVIDIA GeForce GTX 1080**, sm_61. NVRTC 12.9; fusion-opt; LLVM 24.0.0git.

All latencies below are **measured CUDA-event times**. Logical bytes are analytical estimates; they are not DRAM counters. The model's predicted microseconds are uncalibrated.

| Shape (rows × width) | Selected | Selected µs | Tuned split µs | Speedup | Best observed | Regret |
|---|---|---:|---:|---:|---|---:|
| 256 × 256 | cached-t64 | 3.296 | 8.400 | 2.55× | cached-t64 | 0.0% |
| 128 × 1024 | cached-t128 | 4.144 | 11.840 | 2.86× | cached-t128 | 0.0% |
| 32 × 4096 | cached-t256 | 4.640 | 12.944 | 2.79× | cached-t256 | 0.0% |
| 4 × 16384 | partial-t256 | 14.896 | 16.272 | 1.09× | cached-t256 | 57.5% |
| 256 × 4096 | cached-t128 | 58.512 | 111.824 | 1.91× | cached-t256 | 1.1% |
| 1 × 65536 | split-t256 | 41.744 | 41.744 | 1.00× | split-t256 | 0.0% |
| 37 × 1003 | cached-t128 | 3.168 | 6.864 | 2.17× | cached-t128 | 0.0% |
| 8 × 32771 | split-t256 | 43.392 | 43.392 | 1.00× | cached-t256 | 88.6% |

The budgeted selector is within 10% of the best enumerated schedule in **6/8** cases. Worst regret: **88.6%**. All 90 evaluated candidates passed correctness and output-guard checks; maximum absolute error: **9.54e-07**.

## Memory and compiled resource evidence

| Shape | Selected logical bytes (estimate) | Scratch bytes | Registers/thread (compiled maximum) | Local bytes/thread (compiled maximum) |
|---|---:|---:|---:|---:|
| 256 × 256 | 1,048,576 | 0 | 18 | 0 |
| 128 × 1024 | 2,097,152 | 0 | 27 | 0 |
| 32 × 4096 | 2,097,152 | 0 | 32 | 0 |
| 4 × 16384 | 1,835,024 | 262,160 | 17 | 0 |
| 256 × 4096 | 16,777,216 | 0 | 49 | 0 |
| 1 × 65536 | 2,097,156 | 262,148 | 11 | 0 |
| 37 × 1003 | 593,776 | 0 | 27 | 0 |
| 8 × 32771 | 8,389,408 | 1,048,704 | 12 | 0 |

## Limits of this result

- The workload is residual-add → RMSNorm with finite FP32 inputs. It is not a full transformer benchmark.
- This is a hot-buffer CUDA-graph replay measurement. Transfers, compilation, graph creation, and Python overhead are excluded from device latency. Compilation is reported separately in JSON/CSV.
- Selection uses three tuning samples; the reported evaluation samples are separate. Candidate order is randomized per round. No clock locking or exclusive GPU access is assumed.
- The budget applies to the selector's shortlist. This offline experiment additionally compiles and times every legal candidate to construct the oracle; total sweep cost is not the shortlist cost.
- The oracle is the fastest measured member of this finite candidate set. Small differences within sample variation should not be treated as reliable wins.
- Registers/local storage come from the compiled kernel; occupancy is a CUDA API theoretical limit, not measured achieved occupancy. Local storage is not a dynamic spill-traffic counter.
- Model misses and regressions remain in the table. Pascal results do not establish performance on modern GPUs or proprietary accelerators.
