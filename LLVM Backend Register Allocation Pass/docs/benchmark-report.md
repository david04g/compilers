# Benchmark Report Template

## Environment

- LLVM version:
- Host:
- Target triple:
- Build type:
- Baseline allocator: Greedy
- Candidate allocator: custom-ra

## Workloads

| Workload | Source | Notes |
| --- | --- | --- |
| smoke-mir | `test/*.mir` | Correctness and trace coverage |
| representative-c | local C/C++ programs | Compile through `clang -S -emit-llvm` then `llc` |
| SPEC-like | external suite | Optional, not vendored |

## Metrics

| Workload | Baseline spill metric | custom-ra spill metric | Delta | Code size delta | Compile-time delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Not yet run | N/A | N/A | N/A | N/A | N/A |

## Notes

The target improvement is at least a 15% spill-count reduction on selected
pressure-heavy workloads. This report must not claim that target until
`scripts/benchmark.py` has been run against a local LLVM build and representative
workloads.

The benchmark harness runs both Greedy and `custom-ra` with
`-verify-machineinstrs` and derives the spill metric from LLVM statistics whose
names mention spills, reloads, or stack slots. Unsupported-spill failures are
recorded as failed candidate runs, not as successful spill reductions.
