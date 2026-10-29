# Measured stack spilling results

## Result

**9/9 MIR tests and 11/11 executable scenarios pass.** Each executable scenario checks 100 deterministic inputs × 8 loop-count/branch inputs against an independent expected result and LLVM Greedy. That is **8,800 distinct checks**, repeated five times (44,000 checks). Inputs include zeros, maximum unsigned integers, random unsigned integers with wraparound, and exactly representable floating-point values.

The previous allocator compiles 1/11 of the same generated workloads; the updated allocator compiles 11/11. The low-pressure case has zero spill events. Every pressure case has real stack stores, not just stack-slot declarations.

## Performance

LLVM/Clang 23.1.4, MSYS2 UCRT64, Release plugin, x86_64-w64-windows-gnu, host CPU reported as znver3. Every code-generation run uses `-verify-machineinstrs`.

Compilation times are medians of five complete llc process runs with tracing disabled; startup and plugin loading are included. Execution uses five native-process samples, alternating measurement order. Iteration counts are calibrated using Greedy to approximately 0.1 seconds or more, then held equal for the paired candidate run. The table normalizes execution time per call. Timings are local synthetic microbenchmarks, not general speedup claims.

| Workload | Previous compiles | Spill events | Stack loads/stores Greedy → custom | Compile ms Greedy → custom | Runtime ns/call Greedy → custom | Custom/Greedy runtime |
|---|---|---:|---|---|---|---:|
| low64 | yes | 0 | 0/0 → 0/0 | 45.1 → 42.2 | 1.97 → 1.97 | 1.00× |
| pressure20 | no | 7 | 6/6 → 10/23 | 50.1 → 47.9 | 8.85 → 10.16 | 1.15× |
| pressure40 | no | 27 | 27/27 → 47/83 | 54.5 → 64.2 | 17.40 → 32.92 | 1.89× |
| pressure80 | no | 67 | 67/67 → 125/203 | 61.9 → 87.6 | 32.92 → 85.83 | 2.61× |
| call20 | no | 13 | 13/13 → 20/40 | 48.2 → 49.7 | 11.41 → 17.19 | 1.51× |
| branch20 | no | 7 | 13/7 → 19/38 | 53.2 → 51.2 | 8.28 → 10.99 | 1.33× |
| loop20 | no | 11 | 10/10 → 38/29 | 52.7 → 54.4 | 21.25 → 39.58 | 1.86× |
| pressure32 | no | 27 | 27/27 → 47/83 | 51.6 → 55.9 | 16.46 → 35.31 | 2.15× |
| call32 | no | 13 | 13/13 → 20/40 | 51.4 → 52.1 | 10.83 → 17.08 | 1.58× |
| float40 | no | 26 | 25/25 → 115/76 | 50.6 → 62.2 | 30.00 → 116.25 | 3.88× |
| float-call20 | no | 10 | 10/10 → 30/30 | 47.0 → 50.1 | 13.12 → 35.52 | 2.71× |

**Interpretation:** the addition makes previously rejected pressure cases compile correctly. It generally generates more memory traffic and slower code than Greedy. Rebuilding the graph on every spill also increases compile work. This is a traceable correctness milestone, not a spill-reduction optimization.

Stack load/store counts are static machine instructions whose memory operands reference spill slots; they include folded accesses and are not dynamic hardware counters. Raw JSON also records object-file sizes, which include object metadata and should not be treated as pure code size. Benchmark input loads are volatile and values depend on a key loaded last, preventing the pressure workload from becoming a low-pressure streaming reduction.

## Coverage

- MIR: existing straight-line, branch, call regmask, and multi-function trace/DOT cases; formerly rejected forced pressure and loop pressure; new 32-bit, floating-point, and call-pressure fixtures.
- The three new MIR fixtures check actual stack slots, completed allocation, rewritten physical registers, and byte-identical output on a second compilation.
- Native execution: 4, 20, 40, and 80 integer values; values live across calls and branches; loop trip counts 0–7; 32-bit overflow; floating-point register classes and floating-point values live across calls.
- Linux-target handcrafted MIR and Windows-target generated MIR are verified. Native execution and timing cover Windows x86-64 only.
- Floating-point correctness inputs are small exact integers stored as doubles. NaNs, infinities, exception handling, vectors, debug-info stress, and other architectures are not validated by this suite.
