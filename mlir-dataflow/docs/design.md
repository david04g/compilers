# Plan and simulator design

## Compilation path

```text
linalg/tensor module
  -> validate the supported SSA chain and scalar payloads (PlanPass.cpp)
  -> enumerate tile triples, boundary masks, placements (Planner.cpp)
  -> emit per-tile loads, reductions, pointwise operations and transfers
  -> infer buffer-use intervals and allocate bank arenas
  -> reject capacity violations and rank by analytical resource load
  -> write versioned JSON plan + candidate diagnostics
  -> independently schedule, check memory lifetimes, execute, compare (Python)
```

The C++ pass makes the heuristic decision. Python does not parse MLIR, select the
heuristic winner, or silently regenerate a graph from dimensions. The simulator
executes the emitted event stream. Its reference path uses separate whole-array
float64 matmuls and GELU, then compares f32 results with `rtol=2e-4, atol=2e-5`.
Reduction grouping can introduce normal f32 differences.

## Candidate grammar

The two boundary bits encode fusion of matmul/bias and bias/GELU independently:

| Mode | Executable regions |
|---|---|
| split | `[matmul] [bias] [GELU]` |
| matmul-bias | `[matmul bias] [GELU]` |
| bias-gelu | `[matmul] [bias GELU]` |
| fused | `[matmul bias GELU]` |

When present, the second matmul adds a final independent region to every candidate.
`compact` maps every stage of a region to its first stage's CU; `spread` maps each
stage to its stage index modulo the number of CUs. CU IDs are row-major. Region
boundaries always store/reload through off-chip memory. Inside a region, values
stay in SRAM; a CU change emits a `send` event along an X-then-Y route.

Fixed tiles cover `M,N,K`, where `K` is the reduction dimension. Edge event shapes
are clipped, including the final reduction block. The second projection uses the
same triple with its own output/reduction dimensions. The compiler zeroes an
accumulator once per output tile and accumulates all K blocks into it.

## Memory and dependencies

Every CU has configurable banks of equal capacity. Matrix A, accumulators, and
pointwise inputs/outputs use bank 0; matrix B and bias use bank 1 (modulo the bank
count). Allocation uses aligned first-fit ranges with inclusive first/last use
intervals from one prototype tile. Nonoverlapping values may share an offset.

The prototype uses the largest actual edge-clipped tile for that region; smaller
edge tiles reuse its allocation. Fused regions reserve two disjoint copies of the
arena for streaming; split regions reserve one. An event dependency from the last
store of a slot's previous tile prevents that slot from being overwritten early.
Each next region depends on all outstanding previous-region slots, so its arena
can reuse the same physical bank addresses. This conservative schedule makes the
memory model easy to inspect and extend.

Capacity is checked per bank, not against a sum across the grid. Reported
`peak_sram_bytes` is the maximum sum of active region arena reservations, including
alignment and both slots; it is not the sum of every logical tensor size. The
`peak_bank_bytes` map contains per-bank maxima across regions, which need not occur
simultaneously. The simulator independently checks concrete tile lifetimes for
live address overlap after scheduling and executes through those shared offsets.

## Timing model

- `matmul`: `ceil(m*n*k / macs_per_cycle)` per reduction block (MACs, not FLOPs).
- `zero` and `bias`: `ceil(elements / elements_per_cycle)`.
- `GELU`: `gelu_cycles_per_vector * ceil(elements / elements_per_cycle)`.
- DRAM transfer: startup per traversed link plus one DRAM startup, then
  `ceil(bytes / min(dram_bandwidth, route_link_bandwidth))`.
- On-chip send: startup per link plus `ceil(bytes / link_bandwidth)`.

DRAM attaches at CU 0. All DRAM traffic shares one port. Directed XY links are
independent in opposite directions; transfers sharing a directed link contend.
A transfer reserves all route links simultaneously for its entire duration
(circuit model), along with endpoint banks and the DRAM port when relevant.
Each compute event reserves its CU and all banks it accesses. Banks have one port.
Multiple operands in one bank are treated as serviced within the kernel's stated
duration; bank bandwidth and internal access counts are not modeled separately.

The analytical estimate sums, across regions, the maximum resource workload plus
a single-tile dependency-span fill/drain correction. It does not run the discrete
scheduler. The simulator chooses the ready event with the earliest feasible start,
breaking ties by event ID, and reserves its resources until completion. It enforces
dependency completion, slot reuse, and resource contention. This distinction is
why the heuristic can miss the fastest simulated candidate.

Stall totals sum resource waits of events and can overlap; they do not sum to
wall latency. Dependency delays (including slot backpressure) appear separately
as `dependency_ready` in the timeline. Perfetto lanes show every reserved resource.

## JSON ABI, version 1

Top-level fields:

| Field | Meaning |
|---|---|
| `schema_version` | Must be 1 |
| `graph` | Positive static `m,k,n`; `p=0` means no second matmul |
| `hardware` | Grid, banks, capacities, rates, transfer startup |
| `strategy` | `heuristic`, `split`, or `fuse` |
| `selected_plan` | Executable candidate, or null if none is legal |
| `candidates` | Every candidate's diagnostics/cost; executable bodies with `emit-all=true` |

A legal candidate contains `id`, `mode`, `placement`, `tile`, estimates,
`regions`, `buffers`, and `events`. An illegal one contains `legal=false` and a
specific `rejection` string. On no legal plan the compiler still writes diagnostics
and exits unsuccessfully; it never silently falls back to an overflowing plan.

A buffer identifies `id, region, slot, cu, bank, offset, bytes` and prototype use
indices. An event identifies `id, region, tile_index, kind, deps, resources, cycles`,
`reads/writes` buffer IDs, an actual `shape`, and transfer `tensor, origin, bytes`.
Shapes for matmul are `[m,n,k]`; other compute/send shapes are `[m,n]`. Bias loads
use a one-dimensional shape. Events have contiguous zero-based IDs. Tensor names
are A, B, bias, I0, I1, H, W2 and Y; intermediate off-chip arrays begin uninitialized.

An exhaustive result means the fastest plan under this same scheduler and candidate
grammar. It is not an optimal graph partition, placement, allocation, or schedule
over all possible implementations. Keep that qualifier in any performance claim.
