"""Deterministic resource-constrained scheduling and physical-bank execution."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np


def schedule(events):
    """Earliest-ready list scheduling; an event reserves all resources until done.

    Routes use circuit reservations (all directed XY links at once). A bank has
    one port and a CU executes one kernel at a time. This is a latency model,
    not a wall-clock benchmark of NumPy or a cycle-accurate hardware model.
    """
    count = len(events)
    children = [[] for _ in events]
    degree = []
    for i, event in enumerate(events):
        if event["id"] != i or type(event["cycles"]) is not int or event["cycles"] <= 0:
            raise ValueError("event IDs must be contiguous; durations must be positive integers")
        deps = event["deps"]
        if len(set(deps)) != len(deps) or any(type(d) is not int or not 0 <= d < count or d == i for d in deps):
            raise ValueError("invalid event dependency")
        if len(set(event["resources"])) != len(event["resources"]):
            raise ValueError("duplicate resource")
        degree.append(len(deps))
        for dep in deps:
            children[dep].append(i)
    ready = {i for i, d in enumerate(degree) if d == 0}
    available = defaultdict(int)
    ends = [0] * count
    timeline = []
    stalls = Counter()
    while ready:
        def earliest(i):
            event = events[i]
            dependencies_ready = max((ends[d] for d in event["deps"]), default=0)
            resource_ready = max((available[r] for r in event["resources"]), default=0)
            return max(dependencies_ready, resource_ready), i

        start, i = min(map(earliest, ready))
        ready.remove(i)
        event = events[i]
        dep_ready = max((ends[d] for d in event["deps"]), default=0)
        blockers = sorted(r for r in event["resources"] if available[r] == start and start > dep_ready)
        end = start + event["cycles"]
        ends[i] = end
        for resource in event["resources"]:
            available[resource] = end
        if blockers:
            # Attribute each wait exactly once, even if several resources tie.
            stalls[blockers[0]] += start - dep_ready
        timeline.append({"id": i, "kind": event["kind"], "region": event.get("region", 0),
                         "start": start, "end": end, "dependency_ready": dep_ready,
                         "resource_wait_cycles": start - dep_ready, "blocked_by": blockers,
                         "resources": event["resources"]})
        for child in children[i]:
            degree[child] -= 1
            if degree[child] == 0:
                ready.add(child)
    if len(timeline) != count:
        raise ValueError("event graph contains a cycle")
    return {"simulated_cycles": max(ends, default=0), "timeline": timeline,
            "resource_wait_cycles": dict(stalls.most_common())}


def validate(artifact, plan):
    if artifact.get("schema_version") != 1 or not plan or not plan.get("legal"):
        raise ValueError("expected a legal schema_version=1 execution plan")
    hw = artifact["hardware"]
    buffers = {b["id"]: b for b in plan["buffers"]}
    if len(buffers) != len(plan["buffers"]):
        raise ValueError("duplicate buffer ID")
    for buf in buffers.values():
        if not (0 <= buf["cu"] < hw["rows"] * hw["cols"] and 0 <= buf["bank"] < hw["banks_per_cu"]):
            raise ValueError("buffer placement outside hardware")
        if buf["offset"] < 0 or buf["offset"] % 16 or buf["bytes"] <= 0 or buf["bytes"] % 4:
            raise ValueError("invalid buffer extent/alignment")
        if buf["offset"] + buf["bytes"] > hw["bank_bytes"]:
            raise ValueError("buffer exceeds SRAM bank capacity")
    kinds = {"zero", "load", "store", "send", "matmul", "bias", "gelu"}
    for event in plan["events"]:
        if event["kind"] not in kinds or any(type(d) is not int or d <= 0 for d in event["shape"]):
            raise ValueError("invalid event kind/shape")
        for name in event["reads"] + event["writes"]:
            if name not in buffers:
                raise ValueError("event references missing buffer")
            b = buffers[name]
            if b["region"] != event["region"] or f"bank:{b['cu']}:{b['bank']}" not in event["resources"]:
                raise ValueError("event missing bank reservation or using another region's buffer")
    return buffers


def inputs(graph, seed):
    rng = np.random.default_rng(seed)
    m, k, n, p = (graph[x] for x in ("m", "k", "n", "p"))
    def random(shape):
        return (0.25 * rng.standard_normal(shape)).astype(np.float32)
    arrays = {"A": random((m, k)), "B": random((k, n)), "bias": random((n,)),
              "I0": np.full((m, n), np.nan, np.float32), "I1": np.full((m, n), np.nan, np.float32),
              "H": np.full((m, n), np.nan, np.float32),
              "Y": np.full((m, p or n), np.nan, np.float32)}
    if p:
        arrays["W2"] = random((n, p))
    return arrays


def validate_lifetimes(plan, timing, buffers):
    """Reject temporal overlap of distinct buffer values at aliased addresses."""
    lives = {}
    for t in timing["timeline"]:
        event = plan["events"][t["id"]]
        for name in set(event["reads"] + event["writes"]):
            key = (event["region"], event["tile_index"], name)
            lo, hi = lives.get(key, (t["start"], t["end"]))
            lives[key] = min(lo, t["start"]), max(hi, t["end"])
    per_bank = defaultdict(list)
    for (_, _, name), (lo, hi) in lives.items():
        b = buffers[name]
        per_bank[b["cu"], b["bank"]].append((lo, hi, b["offset"], b["offset"] + b["bytes"], name))
    for intervals in per_bank.values():
        active = []
        for current in sorted(intervals):
            lo, hi, offset, end, name = current
            active = [a for a in active if a[1] > lo]
            for other in active:
                if offset < other[3] and other[2] < end:
                    raise ValueError(f"live SRAM alias between {name} and {other[4]}")
            active.append(current)


def reference(arrays):
    # Independent whole-tensor, float64 oracle; no tile schedule or bank accesses.
    x = arrays["A"].astype(np.float64) @ arrays["B"].astype(np.float64) + arrays["bias"]
    result = 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    if "W2" in arrays:
        result = result @ arrays["W2"].astype(np.float64)
    return result.astype(np.float32)


def execute(artifact, plan, timing, seed=0):
    """Interpret emitted events through actual offsets in shared SRAM bank arrays."""
    buffers = validate(artifact, plan)
    arrays = inputs(artifact["graph"], seed)
    expected = reference(arrays)
    banks = {}
    for b in buffers.values():
        banks.setdefault((b["cu"], b["bank"]), np.full(artifact["hardware"]["bank_bytes"] // 4, np.nan, np.float32))

    def view(name, shape):
        b = buffers[name]
        count = math.prod(shape)
        if count * 4 > b["bytes"]:
            raise ValueError(f"event overruns {name}")
        start = b["offset"] // 4
        return banks[b["cu"], b["bank"]][start:start + count].reshape(shape)

    def tensor_view(event):
        tensor = arrays[event["tensor"]]
        if len(event["origin"]) != tensor.ndim or len(event["shape"]) != tensor.ndim:
            raise ValueError("transfer rank mismatch")
        if any(o < 0 or o + s > limit for o, s, limit in zip(event["origin"], event["shape"], tensor.shape)):
            raise ValueError("transfer outside tensor bounds")
        return tensor[tuple(slice(o, o+s) for o, s in zip(event["origin"], event["shape"]))]

    coverage = np.zeros_like(expected, dtype=np.int32)
    # Events are atomic at completion. Single-port bank reservations prevent an
    # overlapping write while a consumer reads; dependency ties sort by ID.
    for entry in sorted(timing["timeline"], key=lambda t: (t["end"], t["id"])):
        event = plan["events"][entry["id"]]
        kind, shape = event["kind"], event["shape"]
        reads, writes = event["reads"], event["writes"]
        if kind == "load":
            view(writes[0], shape)[:] = tensor_view(event)
        elif kind == "store":
            tensor_view(event)[:] = view(reads[0], shape)
            if event["tensor"] == "Y":
                slices = tuple(slice(o, o+s) for o, s in zip(event["origin"], shape))
                coverage[slices] += 1
        elif kind == "zero":
            view(writes[0], shape).fill(0)
        elif kind == "send":
            view(writes[0], shape)[:] = view(reads[0], shape)
        elif kind == "matmul":
            m, n, k = shape
            view(writes[0], (m, n))[:] = (view(reads[2], (m, n)) +
                view(reads[0], (m, k)) @ view(reads[1], (k, n)))
        elif kind == "bias":
            view(writes[0], shape)[:] = view(reads[0], shape) + view(reads[1], (shape[1],))
        elif kind == "gelu":
            x = view(reads[0], shape)
            # Match the f32 scalar payload, including the operation grouping.
            view(writes[0], shape)[:] = (np.float32(0.5) * x) * (np.float32(1) +
                np.tanh(np.float32(0.7978845608028654) * (x + np.float32(0.044715) * ((x*x)*x))))
    if not np.all(coverage == 1):
        raise ValueError("output tiles must cover Y exactly once")
    actual = arrays["Y"]
    if not np.all(np.isfinite(actual)):
        raise ValueError("nonfinite output: missing transfer or uninitialized buffer")
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
    return {"verified": True, "seed": seed,
            "max_abs_error": float(np.max(np.abs(actual - expected))),
            "rtol": 2e-4, "atol": 2e-5}, actual


def simulate(artifact, plan=None, verify=True, seed=0):
    plan = plan if plan is not None else artifact["selected_plan"]
    buffers = validate(artifact, plan)
    timing = schedule(plan["events"])
    validate_lifetimes(plan, timing, buffers)
    offchip = sum(e["bytes"] for e in plan["events"] if e["kind"] in ("load", "store"))
    if offchip != plan["offchip_bytes"]:
        raise ValueError("off-chip accounting disagrees with emitted transfers")
    result = {"plan_id": plan["id"], "mode": plan["mode"], "tile": plan["tile"],
              "placement": plan["placement"], "estimated_cycles": plan["estimated_cycles"],
              "offchip_bytes": offchip, "peak_sram_bytes": plan["peak_sram_bytes"],
              "peak_bank_bytes": plan["peak_bank_bytes"], **timing}
    if verify:
        correctness, _ = execute(artifact, plan, timing, seed)
        result.update(correctness)
    else:
        result["verified"] = False
    return result


def write_trace(path, plan, timing):
    """Chrome/Perfetto trace; one lane for each CU, link, bank and DRAM port."""
    events = []
    for t in timing["timeline"]:
        for resource in t["resources"]:
            events.append({"name": f"{t['id']}:{t['kind']}", "ph": "X", "pid": 1, "tid": resource,
                           "ts": t["start"], "dur": t["end"]-t["start"],
                           "args": {"blocked_by": t["blocked_by"], "wait": t["resource_wait_cycles"]}})
    Path(path).write_text(json.dumps({"displayTimeUnit": "ns", "traceEvents": events}), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-verify", action="store_true", help="timing only; explicitly disables numerical verification")
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--trace", type=Path, help="Chrome/Perfetto trace; one timestamp unit represents one cycle")
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args()
    artifact = json.loads(args.plan.read_text(encoding="utf-8"))
    result = simulate(artifact, verify=not args.no_verify, seed=args.seed)
    if args.explain:
        rejected = [c for c in artifact["candidates"] if not c["legal"]]
        for c in rejected:
            print(f"rejected {c['id']}: {c['rejection']}")
        for resource, wait in list(result["resource_wait_cycles"].items())[:8]:
            print(f"wait on {resource}: {wait} summed event-cycles (may overlap)")
    summary = {k: v for k, v in result.items() if k != "timeline"}
    print(json.dumps(summary, indent=2))
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        write_trace(args.trace, artifact["selected_plan"], result)


if __name__ == "__main__":
    main()
