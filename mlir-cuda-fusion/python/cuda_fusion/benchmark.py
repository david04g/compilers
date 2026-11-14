"""Independent correctness checks and randomized CUDA-graph timing experiments."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time

import numpy as np

from .codegen import emit_cuda, validate_plan
from .cuda import CUDA, NVRTC
from .frontend import compile_graph

DEFAULT_SHAPES = "256x256,128x1024,32x4096,4x16384,256x4096,1x65536,37x1003,8x32771"
RTOL, ATOL = 2e-5, 2e-5
GUARD = np.float32(1234567.0)


def make_inputs(rows, width, seed=0, zero=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, width)).astype(np.float32)
    residual = rng.normal(size=(rows, width)).astype(np.float32)
    gamma = rng.uniform(-1.5, 1.5, width).astype(np.float32)
    if zero:
        x.fill(0)
        residual.fill(0)
    return x, residual, gamma


def reference(x, residual, gamma, epsilon):
    # The add has FP32 semantics. Accumulate the reference reduction in FP64.
    added = np.add(x, residual, dtype=np.float32).astype(np.float64)
    inv = 1.0 / np.sqrt(np.mean(added * added, axis=1, keepdims=True) + epsilon)
    return (added * inv * gamma.astype(np.float64)).astype(np.float32)


def buffers(cuda, inputs):
    x, residual, gamma = inputs
    m, n = x.shape
    output = np.full(m * n + 32, GUARD, dtype=np.float32)
    output[:m * n] = np.nan
    return [cuda.upload(x), cuda.upload(residual), cuda.upload(gamma),
            cuda.upload(np.full_like(x, np.nan)),
            cuda.upload(np.full(m, np.nan, dtype=np.float32)), cuda.upload(output)], output


def verify_candidate(cuda, loaded, pointers, host_output, expected):
    host_output[:expected.size] = np.nan
    host_output[expected.size:] = GUARD
    cuda.check(cuda.cuMemcpyHtoD_v2(pointers[-1], host_output.ctypes.data, host_output.nbytes))
    cuda.check(cuda.cuCtxSynchronize())
    cuda.launch(loaded, pointers)
    cuda.download(pointers[-1], host_output)
    np.testing.assert_array_equal(host_output[expected.size:], GUARD, err_msg="output guard overwritten")
    actual = host_output[:expected.size].reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=False)
    delta = np.abs(actual.astype(np.float64) - expected)
    return {"max_abs_error": float(delta.max()),
            "max_scaled_error": float(np.max(delta / (ATOL + RTOL * np.abs(expected)))),
            "guard_intact": True, "correct": True}


def run_case(plan, nvrtc, out, *, seed=0, samples=9, repetitions=32, budget=3):
    validate_plan(plan)
    if samples < 3 or repetitions < 1 or budget < 1:
        raise ValueError("need samples >= 3, repetitions >= 1, budget >= 1")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    m, n = plan["rows"], plan["width"]
    inputs = make_inputs(m, n, seed)
    expected = reference(*inputs, plan["epsilon"])
    rows = []
    with CUDA() as cuda:
        pointers, host_output = buffers(cuda, inputs)
        graphs = []
        for candidate in plan["candidates"]:
            if not candidate["legal"]:
                continue
            source, launches = emit_cuda(plan, candidate)
            (out / f"{candidate['id']}.cu").write_text(source, encoding="utf-8")
            cubin, log, compile_ms, flags = nvrtc.compile(source, cuda.arch)
            loaded, resources = cuda.load(cubin, launches)
            error = verify_candidate(cuda, loaded, pointers, host_output, expected)
            graph = cuda.capture(loaded, pointers, repetitions)
            graphs.append(graph)
            rows.append({**candidate, **error, "compiled_resources": resources,
                         "compile_ms": compile_ms, "compiler_log": log,
                         "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                         "compile_options": flags, "tuning_samples_us": [], "samples_us": []})
        # Warm every schedule, then interleave the samples to reduce order bias.
        for graph in graphs:
            for _ in range(3):
                cuda.time_graph_us(graph, repetitions)
        rng = random.Random(seed)
        for phase, count in [("tuning_samples_us", 3), ("samples_us", samples)]:
            for _ in range(count):
                order = list(range(len(rows)))
                rng.shuffle(order)
                for index in order:
                    rows[index][phase].append(cuda.time_graph_us(graphs[index], repetitions))
        device = cuda.info
    for row in rows:
        row["tuning_median_us"] = statistics.median(row["tuning_samples_us"])
        row["median_us"] = statistics.median(row["samples_us"])
        row["p25_us"], row["p75_us"] = map(float, np.percentile(row["samples_us"], [25, 75]))
    ranked = sorted(rows, key=lambda c: (c["predicted_us"], c["id"]))
    shortlist = ranked[:budget]
    heuristic = next(c for c in rows if c["id"] == plan["selected"])
    selected = min(shortlist, key=lambda c: c["tuning_median_us"])
    oracle = min(rows, key=lambda c: c["median_us"])
    split_candidates = [c for c in rows if c["mode"] == "split"]
    # Baseline tile choice uses tuning samples, never evaluation samples.
    split = min(split_candidates, key=lambda c: c["tuning_median_us"]) if split_candidates else None
    greedy = min(rows, key=lambda c: (c["kernel_count"], c["estimated_logical_bytes"], c["predicted_us"]))
    summary = {"shape": [m, n], "device": device, "seed": seed,
               "selected_id": selected["id"], "selected_median_us": selected["median_us"],
               "heuristic_id": heuristic["id"], "heuristic_median_us": heuristic["median_us"],
               "oracle_id": oracle["id"], "oracle_median_us": oracle["median_us"],
               "greedy_id": greedy["id"], "greedy_median_us": greedy["median_us"],
               "split_id": split["id"] if split else None,
               "split_median_us": split["median_us"] if split else None,
               "speedup_vs_split": split["median_us"] / selected["median_us"] if split else None,
               "regret_pct": 100 * (selected["median_us"] / oracle["median_us"] - 1),
               "heuristic_regret_pct": 100 * (heuristic["median_us"] / oracle["median_us"] - 1),
               "shortlist_ids": [c["id"] for c in shortlist],
               "shortlist_compile_ms": sum(c["compile_ms"] for c in shortlist),
               "full_sweep_compile_ms": sum(c["compile_ms"] for c in rows),
               "evaluated_candidates": len(rows),
               "rejected_candidates": [c for c in plan["candidates"] if not c["legal"]],
               "max_abs_error": max(c["max_abs_error"] for c in rows),
               "estimated_selected_logical_bytes": selected["estimated_logical_bytes"],
               "selected_scratch_bytes": selected["scratch_bytes"], "candidates": rows}
    return summary


def write_report(report, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = ["rows", "width", "candidate", "mode", "threads", "median_us", "p25_us", "p75_us",
              "predicted_us", "estimated_logical_bytes", "scratch_bytes", "max_registers_per_thread",
              "max_local_bytes_per_thread", "compile_ms", "max_abs_error", "selected"]
    with (out / "candidates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in report["cases"]:
            for c in case["candidates"]:
                row = {k: c[k] for k in fields if k in c}
                row.update(rows=case["shape"][0], width=case["shape"][1], candidate=c["id"],
                           max_registers_per_thread=max(r["registers_per_thread"] for r in c["compiled_resources"]),
                           max_local_bytes_per_thread=max(r["local_bytes_per_thread"] for r in c["compiled_resources"]),
                           selected=c["id"] == case["selected_id"])
                writer.writerow(row)
    lines = ["# GTX 1080 fusion experiment", "", report["method"], "",
             f"Device: **{report['device']['name']}**, sm_{report['device']['compute_capability']}. "
             f"NVRTC {report['nvrtc_version']}; {report['compiler_version']}.", "",
             "All latencies below are **measured CUDA-event times**. Logical bytes are analytical estimates; "
             "they are not DRAM counters. The model's predicted microseconds are uncalibrated.", "",
             "| Shape (rows × width) | Selected | Selected µs | Tuned split µs | Speedup | Best observed | Regret |",
             "|---|---|---:|---:|---:|---|---:|"]
    for c in report["cases"]:
        split = f"{c['split_median_us']:.3f}" if c['split_median_us'] is not None else "n/a"
        speedup = f"{c['speedup_vs_split']:.2f}×" if c['speedup_vs_split'] is not None else "n/a"
        lines.append(f"| {c['shape'][0]} × {c['shape'][1]} | {c['selected_id']} | {c['selected_median_us']:.3f} "
                     f"| {split} | {speedup} | {c['oracle_id']} | {c['regret_pct']:.1f}% |")
    cases = report["cases"]
    wins = sum(c["regret_pct"] <= 10 for c in cases)
    errors = max(c["max_abs_error"] for c in cases)
    lines += ["", f"The budgeted selector is within 10% of the best enumerated schedule in **{wins}/{len(cases)}** cases. "
              f"Worst regret: **{max(c['regret_pct'] for c in cases):.1f}%**. "
              f"All {sum(c['evaluated_candidates'] for c in cases)} evaluated candidates passed correctness and output-guard checks; "
              f"maximum absolute error: **{errors:.3g}**.", "",
              "## Memory and compiled resource evidence", "",
              "| Shape | Selected logical bytes (estimate) | Scratch bytes | Registers/thread (compiled maximum) | Local bytes/thread (compiled maximum) |",
              "|---|---:|---:|---:|---:|"]
    for c in cases:
        chosen = next(r for r in c["candidates"] if r["id"] == c["selected_id"])
        lines.append(f"| {c['shape'][0]} × {c['shape'][1]} | {chosen['estimated_logical_bytes']:,} | "
                     f"{chosen['scratch_bytes']:,} | "
                     f"{max(r['registers_per_thread'] for r in chosen['compiled_resources'])} | "
                     f"{max(r['local_bytes_per_thread'] for r in chosen['compiled_resources'])} |")
    lines += ["", "## Limits of this result", "",
              "- The workload is residual-add → RMSNorm with finite FP32 inputs. It is not a full transformer benchmark.",
              "- This is a hot-buffer CUDA-graph replay measurement. Transfers, compilation, graph creation, and Python overhead are excluded from device latency. Compilation is reported separately in JSON/CSV.",
              "- Selection uses three tuning samples; the reported evaluation samples are separate. Candidate order is randomized per round. No clock locking or exclusive GPU access is assumed.",
              "- The budget applies to the selector's shortlist. This offline experiment additionally compiles and times every legal candidate to construct the oracle; total sweep cost is not the shortlist cost.",
              "- The oracle is the fastest measured member of this finite candidate set. Small differences within sample variation should not be treated as reliable wins.",
              "- Registers/local storage come from the compiled kernel; occupancy is a CUDA API theoretical limit, not measured achieved occupancy. Local storage is not a dynamic spill-traffic counter.",
              "- Model misses and regressions remain in the table. Pascal results do not establish performance on modern GPUs or proprietary accelerators.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compiler", required=True)
    p.add_argument("--output", default="out/benchmark")
    p.add_argument("--shapes", default=DEFAULT_SHAPES)
    p.add_argument("--threads", default="64,128,256")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--samples", type=int, default=9)
    p.add_argument("--repetitions", type=int, default=32)
    p.add_argument("--budget", type=int, default=3)
    args = p.parse_args()
    if args.samples < 3 or args.repetitions < 1 or args.budget < 1:
        p.error("samples >= 3, repetitions >= 1, and budget >= 1 are required")
    try:
        shapes = [tuple(map(int, s.split("x"))) for s in args.shapes.split(",")]
        if not shapes or any(len(s) != 2 or min(s) <= 0 for s in shapes):
            raise ValueError()
    except ValueError:
        p.error("shapes must be positive rowsxwidth pairs")
    out = Path(args.output)
    nvrtc = NVRTC()
    version = subprocess.check_output([args.compiler, "--version"], text=True).strip()
    report = {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "nvrtc_version": ".".join(map(str, nvrtc.version)), "compiler_version": version,
              "python_version": platform.python_version(), "numpy_version": np.__version__,
              "os": platform.system(), "seed": args.seed, "samples": args.samples,
              "repetitions_per_graph": args.repetitions, "tuning_budget": args.budget,
              "method": f"Median of {args.samples} separately timed graph replays, each containing "
                        f"{args.repetitions} block invocations. Three independent tuning samples select "
                        f"among the model's top {args.budget} candidates. All legal candidates are evaluated for comparison.",
              "cases": []}
    for index, (m, n) in enumerate(shapes):
        directory = out / f"{m}x{n}"
        start = time.perf_counter()
        plan = compile_graph(args.compiler, m, n, directory, options=f"threads={args.threads}")
        planner_ms = (time.perf_counter() - start) * 1000
        case = run_case(plan, nvrtc, directory, seed=args.seed + index, samples=args.samples,
                        repetitions=args.repetitions, budget=args.budget)
        case["planner_process_ms"] = planner_ms
        report["device"] = case["device"]
        report["cases"].append(case)
        write_report(report, out)
        print(f"{m}x{n}: {case['selected_id']}, {case['selected_median_us']:.3f} us; "
              f"regret {case['regret_pct']:.1f}%; all {case['evaluated_candidates']} candidates correct", flush=True)
    print(f"Report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
