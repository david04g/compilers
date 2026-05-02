#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


BENCHMARKS = [
    "memory_latency",
    "dependency_chain",
    "independent_mix",
    "memory_heavy",
    "barrier",
]


def resolve_exe(root: Path, requested: str) -> Path:
    candidates = [
        root / requested,
        root / "build" / "Debug" / "gpu_sched_sim.exe",
        root / "build" / "Release" / "gpu_sched_sim.exe",
        root / "build" / "gpu_sched_sim",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GPU scheduling benchmarks.")
    parser.add_argument("--exe", default="build/gpu_sched_sim")
    parser.add_argument("--config", default="configs/default_gpu.json")
    parser.add_argument("--scheduler", default="stall-fill")
    parser.add_argument("--out", default="results/benchmarks")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    exe = resolve_exe(root, args.exe)
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in BENCHMARKS:
        bench_out = out_dir / name
        command = [
            str(exe),
            "--input",
            str(root / "examples" / f"{name}.asm"),
            "--config",
            str(root / args.config),
            "--scheduler",
            args.scheduler,
            "--metrics",
            "--emit-schedule",
            "--emit-dot",
            "--output",
            str(bench_out),
        ]
        subprocess.run(command, check=True)
        metrics = json.loads((bench_out / "metrics.json").read_text())
        rows.append(
            {
                "benchmark": name,
                "baseline_ipc": metrics["baseline"]["ipc"],
                "scheduled_ipc": metrics["scheduled"]["ipc"],
                "stall_reduction_percent": metrics["improvement"][
                    "stall_reduction_percent"
                ],
                "throughput_percent": metrics["improvement"]["throughput_percent"],
            }
        )

    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("| Benchmark | Baseline IPC | Scheduled IPC | Stall Reduction | Throughput Gain |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['benchmark']} | {row['baseline_ipc']:.2f} | "
            f"{row['scheduled_ipc']:.2f} | {row['stall_reduction_percent']:.1f}% | "
            f"{row['throughput_percent']:.1f}% |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
