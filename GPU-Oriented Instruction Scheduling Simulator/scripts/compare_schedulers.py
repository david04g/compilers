#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


SCHEDULERS = ["baseline", "list", "latency-aware", "stall-fill"]


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
    parser = argparse.ArgumentParser(description="Compare all scheduling modes.")
    parser.add_argument("--exe", default="build/gpu_sched_sim")
    parser.add_argument("--input", default="examples/memory_latency.asm")
    parser.add_argument("--config", default="configs/default_gpu.json")
    parser.add_argument("--out", default="results/scheduler_compare")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    exe = resolve_exe(root, args.exe)
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for scheduler in SCHEDULERS:
        scheduler_out = out_dir / scheduler
        command = [
            str(exe),
            "--input",
            str(root / args.input),
            "--config",
            str(root / args.config),
            "--scheduler",
            scheduler,
            "--metrics",
            "--emit-schedule",
            "--output",
            str(scheduler_out),
        ]
        subprocess.run(command, check=True)
        metrics = json.loads((scheduler_out / "metrics.json").read_text())
        rows.append((scheduler, metrics["scheduled"]["ipc"], metrics["improvement"]["throughput_percent"]))

    print("| Scheduler | IPC | Throughput vs Baseline |")
    print("|---|---:|---:|")
    for scheduler, ipc, improvement in rows:
        print(f"| {scheduler} | {ipc:.2f} | {improvement:.1f}% |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
