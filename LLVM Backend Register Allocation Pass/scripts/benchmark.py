#!/usr/bin/env python3
"""Small benchmark harness for comparing LLVM Greedy with custom-ra."""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


STAT_RE = re.compile(r"^\s*([0-9]+)\s+([^#\n]+?)(?:\s+-\s+(.+))?$")
SPILL_STAT_TERMS = (
    "spill",
    "spilled",
    "spills",
    "reload",
    "reloads",
    "stack slot",
)


def parse_stats(stderr):
    stats = {}
    for line in stderr.splitlines():
        match = STAT_RE.match(line)
        if not match:
            continue
        value = int(match.group(1))
        key = " ".join(part for part in match.groups()[1:] if part).strip()
        if key:
            stats[key] = value
    return stats


def spill_metric(stats):
    total = 0
    selected = {}
    for key, value in stats.items():
        lowered = key.lower()
        if any(term in lowered for term in SPILL_STAT_TERMS):
            selected[key] = value
            total += value
    return total, selected


def run_llc(llc, plugin, allocator, mir):
    cmd = [
        llc,
        "-mtriple=x86_64-unknown-linux-gnu",
        "-verify-machineinstrs",
        "-stats",
        "-o",
        "-",
    ]
    if allocator == "custom-ra":
        cmd.extend(["-load", plugin, "-regalloc=custom-ra"])
    else:
        cmd.extend(["-regalloc=greedy"])
    cmd.append(str(mir))

    start = time.perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - start
    stats = parse_stats(proc.stderr)
    spills, spill_stats = spill_metric(stats)
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "spill_metric": spills,
        "spill_stats": spill_stats,
        "all_stats": stats,
        "stdout_bytes": len(proc.stdout.encode()),
        "stderr": proc.stderr,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llc", default="llc")
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--output", default="benchmark-results.json")
    parser.add_argument("mir", nargs="+", type=Path)
    args = parser.parse_args()

    results = {}
    for mir in args.mir:
        results[str(mir)] = {
            "greedy": run_llc(args.llc, args.plugin, "greedy", mir),
            "custom-ra": run_llc(args.llc, args.plugin, "custom-ra", mir),
        }

    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
