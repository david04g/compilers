#!/usr/bin/env python3
"""Executable differential tests and a small, reproducible spill benchmark.

Only the candidate workload uses custom-ra; the C test driver uses Clang's
normal allocator. All llc invocations enable the machine instruction verifier.
"""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time


def run(cmd):
    start = time.perf_counter()
    p = subprocess.run(list(map(str, cmd)), capture_output=True, text=True,
                       timeout=120)
    if p.returncode:
        raise RuntimeError(f"{cmd}\n{p.stdout}\n{p.stderr}")
    return p.stdout, time.perf_counter() - start


def workload(n, call=False, branch=False, loop=False, bits=64, floating=False):
    """Volatile loads prevent rematerialization from hiding stack spilling."""
    ty = "double" if floating else f"i{bits}"
    zero = "0.000000e+00" if floating else "0"
    combine, add = ("fmul", "fadd") if floating else ("xor", "add")
    s = ["declare void @barrier()", f"define {ty} @work(ptr %p, i64 %count) {{", "entry:"]
    if loop:
        s += ["  %empty = icmp eq i64 %count, 0", "  br i1 %empty, label %zero, label %body",
              f"zero:\n  ret {ty} {zero}", "body:",
              "  %iter = phi i64 [0, %entry], [%next, %body]",
              f"  %acc = phi {ty} [{zero}, %entry], [%total, %body]"]
    for i in range(n):
        s += [f"  %p{i} = getelementptr {ty}, ptr %p, i64 {i}",
              f"  %v{i} = load volatile {ty}, ptr %p{i}, align {bits // 8}"]
    if call:
        s += ["  call void @barrier()"]
    # Every loaded value is used with a key loaded LAST. This prevents the
    # backend from turning the intended pressure case into a streaming sum.
    s += [f"  %key = load volatile {ty}, ptr %p, align {bits // 8}"]
    if branch:
        s += ["  %cond = icmp eq i64 %count, 0", "  br i1 %cond, label %left, label %right", "left:"]
    def sums(prefix, order):
        lines = [f"  %{prefix}first = {combine} {ty} %v{order[0]}, %key"]
        prev = f"%{prefix}first"
        for j, i in enumerate(order[1:]):
            name = f"%{prefix}{j}"
            lines += [f"  %{prefix}x{j} = {combine} {ty} %v{i}, %key",
                      f"  {name} = {add} {ty} {prev}, %{prefix}x{j}"]
            prev = name
        return lines, prev
    lines, last = sums("sum", list(range(n)))
    s += lines
    if branch:
        s += [f"  ret {ty} {last}", "right:"]
        lines, last = sums("rev", list(reversed(range(n))))
        s += lines
    if loop:
        s += [f"  %total = {add} {ty} %acc, {last}", "  %next = add i64 %iter, 1",
              "  %more = icmp ult i64 %next, %count", "  br i1 %more, label %body, label %exit",
              f"exit:\n  ret {ty} %total"]
    else:
        s += [f"  ret {ty} {last}"]
    return "\n".join(s + ["}", ""])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llc", required=True)
    ap.add_argument("--clang", required=True)
    ap.add_argument("--plugin", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()
    if a.repeats < 1:
        ap.error("--repeats must be at least 1")
    a.output.mkdir(parents=True, exist_ok=True)
    cases = [("low64", 4, {}), ("pressure20", 20, {}), ("pressure40", 40, {}),
             ("pressure80", 80, {}), ("call20", 20, {"call": True}),
             ("branch20", 20, {"branch": True}), ("loop20", 20, {"loop": True}),
             ("pressure32", 40, {"bits": 32}), ("call32", 20, {"bits": 32, "call": True}),
             ("float40", 40, {"floating": True}),
             ("float-call20", 20, {"floating": True, "call": True})]
    report = {"environment": run([a.llc, "--version"])[0], "cases": []}
    for name, n, opts in cases:
        bits = opts.get("bits", 64)
        ty = "double" if opts.get("floating") else f"uint{bits}_t"
        fill = "(seed % 101)" if opts.get("floating") else f"({ty})(seed>>16)"
        extreme = "100" if opts.get("floating") else f"({ty})-1"
        combine = "*" if opts.get("floating") else "^"
        objects = []
        entry = {"name": name, "values": n}
        for alloc in ("greedy", "custom-ra"):
            stem = a.output / f"{name}-{alloc}"
            ir = stem.with_suffix(".ll")
            ir.write_text(workload(n, **opts).replace("@work(", f"@{alloc.replace('-', '_')}_work("))
            obj = stem.with_suffix(".o")
            cmd = [a.llc, "-verify-machineinstrs"]
            if alloc == "custom-ra":
                trace = stem.with_suffix(".trace.txt")
                trace.write_text("")
                cmd += ["-load", a.plugin, f"-custom-ra-trace={trace}"]
            cmd += [f"-regalloc={alloc}", ir]
            # Measure code generation separately from trace/file collection.
            timings = []
            for _ in range(a.repeats):
                benchcmd = [x for x in cmd if not str(x).startswith("-custom-ra-trace=")]
                _, elapsed = run(benchcmd + ["-filetype=obj", "-o", obj])
                timings.append(elapsed)
            run(cmd + ["-filetype=obj", "-o", obj])
            run([x for x in cmd if not str(x).startswith("-custom-ra-trace=")] +
                ["-stop-after=virtregrewriter", "-o", stem.with_suffix(".mir")])
            run([x for x in cmd if not str(x).startswith("-custom-ra-trace=")] +
                ["-o", stem.with_suffix(".s")])
            mir = stem.with_suffix(".mir").read_text()
            entry[alloc] = {"compile_seconds_median": statistics.median(timings),
                            "object_bytes": obj.stat().st_size,
                            "static_stack_loads": len(re.findall(r"::.*load.*%stack", mir)),
                            "static_stack_stores": len(re.findall(r"::.*store.*%stack", mir))}
            if alloc == "custom-ra":
                entry[alloc]["spill_events"] = trace.read_text().count("spill: round=")
                if name != "low64" and entry[alloc]["static_stack_stores"] == 0:
                    raise AssertionError(f"{name}: expected actual stack stores")
            objects.append(obj)
        driver = a.output / f"{name}-driver.c"
        multiplier = "count" if opts.get("loop") else "1"
        driver.write_text(f'''#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
{ty} greedy_work({ty} *, uint64_t);
{ty} custom_ra_work({ty} *, uint64_t);
__attribute__((noinline)) void barrier(void) {{ __asm__ volatile("" ::: "memory"); }}
static volatile {ty} sink;
static double bench({ty} (*fn)({ty} *, uint64_t), {ty} *p, unsigned iters) {{
  clock_t t=clock();
  for(unsigned i=0;i<iters;++i) sink+=fn(p,3);
  return (double)(clock()-t)/CLOCKS_PER_SEC;
}}
int main(int argc, char **argv) {{
  {ty} p[{n}], expected;
  uint64_t seed=1234567, checks=0;
  for (int test=0; test<100; ++test) {{
    expected=0;
    for(int i=0;i<{n};++i) {{ seed=seed*6364136223846793005ULL+1;
      p[i]=test==0?0:test==1?{extreme}:{fill}; }}
    expected=0;
    for(int i=0;i<{n};++i) expected+=(p[i]{combine}p[0]);
    for(uint64_t count=0;count<8;++count) {{
      {ty} want=expected*({multiplier});
      {ty} g=greedy_work(p,count), c=custom_ra_work(p,count);
      if(g!=want || c!=want) {{ printf("FAIL test=%d count=%llu expected=%llu greedy=%llu custom=%llu\\n",
        test,(unsigned long long)count,(unsigned long long)want,(unsigned long long)g,(unsigned long long)c); return 1; }}
      ++checks;
    }}
  }}
  unsigned iters=300000;
  while(bench(greedy_work,p,iters)<0.1 && iters<153600000) iters*=2;
  double gtime,ctime;
  if(argc>1 && atoi(argv[1])%2) {{
    ctime=bench(custom_ra_work,p,iters); gtime=bench(greedy_work,p,iters);
  }} else {{
    gtime=bench(greedy_work,p,iters); ctime=bench(custom_ra_work,p,iters);
  }}
  printf("PASS checks=%llu iterations=%u greedy_seconds=%.6f custom_seconds=%.6f checksum=%llu\\n",
    (unsigned long long)checks,iters,gtime,ctime,(unsigned long long)sink);
  return 0;
}}
''')
        exe = a.output / (name + (".exe" if os.name == "nt" else ".exe.out"))
        run([a.clang, "-O2", driver, *objects, "-o", exe])
        runs = [run([exe, str(i)])[0].strip() for i in range(a.repeats)]
        (a.output / f"{name}-stdout.txt").write_text("\n".join(runs) + "\n")
        entry["execution"] = runs
        entry["checks"] = 800
        report["cases"].append(entry)
        (a.output / "results.json").write_text(json.dumps(report, indent=2))
        print(name, runs[0], flush=True)


if __name__ == "__main__":
    main()
