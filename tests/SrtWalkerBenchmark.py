#!/usr/bin/env python3
"""Run through heavy.sh: python3 -I tests/SrtWalkerBenchmark.py <resource_materialization_tests>."""

import os
import re
import statistics
import subprocess
import sys


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: SrtWalkerBenchmark.py <resource_materialization_tests>")
    pattern = re.compile(
        r"SrtDrawBenchmark buffers=(\d+) mode=(\w+) repeat=(\d+) "
        r"iterations=(\d+) ns/draw=([\d.]+) checksum=(\d+)"
    )
    timings = {}
    checksums = {}
    # Reverse the order for the second pair to reduce order and warm-up bias.
    for switch in ("0", "1", "1", "0"):
        env = dict(os.environ, KYTY_PERF_SRT_WALKER=switch)
        result = subprocess.run(
            [sys.argv[1], "--benchmark-srt"], env=env, text=True, capture_output=True, check=True
        )
        print(f"KYTY_PERF_SRT_WALKER={switch}", flush=True)
        print(result.stderr, end="", flush=True)
        print(result.stdout, end="", flush=True)
        state_line = f"Performance: KYTY_PERF_SRT_WALKER={switch}"
        if result.stderr.splitlines().count(state_line) != 1:
            raise SystemExit("walker switch state was not logged exactly once")
        rows = list(pattern.finditer(result.stdout))
        if len(rows) != 30:
            raise SystemExit(f"expected 30 timing rows, got {len(rows)}")
        for row in rows:
            buffers, mode, _, iterations, ns, checksum = row.groups()
            key = (int(buffers), mode)
            checksums.setdefault(key, set()).add((iterations, checksum))
            timings.setdefault((switch, *key), []).append(float(ns))
    if any(len(values) != 1 for values in checksums.values()):
        raise SystemExit("baseline and treatment checksums differ")
    print("Summary: medians of 10 samples per path; live data changes each draw")
    for buffers, mode in sorted(checksums):
        old = timings[("0", buffers, mode)]
        new = timings[("1", buffers, mode)]
        baseline = statistics.median(old)
        treatment = statistics.median(new)
        print(
            f"buffers={buffers} mode={mode} "
            f"old_ns={baseline:.2f} new_ns={treatment:.2f} "
            f"reduction={(1 - treatment / baseline) * 100:.2f}% "
            f"old_range={min(old):.2f}..{max(old):.2f} "
            f"new_range={min(new):.2f}..{max(new):.2f}"
        )
    print("SrtWalkerBenchmark: baseline and treatment checksums match")


if __name__ == "__main__":
    main()
