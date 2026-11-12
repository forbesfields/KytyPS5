#!/usr/bin/env python3
"""Check real sampler output without a game or a Vulkan device."""
import json
import os
import subprocess
import sys


def main():
    environment = dict(os.environ, KYTY_PERF_LOG="1")
    result = subprocess.run(
        [sys.argv[1], "--perf-reclaim-log-only"], env=environment,
        capture_output=True, text=True, timeout=45, check=True,
    )
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    records = [json.loads(line.removeprefix("KYTY_PERF "))
               for line in result.stdout.splitlines() if line.startswith("KYTY_PERF {")]
    assert len(records) == 2, f"expected two windows, got {len(records)}"
    fields = (
        "pool_draw", "pool_frame", "pool_pressure", "tess_draws", "tess_vertices",
        "tess_indirect_draws", "tess_patch", "tess_rect", "tess_corner",
        "corner_indirect", "corner_restart", "corner_capture_range", "corner_output_kind",
        "corner_side_effects", "corner_switch_off", "corner_other", "vs_store_native", "mesh_buffer",
    )
    first, second = records
    for field in fields:
        assert first[field] == 7, (field, first[field])
        assert second[field] == 0, (field, second[field])
    assert first["pool_recreate_ms"] == 125.0, first
    assert second["pool_recreate_ms"] == 0.0, second
    for record in records:
        assert record["source"] == "guest_flip" and record["v"] == 1
        assert record["window_s"] >= 5.0
        assert record["frames"] == 0 and record["frame_ms"] == []
        assert record["driver_mib"] is None  # No renderer exists in this fixture.
    print("PerfReclaimLogTests: JSON fields, ms units, window deltas, hidden overlay: PASS")


if __name__ == "__main__":
    main()
