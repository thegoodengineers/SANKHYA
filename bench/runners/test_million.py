#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the million-row runner, its transport generator and its doc section (#751).

Hermetic by default: the generator's KKT construction, its determinism, the runner's
attribution rule and the section's rendering. With SANKHYA_BINARY set (CI's after-tests leg
sets it to build/sankhya), a small transport instance also goes through the runner's own
solve-and-verify path, so the pipeline the million-row CSV comes from is exercised end to
end at a size a pull request can afford.

    python3 bench/runners/test_million.py
"""
from __future__ import annotations

import csv
import hashlib
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import generate_transport_lp  # noqa: E402
import million  # noqa: E402
import million_doc  # noqa: E402


def test_transport_kkt_and_determinism() -> None:
    inst = generate_transport_lp.build(300, 250, 4, 8, seed=3)
    optimum = generate_transport_lp.verify(inst)
    assert optimum > 0, optimum
    assert len(inst["arc_source"]) == 250 * 4
    # Every arc stays inside its sink's window: the normal equations' band.
    for j in range(250):
        first = min(j * 300 // 250, 300 - 8)
        for a in range(inst["sink_start"][j], inst["sink_start"][j + 1]):
            assert first <= inst["arc_source"][a] < first + 8
    with tempfile.TemporaryDirectory() as tmp:
        digests = []
        for name in ("a.mps", "b.mps"):
            path = Path(tmp) / name
            generate_transport_lp.write_mps(
                generate_transport_lp.build(300, 250, 4, 8, seed=3), optimum, path, 3)
            digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
        assert digests[0] == digests[1], "the same seed must write the same bytes"


def test_attribution_rule() -> None:
    base = {"verified": "0", "status": "time_limit", "message": "", "peak_rss_mb": "100"}
    assert million.attribute({**base, "verified": "1", "status": "optimal"}, 16000) == ""
    assert million.attribute(base, 16000) == "iterations"
    assert million.attribute({**base, "status": "killed"}, 16000) == "memory"
    assert million.attribute({**base, "peak_rss_mb": "15000"}, 16000) == "memory"
    declined = "the interior-point polish declined: factor would hold 3e9 nonzeros"
    assert million.attribute({**base, "message": declined}, 16000) == "polish"
    fill = "the factor would hold 3e9 nonzeros, over ipm_max_factor_nonzeros"
    assert million.attribute({**base, "status": "not_solved", "message": fill}, 16000) == "fill"
    assert million.attribute({**base, "status": "optimal"}, 16000) == "verifier"


def test_section_renders_both_tables() -> None:
    row = {key: "" for key in million.CSV_COLUMNS}
    row.update({"family": "transport", "rows": "1000000", "columns": "2000000",
                "nonzeros": "4000000", "arm": "ipm", "route": "ipm", "status": "optimal",
                "relative_error": "0.0", "verified": "1", "wall_seconds": "812.4",
                "peak_rss_mb": "5120", "git_commit": "abc1234", "machine": "m",
                "time_limit": "3600"})
    stuck = {**row, "arm": "pdhg", "route": "pdhg-cpu", "status": "time_limit",
             "verified": "0", "attribution": "iterations", "iterations": "90000",
             "primal_infeasibility": "1e-5", "dual_infeasibility": "2e-4",
             "message": "time limit | reached"}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "million-cpu-abc1234.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=million.CSV_COLUMNS)
            writer.writeheader()
            writer.writerows([row, stuck])
        text = million_doc.million_section(path)
        assert "**1 of 2** arms end `optimal` and verified." in text, text
        assert "| transport | `pdhg` | **iterations** |" in text, text
        assert "time limit / reached" in text, "a pipe in a message must not split the table"
        line = million_doc.readme_scale_line(path)
        assert "transport (1,000,000 rows) `optimal` and verified by `ipm`" in line, line
    assert "No `million-cpu-*.csv`" in million_doc.million_section(None)


def test_runner_solves_and_verifies_a_small_instance() -> None:
    binary = os.environ.get("SANKHYA_BINARY")
    if not binary:
        print("  (SANKHYA_BINARY not set: the end-to-end solve is skipped)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        million.FAMILIES["transport"] = ["generate_transport_lp.py", "--sources", "600",
                                         "--sinks", "500", "--seed", "5"]
        instance = million.generate("transport", work)
        assert instance["rows"] == 1100 and instance["columns"] == 2000, instance
        for arm in ("ipm", "pdhg"):
            row = million.solve(Path(binary), instance, arm, 120.0, work, [], 0.0)
            assert row["status"] == "optimal", row
            assert row["verified"] == "1", row
            objective = float(row["our_objective"])
            optimum = float(instance["optimum"])
            assert abs(objective - optimum) <= 1e-6 * max(1.0, abs(optimum)), (objective,
                                                                               optimum)
            assert float(row["peak_rss_mb"]) > 0, row


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"ok  {name}")
