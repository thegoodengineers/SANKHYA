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
    # The refinery with the supernodal factor at b561fba: killed by the runner at 6,000 s
    # for a 3,600 s limit, 11,806 MB peak on a 16,094 MB machine.
    overran = {**base, "status": "killed", "time_limit": "3600", "wall_seconds": "6000.325",
               "peak_rss_mb": "11806", "message": "exit -9; "}
    assert million.attribute(overran, 16094) == "overran"
    assert million.attribute({**overran, "peak_rss_mb": "15500"}, 16094) == "memory"
    assert million.attribute({**overran, "wall_seconds": "900"}, 16094) == "memory"
    assert million.attribute({**base, "peak_rss_mb": "15000"}, 16000) == "memory"
    declined = "the interior-point polish declined: factor would hold 3e9 nonzeros"
    assert million.attribute({**base, "message": declined}, 16000) == "polish"
    fill = "the factor would hold 3e9 nonzeros, over ipm_max_factor_nonzeros"
    assert million.attribute({**base, "status": "not_solved", "message": fill}, 16000) == "fill"
    assert million.attribute({**base, "status": "optimal"}, 16000) == "verifier"
    stalled = ("the interior-point iteration stalled after 21 iterations (steps 3.4e-11 / "
               "1.4e-11); the best iterate is reported as a feasible point")
    assert million.attribute({**base, "status": "feasible", "message": stalled}, 16000) == (
        "stall")
    crossed = stalled + ("; crossover did not reach a vertex (time_limit after 198900 "
                         "pivots, 3576.36s), the interior point's answer stands")
    assert million.attribute({**base, "status": "feasible", "message": crossed}, 16000) == (
        "crossover")


def test_reattribute_rewrites_only_the_attribution() -> None:
    row = {key: "x" for key in million.CSV_COLUMNS}
    row.update({"family": "transport", "arm": "ipm", "status": "feasible", "verified": "1",
                "peak_rss_mb": "1865", "attribution": "iterations",
                "machine": "cloud container; Linux-x86_64; cpu; 4 cores; 15.7 GiB RAM",
                "message": "the interior-point iteration stalled after 21 iterations"})
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "million-cpu-abc1234.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=million.CSV_COLUMNS)
            writer.writeheader()
            writer.writerow(row)
        million.reattribute(path)
        with path.open(newline="", encoding="utf-8") as handle:
            (after,) = list(csv.DictReader(handle))
        assert after["attribution"] == "stall", after
        assert {k: v for k, v in after.items() if k != "attribution"} == {
            k: v for k, v in row.items() if k != "attribution"}


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


def test_latest_csv_keeps_rows_with_arm_options() -> None:
    row = {key: "" for key in million.CSV_COLUMNS}
    row.update({"git_commit": "abc1234", "solver_options": "algorithm=pdhg",
                "timestamp_utc": "2026-09-30T00:00:00+00:00"})
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for name in ("million-cpu-abc1234.csv", "million-cpu-cap-experiment.csv"):
            with (directory / name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=million.CSV_COLUMNS)
                writer.writeheader()
                writer.writerow(row)
        chosen = million_doc.latest_csv(directory)
        assert chosen is not None and chosen.name == "million-cpu-abc1234.csv", chosen
        assert million_doc.latest_csv(directory / "empty") is None


def test_experiments_are_listed_apart_with_their_options() -> None:
    row = {key: "" for key in million.CSV_COLUMNS}
    row.update({"family": "staircase", "arm": "ipm", "status": "time_limit",
                "relative_error": "1.5e-4", "verified": "0", "wall_seconds": "3613",
                "peak_rss_mb": "4347", "iterations": "65",
                "solver_options": "algorithm=ipm ipm_max_factor_nonzeros=600000000"})
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        assert million_doc.experiments_section(directory) == ""
        for name in ("million-cpu-abc1234.csv", "million-factor-cap-abc1234.csv"):
            with (directory / name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=million.CSV_COLUMNS)
                writer.writeheader()
                writer.writerow(row)
        text = million_doc.experiments_section(directory)
        assert "`million-factor-cap-abc1234.csv`" in text, text
        assert "million-cpu-abc1234" not in text, "the default run is never an experiment"
        assert "ipm_max_factor_nonzeros=600000000" in text, text


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
