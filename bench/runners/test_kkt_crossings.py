#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the relative-KKT crossing columns (#486): what the runners write for a PDHG
row and for any other engine, the ordering check, and the two generated doc tables. Pure
Python, no solve.

    python bench/runners/test_kkt_crossings.py
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kkt_crossings  # noqa: E402
import make_benchmarks_doc  # noqa: E402
import mittelmann  # noqa: E402
import netlib  # noqa: E402
import pdhg_report  # noqa: E402

FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}  {detail}")
    if not condition:
        FAILURES += 1


def blob(algorithm: str, **effort) -> dict:
    """A stats JSON as the writer lays it out: every engine carries the six fields."""
    base = {"kkt_1e4_seconds": "nan", "kkt_1e6_seconds": "nan", "kkt_1e8_seconds": "nan"}
    base.update(effort)
    return {"result": {"algorithm": algorithm}, "effort": base}


def test_cells() -> None:
    pdhg = kkt_crossings.crossings(blob("pdhg-cpu", kkt_1e4_seconds=0.5, kkt_1e6_seconds=1.25,
                                        kkt_1e4_iterations=40, kkt_1e6_iterations=120,
                                        kkt_1e8_iterations=-1))
    check(pdhg == {"kkt_1e4_seconds": 0.5, "kkt_1e6_seconds": 1.25, "kkt_1e8_seconds": "nan",
                   "kkt_1e4_iterations": 40, "kkt_1e6_iterations": 120,
                   "kkt_1e8_iterations": -1},
          "a PDHG row keeps the times and iterations, and the writer's nan and -1 for a level "
          "never reached", str(pdhg))
    merged = kkt_crossings.crossings(blob("pdhg-cpu+ipm", kkt_1e4_seconds=0.1))
    check(merged["kkt_1e4_seconds"] == 0.1, "a polished PDHG run is still a PDHG row")
    simplex = kkt_crossings.crossings(blob("simplex-dual"))
    check(simplex == {k: "" for k in kkt_crossings.ALL_COLUMNS},
          "any other engine gets blanks, not a 'never reached' it never attempted")
    for engine in ("pdhg-cuda-multi", "pdhg-cuda-multi+ipm"):
        cells = kkt_crossings.crossings(blob(engine))
        check(cells == {k: "" for k in kkt_crossings.ALL_COLUMNS},
              f"{engine} records no crossings, so its row is blank, not 'never reached'")
    for engine in ("pdhg-cuda", "pdhg-cuda+ipm"):
        cells = kkt_crossings.crossings(blob(engine, kkt_1e4_seconds=0.2))
        check(cells["kkt_1e4_seconds"] == 0.2, f"{engine} records its crossings")
    bare = kkt_crossings.crossings({"result": {"algorithm": "pdhg-cpu"}, "effort": {}})
    check(all(bare[k] == "nan" for k in kkt_crossings.COLUMNS)
          and all(bare[k] == -1 for k in kkt_crossings.ITERATION_COLUMNS),
          "a PDHG blob without the fields reads as never reached")


def test_consistency() -> None:
    row = dict(zip(kkt_crossings.COLUMNS, ("0.1", "0.4", "2.0")))
    check(kkt_crossings.consistent(row), "increasing crossings are consistent")
    row = dict(zip(kkt_crossings.COLUMNS, ("0.1", "0.1", "0.1")))
    check(kkt_crossings.consistent(row), "equal crossings (one evaluation) are consistent")
    row = dict(zip(kkt_crossings.COLUMNS, ("0.1", "0.4", "nan")))
    check(kkt_crossings.consistent(row), "a tail of levels never reached is consistent")
    row = dict(zip(kkt_crossings.COLUMNS, ("0.5", "0.4", "2.0")))
    check(not kkt_crossings.consistent(row), "a tighter level before a looser one is not")
    row = dict(zip(kkt_crossings.COLUMNS, ("nan", "0.4", "2.0")))
    check(not kkt_crossings.consistent(row), "a looser level missed but a tighter one hit is not")
    check(kkt_crossings.consistent({k: "" for k in kkt_crossings.COLUMNS}),
          "an all-blank (other engine) row is consistent")
    check(not kkt_crossings.consistent({"kkt_1e4_seconds": "0.1"}),
          "a partly blank row is not")
    check(kkt_crossings.cell_text("nan") == "not reached" and kkt_crossings.cell_text("") == "-"
          and kkt_crossings.cell_text("0.01234") == "0.0123", "cell text")


def test_runner_columns() -> None:
    for module in (pdhg_report, netlib, mittelmann):
        missing = [k for k in kkt_crossings.ALL_COLUMNS if k not in module.CSV_COLUMNS]
        check(not missing, f"{module.__name__}.py writes the six crossing columns",
              str(missing))
    for column in ("independently_verified", "verifier_message", "instance_sha256",
                   "absolute_error"):
        check(column in pdhg_report.CSV_COLUMNS, f"pdhg_report.py writes {column}")
    check(pdhg_report.CSV_COLUMNS[:15] == [
        "instance", "algorithm", "tolerance", "restarts_enabled", "status", "objective",
        "published_objective", "relative_error", "iterations", "seconds", "reached_tolerance",
        "git_commit", "machine", "timestamp_utc", "solver_options"],
        "the columns an older pdhg CSV has keep their order; the new ones are appended")
    check(pdhg_report.verdict(Path("x.mps"), Path("missing.sol"), "iteration_limit") == ("", ""),
          "a limit is not handed to the verifier: it claims no point")


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def test_pdhg_doc_table() -> None:
    common = {"published_objective": "1.0", "git_commit": "abc1234", "machine": "m"}
    rows = [
        {**common, "instance": "afiro", "algorithm": "simplex", "status": "optimal",
         "objective": "1.0"},
        {**common, "instance": "afiro", "algorithm": "pdhg", "tolerance": "0.0001",
         "restarts_enabled": "1", "status": "optimal", "objective": "1.0", "iterations": "9",
         "kkt_1e4_seconds": "0.9", "kkt_1e6_seconds": "nan", "kkt_1e8_seconds": "nan",
         "independently_verified": "1"},
        {**common, "instance": "afiro", "algorithm": "pdhg", "tolerance": "1e-08",
         "restarts_enabled": "1", "status": "optimal", "objective": "1.0", "iterations": "40",
         "kkt_1e4_seconds": "0.001", "kkt_1e6_seconds": "0.002", "kkt_1e8_seconds": "nan",
         "independently_verified": "0"},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pdhg-abc1234.csv"
        write_csv(path, pdhg_report.CSV_COLUMNS, rows)
        text = make_benchmarks_doc.pdhg_section(path)
        check("Relative KKT crossings in the 1e-08 run" in text,
              "the PDHG section carries the crossing table for the tightest run")
        check("| `afiro` | 0.001 | 0.002 | not reached | optimal | **NO** |" in text,
              "the tightest run's crossings, status and verdict, a missed level named")
        old = [{k: v for k, v in r.items() if not k.startswith("kkt_")} for r in rows]
        path_old = Path(tmp) / "pdhg-old.csv"
        write_csv(path_old, pdhg_report.CSV_COLUMNS[:15], old)
        check("Relative KKT crossings" not in make_benchmarks_doc.pdhg_section(path_old),
              "a CSV from before #486 renders as it did")


def test_feasibility_standard_table() -> None:
    columns = ["instance", "status", "independently_verified", "solver_seconds",
               *kkt_crossings.COLUMNS]
    default_rows = [{"instance": "brazil3", "status": "time_limit"},
                    {"instance": "qap15", "status": "optimal", "independently_verified": "1"}]
    pdhg_rows = [
        {"instance": "brazil3", "status": "optimal", "independently_verified": "1",
         "kkt_1e4_seconds": "3.0", "kkt_1e6_seconds": "12.5", "kkt_1e8_seconds": "40.0"},
        {"instance": "qap15", "status": "time_limit", "kkt_1e4_seconds": "0.87",
         "kkt_1e6_seconds": "nan", "kkt_1e8_seconds": "nan"},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        default = Path(tmp) / "mittelmann-a.csv"
        pdhg = Path(tmp) / "mittelmann-pdhg-a.csv"
        write_csv(default, columns, default_rows)
        write_csv(pdhg, columns, pdhg_rows)
        text = make_benchmarks_doc.mittelmann_engines_section(default, pdhg, None)
        check("feasibility-page standard" in text and "not an optimal basis" in text,
              "labelled as the feasibility standard and said not to be an optimal basis")
        check("| `brazil3` | 12.5 | optimal | yes |" in text, "a reached level with its verdict")
        check("| `qap15` | not reached | time_limit | - |" in text, "a level never reached")
        check("**1 of 2**" in text, "the count reached")
        write_csv(pdhg, ["instance", "status", "independently_verified", "solver_seconds"],
                  [{k: v for k, v in r.items() if not k.startswith("kkt_")} for r in pdhg_rows])
        text = make_benchmarks_doc.mittelmann_engines_section(default, pdhg, None)
        check("feasibility-page standard" not in text,
              "no table from a PDHG CSV that carries no crossing")


def main() -> int:
    for test in (test_cells, test_consistency, test_runner_columns, test_pdhg_doc_table,
                 test_feasibility_standard_table):
        print(test.__name__)
        test()
    print(f"\n{'OK' if FAILURES == 0 else f'{FAILURES} FAILED'}")
    return 0 if FAILURES == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
