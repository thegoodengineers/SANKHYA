#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the QPLIB MIQP warm-start A/B runner's pure logic (#494, #893).

Hermetic: nothing here downloads a file or runs the solver.

    python bench/runners/test_qplib_miqp.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qplib_miqp as runner  # noqa: E402

FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}  {detail}")
    if not condition:
        FAILURES += 1


def test_selection() -> None:
    entry = lambda kind, convex=True, **extra: {"problem_type": kind,  # noqa: E731
                                                 "listed_convex": convex, **extra}
    check(runner.is_convex_miqp(entry("CBL")), "convex objective, binaries, linear rows")
    check(runner.is_convex_miqp(entry("DML")), "diagonal convex objective, mixed columns")
    check(runner.is_convex_miqp(entry("CIB")), "integer columns, bounds only")
    check(not runner.is_convex_miqp(entry("CCL")), "a continuous QP is qplib.py's")
    check(not runner.is_convex_miqp(entry("LBL")), "a linear objective is a MILP")
    check(not runner.is_convex_miqp(entry("QBL")), "a nonconvex objective")
    check(not runner.is_convex_miqp(entry("CBL", convex=False)), "Cvx not ticked")
    check(not runner.is_convex_miqp(entry("CBQ")), "quadratic constraints")
    check(not runner.is_convex_miqp(entry("CBL", conversion_error="x")),
          "a model the converter refused")
    instances = {"QPLIB_2": entry("CBL"), "QPLIB_1": entry("DML"), "QPLIB_3": entry("CCL")}
    check(runner.select(instances, None) == ["QPLIB_1", "QPLIB_2"],
          "every convex MIQP by name, nothing else")
    check(runner.select(instances, ["QPLIB_3"]) == ["QPLIB_3"], "explicit names win")
    try:
        runner.select(instances, ["QPLIB_9"])
        check(False, "an unknown name is refused")
    except SystemExit:
        check(True, "an unknown name is refused")


def test_arms() -> None:
    cold, warm = runner.arm_options("cold"), runner.arm_options("warm")
    check("miqp_node_ipm=true" in cold and "miqp_node_ipm=true" in warm,
          "the node IPM in both arms")
    check("miqp_node_ipm_warm_start=false" in cold and "miqp_node_ipm_warm_start=true" in warm,
          "the warm start is the only difference")
    check(len(cold) == len(warm) and sum(a != b for a, b in zip(cold, warm)) == 1,
          "exactly one option differs")
    command = runner.command_for(Path("s"), Path("m.qps"), Path("st.json"), Path("x.sol"), 60.0,
                                 "warm")
    check(command[:3] == ["s", "solve", "m.qps"] and "--time-limit" in command
          and command[command.index("--time-limit") + 1] == "60.0", "the solve command")
    check(command.count("--option") == len(warm), "every arm option passed")
    try:
        runner.arm_options("hot")
        check(False, "an unknown arm is refused")
    except ValueError:
        check(True, "an unknown arm is refused")


def test_effort() -> None:
    blob = {"result": {"status": "optimal", "objective": 12.5, "message": ""},
            "model": {"rows": 5, "columns": 150},
            "effort": {"nodes": 40, "qp_node_solves": 41, "qp_node_iterations": 410,
                       "qp_node_warm_starts": 39, "qp_node_warm_fallbacks": 2,
                       "solve_seconds": 1.5}}
    flat = runner.effort_of(blob)
    check(flat["iterations_per_node"] == 10.0, "iterations per node QP solved",
          str(flat["iterations_per_node"]))
    check(flat["objective"] == 12.5 and flat["nodes"] == 40, "objective and nodes")
    blob["effort"]["qp_node_solves"] = 0
    check(runner.effort_of(blob)["iterations_per_node"] is None, "no node QP, no ratio")
    blob["result"]["status"] = "infeasible"
    check(runner.effort_of(blob)["objective"] is None, "no objective without a point")
    old = {"result": {"status": "time_limit", "objective": "inf"}, "effort": {"nodes": 3}}
    flat = runner.effort_of(old)
    check(flat["iterations_per_node"] is None and flat["qp_node_iterations"] is None,
          "a binary without the counts reports none, not zero")


def row(status: str, objective=None) -> dict:
    return {"status": status, "our_objective": objective}


def test_agreement() -> None:
    check(runner.agrees(row("optimal", 100.0), row("optimal", 100.0 + 5e-3)) is True,
          "within kMipRelativeGap of each other")
    check(runner.agrees(row("optimal", 100.0), row("optimal", 100.2)) is False,
          "2e-3 apart relative is a different answer")
    check(runner.agrees(row("optimal", 0.0), row("optimal", 5e-5)) is True,
          "absolute near zero")
    check(runner.agrees(row("infeasible"), row("infeasible")) is True, "both infeasible")
    check(runner.agrees(row("optimal", 1.0), row("infeasible")) is False,
          "a different verdict disagrees")
    check(runner.agrees(row("time_limit", 1.0), row("optimal", 2.0)) is None,
          "a limited arm is not judged")
    check(runner.agrees(row("feasible", 1.0), row("feasible", 2.0)) is None,
          "two incumbents cut off by a clock are not judged")


def full_row(instance: str, arm: str, status: str, nodes: int, iterations: int | None,
             solves: int | None, verified, agrees=None) -> dict:
    per_node = None if not solves or iterations is None else iterations / solves
    return {"instance": instance, "arm": arm, "status": status, "nodes": nodes,
            "qp_node_iterations": iterations, "iterations_per_node": per_node,
            "verified": verified, "agrees_with_cold": agrees}


def test_summary() -> None:
    rows = [
        full_row("A", "cold", "optimal", 10, 200, 10, True),
        full_row("A", "warm", "optimal", 10, 100, 10, True, True),
        full_row("B", "cold", "time_limit", 50, 1000, 50, True),
        full_row("B", "warm", "time_limit", 80, 800, 80, True, None),
        full_row("C", "cold", "optimal", 4, 40, 4, False),
        full_row("C", "warm", "optimal", 4, None, None, True, False),
    ]
    totals = runner.summary(rows)
    check(totals["paired_instances"] == 2, "C is unpaired: its warm arm has no ratio")
    check(totals["cold"]["nodes"] == 64 and totals["warm"]["nodes"] == 94, "node totals")
    check(totals["cold"]["qp_node_iterations"] == 1240
          and totals["warm"]["qp_node_iterations"] == 900, "QP iteration totals")
    expected_cold = runner.shifted_geomean([20.0, 20.0])
    expected_warm = runner.shifted_geomean([10.0, 10.0])
    check(abs(totals["cold"]["iterations_per_node_geomean"] - expected_cold) < 1e-12
          and abs(totals["warm"]["iterations_per_node_geomean"] - expected_warm) < 1e-12,
          "the geomean over the paired instances only")
    check(totals["cold"]["optimal"] == 2 and totals["cold"]["rejected"] == 1
          and totals["warm"]["verified"] == 3, "statuses and verifier verdicts")
    check((totals["agree"], totals["disagree"], totals["not_judged"]) == (1, 1, 1),
          "agreement counted from the warm rows")
    check(abs(runner.shifted_geomean([0.0, 0.0])) < 1e-12, "the shift keeps zero at zero")


def test_csv_columns() -> None:
    for column in ("instance", "arm", "qps_sha256", "our_objective", "reference_objective",
                   "abs_gap", "rel_gap", "status", "wall_seconds", "nodes",
                   "qp_node_iterations", "iterations_per_node", "verified", "git_commit",
                   "machine"):
        check(column in runner.CSV_COLUMNS, f"the CSV has {column}")
    entry = {"problem_type": "CBL", "qps": {"sha256": "ab"}, "integer_columns": 3,
             "reference_objective": 10.0, "maximize": False}
    flat = {"status": "optimal", "objective": 10.0005, "nodes": 7, "qp_node_solves": 8,
            "qp_node_iterations": 80, "iterations_per_node": 10.0, "verified": True,
            "wall_seconds": 1.25}
    made = runner.row_for("QPLIB_1", entry, "warm", flat, 60.0, {"git_commit": "abc",
                                                                   "machine": "m",
                                                                   "timestamp_utc": "t"})
    check(set(made) == set(runner.CSV_COLUMNS), "a row has exactly the CSV's columns")
    check(made["matches_reference"] is True, "5e-5 relative matches within kMipRelativeGap")
    check("miqp_node_ipm_warm_start=true" in made["options"], "the arm's options recorded")


def main() -> int:
    for test in (test_selection, test_arms, test_effort, test_agreement, test_summary,
                 test_csv_columns):
        print(test.__name__)
        test()
    print(f"\n{'OK' if FAILURES == 0 else 'FAILED'}: {FAILURES} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
