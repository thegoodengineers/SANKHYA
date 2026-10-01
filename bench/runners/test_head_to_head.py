#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the head-to-head mode of compare.py (#766): rivals.py's output parsers and .sol
conversion, compare_suite.py's grading, the performance profile and the doc section.

Hermetic: no solver runs and nothing is downloaded. The model used is data/netlib/afiro.mps,
committed, read with the verifier's own reader.

    python bench/runners/test_head_to_head.py
"""
from __future__ import annotations

import csv
import math
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))
import compare_suite  # noqa: E402
import head_to_head_doc  # noqa: E402
import perf_profile  # noqa: E402
import rivals  # noqa: E402
from verify_solution_mps import parse_mps  # noqa: E402
from verify_solution_sol import parse_sol  # noqa: E402

FAILURES = 0
AFIRO = REPO_ROOT / "data" / "netlib" / "afiro.mps"


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}  {detail}")
    if not condition:
        FAILURES += 1


def sgm(values, shift):
    return math.exp(sum(math.log(v + shift) for v in values) / len(values)) - shift


print("output parsers")
CLP_SOLU = """Optimal - objective value      -464.75314
      0 R09                    0             -0.62857143
      1 DEDO31R                0                       0
**    2 X05                   80             -0.34477143
      0 X01                   80                       0
      1 DEDO3 11               0              0.12
"""
rows, cols = rivals._clp_names(CLP_SOLU)
check(rows == ["R09", "DEDO31R", "X05"], "Clp row names, `**` marker dropped", str(rows))
check(cols == ["X01", "DEDO3 11"], "Clp column names, restart at index 0 and a spaced name",
      str(cols))

GLPK_REPORT = """Problem:    AFIRO
   No.   Row name   St   Activity     Lower bound   Upper bound    Marginal
------ ------------ -- ------------- ------------- ------------- -------------
     1 R09          NS             0             0             =     -0.628571
     2 DEDO3 1R     NS             0             0             =         < eps
     3 AVERYLONGROWNAME
                    B             80                          80

   No. Column name  St   Activity     Lower bound   Upper bound    Marginal
------ ------------ -- ------------- ------------- ------------- -------------
     1 X01          B             80             0
"""
rows, cols = rivals._glpk_names(GLPK_REPORT)
check(rows == ["R09", "DEDO3 1R", "AVERYLONGROWNAME"],
      "GLPK row names: fixed twelve columns keep a space, a long name stands alone", str(rows))
check(cols == ["X01"], "GLPK column names", str(cols))

text = "Clp version\nOptimal - objective value      -464.75314\n"
check(rivals.CLP_STATUS.findall(text)[-1][0] == "Optimal", "Clp status line")
check(rivals.CLP_STATUS.findall("Stopped on time - objective value 3\n")[-1][0]
      == "Stopped on time", "Clp time limit line")
check(rivals.supports("glpk", "qp") is False and rivals.supports("glpk", "lp"),
      "GLPK is recorded as unsupported on QPs, not skipped")
check(rivals.run("glpk", AFIRO, "qp", 1.0, Path("unused"))["status"] == "unsupported",
      "run() says unsupported without starting glpsol")

# saveSolution layout: two ints, then objective, row activities, row duals, columns, costs.
blob = struct.pack("ii", 1, 2) + struct.pack("7d", -1.5, 3.0, -0.25, 1.0, 2.0, 0.5, 0.75)
m, n = struct.unpack_from("ii", blob, 0)
values = struct.unpack_from(f"{1 + 2 * m + 2 * n}d", blob, 8)
check((m, n, values[0], values[1 + 2 * m + n:]) == (1, 2, -1.5, (0.5, 0.75)),
      "Clp saveSolution offsets", str(values))

print(".sol conversion and reconcile")
model = parse_mps(AFIRO)
with tempfile.TemporaryDirectory() as tmp:
    sol = Path(tmp) / "x.sol"
    rivals.write_sol(sol, solver="glpk", status="optimal", objective=-2.5,
                     columns={"X01": (1.0, 0.0), "A B": (2.0, -1.0)},
                     rows={"R09": (0.5, 0.25)}, message="note")
    parsed = parse_sol(sol)
    check(parsed.status == "optimal" and parsed.header_float("objective") == -2.5,
          "header round-trips through the verifier's reader")
    check(parsed.col_value.get("A B") == 2.0 and parsed.col_dual.get("A B") == -1.0,
          "a spaced name is quoted and read back whole")

    # The afiro point with names SCIP-style and one row missing.
    cols = {n.replace(" ", "_"): (0.0, 0.0) for n in model.col_names}
    rows = {n: (0.0, 0.0) for n in model.row_names[1:]}
    rivals.write_sol(sol, solver="scip", status="feasible", objective=0.0, columns=cols,
                     rows=rows)
    notes = rivals.reconcile(sol, model, "scip")
    parsed = parse_sol(sol)
    check(all(n in parsed.row_activity for n in model.row_names),
          "a row the solver did not report is filled", str(notes))
    check(not notes, "SCIP's filled rows are not reported as a change (it has no duals)",
          str(notes))
    rivals.write_sol(sol, solver="cbc-clp", status="optimal", objective=0.0, columns=cols,
                     rows=rows)
    notes = rivals.reconcile(sol, model, "cbc-clp")
    check(any("filled" in n for n in notes), "for a solver with duals the fill is noted",
          str(notes))


class FakeModel:
    """Two columns, one row, a constant: enough for the name and constant rules."""
    col_names = ["DEDO3 11", "Y"]
    col_index = {"DEDO3 11": 0, "Y": 1}
    row_names = ["R"]
    row_index = {"R": 0}
    num_rows = 1
    entries = [[(0, 1.0)], [(0, 1.0)]]
    objective_offset = 7.0
    col_cost = [1.0, 2.0]

    def quadratic_objective(self, x):
        return 0.0


fake = FakeModel()
with tempfile.TemporaryDirectory() as tmp:
    sol = Path(tmp) / "x.sol"
    rivals.write_sol(sol, solver="glpk", status="optimal", objective=-4.0,
                     columns={"DEDO311": (1.0, 0.0), "Y": (1.0, 0.0)}, rows={"R": (2.0, 0.0)})
    notes = rivals.reconcile(sol, fake, "glpk")
    parsed = parse_sol(sol)
    check("DEDO3 11" in parsed.col_value, "Clp/GLPK's space-stripped name mapped back")
    check(parsed.header_float("objective") == 10.0,
          "GLPK's objective restated: c'x - c0 = -4 becomes c'x + c0 = 10",
          str(parsed.header_float("objective")))
    check(any("GLPK's constant convention" in n for n in notes), "and the note says so")
    check(compare_suite.point_objective(fake, sol, with_constant=False) == 3.0
          and compare_suite.point_objective(fake, sol, with_constant=True) == 10.0,
          "the graded objective is recomputed from the point, with or without c0")

print("grading")
suite = compare_suite.SUITES["netlib"]
with tempfile.TemporaryDirectory() as tmp:
    missing = Path(tmp) / "none.sol"
    verdict = compare_suite.grade("highs", suite, model, AFIRO, missing,
                                  {"status": "time_limit", "message": ""}, -464.75, None, 120)
    check(not verdict["counted_for_time"] and verdict["verification"] == "not-run",
          "a time limit is not verified and not counted")
    verdict = compare_suite.grade("highs", suite, model, AFIRO, missing,
                                  {"status": "optimal", "message": ""}, -464.75, None, 120)
    check(verdict["independently_verified"] is False and not verdict["counted_for_time"],
          "optimal with no solution file is a rejection, not a pass")
    verdict = compare_suite.grade("glpk", compare_suite.SUITES["maros-meszaros"], model, AFIRO,
                                  missing, {"status": "unsupported"}, 0.0, None, 60)
    check(verdict["verification"] == "unsupported", "unsupported stays unsupported")
check(compare_suite.SUITES["netlib"]["time_limit"] == 120.0
      and compare_suite.SUITES["kennington"]["time_limit"] == 120.0
      and compare_suite.SUITES["maros-meszaros"]["time_limit"] == 60.0,
      "suite limits: 120 s LP, 60 s QP")
check(compare_suite.MATCH_TOLERANCE == 1e-6, "the match tolerance is the runners' 1e-6")
tag = compare_suite.machine_tag("cloud container")
check(tag.startswith("cloud container;") and "cores" in tag and "RAM" in tag,
      "machine tag names the kind, CPU, cores and RAM", tag)

print("performance profile")
ROWS = [
    {"instance": "a", "solver": "sankhya", "counted_for_time": "1", "solver_seconds": "2"},
    {"instance": "a", "solver": "highs", "counted_for_time": "1", "solver_seconds": "1"},
    {"instance": "b", "solver": "sankhya", "counted_for_time": "1", "solver_seconds": "0.01"},
    {"instance": "b", "solver": "highs", "counted_for_time": "1", "solver_seconds": "0.02"},
    {"instance": "c", "solver": "sankhya", "counted_for_time": "0", "solver_seconds": "5"},
    {"instance": "c", "solver": "highs", "counted_for_time": "1", "solver_seconds": "4"},
]
r = perf_profile.ratios(ROWS, ["sankhya", "highs"])
check(r["sankhya"] == [2.0, 1.0, math.inf], "ratios: 2x, a tie below the floor, a failure",
      str(r["sankhya"]))
check(perf_profile.profile(r["highs"], 1.0) == 1.0
      and abs(perf_profile.profile(r["sankhya"], 2.0) - 2 / 3) < 1e-12,
      "rho(tau) counts every instance in the denominator")
image = perf_profile.svg("t", r)
check(image.startswith("<svg") and image.count("<polyline") == 2, "one curve per solver")

print("per-instance wins")
EXTRA = [
    {"instance": "d", "solver": "sankhya", "counted_for_time": "1", "solver_seconds": "1.05"},
    {"instance": "d", "solver": "highs", "counted_for_time": "1", "solver_seconds": "1"},
    {"instance": "e", "solver": "sankhya", "counted_for_time": "1", "solver_seconds": "1"},
    {"instance": "e", "solver": "highs", "counted_for_time": "0", "solver_seconds": "9"},
    {"instance": "f", "solver": "sankhya", "counted_for_time": "0", "solver_seconds": "1"},
    {"instance": "f", "solver": "highs", "counted_for_time": "0", "solver_seconds": "1"},
]
s = head_to_head_doc.per_instance(ROWS + EXTRA, "highs")
check((s["wins"], s["ties"], s["losses"], s["compared"]) == (1, 2, 2, 5),
      "2x slower and a failure lose, under the floor and within 10% tie, a rival's failure "
      "wins, neither counting is left out", str(s))
check(s["median"] == 1.05 and s["median_raw"] == 1.05,
      "median ratio over the three both counted on: floored 2, 1, 1.05; raw 2, 0.5, 1.05",
      str(s))
check(head_to_head_doc.per_instance(ROWS, "glpk") is None, "an absent rival gets no row")

print("doc section")
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "head-to-head-netlib-abc1234.csv"
    base = {c: "" for c in compare_suite.CSV_COLUMNS}
    lines = []
    for inst, solver, status, match, ver, counted, t in [
            ("afiro", "sankhya", "optimal", "1", "1", "1", "0.5"),
            ("afiro", "glpk", "optimal", "1", "1", "1", "0.1"),
            ("truss", "sankhya", "optimal", "1", "1", "1", "3"),
            ("truss", "glpk", "time_limit", "0", "", "0", "120"),
            ("e226", "sankhya", "optimal", "0", "0", "0", "1"),
            ("e226", "glpk", "optimal", "1", "1", "1", "0.2")]:
        lines.append({**base, "suite": "netlib", "instance": inst, "solver": solver,
                      "status": status, "matches_reference": match,
                      "independently_verified": ver, "counted_for_time": counted,
                      "solver_seconds": t, "time_limit": "120", "verification": "primal+dual",
                      "relative_gap": "0.5", "verifier_message": "[FAIL] duality",
                      "git_commit": "abc1234", "machine": "m", "solver_version": "v"})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=compare_suite.CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(lines)
    img = Path(tmp) / "img"
    original = perf_profile.IMG_DIR
    perf_profile.write_figure.__defaults__ = (img,)
    try:
        text = head_to_head_doc.section({"netlib": path}, sgm)
    finally:
        perf_profile.write_figure.__defaults__ = (original,)
    check("The 2 LPs netlib.org ships as EMPS files" in text,
          "the headline excludes the generated truss and stocfor3")
    check("`truss` and `stocfor3`" in text, "and reports them beside it")
    check("`e226` (optimal, rejected by the verifier)" in text, "a rejection is named")
    check("| GLPK | 2 | 0 | 0 | 2 | 5.00x | 5.00x |" in text,
          "the per-instance row: afiro 5x slower, e226 not counted, both losses")
    check("### 4a.3 Maros-Meszaros QP" in text and "no numbers" in text,
          "a missing suite states no numbers")
    check((img / "profile-netlib.svg").exists(), "the profile figure is written")

print(f"\n{'all passed' if FAILURES == 0 else f'{FAILURES} FAILED'}")
sys.exit(1 if FAILURES else 0)
