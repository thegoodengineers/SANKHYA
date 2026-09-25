#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the QPLIB fetcher, the .qplib to QPS conversion, the runner's verdicts and the
doc section (#492).

Hermetic: nothing here downloads a file or runs the solver. The fixtures in testdata/qplib/
are QPLIB's own, fetched from https://qplib.zib.de on 2026-09-25 (QPLIB is licensed under
CC-BY 4.0; Furini et al., Mathematical Programming Computation 11, 2019):

*   instances-excerpt.html, QPLIB_8845-excerpt.html, index-excerpt.html,
    qplib-solu-excerpt.txt: excerpts of instances.html, QPLIB_8845.html, index.html and
    qplib.solu, each headed by its URL and the sha256 of the whole page as fetched;
*   QPLIB_0018.qplib and QPLIB_0018.sol: https://qplib.zib.de/qplib/QPLIB_0018.qplib and
    https://qplib.zib.de/sol/QPLIB_0018.sol byte for byte (their sha256 is checked below).
    QPLIB_0018 is not in the convex selection (its objective is indefinite); it is here
    because it is small and has a published point, which is all the conversion check needs.

    python bench/runners/test_qplib.py
"""
from __future__ import annotations

import csv
import hashlib
import math
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "tools"))
import fetch_qplib as fetch  # noqa: E402
import qplib as runner  # noqa: E402
import qplib_doc as doc  # noqa: E402
import qplib_format as fmt  # noqa: E402
from verify_solution_mps import parse_mps  # noqa: E402

DATA = HERE / "testdata" / "qplib"
FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}  {detail}")
    if not condition:
        FAILURES += 1


def close(a: float | None, b: float, rel: float = 1e-12) -> bool:
    return a is not None and abs(a - b) <= rel * max(1.0, abs(b))


def objective_via_qps(model: fmt.QplibModel, x: list[float]) -> tuple[float, object]:
    """Write the QPS, re-read it with the independent verifier's reader, evaluate at x."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.qps"
        path.write_text(fmt.to_qps(model), encoding="utf-8", newline="\n")
        reread = parse_mps(path)
    point = [0.0] * reread.num_cols
    for j in range(model.n):
        point[reread.col_index[fmt.column_name(j)]] = x[j]
    value = (reread.objective_offset + sum(c * v for c, v in zip(reread.col_cost, point))
             + reread.quadratic_objective(point))
    return value, reread


# ---- the site's pages ---------------------------------------------------------------------

def test_listing() -> None:
    rows = fetch.parse_listing((DATA / "instances-excerpt.html").read_text(encoding="utf-8"))
    by_name = {row["name"]: row for row in rows}
    check(len(rows) == 9, "every row of the excerpt is read", str(len(rows)))
    picked = sorted(row["name"] for row in rows if fetch.selected(row))
    check(picked == ["QPLIB_8790", "QPLIB_8845", "QPLIB_8938", "QPLIB_9002"],
          "selected: convex, O in {C, D}, V = C, C in {N, B, L}", str(picked))
    check(by_name["QPLIB_2456"]["convex"] and not fetch.selected(by_name["QPLIB_2456"]),
          "a convex LCD instance (quadratic constraints) is not selected")
    check(not by_name["QPLIB_0018"]["convex"], "Cvx '-' reads as not convex")
    row = by_name["QPLIB_8845"]
    check((row["o"], row["v"], row["c"], row["nvars"], row["ncons"], row["nz"]) ==
          ("C", "C", "L", 1546, 777, 10999), "QPLIB_8845's cells as the site shows them")
    check(row["qplib_path"] == "qplib/QPLIB_8845.qplib", "the file path is the listing's link")
    check(by_name["QPLIB_8790"]["ncons"] == 0 and by_name["QPLIB_8790"]["c"] == "B",
          "a box-constrained row has 0 constraints")
    changed = (DATA / "instances-excerpt.html").read_text(encoding="utf-8").replace(
        ">Cvx</A>", ">Convex</A>")
    try:
        fetch.parse_listing(changed)
        check(False, "a renamed column is refused")
    except ValueError:
        check(True, "a renamed column is refused")


def test_pages() -> None:
    fields = fetch.parse_instance_page(
        (DATA / "QPLIB_8845-excerpt.html").read_text(encoding="utf-8"))
    check(fields["probtype"] == "CCL" and fields["solobjvalue_text"] == "10907992.49000000",
          "the page's probtype and solobjvalue", fields["solobjvalue_text"])
    check(fields["sol_path"] == "sol/QPLIB_8845.sol" and fields["nobjquadnz"] == "58114"
          and fields["solinfeasibility"] == "3.4106e-13", "the solution link and the counts")
    rows = fetch.parse_listing((DATA / "instances-excerpt.html").read_text(encoding="utf-8"))
    row = next(r for r in rows if r["name"] == "QPLIB_8845")
    check(fetch.page_consistent(row, fields) == [], "page and listing agree")
    check(fetch.page_consistent(dict(row, c="D"), fields) != [], "a disagreement is reported")
    home = fetch.parse_home((DATA / "index-excerpt.html").read_text(encoding="utf-8"))
    check((home["discrete"], home["continuous"]) == (319, 134), "index.html's counts", str(home))
    check(home["licence_statement"] == "QPLIB is licensed under CC-BY 4.0." and
          home["licence_url"] == "https://creativecommons.org/licenses/by/4.0/",
          "the licence as the site states it")
    solu = fetch.parse_solu((DATA / "qplib-solu-excerpt.txt").read_text(encoding="utf-8"))
    check(solu["QPLIB_8845"][:2] == ("best", 10907992.4900000002235174), "a solu value",
          str(solu["QPLIB_8845"]))
    check(solu["QPLIB_9002"] == ("unkn", None, None), "an =unkn= line has no value")
    check(fetch.references_agree("10907992.49000000", solu["QPLIB_8845"][1]) and
          fetch.references_agree("-0.00015624", solu["QPLIB_8790"][1]) and
          not fetch.references_agree("-0.00015625", solu["QPLIB_8790"][1]) and
          fetch.references_agree(None, None) and not fetch.references_agree("1.0", None),
          "page and solu agree to the page's eight decimals, and only then")


# ---- the format ---------------------------------------------------------------------------

# The Appendix B example of the paper made continuous (x3 in [0, 1] instead of binary), with
# a D exponent, comment lines and a two-sided second constraint 1 <= x1 + x3 <= 3.
EXAMPLE = """\
! ---------------
! example problem
! ---------------
EXAMPLE # problem name
CCL # continuous, linear constraints
Minimize # minimize the objective function
3 # variables
2 # general linear constraints
5 # nonzeros in lower triangle of Q^0
1 1 2.0
2 1 -1.0
2 2 2.0
3 2 -1.0
3 3 2.0
-0.2 default value for entries in b_0
1 # non default entries in b_0
2 -0.4D0
0.5 value of q^0
4 # nonzeros in vectors b^i (i=1,...,m)
1 1 1.0
1 2 1.0
2 1 1.0
2 3 1.0
1.0E+20 infinity
1.0 default value for entries in c_l
0 # non default entries in c_l
1.0E+20 default value for entries in c_u
1 # non default entries in c_u
2 3.0
0.0 default value for entries in l
0 # non default entries in l
1.0 default value for entries in u
1 # non default entries in u
2 2.0
1.0 default value for initial values for x
0 # non default entries in x
0.0 default value for initial values for y
0 # non default entries in y
0.0 default value for initial values for z
0 # non default entries in z
0 # non default names for variables
0 # non default names for constraints
"""


def test_hand_computed() -> None:
    model = fmt.parse(EXAMPLE)
    check((model.n, model.m, len(model.q0), model.b0) == (3, 2, 5, [-0.2, -0.4, -0.2]),
          "the example's sizes and b^0, D exponent included", str(model.b0))
    x = [1.0, 2.0, 0.5]
    # By hand, the site's triangle reading: 0.5 * (2*1 - 1*2*1 + 2*4 - 1*0.5*2 + 2*0.25)
    # = 3.75; linear -0.2 - 0.8 - 0.1 = -1.1; constant 0.5. Total 3.15.
    check(close(model.objective(x), 3.15), "objective at (1, 2, 0.5) by hand, triangle reading",
          repr(model.objective(x)))
    value, reread = objective_via_qps(model, x)
    check(close(value, 3.15), "the converted QPS gives the same value through the verifier's "
          "reader", repr(value))
    # The paper's symmetric reading would count each off-diagonal twice: 0.5 * 4.5 - 1.1 + 0.5.
    check(not close(value, 1.65, 1e-6), "and not the paper's symmetric reading, 1.65")
    check(reread.row_lower == [1.0, 1.0] and reread.row_upper == [math.inf, 3.0],
          "one-sided and two-sided rows (RANGES)", f"{reread.row_lower} {reread.row_upper}")
    check(reread.col_lower == [0.0, 0.0, 0.0] and reread.col_upper == [1.0, 2.0, 1.0],
          "variable bounds", f"{reread.col_lower} {reread.col_upper}")
    check(reread.hessian == {(0, 0): 2.0, (1, 0): -0.5, (1, 1): 2.0, (2, 1): -0.5, (2, 2): 2.0},
          "QUADOBJ: diagonal kept, off-diagonal halved", str(reread.hessian))


def test_layouts() -> None:
    box = fmt.parse("B\nCCB\nminimize\n2\n1\n2 1 1.0\n0.0\n1\n1 -1.0\n0.0\n1e30\n"
                    "-1e30\n1\n2 0.0\n5.0\n0\n0.0\n0\n0.0\n0\n0\n0\n")
    check((box.m, box.lower, box.upper) == (0, [-math.inf, 0.0], [5.0, 5.0]),
          "**B: no constraint count, sides or duals; a bound past the infinity is infinite",
          f"{box.lower} {box.upper}")
    value, reread = objective_via_qps(box, [-2.0, 3.0])
    check(close(value, 0.5 * 1.0 * -2.0 * 3.0 + 2.0) and reread.num_rows == 0,
          "**B model through QPS", repr(value))
    check(fmt.read_solution("", box) == ([0.0, 0.0], 0.0),
          "an empty .sol file is the origin with objective 0 (QPLIB_10038's)")
    free = fmt.parse("N\nDCN\nmaximize\n1\n1\n1 1 -2.0\n0.0\n0\n3.0\n"
                     "1.79769313486232E+308\n-1.79769313486232E+308\n0\n"
                     "1.79769313486232E+308\n0\n0.0\n0\n0.0\n0\n0\n0\n")
    check(free.maximize and free.lower == [-math.inf] and free.upper == [math.inf],
          "**N, maximize, the site's 1.79769313486232E+308 infinity")
    value, reread = objective_via_qps(free, [2.0])
    check(reread.maximize and close(value, -4.0 + 3.0), "maximize and the constant via QPS",
          repr(value))
    for text, what in (
        ("D\nCCB\nminimize\n1\n2\n1 1 1.0\n1 1 2.0\n0.0\n0\n0.0\n1e30\n0\n0\n1\n0\n0\n0\n0\n0\n0\n",
         "a Q^0 entry listed twice"),
        ("D\nCCB\nminimize\n1\n1\n1 1 1.0\n0.0\n0\n0.0\n1e30\n0\n0\n1\n0\n0\n0\n0\n0\n0\n0\n7\n",
         "data past the last section"),
    ):
        try:
            fmt.parse(text)
            check(False, f"{what} is refused")
        except ValueError as error:
            check(True, f"{what} is refused", str(error)[:60])
    integer = fmt.parse("I\nCIB\nminimize\n1\n1\n1 1 1.0\n0.0\n0\n0.0\n1e30\n0\n0\n4\n0\n"
                        "0\n0\n0\n0\n0\n0\n")
    try:
        fmt.to_qps(integer)
        check(False, "integer variables are refused by the writer")
    except ValueError:
        check(True, "integer variables are refused by the writer")


def test_real_file() -> None:
    raw = (DATA / "QPLIB_0018.qplib").read_bytes()
    check(hashlib.sha256(raw).hexdigest() ==
          "4e122ceeccc01ebc3780f638e3e8c05ad3a3e96ce26426d7c50783a06737b41d",
          "the fixture is the site's file byte for byte")
    model = fmt.read(DATA / "QPLIB_0018.qplib")
    check((model.problem_type, model.n, model.m, len(model.q0)) == ("QCL", 50, 1, 1275),
          "QPLIB_0018's header: QCL, 50 variables, 1 constraint, 1275 Q^0 entries")
    sol = (DATA / "QPLIB_0018.sol").read_text(encoding="latin-1")
    x, objvar = fmt.read_solution(sol, model)
    check(objvar == -6.38601498159835 and sum(v != 0.0 for v in x) == 4,
          "the published point: objvar and four nonzeros")
    value, _ = objective_via_qps(model, x)
    check(close(value, objvar, 1e-12), "the converted QPS reproduces the published objective",
          f"{value!r} vs {objvar!r}")
    solu = fetch.parse_solu((DATA / "qplib-solu-excerpt.txt").read_text(encoding="utf-8"))
    check(close(value, solu["QPLIB_0018"][1], 1e-9), "and qplib.solu's value to its digits")
    symmetric = model.objective(x) + 0.5 * sum(v * x[h] * x[k] for (h, k), v in model.q0.items()
                                               if h != k)
    check(abs(symmetric - objvar) > 1.0, "the paper's symmetric reading does not reproduce it",
          repr(symmetric))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "QPLIB_0018.qps"
        path.write_text(fmt.to_qps(model), encoding="utf-8", newline="\n")
        verdict = fetch.converter_check(path, model, sol, 2.2204e-16)
    check(verdict["passed"] and verdict["violation_at_point_via_qps"] <= 1e-12,
          "fetch_qplib's conversion check passes on it", str(verdict))
    shifted = sol.replace("x15", "x16")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "QPLIB_0018.qps"
        path.write_text(fmt.to_qps(model), encoding="utf-8", newline="\n")
        verdict = fetch.converter_check(path, model, shifted, 2.2204e-16)
    check(not verdict["passed"], "and fails when the point is mapped to the wrong variable")


# ---- the runner and the doc ---------------------------------------------------------------

class _Residuals:
    def __init__(self, level: float) -> None:
        self.level = level

    def meets(self, tolerance: float, relative: bool) -> bool:
        return self.level <= tolerance


def test_runner() -> None:
    good = runner.judge("optimal", 100.00001, 100.0, _Residuals(1e-9), True, True)
    check(good["passed"] and good["matches_reference"] and close(good["rel_gap"], 1e-7, 1e-6),
          "within 1e-6 of the reference, verified, residuals met: passed", str(good))
    below = runner.judge("optimal", 99.0, 100.0, _Residuals(1e-9), True, True)
    check(below["passed"] is False and below["matches_reference"] is False,
          "an objective below the best known value is a disagreement, not a credit")
    none = runner.judge("optimal", 1.0, None, _Residuals(1e-9), True, True)
    check(none["passed"] is None and none["matches_reference"] is None,
          "no published reference: neither passed nor failed")
    unchecked = runner.judge("optimal", 100.0, 100.0, _Residuals(1e-9), True, False)
    check(unchecked["passed"] is False, "a failed conversion check cannot pass")
    tiny = runner.judge("optimal", -0.0001562431, -0.0001562421091, _Residuals(1e-9), True, True)
    check(tiny["passed"], "a reference below 1 in magnitude is judged absolutely (max(1, |ref|))")
    instances = {"A": {"tier": "small"}, "B": {"tier": "full"}, "C": {"tier": "full",
                                                                      "skipped": "size"}}
    check(runner.select(instances, "small", None) == ["A"] and
          runner.select(instances, "full", None) == ["A", "B"],
          "tiers: small, and full without the instances skipped for size")


def test_doc() -> None:
    manifest = {
        "selection_rule": "instances.html: Cvx ticked, O in {C, D}, V = C, C in {N, B, L}",
        "listing_counts": {"rows": 453}, "tiers": {"small_max_coefficients": 100000},
        "convex_continuous_classes": {"CCL": 5, "DCL": 11, "LCD": 13},
        "licence": {"licence_statement": "QPLIB is licensed under CC-BY 4.0.",
                    "licence_url": "https://creativecommons.org/licenses/by/4.0/"},
        "instances": {"QPLIB_1": {"tier": "small", "reference_objective": 1.0},
                      "QPLIB_2": {"tier": "full", "reference_objective": None},
                      "QPLIB_3": {"tier": "full", "reference_objective": 2.0, "skipped": "size"}},
    }
    empty = doc.section(None, manifest)
    check("Not yet run on `main`" in empty and "**3 instances**" in empty and
          "`QPLIB_3`" in empty and "LCD 13" in empty and "CC-BY 4.0" in empty,
          "no CSV: the selection is still stated, with the skipped and the excluded")
    base = {"git_commit": "abc1234", "machine": "Linux-x86_64", "time_limit": "60",
            "solver_options": "", "reference_objective": "1.0", "our_objective": "1.0",
            "rel_gap": "0.0", "primal_residual_rel": "1e-9", "dual_residual_rel": "1e-9",
            "duality_gap_rel": "1e-9", "iterations": "10", "solver_seconds": "0.1"}
    rows = [
        dict(base, instance="QPLIB_1", engine="auto", engine_options="", status="optimal",
             matches_reference="1", verified="1", passed="1", **{"success_rel_1e-6": "1"}),
        dict(base, instance="QPLIB_1", engine="ipm", engine_options="qp_algorithm=ipm",
             status="time_limit", matches_reference="0", verified="", passed="0",
             **{"success_rel_1e-6": "0"}),
        dict(base, instance="QPLIB_2", engine="auto", engine_options="", status="optimal",
             matches_reference="", verified="1", passed="", reference_objective="",
             **{"success_rel_1e-6": "1"}),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "qplib-abc1234.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=runner.CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        text = doc.section(path, manifest)
    check("`auto`: **passed 1 of 1**" in text and "`ipm` (`qp_algorithm=ipm`): **passed 0 of 1**"
          in text, "the pass count per engine, over the rows with a reference")
    check("`ipm`, **did not reach `optimal`** (1): `QPLIB_1` (time_limit)" in text and
          "`auto`, **no published reference** (1): `QPLIB_2`" in text,
          "every failure named under its engine and reason")


def main() -> int:
    for test in (test_listing, test_pages, test_hand_computed, test_layouts, test_real_file,
                 test_runner, test_doc):
        print(test.__name__)
        test()
    print(f"\n{'FAILED' if FAILURES else 'OK'}: {FAILURES} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
