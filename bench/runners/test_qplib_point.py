#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the check of every QPLIB conversion at QPLIB's published point (#835).

Hermetic: nothing here downloads a file or runs the solver. Each failure class the
whole-library check met has a small inline .qplib model, a .sol file in QPLIB's layout and,
where the class needs it, the head of a GAMS model in the convert tool's layout:

*   binary and integer columns are `b<k>` and `i<k>` in the .sol file, not `x<k>`
    (QPLIB_10056, QPLIB_9030);
*   objvar is not always the GAMS model's first variable (QPLIB_10069 declares it last);
*   with a linear objective objvar can be a variable of the model itself (QPLIB_10035);
*   a row whose terms are large and cancel is judged on its exact value, not on what double
    precision makes of it (QPLIB_10022), and a row the point really misses is reported with
    its scale (QPLIB_4805);
*   a .qplib file whose objective is empty although the published one is not is reported as
    exactly that (QPLIB_10035, 10036, 10037, 10039), not passed and not blamed on the
    conversion.

    python bench/runners/test_qplib_point.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "tools"))
import fetch_qplib as fetch  # noqa: E402
import qplib_format as fmt  # noqa: E402

FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}  {detail}")
    if not condition:
        FAILURES += 1


def verdict(model: fmt.QplibModel, sol: str, gams: list[str] | None,
            solinfeasibility: float | None = 0.0) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.qps"
        path.write_text(fmt.to_qps(model), encoding="utf-8", newline="\n")
        return fetch.converter_check(path, model, sol, solinfeasibility, gams)


def refused(call, what: str) -> None:
    try:
        call()
        check(False, what)
    except ValueError as error:
        check(True, what, str(error)[:70])


# A GAMS head as the convert tool writes it: comment header, then the Variables statement
# wrapped over lines with leading commas, then the type statements, then Equations.
GAMS_OBJVAR_FIRST = """$offlisting
*  Variable counts
*      Total     cont   binary  integer
*          4        1        3        0

Variables  objvar,b2,b3
          ,b4;

Binary Variables  b2,b3,b4;

Equations  e1,e2;
"""

# min 0.5 (2 x1^2 + 2 x2^2 + 2 x3^2) - x1 - 2 x2 - 3 x3, x binary, x1 + x2 + x3 <= 2.
# Its optimum is x = (0, 1, 1): 0.5 (2 + 2) - 2 - 3 = -3.
BINARY_QP = """BIN3
CBL
minimize
3
1
3
1 1 2.0
2 2 2.0
3 3 2.0
0.0
3
1 -1.0
2 -2.0
3 -3.0
0.0
3
1 1 1.0
1 2 1.0
1 3 1.0
1e30
-1e30
0
2.0
0
0.0
0
0.0
0
0.0
0
0
0
"""


def test_gams_variables() -> None:
    names = fmt.parse_gams_variables(GAMS_OBJVAR_FIRST)
    check(names == ["objvar", "b2", "b3", "b4"],
          "the first Variables statement, wrapped with leading commas", str(names))
    refused(lambda: fmt.parse_gams_variables("Positive Variables x2;\nEquations e1;\n"),
            "a head without the Variables statement is refused")
    refused(lambda: fmt.parse_gams_variables("Variables objvar,x2,x2;\n"),
            "a name declared twice is refused")


def test_binary_names() -> None:
    model = fmt.parse(BINARY_QP)
    sol = "objvar                 -3.000000000000000\nb3   1.0\nb4   1.0\n"
    x, objvar = fmt.read_solution(sol, model, fmt.parse_gams_variables(GAMS_OBJVAR_FIRST))
    check(x == [0.0, 1.0, 1.0] and objvar == -3.0,
          "b<k> names map through the GAMS order (QPLIB_10056's class)", str(x))
    x, _ = fmt.read_solution(sol, model)
    check(x == [0.0, 1.0, 1.0], "and by the convert rule without it, objvar first", str(x))
    v = verdict(model, sol, fmt.parse_gams_variables(GAMS_OBJVAR_FIRST))
    check(v["passed"] and v["objective_at_point_via_qps"] == -3.0,
          "the published-point check passes on the converted QPS", str(v))
    integer = fmt.parse(BINARY_QP.replace("CBL", "CIL", 1).replace(
        "1e30\n-1e30\n0\n2.0\n0\n", "1e30\n-1e30\n0\n2.0\n0\n0.0\n0\n1.0\n0\n", 1))
    gams = ["objvar", "i2", "i3", "i4"]
    v = verdict(integer, "objvar -3\ni3 1\ni4 1\n", gams)
    check(v["passed"], "i<k> names of an integer model (QPLIB_9030's class)", str(v))
    v = verdict(model, "objvar -3\nb2 1\nb4 1\n", fmt.parse_gams_variables(GAMS_OBJVAR_FIRST))
    check(not v["passed"], "a point on the wrong columns still fails the objective", str(v))


def test_objvar_last() -> None:
    model = fmt.parse(BINARY_QP)
    gams = ["b1", "b2", "b3", "objvar"]  # QPLIB_10069 declares objvar after the binaries
    sol = "objvar -3\nb2 1\nb3 1\n"
    x, _ = fmt.read_solution(sol, model, gams)
    check(x == [0.0, 1.0, 1.0], "objvar declared last: b<k> is variable k - 1", str(x))
    check(verdict(model, sol, gams)["passed"], "and the check passes")
    refused(lambda: fmt.read_solution("objvar -3\nb1 1\n", model),
            "without the GAMS order a name below 2 cannot be placed, and is refused")
    refused(lambda: fmt.read_solution("objvar -3\nb9 1\n", model, gams),
            "a .sol name the GAMS model does not declare is refused")


def test_objvar_is_a_variable() -> None:
    # min objvar subject to objvar - x2 - x3 >= 0, x2 + x3 >= 1: a linear objective whose
    # variable 1 is objvar itself, as in QPLIB_10035 (n GAMS names for n variables).
    model = fmt.parse("LIN\nLCL\nminimize\n3\n2\n0.0\n1\n1 1.0\n0.0\n5\n1 1 1.0\n1 2 -1.0\n"
                      "1 3 -1.0\n2 2 1.0\n2 3 1.0\n1e30\n0.0\n1\n2 1.0\n1e30\n0\n0.0\n0\n"
                      "10.0\n0\n0.0\n0\n0.0\n0\n0.0\n0\n0\n0\n")
    gams = ["objvar", "x2", "x3"]
    sol = "objvar 1.0\nx2 0.25\nx3 0.75\n"
    x, objvar = fmt.read_solution(sol, model, gams)
    check(x == [1.0, 0.25, 0.75] and objvar == 1.0,
          "objvar is variable 1 of the model and gets its value", str(x))
    v = verdict(model, sol, gams)
    check(v["passed"] and v["objective_at_point_via_qps"] == 1.0, "and the check passes", str(v))
    refused(lambda: fmt.read_solution(sol, model, ["objvar", "x2"]),
            "a GAMS model with neither n nor n + 1 variables is refused")
    # QPLIB_10035, 10036, 10037 and 10039 are this model with the objective coefficient of
    # objvar missing from the .qplib file: the point is feasible, the objective cannot match,
    # and the check says so rather than passing or blaming the conversion.
    empty = fmt.parse("LIN\nLCL\nminimize\n3\n2\n0.0\n0\n0.0\n5\n1 1 1.0\n1 2 -1.0\n"
                      "1 3 -1.0\n2 2 1.0\n2 3 1.0\n1e30\n0.0\n1\n2 1.0\n1e30\n0\n0.0\n0\n"
                      "10.0\n0\n0.0\n0\n0.0\n0\n0.0\n0\n0\n0\n")
    v = verdict(empty, sol, gams)
    check(not v["passed"] and v["violation_at_point_via_qps"] == 0.0
          and "identically zero" in v["reasons"][0] and len(v["reasons"]) == 1,
          "an empty .qplib objective is reported as the reason, the point still feasible",
          str(v.get("reasons")))


def test_exact_rows() -> None:
    # x1 + 1e17 x2 - 1e17 x3 >= 1 at x = (1, 1, 1): exactly 1, feasible. Summed in column
    # order in double precision, 1 + 1e17 rounds to 1e17 and the row reads 0, a violation
    # of 1: QPLIB_10022's row 9157 has the same cancellation at a scale of 6.7e9.
    model = fmt.parse("BIG\nLCL\nminimize\n3\n1\n0.0\n0\n0.0\n3\n1 1 1.0\n1 2 1e17\n"
                      "1 3 -1e17\n1e30\n1.0\n0\n1e30\n0\n0.0\n0\n2.0\n0\n0.0\n0\n0.0\n0\n"
                      "0.0\n0\n0\n0\n")
    gams = ["objvar", "x2", "x3", "x4"]
    v = verdict(model, "x2 1\nx3 1\nx4 1\n", gams)
    check(v["passed"] and v["violation_at_point_via_qps"] == 0.0
          and v.get("rows_evaluated_exactly") == 1,
          "a row that fails only in floating point is judged on its exact value", str(v))
    v = verdict(model, "x2 0.5\nx3 1\nx4 1\n", gams)
    check(not v["passed"] and v["violation_at_point_via_qps"] == 0.5
          and "row c1" in v["reasons"][0] and "2e+17 in magnitude" in v["reasons"][0],
          "and a row that really is violated still fails, by its exact amount, with the row "
          "and its scale named", str(v.get("reasons")))


def main() -> int:
    for test in (test_gams_variables, test_binary_names, test_objvar_last,
                 test_objvar_is_a_variable, test_exact_rows):
        print(test.__name__)
        test()
    print(f"\n{'OK' if FAILURES == 0 else 'FAILED'}: {FAILURES} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
