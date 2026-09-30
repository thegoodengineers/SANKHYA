# SPDX-License-Identifier: Apache-2.0
"""The exact sensitivity derivation (#757), run from test_verify_solution.py's main().

The factor is checked against its own definition (B x = b and B^T y = c hold exactly), and
the shadow price interval against a degenerate LP whose one-sided derivatives are worked out
by hand in the comments below.
"""
from __future__ import annotations

import os
import random
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import verify_solution as vs
import verify_solution_sensitivity as sens


def _lu_matches_its_definition(check) -> None:
    rng = random.Random(757)
    worst = "none"
    ok = True
    for trial in range(40):
        m = rng.randint(1, 9)
        while True:
            columns = [{i: Fraction(rng.choice([-3, -1, 1, 2, 0.1, 0.35]))
                        for i in range(m) if rng.random() < 0.5} for _ in range(m)]
            for p in range(m):
                columns[p].setdefault(p, Fraction(1))  # a nonzero diagonal: nonsingular often
            try:
                lu = sens.ExactLU(columns, deadline=float("inf"))
                break
            except sens.Declined:
                continue
        b = {i: Fraction(rng.randint(-5, 5)) for i in range(m)}
        x = lu.solve(b)
        for i in range(m):
            if sum((col.get(i, 0) * x.get(p, 0) for p, col in enumerate(columns)),
                   Fraction(0)) != b[i]:
                ok, worst = False, f"B x != b, trial {trial}"
        c = {p: Fraction(rng.randint(-5, 5)) for p in range(m)}
        y = lu.solve_transpose(c)
        for p, col in enumerate(columns):
            if sum((v * y.get(i, 0) for i, v in col.items()), Fraction(0)) != c[p]:
                ok, worst = False, f"B^T y != c, trial {trial}"
        # Product-form etas: replace a column, and the solves must be those of the new
        # basis, while the factor the eta was taken from still solves the old one.
        a = {i: Fraction(rng.randint(-4, 4)) for i in range(m) if rng.random() < 0.6}
        alpha = lu.solve(a)
        position = next((p for p in range(m) if alpha.get(p)), None)
        if position is None:
            continue
        updated = lu.with_eta(position, alpha)
        changed = list(columns)
        changed[position] = a
        for factor, cols, label in ((updated, changed, "after an eta"),
                                    (lu, columns, "the original, after an eta was taken")):
            x = factor.solve(b)
            y = factor.solve_transpose(c)
            for i in range(m):
                if sum((col.get(i, 0) * x.get(p, 0) for p, col in enumerate(cols)),
                       Fraction(0)) != b[i]:
                    ok, worst = False, f"B x != b {label}, trial {trial}"
            for p, col in enumerate(cols):
                if sum((v * y.get(i, 0) for i, v in col.items()), Fraction(0)) != c[p]:
                    ok, worst = False, f"B^T y != c {label}, trial {trial}"
    check(ok, "exact LU: B x = b and B^T y = c hold exactly on 40 random bases, and after "
          "a product-form eta", worst)


def _degenerate_model() -> vs.Model:
    """min x + y  s.t.  r1: x + y >= 2,  r2: x >= 1,  r3: y >= 1,  x, y >= 0.

    The optimum (1, 1) has all three rows tight on two columns: degenerate, and the dual is
    not unique. Shifting r1 to x + y >= 2 + t costs 2 + t for t > 0 and nothing for t < 0 (r2
    and r3 still force 2): left derivative 0, right 1. Shifting r2 to x >= 1 + t costs 2 + t
    for t > 0; for t < 0, x = 1 + t and y = 1 - t keep it at 2: left 0, right 1. r3 likewise.
    """
    model = vs.Model()
    for name in ("x", "y"):
        model.add_column(name, False)
    model.col_cost = [1.0, 1.0]
    model.row_names = ["r1", "r2", "r3"]
    model.row_index = {name: i for i, name in enumerate(model.row_names)}
    model.row_lower = [2.0, 1.0, 1.0]
    model.row_upper = [vs.INF, vs.INF, vs.INF]
    model.entries = [[(0, 1.0), (1, 1.0)], [(0, 1.0), (2, 1.0)]]
    return model


def _basis(model, cols, rows, values, activities) -> vs.Solution:
    solution = vs.Solution()
    solution.header["status"] = "optimal"
    for name, st, value in zip(model.col_names, cols, values):
        solution.col_status[name], solution.col_value[name] = st, value
    for name, st, value in zip(model.row_names, rows, activities):
        solution.row_status[name], solution.row_activity[name] = st, value
    return solution


def _shadow_price_interval_at_a_degenerate_optimum(check) -> None:
    model = _degenerate_model()
    # Basis {x, y, s1}: s1 basic AT its bound 2 (the degeneracy), y = (0, 1, 1).
    solution = _basis(model, ["basic", "basic"], ["basic", "at_lower", "at_lower"],
                      [1.0, 1.0], [2.0, 1.0, 1.0])
    exact = sens.derive(model, solution)
    rows = exact["rows"]
    want = {"r1": (0, 1), "r2": (0, 1), "r3": (0, 1)}
    got = {name: (rows[name]["left"], rows[name]["right"]) for name in want}
    check(got == want, "degenerate LP: left and right shadow prices of every row",
          f"{got}")
    check(rows["r1"]["dual"] == 0 and rows["r2"]["dual"] == 1,
          "degenerate LP: the basis's own duals lie inside the intervals",
          f"y = {[rows[n]['dual'] for n in ('r1', 'r2', 'r3')]}")
    # The same vertex with another basis {x, y, s2} gives other duals, the same intervals.
    other = _basis(model, ["basic", "basic"], ["at_lower", "basic", "at_lower"],
                   [1.0, 1.0], [2.0, 1.0, 1.0])
    exact_other = sens.derive(model, other)["rows"]
    got_other = {name: (exact_other[name]["left"], exact_other[name]["right"])
                 for name in want}
    check(got_other == want and exact_other["r1"]["dual"] == 1,
          "degenerate LP: a second basis of the same vertex, other duals, same intervals",
          f"{got_other}")
    # Maximise -x - y is the same problem with every price negated.
    model.maximize = True
    model.col_cost = [-1.0, -1.0]
    exact_max = sens.derive(model, solution)["rows"]
    got_max = {name: (exact_max[name]["left"], exact_max[name]["right"]) for name in want}
    check(got_max == {"r1": (0, -1), "r2": (0, -1), "r3": (0, -1)},
          "degenerate LP, maximised: the interval in the model's own sense", f"{got_max}")


def _a_non_optimal_basis_is_declined(check) -> None:
    model = _degenerate_model()
    # {x, s2, s3} basic with y at its lower bound 0: x = 2 satisfies every row; the dual
    # y1 = 1 prices y at 1 - 1 = 0 - fine - so make it primal infeasible instead: x at 0.
    solution = _basis(model, ["at_lower", "at_lower"], ["basic", "basic", "basic"],
                      [0.0, 0.0], [0.0, 0.0, 0.0])
    try:
        sens.derive(model, solution)
        check(False, "a basis that is not exactly primal feasible is declined")
    except sens.Declined as why:
        check("primal feasible" in str(why),
              "a basis that is not exactly primal feasible is declined", str(why))


DEGENERATE_MPS = """\
NAME          DEGEN
ROWS
 N  COST
 G  R1
 G  R2
 G  R3
COLUMNS
    X         COST         1.0   R1           1.0
    X         R2           1.0
    Y         COST         1.0   R1           1.0
    Y         R3           1.0
RHS
    RHS       R1           2.0   R2           1.0
    RHS       R3           1.0
ENDATA
"""

REPO = Path(__file__).resolve().parents[1]


def _binary() -> Path | None:
    env = os.environ.get("SANKHYA_BIN")
    if env and Path(env).exists():
        return Path(env)
    for candidate in ("build/sankhya", "build/sankhya.exe"):
        if (REPO / candidate).exists():
            return REPO / candidate
    return None


def _verified(mps: Path, sol: Path) -> tuple[bool, str]:
    report = vs.verify(vs.parse_mps(mps), vs.parse_sol(sol), vs.DEFAULT_PRIMAL_TOL,
                       vs.DEFAULT_DUAL_TOL, vs.DEFAULT_INTEGER_TOL, vs.DEFAULT_DUALITY_TOL)
    failed = [f"{name}: {detail}" for ok, name, detail in report.lines if not ok]
    return report.failures == 0, "; ".join(failed)[:300]


def _rows_block(text: str) -> dict[str, list[str]]:
    rows, inside = {}, False
    for line in text.splitlines():
        if line.startswith("begin exact_sensitivity_rows"):
            inside = True
        elif line.startswith("end exact_sensitivity_rows"):
            inside = False
        elif inside:
            fields = line.split()
            rows[fields[0]] = fields[1:]
    return rows


def _solve_and_verify(check, binary: Path, mps: Path, label: str, tmp: Path) -> str:
    sol = tmp / f"{label}.sol"
    subprocess.run([str(binary), "solve", str(mps), "--ranging", "--option", "exact=true",
                    "--option", "log_to_console=false", "--write-sol", str(sol)],
                   capture_output=True, text=True, check=False)
    text = sol.read_text() if sol.exists() else ""
    check("begin exact_sensitivity_rows" in text,
          f"{label}: the solver writes a certified sensitivity report",
          next((line for line in text.splitlines() if line.startswith("sensitivity")), ""))
    if "begin exact_sensitivity_rows" not in text:
        return text
    ok, why = _verified(mps, sol)
    check(ok, f"{label}: the verifier re-derives every exact value and accepts it", why)
    # A row's exact dual moved by 1/7 must be rejected.
    lines = text.splitlines()
    start = next(k for k, line in enumerate(lines)
                 if line.startswith("begin exact_sensitivity_rows"))
    fields = lines[start + 1].split()
    value = Fraction(fields[2]) + Fraction(1, 7)
    fields[2] = f"{value.numerator}/{value.denominator}"
    lines[start + 1] = " ".join(fields)
    bad = tmp / f"{label}-bad.sol"
    bad.write_text("\n".join(lines) + "\n")
    ok, _ = _verified(mps, bad)
    check(not ok, f"{label}: a tampered exact dual is rejected")
    return text


def _end_to_end(check) -> None:
    binary = _binary()
    if binary is None:
        print("  certified sensitivity end to end: SKIPPED, no built solver")
        return
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        degenerate = tmp / "degen.mps"
        degenerate.write_text(DEGENERATE_MPS)
        text = _solve_and_verify(check, binary, degenerate, "degenerate LP", tmp)
        intervals = {name: fields[4:6] for name, fields in _rows_block(text).items()}
        check(intervals == {"R1": ["0/1", "1/1"], "R2": ["0/1", "1/1"], "R3": ["0/1", "1/1"]},
              "degenerate LP: the solver reports the two-sided shadow price of every row",
              f"{intervals}")
        _solve_and_verify(check, binary, REPO / "demo" / "crude_blend.mps", "crude blend", tmp)


def run(check) -> None:
    _lu_matches_its_definition(check)
    _shadow_price_interval_at_a_degenerate_optimum(check)
    _a_non_optimal_basis_is_declined(check)
    _end_to_end(check)
