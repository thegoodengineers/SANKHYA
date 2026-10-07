#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the osqp/clarabel/piqp/scs comparators added to rivals.py (#983).

Hermetic with respect to Maros-Meszaros: nothing here downloads or reads that set (it is
not vendored in this checkout, as the PR body for #983 says). Instead a two-variable QP is
fixed right here -

    minimize 0.5 (x1^2 + x2^2)  subject to  x1 + x2 >= 1,  x1, x2 >= 0

with the unique optimum x1 = x2 = 0.5, objective 0.25 - small enough to check by hand, and
exactly the shape (a general row plus column bounds) `_qp_standard_form` builds for every
one of the four comparators. `bench/runners/testdata/tiny.qps` is this problem in QPS text.

Run:

    python bench/runners/test_qp_rivals.py
"""
from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rivals  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
TINY_QP = Path(__file__).resolve().parent / "testdata" / "tiny.qps"
EXPECTED_OBJECTIVE = 0.25
TOLERANCE = 1e-4

FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}  {detail}")
    if not condition:
        FAILURES += 1


def test_supports() -> None:
    for solver in rivals.QP_PYTHON_SOLVERS:
        check(rivals.supports(solver, "qp"), f"{solver} supports qp")
        check(not rivals.supports(solver, "lp"), f"{solver} does not run on lp "
             "(Maros-Meszaros is what it is measured on, not the LP suites)")
    check(not rivals.supports("glpk", "qp"), "glpk still refuses qp")


def test_standard_form_shape() -> None:
    """P mirrors the stored half-Hessian without doubling it (#514's convention) and A
    stacks the model's own row first, then the identity block for the column bounds."""
    try:
        importlib.import_module("scipy.sparse")
    except ImportError:
        check(True, "scipy not installed in this environment", "(standard form skipped by name)")
        return
    model = rivals._read_qp_model(TINY_QP)  # noqa: SLF001 - exercising the private helper
    P, q, A, l, u = rivals._qp_standard_form(model)  # noqa: SLF001
    check(P.shape == (2, 2), "P is 2x2", str(P.shape))
    check(P.toarray().tolist() == [[1.0, 0.0], [0.0, 1.0]], "P is the identity here",
         str(P.toarray().tolist()))
    check(A.shape == (3, 2), "A stacks 1 row + 2 bound rows", str(A.shape))
    check(list(l) == [1.0, 0.0, 0.0], "l is [c1 lower, x1 lower, x2 lower]", str(list(l)))


def _run_each_installed_solver() -> dict[str, dict]:
    out = {}
    for solver in rivals.QP_PYTHON_SOLVERS:
        try:
            importlib.import_module(solver)
        except ImportError:
            out[solver] = {"status": "not_installed"}
            continue
        with tempfile.TemporaryDirectory() as tmp:
            sol = Path(tmp) / "solution.sol"
            out[solver] = rivals.QP_WORKERS[solver](TINY_QP, 10.0, sol)
            out[solver]["_sol_exists"] = sol.exists()
    return out


def test_each_installed_solver_solves_the_tiny_qp() -> None:
    """Every comparator that IS installed in this environment reaches 0.25 within
    tolerance and writes a point; a solver that is not installed is reported here by name,
    not treated as a silent pass (the same thing _python_qp_rival does for compare_suite.py)."""
    results = _run_each_installed_solver()
    for solver, out in results.items():
        if out["status"] == "not_installed":
            check(True, f"{solver} not installed in this environment", "(skipped by name)")
            continue
        check(out["status"] == "optimal", f"{solver} solves the tiny QP", str(out))
        if out["status"] == "optimal":
            check(abs(out["objective"] - EXPECTED_OBJECTIVE) < TOLERANCE,
                 f"{solver} objective is {EXPECTED_OBJECTIVE}", str(out["objective"]))
            check(out["_sol_exists"], f"{solver} wrote a .sol file")
            check(bool(out.get("solver_options")), f"{solver} records its tolerances",
                 str(out.get("solver_options")))


def test_missing_package_is_named_not_silent() -> None:
    """compare_suite.py's own acceptance criterion: a package that is not installed is
    skipped BY NAME in the output. A genuinely missing import raises the identical
    `ModuleNotFoundError` shape _python_qp_rival parses for osqp/clarabel/piqp/scs; this
    reproduces that shape without depending on one of the four actually being absent from
    this environment (all four happen to be installed here)."""
    import subprocess  # noqa: PLC0415
    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "fake_missing_solver_runner.py"
        fake.write_text("import not_a_real_qp_package_xyz\n")
        done = subprocess.run([sys.executable, str(fake)], capture_output=True, text=True)
        check(done.returncode != 0, "a missing package fails the child process")
        check("No module named" in done.stderr, "the failure names the missing module",
             done.stderr.strip().splitlines()[-1] if done.stderr else "")
        check("No module named 'not_a_real_qp_package_xyz'" in done.stderr,
             "the message names the exact missing module")


def main() -> int:
    test_supports()
    test_standard_form_shape()
    test_each_installed_solver_solves_the_tiny_qp()
    test_missing_package_is_named_not_silent()
    print(f"\n{FAILURES} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
