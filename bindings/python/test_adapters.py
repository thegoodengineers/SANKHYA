#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the optional PuLP, Pyomo and CVXPY adapters (#536).

Hand-rolled, matching bindings/python/test_bindings.py, and separate from it: PuLP, Pyomo
and CVXPY are OPTIONAL dependencies of the adapters, never of the core `sankhya` package, so
importing this file must not be a precondition for that suite. Each adapter's tests are
skipped, not failed, when its library is not installed - the acceptance criterion this
answers is "optional dependencies only", and a skip says that was honoured, where a failure
would say the opposite.

Every test compares the adapter's answer against a DIRECT sankhya solve of the same model
built through the plain Python bindings, per #536's own acceptance criterion - so a defect
that changes the *translation* (a bound, a sign, a sense) surfaces as a mismatch here even
if both sides happen to reach a locally-consistent answer.

    PYTHONPATH=bindings/python python bindings/python/test_adapters.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sankhya  # noqa: E402

FAILURES = 0
SKIPPED = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    if condition:
        print(f"  [PASS] {name}" + (f"  {detail}" if detail else ""))
    else:
        FAILURES += 1
        print(f"  [FAIL] {name}" + (f"  {detail}" if detail else ""))


def skip(name: str, why: str) -> None:
    global SKIPPED
    SKIPPED += 1
    print(f"  [SKIP] {name}  ({why})")


def near(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


# ---- The reference: the same three models, solved directly ------------------------------

def _direct_lp() -> "sankhya.Result":
    #   maximise 3x + 2y  s.t.  x + y <= 4,  x + 3y <= 6,  0 <= x <= 3,  y >= 0
    model = sankhya.Model(maximize=True)
    x = model.add_column(cost=3.0, upper=3.0, name="x")
    y = model.add_column(cost=2.0, name="y")
    model.add_row({x: 1.0, y: 1.0}, upper=4.0)
    model.add_row({x: 1.0, y: 3.0}, upper=6.0)
    return model.solve(log_to_console=False)


def _direct_milp() -> "sankhya.Result":
    #   maximise a + b  s.t.  2a + 2b <= 3,  a, b in {0, 1}
    model = sankhya.Model(maximize=True)
    a = model.add_column(cost=1.0, upper=1.0, integer=True, name="a")
    b = model.add_column(cost=1.0, upper=1.0, integer=True, name="b")
    model.add_row({a: 2.0, b: 2.0}, upper=3.0)
    return model.solve(log_to_console=False)


def _direct_infeasible() -> "sankhya.Result":
    model = sankhya.Model()
    z = model.add_column(upper=1.0, name="z")
    model.add_row({z: 1.0}, lower=5.0)
    return model.solve(log_to_console=False)


# ---- PuLP ---------------------------------------------------------------------------------

def test_pulp_lp_matches_a_direct_solve() -> None:
    try:
        import pulp
    except ImportError as error:
        skip("test_pulp_lp_matches_a_direct_solve", str(error))
        return
    import sankhya.adapters.pulp_solver as sankhya_pulp

    reference = _direct_lp()
    prob = pulp.LpProblem("blend", pulp.LpMaximize)
    x = pulp.LpVariable("x", lowBound=0, upBound=3)
    y = pulp.LpVariable("y", lowBound=0)
    prob += 3 * x + 2 * y
    prob += x + y <= 4
    prob += x + 3 * y <= 6
    status = prob.solve(sankhya_pulp.SANKHYA(msg=False))

    check(pulp.LpStatus[status] == "Optimal", "PuLP LP status", pulp.LpStatus[status])
    check(near(pulp.value(prob.objective), reference.objective), "PuLP LP objective matches",
          f"{pulp.value(prob.objective)} vs {reference.objective}")
    check(near(x.varValue, reference.x[0]) and near(y.varValue, reference.x[1]),
          "PuLP LP primal values match", f"{[x.varValue, y.varValue]} vs {reference.x}")


def test_pulp_milp_matches_a_direct_solve() -> None:
    try:
        import pulp
    except ImportError as error:
        skip("test_pulp_milp_matches_a_direct_solve", str(error))
        return
    import sankhya.adapters.pulp_solver as sankhya_pulp

    reference = _direct_milp()
    prob = pulp.LpProblem("knapsack", pulp.LpMaximize)
    a = pulp.LpVariable("a", cat=pulp.LpBinary)
    b = pulp.LpVariable("b", cat=pulp.LpBinary)
    prob += a + b
    prob += 2 * a + 2 * b <= 3
    status = prob.solve(sankhya_pulp.SANKHYA(msg=False))

    check(pulp.LpStatus[status] == "Optimal", "PuLP MILP status", pulp.LpStatus[status])
    check(near(pulp.value(prob.objective), reference.objective), "PuLP MILP objective matches",
          f"{pulp.value(prob.objective)} vs {reference.objective}")
    check(all(abs(v - round(v)) < 1e-6 for v in (a.varValue, b.varValue)),
          "PuLP MILP values are integral", str([a.varValue, b.varValue]))


def test_pulp_infeasible_is_reported() -> None:
    try:
        import pulp
    except ImportError as error:
        skip("test_pulp_infeasible_is_reported", str(error))
        return
    import sankhya.adapters.pulp_solver as sankhya_pulp

    prob = pulp.LpProblem("bad", pulp.LpMinimize)
    z = pulp.LpVariable("z", lowBound=0, upBound=1)
    prob += z
    prob += z >= 5
    status = prob.solve(sankhya_pulp.SANKHYA(msg=False))
    check(pulp.LpStatus[status] == "Infeasible", "PuLP infeasible status",
          pulp.LpStatus[status])


# ---- CVXPY --------------------------------------------------------------------------------

def test_cvxpy_lp_matches_a_direct_solve() -> None:
    try:
        import cvxpy as cp
    except ImportError as error:
        skip("test_cvxpy_lp_matches_a_direct_solve", str(error))
        return
    import sankhya.adapters.cvxpy_solver  # noqa: F401 - registers "SANKHYA" on import

    reference = _direct_lp()
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")
    problem = cp.Problem(cp.Maximize(3 * x + 2 * y),
                         [x + y <= 4, x + 3 * y <= 6, x >= 0, x <= 3, y >= 0])
    value = problem.solve(method="SANKHYA")

    check(problem.status == cp.OPTIMAL, "CVXPY LP status", problem.status)
    check(near(value, reference.objective), "CVXPY LP objective matches",
          f"{value} vs {reference.objective}")
    check(near(x.value, reference.x[0]) and near(y.value, reference.x[1]),
          "CVXPY LP primal values match", f"{[x.value, y.value]} vs {reference.x}")


def test_cvxpy_vector_variable_and_matrix_constraint() -> None:
    """The shape a real CVXPY model actually uses - `A @ x`, not one scalar per row."""
    try:
        import cvxpy as cp
        import numpy as np
    except ImportError as error:
        skip("test_cvxpy_vector_variable_and_matrix_constraint", str(error))
        return
    import sankhya.adapters.cvxpy_solver  # noqa: F401

    # minimise x0 + 2x1 - x2  s.t.  x0+x1+x2 = 10,  0 <= x <= 5.
    # Cheapest way to spend the budget: x2 (coefficient -1) to its cap, x0 (coefficient 1,
    # cheaper than x1's 2) for the rest -> x = [5, 0, 5], objective 5 + 0 - 5 = 0.
    c = np.array([1.0, 2.0, -1.0])
    a_row = np.array([[1.0, 1.0, 1.0]])
    b = np.array([10.0])
    x = cp.Variable(3, nonneg=True)
    problem = cp.Problem(cp.Minimize(c @ x), [a_row @ x == b, x <= 5])
    value = problem.solve(method="SANKHYA")

    check(problem.status == cp.OPTIMAL, "CVXPY vector LP status", problem.status)
    check(near(value, 0.0, tol=1e-6), "CVXPY vector LP objective", str(value))
    check(near(x.value[0], 5.0) and near(x.value[1], 0.0) and near(x.value[2], 5.0),
          "CVXPY vector LP primal values", str(x.value))


def test_cvxpy_milp_matches_a_direct_solve() -> None:
    try:
        import cvxpy as cp
    except ImportError as error:
        skip("test_cvxpy_milp_matches_a_direct_solve", str(error))
        return
    import sankhya.adapters.cvxpy_solver  # noqa: F401

    reference = _direct_milp()
    a = cp.Variable(boolean=True)
    b = cp.Variable(boolean=True)
    problem = cp.Problem(cp.Maximize(a + b), [2 * a + 2 * b <= 3])
    value = problem.solve(method="SANKHYA")

    check(problem.status == cp.OPTIMAL, "CVXPY MILP status", problem.status)
    check(near(value, reference.objective), "CVXPY MILP objective matches",
          f"{value} vs {reference.objective}")


def test_cvxpy_infeasible_is_reported() -> None:
    try:
        import cvxpy as cp
    except ImportError as error:
        skip("test_cvxpy_infeasible_is_reported", str(error))
        return
    import sankhya.adapters.cvxpy_solver  # noqa: F401

    z = cp.Variable(nonneg=True)
    problem = cp.Problem(cp.Minimize(z), [z >= 5, z <= 1])
    problem.solve(method="SANKHYA")
    check(problem.status == cp.INFEASIBLE, "CVXPY infeasible status", problem.status)


def test_cvxpy_quadratic_objective_is_refused_not_silently_relaxed() -> None:
    """#536 scopes this adapter to LP/MILP; a QP must be REFUSED, not solved as if linear."""
    try:
        import cvxpy as cp
    except ImportError as error:
        skip("test_cvxpy_quadratic_objective_is_refused_not_silently_relaxed", str(error))
        return
    import sankhya.adapters.cvxpy_solver  # noqa: F401

    x = cp.Variable(name="x")
    problem = cp.Problem(cp.Minimize(cp.square(x)), [x >= 1])
    raised = False
    try:
        problem.solve(method="SANKHYA")
    except ValueError as error:
        raised = "affine" in str(error)
    check(raised, "a quadratic CVXPY objective is refused rather than mis-solved as linear")


# ---- Pyomo --------------------------------------------------------------------------------

def test_pyomo_lp_matches_a_direct_solve() -> None:
    try:
        import pyomo.environ as pyo
    except ImportError as error:
        skip("test_pyomo_lp_matches_a_direct_solve", str(error))
        return
    import sankhya.adapters.pyomo_plugin  # noqa: F401 - registers "sankhya" on import

    reference = _direct_lp()
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(0, 3))
    model.y = pyo.Var(bounds=(0, None))
    model.obj = pyo.Objective(expr=3 * model.x + 2 * model.y, sense=pyo.maximize)
    model.c1 = pyo.Constraint(expr=model.x + model.y <= 4)
    model.c2 = pyo.Constraint(expr=model.x + 3 * model.y <= 6)
    results = pyo.SolverFactory("sankhya").solve(model)

    ok = str(results.solver.termination_condition) == "optimal"
    check(ok, "Pyomo LP termination condition", str(results.solver.termination_condition))
    check(near(pyo.value(model.obj), reference.objective), "Pyomo LP objective matches",
          f"{pyo.value(model.obj)} vs {reference.objective}")
    check(near(pyo.value(model.x), reference.x[0]) and near(pyo.value(model.y), reference.x[1]),
          "Pyomo LP primal values match",
          f"{[pyo.value(model.x), pyo.value(model.y)]} vs {reference.x}")


def test_pyomo_indexed_milp_matches_a_direct_solve() -> None:
    """Indexed components (`Var(RangeSet)`), the shape a real Pyomo model actually uses."""
    try:
        import pyomo.environ as pyo
    except ImportError as error:
        skip("test_pyomo_indexed_milp_matches_a_direct_solve", str(error))
        return
    import sankhya.adapters.pyomo_plugin  # noqa: F401

    reference = _direct_milp()
    model = pyo.ConcreteModel()
    model.I = pyo.RangeSet(0, 1)
    model.a = pyo.Var(model.I, domain=pyo.Binary)
    model.obj = pyo.Objective(expr=sum(model.a[i] for i in model.I), sense=pyo.maximize)
    model.c = pyo.Constraint(expr=sum(2 * model.a[i] for i in model.I) <= 3)
    results = pyo.SolverFactory("sankhya").solve(model)

    ok = str(results.solver.termination_condition) == "optimal"
    check(ok, "Pyomo indexed MILP termination condition",
          str(results.solver.termination_condition))
    check(near(pyo.value(model.obj), reference.objective),
          "Pyomo indexed MILP objective matches",
          f"{pyo.value(model.obj)} vs {reference.objective}")
    values = [pyo.value(model.a[i]) for i in model.I]
    check(all(abs(v - round(v)) < 1e-6 for v in values), "Pyomo indexed MILP values integral",
          str(values))


def test_pyomo_infeasible_is_reported() -> None:
    try:
        import pyomo.environ as pyo
    except ImportError as error:
        skip("test_pyomo_infeasible_is_reported", str(error))
        return
    import sankhya.adapters.pyomo_plugin  # noqa: F401

    model = pyo.ConcreteModel()
    model.z = pyo.Var(bounds=(0, 1))
    model.obj = pyo.Objective(expr=model.z)
    model.c = pyo.Constraint(expr=model.z >= 5)
    results = pyo.SolverFactory("sankhya").solve(model)
    check("nfeasible" in str(results.solver.termination_condition),
          "Pyomo infeasible termination condition",
          str(results.solver.termination_condition))


def main() -> int:
    print(f"SANKHYA adapter bindings, against solver version {sankhya.version()}\n")
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            print(name)
            function()
    print()
    print(f"{SKIPPED} test(s) skipped (optional dependency not installed)")
    if FAILURES == 0:
        print("ALL TESTS PASSED")
        return 0
    print(f"{FAILURES} check(s) FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
