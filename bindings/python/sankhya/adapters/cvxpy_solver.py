# SPDX-License-Identifier: Apache-2.0
"""A CVXPY solver plugin over the SANKHYA Python bindings (#536), for LP and MILP problems.

    import cvxpy as cp
    import sankhya.adapters.cvxpy_solver  # registers "SANKHYA" as a solve method, on import

    x = cp.Variable(2, nonneg=True)
    problem = cp.Problem(cp.Maximize(c @ x), [A @ x <= b])
    problem.solve(method="SANKHYA")

CVXPY's own solver plugins hook into its reduction chain (DCP -> cone canonicalization ->
a solver-specific matrix stuffing), which is powerful but is CVXPY's internal machinery and
changes shape between CVXPY releases - coupling to it is coupling to CVXPY's implementation,
not its documented interface. `Problem.solve(method="...")`, by contrast, IS CVXPY's own
public escape hatch for exactly this (`Problem.register_solve`, `problem.py`): a name
registered against a callable that receives the `Problem` and is responsible for setting
`problem._value`, `problem._status` and every variable's value as a side effect - the same
contract `Problem._solve` itself fulfils. This adapter therefore walks the problem's PUBLIC
surface directly - `problem.variables()`, `problem.constraints`, `problem.objective` -
rather than CVXPY's internal reduction output.

COEFFICIENT EXTRACTION. An affine CVXPY expression has no public "give me your coefficient
matrix" accessor outside the reduction chain this module deliberately avoids, so a
coefficient is read the way any black-box affine function's Jacobian is: probe it. Every
variable's `.value` is set to an all-zero array to read the constant term, then each
variable's own flattened components are perturbed to 1.0 one at a time (all others held at
zero) and the resulting change in the expression's value IS that component's coefficient,
by definition of affine. This costs one evaluation per (expression, flattened-variable-
component) pair - fine for the model sizes an LP/MILP example uses, and correct regardless
of how CVXPY represents the expression internally, which is the trade this file is making
throughout: a slower but version-stable read of the public surface over a fast but fragile
read of a private one.

SCOPE. LP and MILP (boolean and integer variables); QP is not attempted by this adapter -
extracting a quadratic form the same black-box way would need one probe per PAIR of
variable components, and CVXPY's own quadratic canonicalization (`cp.quad_form`, `**2`,
`cp.sum_squares`) is exactly the internal machinery this file avoids depending on. Variables
of any shape are supported (each element becomes its own SANKHYA column); constraints must
be affine (`<=`, `>=`, `==`) between affine expressions, which is what `is_affine()` on the
normalised form checks before anything is built.
"""

from __future__ import annotations

import numpy as np

import cvxpy as cp

import sankhya

__all__ = ["SANKHYA"]

# What CVXPY's Solution.status may hold, and how SANKHYA's fuller vocabulary maps onto it.
# cvxpy.settings has no "feasible but not proven optimal" word, the same gap PuLP has, but
# it does have USER_LIMIT: the solver stopped short of a proof and a point may be available.
# That is what a `feasible` answer and every limit status are. INFEASIBLE_OR_UNBOUNDED is
# NOT an honest stand-in for them: it asserts a verdict about the problem that a feasible
# point contradicts, and a caller who branches on it would treat a solvable model as
# unsolvable. It is used only where the solver itself said infeasible_or_unbounded.
_STATUS_TO_CVXPY = {
    "optimal": cp.settings.OPTIMAL,
    "infeasible": cp.settings.INFEASIBLE,
    "unbounded": cp.settings.UNBOUNDED,
    "infeasible_or_unbounded": cp.settings.INFEASIBLE_OR_UNBOUNDED,
    "feasible": cp.settings.USER_LIMIT,
    "iteration_limit": cp.settings.USER_LIMIT,
    "time_limit": cp.settings.USER_LIMIT,
    "node_limit": cp.settings.USER_LIMIT,
    "interrupted": cp.settings.USER_LIMIT,
    "model_error": cp.settings.SOLVER_ERROR,
    "numerical_error": cp.settings.SOLVER_ERROR,
    "not_solved": cp.settings.SOLVER_ERROR,
}


def SANKHYA(problem: "cp.Problem", **options: object) -> float:
    """A CVXPY ``method=`` solve function, registered under the name ``"SANKHYA"`` on
    import: ``problem.solve(method="SANKHYA", ...)``.

    Fulfils the same contract ``Problem._solve`` itself does (``problem.py``,
    ``register_solve``): set ``problem._value``, ``problem._status`` and every variable's
    value as a side effect, and return the objective value. Keyword arguments are passed
    straight through as SANKHYA options, exactly as CVXPY's own solver keywords are for a
    registered solver - ``problem.solve(method="SANKHYA", mip_relative_gap=1e-6)``.
    """
    model, columns = _build_model(problem)
    opts = dict(options)
    opts.setdefault("log_to_console", False)
    result = model.solve(sankhya.Options(**opts))

    if result.claims_a_point:
        values = result.x
        for variable, slots in columns.items():
            if variable.shape == ():
                variable.save_value(values[slots[()]])
            else:
                array = np.zeros(variable.shape)
                for flat_index, column in slots.items():
                    array[flat_index] = values[column]
                variable.save_value(array)

    else:
        # No point was reported (a limit before any incumbent, #562): the zeros the
        # coefficient probing left in the variables are not values anyone computed.
        for variable in columns:
            variable.value = None

    problem._status = _STATUS_TO_CVXPY.get(result.status, cp.settings.SOLVER_ERROR)
    if result.claims_a_point:
        problem._value = result.objective
    elif result.status == "infeasible":
        # CVXPY's own convention for an infeasible problem: the worst value of the sense.
        problem._value = -np.inf if problem.objective.NAME == "maximize" else np.inf
    elif result.status == "unbounded":
        problem._value = np.inf if problem.objective.NAME == "maximize" else -np.inf
    else:
        # A limit or an error with nothing in hand is not a value, and CVXPY reads None as
        # exactly that; +-inf here would read as a verdict about the problem.
        problem._value = None
    return problem._value


def register() -> None:
    """Idempotently register ``"SANKHYA"`` with ``cvxpy.Problem.register_solve``.

    Called automatically on import - the point of ``import sankhya.adapters.cvxpy_solver``
    being the whole change a caller makes.
    """
    cp.Problem.register_solve("SANKHYA", SANKHYA)


register()


def _flat_indices(shape: tuple[int, ...]):
    """Every flattened index into an array of `shape`, `()` itself for a scalar."""
    if shape == ():
        yield ()
        return
    yield from np.ndindex(*shape)


def _zero(variable: "cp.Variable"):
    return np.zeros(variable.shape) if variable.shape else 0.0


def _set_all_zero(variables: list) -> None:
    for variable in variables:
        variable.value = _zero(variable)


def _probe(variable: "cp.Variable", index) -> None:
    """Perturb one component of `variable` to 1.0; the caller resets it afterward."""
    if variable.shape == ():
        variable.value = 1.0
    else:
        array = np.array(variable.value, dtype=float, copy=True)
        array[index] = 1.0
        variable.value = array


def _reset(variable: "cp.Variable") -> None:
    variable.value = _zero(variable)


def _affine_rows(expr: "cp.Expression", variables: list,
                 columns: dict) -> tuple[list[dict[int, float]], list[float]]:
    """One `{column: coefficient}` dict and constant per FLATTENED element of `expr`.

    `expr` need not be scalar: `A @ x <= b` is one CVXPY constraint whose expression has
    `A`'s row count as its shape, and becomes that many SANKHYA rows here - each probed
    together (one perturbation of one variable component moves every output element at
    once, read off in the same pass) rather than one flattened expression at a time.
    """
    if not expr.is_affine():
        raise ValueError(f"SANKHYA's CVXPY adapter handles LP/MILP only: '{expr}' is not "
                         "affine (a quadratic or other nonlinear term is present)")
    _set_all_zero(variables)
    shape = expr.shape
    size = int(np.prod(shape)) if shape else 1
    base = np.asarray(expr.value, dtype=float).reshape(size)
    coefficients: list[dict[int, float]] = [dict() for _ in range(size)]
    for variable in variables:
        for index in _flat_indices(variable.shape):
            column = columns[variable][index]
            _probe(variable, index)
            perturbed = np.asarray(expr.value, dtype=float).reshape(size)
            delta = perturbed - base
            _reset(variable)
            for flat in range(size):
                value = delta[flat]
                if value != 0.0:
                    coefficients[flat][column] = coefficients[flat].get(column, 0.0) + value
    return coefficients, base.tolist()


def _variable_bounds(variable: "cp.Variable") -> tuple[float | None, float | None]:
    attrs = variable.attributes
    if attrs.get("bounds") is not None:
        lo, hi = attrs["bounds"]
        return (None if lo is None else float(np.min(lo)),
                None if hi is None else float(np.max(hi)))
    if attrs.get("boolean"):
        return 0.0, 1.0
    # `pos`/`neg` are STRICT in CVXPY's own sense; an LP has no strict bound to give them, so
    # (per every LP interface's usual convention) the closed bound at 0 is what is offered -
    # the nearest representable statement, not a claim of strictness this format cannot make.
    if attrs.get("nonneg") or attrs.get("pos"):
        return 0.0, None
    if attrs.get("nonpos") or attrs.get("neg"):
        return None, 0.0
    return None, None


def _build_model(problem: "cp.Problem"):
    variables = problem.variables()
    model = sankhya.Model(maximize=(problem.objective.NAME == "maximize"))

    columns: dict["cp.Variable", dict] = {}
    for variable in variables:
        lower, upper = _variable_bounds(variable)
        integer = bool(variable.attributes.get("integer") or variable.attributes.get("boolean"))
        slots: dict = {}
        base_name = variable.name()
        for index in _flat_indices(variable.shape):
            name = base_name if index == () else f"{base_name}[{','.join(map(str, index))}]"
            slots[index] = model.add_column(
                lower=-sankhya.INFINITY if lower is None else lower,
                upper=upper, integer=integer, name=name)
        columns[variable] = slots

    # The objective is always scalar (CVXPY enforces this), so its one row is the [0]th.
    obj_coef, obj_constant = _affine_rows(problem.objective.expr, variables, columns)
    for column, coeff in obj_coef[0].items():
        model.set_cost(column, coeff)
    model.set_objective_offset(obj_constant[0])

    for constraint in problem.constraints:
        # Every affine CVXPY constraint normalises to `net <sense> 0`: Inequality(a, b) is
        # `a <= b`, i.e. `a - b <= 0`; Equality likewise. Reading it through that difference
        # means the direction (`<=` vs `>=`, which side the variables landed on) never has
        # to be inspected - only which SIGN of bound the constraint's own type wants.
        net = constraint.args[0] - constraint.args[1]
        row_coefficients, row_constants = _affine_rows(net, variables, columns)
        base_name = constraint.name() if hasattr(constraint, "name") else f"c{id(constraint)}"
        equality = isinstance(constraint, cp.constraints.Equality)
        if not equality and not isinstance(constraint, cp.constraints.Inequality):
            raise ValueError(
                "SANKHYA's CVXPY adapter handles affine <=, >= and == constraints only; "
                f"got a {type(constraint).__name__}")
        for flat, (coefficients, constant) in enumerate(zip(row_coefficients, row_constants)):
            bound = -constant
            name = base_name if len(row_constants) == 1 else f"{base_name}[{flat}]"
            if equality:
                model.add_row(coefficients, lower=bound, upper=bound, name=name)
            else:
                model.add_row(coefficients, upper=bound, name=name)

    return model, columns
