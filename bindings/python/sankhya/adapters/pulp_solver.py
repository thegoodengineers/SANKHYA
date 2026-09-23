# SPDX-License-Identifier: Apache-2.0
"""A PuLP solver backend over the SANKHYA Python bindings (#536).

    import pulp
    import sankhya.adapters.pulp_solver as sankhya_pulp

    problem = pulp.LpProblem("blend", pulp.LpMaximize)
    ...
    problem.solve(sankhya_pulp.SANKHYA(msg=False))

PuLP's own ``LpSolver`` base class (``pulp/apis/core.py``) is the interface this
implements - ``actualSolve(self, lp)`` receives the ``LpProblem`` and is responsible for
setting each variable's ``varValue`` and returning one of PuLP's own status codes. That
shape (one method, in-memory model in, values and a status code out) is what every other
PuLP backend (CBC, GLPK, CPLEX, ...) already implements; matching it is interface
compatibility; nothing here reads or reuses another solver's source.

STATUS MAPPING. PuLP's vocabulary is narrower than SANKHYA's: ``LpStatusOptimal`` /
``Infeasible`` / ``Unbounded`` / ``NotSolved`` / ``Undefined``, with no separate word for "a
feasible point exists but optimality was not proven" - the exact distinction SANKHYA's own
``feasible`` exists to make (see ``Result.status`` docstring). Reporting ``feasible`` as
``LpStatusOptimal`` would silently drop that distinction for a caller who trusts PuLP's
status the way its own solvers document it; it is reported as ``LpStatusNotSolved`` instead,
which is what PuLP itself uses elsewhere for "stopped without a proof". The unfathomed
integer values are still returned, exactly as ``pulp.PULP_CBC_CMD`` does at its own time
limit.
"""

from __future__ import annotations

import pulp

import sankhya

__all__ = ["SANKHYA"]

# LpConstraintLE = -1, LpConstraintEQ = 0, LpConstraintGE = 1 in every PuLP release this was
# written against; named here rather than compared as literals so a version that renumbers
# them still reads correctly.
_LE = pulp.LpConstraintLE
_GE = pulp.LpConstraintGE


class SANKHYA(pulp.LpSolver):
    """Solve a PuLP ``LpProblem`` with SANKHYA.

    Keyword arguments beyond PuLP's own (``mip``, ``msg``, ``timeLimit``, ``warmStart``) are
    passed straight through as SANKHYA options - ``SANKHYA(gap_rel=1e-6)`` sets
    ``mip_relative_gap``, for instance, exactly as PuLP's own solvers forward theirs.
    """

    name = "SANKHYA"

    def __init__(self, mip: bool = True, msg: bool = False, timeLimit: float | None = None,
                warmStart: bool = False, **options: object) -> None:
        super().__init__(mip=mip, msg=msg, timeLimit=timeLimit)
        self._warm_start = warmStart
        self._options = options

    def available(self) -> bool:
        return True

    def actualSolve(self, lp: "pulp.LpProblem") -> int:
        # mip=False means "solve the LP relaxation": no column is marked integer, so the
        # model IS the relaxation rather than a MILP solve() would have to be told to relax.
        model, var_index = _build_model(lp, relax_integers=not self.mip)

        solver_options = dict(self._options)
        solver_options.setdefault("log_to_console", self.msg)
        if self.timeLimit is not None:
            solver_options.setdefault("time_limit", float(self.timeLimit))

        options = sankhya.Options(**solver_options)
        start = None
        if self._warm_start:
            start = getattr(lp, "_sankhya_last_result", None)
        result = model.solve(options, start=start)
        lp._sankhya_last_result = result

        _assign_values(lp, var_index, result)
        lp.status = _STATUS_TO_PULP.get(result.status, pulp.LpStatusUndefined)
        # solutionStatus (added in newer PuLP) distinguishes "not proven" cases the plain
        # status cannot; set it when the attribute exists so callers that read it get the
        # finer answer, without requiring it of older PuLP versions.
        if hasattr(pulp, "LpSolutionOptimal"):
            lp.sol_status = _RESULT_TO_SOL_STATUS.get(
                result.status, getattr(pulp, "LpSolutionNoSolutionFound", pulp.LpStatusUndefined))
        return lp.status


_STATUS_TO_PULP = {
    "optimal": pulp.LpStatusOptimal,
    "infeasible": pulp.LpStatusInfeasible,
    "unbounded": pulp.LpStatusUnbounded,
    "infeasible_or_unbounded": pulp.LpStatusUnbounded,
    "model_error": pulp.LpStatusUndefined,
    "numerical_error": pulp.LpStatusUndefined,
    "not_solved": pulp.LpStatusNotSolved,
    # A point exists but optimality (feasible) or even a point (the limits without one) was
    # not proven: PuLP's closest word is "not solved", not "optimal" - see the module
    # docstring's STATUS MAPPING note.
    "feasible": pulp.LpStatusNotSolved,
    "iteration_limit": pulp.LpStatusNotSolved,
    "time_limit": pulp.LpStatusNotSolved,
    "node_limit": pulp.LpStatusNotSolved,
    "interrupted": pulp.LpStatusNotSolved,
}


def _sol_status_map() -> dict[str, object]:
    if not hasattr(pulp, "LpSolutionOptimal"):
        return {}
    no_solution = getattr(pulp, "LpSolutionNoSolutionFound", pulp.LpSolutionInfeasible)
    return {
        "optimal": pulp.LpSolutionOptimal,
        "feasible": pulp.LpSolutionIntegerFeasible,
        "iteration_limit": pulp.LpSolutionIntegerFeasible,
        "time_limit": pulp.LpSolutionIntegerFeasible,
        "node_limit": pulp.LpSolutionIntegerFeasible,
        "interrupted": pulp.LpSolutionIntegerFeasible,
        "infeasible": pulp.LpSolutionInfeasible,
        "unbounded": pulp.LpSolutionUnbounded,
        "infeasible_or_unbounded": no_solution,
        "model_error": no_solution,
        "numerical_error": no_solution,
        "not_solved": no_solution,
    }


_RESULT_TO_SOL_STATUS = _sol_status_map()


def _build_model(lp: "pulp.LpProblem", relax_integers: bool = False):
    model = sankhya.Model(maximize=(lp.sense == pulp.LpMaximize))

    # Every coefficient the objective actually names, keyed by variable name - gathered
    # before any column is added so add_column's own cost argument can be used directly
    # rather than a separate set_cost pass per variable.
    objective_coef: dict[str, float] = {}
    if lp.objective is not None:
        for var, coeff in lp.objective.items():
            objective_coef[var.name] = objective_coef.get(var.name, 0.0) + float(coeff)

    var_index: dict[str, int] = {}
    for variable in lp.variables():
        lower = variable.lowBound
        upper = variable.upBound
        integer = variable.cat in (pulp.LpInteger, pulp.LpBinary) and not relax_integers
        if variable.cat == pulp.LpBinary:
            lower, upper = 0.0, 1.0
        var_index[variable.name] = model.add_column(
            cost=objective_coef.get(variable.name, 0.0),
            lower=-sankhya.INFINITY if lower is None else float(lower),
            upper=None if upper is None else float(upper),
            integer=integer,
            name=variable.name,
        )

    if lp.objective is not None:
        model.set_objective_offset(float(lp.objective.constant))

    for name, constraint in lp.constraints.items():
        coefficients = {
            var_index[var.name]: float(coeff) for var, coeff in constraint.items()
        }
        # PuLP's LpConstraint IS an LpAffineExpression: the row it names is
        # `expression + constant <sense> 0`, so the bound this row carries is -constant on
        # whichever side `sense` selects.
        bound = -float(constraint.constant)
        if constraint.sense == _LE:
            model.add_row(coefficients, upper=bound, name=name)
        elif constraint.sense == _GE:
            model.add_row(coefficients, lower=bound, name=name)
        else:
            model.add_row(coefficients, lower=bound, upper=bound, name=name)

    return model, var_index


def _assign_values(lp: "pulp.LpProblem", var_index: dict[str, int],
                   result: "sankhya.Result") -> None:
    if not result.claims_a_point:
        for variable in lp.variables():
            variable.varValue = None
        return
    values = result.x
    for variable in lp.variables():
        variable.varValue = values[var_index[variable.name]]
