# SPDX-License-Identifier: Apache-2.0
"""A Pyomo solver plugin over the SANKHYA Python bindings (#536).

    import pyomo.environ as pyo
    import sankhya.adapters.pyomo_plugin  # registers "sankhya", on import

    results = pyo.SolverFactory("sankhya").solve(model)

Registered against Pyomo's newer ``pyomo.contrib.solver`` interface
(``SolverBase.solve(model) -> Results``, no NL file, no subprocess - the model is read and
the solution written back in memory), which is what recent built-in Pyomo interfaces
(HiGHS, Ipopt's new binding) already use in place of the classic ``OptSolver``/NL-writer
path. ``SolverFactory.register(..., legacy_name=...)`` (`pyomo/contrib/solver/common/
factory.py`) registers the SAME plugin with the classic ``pyomo.opt.base.solvers.
SolverFactory`` too, so ``pyo.SolverFactory("sankhya").solve(model)`` - the call a Pyomo
user already knows from every other solver - works unchanged.

COEFFICIENT EXTRACTION uses ``pyomo.repn.standard_repn.generate_standard_repn``, the same
public utility Pyomo's own NL and LP file writers use to read a linear (and, for the
quadratic terms it also reports, convex-quadratic) expression apart - not a probe of the
kind the CVXPY adapter needs, because Pyomo constraints and variables already expose their
structure directly: ``Constraint.lower`` / ``.body`` / ``.upper``, ``Var.lb`` / ``.ub`` /
``.is_integer()``. Indexed components (``model.x = pyo.Var(range(5))``) are read through
``component_data_objects``, which is Pyomo's own way of turning an indexed component into
its individual scalar entries, each with its own name (``x[3]``).

SCOPE. LP and MILP; QP is not attempted - ``generate_standard_repn`` reports a quadratic
expression's terms too, but mapping them onto ``Model.set_quadratic`` correctly (the
lower-triangular, 0.5-factor convention `include/sankhya/model.hpp` documents) needs
dedicated testing this pass did not have time for, so a quadratic objective or constraint
is refused with a clear message rather than silently solved as if it were linear.
"""

from __future__ import annotations

import pyomo.environ as pyo
from pyomo.common.collections import ComponentMap
from pyomo.contrib.solver.common.base import SolverBase
from pyomo.contrib.solver.common.factory import SolverFactory
from pyomo.contrib.solver.common.results import Results, SolutionStatus, TerminationCondition
from pyomo.contrib.solver.common.solution_loader import SolutionLoader
from pyomo.contrib.solver.common.util import NoSolutionError
from pyomo.repn.standard_repn import generate_standard_repn

import sankhya

__all__ = ["SankhyaSolver"]


class _SolutionLoader(SolutionLoader):
    """Reports the primal values `solve()` already wrote onto the model.

    Duals and reduced costs are not reported (`Solution.row_duals` / `.reduced_costs` exist
    on the SANKHYA side; wiring them through `get_duals`/`get_reduced_costs` is future work
    for this adapter, not attempted this pass) - a caller that asks for them gets
    `NoSolutionError`, the same exception Pyomo's own loaders raise for data they were never
    given, rather than an empty answer that could be mistaken for a zero dual.
    """

    def __init__(self, pyomo_model, primals: dict | None) -> None:
        self._pyomo_model = pyomo_model
        self._primals = primals

    def get_number_of_solutions(self) -> int:
        return 1 if self._primals is not None else 0

    def get_vars(self, vars_to_load=None):
        if self._primals is None:
            raise NoSolutionError("SANKHYA did not report a point for this solve")
        # A ComponentMap, as Pyomo's own loaders return: VarData is not hashable, so a
        # dict keyed by it raises the moment the legacy wrapper loads a solution.
        if vars_to_load is None:
            return ComponentMap(self._primals.items())
        return ComponentMap((v, self._primals[v]) for v in vars_to_load)

    def get_duals(self, cons_to_load=None):
        raise NoSolutionError(
            "SANKHYA's Pyomo adapter does not report constraint duals yet")

    def get_reduced_costs(self, vars_to_load=None):
        raise NoSolutionError(
            "SANKHYA's Pyomo adapter does not report reduced costs yet")

_TERMINATION = {
    "optimal": TerminationCondition.convergenceCriteriaSatisfied,
    "infeasible": TerminationCondition.provenInfeasible,
    "unbounded": TerminationCondition.unbounded,
    "infeasible_or_unbounded": TerminationCondition.infeasibleOrUnbounded,
    "iteration_limit": TerminationCondition.iterationLimit,
    "time_limit": TerminationCondition.maxTimeLimit,
    "node_limit": TerminationCondition.iterationLimit,
    "interrupted": TerminationCondition.interrupted,
    # `feasible` is deliberately NOT convergenceCriteriaSatisfied: it is SANKHYA's own word
    # for "a point exists, optimality not proven", and Pyomo's TerminationCondition has no
    # single value that says that either - `unknown` is the honest one, not a claim of
    # either convergence or a limit having been hit.
    "feasible": TerminationCondition.unknown,
    "numerical_error": TerminationCondition.error,
    "model_error": TerminationCondition.error,
    "not_solved": TerminationCondition.unknown,
}

_SOLUTION_STATUS = {
    "optimal": SolutionStatus.optimal,
    "feasible": SolutionStatus.feasible,
    "iteration_limit": SolutionStatus.feasible,
    "time_limit": SolutionStatus.feasible,
    "node_limit": SolutionStatus.feasible,
    "interrupted": SolutionStatus.feasible,
    "infeasible": SolutionStatus.infeasible,
}


@SolverFactory.register("sankhya", legacy_name="sankhya",
                        doc="SANKHYA LP/MILP over pyomo.contrib.solver")
class SankhyaSolver(SolverBase):
    """``pyo.SolverFactory("sankhya").solve(model, **options)``."""

    def available(self):
        return self.Availability.FullLicense

    def version(self):
        return tuple(int(part) for part in sankhya.version().split()[0].split("."))

    def solve(self, model, **kwargs) -> Results:
        sankhya_model, variables, columns = _build_model(model)
        options = dict(kwargs)
        # The legacy path - pyo.SolverFactory("sankhya").solve(model, options={...},
        # timelimit=...) - stores the caller's options in self.config.solver_options and
        # the limit in self.config.time_limit, then calls solve(model) with no kwargs.
        # Read them from there too, explicit kwargs winning, or a node_limit given that way
        # is silently dropped and the caller gets a full solve they did not ask for.
        config = getattr(self, "config", None)
        stored = getattr(config, "solver_options", None)
        if stored is not None:
            stored_values = stored.value() if hasattr(stored, "value") else dict(stored)
            for key, value in dict(stored_values).items():
                options.setdefault(key, value)
        stored_limit = getattr(config, "time_limit", None)
        if stored_limit is not None:
            options.setdefault("time_limit", stored_limit)
        options.setdefault("log_to_console", False)
        # Pyomo's own generic kwargs are not SANKHYA option names; translated rather than
        # forwarded blind, which would otherwise raise "unknown option" on every solve.
        if "time_limit" in options and options["time_limit"] is not None:
            options["time_limit"] = float(options.pop("time_limit"))
        result = sankhya_model.solve(sankhya.Options(**options))

        primals = None
        if result.claims_a_point:
            values = result.x
            # ComponentMap, not a plain dict: Pyomo's own scalar Var data objects
            # (ScalarVar) are not hashable, and get_vars()'s contract is a mapping FROM the
            # variable object, so a plain dict cannot be this loader's storage either.
            primals = ComponentMap(
                (variable, values[columns[id(variable)]]) for variable in variables)
            for variable, value in primals.items():
                variable.set_value(value, skip_validation=True)

        results = Results()
        results.solution_loader = _SolutionLoader(model, primals)
        results.termination_condition = _TERMINATION.get(result.status,
                                                          TerminationCondition.unknown)
        # A limit that stopped before any point (#562) is not a feasible solution, whatever
        # the status word maps to when a point exists.
        results.solution_status = (
            SolutionStatus.noSolution if primals is None
            else _SOLUTION_STATUS.get(result.status, SolutionStatus.noSolution))
        results.incumbent_objective = result.objective if result.claims_a_point else None
        results.solver_name = "sankhya"
        results.solver_version = sankhya.version()
        results.extra_info = {"sankhya_status": result.status, "sankhya_message": result.message}
        return results


def _variable_bounds(variable) -> tuple[float | None, float | None]:
    lower = variable.lb
    upper = variable.ub
    return (None if lower is None else float(lower)), (None if upper is None else float(upper))


def _build_model(model: "pyo.BlockData"):
    variables = list(model.component_data_objects(pyo.Var, active=True))
    objectives = list(model.component_data_objects(pyo.Objective, active=True))
    if len(objectives) != 1:
        raise ValueError(
            f"SANKHYA's Pyomo adapter expects exactly one active Objective, found "
            f"{len(objectives)}")
    objective = objectives[0]

    sankhya_model = sankhya.Model(maximize=(objective.sense == pyo.maximize))
    # Keyed by id(): Pyomo's per-index Var data objects (ScalarVar / _GeneralVarData) are
    # not hashable, so the object itself cannot be a dict key - id() stays valid for exactly
    # as long as `variables` (and every repn built from this model) is alive, which is the
    # whole of this function and the caller's use of its result.
    columns: dict[int, int] = {}
    for variable in variables:
        lower, upper = _variable_bounds(variable)
        columns[id(variable)] = sankhya_model.add_column(
            lower=-sankhya.INFINITY if lower is None else lower,
            upper=upper,
            integer=bool(variable.is_integer() or variable.is_binary()),
            name=variable.name,
        )

    obj_repn = generate_standard_repn(objective.expr, quadratic=True)
    if obj_repn.quadratic_vars:
        raise ValueError(
            "SANKHYA's Pyomo adapter handles LP/MILP only: the objective has a quadratic "
            "term (QP support is not implemented by this adapter yet)")
    for var, coeff in zip(obj_repn.linear_vars, obj_repn.linear_coefs):
        sankhya_model.set_cost(columns[id(var)], float(coeff))
    sankhya_model.set_objective_offset(float(obj_repn.constant))

    for constraint in model.component_data_objects(pyo.Constraint, active=True):
        repn = generate_standard_repn(constraint.body, quadratic=True)
        if repn.quadratic_vars:
            raise ValueError(
                f"SANKHYA's Pyomo adapter handles LP/MILP only: constraint "
                f"'{constraint.name}' has a quadratic term")
        coefficients = {
            columns[id(var)]: float(coeff)
            for var, coeff in zip(repn.linear_vars, repn.linear_coefs)
        }
        # generate_standard_repn's constant is folded OUT of constraint.lower/upper already
        # (Pyomo normalises `expr + k <= b` to `body <= b - k` when it builds the
        # Constraint), unlike the affine-probe adapters above that have to subtract it
        # themselves - one more way Pyomo's own structure does this adapter's work for it.
        lower = None if constraint.lower is None else float(constraint.lower) - repn.constant
        upper = None if constraint.upper is None else float(constraint.upper) - repn.constant
        sankhya_model.add_row(coefficients, lower=lower, upper=upper, name=constraint.name)

    return sankhya_model, variables, columns
