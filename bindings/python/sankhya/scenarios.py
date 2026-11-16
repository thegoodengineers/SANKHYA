# SPDX-License-Identifier: Apache-2.0
"""One LP under many cost, bound and right-hand-side sets in one call (#752).

    >>> import sankhya
    >>> model = sankhya.Model.read("demo/crude_blend.mps")              # doctest: +SKIP
    >>> runs = model.solve_scenarios([
    ...     [("cost", 0, 2.10)],                     # Arab Light dearer by 0.30 $/bbl
    ...     [("rhs", 1, 42.0)],                      # the diesel pool raised to 42
    ... ], log_to_console=False)                                       # doctest: +SKIP
    >>> [(r.verified, round(r.result.objective, 4)) for r in runs]      # doctest: +SKIP

Each scenario is a list of ``(target, index, value)`` overrides of the BASE model, where
target is one of ``cost``, ``col_lower``, ``col_upper`` (index a column) or ``row_lower``,
``row_upper``, ``rhs`` (index a row; ``rhs`` moves every finite side of the row). Scenarios
do not accumulate. The model is solved once and every scenario restarts from a basis
(sankhya_solve_scenarios in the C API, the same core `sankhya scenarios` runs).

A result with ``verified`` False was NOT checked against its scenario's model and must not be
called optimal, whatever ``result.status`` says.
"""

from __future__ import annotations

import collections
import ctypes
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from . import Model, Options

# The order of sankhya_scenario_target in include/sankhya/sankhya.h.
TARGETS = ("cost", "col_lower", "col_upper", "row_lower", "row_upper", "rhs")

ScenarioResult = collections.namedtuple("ScenarioResult", ["result", "verified", "safe_bound"])
ScenarioResult.__doc__ = ("One scenario's answer: the Result, whether it was checked against "
                          "the scenario's own model, and a safe bound on its optimum (NaN "
                          "when none was proved).")


def solve_scenarios(model: "Model", scenarios: Sequence[Iterable[tuple[str, int, float]]],
                    options: "Options | None" = None,
                    **overrides: object) -> list[ScenarioResult]:
    """Solve ``model`` under every scenario; see the module docstring."""
    from . import Options, Result, _check, _library

    if overrides:
        options = options or Options()
        for name, value in overrides.items():
            options.set(name, value)

    begin = [0]
    targets: list[int] = []
    indices: list[int] = []
    values: list[float] = []
    for k, scenario in enumerate(scenarios):
        for entry in scenario:
            target, index, value = entry
            if target not in TARGETS:
                raise ValueError(f"scenario {k}: target {target!r} is not one of {TARGETS}")
            targets.append(TARGETS.index(target))
            indices.append(int(index))
            values.append(float(value))
        begin.append(len(targets))

    count = len(begin) - 1
    n = max(1, len(targets))
    c_begin = (ctypes.c_int * len(begin))(*begin)
    c_targets = (ctypes.c_int * n)(*targets)
    c_indices = (ctypes.c_int * n)(*indices)
    c_values = (ctypes.c_double * n)(*values)
    handles = (ctypes.c_void_p * max(1, count))()
    verified = (ctypes.c_int * max(1, count))()
    bounds = (ctypes.c_double * max(1, count))()
    _check(_library().sankhya_solve_scenarios(
        model._handle, options._handle if options else None, count, c_begin, c_targets,
        c_indices, c_values, handles, verified, bounds), "solving scenarios")
    cols, rows = model.num_cols, model.num_rows
    return [ScenarioResult(Result(handles[k], cols, rows), bool(verified[k]), bounds[k])
            for k in range(count)]
