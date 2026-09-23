#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A solution report in the planner's terms: binding constraints, shadow prices, ranging
and reduced costs, all by the model's own names, sorted by economic impact (#527).

    python tools/report.py model.mps solution.sol --format md  > report.md
    python tools/report.py model.mps solution.sol --format json > report.json

A DETERMINISTIC TEMPLATE, not a language model: every number in the report is read once
from the parsed `.sol` file (`verify_solution_sol.py`, the same independent reader
`verify_solution.py` checks a solve against - no code shared with the solver) and placed
into the report unchanged, so "why does the report say X" always has the same one-line
answer: because the `.sol` file said so, at that name. `tests/tools/test_report.py` checks
exactly that - every number this script prints is looked up again in the parsed `.sol` file
and compared, so a transcription or rounding bug here cannot silently drift from the file a
planner could also open by hand.

WHAT COUNTS AS "BINDING". A row is binding when its activity sits at one of its own bounds,
within `--tol` (default the project's primal feasibility tolerance, `tolerances.hpp`'s
1e-7) - the same test the solver's own feasibility check makes, not a separate judgement
call this script invents. A column is "at a bound" by the same test against its own bounds.
Both lists are sorted by |shadow price| / |reduced cost| descending: the row a planner
should look at first is the one moving the objective fastest per unit of slack, which is
what "sorted by economic impact" means operationally.

MILP: when any column is integer, the row and column duals in the `.sol` file are the FIXED-
INTEGER LP's - the relaxation solved with every integer column's optimal value fixed as
both its lower and upper bound, which is what a MILP's `.sol` file already carries (there is
no other dual a MIP has). The report labels this explicitly rather than presenting a MILP's
duals as if they meant what an LP's do.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_solution_mps import Model, parse_mps  # noqa: E402
from verify_solution_sol import Solution, parse_sol  # noqa: E402

DEFAULT_TOLERANCE = 1e-7


def _near(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _ranging(decrease: dict, increase: dict, name: str) -> dict | None:
    """The .sol ranging pair for `name`: how far the coefficient (a cost for a column, the
    active bound for a row) may FALL and may RISE before the optimal basis changes. These
    are distances, as src/io/writer.cpp prints them (`name allow_decrease allow_increase`),
    not the two ends of an interval; `inf` means no limit in that direction."""
    if name not in decrease or name not in increase:
        return None
    return {"allow_decrease": decrease[name], "allow_increase": increase[name]}


def binding_rows(model: Model, solution: Solution, tol: float) -> list[dict]:
    rows = []
    for i, name in enumerate(model.row_names):
        activity = solution.row_activity.get(name)
        if activity is None:
            continue
        lower, upper = model.row_lower[i], model.row_upper[i]
        at_lower = lower > -1e29 and _near(activity, lower, tol)
        at_upper = upper < 1e29 and _near(activity, upper, tol)
        if not (at_lower or at_upper):
            continue
        shadow_price = solution.row_dual.get(name, 0.0)
        rows.append({
            "name": name,
            "activity": activity,
            "bound": lower if at_lower else upper,
            "at": "lower" if at_lower else "upper",
            "shadow_price": shadow_price,
            "ranging": _ranging(solution.row_ranging_lower, solution.row_ranging_upper, name),
        })
    rows.sort(key=lambda r: -abs(r["shadow_price"]))
    return rows


def nonbasic_columns(model: Model, solution: Solution, tol: float) -> list[dict]:
    columns = []
    for i, name in enumerate(model.col_names):
        reduced_cost = solution.col_dual.get(name)
        if reduced_cost is None or abs(reduced_cost) <= 1e-9:
            continue
        columns.append({
            "name": name,
            "value": solution.col_value.get(name, 0.0),
            "reduced_cost": reduced_cost,
            "ranging": _ranging(solution.col_ranging_lower, solution.col_ranging_upper, name),
        })
    columns.sort(key=lambda c: -abs(c["reduced_cost"]))
    return columns


def columns_at_bounds(model: Model, solution: Solution, tol: float) -> list[dict]:
    columns = []
    for i, name in enumerate(model.col_names):
        value = solution.col_value.get(name)
        if value is None:
            continue
        lower, upper = model.col_lower[i], model.col_upper[i]
        at_lower = lower > -1e29 and _near(value, lower, tol)
        at_upper = upper < 1e29 and _near(value, upper, tol)
        if not (at_lower or at_upper) or lower == upper:
            continue  # a fixed column (lower == upper) is not "at a bound", it has no other
        columns.append({
            "name": name,
            "value": value,
            "bound": lower if at_lower else upper,
            "at": "lower" if at_lower else "upper",
        })
    return columns


def build_report(model: Model, solution: Solution, tol: float) -> dict:
    is_milp = any(model.col_integer)
    objective = solution.header_float("objective")
    return {
        "model": model.name or "(unnamed)",
        "status": solution.status,
        "objective": objective,
        "sense": "maximize" if model.maximize else "minimize",
        "is_milp": is_milp,
        "duals_note": (
            "duals are the fixed-integer LP's (every integer column fixed at its optimal "
            "value); a MIP has no other dual" if is_milp else None
        ),
        "binding_constraints": binding_rows(model, solution, tol),
        "nonbasic_variables": nonbasic_columns(model, solution, tol),
        "variables_at_bounds": columns_at_bounds(model, solution, tol),
    }


def _fmt(value: float) -> str:
    return f"{value:.10g}"


def render_markdown(report: dict) -> str:
    lines = [f"# Solution report - {report['model']}", ""]
    lines.append(f"**Status:** {report['status']}  ")
    if report["objective"] is not None:
        lines.append(f"**Objective ({report['sense']}):** {_fmt(report['objective'])}  ")
    if report["duals_note"]:
        lines.append(f"**Note:** {report['duals_note']}  ")
    lines.append("")

    lines.append("## Binding constraints, by shadow price")
    if report["binding_constraints"]:
        lines.append("| constraint | at | activity | shadow price | bound may fall / rise by |")
        lines.append("|---|---|---:|---:|---|")
        for row in report["binding_constraints"]:
            ranging = (f"-{_fmt(row['ranging']['allow_decrease'])} / "
                       f"+{_fmt(row['ranging']['allow_increase'])}"
                       if row["ranging"] else "-")
            lines.append(f"| {row['name']} | {row['at']} | {_fmt(row['activity'])} | "
                         f"{_fmt(row['shadow_price'])} | {ranging} |")
    else:
        lines.append("*(none)*")
    lines.append("")

    lines.append("## Nonbasic variables, by reduced cost")
    if report["nonbasic_variables"]:
        lines.append("| variable | value | reduced cost | cost may fall / rise by |")
        lines.append("|---|---:|---:|---|")
        for col in report["nonbasic_variables"]:
            ranging = (f"-{_fmt(col['ranging']['allow_decrease'])} / "
                       f"+{_fmt(col['ranging']['allow_increase'])}"
                       if col["ranging"] else "-")
            lines.append(f"| {col['name']} | {_fmt(col['value'])} | "
                         f"{_fmt(col['reduced_cost'])} | {ranging} |")
    else:
        lines.append("*(none)*")
    lines.append("")

    lines.append("## Variables at their bounds")
    if report["variables_at_bounds"]:
        lines.append("| variable | at | value |")
        lines.append("|---|---|---:|")
        for col in report["variables_at_bounds"]:
            lines.append(f"| {col['name']} | {col['at']} | {_fmt(col['value'])} |")
    else:
        lines.append("*(none)*")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("solution", type=Path)
    parser.add_argument("--format", choices=("md", "json"), default="md")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    model = parse_mps(args.model)
    solution = parse_sol(args.solution)
    report = build_report(model, solution, args.tol)

    text = (json.dumps(report, indent=2) if args.format == "json"
           else render_markdown(report))
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
