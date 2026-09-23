#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Infeasibility repair: the smallest weighted relaxation that makes a model feasible (#523).

An IIS (`src/core/iis.cpp`, exposed through the C API and Python bindings) says WHICH
constraints conflict. This says HOW FAR each one would have to move: elastic programming
(Chinneck, *Feasibility and Infeasibility in Optimization*, Springer 2008, ch. 8) - one
non-negative "elastic" column per relaxable row (and, with `--bounds`, per relaxable
column bound), minimizing their weighted sum (L1) instead of the model's own objective,
solved by the SAME LP engine every other model in this project goes through. No new solver
algorithm: this is a MODEL TRANSFORMATION plus a normal `solve()` call, exactly the way
presolve and postsolve already wrap the same seam.

    python tools/repair_infeasibility.py model.mps
    python tools/repair_infeasibility.py model.mps --bounds --format json
    python tools/repair_infeasibility.py model.mps --optimize   # phase 2: re-optimize the
                                                                 # original objective subject
                                                                 # to the minimal relaxation

THE TRANSFORMATION. Row i, `l_i <= (Ax)_i <= u_i`, becomes `l_i <= (Ax)_i + e_lo_i - e_hi_i
<= u_i` with `e_lo_i, e_hi_i >= 0` new columns (only the side(s) that have a finite bound
get one). A single elastic pair per row is enough for a RANGE row too, not only an
equality: minimising their sum never pays for both at once, because for any value the true
`(Ax)_i` lands at, satisfying whichever single bound it violates (if either) is always at
least as cheap as also moving the elastic variable on the side that was not violated - so
the minimiser never uses more of either than the repair actually needs. `--bounds` does the
same for a column's own bounds by moving them into a new row of exactly this shape and
freeing the column itself to (-inf, inf), so one mechanism covers rows and bounds alike.

Phase 1 replaces the model's objective with the weighted elastic sum (default weight 1 per
row/bound, `--weights name=w,...` to override) and minimizes it. `--optimize` runs phase 2:
with every elastic column's UPPER bound fixed at its phase-1 value (so the repair found
cannot get any bigger) and the original objective restored, re-solve - the cheapest way
through a model at least as feasible as phase 1 already proved possible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bindings" / "python"))

from verify_solution_mps import Model as SourceModel, parse_mps  # noqa: E402

TOLERANCE = 1e-7


def parse_weights(pairs: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for pair in pairs:
        name, _, value = pair.partition("=")
        if not value:
            raise ValueError(f"--weights expects name=value, got {pair!r}")
        weights[name] = float(value)
    return weights


def build_elastic_model(source: SourceModel, *, include_bounds: bool,
                        weights: dict[str, float]):
    import sankhya

    model = sankhya.Model(maximize=False)  # phase 1 always MINIMIZES the elastic sum
    for j in range(source.num_cols):
        model.add_column(cost=0.0, lower=source.col_lower[j], upper=source.col_upper[j],
                         integer=source.col_integer[j], name=source.col_names[j])
    for i in range(source.num_rows):
        model.add_row(lower=source.row_lower[i], upper=source.row_upper[i],
                      name=source.row_names[i])
    for j, entries in enumerate(source.entries):
        for row, value in entries:
            model.set_coefficient(row, j, value)

    # Elastic columns, one pair per row that has a finite bound to relax. `elastics` maps
    # each elastic column's index back to (kind, name, sense) so the report can read the
    # solved values off by name afterwards.
    elastics: list[tuple[str, str, str, int]] = []  # (kind, target_name, "lower"/"upper", col)
    inf = float(sankhya.INFINITY)
    for i in range(source.num_rows):
        name = source.row_names[i]
        weight = weights.get(name, 1.0)
        if source.row_lower[i] > -inf:
            col = model.add_column(cost=weight, lower=0.0, upper=None,
                                   name=f"ELASTIC_LO[{name}]")
            model.set_coefficient(i, col, 1.0)
            elastics.append(("row", name, "lower", col))
        if source.row_upper[i] < inf:
            col = model.add_column(cost=weight, lower=0.0, upper=None,
                                   name=f"ELASTIC_HI[{name}]")
            model.set_coefficient(i, col, -1.0)
            elastics.append(("row", name, "upper", col))

    if include_bounds:
        for j in range(source.num_cols):
            name = source.col_names[j]
            weight = weights.get(name, 1.0)
            lower, upper = source.col_lower[j], source.col_upper[j]
            if lower <= -inf and upper >= inf:
                continue  # already free - nothing to relax
            row = model.add_row(lower=lower, upper=upper, name=f"BOUND[{name}]")
            model.set_coefficient(row, j, 1.0)
            if lower > -inf:
                col = model.add_column(cost=weight, lower=0.0, upper=None,
                                       name=f"ELASTIC_LO[BOUND[{name}]]")
                model.set_coefficient(row, col, 1.0)
                elastics.append(("bound", name, "lower", col))
            if upper < inf:
                col = model.add_column(cost=weight, lower=0.0, upper=None,
                                       name=f"ELASTIC_HI[BOUND[{name}]]")
                model.set_coefficient(row, col, -1.0)
                elastics.append(("bound", name, "upper", col))
            # The column's own bound no longer enforces anything - the new row (with its
            # elastic slack) does, so the column is freed to let the row be the only thing
            # that can bind it.
            model.set_col_bounds(j, -inf, None)

    return model, elastics


def repair(source: SourceModel, *, include_bounds: bool, weights: dict[str, float],
          optimize: bool, log: bool):
    model, elastics = build_elastic_model(source, include_bounds=include_bounds,
                                          weights=weights)
    phase1 = model.solve(log_to_console=log, presolve=True)
    if phase1.status not in ("optimal", "feasible"):
        return {
            "phase1_status": phase1.status,
            "repairable": False,
            "message": (f"the elastic relaxation itself is {phase1.status} - the model "
                       "cannot be made feasible by relaxing the rows/bounds offered "
                       "(e.g. a column's own bounds conflict, lower > upper, which no row "
                       "relaxation touches)"),
            "relaxations": [],
        }

    values = phase1.x
    relaxations = []
    for kind, name, side, col in elastics:
        amount = values[col]
        if amount > TOLERANCE:
            relaxations.append({"kind": kind, "name": name, "side": side, "amount": amount})
    relaxations.sort(key=lambda r: -r["amount"])

    total = sum(r["amount"] * weights.get(r["name"], 1.0) for r in relaxations)
    report = {
        "phase1_status": phase1.status,
        "repairable": True,
        "total_weighted_relaxation": total,
        "relaxations": relaxations,
    }

    if optimize and relaxations:
        # Phase 2: cap every elastic column at what phase 1 actually used (never more - the
        # repair already found is a CEILING, not a floor, for phase 2 to spend), restore the
        # original objective, re-solve.
        for _, _, _, col in elastics:
            model.set_col_bounds(col, 0.0, values[col])
        for j in range(source.num_cols):
            model.set_cost(j, source.col_cost[j])
        for _, _, _, col in elastics:
            model.set_cost(col, 0.0)
        phase2 = model.solve(log_to_console=log, presolve=True, start=phase1)
        report["phase2_status"] = phase2.status
        report["phase2_objective"] = phase2.objective if phase2.claims_a_point else None

    return report


def render(report: dict, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(report, indent=2)
    lines = []
    if not report["repairable"]:
        lines.append(f"NOT REPAIRABLE: {report['message']}")
        return "\n".join(lines)
    lines.append(f"Repaired (phase 1: {report['phase1_status']}, "
                 f"total weighted relaxation {report['total_weighted_relaxation']:.6g}):")
    if not report["relaxations"]:
        lines.append("  (the model was already feasible - no relaxation was needed)")
    for r in report["relaxations"]:
        label = r["name"] if r["kind"] == "row" else f"bound on {r['name']}"
        lines.append(f"  {label}: {r['side']} moved by {r['amount']:.6g}")
    if "phase2_status" in report:
        lines.append(f"\nPhase 2 (original objective, relaxation capped at phase 1's): "
                     f"{report['phase2_status']}"
                     + (f", objective {report['phase2_objective']:.10g}"
                        if report["phase2_objective"] is not None else ""))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("--bounds", action="store_true", dest="include_bounds",
                        help="also make column bounds relaxable, not only rows")
    parser.add_argument("--weights", action="append", default=[], metavar="NAME=WEIGHT")
    parser.add_argument("--optimize", action="store_true",
                        help="phase 2: re-optimize the original objective, relaxation capped")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--log", action="store_true", help="show the solver's own log")
    args = parser.parse_args()

    source = parse_mps(args.model)
    weights = parse_weights(args.weights)
    report = repair(source, include_bounds=args.include_bounds, weights=weights,
                    optimize=args.optimize, log=args.log)
    print(render(report, args.format))
    return 0 if report["repairable"] else 1


if __name__ == "__main__":
    sys.exit(main())
