#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Parametric LP: the optimal value as a function of one cost or one right-hand side (#522).

    python tools/parametric.py model.mps --cost X --from 60 --to 90 --out curve.csv
    python tools/parametric.py model.mps --row DEMAND --side upper --from 100 --to 200

Sensitivity ranging (#220, `--ranging`) already answers "how far can this ONE coefficient
move before the CURRENT basis stops being optimal" - which is exactly the ratio test a
dedicated parametric simplex (Gass & Saaty, *The computational algorithm for the parametric
objective function*, Naval Research Logistics Quarterly 2, 1955; Murty, *Linear
Programming*, Wiley 1983, ch. 8) performs to find its next breakpoint. This tool walks
BETWEEN breakpoints the same way: solve at the current parameter value, read how far
`col_ranging_upper[j]` (a cost) or `row_ranging_upper[i]` (a bound) says the basis can move
before it changes, jump the parameter there - warm-started (#218) from the basis that
already proved optimal at the point being left - and repeat until the requested range is
covered.

WHERE THIS DIFFERS FROM A DEDICATED PARAMETRIC SIMPLEX. A true parametric simplex changes
ONE non-basic column (the entering/leaving pivot) at each breakpoint and continues the SAME
pivot sequence; this tool re-solves at each breakpoint instead - the ranging interval STILL
correctly names the breakpoint, but where a dedicated implementation would take exactly one
pivot to cross it, this takes a full (fast, warm-started) re-solve. #522's own acceptance
criterion - "each breakpoint's objective verified by an independent re-solve at that
parameter value" - is what this trade buys back: every point on the reported curve already
IS an independent solve, not a value carried forward from ranging's own linear estimate.

Output: a CSV of breakpoints (parameter value, objective, status, which columns entered or
left the basis since the previous breakpoint) and a one-line plain description of the change
at each.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bindings" / "python"))

from verify_solution_mps import parse_mps  # noqa: E402

TOLERANCE = 1e-9
MAX_BREAKPOINTS = 2000  # a runaway direction (a bound that never binds again) still halts


def _describe_basis_change(previous: dict[str, str] | None,
                           current: dict[str, str]) -> str:
    if previous is None:
        return "initial basis"
    entered = [name for name, status in current.items()
              if status == "basic" and previous.get(name) != "basic"]
    left = [name for name, status in current.items()
           if status != "basic" and previous.get(name) == "basic"]
    if not entered and not left:
        return "no basis change (objective still moves linearly)"
    parts = []
    if entered:
        parts.append(f"entered: {', '.join(sorted(entered))}")
    if left:
        parts.append(f"left: {', '.join(sorted(left))}")
    return "; ".join(parts)


def sweep_cost(model_path: Path, column: str, from_value: float, to_value: float,
              options: dict) -> list[dict]:
    import sankhya

    source = parse_mps(model_path)
    if column not in source.col_index:
        raise ValueError(f"no column named {column!r} in {model_path}")
    j = source.col_index[column]

    model = _rebuild(sankhya, source)
    direction = 1.0 if to_value >= from_value else -1.0
    t = from_value
    model.set_cost(j, t)

    rows = []
    previous_result = None
    previous_basis: dict[str, str] | None = None
    for _ in range(MAX_BREAKPOINTS):
        result = model.solve(ranging=True, start=previous_result, log_to_console=False,
                             **options)
        basis = dict(zip(source.col_names, result.col_statuses))
        rows.append({
            "parameter": t, "objective": result.objective if result.claims_a_point else "",
            "status": result.status,
            "change": _describe_basis_change(previous_basis, basis),
        })
        previous_result, previous_basis = result, basis

        if (direction > 0 and t >= to_value - TOLERANCE) or (
                direction < 0 and t <= to_value + TOLERANCE):
            break
        step = result.col_ranging_upper[j] if direction > 0 else -result.col_ranging_lower[j]
        if step == 0.0 or not _finite(step):
            t = to_value  # the basis never changes again in this direction - one more solve
        else:
            t = t + step if direction > 0 else t + step
            if direction > 0:
                t = min(t, to_value)
            else:
                t = max(t, to_value)
        model.set_cost(j, t)
    return rows


def sweep_row(model_path: Path, row: str, side: str, from_value: float, to_value: float,
             options: dict) -> list[dict]:
    import sankhya

    source = parse_mps(model_path)
    if row not in source.row_index:
        raise ValueError(f"no row named {row!r} in {model_path}")
    i = source.row_index[row]

    model = _rebuild(sankhya, source)
    direction = 1.0 if to_value >= from_value else -1.0
    t = from_value
    lower, upper = source.row_lower[i], source.row_upper[i]

    def apply(value: float) -> None:
        nonlocal lower, upper
        if side == "upper":
            upper = value
        else:
            lower = value
        model.set_row_bounds(i, lower if lower > -1e29 else None,
                            upper if upper < 1e29 else None)

    apply(t)
    rows = []
    previous_result = None
    previous_basis: dict[str, str] | None = None
    for _ in range(MAX_BREAKPOINTS):
        result = model.solve(ranging=True, start=previous_result, log_to_console=False,
                             **options)
        basis = dict(zip(source.col_names, result.col_statuses))
        rows.append({
            "parameter": t, "objective": result.objective if result.claims_a_point else "",
            "status": result.status,
            "change": _describe_basis_change(previous_basis, basis),
        })
        previous_result, previous_basis = result, basis

        if (direction > 0 and t >= to_value - TOLERANCE) or (
                direction < 0 and t <= to_value + TOLERANCE):
            break
        step = result.row_ranging_upper[i] if direction > 0 else -result.row_ranging_lower[i]
        if step == 0.0 or not _finite(step):
            t = to_value
        else:
            t = t + step
            t = min(t, to_value) if direction > 0 else max(t, to_value)
        apply(t)
    return rows


def _finite(value: float) -> bool:
    return abs(value) < 1e29


def _rebuild(sankhya, source):
    model = sankhya.Model(maximize=source.maximize)
    for j in range(source.num_cols):
        model.add_column(cost=source.col_cost[j], lower=source.col_lower[j],
                         upper=(None if source.col_upper[j] >= 1e29 else source.col_upper[j]),
                         integer=source.col_integer[j], name=source.col_names[j])
    for i in range(source.num_rows):
        model.add_row(lower=(None if source.row_lower[i] <= -1e29 else source.row_lower[i]),
                      upper=(None if source.row_upper[i] >= 1e29 else source.row_upper[i]),
                      name=source.row_names[i])
    for j, entries in enumerate(source.entries):
        for row, value in entries:
            model.set_coefficient(row, j, value)
    return model


def provenance(model_path: Path) -> dict:
    """What every CSV under bench/results carries beside its numbers: the commit the
    solver was built from, the machine, and the sha256 of the instance (ENGINEERING_RULES,
    evidence rules). The library reports no commit of its own, so the stamp is read from
    the CLI built beside it, through bench/runners/stamp.py (#433)."""
    import hashlib
    import platform
    import sankhya
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "runners"))
    import stamp  # noqa: E402
    digest = hashlib.sha256()
    with open(model_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    try:
        commit = stamp.stamp(sankhya.locate_executable())
    except Exception:  # no CLI beside the library: say so rather than guess from the tree
        commit = "unknown"
    return {"git_commit": commit, "machine": f"{platform.system()}-{platform.machine()}",
            "instance_sha256": digest.hexdigest()}


def write_csv(rows: list[dict], path: Path | None, stamp_columns: dict | None = None) -> None:
    handle = open(path, "w", newline="", encoding="utf-8") if path else sys.stdout
    extra = stamp_columns or {}
    try:
        writer = csv.DictWriter(handle, fieldnames=["parameter", "objective", "status",
                                                     "change", *extra.keys()])
        writer.writeheader()
        writer.writerows({**row, **extra} for row in rows)
    finally:
        if path:
            handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--cost", metavar="COLUMN", help="vary this column's objective cost")
    target.add_argument("--row", metavar="ROW", help="vary this row's bound")
    parser.add_argument("--side", choices=("lower", "upper"), default="upper",
                        help="which bound of --row to vary (ignored for --cost)")
    parser.add_argument("--from", dest="from_value", type=float, required=True)
    parser.add_argument("--to", dest="to_value", type=float, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--option", action="append", default=[], dest="raw_options",
                        metavar="NAME=VALUE")
    args = parser.parse_args()

    options: dict[str, object] = {}
    for assignment in args.raw_options:
        name, _, value = assignment.partition("=")
        options[name] = value

    if args.cost is not None:
        rows = sweep_cost(args.model, args.cost, args.from_value, args.to_value, options)
    else:
        rows = sweep_row(args.model, args.row, args.side, args.from_value, args.to_value,
                         options)

    write_csv(rows, args.out, provenance(args.model))
    if args.out:
        print(f"wrote {len(rows)} breakpoint(s) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
