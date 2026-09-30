#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Floating-point sensitivity against exact sensitivity, over Netlib and the demo LPs (#757).

For every instance: solve with `--ranging --option exact=true`, then re-derive every dual,
reduced cost, cost range and RHS range from the basis the .sol states, in arbitrary-precision
rational arithmetic (tools/verify_solution_sensitivity.py, which shares no code with the
solver), and compare each floating-point value with its exact one. A row per instance: how
many values were compared, how many differ by more than the certification tolerance (1e-9
relative to max(1, |exact|), tolerances.hpp kSensitivityAgreement), the largest relative
disagreement and where it is, how many rows have a two-sided shadow price, and what the
solver's own certification said (computed, or declined past its exact_seconds budget), and
whether the exact repair (#757, option exact_repair) had to move the basis first: the
exact_repair column is `optimal` when the reported basis is exactly optimal, with the exact
pivots and bound flips it took from the engine's basis (0 and 0 when none were needed).

The exact derivation has a time budget per instance; an instance past it is a row that says
so, not a dropped row. The solver's own exact modules get --exact-seconds each (option
exact_seconds), stated in the CSV's exact_seconds column.

    python bench/runners/exact_sensitivity.py --budget 120 --exact-seconds 120 \
        --machine-kind "cloud container"
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stamp  # noqa: E402
from compare_suite import machine_tag  # noqa: E402
import verify_solution_sensitivity as sens  # noqa: E402
from verify_solution_mps import parse_mps  # noqa: E402
from verify_solution_sol import parse_sol  # noqa: E402

INF = float("inf")
DEMO_LPS = ["crude_blend.mps", "crude_blend_infeasible.mps"]

COLUMNS = [
    "instance", "instance_sha256", "rows", "columns", "status", "our_objective",
    "published_objective", "absolute_gap", "relative_gap", "exact_repair",
    "exact_repair_pivots", "exact_repair_flips", "exact_verification", "solver_certification",
    "exact_derivation", "values_compared", "values_disagreeing", "max_relative_disagreement",
    "worst_value", "rows_with_two_sided_shadow_price", "derive_seconds", "wall_seconds",
    "iterations", "exact_seconds", "git_commit", "machine", "timestamp_utc",
]


def relative(reported: float, exact) -> float:
    if exact in (INF, -INF):
        return 0.0 if reported == exact else INF
    if reported in (INF, -INF) or reported != reported:
        return INF
    return float(abs(Fraction(reported) - exact) / max(Fraction(1), abs(exact)))


def compare(solution, exact) -> tuple[int, int, float, str, int]:
    compared = disagreeing = kinks = 0
    worst, where = 0.0, ""
    pairs = []
    for name, e in exact["columns"].items():
        pairs += [(f"{name} reduced cost", solution.col_dual.get(name, 0.0), e["reduced_cost"]),
                  (f"{name} cost decrease", solution.col_ranging_lower.get(name, INF),
                   e["range_lower"]),
                  (f"{name} cost increase", solution.col_ranging_upper.get(name, INF),
                   e["range_upper"])]
    for name, e in exact["rows"].items():
        pairs += [(f"{name} dual", solution.row_dual.get(name, 0.0), e["dual"]),
                  (f"{name} rhs decrease", solution.row_ranging_lower.get(name, INF),
                   e["range_lower"]),
                  (f"{name} rhs increase", solution.row_ranging_upper.get(name, INF),
                   e["range_upper"])]
        kinks += e["left"] != e["right"]
    for label, reported, value in pairs:
        compared += 1
        r = relative(reported, value)
        if not sens.agrees(reported, value):
            disagreeing += 1
        if r > worst:
            worst, where = r, label
    return compared, disagreeing, worst, where, kinks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=REPO_ROOT / "build" / "sankhya.exe")
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--budget", type=float, default=120.0,
                        help="seconds for the exact derivation of one instance")
    parser.add_argument("--exact-seconds", type=float, default=30.0,
                        help="option exact_seconds for the solver's own exact modules")
    parser.add_argument("--machine-kind", default=None,
                        help="what kind of machine this is, e.g. 'cloud container'")
    parser.add_argument("--instances", nargs="*")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if not args.binary.exists():
        args.binary = REPO_ROOT / "build" / "sankhya"
    commit = stamp.stamp(args.binary)
    machine = machine_tag(args.machine_kind)
    when = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    reference = json.loads((REPO_ROOT / "data" / "netlib" / "reference.json").read_text())
    published = {k: v.get("published_optimal") for k, v in reference["instances"].items()}

    files = [REPO_ROOT / "demo" / d for d in DEMO_LPS]
    files += sorted((REPO_ROOT / "data" / "netlib").glob("*.mps"))
    if args.instances:
        files = [f for f in files if f.stem in args.instances]
    rows = []
    for mps in files:
        with tempfile.TemporaryDirectory() as tmp:
            sol_path = Path(tmp) / "s.sol"
            started = time.perf_counter()
            subprocess.run([str(args.binary), "solve", str(mps), "--ranging",
                            "--option", "exact=true", "--option", "log_to_console=false",
                            "--option", f"exact_seconds={args.exact_seconds}",
                            "--time-limit", str(args.time_limit), "--write-sol", str(sol_path)],
                           capture_output=True, text=True, check=False)
            wall = time.perf_counter() - started
            model = parse_mps(mps)
            solution = parse_sol(sol_path) if sol_path.exists() else None
        row = {"instance": mps.stem, "instance_sha256": hashlib.sha256(mps.read_bytes()).hexdigest(),
               "rows": model.num_rows, "columns": model.num_cols,
               "status": solution.status if solution else "no_output",
               "wall_seconds": f"{wall:.3f}", "exact_seconds": args.exact_seconds,
               "git_commit": commit, "machine": machine,
               "timestamp_utc": when}
        ref = published.get(mps.stem)
        if solution is not None:
            row["iterations"] = solution.header.get("iterations", "")
            ours = solution.header_float("objective")
            row["our_objective"] = "" if ours is None else repr(ours)
            row["solver_certification"] = solution.header.get("sensitivity_verification", "")
            row["exact_verification"] = solution.header.get("exact_verification", "")
            for key in ("exact_repair", "exact_repair_pivots", "exact_repair_flips"):
                row[key] = solution.header.get(key, "")
            if ref is not None and ours is not None:
                row["published_objective"] = repr(ref)
                row["absolute_gap"] = repr(abs(ours - ref))
                row["relative_gap"] = repr(abs(ours - ref) / max(1.0, abs(ref)))
        if solution is not None and solution.status == "optimal" and solution.col_ranging_lower:
            started = time.perf_counter()
            try:
                exact = sens.derive(model, solution, args.budget)
                compared, bad, worst, where, kinks = compare(solution, exact)
                row.update({"exact_derivation": "done", "values_compared": compared,
                            "values_disagreeing": bad, "max_relative_disagreement": repr(worst),
                            "worst_value": where, "rows_with_two_sided_shadow_price": kinks})
            except sens.Declined as why:
                row["exact_derivation"] = f"declined: {why}"
            row["derive_seconds"] = f"{time.perf_counter() - started:.3f}"
        else:
            row["exact_derivation"] = "no optimal basis with ranging"
        rows.append(row)
        print(f"{row['instance']:<12} {row['status']:<12} {row.get('exact_repair', ''):<8}"
              f" {row.get('exact_repair_pivots', '')!s:>4}p {row.get('solver_certification', ''):<9}"
              f" {row['exact_derivation'][:40]:<40} {row.get('values_disagreeing', '')!s:>5}/"
              f"{row.get('values_compared', '')!s:<6} max {row.get('max_relative_disagreement', '')}"
              f" kinks {row.get('rows_with_two_sided_shadow_price', '')}", flush=True)
    out = args.out or REPO_ROOT / "bench" / "results" / f"exact-sensitivity-{commit}.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    done = [r for r in rows if r.get("exact_derivation") == "done"]
    print(f"exact derivation done on {len(done)} of {len(rows)}; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
