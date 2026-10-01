#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Infeasibility repair over Netlib's infeasible set, each repair checked by the verifier (#523).

For every instance in data/netlib-infeasible:

  1. tools/repair_infeasibility.py finds the smallest weighted relaxation of the rows and the
     column bounds (elastic programming, Chinneck 2008 ch. 8) that makes the model feasible;
  2. the relaxation is applied to the model as read from the file - a row's lower side moved
     down by its amount, an upper side up, the same for a column bound - and the repaired
     model is written out as free MPS;
  3. the repaired model, with its objective set to zero so that the question asked is only
     "is there a feasible point", is solved by the CLI from scratch like any other model;
  4. tools/verify_solution.py checks that answer against the repaired model with its own
     reader and arithmetic.

A pass is a point the verifier accepts on the repaired model: status optimal or feasible, and
every row and bound satisfied. A run with no point (a limit, numerical_error) is not a pass. The repair tool and the verifier share only the MPS parser, which reads the
original file; the repaired file is written here and read back by both the CLI and the
verifier like any other model.

    python bench/runners/netlib_repair.py --binary build/sankhya --out bench/results/netlib-repair-<sha>.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "bench" / "runners"))
from verify_solution_mps import parse_mps  # noqa: E402
import repair_infeasibility  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "netlib-infeasible"
INF = 1e30


def finite(v: float) -> bool:
    return math.isfinite(v) and abs(v) < INF


def apply_repair(model, relaxations) -> None:
    for r in relaxations:
        amount = float(r["amount"])
        if r["kind"] == "row":
            i = model.row_index[r["name"]]
            if r["side"] == "lower":
                model.row_lower[i] -= amount
            else:
                model.row_upper[i] += amount
        else:
            j = model.col_index[r["name"]]
            if r["side"] == "lower":
                model.col_lower[j] -= amount
            else:
                model.col_upper[j] += amount


def free_name(name: str) -> str:
    """Fixed-format MPS allows a name that starts with '$' (pang has $0AICCB); free format
    reads a field starting with '$' as a comment. Such a name is written with a prefix, in the
    repaired file only, which both the CLI and the verifier then read."""
    return "D_" + name if name.startswith("$") else name


def write_free_mps(model, path: Path) -> None:
    """A plain free-MPS writer for an LP: N, E, L, G rows, RANGES, and every column bound."""
    model.col_names = [free_name(n) for n in model.col_names]
    model.row_names = [free_name(n) for n in model.row_names]
    lines = [f"NAME {model.name or 'REPAIRED'}"]
    if model.maximize:
        lines += ["OBJSENSE", "    MAX"]
    lines.append("ROWS")
    lines.append(" N  OBJ_REPAIR")
    kinds, rhs, ranges = [], [], []
    for i, name in enumerate(model.row_names):
        lo, hi = model.row_lower[i], model.row_upper[i]
        if finite(lo) and finite(hi) and lo == hi:
            kinds.append("E"); rhs.append(lo); ranges.append(None)
        elif finite(lo) and finite(hi):
            kinds.append("L"); rhs.append(hi); ranges.append(hi - lo)
        elif finite(hi):
            kinds.append("L"); rhs.append(hi); ranges.append(None)
        elif finite(lo):
            kinds.append("G"); rhs.append(lo); ranges.append(None)
        else:
            kinds.append("N"); rhs.append(None); ranges.append(None)
        lines.append(f" {kinds[-1]}  {name}")
    lines.append("COLUMNS")
    for j, name in enumerate(model.col_names):
        written = len(lines)
        if model.col_cost[j] != 0.0:
            lines.append(f"    {name}  OBJ_REPAIR  {model.col_cost[j]!r}")
        for i, v in model.entries[j]:
            if kinds[i] != "N":
                lines.append(f"    {name}  {model.row_names[i]}  {v!r}")
        if len(lines) == written:  # an empty column still has to be declared
            lines.append(f"    {name}  OBJ_REPAIR  0.0")
    lines.append("RHS")
    if model.objective_offset:
        lines.append(f"    RHS  OBJ_REPAIR  {-model.objective_offset!r}")
    for i, name in enumerate(model.row_names):
        if rhs[i] is not None and rhs[i] != 0.0:
            lines.append(f"    RHS  {name}  {rhs[i]!r}")
    if any(r is not None for r in ranges):
        lines.append("RANGES")
        for i, name in enumerate(model.row_names):
            if ranges[i] is not None:
                lines.append(f"    RNG  {name}  {ranges[i]!r}")
    lines.append("BOUNDS")
    for j, name in enumerate(model.col_names):
        lo, hi = model.col_lower[j], model.col_upper[j]
        if not finite(lo) and not finite(hi):
            lines.append(f" FR BND  {name}")
            continue
        if finite(lo) and finite(hi) and lo == hi:
            lines.append(f" FX BND  {name}  {lo!r}")
            continue
        if not finite(lo):
            lines.append(f" MI BND  {name}")
        elif lo != 0.0:
            lines.append(f" LO BND  {name}  {lo!r}")
        if finite(hi):
            lines.append(f" UP BND  {name}  {hi!r}")
    lines.append("ENDATA")
    path.write_text("\n".join(lines) + "\n")


def field(text: str, key: str) -> str:
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == key:
            return parts[1]
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--machine", default=f"{platform.system()}-{platform.machine()}")
    parser.add_argument("--instances", nargs="*")
    args = parser.parse_args()

    manifest = json.loads((DATA_DIR / "reference.json").read_text())
    names = sorted(manifest.get("instances", manifest))
    if args.instances:
        names = [n for n in names if n in args.instances]
    commit = subprocess.run([str(args.binary), "--version"], capture_output=True,
                            text=True).stdout.split("(")[1].split(",")[0]
    rows = []
    work = Path(tempfile.mkdtemp(prefix="sankhya-repair-"))
    for name in names:
        mps = DATA_DIR / f"{name}.mps"
        t0 = time.perf_counter()
        source = parse_mps(mps)
        try:
            report = repair_infeasibility.repair(source, include_bounds=True, weights={},
                                                 optimize=False, log=False)
        except Exception as error:  # noqa: BLE001 - recorded, not raised
            report = {"repairable": False, "phase1_status": "error",
                      "message": str(error)[:200], "relaxations": []}
        repair_seconds = time.perf_counter() - t0
        row = {"instance": name, "rows": len(source.row_names), "cols": len(source.col_names),
               "repair_status": report.get("phase1_status", ""),
               "repairable": int(bool(report.get("repairable"))),
               "relaxed_items": len(report.get("relaxations", [])),
               "total_relaxation": report.get("total_weighted_relaxation", ""),
               "largest_move": (f"{report['relaxations'][0]['name']} {report['relaxations'][0]['side']} "
                                f"{report['relaxations'][0]['amount']:.6g}"
                                if report.get("relaxations") else ""),
               "repair_seconds": round(repair_seconds, 3),
               "resolve_status": "", "verified": 0, "verifier_message": "",
               "git_commit": commit, "machine": args.machine, "time_limit": args.time_limit}
        if report.get("repairable"):
            repaired = parse_mps(mps)
            apply_repair(repaired, report["relaxations"])
            repaired.col_cost = [0.0] * len(repaired.col_cost)  # feasibility only
            repaired.objective_offset = 0.0
            out_mps, out_sol = work / f"{name}_repaired.mps", work / f"{name}_repaired.sol"
            write_free_mps(repaired, out_mps)
            solve = subprocess.run([str(args.binary), "solve", str(out_mps), "--write-sol",
                                    str(out_sol), "--option", f"time_limit={args.time_limit}"],
                                   capture_output=True, text=True)
            row["resolve_status"] = field(solve.stdout, "status") or field(solve.stdout, "Result:")
            if out_sol.exists():
                check = subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "verify_solution.py"),
                                        str(out_mps), str(out_sol), "--quiet"],
                                       capture_output=True, text=True)
                row["verified"] = int(check.returncode == 0
                                      and row["resolve_status"] in ("optimal", "feasible"))
                row["verifier_message"] = (check.stdout.strip().splitlines() or [""])[-1][:160]
        rows.append(row)
        print(f"{name:12s} repair={row['repair_status']:10s} items={row['relaxed_items']:4d} "
              f"resolve={row['resolve_status']:10s} verified={row['verified']}")
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    passed = sum(r["verified"] for r in rows)
    print(f"{passed}/{len(rows)} repaired models solved and accepted by tools/verify_solution.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
