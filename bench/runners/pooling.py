#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run SANKHYA over the standard pooling instances and emit the evidence CSV (#516).

Thirteen instances (data/pooling/reference.json) in three formulations each (P, Q, PQ;
bench/runners/pooling_models.py): 39 runs. Every one is a non-convex QCQP. A run PASSES only
when all of these hold:

  1. the status is `optimal`, which a global method may claim only when its proven bound
     closes the gap (#514);
  2. the objective is within the pass tolerance of the published global optimum;
  3. the proven lower bound, when the solver reports one, does not exceed the published
     optimum by more than that tolerance - a bound above a known feasible value is wrong;
  4. tools/verify_solution.py, which re-reads the model with its own parser, accepts the
     .sol file against the ORIGINAL model, quadratic rows included.

What the solver does not do is recorded as what it is. A model the reader refuses is
`refused` with the reader's own message, never a skipped row: a benchmark that drops the
instances it cannot read reports a pass rate over a different set.

The CSV has the columns ENGINEERING_RULES.md requires (instance, sha256, our and the
reference objective, both gaps, status, wall time, iterations and nodes, commit, machine),
plus the formulation and the root relaxation bound, which is what the PQ-versus-P comparison
of #516 reads.

Usage:
    python bench/runners/pooling.py
    python bench/runners/pooling.py --binary build/sankhya --time-limit 60
    python bench/runners/pooling.py --instances haverly1 foulds2 --formulations pq
    python bench/runners/pooling.py --solver-option nonconvex=global
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stamp  # noqa: E402  (#433: stamps from the binary)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "pooling"
RESULTS_DIR = REPO_ROOT / "bench" / "results"
VERIFIER = REPO_ROOT / "tools" / "verify_solution.py"
FORMULATIONS = ("p", "q", "pq")

# The MIP gap convention of include/sankhya/tolerances.hpp: relative 1e-4 with an absolute
# floor of 1e-6. The published optima are proven to about 1e-8 relative (reference.json), so
# this is the width of OUR claim, not of the reference's uncertainty.
PASS_RELATIVE = 1e-4
PASS_ABSOLUTE = 1e-6

CSV_COLUMNS = [
    "instance",
    "formulation",
    "model_file",
    "instance_sha256",
    # 0 when the file on disk is not the one reference.json recorded (line-ending blind).
    "matches_manifest",
    "rows",
    "columns",
    "quadratic_rows",
    "status",
    "message",
    "our_objective",
    "reference_objective",
    "absolute_gap",
    "relative_gap",
    "matches_reference",
    # The proven lower bound at the end and at the root relaxation; empty when the solver
    # reports none (every engine that refuses the model).
    "dual_bound",
    "root_bound",
    "bound_valid",
    "independently_verified",
    "verifier_message",
    "passed",
    "failure_reason",
    "wall_seconds",
    "solver_seconds",
    "iterations",
    "nodes",
    "algorithm",
    "time_limit",
    "git_commit",
    "machine",
    "timestamp_utc",
    "solver_options",
]


def as_number(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def tolerance(reference: float) -> float:
    return max(PASS_ABSOLUTE, PASS_RELATIVE * abs(reference))


def normalized_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def count_quadratic_rows(path: Path) -> int:
    return sum(1 for line in path.read_text(errors="replace").splitlines()
               if line.upper().startswith("QCMATRIX"))


def classify(flat: dict, reference: float) -> str:
    """Why a run is not a pass, or "" when it is. In the order the evidence is read."""
    status = flat["status"]
    if status == "refused":
        return "refused at read: " + ("quadratic constraints (QCMATRIX)"
                                      if "QCMATRIX" in flat.get("message", "")
                                      else "model not read")
    if status in ("crashed", "no_output"):
        return f"solver {status.replace('_', ' ')}"
    ours = as_number(flat.get("objective"))
    bound = as_number(flat.get("dual_bound"))
    if bound is not None and bound > reference + tolerance(reference):
        return "proven bound above the published optimum"
    if status == "not_solved":
        return "refused by the solver"
    if status in ("infeasible", "unbounded"):
        return f"wrong verdict: {status}"
    if status == "feasible" or (status != "optimal" and ours is not None):
        return "gap not closed" if ours is None or abs(ours - reference) <= tolerance(
            reference) else "gap not closed, incumbent worse than the optimum"
    if status != "optimal":
        return f"no verdict: {status}"
    if ours is None or abs(ours - reference) > tolerance(reference):
        return "optimal claimed at a different objective"
    if flat.get("verifier_rc") is None:
        return "not verified"
    if flat["verifier_rc"] != 0:
        return "rejected by the verifier"
    return ""


def verifier_failures(text: str) -> str:
    failing = [line.strip() for line in text.splitlines() if "[FAIL]" in line]
    if failing:
        return "; ".join(failing)[:400]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1][:400] if lines else ""


def default_binary() -> Path:
    sys.path.insert(0, str(REPO_ROOT / "bindings" / "python"))
    import sankhya
    try:
        return sankhya.locate_executable()
    except sankhya.SankhyaError as error:
        raise SystemExit(str(error))


def run_one(binary: Path, mps: Path, time_limit: float, verify: bool,
            solver_options: list[str]) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        stats_path = Path(tmp) / "stats.json"
        sol_path = Path(tmp) / "solution.sol"
        command = [str(binary), "solve", str(mps), "--stats", str(stats_path), "--write-sol",
                   str(sol_path), "--time-limit", str(time_limit), "--option",
                   "log_to_console=false"]
        for option in solver_options:
            command += ["--option", option]
        started = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True)
        wall = time.perf_counter() - started
        base = {"wall_seconds": wall, "verifier_rc": None, "verifier_text": ""}
        if not stats_path.exists():
            # Exit code 3 is the CLI's "could not load the model" (apps/sankhya-cli).
            status = ("refused" if completed.returncode == 3 else
                      "crashed" if completed.returncode not in (0, 1) else "no_output")
            return {**base, "status": status,
                    "message": (completed.stderr or completed.stdout).strip()[:300]}
        blob = json.loads(stats_path.read_text())
        result, model, effort = (blob.get(key, {}) for key in ("result", "model", "effort"))
        flat = {
            **base, "status": result.get("status", "unknown"),
            "message": result.get("message", "")[:300],
            "objective": as_number(result.get("objective")),
            "dual_bound": as_number(result.get("dual_bound")),
            "root_bound": as_number(effort.get("root_bound")),
            "algorithm": result.get("algorithm", ""), "rows": model.get("rows", ""),
            "columns": model.get("columns", ""), "iterations": effort.get("iterations", ""),
            "nodes": effort.get("nodes", ""), "solver_seconds": effort.get("solve_seconds", ""),
        }
        if verify and sol_path.exists():
            check = subprocess.run([sys.executable, str(VERIFIER), str(mps), str(sol_path)],
                                   capture_output=True, text=True)
            flat["verifier_rc"] = check.returncode
            flat["verifier_text"] = check.stdout + check.stderr
        return flat


def csv_row(name: str, formulation: str, mps: Path, entry: dict, flat: dict,
            reference: float, context: dict) -> dict:
    ours = as_number(flat.get("objective"))
    bound = as_number(flat.get("dual_bound"))
    absolute = None if ours is None else abs(ours - reference)
    relative = None if absolute is None else absolute / max(1.0, abs(reference))
    reason = classify(flat, reference)
    recorded = entry.get("models", {}).get(formulation, {}).get("sha256")
    verified = ("" if flat.get("verifier_rc") is None else int(flat["verifier_rc"] == 0))
    return {
        "instance": name, "formulation": formulation, "model_file": mps.name,
        "instance_sha256": hashlib.sha256(mps.read_bytes()).hexdigest(),
        "matches_manifest": int(recorded == normalized_sha256(mps)),
        "rows": flat.get("rows", ""), "columns": flat.get("columns", ""),
        "quadratic_rows": count_quadratic_rows(mps),
        "status": flat["status"], "message": flat.get("message", ""),
        "our_objective": "" if ours is None else repr(ours),
        "reference_objective": repr(reference),
        "absolute_gap": "" if absolute is None else repr(absolute),
        "relative_gap": "" if relative is None else repr(relative),
        "matches_reference": "" if ours is None else int(absolute <= tolerance(reference)),
        "dual_bound": "" if bound is None else repr(bound),
        "root_bound": "" if flat.get("root_bound") is None else repr(flat["root_bound"]),
        "bound_valid": "" if bound is None else int(bound <= reference + tolerance(reference)),
        "independently_verified": verified,
        "verifier_message": verifier_failures(flat.get("verifier_text", "")),
        "passed": int(reason == ""), "failure_reason": reason,
        "wall_seconds": round(flat.get("wall_seconds", 0.0), 6),
        "solver_seconds": flat.get("solver_seconds", ""),
        "iterations": flat.get("iterations", ""), "nodes": flat.get("nodes", ""),
        "algorithm": flat.get("algorithm", ""), **context,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--instances", nargs="*")
    parser.add_argument("--formulations", nargs="*", default=list(FORMULATIONS),
                        choices=FORMULATIONS)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--solver-option", action="append", default=[], metavar="KEY=VALUE",
                        help="passed as --option KEY=VALUE; repeatable, recorded in the CSV")
    parser.add_argument("--out", type=Path, default=None,
                        help="destination CSV; relative paths resolve against the repo root")
    args = parser.parse_args()

    reference_doc = json.loads((DATA_DIR / "reference.json").read_text(encoding="utf-8"))
    instances = reference_doc["instances"]
    names = sorted(args.instances or instances)
    binary = args.binary or default_binary()
    commit = stamp.stamp(args.binary)
    context = {
        "time_limit": args.time_limit, "git_commit": commit,
        "machine": f"{platform.system()}-{platform.machine()}",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"),
        "solver_options": " ".join(args.solver_option),
    }
    print(f"solver   {binary}")
    print(f"commit   {commit}   machine {context['machine']}   time limit "
          f"{args.time_limit:g}s" + (f"   options {context['solver_options']}"
                                     if context["solver_options"] else ""))
    print()
    print(f"{'instance':<10}{'form':<5}{'status':<12}{'objective':>16}{'reference':>16}"
          f"{'root bound':>14}{'time':>9}  result")
    print("-" * 104)

    rows: list[dict] = []
    for name in names:
        entry = instances.get(name)
        if entry is None:
            print(f"{name:<10} not in data/pooling/reference.json")
            continue
        reference = float(entry["reference_objective"])
        for formulation in args.formulations:
            mps = DATA_DIR / f"{name}_{formulation}.mps"
            if not mps.exists():
                print(f"{name:<10}{formulation:<5}MISSING (run pooling_models.py)")
                continue
            flat = run_one(binary, mps, args.time_limit, not args.no_verify,
                           args.solver_option)
            row = csv_row(name, formulation, mps, entry, flat, reference, context)
            rows.append(row)
            ours = row["our_objective"]
            print(f"{name:<10}{formulation:<5}{row['status']:<12}"
                  f"{(f'{float(ours):.8g}' if ours else '-'):>16}{reference:>16.10g}"
                  f"{(f'{float(row['root_bound']):.6g}' if row['root_bound'] else '-'):>14}"
                  f"{row['wall_seconds']:>8.2f}s  "
                  f"{'PASS' if row['passed'] else 'FAIL: ' + row['failure_reason']}")
            if not row["matches_manifest"]:
                print(f"{'':15}the file on disk is not the one reference.json recorded")

    passes = sum(row["passed"] for row in rows)
    print("-" * 104)
    print(f"{passes}/{len(rows)} runs reached the published global optimum, claimed optimal, "
          "and were accepted by tools/verify_solution.py")
    grouped: dict[str, list[str]] = {}
    for row in rows:
        if row["failure_reason"]:
            grouped.setdefault(row["failure_reason"], []).append(
                f"{row['instance']}_{row['formulation']}")
    for reason, failed in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        print(f"  {reason} ({len(failed)}): {', '.join(failed)}")

    whole = set(names) == set(instances) and set(args.formulations) == set(FORMULATIONS)
    out_path = args.out or (RESULTS_DIR / f"pooling-{'' if whole else 'partial-'}{commit}.csv")
    out_path = (REPO_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")
    return 0 if rows and passes == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
