#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run SANKHYA over the Maros-Meszaros convex QP set and emit the evidence CSV (#491).

The QP counterpart of netlib.py, and it keeps netlib.py's columns: instance, sha256 of the
instance file, our objective, the PUBLISHED reference objective, absolute and relative gap,
status, wall time, iterations, git commit, machine, and the independent verifier's verdict.
What it adds is the QP benchmark methodology's three optimality measures (qp_residuals.py):
primal residual, dual residual and duality gap, each recomputed from the model and the .sol
file, absolute and relative, and success judged at the two levels the published
Maros-Meszaros report uses, 1e-6 and 1e-9.

A row reports three separate things, because each can fail while the others hold:

*   `matches_reference`: the status is optimal and the objective is within 1e-6 relative of
    the readme's OPT. Those values are BPMPD's, printed to eight significant digits, so this
    is as tight as the published number supports - not a test at 1e-9.
*   `success_rel_1e-6`, `success_rel_1e-9`: the status is optimal and all three RELATIVE
    measures are within the level (#491). `success_abs_*` apply the same test to the
    absolute measures, which is qpbenchmark's own criterion and the one to use when
    comparing with its published tables.
*   `independently_verified`: tools/verify_solution.py's verdict, which re-derives the KKT
    conditions at the project's own tolerances (1e-7 primal and dual).

`passed` is the conjunction the project stands behind: optimal, verified, matching the
reference, and successful at 1e-6 relative.

The solver runs at its defaults (the QP engine stops at qp_tolerance = 1e-8 on its own
absolute residuals) and both levels are judged from that one run. qpbenchmark instead
reconfigures each solver per level; `--solver-option qp_tolerance=1e-10` does the same here,
and is recorded in the CSV so such a run is never mistaken for the default one.

Output: bench/results/maros-meszaros-<commit>.csv for a run over every instance in the
manifest; any narrower selection writes maros-meszaros-partial-<commit>.csv, which
docs/BENCHMARKS.md never reads. The published report's per-instance limit is 1000 s, and
that is the default here.

    python bench/runners/fetch_maros_meszaros.py
    python bench/runners/maros_meszaros.py
    python bench/runners/maros_meszaros.py --smallest 30 --time-limit 30 --out /tmp/mm.csv
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qp_residuals  # noqa: E402
import stamp  # noqa: E402  (#433: stamps from the binary)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "maros-meszaros"
RESULTS_DIR = REPO_ROOT / "bench" / "results"
VERIFIER = REPO_ROOT / "tools" / "verify_solution.py"

REFERENCE_TOLERANCE = 1e-6
DEFAULT_TIME_LIMIT = 1000.0  # the published Maros-Meszaros report's runtime limit
SHIFT_SECONDS = 10.0  # the published report's shift for its shifted geometric means

CSV_COLUMNS = [
    "instance", "instance_sha256", "rows", "columns", "nonzeros", "status", "message",
    "our_objective", "reference_objective", "absolute_gap", "relative_gap",
    "matches_reference",
    "primal_residual", "dual_residual", "duality_gap",
    "primal_residual_rel", "dual_residual_rel", "duality_gap_rel",
    "success_rel_1e-6", "success_rel_1e-9", "success_abs_1e-6", "success_abs_1e-9",
    "independently_verified", "verifier_message", "passed",
    "wall_seconds", "solver_seconds", "iterations", "algorithm", "time_limit",
    "git_commit", "machine", "timestamp_utc", "solver_options",
]

# Statuses whose .sol carries a point the residuals can be measured at.
STATUSES_WITH_A_POINT = ("optimal", "feasible", "iteration_limit", "time_limit")


def default_binary() -> Path:
    sys.path.insert(0, str(REPO_ROOT / "bindings" / "python"))
    import sankhya
    try:
        return sankhya.locate_executable()
    except sankhya.SankhyaError as error:
        raise SystemExit(str(error))


def as_number(value) -> float | None:
    """The stats writer spells non-finite values as strings; float() reads all of them."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def select(instances: dict, names: list[str] | None, smallest: int | None) -> list[str]:
    """Explicit names win; otherwise the `smallest` instances by QPS file size, ties broken
    by name so the subset is the same on every machine; otherwise every instance."""
    if names:
        unknown = [name for name in names if name not in instances]
        if unknown:
            raise SystemExit(f"not in the manifest: {', '.join(unknown)}")
        return list(names)
    ordered = sorted(instances, key=lambda name: (instances[name]["qps_bytes"], name))
    return ordered[:smallest] if smallest else sorted(instances)


def judge(status: str, ours: float | None, reference: float,
          residuals: qp_residuals.Residuals | None) -> dict:
    """Every per-row verdict, from the status, the objective and the residuals alone."""
    optimal = status == "optimal"
    gap = None if ours is None else abs(ours - reference) / max(1.0, abs(reference))
    verdict = {
        "absolute_gap": None if ours is None else abs(ours - reference),
        "relative_gap": gap,
        "matches_reference": bool(optimal and gap is not None and gap <= REFERENCE_TOLERANCE),
    }
    for level, label in zip(qp_residuals.LEVELS, ("1e-6", "1e-9")):
        for relative, kind in ((True, "rel"), (False, "abs")):
            verdict[f"success_{kind}_{label}"] = bool(
                optimal and residuals is not None and residuals.meets(level, relative))
    return verdict


def run_one(binary: Path, qps: Path, time_limit: float, verify: bool,
            solver_options: list[str]) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        stats_path = Path(tmp) / "stats.json"
        sol_path = Path(tmp) / "solution.sol"
        command = [str(binary), "solve", str(qps), "--stats", str(stats_path),
                   "--write-sol", str(sol_path), "--time-limit", str(time_limit),
                   "--option", "log_to_console=false"]
        for option in solver_options:
            command += ["--option", option]
        started = time.perf_counter()
        # The solver enforces its own limit; the margin here only catches a hung process.
        try:
            completed = subprocess.run(command, capture_output=True, text=True,
                                       timeout=time_limit + 120)
        except subprocess.TimeoutExpired:
            return {"status": "hung", "wall_seconds": time.perf_counter() - started,
                    "message": f"no exit within {time_limit + 120:g} s"}
        wall = time.perf_counter() - started
        if not stats_path.exists():
            stderr = completed.stderr.strip()
            status = ("read_error" if "error:" in stderr
                      else "crashed" if completed.returncode not in (0, 1) else "no_output")
            return {"status": status, "wall_seconds": wall, "message": stderr[:300]}

        blob = json.loads(stats_path.read_text(encoding="utf-8"))
        result, model, effort = blob.get("result", {}), blob.get("model", {}), blob.get("effort", {})
        flat = {
            "status": result.get("status", "unknown"), "message": result.get("message", ""),
            "objective": as_number(result.get("objective")), "rows": model.get("rows", ""),
            "columns": model.get("columns", ""), "nonzeros": model.get("nonzeros", ""),
            "iterations": effort.get("iterations", ""),
            "solver_seconds": effort.get("solve_seconds", ""),
            "algorithm": result.get("algorithm", ""), "wall_seconds": wall,
            "verified": None, "verifier_message": "", "residuals": None,
        }
        if sol_path.exists() and flat["status"] in STATUSES_WITH_A_POINT:
            try:
                flat["residuals"] = qp_residuals.compute_files(qps, sol_path)
            except (OSError, ValueError) as error:
                flat["message"] = (flat["message"] + f"; residuals not computed: {error}")[:300]
        if verify and sol_path.exists() and flat["status"] in ("optimal", "feasible"):
            check = subprocess.run([sys.executable, str(VERIFIER), str(qps), str(sol_path)],
                                   capture_output=True, text=True)
            flat["verified"] = check.returncode == 0
            if check.returncode != 0:
                failing = [line.strip() for line in check.stdout.splitlines() if "[FAIL]" in line]
                flat["verifier_message"] = ("; ".join(failing)
                                            or (check.stdout + check.stderr).strip())[:300]
        return flat


def shifted_geometric_mean(values: list[float], shift: float = SHIFT_SECONDS) -> float:
    if not values:
        return float("nan")
    return math.exp(sum(math.log(max(v, 0.0) + shift) for v in values) / len(values)) - shift


def text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        return repr(value)
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT)
    parser.add_argument("--instances", nargs="*")
    parser.add_argument("--smallest", type=int, default=None,
                        help="run only the N smallest instances by QPS file size")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--solver-option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--out", type=Path, default=None,
                        help="destination CSV; relative paths resolve against the repo root")
    args = parser.parse_args()

    manifest_path = DATA_DIR / "reference.json"
    if not manifest_path.exists():
        raise SystemExit("no data/maros-meszaros/reference.json; "
                         "run bench/runners/fetch_maros_meszaros.py")
    instances = json.loads(manifest_path.read_text(encoding="utf-8"))["instances"]
    names = select(instances, args.instances, args.smallest)
    binary = args.binary or default_binary()
    commit, machine = stamp.stamp(args.binary), f"{platform.system()}-{platform.machine()}"
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    solver_options = " ".join(args.solver_option)

    print(f"solver   {binary}\ncommit   {commit}   machine {machine}   "
          f"time limit {args.time_limit:g}s   {len(names)} of {len(instances)} instances"
          + (f"   options {solver_options}" if solver_options else ""))
    print(f"\n{'instance':<10}{'status':<16}{'our objective':>20}{'reference':>16}{'rel gap':>9}"
          f"{'prim rel':>10}{'dual rel':>10}{'gap rel':>10}{'time':>8}  ver  1e-6 1e-9")
    print("-" * 124)
    rows: list[dict] = []
    for name in names:
        entry = instances[name]
        qps = DATA_DIR / f"{name}.qps"
        if not qps.exists():
            print(f"{name:<10}MISSING (run fetch_maros_meszaros.py)")
            continue
        flat = run_one(binary, qps, args.time_limit, not args.no_verify, args.solver_option)
        residuals = flat.get("residuals")
        ours = flat.get("objective")
        verdict = judge(flat["status"], ours, entry["reference_objective"], residuals)
        verified = flat.get("verified")
        passed = bool(verdict["matches_reference"] and verdict["success_rel_1e-6"]
                      and verified is True)
        row = {
            "instance": name, "instance_sha256": entry["qps_sha256"],
            "rows": flat.get("rows", ""), "columns": flat.get("columns", ""),
            "nonzeros": flat.get("nonzeros", ""), "status": flat["status"],
            "message": str(flat.get("message", ""))[:200], "our_objective": ours,
            "reference_objective": entry["reference_objective"],
            **verdict,
            "primal_residual": residuals.primal if residuals else None,
            "dual_residual": residuals.dual if residuals else None,
            "duality_gap": residuals.gap if residuals else None,
            "primal_residual_rel": residuals.primal_rel if residuals else None,
            "dual_residual_rel": residuals.dual_rel if residuals else None,
            "duality_gap_rel": residuals.gap_rel if residuals else None,
            "independently_verified": verified,
            "verifier_message": flat.get("verifier_message", ""), "passed": passed,
            "wall_seconds": round(flat.get("wall_seconds", 0.0), 6),
            "solver_seconds": flat.get("solver_seconds", ""),
            "iterations": flat.get("iterations", ""), "algorithm": flat.get("algorithm", ""),
            "time_limit": args.time_limit, "git_commit": commit, "machine": machine,
            "timestamp_utc": timestamp, "solver_options": solver_options,
        }
        rows.append({key: text(row[key]) for key in CSV_COLUMNS})

        def cell(value, width):
            return f"{'-':>{width}}" if value is None else f"{value:>{width}.1e}"
        ours_text = "-" if ours is None else f"{ours:.10e}"
        print(f"{name:<10}{flat['status']:<16}{ours_text:>20}{entry['reference_objective']:>16.8e}"
              f"{cell(verdict['relative_gap'], 9)}"
              f"{cell(residuals.primal_rel if residuals else None, 10)}"
              f"{cell(residuals.dual_rel if residuals else None, 10)}"
              f"{cell(residuals.gap_rel if residuals else None, 10)}"
              f"{flat.get('wall_seconds', 0.0):>7.1f}s  "
              f"{ {True: 'yes', False: 'NO ', None: ' - '}[verified]}  "
              f"{'yes ' if verdict['success_rel_1e-6'] else 'no  '} "
              f"{'yes' if verdict['success_rel_1e-9'] else 'no'}")
        if flat.get("verifier_message"):
            print(f"{'':10}verifier: {flat['verifier_message'][:100]}")

    total = len(rows)
    count = lambda key: sum(row[key] == "1" for row in rows)  # noqa: E731
    print("-" * 124)
    print(f"optimal {sum(r['status'] == 'optimal' for r in rows)}/{total}   "
          f"matches reference (1e-6 rel) {count('matches_reference')}/{total}   "
          f"verified {count('independently_verified')}/{total}")
    print(f"success at 1e-6: {count('success_rel_1e-6')}/{total} relative, "
          f"{count('success_abs_1e-6')}/{total} absolute   "
          f"success at 1e-9: {count('success_rel_1e-9')}/{total} relative, "
          f"{count('success_abs_1e-9')}/{total} absolute")
    # The published report charges a failure the full time limit; so does this.
    charged = [float(r["solver_seconds"]) if r["success_rel_1e-6"] == "1" and r["solver_seconds"]
               else args.time_limit for r in rows]
    if charged:
        print(f"shifted geometric mean solver time (shift {SHIFT_SECONDS:g} s, a failure at "
              f"1e-6 charged the {args.time_limit:g} s limit): "
              f"{shifted_geometric_mean(charged):.2f} s")
    failed = [r["instance"] for r in rows if r["passed"] != "1"]
    print(f"passed {total - len(failed)}/{total}"
          + (f"; failed: {', '.join(failed)}" if failed else ""))

    full_set = not args.instances and not args.smallest
    default_name = f"maros-meszaros-{commit}.csv" if full_set else \
        f"maros-meszaros-partial-{commit}.csv"
    out_path = (REPO_ROOT / args.out).resolve() if args.out else RESULTS_DIR / default_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")
    return 0 if rows and not failed else 1


if __name__ == "__main__":
    sys.exit(main())
