#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""What the engine layer costs, beside what the engines cost (#297).

#297 asks that dispatch - classifying the model, choosing an engine, constructing it - be
negligible against the optimization it dispatches, and asks for the evidence. solve() times
both as profiler regions (#285): `classification` and `engine selection` (which includes the
one-time construction of the built-in registry, since every CLI run is a fresh process and
pays it), beside `presolve`, `engine`, `postsolve` and `verification`. This runner solves a
small set spanning LP, a generated sparse LP (--large-rows), MILP, QP and MIQP, and CPU PDHG - and CUDA PDHG when
the binary has it - with --option profile=basic, reads those regions from profile_out, and
writes one CSV row per run.

Every answer that claims a point also goes through tools/verify_solution.py, the independent
checker, so each engine's result is checked outside the process that produced it. A build
without the CUDA backend records no GPU row at all rather than a CPU number under a GPU name;
the summary says the GPU case was skipped and why.

Columns: setup_seconds is classification plus engine selection (the dispatch this issue is
about), solve_seconds the engine region, postprocess_seconds postsolve plus verification,
total_seconds the whole solve() region, wall_seconds the process including reading the file.

    python bench/runners/engine_dispatch.py
    python bench/runners/engine_dispatch.py --reps 5 --large-rows 5000 --time-limit 600
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
VERIFIER = REPO_ROOT / "tools" / "verify_solution.py"
GENERATOR = REPO_ROOT / "bench" / "runners" / "generate_large_lp.py"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from miplib import find_binary, git_commit  # noqa: E402  (same stamp and lookup as MIPLIB)

CSV_COLUMNS = [
    "case",
    "problem_class",
    "instance",
    "instance_sha256",
    "requested",
    "engine",
    "engine_rule",
    "rep",
    "status",
    "objective",
    "iterations",
    "nodes",
    "classification_seconds",
    "selection_seconds",
    "setup_seconds",
    "presolve_seconds",
    "solve_seconds",
    "postprocess_seconds",
    "total_seconds",
    "wall_seconds",
    "setup_share",
    "independently_verified",
    "git_commit",
    "machine",
    "timestamp_utc",
    "solver_options",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cases(large: Path, have_gpu: bool) -> list[tuple[str, str, Path, list[str]]]:
    """(case, class, instance, extra --option values). Every LP engine is named once on the
    same small model so each wrapper is on the record, then auto on each class."""
    afiro = REPO_ROOT / "data" / "netlib" / "afiro.mps"
    out = [("lp auto", "LP", afiro, [])]
    for name in ("simplex", "dual-simplex", "ipm", "pdhg"):
        out.append((f"lp {name}", "LP", afiro, [f"algorithm={name}"]))
    out += [
        ("lp medium auto", "LP", REPO_ROOT / "data" / "netlib" / "25fv47.mps", []),
        ("sparse lp auto", "LP", large, []),
        ("cpu pdhg", "LP", large, ["algorithm=pdhg"]),
        ("milp", "MILP", REPO_ROOT / "demo" / "blend_milp.mps", []),
        ("milp lot sizing", "MILP", REPO_ROOT / "data" / "casestudies" / "lot_sizing.mps", []),
        ("qp", "QP", REPO_ROOT / "demo" / "qp_blend.mps", []),
        ("miqp", "MIQP", REPO_ROOT / "demo" / "miqp_blend.mps", []),
    ]
    if have_gpu:
        out.append(("cuda pdhg", "LP", large, ["algorithm=pdhg", "gpu=true"]))
    return out


def engines_in(binary: Path) -> list[str]:
    listing = subprocess.run([str(binary), "engines", "--format", "json"],
                             capture_output=True, text=True, check=False)
    try:
        return [row["name"] for row in json.loads(listing.stdout)["engines"]]
    except (json.JSONDecodeError, KeyError):
        return []


def region(profile: dict, path: str) -> float:
    for row in profile.get("regions", []):
        if row.get("path") == path:
            return float(row.get("inclusive_seconds", 0.0))
    return 0.0


def run(binary: Path, instance: Path, options: list[str], time_limit: float,
        verify: bool) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        stats_path = Path(tmp) / "stats.json"
        profile_path = Path(tmp) / "profile.json"
        sol_path = Path(tmp) / "solution.sol"
        command = [str(binary), "solve", str(instance),
                   "--time-limit", str(time_limit),
                   "--stats", str(stats_path),
                   "--write-sol", str(sol_path),
                   "--option", "log_to_console=false",
                   "--option", "profile=basic",
                   "--option", f"profile_out={profile_path}"]
        for option in options:
            command += ["--option", option]
        started = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True)
        wall = time.perf_counter() - started
        if not stats_path.exists() or not profile_path.exists():
            return {"status": "crashed" if completed.returncode not in (0, 1) else "no_output",
                    "wall_seconds": wall, "verified": None}
        blob = json.loads(stats_path.read_text())
        profile = json.loads(profile_path.read_text())
        result = blob.get("result", {})
        effort = blob.get("effort", {})
        flat = {
            "status": result.get("status", "unknown"),
            "engine": result.get("algorithm", ""),
            "engine_rule": result.get("engine_rule", ""),
            "objective": result.get("objective"),
            "iterations": effort.get("iterations", ""),
            "nodes": effort.get("nodes", ""),
            "classification": region(profile, "solve/classification"),
            "selection": region(profile, "solve/engine selection"),
            "presolve": region(profile, "solve/presolve"),
            "engine_time": region(profile, "solve/engine"),
            "postprocess": (region(profile, "solve/postsolve")
                            + region(profile, "solve/verification")),
            "total": region(profile, "solve"),
            "wall_seconds": wall,
            "verified": None,
        }
        if verify and sol_path.exists() and flat["status"] in ("optimal", "feasible"):
            check = subprocess.run(
                [sys.executable, str(VERIFIER), str(instance), str(sol_path), "--quiet"],
                capture_output=True, text=True)
            flat["verified"] = check.returncode == 0
        return flat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--large-rows", type=int, default=2000,
                        help="rows and columns of the generated sparse LP (seed 42, 5 "
                             "nonzeros per column). 2,000 is the size every CPU route "
                             "finishes inside a few tens of seconds on the development box; at "
                             "5,000 the dual simplex does not finish in 120 s "
                             "(bench/results/scale-e134aeb.csv)")
    parser.add_argument("--cases", nargs="*", default=None,
                        help="run only the cases whose name contains one of these words")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--solver-option", action="append", default=[], metavar="KEY=VALUE",
                        help="passed to every solve as --option KEY=VALUE; repeatable, and "
                             "recorded in the CSV")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    binary = find_binary(args.binary)
    if binary is None:
        print("no solver binary; build first", file=sys.stderr)
        return 1

    have_gpu = "pdhg-gpu" in engines_in(binary)
    commit = git_commit()
    machine = f"{platform.system()}-{platform.machine()}"
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    solver_options = " ".join(args.solver_option)

    rows = []
    with tempfile.TemporaryDirectory() as scratch:
        large = Path(scratch) / f"random-sparse-{args.large_rows}.mps"
        subprocess.run([sys.executable, str(GENERATOR), "--rows", str(args.large_rows),
                        "--cols", str(args.large_rows), "--nnz-per-col", "5", "--seed", "42",
                        "--out", str(large)], check=True, capture_output=True)
        print(f"commit   {commit}   machine {machine}   reps {args.reps}"
              + (f"   options {solver_options}" if solver_options else ""))
        print()
        print(f"{'case':<22}{'engine':<22}{'status':<10}{'setup ms':>10}{'solve ms':>12}"
              f"{'total ms':>12}{'setup %':>9}  ver")
        print("-" * 102)
        for case, problem_class, instance, extra in cases(large, have_gpu):
            if args.cases and not any(word in case for word in args.cases):
                continue
            options = extra + args.solver_option
            shown_name = instance.name
            digest = sha256_file(instance)
            case_rows = []
            for rep in range(1, args.reps + 1):
                blob = run(binary, instance, options, args.time_limit, not args.no_verify)
                setup = blob.get("classification", 0.0) + blob.get("selection", 0.0)
                total = blob.get("total", 0.0)
                share = setup / total if total > 0 else None
                row = {
                    "case": case,
                    "problem_class": problem_class,
                    "instance": shown_name,
                    "instance_sha256": digest,
                    "requested": " ".join(extra),
                    "engine": blob.get("engine", ""),
                    "engine_rule": blob.get("engine_rule", ""),
                    "rep": rep,
                    "status": blob["status"],
                    "objective": "" if blob.get("objective") is None else repr(blob["objective"]),
                    "iterations": blob.get("iterations", ""),
                    "nodes": blob.get("nodes", ""),
                    "classification_seconds": f"{blob.get('classification', 0.0):.9f}",
                    "selection_seconds": f"{blob.get('selection', 0.0):.9f}",
                    "setup_seconds": f"{setup:.9f}",
                    "presolve_seconds": f"{blob.get('presolve', 0.0):.9f}",
                    "solve_seconds": f"{blob.get('engine_time', 0.0):.9f}",
                    "postprocess_seconds": f"{blob.get('postprocess', 0.0):.9f}",
                    "total_seconds": f"{total:.9f}",
                    "wall_seconds": f"{blob['wall_seconds']:.6f}",
                    "setup_share": "" if share is None else f"{share:.6f}",
                    "independently_verified": ("" if blob["verified"] is None
                                               else int(blob["verified"])),
                    "git_commit": commit,
                    "machine": machine,
                    "timestamp_utc": stamp,
                    "solver_options": solver_options,
                }
                case_rows.append(row)
            rows += case_rows
            # The median over the repetitions, so one cold-cache run does not speak for all.
            setup_ms = statistics.median(float(r["setup_seconds"]) for r in case_rows) * 1e3
            solve_ms = statistics.median(float(r["solve_seconds"]) for r in case_rows) * 1e3
            total_ms = statistics.median(float(r["total_seconds"]) for r in case_rows) * 1e3
            share_pct = 100.0 * setup_ms / total_ms if total_ms > 0 else float("nan")
            verified = {r["independently_verified"] for r in case_rows}
            ver = "yes" if verified == {1} else ("-" if verified == {""} else "NO")
            last = case_rows[-1]
            print(f"{case:<22}{last['engine']:<22}{last['status']:<10}{setup_ms:>10.3f}"
                  f"{solve_ms:>12.3f}{total_ms:>12.3f}{share_pct:>8.2f}%  {ver}", flush=True)

    print("-" * 102)
    if not have_gpu:
        print("cuda pdhg: skipped - this binary has no pdhg-gpu engine (SANKHYA_ENABLE_CUDA "
              "is off); no GPU number is recorded")
    unverified = sorted({r["case"] for r in rows if r["independently_verified"] == 0})
    if unverified:
        # Naming them is not optional: an answer the checker refused is not an answer.
        print(f"REFUSED by the independent verifier: {', '.join(unverified)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.out or (RESULTS_DIR / f"engine-dispatch-{commit}.csv")
    out_path = (REPO_ROOT / out_path).resolve()
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    try:
        shown = out_path.relative_to(REPO_ROOT)
    except ValueError:
        shown = out_path
    print(f"wrote {shown}")
    return 1 if unverified else 0


if __name__ == "__main__":
    sys.exit(main())
