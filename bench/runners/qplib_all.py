#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Every QPLIB instance through the reader and the default solve dispatcher (#835).

qplib.py (#492) judges the convex continuous QPs. This one takes every instance
fetch_qplib_all.py converted - integer, nonconvex and quadratically constrained ones too - and
asks, one CSV row per instance, what the CLI does with it under its DEFAULTS (no engine
option, no `nonconvex=global`) and the --time-limit (60 s):

*   `read_ok`: the converted QPS file was read. A file the CLI refuses at read time is not
    read OK, whatever the reason; `refusal` then says whether the refusal was a deliberate one
    (quadratic constraints) or a reader error, which is a bug.
*   `engine`: the algorithm the dispatcher ran (the stats file's `algorithm`).
*   `refusal`: `nonconvex`, `quadratic_constraints`, `conversion` (qplib_format.py could not
    write the model), `reader_error`, `crashed`, `hung`, or empty. A nonconvex refusal is
    recognised by the solver's own message ("is not convex"), whichever status carries it.
*   `solved`: status optimal.
*   `matches_reference`: solved and |ours - ref| / max(1, |ref|) within `tolerance`: 1e-6
    (qplib.py's) for a continuous model, and for a model with integer columns kMipRelativeGap
    (1e-4, include/sankhya/tolerances.hpp), the gap the branch and bound stops at, so a
    correct answer can differ from the reference by that much. The reference is qplib.solu's
    best known point, not a proven optimum. `beats_reference` flags an objective BETTER than
    it by more than the tolerance, which for a model QPLIB has solved to global optimality
    means a wrong answer, and is reported as such, never credited.
*   `verified`: tools/verify_solution.py's verdict on the QPS file and the written .sol file,
    for any status that reports a point.

    python bench/runners/fetch_qplib_all.py
    python bench/runners/qplib_all.py --time-limit 60
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_suite  # noqa: E402  (machine_tag)
import maros_meszaros as mm  # noqa: E402  (default_binary, as_number, text, VERIFIER)
import stamp  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "qplib"
RESULTS_DIR = REPO_ROOT / "bench" / "results"
CONTINUOUS_TOLERANCE = 1e-6
INTEGER_TOLERANCE = 1e-4  # kMipRelativeGap
POINT_STATUSES = ("optimal", "feasible", "time_limit", "iteration_limit", "node_limit",
                  "interrupted", "locally_optimal")

CSV_COLUMNS = [
    "instance", "problem_type", "listed_convex", "instance_sha256", "qps_sha256", "rows",
    "cols", "nnz", "integer_columns", "read_ok", "engine", "refusal", "status", "message",
    "solved", "our_objective", "reference_objective", "abs_gap", "rel_gap", "tolerance",
    "matches_reference", "beats_reference", "verified", "verifier_message", "wall_seconds",
    "solver_seconds", "iterations", "nodes", "time_limit", "jobs", "git_commit", "machine",
    "timestamp_utc",
]


def classify(status: str, message: str) -> str:
    """The refusal a status and message amount to, or empty when there was none."""
    lowered = message.lower()
    if status == "read_error":
        return ("quadratic_constraints" if "quadratic constraints are not supported" in lowered
                else "reader_error")
    if status in ("crashed", "hung", "no_output"):
        return "crashed" if status == "no_output" else status
    if status in POINT_STATUSES:
        return ""
    if "is not convex" in lowered or "non-convex" in lowered or "nonconvex" in lowered:
        return "nonconvex"
    return ""


def judge(status: str, ours, reference, integer: bool, maximize: bool | None) -> dict:
    tolerance = INTEGER_TOLERANCE if integer else CONTINUOUS_TOLERANCE
    solved = status == "optimal"
    if ours is None or reference is None or status not in POINT_STATUSES:
        return {"solved": solved, "abs_gap": None, "rel_gap": None, "tolerance": tolerance,
                "matches_reference": None if reference is None else False,
                "beats_reference": None}
    gap = abs(ours - reference)
    rel = gap / max(1.0, abs(reference))
    better = (ours > reference) if maximize else (ours < reference)
    return {"solved": solved, "abs_gap": gap, "rel_gap": rel, "tolerance": tolerance,
            "matches_reference": bool(solved and rel <= tolerance),
            "beats_reference": bool(rel > tolerance and better)}


def solve(binary: Path, qps: Path, time_limit: float, verify: bool) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        stats_path, sol_path = Path(tmp) / "stats.json", Path(tmp) / "solution.sol"
        command = [str(binary), "solve", str(qps), "--stats", str(stats_path), "--write-sol",
                   str(sol_path), "--time-limit", str(time_limit),
                   "--option", "log_to_console=false"]
        started = time.perf_counter()
        try:
            done = subprocess.run(command, capture_output=True, text=True,
                                  timeout=time_limit + 120)
        except subprocess.TimeoutExpired:
            return {"status": "hung", "wall_seconds": time.perf_counter() - started,
                    "message": f"no exit within {time_limit + 120:g} s"}
        wall = time.perf_counter() - started
        if not stats_path.exists():
            stderr = done.stderr.strip()
            status = ("read_error" if "error:" in stderr and done.returncode >= 0
                      else "crashed" if done.returncode not in (0, 1) else "no_output")
            return {"status": status, "wall_seconds": wall,
                    "message": (stderr or f"exit code {done.returncode}")[:300]}
        blob = json.loads(stats_path.read_text(encoding="utf-8"))
        result, model = blob.get("result", {}), blob.get("model", {})
        effort = blob.get("effort", {})
        flat = {"status": result.get("status", "unknown"), "message": result.get("message", ""),
                "objective": mm.as_number(result.get("objective"))
                if result.get("status") in POINT_STATUSES else None,
                "engine": result.get("algorithm", ""), "rows": model.get("rows", ""),
                "cols": model.get("columns", ""), "nnz": model.get("nonzeros", ""),
                "iterations": effort.get("iterations", ""), "nodes": effort.get("nodes", ""),
                "solver_seconds": effort.get("solve_seconds", ""), "wall_seconds": wall,
                "verified": None, "verifier_message": ""}
        if verify and sol_path.exists() and flat["status"] in POINT_STATUSES:
            check = subprocess.run([sys.executable, str(mm.VERIFIER), str(qps), str(sol_path)],
                                   capture_output=True, text=True)
            flat["verified"] = check.returncode == 0
            if check.returncode != 0:
                failing = [line.strip() for line in check.stdout.splitlines() if "[FAIL]" in line]
                flat["verifier_message"] = ("; ".join(failing)
                                            or (check.stdout + check.stderr).strip())[:300]
        return flat


def row_for(name: str, entry: dict, binary: Path, args, stamp_fields: dict) -> dict:
    qps = DATA_DIR / f"{name}.qps"
    if "conversion_error" in entry:
        flat = {"status": "not_converted", "message": entry["conversion_error"],
                "wall_seconds": 0.0}
        refusal = "conversion"
    elif not qps.exists():
        flat = {"status": "missing", "message": "run bench/runners/fetch_qplib_all.py",
                "wall_seconds": 0.0}
        refusal = "conversion"
    else:
        flat = solve(binary, qps, args.time_limit, not args.no_verify)
        refusal = classify(flat["status"], str(flat.get("message", "")))
    integer = entry["problem_type"][1] != "C"
    verdict = judge(flat["status"], flat.get("objective"), entry.get("reference_objective"),
                    integer, entry.get("maximize"))
    row = {
        "instance": name, "problem_type": entry["problem_type"],
        "listed_convex": entry.get("listed_convex"),
        "instance_sha256": entry.get("qplib", {}).get("sha256", ""),
        "qps_sha256": entry.get("qps", {}).get("sha256", ""),
        "rows": flat.get("rows", ""), "cols": flat.get("cols", ""), "nnz": flat.get("nnz", ""),
        "integer_columns": entry.get("integer_columns", ""),
        "read_ok": flat["status"] not in ("read_error", "not_converted", "missing", "crashed",
                                          "hung", "no_output"),
        "engine": flat.get("engine", ""), "refusal": refusal, "status": flat["status"],
        "message": str(flat.get("message", ""))[:200], "our_objective": flat.get("objective"),
        "reference_objective": entry.get("reference_objective"), **verdict,
        "verified": flat.get("verified"), "verifier_message": flat.get("verifier_message", ""),
        "wall_seconds": round(flat.get("wall_seconds", 0.0), 6),
        "solver_seconds": flat.get("solver_seconds", ""),
        "iterations": flat.get("iterations", ""), "nodes": flat.get("nodes", ""),
        "time_limit": args.time_limit, "jobs": args.jobs, **stamp_fields,
    }
    return {key: mm.text(row[key]) for key in CSV_COLUMNS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--instances", nargs="*")
    parser.add_argument("--jobs", type=int, default=1,
                        help="instances solved at once; recorded in every row")
    parser.add_argument("--machine-kind", default="cloud container")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    manifest_path = DATA_DIR / "all.json"
    if not manifest_path.exists():
        raise SystemExit("no data/qplib/all.json; run bench/runners/fetch_qplib_all.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    instances = manifest["instances"]
    names = args.instances or sorted(instances)
    binary = args.binary or mm.default_binary()
    stamp_fields = {
        "git_commit": stamp.stamp(binary), "machine": compare_suite.machine_tag(args.machine_kind),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    print(f"solver {binary}  commit {stamp_fields['git_commit']}\nmachine {stamp_fields['machine']}"
          f"\n{len(names)} instances at {args.time_limit:g} s, {args.jobs} at a time")

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(row_for, n, instances[n], binary, args, stamp_fields): n
                   for n in names}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"{row['instance']:<13}{row['problem_type']:<5}{row['status']:<18}"
                  f"{row['engine']:<22}{row['refusal']:<22}{row['our_objective']:>24} "
                  f"{row['reference_objective']:>22} match={row['matches_reference'] or '-'} "
                  f"ver={row['verified'] or '-'} {float(row['wall_seconds']):.1f}s", flush=True)
    rows.sort(key=lambda r: r["instance"])

    partial = bool(args.instances) or manifest.get("partial")
    default_name = f"qplib-all{'-partial' if partial else ''}-{stamp_fields['git_commit']}.csv"
    out_path = (REPO_ROOT / args.out).resolve() if args.out else RESULTS_DIR / default_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    count = lambda key, value="1": sum(r[key] == value for r in rows)  # noqa: E731
    print(f"{len(rows)} instances: read {count('read_ok')}, solved {count('solved')}, matched "
          f"{count('matches_reference')}, verified {count('verified')}, beats the reference "
          f"{count('beats_reference')}, verifier rejected {count('verified', '0')}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
