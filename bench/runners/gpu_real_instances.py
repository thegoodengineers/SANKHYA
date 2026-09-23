#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GPU PDHG vs CPU PDHG on non-synthetic instances (issue #446).

Runs CPU and GPU PDHG on the refinery planning year (779,640 rows, generated
from bench/runners/generate_refinery_lp.py) and on the two Mittelmann LP
instances that CPU PDHG already solves: chromaticindex1024-7 (67,583 rows) and
brazil3 (14,646 rows).

All three are run at 1e-4 and 1e-8, PDHG alone (pdhg_polish=false), solver
clock, warm-up GPU solve per instance — the same protocol as gpu_report.py.

Usage:
    # Mittelmann instances must be fetched first:
    python bench/runners/fetch_mittelmann.py
    python bench/runners/gpu_real_instances.py --binary build_gpu/sankhya
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
MITTELMANN_DIR = REPO_ROOT / "data" / "mittelmann"

CSV_COLUMNS = [
    "instance", "rows", "cols", "nnz", "algorithm", "tolerance",
    "status", "objective", "iterations", "seconds", "wall_seconds",
    "reached_tolerance", "primal_residual", "dual_residual",
    "git_commit", "machine", "gpu", "timestamp_utc",
]

COMMON_OPTIONS = ["log_to_console=false", "algorithm=pdhg", "pdhg_polish=false",
                  "pdhg_stop_at_request=true"]

TOLERANCES = [1e-4, 1e-8]

# Mittelmann instances CPU PDHG already finishes (issue #446).
MITTELMANN_INSTANCES = ["chromaticindex1024-7", "brazil3"]


def git_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                       capture_output=True, text=True, check=False)
    commit = r.stdout.strip() or "unknown"
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                            cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if status.stdout.strip():
        commit += "-dirty"
    return commit


def gpu_description(binary: Path) -> str:
    r = subprocess.run([str(binary), "--version"], capture_output=True, text=True, check=False)
    match = re.search(r"GPU ([^)]*\))", r.stdout)
    return match.group(1).strip() if match else r.stdout.strip()


def as_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_solve(binary: Path, mps: Path, algorithm: str, tolerance: float,
              time_limit: float, extra_options: list[str] | None = None) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        stats = Path(tmp) / "s.json"
        command = [str(binary), "solve", str(mps), "--stats", str(stats),
                   "--time-limit", str(time_limit)]
        for option in COMMON_OPTIONS + [f"pdhg_tolerance={tolerance:g}"] + (extra_options or []):
            command += ["--option", option]
        if algorithm == "pdhg-cuda":
            command += ["--option", "gpu=true"]
        started = time.perf_counter()
        subprocess.run(command, capture_output=True, text=True)
        seconds = time.perf_counter() - started
        if not stats.exists():
            return {"status": "no_output", "objective": None, "iterations": "",
                    "seconds": seconds, "wall": seconds}
        blob = json.loads(stats.read_text())
        solver = as_number(blob.get("effort", {}).get("solve_seconds"))
        result = blob.get("result", {})
        msg = result.get("message", "")
        primal = re.search(r"absolute primal\s+([\d.e+\-]+)", msg) or \
                 re.search(r"\bprimal\s+([\d.e+\-]+)\s+vs", msg)
        dual = re.search(r"absolute primal[\d.e+\-\s]+,\s*dual\s+([\d.e+\-]+)", msg) or \
               re.search(r",\s*dual\s+([\d.e+\-]+)\s+vs", msg)
        return {
            "status": result.get("status", "unknown"),
            "objective": as_number(result.get("objective")),
            "iterations": blob.get("effort", {}).get("iterations", ""),
            "seconds": seconds if solver is None else solver,
            "wall": seconds,
            "primal_residual": primal.group(1) if primal else "",
            "dual_residual": dual.group(1) if dual else "",
        }


def mps_dimensions(mps: Path) -> tuple[int, int, int]:
    """Count rows, cols, nonzeros from a free-format MPS file (approx)."""
    rows = cols = nnz = 0
    section = ""
    open_fn = open
    if str(mps).endswith(".gz"):
        import gzip
        open_fn = gzip.open
    with open_fn(mps, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            tok = line.split()
            if not tok:
                continue
            if tok[0] in ("ROWS", "COLUMNS", "RHS", "BOUNDS", "RANGES", "ENDATA"):
                section = tok[0]
                continue
            if section == "ROWS" and tok[0] != "N":
                rows += 1
            elif section == "COLUMNS":
                cols += 1
                nnz += len(tok) // 2
    return rows, cols, nnz


def find_mittelmann_mps(name: str) -> Path | None:
    """Find a Mittelmann LP file by instance name (any .mps or .mps.gz)."""
    for ext in (".mps", ".mps.gz"):
        p = MITTELMANN_DIR / (name + ext)
        if p.exists():
            return p
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    commit = git_commit()
    machine = f"{platform.system()}-{platform.machine()}"
    gpu = gpu_description(args.binary)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    result_rows: list[dict] = []

    algorithms = ["pdhg-cpu", "pdhg-cuda"]

    print("GPU vs CPU PDHG on real instances")
    print(f"binary: {args.binary}")
    print(f"commit: {commit}  machine: {machine}  gpu: {gpu}\n")
    print(f"{'instance':>30}  {'alg':>12}  {'tol':>6}  {'status':>12}  {'seconds':>9}  {'speedup':>8}")
    print("-" * 90)

    cpu_times: dict[tuple, float] = {}

    # --- Refinery planning year (generate on the fly) ---
    with tempfile.TemporaryDirectory() as tmp_dir:
        refinery_mps = Path(tmp_dir) / "refinery_year.mps"
        print("generating refinery year LP (779,640 rows)...")
        gen = subprocess.run(
            [sys.executable, str(REPO_ROOT / "bench" / "runners" / "generate_refinery_lp.py"),
             "--periods", "8760", "--seed", "42", "--out", str(refinery_mps)],
            capture_output=True, text=True, check=False,
        )
        if gen.returncode != 0 or not refinery_mps.exists():
            print(f"  ERROR generating refinery LP: {gen.stderr[:200]}")
        else:
            # warm-up GPU
            run_solve(args.binary, refinery_mps, "pdhg-cuda", TOLERANCES[0], args.time_limit)
            r, c, nz = mps_dimensions(refinery_mps)
            for tol in TOLERANCES:
                for alg in algorithms:
                    result = run_solve(args.binary, refinery_mps, alg, tol, args.time_limit)
                    key = ("refinery_year", tol, "pdhg-cpu")
                    if alg == "pdhg-cpu":
                        cpu_times[key] = result["seconds"]
                    cpu_t = cpu_times.get(("refinery_year", tol, "pdhg-cpu"))
                    speedup = (cpu_t / result["seconds"]) if (alg != "pdhg-cpu" and cpu_t) else 1.0
                    speedup_str = f"{speedup:.2f}x" if alg != "pdhg-cpu" else "baseline"
                    print(f"{'refinery_year':>30}  {alg:>12}  {tol:>6.0e}  "
                          f"{result['status']:>12}  {result['seconds']:>9.3f}  {speedup_str:>8}")
                    result_rows.append({
                        "instance": "refinery_year",
                        "rows": r, "cols": c, "nnz": nz,
                        "algorithm": alg,
                        "tolerance": f"{tol:.0e}",
                        "status": result["status"],
                        "objective": "" if result["objective"] is None else repr(result["objective"]),
                        "iterations": result["iterations"],
                        "seconds": round(result["seconds"], 6),
                        "wall_seconds": round(result["wall"], 6),
                        "reached_tolerance": int(result["status"] in ("optimal", "feasible")),
                        "primal_residual": result.get("primal_residual", ""),
                        "dual_residual": result.get("dual_residual", ""),
                        "git_commit": commit, "machine": machine, "gpu": gpu,
                        "timestamp_utc": timestamp,
                    })

    # --- Mittelmann instances ---
    for name in MITTELMANN_INSTANCES:
        mps = find_mittelmann_mps(name)
        if mps is None:
            print(f"  SKIP {name}: not in {MITTELMANN_DIR} — run fetch_mittelmann.py first")
            continue
        # warm-up GPU
        run_solve(args.binary, mps, "pdhg-cuda", TOLERANCES[0], args.time_limit)
        r, c, nz = mps_dimensions(mps)
        for tol in TOLERANCES:
            for alg in algorithms:
                result = run_solve(args.binary, mps, alg, tol, args.time_limit)
                key = (name, tol, "pdhg-cpu")
                if alg == "pdhg-cpu":
                    cpu_times[key] = result["seconds"]
                cpu_t = cpu_times.get((name, tol, "pdhg-cpu"))
                speedup = (cpu_t / result["seconds"]) if (alg != "pdhg-cpu" and cpu_t) else 1.0
                speedup_str = f"{speedup:.2f}x" if alg != "pdhg-cpu" else "baseline"
                print(f"{name:>30}  {alg:>12}  {tol:>6.0e}  "
                      f"{result['status']:>12}  {result['seconds']:>9.3f}  {speedup_str:>8}")
                result_rows.append({
                    "instance": name,
                    "rows": r, "cols": c, "nnz": nz,
                    "algorithm": alg,
                    "tolerance": f"{tol:.0e}",
                    "status": result["status"],
                    "objective": "" if result["objective"] is None else repr(result["objective"]),
                    "iterations": result["iterations"],
                    "seconds": round(result["seconds"], 6),
                    "wall_seconds": round(result["wall"], 6),
                    "reached_tolerance": int(result["status"] in ("optimal", "feasible")),
                    "primal_residual": result.get("primal_residual", ""),
                    "dual_residual": result.get("dual_residual", ""),
                    "git_commit": commit, "machine": machine, "gpu": gpu,
                    "timestamp_utc": timestamp,
                })

    if not result_rows:
        print("no results — check that Mittelmann instances are fetched and GPU build works")
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or (RESULTS_DIR / f"gpu-real-{commit}.csv")
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(result_rows)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
