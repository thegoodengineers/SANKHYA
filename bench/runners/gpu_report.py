#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU vs GPU PDHG crossover benchmark (issue #19).

Generates synthetic LPs of increasing size from a known KKT optimum (same construction as
generate_large_lp.py) and solves each with both the CPU and GPU PDHG engines at two
tolerances.  The goal is the crossover point: the size at which the GPU backend overtakes
the CPU backend in wall-clock time, and by how much at the largest sizes.

Usage:
    python bench/runners/gpu_report.py --binary build_gpu/sankhya
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import platform
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import stamp  # noqa: E402  (#433: stamps from the binary)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"

CSV_COLUMNS = [
    "instance", "rows", "cols", "nnz", "algorithm", "tolerance",
    "status", "objective", "published_objective", "relative_error",
    "iterations", "seconds", "seconds_min", "seconds_max", "repeats",
    "wall_seconds", "reached_tolerance",
    "primal_residual", "dual_residual",
    "git_commit", "machine", "gpu", "timestamp_utc",
]

# THE TWO ENGINES ARE COMPARED ON THE SAME WORK. `algorithm=pdhg` alone hands a finished
# PDHG answer to the interior point for a polish (`pdhg_polish`, #229) on both paths, so
# the first measurement of this script compared PDHG-plus-polish against PDHG-plus-polish
# and every CPU row reached 1e-12 whatever the tolerance asked for. The polish is off here:
# what is timed is PDHG to the requested tolerance and nothing else, from the solver's own
# clock (`effort.solve_seconds` in the stats blob), so process start-up and the MPS read
# are not in the number. A first GPU solve also pays CUDA's context creation once per
# process, which is a property of the driver and not of the algorithm: every size's GPU
# measurements are preceded by a warm-up solve whose time is discarded, and the wall clock
# is still recorded beside the solver clock so the overhead stays visible.
# `pdhg_tolerance` on its own changes nothing looser than the project's absolute
# tolerances (#180): the loop runs on until the point meets them, which is why the first
# measurement had the same iteration count at 1e-4 and 1e-8. `pdhg_stop_at_request` makes
# the requested tolerance the stopping rule, so the two columns measure two different
# amounts of work.
COMMON_OPTIONS = ["log_to_console=false", "algorithm=pdhg", "pdhg_polish=false",
                  "pdhg_stop_at_request=true"]

# Problem sizes: (rows, cols, nnz_per_col)
SIZES = [
    (200,   200,   5),
    (500,   500,   5),
    (1000,  1000,  5),
    (2000,  2000,  5),
    (5000,  5000,  8),
    (10000, 10000, 8),
]

TOLERANCES = [1e-4, 1e-8]


def git_commit(binary=None) -> str:
    """The commit this CSV is stamped with: the binary's own, read from `sankhya
    version`, with `-dirty` from the tree; HEAD only when no binary answers (#433,
    bench/runners/stamp.py)."""
    return stamp.stamp(binary)

def gpu_description(binary: Path) -> str:
    """The device the binary sees, from `sankhya --version` (name, compute, VRAM), so the
    CSV says which card produced it instead of the doc generator guessing."""
    r = subprocess.run([str(binary), "--version"], capture_output=True, text=True,
                       check=False)
    match = re.search(r"GPU ([^)]*\))", r.stdout)
    return match.group(1).strip() if match else r.stdout.strip()


def as_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def generate_lp(rows: int, cols: int, nnz_per_col: int, seed: int, out: Path) -> float:
    """Generate a KKT LP and return its analytic optimum."""
    rng = random.Random(seed)
    # Choose a feasible primal point
    x_star = [rng.randint(1, 10) if rng.random() > 0.3 else 0 for _ in range(cols)]
    # Choose positive dual multipliers on a subset of rows
    y_star = [rng.randint(1, 5) if rng.random() > 0.5 else 0 for _ in range(rows)]
    # Build sparse A (each column has nnz_per_col entries)
    A: list[list[tuple[int, int]]] = [[] for _ in range(cols)]
    for j in range(cols):
        row_indices = rng.sample(range(rows), min(nnz_per_col, rows))
        for i in row_indices:
            A[j].append((i, rng.randint(1, 5)))
    # c_j = sum_i A_ij * y*_i + d_j, where d_j >= 0 and d_j=0 when x*_j > 0
    c = []
    for j in range(cols):
        dual_term = sum(a * y_star[i] for i, a in A[j])
        d_j = 0 if x_star[j] > 0 else rng.randint(0, 3)
        c.append(dual_term + d_j)
    # b_i = A_i x* (active rows are tight; inactive rows get a slack)
    b = [0.0] * rows
    for j in range(cols):
        for i, a in A[j]:
            b[i] += a * x_star[j]
    for i in range(rows):
        if y_star[i] == 0:
            b[i] -= rng.randint(1, 3)  # make row inactive (slack > 0)
    analytic_optimum = sum(c[j] * x_star[j] for j in range(cols))
    # Write MPS
    lines = ["NAME          KKT_LP", "ROWS", " N  obj"]
    for i in range(rows):
        lines.append(f" G  r{i}")
    lines.append("COLUMNS")
    for j in range(cols):
        if c[j] != 0:
            lines.append(f"    x{j}  obj  {c[j]}")
        for i, a in A[j]:
            lines.append(f"    x{j}  r{i}  {a}")
    lines.append("RHS")
    for i in range(rows):
        if b[i] != 0:
            lines.append(f"    rhs  r{i}  {b[i]}")
    lines.append("BOUNDS")
    for j in range(cols):
        ub = x_star[j] + rng.randint(0, 3) + 1
        lines.append(f" UP BND  x{j}  {ub}")
    lines.append("ENDATA")
    out.write_text("\n".join(lines) + "\n", encoding="ascii")
    return float(analytic_optimum)


def run_solve(binary: Path, mps: Path, algorithm: str, tolerance: float,
              time_limit: float) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        stats = Path(tmp) / "s.json"
        # Both CPU and GPU use algorithm=pdhg; GPU adds gpu=true.
        command = [str(binary), "solve", str(mps), "--stats", str(stats),
                   "--time-limit", str(time_limit)]
        for option in COMMON_OPTIONS + [f"pdhg_tolerance={tolerance:g}"]:
            command += ["--option", option]
        if algorithm == "pdhg-cuda":
            command += ["--option", "gpu=true"]
        started = time.perf_counter()
        subprocess.run(command, capture_output=True, text=True)
        seconds = time.perf_counter() - started
        if not stats.exists():
            return {"status": "no_output", "objective": None, "iterations": "",
                    "seconds": seconds, "wall": seconds, "algorithm_used": algorithm}
        blob = json.loads(stats.read_text())
        solver = as_number(blob.get("effort", {}).get("solve_seconds"))
        result = blob.get("result", {})
        msg = result.get("message", "")
        # Parse residuals from the solver's own message.
        # optimal:  "...absolute primal P, dual D, relative gap G"
        # feasible: "...(primal P vs ..., dual D vs ..., ...)"
        primal = re.search(r"absolute primal\s+([\d.e+\-]+)", msg) or \
                 re.search(r"\bprimal\s+([\d.e+\-]+)\s+vs", msg)
        dual = re.search(r"absolute primal[\d.e+\-\s]+,\s*dual\s+([\d.e+\-]+)", msg) or \
               re.search(r",\s*dual\s+([\d.e+\-]+)\s+vs", msg)
        return {
            "status": result.get("status", "unknown"),
            "objective": as_number(result.get("objective")),
            "iterations": blob.get("effort", {}).get("iterations", ""),
            "algorithm_used": result.get("algorithm", algorithm),
            "seconds": seconds if solver is None else solver,
            "wall": seconds,
            "primal_residual": primal.group(1) if primal else "",
            "dual_residual": dual.group(1) if dual else "",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5,
                        help="solves per (size, algorithm, tolerance) cell; median is reported")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    commit = git_commit(args.binary)
    machine = f"{platform.system()}-{platform.machine()}"
    gpu = gpu_description(args.binary)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    rows: list[dict] = []

    algorithms = ["pdhg-cpu", "pdhg-cuda"]

    print("GPU vs CPU PDHG crossover benchmark")
    print(f"binary: {args.binary}")
    print(f"commit: {commit}  machine: {machine}  gpu: {gpu}  repeats: {args.repeats}\n")
    print(f"{'size':>12}  {'alg':>12}  {'tol':>6}  {'status':>12}  {'median':>9}  "
          f"{'min':>9}  {'max':>9}  {'iters':>8}  {'speedup':>8}")
    print("-" * 100)

    cpu_medians: dict[tuple, float] = {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        for (nrows, ncols, nnz) in SIZES:
            mps_path = Path(tmp_dir) / f"kkt_{nrows}x{ncols}.mps"
            optimum = generate_lp(nrows, ncols, nnz, args.seed, mps_path)
            nnz_total = nrows * nnz  # approximate

            # Warm-up: the first CUDA call in a process creates the context (seconds on a
            # laptop card); that is paid once here and thrown away.
            run_solve(args.binary, mps_path, "pdhg-cuda", TOLERANCES[0], args.time_limit)
            for tol in TOLERANCES:
                for alg in algorithms:
                    # Run the cell args.repeats times; report median/min/max so run-to-run
                    # variance on a laptop GPU (clock boost, thermal state) is visible.
                    sample_results = [
                        run_solve(args.binary, mps_path, alg, tol, args.time_limit)
                        for _ in range(args.repeats)
                    ]
                    # Pick the result whose solve time is closest to the median for metadata.
                    sample_seconds = [r["seconds"] for r in sample_results]
                    med = statistics.median(sample_seconds)
                    result = min(sample_results,
                                 key=lambda r: abs(r["seconds"] - med))
                    sec_min = min(sample_seconds)
                    sec_max = max(sample_seconds)

                    obj = result["objective"]
                    err = (abs(obj - optimum) / max(1.0, abs(optimum))) if obj is not None else None
                    key = (nrows, tol, "pdhg-cpu")
                    if alg == "pdhg-cpu":
                        cpu_medians[key] = med
                    cpu_t = cpu_medians.get((nrows, tol, "pdhg-cpu"))
                    speedup = (cpu_t / med) if (alg != "pdhg-cpu" and cpu_t) else 1.0

                    rows.append({
                        "instance": f"kkt_{nrows}x{ncols}",
                        "rows": nrows, "cols": ncols, "nnz": nnz_total,
                        "algorithm": result.get("algorithm_used", alg),
                        "tolerance": f"{tol:.0e}",
                        "status": result["status"],
                        "objective": "" if obj is None else repr(obj),
                        "published_objective": repr(optimum),
                        "relative_error": "" if err is None else repr(err),
                        "iterations": result["iterations"],
                        "seconds": round(med, 6),
                        "seconds_min": round(sec_min, 6),
                        "seconds_max": round(sec_max, 6),
                        "repeats": args.repeats,
                        "wall_seconds": round(result["wall"], 6),
                        "reached_tolerance": int(result["status"] in ("optimal", "feasible")),
                        "primal_residual": result.get("primal_residual", ""),
                        "dual_residual": result.get("dual_residual", ""),
                        "git_commit": commit, "machine": machine, "gpu": gpu,
                        "timestamp_utc": timestamp,
                    })

                    speedup_str = f"{speedup:.2f}x" if alg != "pdhg-cpu" else "baseline"
                    print(f"{nrows:>5}x{ncols:<6}  {alg:>12}  {tol:>6.0e}  "
                          f"{result['status']:>12}  {med:>9.3f}  "
                          f"{sec_min:>9.3f}  {sec_max:>9.3f}  "
                          f"{str(result['iterations']):>8}  {speedup_str:>8}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or (RESULTS_DIR / f"gpu-{commit}.csv")
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
