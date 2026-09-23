#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU PDHG thread scaling with the row-parallel A x (#487).

A GPU speed-up over a single-threaded CPU loop is the number most easily dismissed. This
measures what the CPU side does with its cores: PDHG alone, `pdhg_parallel_spmv=true`, a
FIXED iteration count per instance so every thread count does the same arithmetic, at 1, 2,
4 and 8 threads, on the synthetic ladder gpu_report.py uses and on the refinery year.

    python bench/runners/pdhg_threads.py --binary build/sankhya --threads 1,2,4,8

Output: bench/results/pdhg-threads-<sha>.csv (the sha from the binary, #433), one row per
instance and thread count: rows, cols, nnz, threads, iterations (the same in every row of an
instance), objective (identical across thread counts by construction: the test in
tests/unit/test_pdhg_parallel_spmv.cpp holds it to the bit), solver seconds, wall seconds,
speedup over one thread, and the machine. `--serial` adds a row per instance with the option
off at one thread, the product PDHG used before #487, for the cost of the transpose.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stamp  # noqa: E402
from gpu_report import SIZES, generate_lp  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"

COLUMNS = ["instance", "rows", "cols", "nnz", "threads", "parallel_spmv", "iterations", "status",
           "objective", "solver_seconds", "wall_seconds", "speedup_vs_one_thread", "git_commit",
           "machine", "timestamp_utc"]

# Iterations per instance: enough that the per-iteration cost dominates the transpose and
# the start-up, few enough that the whole ladder runs in minutes.
DEFAULT_ITERATIONS = 2000


def solve(binary: Path, mps: Path, threads: int, parallel: bool, iterations: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        stats = Path(tmp) / "s.json"
        command = [str(binary), "solve", str(mps), "--stats", str(stats)]
        for option in ("log_to_console=false", "algorithm=pdhg", "pdhg_polish=false",
                       "pdhg_restart=true", "presolve=false", f"threads={threads}",
                       f"pdhg_parallel_spmv={'true' if parallel else 'false'}",
                       f"iteration_limit={iterations}"):
            command += ["--option", option]
        started = time.perf_counter()
        subprocess.run(command, capture_output=True, text=True)
        wall = time.perf_counter() - started
        if not stats.exists():
            return {"status": "no_output", "objective": "", "iterations": "", "seconds": wall,
                    "wall": wall}
        blob = json.loads(stats.read_text())
        return {
            "status": blob.get("result", {}).get("status", "unknown"),
            "objective": blob.get("result", {}).get("objective", ""),
            "iterations": blob.get("effort", {}).get("iterations", ""),
            "seconds": float(blob.get("effort", {}).get("solve_seconds", wall)),
            "wall": wall,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--threads", default="1,2,4,8")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--serial", action="store_true",
                        help="also the serial product at one thread, per instance")
    parser.add_argument("--no-refinery", action="store_true",
                        help="skip the 779,640-row refinery year")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    threads = [int(t) for t in args.threads.split(",")]

    commit = stamp.stamp(args.binary)
    machine = f"{platform.system()}-{platform.machine()}"
    stamp_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    out = args.out or RESULTS_DIR / f"pdhg-threads-{commit}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"solver {commit}; threads {threads}; {args.iterations} iterations per solve")
    print(f"{'instance':>18} {'threads':>7} {'A x':>8} {'iters':>6} {'solver s':>9} {'speedup':>8}")
    with tempfile.TemporaryDirectory() as tmp_dir, out.open("w", newline="",
                                                             encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        instances: list[tuple[str, Path, int, int, int]] = []
        for nrows, ncols, nnz in SIZES:
            if nrows < 1000:
                continue  # below the sizes where a thread has anything to do
            mps = Path(tmp_dir) / f"kkt_{nrows}x{ncols}.mps"
            generate_lp(nrows, ncols, nnz, args.seed, mps)
            instances.append((f"kkt_{nrows}x{ncols}", mps, nrows, ncols, nrows * nnz))
        if not args.no_refinery:
            refinery = Path(tmp_dir) / "refinery_year.mps"
            print("generating refinery year LP (779,640 rows)...")
            subprocess.run([sys.executable, str(REPO_ROOT / "bench/runners/generate_refinery_lp.py"),
                            "--periods", "8760", "--seed", "42", "--out", str(refinery)],
                           check=True, capture_output=True)
            instances.append(("refinery_year", refinery, 779640, 0, 0))

        for name, mps, nrows, ncols, nnz in instances:
            base = None
            plan = ([(1, False)] if args.serial else []) + [(t, True) for t in threads]
            for t, parallel in plan:
                r = solve(args.binary, mps, t, parallel, args.iterations)
                if parallel and t == 1:
                    base = r["seconds"]
                speedup = "" if (base is None or not parallel) else f"{base / r['seconds']:.3f}"
                row = {
                    "instance": name, "rows": nrows, "cols": ncols, "nnz": nnz, "threads": t,
                    "parallel_spmv": int(parallel), "iterations": r["iterations"],
                    "status": r["status"], "objective": r["objective"],
                    "solver_seconds": f"{r['seconds']:.6f}", "wall_seconds": f"{r['wall']:.6f}",
                    "speedup_vs_one_thread": speedup, "git_commit": commit, "machine": machine,
                    "timestamp_utc": stamp_utc,
                }
                writer.writerow(row)
                handle.flush()
                print(f"{name:>18} {t:>7} {'parallel' if parallel else 'serial':>8} "
                      f"{str(r['iterations']):>6} {r['seconds']:>9.3f} {speedup:>8}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
