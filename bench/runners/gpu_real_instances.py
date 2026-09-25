#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GPU PDHG vs CPU PDHG on non-synthetic instances (issue #446).

Runs CPU and GPU PDHG on the refinery planning year (779,640 rows, generated
from bench/runners/generate_refinery_lp.py) and on the two Mittelmann LP
instances that CPU PDHG already solves: chromaticindex1024-7 (67,583 rows) and
brazil3 (14,646 rows).

All three are run at 1e-4 and 1e-8, PDHG alone (pdhg_polish=false), solver
clock, warm-up GPU solve per instance — the same protocol as gpu_report.py.

Three arms per instance and tolerance (#488, bench/runners/gpu_arms.py): the CPU at one
thread with the serial A x (`cpu-1t`, the default configuration), the CPU at
`--cpu-threads N` with `pdhg_parallel_spmv=true` (`cpu-<N>t`), and the card (`gpu`). The
speedup is printed against both CPU arms. Each instance gets one reference objective - the
generator's analytic optimum for the refinery year, HiGHS as a separate process for the
Mittelmann files - and every row carries its absolute and relative gap to it.

Usage:
    # Mittelmann instances must be fetched first:
    python bench/runners/fetch_mittelmann.py
    python bench/runners/gpu_real_instances.py --binary build_gpu/sankhya --card l4
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_arms  # noqa: E402  (#488: the CPU arms, the reference and the gap arithmetic)
import stamp  # noqa: E402  (#433, #589: the CSV names the commit the BINARY was built from)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
MITTELMANN_DIR = REPO_ROOT / "data" / "mittelmann"

CSV_COLUMNS = [
    "instance", "instance_sha256", "rows", "cols", "nnz", "algorithm", "arm", "cpu_threads",
    "parallel_spmv", "tolerance", "status", "objective", "reference_objective",
    "reference_source", "abs_gap", "rel_gap", "iterations", "seconds", "wall_seconds",
    "reached_tolerance", "primal_residual", "dual_residual",
    "kkt_1e4_seconds", "kkt_1e6_seconds", "kkt_1e8_seconds",
    "git_commit", "machine", "gpu", "timestamp_utc", "solver_options",
]

COMMON_OPTIONS = ["log_to_console=false", "algorithm=pdhg", "pdhg_polish=false",
                  "pdhg_stop_at_request=true"]

TOLERANCES = [1e-4, 1e-8]

# Mittelmann instances CPU PDHG already finishes (issue #446).
MITTELMANN_INSTANCES = ["chromaticindex1024-7", "brazil3"]


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
        primal, dual = gpu_arms.residuals(blob)
        return {
            "status": result.get("status", "unknown"),
            "objective": as_number(result.get("objective")),
            "iterations": blob.get("effort", {}).get("iterations", ""),
            "seconds": seconds if solver is None else solver,
            "wall": seconds,
            "primal_residual": primal,
            "dual_residual": dual,
            "kkt_1e4_seconds": blob.get("effort", {}).get("kkt_1e4_seconds", ""),
            "kkt_1e6_seconds": blob.get("effort", {}).get("kkt_1e6_seconds", ""),
            "kkt_1e8_seconds": blob.get("effort", {}).get("kkt_1e8_seconds", ""),
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


def build_row(name: str, dims: tuple[int, int, int], digest: str, arm: tuple, tol: float,
              result: dict, reference: float | None, reference_source: str,
              stamp_fields: dict) -> dict:
    """One CSV row. Pure, so test_gpu_runners.py can pin it with a synthetic result."""
    r, c, nz = dims
    row = {
        "instance": name,
        "rows": r, "cols": c, "nnz": nz,
        "algorithm": arm[1],
        "tolerance": f"{tol:.0e}",
        "status": result["status"],
        "objective": "" if result["objective"] is None else repr(result["objective"]),
        "iterations": result["iterations"],
        "seconds": round(result["seconds"], 6),
        "wall_seconds": round(result["wall"], 6),
        "reached_tolerance": int(result["status"] in ("optimal", "feasible")),
        "primal_residual": result.get("primal_residual", ""),
        "dual_residual": result.get("dual_residual", ""),
        "kkt_1e4_seconds": result.get("kkt_1e4_seconds", ""),
        "kkt_1e6_seconds": result.get("kkt_1e6_seconds", ""),
        "kkt_1e8_seconds": result.get("kkt_1e8_seconds", ""),
    }
    row.update(gpu_arms.fairness_cells(arm, digest, result["objective"], reference,
                                       reference_source))
    row.update(stamp_fields)
    return row


def measure(name: str, mps: Path, args, stamp_fields: dict) -> list[dict]:
    """Every arm at every tolerance on one instance, with one reference for all of them."""
    digest = gpu_arms.sha256_file(mps)
    dims = mps_dimensions(mps)
    reference, source, note = gpu_arms.reference_objective(
        mps, args.reference_time_limit, use_highs=not args.no_reference)
    print(f"{name}: sha256 {digest[:16]}..., reference {reference!r} ({source}; {note})")
    # warm-up GPU
    run_solve(args.binary, mps, "pdhg-cuda", TOLERANCES[0], args.time_limit, args.solver_option)
    rows: list[dict] = []
    for tol in TOLERANCES:
        cpu_seconds: dict[str, float] = {}
        for arm in gpu_arms.all_arms(args.cpu_threads):
            label, algorithm, arm_options = arm[0], arm[1], arm[2]
            result = run_solve(args.binary, mps, algorithm, tol, args.time_limit,
                               arm_options + args.solver_option)
            row = build_row(name, dims, digest, arm, tol, result, reference, source,
                            stamp_fields)
            if algorithm == "pdhg-cpu":
                cpu_seconds[label] = result["seconds"]
                versus = "baseline"
            else:
                versus = "  ".join(f"{gpu_arms.speedup(s, result['seconds'])} vs {cpu}"
                                   for cpu, s in cpu_seconds.items())
            print(f"{name:>24}  {label:>8}  {tol:>6.0e}  {result['status']:>15}  "
                  f"{result['seconds']:>9.3f}  {row['rel_gap'] or '-':>10}  {versus}")
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--card", default="",
                        help="short card name (l4, a100, ...): the output becomes "
                             "gpu-real-<card>-<commit>.csv, which the doc's section 1g.3 reads")
    parser.add_argument("--cpu-threads", type=int, default=16,
                        help="threads of the parallel CPU arm, run with pdhg_parallel_spmv=true "
                             "(#488); the serial one-thread arm always runs as well. 1 drops "
                             "the parallel arm.")
    parser.add_argument("--reference-time-limit", type=float, default=600.0,
                        help="HiGHS's own limit for the reference objective, once per instance")
    parser.add_argument("--no-reference", action="store_true",
                        help="skip HiGHS; the reference is blank unless the file states one")
    parser.add_argument("--solver-option", action="append", default=[], metavar="KEY=VALUE",
                        help="passed to every solve, CPU and GPU arm alike, and recorded in the "
                             "solver_options column: the A/B of an option (gpu_on_device_loop, "
                             "pdhg_two_matvec, ...). Such a run is written to gpu-ab-<commit>-<tag>.csv, "
                             "which the benchmark doc's gpu-real-* tier never reads.")
    parser.add_argument("--skip-refinery", action="store_true",
                        help="only the Mittelmann instances (the 779,640-row refinery year is the "
                             "slow part)")
    args = parser.parse_args()
    solver_options = " ".join(args.solver_option)

    commit = stamp.stamp(args.binary)
    machine = f"{platform.system()}-{platform.machine()}"
    gpu = gpu_description(args.binary)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    stamp_fields = {"git_commit": commit, "machine": machine, "gpu": gpu,
                    "timestamp_utc": timestamp, "solver_options": solver_options}
    result_rows: list[dict] = []

    print("GPU vs CPU PDHG on real instances")
    print(f"binary: {args.binary}")
    print(f"commit: {commit}  machine: {machine}  gpu: {gpu}")
    print(f"CPU arms: {', '.join(a[0] for a in gpu_arms.cpu_arms(args.cpu_threads))}\n")
    print(f"{'instance':>24}  {'arm':>8}  {'tol':>6}  {'status':>15}  {'seconds':>9}  "
          f"{'rel gap':>10}  speedup")
    print("-" * 100)

    # --- Refinery planning year (generate on the fly) ---
    with tempfile.TemporaryDirectory() as tmp_dir:
        refinery_mps = Path(tmp_dir) / "refinery_year.mps"
        gen = None
        if args.skip_refinery:
            print("refinery year skipped (--skip-refinery)")
        else:
            print("generating refinery year LP (779,640 rows)...")
            gen = subprocess.run(
                [sys.executable, str(REPO_ROOT / "bench" / "runners" / "generate_refinery_lp.py"),
                 "--periods", "8760", "--seed", "42", "--out", str(refinery_mps)],
                capture_output=True, text=True, check=False,
            )
        if gen is None:
            pass
        elif gen.returncode != 0 or not refinery_mps.exists():
            print(f"  ERROR generating refinery LP: {gen.stderr[:200]}")
        else:
            result_rows += measure("refinery_year", refinery_mps, args, stamp_fields)

    # --- Mittelmann instances ---
    for name in MITTELMANN_INSTANCES:
        mps = find_mittelmann_mps(name)
        if mps is None:
            print(f"  SKIP {name}: not in {MITTELMANN_DIR} — run fetch_mittelmann.py first")
            continue
        result_rows += measure(name, mps, args, stamp_fields)

    if not result_rows:
        print("no results — check that Mittelmann instances are fetched and GPU build works")
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    card = f"{args.card}-" if args.card else ""
    if args.out is not None:
        out = args.out
    elif solver_options:
        tag = re.sub(r"[^A-Za-z0-9]+", "-", solver_options).strip("-")
        out = RESULTS_DIR / f"gpu-ab-{commit}-{tag}.csv"
    else:
        out = RESULTS_DIR / f"gpu-real-{card}{commit}.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(result_rows)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
