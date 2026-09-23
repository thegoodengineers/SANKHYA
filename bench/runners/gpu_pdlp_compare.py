#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GPU PDHG vs OR-Tools PDLP head-to-head benchmark (issue #447).

Runs our GPU PDHG and OR-Tools PDLP on the same instances, each as a separate
process (the pattern from bench/runners/cross_check_highs.py).  OR-Tools ships
PDLP via its published `ortools` PyPI wheel — no source is read or vendored,
exactly as highspy is handled.  See PROVENANCE.md row 21 for the judgement call
on running OR-Tools PDLP as a comparison target.

Instances: the refinery planning year (779,640 rows, generated on the fly) and
the two Mittelmann LPs that PDHG already finishes: chromaticindex1024-7 and
brazil3 — the same set as gpu_real_instances.py, so the two CSVs are comparable.

Usage:
    pip install ortools          # installs the published wheel
    python bench/runners/fetch_mittelmann.py
    python bench/runners/gpu_pdlp_compare.py --binary build_gpu/sankhya
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
import stamp  # noqa: E402  (#433, #589: the CSV names the commit the BINARY was built from)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
MITTELMANN_DIR = REPO_ROOT / "data" / "mittelmann"

CSV_COLUMNS = [
    "instance", "rows", "cols", "nnz", "solver", "tolerance",
    "status", "objective", "iterations", "seconds", "wall_seconds",
    "git_commit", "machine", "gpu", "timestamp_utc",
]

COMMON_OUR_OPTIONS = ["log_to_console=false", "algorithm=pdhg", "pdhg_polish=false",
                      "pdhg_stop_at_request=true"]

TOLERANCES = [1e-4, 1e-8]

MITTELMANN_INSTANCES = ["chromaticindex1024-7", "brazil3"]

# OR-Tools PDLP solver script, run as a subprocess to keep provenance clean.
# Uses ortools.linear_solver.python.model_builder for MPS loading (the supported
# Python API path) and SetSolverSpecificParametersAsString with a PDLP text proto
# to set termination tolerances (PDLP ignores MPSolverParameters knobs).
PDLP_RUNNER_SCRIPT = r"""
import sys, time, json
try:
    from ortools.linear_solver.python import model_builder as mb
except ImportError:
    print(json.dumps({"error": "ortools not installed"}))
    sys.exit(1)

mps_file, tol_str, time_limit_str = sys.argv[1], sys.argv[2], sys.argv[3]
tol = float(tol_str)
time_limit = float(time_limit_str)

# Load MPS via the model_builder API (handles .mps and .mps.gz transparently).
try:
    model = mb.Model()
    model.import_from_mps_file(mps_file)
except Exception as exc:
    print(json.dumps({"error": f"import_from_mps_file failed: {exc}"}))
    sys.exit(1)

solver = mb.ModelSolver("PDLP")
if solver is None:
    print(json.dumps({"error": "PDLP not available in this ortools build"}))
    sys.exit(1)

solver.set_time_limit_in_seconds(time_limit)
# PDLP termination tolerances — must be set via solver-specific text proto;
# MPSolverParameters knobs are not forwarded to PDLP.
solver.set_solver_specific_parameters(
    f"termination_criteria {{ eps_optimal_relative: {tol} eps_optimal_absolute: {tol} }}"
)

t0 = time.perf_counter()
result_status = solver.solve(model)
elapsed = time.perf_counter() - t0

status_str = str(result_status).split(".")[-1].lower()
feasible = status_str in ("optimal", "feasible")
print(json.dumps({
    "status": status_str,
    "objective": solver.objective_value if feasible else None,
    "wall_seconds": elapsed,
}))
"""


def gpu_description(binary: Path) -> str:
    r = subprocess.run([str(binary), "--version"], capture_output=True, text=True, check=False)
    match = re.search(r"GPU ([^)]*\))", r.stdout)
    return match.group(1).strip() if match else r.stdout.strip()


def as_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_our_gpu(binary: Path, mps: Path, tolerance: float, time_limit: float) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        stats = Path(tmp) / "s.json"
        command = [str(binary), "solve", str(mps), "--stats", str(stats),
                   "--time-limit", str(time_limit)]
        for option in COMMON_OUR_OPTIONS + [f"pdhg_tolerance={tolerance:g}"]:
            command += ["--option", option]
        command += ["--option", "gpu=true"]
        started = time.perf_counter()
        subprocess.run(command, capture_output=True, text=True)
        elapsed = time.perf_counter() - started
        if not stats.exists():
            return {"status": "no_output", "objective": None, "iterations": "",
                    "seconds": elapsed, "wall": elapsed}
        blob = json.loads(stats.read_text())
        solver_s = as_number(blob.get("effort", {}).get("solve_seconds"))
        result = blob.get("result", {})
        return {
            "status": result.get("status", "unknown"),
            "objective": as_number(result.get("objective")),
            "iterations": blob.get("effort", {}).get("iterations", ""),
            "seconds": elapsed if solver_s is None else solver_s,
            "wall": elapsed,
        }


def run_pdlp(mps: Path, tolerance: float, time_limit: float) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "pdlp_run.py"
        script.write_text(PDLP_RUNNER_SCRIPT, encoding="utf-8")
        started = time.perf_counter()
        r = subprocess.run(
            [sys.executable, str(script), str(mps), str(tolerance), str(time_limit)],
            capture_output=True, text=True,
        )
        elapsed = time.perf_counter() - started
        try:
            result = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:
            result = {"error": r.stderr[:200] or r.stdout[:200]}
        if "error" in result:
            return {"status": f"error: {result['error']}", "objective": None,
                    "iterations": "", "seconds": elapsed, "wall": elapsed}
        return {
            "status": result.get("status", "unknown"),
            "objective": result.get("objective"),
            "iterations": "",
            "seconds": result.get("wall_seconds", elapsed),
            "wall": elapsed,
        }


def find_mittelmann_mps(name: str) -> Path | None:
    for ext in (".mps", ".mps.gz"):
        p = MITTELMANN_DIR / (name + ext)
        if p.exists():
            return p
    return None


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, required=False, default=None)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--pdlp-only", action="store_true",
                        help="run only the OR-Tools PDLP side (no GPU binary needed); "
                             "useful for verifying the PDLP API fixes without the card")
    args = parser.parse_args()

    if not args.pdlp_only and args.binary is None:
        parser.error("--binary is required unless --pdlp-only is set")

    commit = stamp.stamp(args.binary)
    machine = f"{platform.system()}-{platform.machine()}"
    gpu = gpu_description(args.binary) if args.binary else "pdlp-only"
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    result_rows: list[dict] = []

    print("GPU PDHG vs OR-Tools PDLP head-to-head" + (" [PDLP only]" if args.pdlp_only else ""))
    if args.binary:
        print(f"binary: {args.binary}")
    print(f"commit: {commit}  machine: {machine}  gpu: {gpu}\n")
    print(f"{'instance':>30}  {'solver':>14}  {'tol':>6}  {'status':>12}  {'seconds':>9}  {'ratio':>8}")
    print("-" * 90)

    def process_instance(name: str, mps: Path) -> None:
        r, c, nz = mps_dimensions(mps)
        if not args.pdlp_only:
            # warm-up GPU for this instance
            run_our_gpu(args.binary, mps, TOLERANCES[0], args.time_limit)
        for tol in TOLERANCES:
            our = run_our_gpu(args.binary, mps, tol, args.time_limit) if not args.pdlp_only \
                else {"status": "skipped", "objective": None, "iterations": "", "seconds": 0.0, "wall": 0.0}
            pdlp = run_pdlp(mps, tol, args.time_limit)
            ratio_str = "—"
            if pdlp["seconds"] and our["seconds"]:
                ratio = pdlp["seconds"] / our["seconds"]
                ratio_str = f"{ratio:.2f}x"
            print(f"{name:>30}  {'sankhya-gpu':>14}  {tol:>6.0e}  "
                  f"{our['status']:>12}  {our['seconds']:>9.3f}  {'—':>8}")
            print(f"{'':>30}  {'ortools-pdlp':>14}  {tol:>6.0e}  "
                  f"{pdlp['status']:>12}  {pdlp['seconds']:>9.3f}  {ratio_str:>8}")
            for solver_name, res in [("sankhya-gpu", our), ("ortools-pdlp", pdlp)]:
                result_rows.append({
                    "instance": name,
                    "rows": r, "cols": c, "nnz": nz,
                    "solver": solver_name,
                    "tolerance": f"{tol:.0e}",
                    "status": res["status"],
                    "objective": "" if res["objective"] is None else repr(res["objective"]),
                    "iterations": res.get("iterations", ""),
                    "seconds": round(res["seconds"], 6),
                    "wall_seconds": round(res.get("wall", res["seconds"]), 6),
                    "git_commit": commit, "machine": machine, "gpu": gpu,
                    "timestamp_utc": timestamp,
                })

    # Refinery year
    with tempfile.TemporaryDirectory() as tmp_dir:
        refinery_mps = Path(tmp_dir) / "refinery_year.mps"
        print("generating refinery year LP (779,640 rows)...")
        gen = subprocess.run(
            [sys.executable, str(REPO_ROOT / "bench" / "runners" / "generate_refinery_lp.py"),
             "--periods", "8760", "--seed", "42", "--out", str(refinery_mps)],
            capture_output=True, text=True, check=False,
        )
        if gen.returncode != 0 or not refinery_mps.exists():
            print(f"  ERROR: {gen.stderr[:200]}")
        else:
            process_instance("refinery_year", refinery_mps)

    # Mittelmann instances
    for name in MITTELMANN_INSTANCES:
        mps = find_mittelmann_mps(name)
        if mps is None:
            print(f"  SKIP {name}: not in {MITTELMANN_DIR}")
            continue
        process_instance(name, mps)

    if not result_rows:
        print("no results")
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or (RESULTS_DIR / f"gpu-pdlp-{commit}.csv")
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(result_rows)
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
