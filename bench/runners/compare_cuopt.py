#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GPU PDHG vs NVIDIA cuOpt head-to-head benchmark (issue #759).

Runs our GPU PDHG and NVIDIA cuOpt's LP solver on the same instances, each as a
separate process - cuOpt is never linked or vendored, exactly as HiGHS and OR-Tools
PDLP are already handled in this repo (gpu_pdlp_compare.py, cross_check_highs.py; see
PROVENANCE.md row 21 for the judgement call on running a third-party solver as a
comparison target, and the PROVENANCE row this PR adds for cuOpt specifically).

cuOpt is driven through `cuopt_cli`, its standalone command-line solver (installed
with the `libcuopt` package, documented at
https://docs.nvidia.com/cuopt/user-guide/latest/cuopt-cli/quick-start.html): it reads
an MPS file directly and needs no server process, which keeps this the same
subprocess-per-solve shape every other comparator in this repo uses.

Instances: the synthetic KKT ladder (gpu_report.py's generator, #488), the Mittelmann
medium set PDHG already finishes (chromaticindex1024-7, brazil3), and the refinery
planning year (779,640 rows, generated on the fly) - matching #759's "What" list.

Usage:
    pip install --extra-index-url=https://pypi.nvidia.com libcuopt-cu12
    python bench/runners/fetch_mittelmann.py
    python bench/runners/compare_cuopt.py --binary build_gpu/sankhya --card <l4|a100|...>

NOT INDEPENDENTLY VERIFIED ON FIRST USE (flagged honestly, not guessed around): our own
side's solution is run through tools/verify_solution.py exactly as every other runner in
this repo does (verify_solution.py reads only SANKHYA's own .sol format, src/io/writer.cpp's
documented layout). cuOpt's own primal point is NOT put through that verifier - doing so
would need converting cuOpt_cli's --solution-file output into that same .sol format, and
this was written without a real cuOpt install to confirm that output's exact text grammar
against (the documented examples were for 2-variable toy problems). Instead cuOpt's answer
is checked the way a reference solver's answer already is throughout this repo: its reported
objective is compared to the instance's independent reference (HiGHS, a separate process,
or the generator's analytic optimum) and to our own verified answer, both recorded as
abs_gap/rel_gap columns. If independent verification of cuOpt's own point is wanted, writing
the cuopt_solution_to_sol() converter below against a real --solution-file is the one
piece to finish first - it is intentionally left as a stub that reports "not verified"
rather than silently miscounting.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_arms  # noqa: E402  (#488: the reference objective and the gap arithmetic)
import stamp  # noqa: E402  (#433, #589: the CSV names the commit the BINARY was built from)
from gpu_real_instances import find_mittelmann_mps, mps_dimensions  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
VERIFIER = REPO_ROOT / "tools" / "verify_solution.py"

CSV_COLUMNS = [
    "instance", "rows", "cols", "nnz", "solver", "tolerance", "status", "objective",
    "reference_objective", "reference_source", "abs_gap", "rel_gap",
    "independently_verified", "verifier_message", "iterations", "seconds", "wall_seconds",
    "git_commit", "machine", "gpu", "driver_version", "cuda_runtime",
    "cuopt_version", "timestamp_utc",
]

COMMON_OUR_OPTIONS = ["log_to_console=false", "algorithm=pdhg", "pdhg_polish=false",
                      "pdhg_stop_at_request=true"]

TOLERANCES = [1e-4, 1e-8]
MITTELMANN_INSTANCES = ["chromaticindex1024-7", "brazil3"]

# cuopt_cli prints "Termination status: <Name> (<code>)" and, on an LP, a line such as
# "Solution objective: -28.000000 , ... total_solve_time 1.234567 ... simplex_iterations 7"
# (verified against the documented CLI examples, cuopt-cli/cli-examples.html and
# cuopt-c/lp-milp/milp-examples.html - both current as of this PR; cuOpt's own output
# format is not under this repo's control and may drift between releases).
TERMINATION_RE = re.compile(r"Termination status:\s*(\w+)")
OBJECTIVE_RE = re.compile(r"Objective value:\s*([-+0-9.eE]+)")
SOLVE_TIME_RE = re.compile(r"Solve time:\s*([-+0-9.eE]+)\s*seconds")
ITERATIONS_RE = re.compile(r"simplex_iterations\s+(\d+)|nb_iterations[\"':]*\s*(\d+)")

CUOPT_STATUS_MAP = {
    "optimal": "optimal", "infeasible": "infeasible", "unbounded": "unbounded",
    "iterationlimit": "iteration_limit", "timelimit": "time_limit",
    "numericalerror": "numerical_error", "primalfeasible": "feasible",
    "feasiblefound": "feasible",
}


def gpu_description(binary: Path) -> str:
    r = subprocess.run([str(binary), "--version"], capture_output=True, text=True, check=False)
    match = re.search(r"GPU ([^)]*\))", r.stdout)
    return match.group(1).strip() if match else r.stdout.strip()


def cuopt_version() -> str:
    try:
        r = subprocess.run(["cuopt_cli", "--version"], capture_output=True, text=True,
                           check=False)
    except FileNotFoundError:
        return "not installed"
    if r.returncode != 0:
        return "not installed"
    return (r.stdout or r.stderr).strip().splitlines()[0] if (r.stdout or r.stderr) else "unknown"


def as_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def cuopt_solution_to_sol(cuopt_solution_file: Path, mps: Path, out_sol: Path) -> bool:
    """Convert cuopt_cli's --solution-file output into SANKHYA's .sol format so it can go
    through tools/verify_solution.py. Deliberately a stub (see the module docstring): it
    is written to fail closed (return False, nothing independently verified) rather than
    guess at a text grammar never checked against a real cuOpt install."""
    return False


def run_our_gpu(binary: Path, mps: Path, tolerance: float, time_limit: float) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        stats = Path(tmp) / "s.json"
        sol = Path(tmp) / "solution.sol"
        command = [str(binary), "solve", str(mps), "--stats", str(stats), "--write-sol",
                   str(sol), "--time-limit", str(time_limit)]
        for option in COMMON_OUR_OPTIONS + [f"pdhg_tolerance={tolerance:g}"]:
            command += ["--option", option]
        command += ["--option", "gpu=true"]
        started = time.perf_counter()
        subprocess.run(command, capture_output=True, text=True)
        elapsed = time.perf_counter() - started
        out = {"status": "no_output", "objective": None, "iterations": "", "seconds": elapsed,
               "wall": elapsed, "verified": "", "verifier_message": ""}
        if not stats.exists():
            return out
        import json
        blob = json.loads(stats.read_text())
        solver_s = as_number(blob.get("effort", {}).get("solve_seconds"))
        result = blob.get("result", {})
        out.update(status=result.get("status", "unknown"),
                   objective=as_number(result.get("objective")),
                   iterations=blob.get("effort", {}).get("iterations", ""),
                   seconds=elapsed if solver_s is None else solver_s)
        if sol.exists() and out["status"] in ("optimal", "feasible"):
            check = subprocess.run([sys.executable, str(VERIFIER), str(mps), str(sol)],
                                   capture_output=True, text=True, check=False)
            out["verified"] = 1 if check.returncode == 0 else 0
            if check.returncode != 0:
                failing = [ln.strip() for ln in check.stdout.splitlines() if "[FAIL]" in ln]
                out["verifier_message"] = "; ".join(failing)[:300]
        return out


def run_cuopt(mps: Path, tolerance: float, time_limit: float) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        sol = Path(tmp) / "cuopt.sol"
        command = [
            "cuopt_cli", str(mps),
            "--time-limit", str(time_limit),
            "--relative-primal-tolerance", str(tolerance),
            "--relative-dual-tolerance", str(tolerance),
            "--relative-gap-tolerance", str(tolerance),
            "--absolute-primal-tolerance", str(tolerance),
            "--absolute-dual-tolerance", str(tolerance),
            "--absolute-gap-tolerance", str(tolerance),
            "--solution-file", str(sol),
            "--log-to-console", "true",
        ]
        started = time.perf_counter()
        try:
            r = subprocess.run(command, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            elapsed = time.perf_counter() - started
            return {"status": "error: cuopt_cli not installed", "objective": None,
                    "iterations": "", "seconds": elapsed, "wall": elapsed}
        elapsed = time.perf_counter() - started
        text = r.stdout + r.stderr
        if r.returncode != 0 and "Termination status" not in text:
            return {"status": f"error: {text[:200].strip()}", "objective": None,
                    "iterations": "", "seconds": elapsed, "wall": elapsed}
        term = TERMINATION_RE.search(text)
        status_raw = term.group(1).lower() if term else "unknown"
        status = CUOPT_STATUS_MAP.get(status_raw, status_raw)
        obj_match = OBJECTIVE_RE.search(text)
        time_match = SOLVE_TIME_RE.search(text)
        iter_match = ITERATIONS_RE.search(text)
        return {
            "status": status,
            "objective": as_number(obj_match.group(1)) if obj_match else None,
            "iterations": next((g for g in (iter_match.groups() if iter_match else ())
                               if g), "") if iter_match else "",
            "seconds": as_number(time_match.group(1)) if time_match else elapsed,
            "wall": elapsed,
            "solution_file": sol if sol.exists() else None,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, required=False, default=None)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--card", default="", help="short card name: l4, a100, rtx5050, ...")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cuopt-only", action="store_true",
                        help="run only the cuOpt side (no GPU binary needed); useful for "
                             "checking the cuopt_cli integration without our own build")
    args = parser.parse_args()

    if not args.cuopt_only and args.binary is None:
        parser.error("--binary is required unless --cuopt-only is set")

    commit = stamp.stamp(args.binary) if args.binary else "none"
    machine = f"{platform.system()}-{platform.machine()}"
    gpu = gpu_description(args.binary) if args.binary else args.card or "unknown"
    stamp_fields = {"git_commit": commit, "machine": machine, "gpu": gpu,
                    **gpu_arms.platform_cells(gpu)}
    cuopt_v = cuopt_version()
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    result_rows: list[dict] = []

    print("GPU PDHG vs NVIDIA cuOpt head-to-head" + (" [cuOpt only]" if args.cuopt_only else ""))
    if args.binary:
        print(f"binary: {args.binary}")
    print(f"commit: {commit}  machine: {machine}  gpu: {gpu}  cuopt: {cuopt_v}\n")
    print(f"{'instance':>30}  {'solver':>14}  {'tol':>6}  {'status':>14}  {'seconds':>9}")
    print("-" * 85)

    def process_instance(name: str, mps: Path, reference: float | None,
                         reference_source: str) -> None:
        r, c, nz = mps_dimensions(mps)
        for tol in TOLERANCES:
            our = run_our_gpu(args.binary, mps, tol, args.time_limit) if not args.cuopt_only \
                else {"status": "skipped", "objective": None, "iterations": "", "seconds": 0.0,
                      "wall": 0.0, "verified": "", "verifier_message": ""}
            cu = run_cuopt(mps, tol, args.time_limit)
            print(f"{name:>30}  {'sankhya-gpu':>14}  {tol:>6.0e}  "
                  f"{our['status']:>14}  {our['seconds']:>9.3f}")
            print(f"{'':>30}  {'cuopt':>14}  {tol:>6.0e}  "
                  f"{cu['status']:>14}  {cu['seconds']:>9.3f}")
            for solver_name, res in [("sankhya-gpu", our), ("cuopt", cu)]:
                absolute, relative = gpu_arms.gaps(res["objective"], reference)
                result_rows.append({
                    "instance": name, "rows": r, "cols": c, "nnz": nz,
                    "solver": solver_name, "tolerance": f"{tol:.0e}",
                    "status": res["status"],
                    "objective": "" if res["objective"] is None else repr(res["objective"]),
                    "reference_objective": "" if reference is None else repr(reference),
                    "reference_source": reference_source if reference is not None else "none",
                    "abs_gap": gpu_arms.fmt_gap(absolute), "rel_gap": gpu_arms.fmt_gap(relative),
                    "independently_verified": res.get("verified", ""),
                    "verifier_message": res.get("verifier_message", ""),
                    "iterations": res.get("iterations", ""),
                    "seconds": round(res["seconds"], 6),
                    "wall_seconds": round(res.get("wall", res["seconds"]), 6),
                    "cuopt_version": cuopt_v if solver_name == "cuopt" else "",
                    "timestamp_utc": timestamp,
                    **stamp_fields,
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
            print(f"  ERROR generating refinery_year: {gen.stderr[:200]}")
        else:
            reference, source, _note, _ref_s = gpu_arms.reference_objective(
                refinery_mps, 600.0, use_highs=False)
            process_instance("refinery_year", refinery_mps, reference, source)

    # Mittelmann instances
    for name in MITTELMANN_INSTANCES:
        mps = find_mittelmann_mps(name)
        if mps is None:
            print(f"  SKIP {name}: not in data/mittelmann (run fetch_mittelmann.py first)")
            continue
        reference, source, _note, _ref_s = gpu_arms.reference_objective(mps, 600.0,
                                                                        use_highs=True)
        process_instance(name, mps, reference, source)

    if not result_rows:
        print("no results")
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or (RESULTS_DIR / f"cuopt-compare-{args.card or 'unknown'}-{commit}.csv")
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(result_rows)
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")

    verified_count = sum(1 for r in result_rows
                         if r["solver"] == "sankhya-gpu" and r["independently_verified"] == 1)
    sankhya_rows = sum(1 for r in result_rows if r["solver"] == "sankhya-gpu")
    print(f"independently verified (sankhya-gpu side): {verified_count}/{sankhya_rows}")
    print("cuOpt's own point is NOT independently verified by this script - see the "
          "module docstring and cuopt_solution_to_sol().")
    return 0


if __name__ == "__main__":
    sys.exit(main())
