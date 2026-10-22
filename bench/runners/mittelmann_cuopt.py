#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Our GPU PDHG against NVIDIA cuOpt on Mittelmann's LPfeas instances, same card, same time
limit, both answers written as .sol files and judged by tools/verify_solution.py (#759).

THE SET is `bench/runners/lpfeas14.json`: the 14 smallest archives, by compressed size,
among the instances of Mittelmann's LPfeas benchmark (https://plato.asu.edu/ftp/lpfeas.html)
whose file sits at the top level of https://plato.asu.edu/ftp/lptestset/. The manifest pins
each archive's URL, size and sha256 (the ten it shares with data/mittelmann/reference.json
match it byte for byte) and the dimensions the benchmark page prints. Mittelmann publishes
solver TIMES, not objective values, so the reference objective is HiGHS's: the manifest
carries the two a committed CSV already holds, and `--reference-time-limit` lets HiGHS, run
as a separate process on the host, supply the rest (blank, never guessed, when it does not
finish optimal).

cuOpt RUNS AS A SEPARATE PROCESS through its published Python package (`cuopt-cu12` on
PyPI; `pip install --extra-index-url=https://pypi.nvidia.com cuopt-cu12`), this script
re-invoked as a child with `--worker`, exactly as rivals.py drives HiGHS and SCIP: nothing of
cuOpt is linked, vendored or read, and the calls are the ones its API reference documents
(`Problem.readMPS`, `SolverSettings.set_parameter`, `Problem.solve`, the `Status`, `SolveTime`,
`ObjValue`, `Value`, `ReducedCost` and `DualValue` attributes). Its point, reduced costs and
duals are written into our .sol layout by rivals.write_sol; the row activities are A x
recomputed from its point with the verifier's own MPS parser, so the file claims nothing
cuOpt did not report. See docs/PROVENANCE.md, judgement call 39.

BOTH SIDES GET THE SAME REQUEST: relative primal, dual and gap tolerance `tol` and one time
limit. Ours runs `algorithm=pdhg gpu=true pdhg_polish=false pdhg_stop_at_request=true`
(gpu_real_instances.run_solve, #488); cuOpt runs its PDLP (`method=PDLP`, the GPU engine
the comparison is about; `--cuopt-method` changes it and the CSV records it). EVERY ANSWER
IS VERIFIED TWICE: at the verifier's defaults (absolute 1e-7, the project standard, the
`independently_verified` column) and at the requested tolerance passed as the verifier's
primal and dual tolerance (`verified_at_tolerance`), so a first-order answer at 1e-4 that
is a 1e-4 answer and not a 1e-7 one is reported as exactly that, for both solvers alike.

Usage:
    python bench/runners/mittelmann_cuopt.py --fetch                 # instances, pinned
    python bench/runners/mittelmann_cuopt.py --binary build_gpu/sankhya --card a100 \\
        --time-limit 300 --tolerances 1e-4 1e-8
"""
from __future__ import annotations

import argparse
import bz2
import csv
import datetime
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_data  # noqa: E402  (Netlib's emps decoder for the packed archives)
import fetch_mittelmann  # noqa: E402  (download with retries, packed-format detection)
import gpu_arms  # noqa: E402  (#488: gaps, platform cells, sha256)
import rivals  # noqa: E402  (write_sol: the .sol layout the verifier reads)
import stamp  # noqa: E402  (#433: the CSV names the commit the binary was built from)
from gpu_real_instances import gpu_description  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
DATA_DIR = REPO_ROOT / "data" / "mittelmann"
MANIFEST = Path(__file__).resolve().parent / "lpfeas14.json"
VERIFIER = REPO_ROOT / "tools" / "verify_solution.py"

CSV_COLUMNS = [
    "instance", "instance_sha256", "rows", "cols", "nnz", "solver", "method", "tolerance",
    "time_limit", "status", "objective", "reference_objective", "reference_source", "abs_gap",
    "rel_gap", "independently_verified", "verified_at_tolerance", "verifier_message",
    "iterations", "seconds", "wall_seconds", "git_commit", "machine", "gpu", "driver_version",
    "cuda_runtime", "cuopt_version", "timestamp_utc",
]

# cuOpt's Status names (its Python API's `Problem.Status.name`), lowercased, to ours.
CUOPT_STATUS = {
    "optimal": "optimal", "infeasible": "infeasible", "primalinfeasible": "infeasible",
    "unbounded": "unbounded", "dualinfeasible": "unbounded", "iterationlimit": "iteration_limit",
    "timelimit": "time_limit", "numericalerror": "numerical_error", "primalfeasible": "feasible",
    "feasiblefound": "feasible",
}


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def fetch(manifest: dict, force: bool = False) -> list[Path]:
    """Every instance of the manifest into data/mittelmann/, the archive refused unless its
    sha256 is the pinned one (the MIPLIB 3 rule, #761), decoded from Netlib's packed format
    where the archive is one (fetch_mittelmann.is_packed)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, entry in manifest["instances"].items():
        archive = DATA_DIR / entry["archive"]
        mps = DATA_DIR / f"{name}.mps"
        if force or not archive.exists():
            print(f"fetching {entry['url']}")
            fetch_mittelmann.download(entry["url"], archive)
        digest = fetch_mittelmann.sha256_of(archive)
        if digest != entry["archive_sha256"]:
            archive.unlink()
            raise SystemExit(f"{archive}: sha256 {digest[:12]} is not the manifest's "
                             f"{entry['archive_sha256'][:12]}; the file was removed")
        if force or not mps.exists():
            with bz2.open(archive, "rb") as src, mps.open("wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            if fetch_mittelmann.is_packed(mps):
                packed = mps.with_suffix(".packed")
                mps.replace(packed)
                emps, _ = fetch_data.build_emps(DATA_DIR)
                fetch_data.decompress(emps, packed, mps)
        mps_sha = gpu_arms.sha256_file(mps)
        pinned = entry.get("mps_sha256")
        if pinned and mps_sha != pinned:
            print(f"  {name}: decoded MPS sha256 {mps_sha[:12]} differs from the manifest's "
                  f"{pinned[:12]}; the CSV records what was solved")
        print(f"  {name}: {archive.stat().st_size / 1e6:.1f} MB archive, "
              f"{mps.stat().st_size / 1e6:.1f} MB MPS, sha256 {mps_sha[:12]}")
        paths.append(mps)
    return paths


# ------------------------------------------------------------------------------------------
# cuOpt, in a child process
# ------------------------------------------------------------------------------------------

def cuopt_worker(mps: Path, tolerance: float, time_limit: float, method: str,
                 sol: Path) -> dict:
    """Runs inside the child: cuOpt's documented Python API, nothing else."""
    import importlib  # noqa: PLC0415
    problem_module = importlib.import_module("cuopt.linear_programming.problem")
    settings_module = importlib.import_module("cuopt.linear_programming.solver_settings")
    parameters = importlib.import_module("cuopt.linear_programming.solver.solver_parameters")
    started = time.perf_counter()
    problem = problem_module.Problem.readMPS(str(mps))
    read_seconds = time.perf_counter() - started
    settings = settings_module.SolverSettings()
    settings.set_parameter("time_limit", float(time_limit))
    for key in ("absolute_primal_tolerance", "relative_primal_tolerance",
                "absolute_dual_tolerance", "relative_dual_tolerance",
                "absolute_gap_tolerance", "relative_gap_tolerance"):
        settings.set_parameter(key, float(tolerance))
    if method:
        # SolverMethod lives in solver_settings in 26.8.0 and in solver_parameters in the
        # 26.10 reference; take whichever the installed package has.
        enum = getattr(settings_module, "SolverMethod", None) or getattr(parameters, "SolverMethod")
        settings.set_parameter(parameters.CUOPT_METHOD, getattr(enum, method))
    started = time.perf_counter()
    problem.solve(settings)
    wall = time.perf_counter() - started
    status_name = str(getattr(problem.Status, "name", problem.Status))
    status = CUOPT_STATUS.get(status_name.lower().replace("_", ""), status_name.lower())
    stats = getattr(problem, "SolutionStats", None)
    iterations = ""
    for attribute in ("num_iterations", "nb_iterations", "iterations", "n_iterations",
                      "pdlp_iterations", "simplex_iterations"):
        if stats is not None and getattr(stats, attribute, None) is not None:
            iterations = getattr(stats, attribute)
            break
    out = {"status": status, "message": status_name, "objective": None,
           "seconds": float(getattr(problem, "SolveTime", wall)), "wall": wall,
           "read_seconds": read_seconds, "iterations": iterations}
    if status in ("optimal", "feasible"):
        out["objective"] = float(problem.ObjValue)
        columns = {v.VariableName: (float(v.Value), float(getattr(v, "ReducedCost", 0.0)))
                   for v in problem.getVariables()}
        rows = {c.ConstraintName: (0.0, float(getattr(c, "DualValue", 0.0)))
                for c in problem.getConstraints()}
        # PRIMAL ONLY, like SCIP in rivals.py: on afiro at 1e-4 cuOpt's DualValue and
        # ReducedCost vectors close neither the reduced-cost identity nor strong duality in
        # our convention under any sign choice (22 Oct 2026, the four combinations tried),
        # so the file is written `feasible` and the verifier judges the point, its bounds,
        # its rows and its objective, and skips the dual conditions. The duals are kept in
        # the file as cuOpt gave them, for the record.
        rivals.write_sol(sol, solver="cuopt", status="feasible", objective=out["objective"],
                         columns=columns, rows=rows,
                         message=f"cuOpt {status_name}; primal only (its duals are not in "
                                 "our convention); row activities recomputed as A x")
    return out


def fill_row_activities(sol: Path, mps: Path) -> None:
    """Replace the placeholder row activities with A x from the file's own point, computed
    with the verifier's MPS parser, so the .sol is consistent before the verifier reads it."""
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from verify_solution_mps import parse_mps  # noqa: PLC0415
    from verify_solution_sol import parse_sol  # noqa: PLC0415
    model = parse_mps(mps)
    solution = parse_sol(sol)
    activity = [0.0] * model.num_rows
    for j, name in enumerate(model.col_names):
        x = solution.col_value.get(name, 0.0)
        if x:
            for i, value in model.entries[j]:
                activity[i] += value * x
    rows = {name: (activity[i], solution.row_dual.get(name, 0.0))
            for i, name in enumerate(model.row_names)}
    columns = {n: (solution.col_value[n], solution.col_dual.get(n, 0.0))
               for n in solution.col_value}
    rivals.write_sol(sol, solver="cuopt", status=solution.status,
                     objective=solution.header_float("objective"), columns=columns, rows=rows,
                     message=solution.header.get("message", ""))


def run_cuopt(mps: Path, tolerance: float, time_limit: float, method: str, sol: Path) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(mps),
                   repr(float(tolerance)), repr(float(time_limit)), method, str(sol),
                   str(report)]
        started = time.perf_counter()
        try:
            done = subprocess.run(command, capture_output=True, text=True,
                                  timeout=time_limit + rivals.HANG_MARGIN_SECONDS + 600.0)
        except subprocess.TimeoutExpired:
            return {"status": "hung", "objective": None, "iterations": "",
                    "seconds": time.perf_counter() - started,
                    "wall": time.perf_counter() - started, "message": "no exit"}
        wall = time.perf_counter() - started
        if not report.exists():
            last = (done.stderr.strip().splitlines() or ["no output"])[-1][:300]
            return {"status": "crashed", "objective": None, "iterations": "", "seconds": wall,
                    "wall": wall, "message": last}
        out = json.loads(report.read_text(encoding="utf-8"))
    if sol.exists():
        fill_row_activities(sol, mps)
    return out


def run_ours(binary: Path, mps: Path, tolerance: float, time_limit: float, sol: Path) -> dict:
    """One GPU PDHG solve, the same options as gpu_real_instances.run_solve (#488), with
    the .sol kept for the two verifications."""
    with tempfile.TemporaryDirectory() as tmp:
        stats = Path(tmp) / "stats.json"
        command = [str(binary), "solve", str(mps), "--stats", str(stats), "--write-sol",
                   str(sol), "--time-limit", str(time_limit)]
        for option in ("log_to_console=false", "algorithm=pdhg", "pdhg_polish=false",
                       "pdhg_stop_at_request=true", f"pdhg_tolerance={tolerance:g}", "gpu=true"):
            command += ["--option", option]
        started = time.perf_counter()
        done = subprocess.run(command, capture_output=True, text=True, check=False)
        wall = time.perf_counter() - started
        if not stats.exists():
            return {"status": "no_output", "objective": None, "iterations": "", "seconds": wall,
                    "wall": wall, "message": done.stderr.strip()[-300:]}
        blob = json.loads(stats.read_text())
        effort, result = blob.get("effort", {}), blob.get("result", {})
        seconds = effort.get("solve_seconds")
        return {"status": result.get("status", "unknown"), "objective": result.get("objective"),
                "iterations": effort.get("iterations", ""),
                "seconds": wall if seconds is None else float(seconds), "wall": wall,
                "message": result.get("message", "")}


def verify(mps: Path, sol: Path, tolerance: float | None) -> tuple[int, str]:
    """(1 or 0, the failing checks) from tools/verify_solution.py; at the requested
    tolerance when one is given, at the verifier's defaults otherwise."""
    command = [sys.executable, str(VERIFIER), str(mps), str(sol)]
    if tolerance is not None:
        command += ["--primal-tolerance", repr(tolerance), "--dual-tolerance", repr(tolerance)]
    check = subprocess.run(command, capture_output=True, text=True, check=False)
    failing = [ln.strip() for ln in check.stdout.splitlines() if "[FAIL]" in ln]
    return (1 if check.returncode == 0 else 0), "; ".join(failing)[:300]


def cuopt_version() -> str:
    script = "import importlib.metadata as m; print(m.version('cuopt-cu12'))"
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          check=False)
    return done.stdout.strip() if done.returncode == 0 else "not installed"


# ------------------------------------------------------------------------------------------
# The rows
# ------------------------------------------------------------------------------------------

def build_row(name: str, entry: dict, digest: str, solver: str, method: str, tolerance: float,
              time_limit: float, result: dict, verified: tuple, at_tolerance: tuple,
              reference: tuple, stamp_fields: dict) -> dict:
    """One CSV row; pure, so test_mittelmann_cuopt.py pins it without a card."""
    absolute, relative = gpu_arms.gaps(result.get("objective"), reference[0])
    return {
        "instance": name, "instance_sha256": digest,
        "rows": entry.get("rows", ""), "cols": entry.get("columns", ""),
        "nnz": entry.get("nonzeros", ""),
        "solver": solver, "method": method, "tolerance": f"{tolerance:.0e}",
        "time_limit": time_limit, "status": result.get("status", "unknown"),
        "objective": "" if result.get("objective") is None else repr(result["objective"]),
        "reference_objective": "" if reference[0] is None else repr(reference[0]),
        "reference_source": reference[1] if reference[0] is not None else "none",
        "abs_gap": gpu_arms.fmt_gap(absolute), "rel_gap": gpu_arms.fmt_gap(relative),
        "independently_verified": verified[0], "verified_at_tolerance": at_tolerance[0],
        "verifier_message": (verified[1] or at_tolerance[1] or result.get("message", ""))[:300],
        "iterations": result.get("iterations", ""),
        "seconds": round(float(result.get("seconds", 0.0)), 6),
        "wall_seconds": round(float(result.get("wall", result.get("seconds", 0.0))), 6),
        **stamp_fields,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", nargs=6, metavar="ARG", help=argparse.SUPPRESS)
    parser.add_argument("--fetch", action="store_true", help="fetch the 14 instances and stop")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--card", default="", help="short card name for the CSV file name")
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--tolerances", type=float, nargs="+", default=[1e-4, 1e-8])
    parser.add_argument("--cuopt-method", default="PDLP",
                        help="cuOpt's method: PDLP (default, the GPU engine), DualSimplex, "
                             "Barrier, Concurrent; empty keeps cuOpt's own default")
    parser.add_argument("--reference-time-limit", type=float, default=600.0,
                        help="HiGHS's own limit for a reference the manifest lacks; 0 skips "
                             "HiGHS and leaves the reference blank")
    parser.add_argument("--instances", nargs="*", help="a subset of the manifest's names")
    parser.add_argument("--skip-ours", action="store_true", help="cuOpt only")
    parser.add_argument("--skip-cuopt", action="store_true", help="our GPU PDHG only")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.worker:
        mps, tolerance, time_limit, method, sol, report = args.worker
        out = cuopt_worker(Path(mps), float(tolerance), float(time_limit), method, Path(sol))
        Path(report).write_text(json.dumps(out), encoding="utf-8")
        return 0

    manifest = load_manifest()
    if args.fetch:
        fetch(manifest)
        return 0
    if args.binary is None and not args.skip_ours:
        parser.error("--binary is required unless --skip-ours")

    names = args.instances or list(manifest["instances"])
    unknown = [n for n in names if n not in manifest["instances"]]
    if unknown:
        parser.error(f"not in {MANIFEST.name}: {', '.join(unknown)}")
    commit = stamp.stamp(args.binary) if args.binary else stamp.stamp()
    gpu = gpu_description(args.binary) if args.binary else ""
    stamp_fields = {"git_commit": commit,
                    "machine": f"{platform.system()}-{platform.machine()}", "gpu": gpu,
                    **gpu_arms.platform_cells(gpu),
                    "cuopt_version": "" if args.skip_cuopt else cuopt_version(),
                    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
                    .isoformat(timespec="seconds")}
    print(f"GPU PDHG vs cuOpt on LPfeas-14: commit {commit}  gpu {gpu}  "
          f"cuopt {stamp_fields['cuopt_version']}  limit {args.time_limit:g} s")
    print(f"{'instance':>20} {'solver':>8} {'tol':>6} {'status':>15} {'seconds':>9} "
          f"{'rel gap':>10} {'verified':>8} {'at tol':>6}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or RESULTS_DIR / f"lpfeas14-cuopt-{args.card or 'unknown'}-{commit}.csv"
    # RESUMABLE: every row is appended and flushed as it is produced, and a (instance,
    # solver, tolerance) already in the file is not run again, so a dropped connection or
    # a killed run loses nothing but the solve in flight.
    done_keys: set[tuple] = set()
    if out.exists():
        with out.open(newline="", encoding="utf-8") as handle:
            done_keys = {(r["instance"], r["solver"], r["tolerance"])
                         for r in csv.DictReader(handle)}
        print(f"resuming {out}: {len(done_keys)} rows already there")
    handle = out.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
    if not done_keys:
        writer.writeheader()
        handle.flush()
    rows: list[dict] = []
    for name in names:
        entry = manifest["instances"][name]
        mps = DATA_DIR / f"{name}.mps"
        if not mps.exists():
            print(f"  SKIP {name}: run --fetch first")
            continue
        digest = gpu_arms.sha256_file(mps)
        if entry.get("reference_objective") is not None:
            reference = (entry["reference_objective"], entry["reference_source"])
        elif args.reference_time_limit > 0:
            value, source, _note, _s = gpu_arms.reference_objective(
                mps, args.reference_time_limit, use_highs=True)
            reference = (value, source)
        else:
            reference = (None, "none")
        for tolerance in args.tolerances:
            legs = []
            if not args.skip_ours:
                legs.append(("sankhya-gpu", "pdhg-cuda"))
            if not args.skip_cuopt:
                legs.append(("cuopt", args.cuopt_method or "default"))
            for solver, method in legs:
                if (name, solver, f"{tolerance:.0e}") in done_keys:
                    continue
                with tempfile.TemporaryDirectory() as tmp:
                    sol = Path(tmp) / "solution.sol"
                    if solver == "sankhya-gpu":
                        result = run_ours(args.binary, mps, tolerance, args.time_limit, sol)
                        verified, at_tolerance = ("", ""), ("", "")
                        if sol.exists() and result["status"] in ("optimal", "feasible"):
                            verified = verify(mps, sol, None)
                            at_tolerance = verify(mps, sol, tolerance)
                    else:
                        result = run_cuopt(mps, tolerance, args.time_limit,
                                           args.cuopt_method, sol)
                        verified, at_tolerance = ("", ""), ("", "")
                        if sol.exists():
                            verified = verify(mps, sol, None)
                            at_tolerance = verify(mps, sol, tolerance)
                row = build_row(name, entry, digest, solver, method, tolerance,
                                args.time_limit, result, verified, at_tolerance, reference,
                                stamp_fields)
                rows.append(row)
                writer.writerow(row)
                handle.flush()
                print(f"{name:>20} {solver:>8} {tolerance:>6.0e} {row['status']:>15} "
                      f"{row['seconds']:>9.3f} {row['rel_gap'] or '-':>10} "
                      f"{str(row['independently_verified']):>8} "
                      f"{str(row['verified_at_tolerance']):>6}")
    if not rows:
        print("no results")
        return 1
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or RESULTS_DIR / f"lpfeas14-cuopt-{args.card or 'unknown'}-{commit}.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(main())
