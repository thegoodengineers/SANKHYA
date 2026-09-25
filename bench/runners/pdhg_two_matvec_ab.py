#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PDHG per-iteration time with two sparse products per iteration against three (#479).

`pdhg_two_matvec` keeps A x_k from the previous step, computes A x_{k+1} once and derives
A xbar and A dx by vector arithmetic, so an iteration takes A'y and A x_{k+1} instead of
A'y, A xbar and A dx. This measures what that is worth per iteration, on the CPU engine and
on the CUDA engine, each option value against the other on the same binary: a FIXED
iteration count per instance (pdhg_tolerance at its 1e-14 floor, so a run stops on the count
unless it genuinely converges first), PDHG alone, the median of --repeats solves per arm,
the arms interleaved within each repeat so a drifting machine load lands on both.

    python bench/runners/pdhg_two_matvec_ab.py --binary build/sankhya
    python bench/runners/pdhg_two_matvec_ab.py --binary build/sankhya --no-cuda
    python bench/runners/pdhg_two_matvec_ab.py --binary build/sankhya --no-cpu
        --solver-option gpu_on_device_loop=true      (one command line)

Instances: the nine committed Netlib instances (data/netlib, published optimum from
data/netlib/reference.json) and the synthetic KKT ladder of gpu_report.py (analytic optimum),
where the products are the cost of an iteration. `--mps PATH` adds any other file (no
reference). The objective after a fixed iteration count is not a converged answer; the
reference and gap columns say how far each arm got, and are there so a row that differs
between the arms by more than rounding is visible.

Output: bench/results/pdhg-two-matvec-<sha>.csv (the sha from the binary, #433), one row per
instance, engine and option value, with the columns ENGINEERING_RULES.md requires: instance,
sha256 of the instance file, objective, reference objective, absolute and relative gap,
status, solver and wall seconds, iterations, git commit, machine. seconds_per_iteration is
the MARGINAL solver time per iteration: each arm is also run at a fifth of the count, and
the median over repeats of (t - t_short) / (k - k_short) is reported, so the one-off set-up
(on the device the context and the upload are most of a short solve) cancels. An instance
that converges before twice the short count has no marginal time and says so.
ratio_to_three_products is the two-product row's per-iteration time over the
three-product row's.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import platform
import re
import statistics
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
NETLIB_DIR = REPO_ROOT / "data" / "netlib"

NETLIB_NINE = ["afiro", "sc50a", "sc50b", "adlittle", "blend", "share2b", "sc105", "stocfor1",
               "israel"]

COLUMNS = ["instance", "instance_sha256", "rows", "cols", "nnz", "engine", "two_matvec",
           "iterations_requested", "iterations", "status", "objective", "reference_objective",
           "absolute_gap", "relative_gap", "solver_seconds", "wall_seconds",
           "short_iterations", "short_solver_seconds", "seconds_per_iteration", "ratio_to_three_products", "repeats", "algorithm_ran",
           "solver_options", "git_commit", "machine", "gpu", "timestamp_utc"]

DEFAULT_ITERATIONS = 10000

# --solver-option values, passed to every solve of both arms (e.g. gpu_on_device_loop=true)
# and recorded in the solver_options column.
EXTRA_OPTIONS: list[str] = []


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_description(binary: Path) -> str:
    result = subprocess.run([str(binary), "--version"], capture_output=True, text=True,
                            check=False)
    match = re.search(r"GPU ([^)]*\))", result.stdout)
    return match.group(1).strip() if match else ""


def solve(binary: Path, mps: Path, cuda: bool, two_matvec: bool, iterations: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        stats = Path(tmp) / "s.json"
        command = [str(binary), "solve", str(mps), "--stats", str(stats)]
        for option in ("log_to_console=false", "algorithm=pdhg", "pdhg_polish=false",
                       "presolve=false", "pdhg_tolerance=1e-14",
                       f"iteration_limit={iterations}",
                       f"pdhg_two_matvec={'true' if two_matvec else 'false'}",
                       f"gpu={'true' if cuda else 'false'}", *EXTRA_OPTIONS):
            command += ["--option", option]
        started = time.perf_counter()
        subprocess.run(command, capture_output=True, text=True, check=False)
        wall = time.perf_counter() - started
        if not stats.exists():
            return {"status": "no_output", "objective": None, "iterations": 0,
                    "seconds": wall, "wall": wall, "algorithm": "", "rows": "", "cols": "",
                    "nnz": ""}
        blob = json.loads(stats.read_text())
        result = blob.get("result", {})
        effort = blob.get("effort", {})
        model = blob.get("model", {})
        objective = result.get("objective")
        return {
            "status": result.get("status", "unknown"),
            "objective": float(objective) if isinstance(objective, (int, float)) else None,
            "iterations": int(effort.get("iterations", 0) or 0),
            "seconds": float(effort.get("solve_seconds", wall)),
            "wall": wall,
            "algorithm": result.get("algorithm", ""),
            "rows": model.get("rows", ""),
            "cols": model.get("columns", ""),
            "nnz": model.get("nonzeros", ""),
        }


def netlib_references() -> dict[str, float]:
    blob = json.loads((NETLIB_DIR / "reference.json").read_text())
    return {name: float(entry["published_optimal"])
            for name, entry in blob["instances"].items()
            if entry.get("published_optimal") is not None}


def measure(binary: Path, name: str, mps: Path, reference: float | None, engines: list[bool],
            iterations: int, repeats: int, common: dict) -> list[dict]:
    digest = sha256_file(mps)
    short = max(1, iterations // 5)
    rows = []
    for cuda in engines:
        if cuda:
            solve(binary, mps, True, False, short)  # device warm-up
        runs: dict[bool, list[tuple[dict, dict]]] = {False: [], True: []}
        for _ in range(repeats):
            for two in (False, True):  # interleaved: A B A B
                runs[two].append((solve(binary, mps, cuda, two, short),
                                  solve(binary, mps, cuda, two, iterations)))
        per_iteration: dict[bool, float | None] = {}
        for two in (False, True):
            done = runs[two]
            last = done[-1][1]
            # The MARGINAL time per iteration, (t_long - t_short) / (k_long - k_short), per
            # repeat: the set-up both runs pay once (reading, scaling, the power iteration,
            # and on the device the context, the upload and the cuSPARSE buffers) cancels.
            # On the device the set-up is most of a short solve, so the plain quotient t / k
            # would measure the set-up, not the iteration.
            marginal = [(long_run["seconds"] - short_run["seconds"])
                        / (long_run["iterations"] - short_run["iterations"])
                        for short_run, long_run in done
                        if long_run["iterations"] - short_run["iterations"] >= short]
            per_iteration[two] = statistics.median(marginal) if marginal else None
            seconds = statistics.median(long_run["seconds"] for _, long_run in done)
            short_seconds = statistics.median(short_run["seconds"] for short_run, _ in done)
            wall = statistics.median(long_run["wall"] for _, long_run in done)
            count = last["iterations"]
            objective = last["objective"]
            absolute = relative = ""
            if objective is not None and reference is not None:
                absolute = abs(objective - reference)
                relative = absolute / max(1.0, abs(reference))
            rows.append({
                "instance": name, "instance_sha256": digest,
                "rows": last["rows"], "cols": last["cols"], "nnz": last["nnz"],
                "engine": "cuda" if cuda else "cpu", "two_matvec": str(two).lower(),
                "iterations_requested": iterations, "iterations": count,
                "status": last["status"],
                "objective": "" if objective is None else repr(objective),
                "reference_objective": "" if reference is None else repr(reference),
                "absolute_gap": absolute, "relative_gap": relative,
                "solver_seconds": round(seconds, 6), "wall_seconds": round(wall, 6),
                "short_iterations": done[-1][0]["iterations"],
                "short_solver_seconds": round(short_seconds, 6),
                "seconds_per_iteration": ("" if per_iteration[two] is None
                                          else f"{per_iteration[two]:.6e}"),
                "ratio_to_three_products": "", "repeats": repeats,
                "algorithm_ran": last["algorithm"], **common,
            })
        engine = "cuda" if cuda else "cpu"
        if per_iteration[False] and per_iteration[True]:
            ratio = per_iteration[True] / per_iteration[False]
            rows[-1]["ratio_to_three_products"] = f"{ratio:.4f}"
            rows[-2]["ratio_to_three_products"] = "1.0000"
            print(f"{name:>22}  {engine:>4}  {rows[-2]['iterations']:>6} / "
                  f"{rows[-1]['iterations']:>6} it  three {per_iteration[False] * 1e6:10.2f} us"
                  f"  two {per_iteration[True] * 1e6:10.2f} us  ratio {ratio:.3f}"
                  f"  ran {rows[-1]['algorithm_ran']}", flush=True)
        else:
            print(f"{name:>22}  {engine:>4}  converged before {2 * short} iterations "
                  f"({rows[-2]['iterations']} / {rows[-1]['iterations']}): no marginal time",
                  flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-cuda", action="store_true", help="the CPU engine only")
    parser.add_argument("--no-netlib", action="store_true")
    parser.add_argument("--no-ladder", action="store_true")
    parser.add_argument("--mps", type=Path, action="append", default=[],
                        help="another instance, no reference objective")
    parser.add_argument("--no-cpu", action="store_true", help="the CUDA engine only")
    parser.add_argument("--solver-option", action="append", default=[], metavar="KEY=VALUE",
                        help="passed to every solve of both arms, e.g. gpu_on_device_loop=true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.no_cuda and args.no_cpu:
        parser.error("--no-cuda and --no-cpu leave nothing to run")
    EXTRA_OPTIONS.extend(args.solver_option)

    commit = stamp.stamp(args.binary)
    gpu = "" if args.no_cuda else gpu_description(args.binary)
    common = {"git_commit": commit,
              "machine": f"{platform.system()}-{platform.machine()}",
              "gpu": gpu,
              "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "solver_options": " ".join(EXTRA_OPTIONS)}
    engines = ([] if args.no_cpu else [False]) + ([] if args.no_cuda else [True])
    print(f"binary {args.binary}  commit {commit}  gpu {gpu or 'none'}  "
          f"{args.iterations} iterations x {args.repeats} repeats  "
          f"options {' '.join(EXTRA_OPTIONS) or 'none'}", flush=True)

    rows: list[dict] = []
    if not args.no_netlib:
        references = netlib_references()
        for name in NETLIB_NINE:
            rows += measure(args.binary, name, NETLIB_DIR / f"{name}.mps",
                            references.get(name), engines, args.iterations, args.repeats,
                            common)
    with tempfile.TemporaryDirectory() as tmp:
        if not args.no_ladder:
            for m, n, per_col in SIZES:
                mps = Path(tmp) / f"kkt_{m}x{n}.mps"
                optimum = generate_lp(m, n, per_col, seed=42, out=mps)
                rows += measure(args.binary, f"kkt_{m}x{n}", mps, optimum, engines,
                                args.iterations, args.repeats, common)
        for mps in args.mps:
            rows += measure(args.binary, mps.name, mps, None, engines, args.iterations,
                            args.repeats, common)

    tag = re.sub(r"[^A-Za-z0-9]+", "-", " ".join(EXTRA_OPTIONS)).strip("-")
    out = args.out or RESULTS_DIR / f"pdhg-two-matvec-{commit}{'-' + tag if tag else ''}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
