#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PDHG feasibility polishing off against on, stopped at an iteration cap (#483).

`pdhg_feasibility_polish` (#915) polishes the point PDHG is about to hand back when it is not
primal feasible to 1e-8. Its effect is easiest to see on a run that stops short, so every
solve here is `algorithm=pdhg` with `iteration_limit` (300 by default) and the two arms
differ only in that option. `pdhg_polish=false` in both arms: the default interior-point
finish (#229) would otherwise take over the point at the limit, and the row would measure
the interior point, not the feasibility polish.

Per row, from the CLI's --stats JSON:

- status, termination_reason, iterations (polish steps included, so it can exceed the cap);
- the engine's own relative residuals of the reported point (`engine_relative_primal`,
  `_dual`, `_gap`), read from the result message the engine writes, which is the only place
  the JSON carries them; the message's feasibility polish clause gives the number of polish
  phases that ran (kept or not), their steps and whether the reported point is a polished
  one;
- the absolute `primal_infeasibility`, `dual_infeasibility` and `complementarity_violation`
  of the `quality` block, measured on the original model. The scaled pair of
  sankhya::Solution (`primal_infeasibility_scaled`, `dual_infeasibility_scaled`) is not in the
  JSON, so it is not here;
- relative_gap, absolute_gap, dual_bound and the objective, beside the published optimum
  (Netlib's, or the analytic optimum generate_refinery_lp.py writes into its file);
- tools/verify_solution.py on the written .sol, at its defaults: exit 0 VERIFIED, 1
  REJECTED, 3 NOT A SOLUTION (every check passed but the stated infeasibility is above the
  1e-4 ceiling, #461). Its worst row violation, absolute and relative to the row's terms
  (the same scale as primal_infeasibility_scaled), is recorded too, recomputed from the
  original file by code that shares nothing with the solver.

Usage:

    python bench/runners/pdhg_polish_ab.py --binary build/sankhya.exe

Writes bench/results/feasibility-polish-ab-netlib-refinery-<commit>.csv (or --out), a name
no glob of make_benchmarks_doc.py or latest_result.py reads.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import gpu_arms  # noqa: E402  (analytic_optimum: the refinery generator's optimum comment)
import netlib  # noqa: E402  (shared helpers: sha256, stamp, machine tag, binary lookup)

REPO_ROOT = netlib.REPO_ROOT
DATA_DIR = netlib.DATA_DIR
DEFAULT_INSTANCES = ["afiro", "sc50a", "adlittle", "blend", "share2b"]
REFINERY = REPO_ROOT / "bench" / "runners" / "generate_refinery_lp.py"
ARMS = (("off", "pdhg_feasibility_polish=false"), ("on", "pdhg_feasibility_polish=true"))

COLUMNS = [
    "instance", "instance_sha256", "arm", "iteration_limit", "status", "termination_reason",
    "iterations", "feasibility_polishes", "feasibility_polish_iterations",
    "reported_point_polished", "engine_relative_primal", "engine_relative_dual",
    "engine_relative_gap", "primal_infeasibility", "dual_infeasibility",
    "complementarity_violation", "relative_gap", "absolute_gap", "dual_bound", "objective",
    "published_objective", "objective_relative_error", "verifier_exit", "verifier_verdict",
    "independently_verified", "verifier_row_violation", "verifier_row_violation_relative",
    "verifier_failures", "solve_seconds", "wall_seconds", "algorithm", "git_commit", "machine",
    "timestamp_utc", "solver_options",
]

_NUMBER = r"([-+]?(?:\d+(?:\.\d*)?(?:e[-+]?\d+)?|inf|nan))"
_RESIDUALS = re.compile(rf"relative primal {_NUMBER}, dual {_NUMBER}, gap {_NUMBER}")
_POLISH = re.compile(r"feasibility polish \(#483\): (\d+) polishes, (\d+) of (\d+) "
                     r"iterations; reported point (polished|not polished)")
_ROW = re.compile(r"\[(?:PASS|FAIL)\] row activity.*?worst violation (\S+) \((\S+) relative")
VERDICTS = {0: "VERIFIED", 1: "REJECTED", 2: "UNREADABLE", 3: "NOT A SOLUTION"}


def parse_message(message: str) -> dict:
    """The engine's relative residuals of the reported point and the polish clause, out of
    the result message pdhg.cpp writes. The LAST residual triple is the reported point's:
    with polishing on, the clause after the stop reason restates them."""
    out = {"engine_relative_primal": "", "engine_relative_dual": "", "engine_relative_gap": "",
           "feasibility_polishes": "", "feasibility_polish_iterations": "",
           "reported_point_polished": ""}
    triples = _RESIDUALS.findall(message or "")
    if triples:
        primal, dual, gap = triples[-1]
        out.update(engine_relative_primal=primal, engine_relative_dual=dual,
                   engine_relative_gap=gap)
    polish = _POLISH.search(message or "")
    if polish:
        out.update(feasibility_polishes=int(polish.group(1)),
                   feasibility_polish_iterations=int(polish.group(2)),
                   reported_point_polished=int(polish.group(4) == "polished"))
    return out


def parse_verifier(returncode: int, stdout: str) -> dict:
    row = _ROW.search(stdout or "")
    failing = [line.strip() for line in (stdout or "").splitlines() if "[FAIL]" in line]
    return {
        "verifier_exit": returncode,
        "verifier_verdict": VERDICTS.get(returncode, f"exit {returncode}"),
        "independently_verified": int(returncode == 0),
        "verifier_row_violation": row.group(1) if row else "",
        "verifier_row_violation_relative": row.group(2) if row else "",
        "verifier_failures": "; ".join(failing)[:300],
    }


def relative_error(value, reference) -> str:
    if value is None or reference is None:
        return ""
    return repr(abs(value - reference) / max(1.0, abs(reference)))


def generate_refinery(periods: int, seed: int, directory: Path) -> Path:
    out = directory / f"refinery-{periods}.mps"
    subprocess.run([sys.executable, str(REFINERY), "--periods", str(periods), "--seed",
                    str(seed), "--out", str(out)], check=True, capture_output=True, text=True)
    return out


def run(binary: Path, mps: Path, options: list[str], time_limit: float) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        stats, sol = Path(tmp) / "s.json", Path(tmp) / "s.sol"
        command = [str(binary), "solve", str(mps), "--stats", str(stats), "--write-sol",
                   str(sol), "--time-limit", str(time_limit)]
        for option in options:
            command += ["--option", option]
        started = time.perf_counter()
        subprocess.run(command, capture_output=True, text=True)
        wall = time.perf_counter() - started
        if not stats.exists():
            return {"status": "no_output", "wall_seconds": round(wall, 6)}
        blob = json.loads(stats.read_text())
        result, quality = blob.get("result", {}), blob.get("quality", {})
        effort, limits = blob.get("effort", {}), blob.get("limits", {})
        out = {
            "status": result.get("status", "unknown"),
            "termination_reason": limits.get("termination_reason", ""),
            "iterations": effort.get("iterations", ""),
            "primal_infeasibility": quality.get("primal_infeasibility", ""),
            "dual_infeasibility": quality.get("dual_infeasibility", ""),
            "complementarity_violation": quality.get("complementarity_violation", ""),
            "relative_gap": result.get("relative_gap", ""),
            "absolute_gap": result.get("absolute_gap", ""),
            "dual_bound": result.get("dual_bound", ""),
            "objective": result.get("objective", ""),
            "solve_seconds": effort.get("solve_seconds", ""),
            "wall_seconds": round(wall, 6),
            "algorithm": result.get("algorithm", ""),
            **parse_message(result.get("message", "")),
        }
        if sol.exists():
            check = subprocess.run([sys.executable, str(netlib.VERIFIER), str(mps), str(sol)],
                                   capture_output=True, text=True)
            out.update(parse_verifier(check.returncode, check.stdout))
        return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--instances", nargs="*", default=DEFAULT_INSTANCES, metavar="NAME",
                        help="Netlib instances (default: %(default)s)")
    parser.add_argument("--refinery", type=int, action="append", default=None, metavar="T",
                        help="add the generated refinery LP at T periods; repeatable "
                             "(default: 12)")
    parser.add_argument("--refinery-seed", type=int, default=7)
    parser.add_argument("--iteration-limit", type=int, default=300)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--solver-option", action="append", default=[], metavar="KEY=VALUE",
                        help="pass --option KEY=VALUE to every solve in both arms")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    binary = args.binary or netlib.default_binary()
    reference = json.loads((DATA_DIR / "reference.json").read_text())["instances"]
    unknown = [n for n in args.instances if n not in reference]
    missing = [n for n in args.instances if not (DATA_DIR / f"{n}.mps").exists()]
    if unknown or missing:
        raise SystemExit("not in data/netlib (run bench/runners/fetch_data.py): "
                         + ", ".join(sorted(set(unknown + missing))))
    commit, machine = netlib.git_commit(args.binary), netlib.machine_tag()
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    base = ["log_to_console=false", "algorithm=pdhg",
            f"iteration_limit={args.iteration_limit}", "pdhg_polish=false",
            *args.solver_option]

    instances = [(n, DATA_DIR / f"{n}.mps", reference[n]["published_optimal"])
                 for n in args.instances]
    scratch = tempfile.TemporaryDirectory()
    for periods in (args.refinery if args.refinery is not None else [12]):
        mps = generate_refinery(periods, args.refinery_seed, Path(scratch.name))
        instances.append((f"refinery-{periods}", mps, gpu_arms.analytic_optimum(mps)))

    print(f"solver {binary}  commit {commit}  iteration_limit {args.iteration_limit}")
    rows = []
    for name, mps, published in instances:
        sha = netlib.sha256_file(mps)
        for arm, option in ARMS:
            options = [*base, option]
            r = run(binary, mps, options, args.time_limit)
            objective = netlib.as_number(r.get("objective"))
            rows.append({
                "instance": name, "instance_sha256": sha, "arm": arm,
                "iteration_limit": args.iteration_limit, **r,
                "objective": "" if objective is None else repr(objective),
                "published_objective": "" if published is None else repr(published),
                "objective_relative_error": relative_error(objective, published),
                "git_commit": commit, "machine": machine, "timestamp_utc": stamp,
                "solver_options": " ".join(options),
            })
            row = rows[-1]
            print(f"{name:<12} {arm:<4} {row['status']:<16} it {row['iterations']!s:>6} "
                  f"rel primal {row.get('engine_relative_primal', '')!s:>10} "
                  f"dual {row.get('engine_relative_dual', '')!s:>10} "
                  f"gap {row.get('engine_relative_gap', '')!s:>10} "
                  f"row viol rel {row.get('verifier_row_violation_relative', '')!s:>10} "
                  f"{row.get('verifier_verdict', '-')}")
    scratch.cleanup()

    out = args.out or (netlib.RESULTS_DIR /
                       f"feasibility-polish-ab-netlib-refinery-{commit}.csv")
    out = (REPO_ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {netlib.display_path(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
