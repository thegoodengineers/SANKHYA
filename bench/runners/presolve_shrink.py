#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""How much presolve shrinks each Netlib instance, what it costs, and where a solve's
fixed overhead goes - one CSV row per instance, arm and repeat.

Per row: rows / columns / nonzeros before and after presolve, the count of every reduction
that fired, presolve's own seconds, and the solve split by the profiler's phases
(presolve, engine, postsolve, verification; `profile=basic`, #285). `overhead_seconds` is
the solver clock minus the engine, which is what a tiny instance pays whatever the
simplex does. `process_seconds` is wall minus the solver clock: process start, reading
the MPS file and writing the solution, none of which the head-to-head times.

Each row is graded exactly as bench/runners/netlib.py grades it: the published optimum
(objective-row constant excluded), Koch's exact optimum when fetched, and
tools/verify_solution.py on the written .sol. An arm that shrinks the model and breaks an
answer shows up here, not later.

Arms are option sets, so an A/B of a reduction is one command:

    python bench/runners/presolve_shrink.py --binary build/sankhya.exe \\
        --arm base --arm dual_fixing:presolve_dual_fixing=true --repeats 3

`--highs` adds a yardstick: the size HiGHS's presolve reaches on the same file, run as a
black box through the highspy wheel exactly as bench/runners/rivals.py runs it (its
binary, never its source), so the gap between the two reductions is measured rather than
guessed.

`--smallest N` keeps the N Netlib instances with the smallest MPS files, and `--refinery T`
(repeatable) adds the refinery planning LP of bench/runners/generate_refinery_lp.py at T
periods, generated into a temporary directory with `--refinery-seed` (7, the generator's
documented example) and graded against the analytic optimum the generator writes into the
file. The bound propagation A/B of #485 is one command:

    python bench/runners/presolve_shrink.py --binary build/sankhya.exe --time-limit 20         --smallest 30 --refinery 12 --refinery 365         --arm off:presolve_bound_propagation=false --arm on:presolve_bound_propagation=true         --out bench/results/bound-propagation-ab-netlib30-refinery-<commit>.csv

Writes bench/results/presolve-shrink-<commit>.csv (or --out).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import netlib  # noqa: E402  (shared helpers: sha256, stamp, binary lookup)

REPO_ROOT = netlib.REPO_ROOT
DATA_DIR = netlib.DATA_DIR
TOL = netlib.PASS_RELATIVE_TOLERANCE
PHASES = ("presolve", "engine", "postsolve", "verification")

COLUMNS = [
    "instance", "instance_sha256", "arm", "rep", "status", "our_objective",
    "published_objective", "relative_gap", "matches_published", "exact_objective",
    "matches_exact", "independently_verified",
    "rows_before", "rows_after", "cols_before", "cols_after", "nnz_before", "nnz_after",
    "presolve_passes", "presolve_seconds", "reductions",
    "highs_rows_after", "highs_cols_after", "highs_nnz_after", "highs_version",
    *(f"phase_{p}_seconds" for p in PHASES),
    "solver_seconds", "overhead_seconds", "wall_seconds", "process_seconds", "iterations",
    "algorithm", "git_commit", "machine", "timestamp_utc", "solver_options",
    # Appended (#485), so a reader of an older CSV by name is unaffected: the two counts the
    # bound propagation A/B is about, out of the `reductions` text, and why presolve stopped.
    "propagated_bounds", "implied_free_column_singletons", "presolve_termination",
]
REFINERY = REPO_ROOT / "bench" / "runners" / "generate_refinery_lp.py"


def parse_arm(text: str) -> tuple[str, list[str]]:
    name, _, rest = text.partition(":")
    return name, [o for o in rest.split(",") if o]


def smallest_by_size(names, size_of, count: int) -> list[str]:
    """The `count` names with the smallest files, ties broken by name, so the set is the
    same on every machine that has the same files."""
    return sorted(sorted(names, key=lambda n: (size_of(n), n))[:count])


def analytic_optimum(text: str) -> float | None:
    """The `* analytic optimum:` comment generate_refinery_lp.py writes into the MPS header
    (#211): c^T x* of the plan it built optimal, from exact rationals."""
    for line in text.splitlines()[:20]:
        match = re.match(r"\*\s*analytic optimum:\s*(\S+)", line)
        if match:
            return float(match.group(1))
    return None


def generate_refinery(periods: int, seed: int, directory: Path) -> Path:
    out = directory / f"refinery-{periods}.mps"
    subprocess.run([sys.executable, str(REFINERY), "--periods", str(periods), "--seed",
                    str(seed), "--out", str(out)], check=True, capture_output=True, text=True)
    return out


def phase_seconds(profile: dict) -> dict:
    out = {p: 0.0 for p in PHASES}
    for region in profile.get("regions", []):
        # Direct children of the root "solve" region only: an "engine" nested inside
        # another phase would be counted twice otherwise.
        parent, _, name = region.get("path", "").rpartition("/")
        if parent == "solve" and name in out:
            out[name] += float(region.get("inclusive_seconds", 0.0))
    return out


def run(binary: Path, mps: Path, options: list[str], time_limit: float) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        stats, sol, prof = (Path(tmp) / n for n in ("s.json", "s.sol", "p.json"))
        command = [str(binary), "solve", str(mps), "--stats", str(stats), "--write-sol",
                   str(sol), "--time-limit", str(time_limit), "--option",
                   "log_to_console=false", "--option", "profile=basic", "--option",
                   f"profile_out={prof}"]
        for option in options:
            command += ["--option", option]
        started = time.perf_counter()
        subprocess.run(command, capture_output=True, text=True)
        wall = time.perf_counter() - started
        if not stats.exists():
            return {"status": "no_output", "wall": wall}
        blob = json.loads(stats.read_text())
        profile = json.loads(prof.read_text()) if prof.exists() else {}
        result, effort = blob.get("result", {}), blob.get("effort", {})
        out = {
            "status": result.get("status", "unknown"),
            "objective": netlib.as_number(result.get("objective")),
            "offset": netlib.as_number(blob.get("model", {}).get("objective_offset")) or 0.0,
            "algorithm": result.get("algorithm", ""),
            "iterations": effort.get("iterations", ""),
            "solver_seconds": netlib.as_number(effort.get("solve_seconds")) or 0.0,
            "presolve": blob.get("presolve", {}),
            "phases": phase_seconds(profile),
            "wall": wall,
            "verified": None,
        }
        if sol.exists() and out["status"] in ("optimal", "feasible"):
            check = subprocess.run([sys.executable, str(netlib.VERIFIER), str(mps), str(sol),
                                    "--quiet"], capture_output=True, text=True)
            out["verified"] = check.returncode == 0
        return out


def highs_presolved(mps: Path) -> dict:
    """Rows, columns and nonzeros after HiGHS's presolve, one thread, as a black box."""
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.readModel(str(mps))
    h.presolve()
    lp = h.getPresolvedLp()
    return {"highs_rows_after": lp.num_row_, "highs_cols_after": lp.num_col_,
            "highs_nnz_after": len(lp.a_matrix_.value_),
            "highs_version": f"{h.versionMajor()}.{h.versionMinor()}.{h.versionPatch()}"}


def gap(value, reference):
    if value is None or reference is None:
        return None
    return abs(value - reference) / max(1.0, abs(reference))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--arm", action="append", default=[], metavar="NAME[:K=V,...]",
                        help="an option set to run; repeatable. Default: one arm, 'base'.")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--instances", nargs="*")
    parser.add_argument("--smallest", type=int, default=None, metavar="N",
                        help="only the N Netlib instances with the smallest MPS files")
    parser.add_argument("--refinery", type=int, action="append", default=[], metavar="T",
                        help="add the generated refinery LP at T periods; repeatable")
    parser.add_argument("--refinery-seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--highs", action="store_true",
                        help="add the size HiGHS's presolve reaches (needs highspy)")
    args = parser.parse_args()

    binary = args.binary or netlib.default_binary()
    reference = json.loads((DATA_DIR / "reference.json").read_text())["instances"]
    exact_path = DATA_DIR / "koch_exact.json"
    exact = json.loads(exact_path.read_text())["instances"] if exact_path.exists() else {}
    arms = [parse_arm(a) for a in (args.arm or ["base"])]
    commit, machine = netlib.git_commit(args.binary), netlib.machine_tag()
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    # Process start alone, so process_seconds can be read against it.
    starts = []
    for _ in range(5):
        began = time.perf_counter()
        subprocess.run([str(binary), "version"], capture_output=True)
        starts.append(time.perf_counter() - began)
    print(f"solver {binary}  commit {commit}  process start median "
          f"{statistics.median(starts) * 1e3:.1f} ms")

    names = [n for n in sorted(args.instances or reference)
             if (DATA_DIR / f"{n}.mps").exists()]
    if args.smallest is not None:
        names = smallest_by_size(names, lambda n: (DATA_DIR / f"{n}.mps").stat().st_size,
                                 args.smallest)
    # (name, file, published objective, exact objective text or None)
    instances = [(n, DATA_DIR / f"{n}.mps", reference[n]["published_optimal"],
                  exact.get(n, {}).get("exact_objective")) for n in names]
    scratch = tempfile.TemporaryDirectory()
    for periods in args.refinery:
        mps = generate_refinery(periods, args.refinery_seed, Path(scratch.name))
        instances.append((f"refinery-{periods}", mps,
                          analytic_optimum(mps.read_text(errors="replace")[:4096]), None))

    rows = []
    for name, mps, published, exact_text in instances:
        sha = netlib.sha256_file(mps)
        yardstick = highs_presolved(mps) if args.highs else {}
        # Interleave arms inside each repeat so drift in machine load hits every arm alike.
        for rep in range(1, args.repeats + 1):
            for arm, options in arms:
                r = run(binary, mps, options, args.time_limit)
                ours, offset = r.get("objective"), r.get("offset", 0.0)
                g = gap(ours, published)
                g_off = gap(None if ours is None else ours - offset, published)
                optimal = r["status"] == "optimal" and r.get("verified") is not False
                matches = optimal and any(x is not None and x <= TOL for x in (g, g_off))
                g_exact = gap(None if ours is None else ours - offset,
                              None if exact_text is None else float(exact_text))
                pre = r.get("presolve", {})
                phases = r.get("phases", {p: 0.0 for p in PHASES})
                fired = {k: v for k, v in pre.get("reductions", {}).items() if v}
                solver = r.get("solver_seconds", 0.0)
                rows.append({
                    "instance": name, "instance_sha256": sha, "arm": arm, "rep": rep,
                    "status": r["status"],
                    "our_objective": "" if ours is None else repr(ours),
                    "published_objective": "" if published is None else repr(published),
                    "relative_gap": "" if g is None else
                    repr(min(x for x in (g, g_off) if x is not None)),
                    "matches_published": int(matches),
                    "exact_objective": exact_text or "",
                    "matches_exact": "" if exact_text is None else
                    int(optimal and g_exact is not None and g_exact <= TOL),
                    "independently_verified": "" if r.get("verified") is None
                    else int(r["verified"]),
                    "rows_before": pre.get("rows", {}).get("before", ""),
                    "rows_after": pre.get("rows", {}).get("after", ""),
                    "cols_before": pre.get("columns", {}).get("before", ""),
                    "cols_after": pre.get("columns", {}).get("after", ""),
                    "nnz_before": pre.get("nonzeros", {}).get("before", ""),
                    "nnz_after": pre.get("nonzeros", {}).get("after", ""),
                    "presolve_passes": pre.get("passes", ""),
                    "presolve_seconds": pre.get("seconds", ""),
                    "reductions": " ".join(f"{k}={v}" for k, v in sorted(fired.items())),
                    **yardstick,
                    **{f"phase_{p}_seconds": round(phases[p], 6) for p in PHASES},
                    "solver_seconds": solver,
                    "overhead_seconds": round(solver - phases["engine"], 6),
                    "wall_seconds": round(r["wall"], 6),
                    "process_seconds": round(r["wall"] - solver, 6),
                    "iterations": r.get("iterations", ""),
                    "algorithm": r.get("algorithm", ""),
                    "git_commit": commit, "machine": machine, "timestamp_utc": stamp,
                    "solver_options": " ".join(options),
                    "propagated_bounds": pre.get("reductions", {}).get("propagated_bounds", ""),
                    "implied_free_column_singletons":
                    pre.get("reductions", {}).get("implied_free_column_singletons", ""),
                    "presolve_termination": pre.get("termination", ""),
                })
                row = rows[-1]
                print(f"{name:<10} {arm:<14} r{rep} {row['status']:<9} "
                      f"rows {row['rows_before']}->{row['rows_after']} "
                      f"cols {row['cols_before']}->{row['cols_after']} "
                      f"pre {float(row['presolve_seconds'] or 0) * 1e3:7.1f}ms "
                      f"solve {solver:8.3f}s it {row['iterations']} "
                      f"{'ok' if matches else 'MISS'}"
                      f"{'' if r.get('verified') is not False else ' REJECTED'}")

    out = args.out or (netlib.RESULTS_DIR / f"presolve-shrink-{commit}.csv")
    out = (REPO_ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, restval="")
        writer.writeheader()
        writer.writerows(rows)
    scratch.cleanup()
    print(f"wrote {netlib.display_path(out)}")
    rejected = sorted({r["instance"] for r in rows if r["independently_verified"] == 0})
    if rejected:
        print(f"REJECTED BY THE VERIFIER: {', '.join(rejected)}")
    return 1 if rejected else 0


if __name__ == "__main__":
    sys.exit(main())
