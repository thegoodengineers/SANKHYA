#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Interleaved A/B of two SANKHYA binaries over the Netlib instances (#766).

A speed claim for a perf PR needs the base and the change measured on the same box in the
same minutes, not a new run held against an old CSV. So this runner alternates the two
binaries instance by instance, flipping which one goes first on every instance and every
repeat, and records every single solve. Drift in the machine (thermal, a neighbour's load)
then lands on both arms alike instead of on whichever one ran second.

Per solve it writes the ENGINEERING_RULES.md fields (instance, sha256, objective, published
objective, absolute and relative gap, status, wall and solver time, iterations, the binary's
own commit, machine), plus `arm` and `repeat`. The summary takes, per instance and arm, the
median solver time over the repeats, and the shifted geometric mean with a 10 s shift over
the instances, a run that is not counted charged the full time limit. Counted means what
compare_suite.py's `counted_for_time` means: within 1e-6 of Koch's exact optimum
(data/netlib/koch_exact.json, objective-row constant excluded) and accepted by the verifier.
That is the convention of docs/BENCHMARKS.md section 4a (bench/runners/head_to_head_doc.py),
so the SGM is comparable with the HiGHS column of the head-to-head CSV.

The independent verifier (tools/verify_solution.py) runs on the first repeat of each arm; the
later repeats are timing only and are marked `verified` empty.

Usage:
    python bench/runners/netlib_ab.py --base /path/main/build/sankhya \\
        --new /path/pr/build/sankhya --repeats 3 --out-prefix bench/results/ab-netlib-pr825
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import netlib  # noqa: E402
import stamp  # noqa: E402

SHIFT_SECONDS = 10.0
GENERATED = ("truss", "stocfor3")  # Fortran generators on netlib.org, not EMPS files (#745)
RUN_FIELDS = ["arm", "repeat", "instance", "instance_sha256", "status", "our_objective",
              "published_objective", "absolute_gap", "relative_gap", "matches_published",
              "exact_objective", "exact_relative_gap", "matches_exact", "independently_verified",
              "wall_seconds", "solver_seconds", "iterations",
              "time_limit", "solver_options", "git_commit", "machine", "timestamp_utc"]
SUMMARY_FIELDS = ["instance", "instance_sha256", "status_base", "status_new", "counted_base",
                  "counted_new", "iterations_base", "iterations_new",
                  "median_solver_seconds_base", "median_solver_seconds_new", "ratio",
                  "repeats", "commit_base", "commit_new", "machine"]


def sgm(values: list[float], shift: float = SHIFT_SECONDS) -> float:
    return math.exp(sum(math.log(v + shift) for v in values) / len(values)) - shift


def solve(binary: Path, name: str, published: float, exact: str | None, args,
          verify: bool) -> dict:
    mps = netlib.DATA_DIR / f"{name}.mps"
    options = [f"threads={args.threads}", *args.solver_option]
    blob = netlib.run_one(binary, mps, args.time_limit, verify, options)
    ours = blob.get("objective")
    gap = None if ours is None else abs(ours - published) / max(1.0, abs(published))
    matches = bool(blob["status"] == "optimal" and gap is not None
                   and gap <= netlib.PASS_RELATIVE_TOLERANCE
                   and blob.get("verified") is not False)
    offset = blob.get("objective_offset") or 0.0
    exact_gap = (None if ours is None or exact is None else
                 abs((ours - offset) - float(exact)) / max(1.0, abs(float(exact))))
    matches_exact = bool(blob["status"] == "optimal" and exact_gap is not None
                         and exact_gap <= netlib.PASS_RELATIVE_TOLERANCE
                         and blob.get("verified") is not False)
    return {
        "instance": name, "status": blob["status"],
        "our_objective": "" if ours is None else repr(ours),
        "published_objective": repr(published),
        "absolute_gap": "" if ours is None else repr(abs(ours - published)),
        "relative_gap": "" if gap is None else repr(gap),
        "matches_published": int(matches),
        "exact_objective": exact or "",
        "exact_relative_gap": "" if exact_gap is None else repr(exact_gap),
        "matches_exact": "" if exact is None else int(matches_exact),
        "independently_verified": "" if blob.get("verified") is None else int(blob["verified"]),
        "wall_seconds": round(blob.get("wall_seconds", 0.0), 6),
        "solver_seconds": blob.get("solver_seconds", ""),
        "iterations": blob.get("iterations", ""),
        "time_limit": args.time_limit, "solver_options": " ".join(options),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--new", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--instances", nargs="*")
    parser.add_argument("--solver-option", action="append", default=[])
    parser.add_argument("--machine", default=netlib.machine_tag())
    parser.add_argument("--out-prefix", required=True,
                        help="writes <prefix>-runs.csv and <prefix>-summary.csv")
    args = parser.parse_args()

    blob = json.loads((netlib.DATA_DIR / "reference.json").read_text())
    reference = blob["instances"]
    exact_path = netlib.DATA_DIR / "koch_exact.json"
    exact = json.loads(exact_path.read_text())["instances"] if exact_path.exists() else {}
    names = sorted(args.instances or [n for n in reference if n not in GENERATED])
    arms = {"base": args.base, "new": args.new}
    commits = {arm: stamp.binary_commit(path) or "unknown" for arm, path in arms.items()}
    shas = {n: netlib.sha256_file(netlib.DATA_DIR / f"{n}.mps") for n in names}
    print(f"base {commits['base']}  new {commits['new']}  {len(names)} instances, "
          f"{args.repeats} repeats, threads={args.threads}", flush=True)

    runs: list[dict] = []
    for repeat in range(1, args.repeats + 1):
        for index, name in enumerate(names):
            order = ("base", "new") if (index + repeat) % 2 else ("new", "base")
            for arm in order:
                row = solve(arms[arm], name, reference[name]["published_optimal"],
                            exact.get(name, {}).get("exact_objective"), args,
                            verify=repeat == 1)
                row.update(arm=arm, repeat=repeat, instance_sha256=shas[name],
                           git_commit=commits[arm], machine=args.machine,
                           timestamp_utc=datetime.datetime.now(datetime.timezone.utc)
                           .isoformat(timespec="seconds"))
                runs.append(row)
            print(f"r{repeat} {name:<10} " + "  ".join(
                f"{r['arm']} {r['status']} {r['solver_seconds']}" for r in runs[-2:]),
                flush=True)

    prefix = Path(args.out_prefix)
    with open(f"{prefix}-runs.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_FIELDS)
        writer.writeheader()
        writer.writerows(runs)

    summary, charged = [], {"base": [], "new": []}
    for name in names:
        row = {"instance": name, "instance_sha256": shas[name], "repeats": args.repeats,
               "commit_base": commits["base"], "commit_new": commits["new"],
               "machine": args.machine}
        for arm in arms:
            mine = [r for r in runs if r["instance"] == name and r["arm"] == arm]
            first = next(r for r in mine if r["repeat"] == 1)
            # compare_suite.py's counted_for_time: the exact grade where there is one.
            grade = first["matches_exact"] if first["matches_exact"] != "" else \
                first["matches_published"]
            ok = grade == 1 and first["independently_verified"] == 1
            seconds = statistics.median(float(r["solver_seconds"] or args.time_limit)
                                        for r in mine)
            charged[arm].append(seconds if ok else args.time_limit)
            row.update({f"status_{arm}": first["status"], f"counted_{arm}": int(ok),
                        f"iterations_{arm}": first["iterations"],
                        f"median_solver_seconds_{arm}": round(seconds, 6)})
        base, new = row["median_solver_seconds_base"], row["median_solver_seconds_new"]
        row["ratio"] = round(new / base, 4) if base > 0 else ""
        summary.append(row)
    with open(f"{prefix}-summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)

    for arm in arms:
        print(f"{arm:<5} {commits[arm]}  counted {sum(r[f'counted_{arm}'] for r in summary)}"
              f"/{len(summary)}  SGM (shift {SHIFT_SECONDS:g} s) "
              f"{sgm(charged[arm]):.4f} s")
    print(f"new/base SGM ratio {sgm(charged['new']) / sgm(charged['base']):.4f}")
    print(f"wrote {prefix}-runs.csv and {prefix}-summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
