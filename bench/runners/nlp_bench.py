#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The nonlinear engine on a set of .nl models, verified, to a results CSV (NLP stages 2-3).

    python bench/runners/nlp_bench.py [--data data/nlp/hs] [--binary build/sankhya]
                                      [--time-limit 60] [--match-tolerance 1e-6] [--out CSV]

The sets: data/nlp/hs, the Hock-Schittkowski problems (bench/runners/hs_mod_to_nl.py says
where they come from), and data/nlp/minlplib, convex MINLPs from MINLPLib with their published
primal bounds (match them at the MIP gap target, --match-tolerance 1e-4). For every problem in
the set's REFERENCE.csv: `sankhya solve <problem>.nl --write-sol`, then tools/verify_solution.py on
the model and the answer - the independent checker, with its own .nl reader and its own
derivatives - and one CSV row with the columns ENGINEERING_RULES.md asks for: instance,
sha256 of the instance file, our objective, the published reference objective, the absolute
and relative gap, status, wall time, iterations, git commit (from the binary, stamp.py) and
machine, plus whether the checker verified the answer and whether it matches the published
value to --match-tolerance relative.

A local optimum that is not the published one is a legitimate outcome of a local method and
is recorded as `match` = no, never hidden; `optimal` (a global claim, made only for a model
proved convex) that is worse than the published value would be a wrong answer.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import stamp

REPO_ROOT = Path(__file__).resolve().parents[2]
FIELDS = ["instance", "sha256", "status", "objective", "reference_objective", "abs_gap",
          "rel_gap", "match", "verified", "seconds", "iterations", "git_commit", "machine",
          "time_limit", "timestamp_utc"]


def field(stdout: str, name: str) -> str:
    for line in stdout.splitlines():
        if line.startswith(name + " ") or line.startswith(name + "\t"):
            return line[len(name):].strip()
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=REPO_ROOT / "build" / "sankhya")
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "nlp" / "hs")
    parser.add_argument("--match-tolerance", type=float, default=1e-6,
                        help="relative gap to the published objective that counts as a match")
    args = parser.parse_args()
    data_dir = args.data
    commit = stamp.stamp(args.binary)
    machine = f"{platform.system()}-{platform.machine()}"
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    out = args.out or REPO_ROOT / "bench" / "results" / f"nlp-{data_dir.name}-{commit}-{platform.node()}.csv"
    rows = list(csv.DictReader(open(data_dir / "REFERENCE.csv", encoding="utf-8")))
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for ref in rows:
            name = ref["problem"]
            nl = data_dir / f"{name}.nl"
            sol = Path(tmp) / f"{name}.sol"
            started = time.monotonic()
            run = subprocess.run([str(args.binary), "solve", str(nl), "--write-sol", str(sol),
                                  "--option", f"time_limit={args.time_limit}"],
                                 capture_output=True, text=True, check=False)
            seconds = time.monotonic() - started
            status = field(run.stdout, "status") or "crashed"
            objective = field(run.stdout, "objective")
            reference = float(ref["reference_objective"])
            verified = "no point"
            if sol.exists() and status in ("optimal", "locally_optimal", "feasible"):
                check = subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "verify_solution.py"),
                                        str(nl), str(sol), "--quiet"],
                                       capture_output=True, text=True, check=False)
                verified = "yes" if check.returncode == 0 else "REJECTED"
            abs_gap = rel_gap = ""
            match = "no"
            if objective:
                gap = abs(float(objective) - reference)
                abs_gap, rel_gap = f"{gap:.6e}", f"{gap / max(1.0, abs(reference)):.6e}"
                if status in ("optimal", "locally_optimal") and float(rel_gap) <= args.match_tolerance:
                    match = "yes"
            results.append({
                "instance": name, "sha256": hashlib.sha256(nl.read_bytes()).hexdigest(),
                "status": status, "objective": objective, "reference_objective": ref["reference_objective"],
                "abs_gap": abs_gap, "rel_gap": rel_gap, "match": match, "verified": verified,
                "seconds": f"{seconds:.3f}", "iterations": field(run.stdout, "iterations"),
                "git_commit": commit, "machine": machine, "time_limit": args.time_limit,
                "timestamp_utc": now})
            print(f"{name:<8} {status:<20} {objective:>22} {ref['reference_objective']:>22} "
                  f"match={match:<3} verified={verified}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)
    matched = sum(r["match"] == "yes" for r in results)
    rejected = sum(r["verified"] == "REJECTED" for r in results)
    print(f"\nmatched {matched} of {len(results)}; verifier rejected {rejected}; wrote {out}")
    return 1 if rejected else 0


if __name__ == "__main__":
    sys.exit(main())
