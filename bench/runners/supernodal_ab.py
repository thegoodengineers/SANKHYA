#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Supernodal factorization A/B (#470): four interior-point arms over Kennington cre-a, cre-c,
ken-07, pds-02 and the refinery ladder at 365 and 1500 periods.

    base-scalar-t1   base binary, threads=1
    base-super-t1    base binary, threads=1, ipm_supernodal=true
    new-super-t1     new binary,  threads=1, ipm_supernodal=true
    new-super-t4     new binary,  threads=4, ipm_supernodal=true

With base and new the same build, base-super-t1 against new-super-t1 is a repeat of one arm,
the run's own noise floor. Every solve goes through the existing runner (kennington.py, which
verifies the solution with tools/verify_solution.py, or scale.py, which grades it against
the generator's analytic optimum) and writes its own CSV, with profile=detailed so the
factorization time and count come from the solver's profile JSON. One summary CSV
`supernodal-ab-<base>-vs-<new>.csv` then puts, per arm and instance: factor seconds,
factorizations, iterations, status and whether the answer was verified.

Usage:
    python bench/runners/supernodal_ab.py --base base/build/sankhya --new build/sankhya \\
        --time-limit 600 --machine "..."
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import stamp

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "bench" / "results"
KENNINGTON = ["cre-a", "cre-c", "ken-07", "pds-02"]
REFINERY_PERIODS = [365, 1500]
FIELDS = ["arm", "set", "instance", "status", "verified", "verified_by", "factor_seconds",
          "factorizations", "iterations", "solver_seconds", "time_limit", "git_commit",
          "machine", "timestamp_utc", "solver_options"]


def factor_profile(path: Path) -> dict:
    """Factorization seconds and calls summed over every region named `factorization`."""
    if not path.exists():
        return {"factor_seconds": "", "factorizations": ""}
    blob = json.loads(path.read_text(encoding="utf-8"))
    regions = [r for r in blob["regions"] if r["path"].endswith("factorization")]
    return {"factor_seconds": round(sum(r["inclusive_seconds"] for r in regions), 4),
            "factorizations": sum(r["calls"] for r in regions)}


def only_row(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--new", required=True, type=Path)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--instances", nargs="*", default=KENNINGTON)
    parser.add_argument("--periods", type=int, nargs="*", default=REFINERY_PERIODS)
    args = parser.parse_args()

    commits = {"base": stamp.binary_commit(args.base) or "unknown",
               "new": stamp.binary_commit(args.new) or "unknown"}
    arms = {"base-scalar-t1": ("base", ["threads=1"]),
            "base-super-t1": ("base", ["threads=1", "ipm_supernodal=true"]),
            "new-super-t1": ("new", ["threads=1", "ipm_supernodal=true"]),
            "new-super-t4": ("new", ["threads=4", "ipm_supernodal=true"])}
    limit = f"{args.time_limit:g}"
    work = Path(tempfile.mkdtemp(prefix="supernodal-ab-"))
    summary = []
    for arm, (side, options) in arms.items():
        binary, commit = (args.base if side == "base" else args.new), commits[side]
        jobs = [("kennington", name, f"supernodal-ab-kennington-{arm}-{name}-{commit}.csv")
                for name in args.instances]
        jobs += [("refinery", f"refinery-{t}", f"supernodal-ab-refinery-{arm}-{t}-{commit}.csv")
                 for t in args.periods]
        for family, name, csv_name in jobs:
            profile = work / f"{arm}-{name}.json"
            out = RESULTS / csv_name
            solver_options = options + ["profile=detailed", f"profile_out={profile}"]
            if family == "kennington":
                cmd = [sys.executable, "bench/runners/kennington.py", "--instances", name]
                solver_options.append("algorithm=ipm")
            else:
                cmd = [sys.executable, "bench/runners/scale.py", "--structure", "refinery",
                       "--sizes", name.split("-")[1], "--engines", "ipm",
                       "--keep", str(work)]
            cmd += ["--binary", str(binary), "--time-limit", limit, "--machine", args.machine,
                    "--out", str(out)]
            for option in solver_options:
                cmd += ["--solver-option", option]
            print(f"== {arm} {name}", flush=True)
            subprocess.run(cmd, cwd=REPO_ROOT, check=False)
            row = only_row(out) if out.exists() else {}
            if family == "kennington":
                verified, by = row.get("independently_verified", ""), "tools/verify_solution.py"
            else:
                verified, by = row.get("reached_optimum", ""), "analytic optimum (scale.py)"
            summary.append({
                "arm": arm, "set": family, "instance": name,
                "status": row.get("status", "no_csv"), "verified": verified, "verified_by": by,
                **factor_profile(profile),
                "iterations": row.get("iterations", ""),
                "solver_seconds": row.get("solver_seconds", ""), "time_limit": limit,
                "git_commit": commit, "machine": args.machine,
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
                .isoformat(timespec="seconds"),
                "solver_options": " ".join(options)})

    out = RESULTS / f"supernodal-ab-{commits['base']}-vs-{commits['new']}.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(f"{row['arm']:<15} {row['instance']:<15} {row['status']:<12} "
              f"verified {row['verified'] or '-':<2} factor {row['factor_seconds']!s:>10} s "
              f"x{row['factorizations']!s:<5} iters {row['iterations']}")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
