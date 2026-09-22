#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The per-heuristic MIPLIB A/B (#414): one leg per heuristic, at one commit, in one sitting.

Every leg runs bench/runners/miplib.py over the same 30 instances at the same time limit,
with exactly one thing different from the defaults, and writes a CSV whose NAME says what
that thing was:

    bench/results/miplib-<sha>.csv                 the baseline, nothing set
    bench/results/miplib-heur-<leg>.csv            one heuristic (or one dive setting) on

so the legs can be read against the baseline by bench/runners/make_benchmarks_doc.py and
by anyone with a spreadsheet. A leg is only evidence when the tree is clean and every CSV
carries the same commit stamp, which this script refuses to violate: a modified tracked
file stops it before the first solve.

Run it alone on the machine, on mains, and do nothing else while it runs: the legs take a
few hours between them, and a loaded machine moves the time-limit rows.

    python bench/runners/miplib_heuristics_ab.py                 every leg, 60 s each
    python bench/runners/miplib_heuristics_ab.py --legs rens dive-guided
    python bench/runners/miplib_heuristics_ab.py --summary       read the CSVs, run nothing
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
RUNNER = REPO_ROOT / "bench" / "runners" / "miplib.py"

# Leg name -> the solver options that leg sets. Each is one heuristic's own switch, so the
# leg measures that heuristic alone against the defaults (#414's discipline point: one
# switch, one A/B). The dive-tree leg keeps the default fractional dive and runs it below
# the root too; dive-backtrack keeps it at the root and adds the one-level backtrack; all
# turns on the master switch, which is every heuristic at once.
LEGS: dict[str, list[str]] = {
    "lock-rounding": ["mip_heur_lock_rounding=on"],
    "repair": ["mip_heur_repair=on"],
    "pump": ["mip_heur_pump=on"],
    "rins": ["mip_heur_rins=on"],
    "rens": ["mip_heur_rens=on"],
    "dive-coefficient": ["mip_heur_dive_coefficient=on"],
    "dive-vector-length": ["mip_heur_dive_vector_length=on"],
    "dive-guided": ["mip_heur_dive_guided=on"],
    "dive-backtrack": ["mip_dive_backtrack=true"],
    "dive-tree": ["mip_dive_frequency=50"],
    "all": ["mip_heuristics=true"],
}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=False).stdout.strip()


def tree_is_clean() -> bool:
    """Modified tracked files, ignoring the tier manifests the fetch scripts rewrite."""
    status = git("status", "--porcelain", "--untracked-files=no", "--", ".",
                 ":!data/netlib/reference.json", ":!data/mittelmann/reference.json")
    return not status.strip()


def leg_path(leg: str, commit: str) -> Path:
    return RESULTS_DIR / (f"miplib-{commit}.csv" if leg == "baseline"
                          else f"miplib-heur-{leg}.csv")


def run_leg(leg: str, options: list[str], time_limit: float, commit: str) -> int:
    command = [sys.executable, str(RUNNER), "--time-limit", str(time_limit)]
    if leg != "baseline":
        command += ["--out", str(leg_path(leg, commit).relative_to(REPO_ROOT))]
    for option in options:
        command += ["--solver-option", option]
    print(f"=== leg {leg}: {' '.join(options) or '(defaults)'}", flush=True)
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def read_csv(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["instance"]: row for row in csv.DictReader(handle)}


def as_float(row: dict, key: str) -> float | None:
    raw = (row.get(key) or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def distance(row: dict) -> float | None:
    """How far the incumbent is from the published optimum, relative; None with no point."""
    ours = as_float(row, "our_objective")
    published = as_float(row, "published_objective")
    if ours is None or published is None:
        return None
    return abs(ours - published) / max(1.0, abs(published))


def compare(base: dict[str, dict], leg: dict[str, dict]) -> dict:
    """Matched and proved counts, incumbents moved either way, and the node ratio over the
    instances that end with the same status - the same reading the cuts A/B uses."""
    common = [n for n in base if n in leg]
    matched = sum(int(leg[n].get("matched_published") or 0) for n in common)
    proved = sum(int(leg[n].get("proved_optimal") or 0) for n in common)
    better, worse = [], []
    for n in common:
        d0, d1 = distance(base[n]), distance(leg[n])
        if d0 is None and d1 is None:
            continue
        if d1 is None:
            worse.append(n)
        elif d0 is None or d1 < d0 - 1e-9:
            better.append(n)
        elif d1 > d0 + 1e-9:
            worse.append(n)
    same = [n for n in common if base[n]["status"] == leg[n]["status"]]

    def nodes(row: dict) -> int:
        try:
            return int(row.get("nodes") or 0)
        except ValueError:
            return 0

    nodes_base = sum(nodes(base[n]) for n in same)
    nodes_leg = sum(nodes(leg[n]) for n in same)
    ratio = nodes_leg / nodes_base if nodes_base else float("nan")
    stamps = sorted({r.get("git_commit", "") for r in list(base.values()) + list(leg.values())})
    return {"instances": len(common), "matched": matched, "proved": proved, "better": better,
            "worse": worse, "node_ratio": ratio, "same": len(same), "stamps": stamps}


def summary(commit: str) -> int:
    base_path = leg_path("baseline", commit)
    if not base_path.exists():
        print(f"no baseline {base_path.name}; run the baseline leg first", file=sys.stderr)
        return 1
    base = read_csv(base_path)
    matched0 = sum(int(r.get("matched_published") or 0) for r in base.values())
    proved0 = sum(int(r.get("proved_optimal") or 0) for r in base.values())
    print(f"baseline {base_path.name}: {matched0} matched, {proved0} proved, "
          f"{len(base)} instances")
    print(f"{'leg':<20}{'matched':>8}{'proved':>7}{'better':>7}{'worse':>6}{'nodes':>8}  "
          f"instances moved")
    print("-" * 100)
    for leg in LEGS:
        path = leg_path(leg, commit)
        if not path.exists():
            print(f"{leg:<20}{'not run':>8}")
            continue
        c = compare(base, read_csv(path))
        stamp = "" if len(c["stamps"]) == 1 and not c["stamps"][0].endswith("-dirty") else \
            f"  NOT ONE MEASUREMENT: stamps {c['stamps']}"
        moved = "; ".join([f"+{n}" for n in c["better"]] + [f"-{n}" for n in c["worse"]])
        print(f"{leg:<20}{c['matched']:>8}{c['proved']:>7}{len(c['better']):>7}"
              f"{len(c['worse']):>6}{c['node_ratio']:>8.3f}  {moved}{stamp}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--legs", nargs="*", default=None,
                        help="legs to run, from: baseline " + " ".join(LEGS))
    parser.add_argument("--summary", action="store_true",
                        help="read the CSVs already in bench/results and print the table")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="run on a modified tree anyway (the CSVs are then stamped "
                             "-dirty and are not evidence)")
    args = parser.parse_args()

    commit = git("rev-parse", "--short", "HEAD")
    if args.summary:
        return summary(commit)
    if not tree_is_clean() and not args.allow_dirty:
        print("the tree has modified tracked files; a leg measured here would be stamped "
              "-dirty and could not be compared. Commit or stash first.", file=sys.stderr)
        return 1

    legs = args.legs or ["baseline", *LEGS]
    unknown = [leg for leg in legs if leg != "baseline" and leg not in LEGS]
    if unknown:
        print(f"unknown legs: {unknown}", file=sys.stderr)
        return 2
    for leg in legs:
        code = run_leg(leg, [] if leg == "baseline" else LEGS[leg], args.time_limit, commit)
        if code != 0:
            print(f"leg {leg} failed with exit code {code}", file=sys.stderr)
            return code
    return summary(commit)


if __name__ == "__main__":
    sys.exit(main())
