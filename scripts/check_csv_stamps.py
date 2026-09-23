#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify that every git_commit stamp in bench/results/*.csv is an ancestor of HEAD.

A CSV that names a commit must have been produced from that commit's tree.  If the commit
is not in main's history (e.g. because it lived only on a PR branch and was squash-merged),
the stamp cannot be used to reproduce the result on a fresh clone.  This check catches
that class of process bug before a PR lands.

Exit 0 when all stamps pass.  Exit 1 and print the offending files when any fail.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "bench" / "results"


def is_ancestor(commit: str) -> bool:
    r = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=REPO_ROOT, capture_output=True, check=False,
    )
    return r.returncode == 0


def check_csv(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if "git_commit" not in (reader.fieldnames or []):
                return errors
            seen: set[str] = set()
            for row in reader:
                commit = row.get("git_commit", "").strip()
                if not commit or commit in seen or commit == "unknown":
                    continue
                seen.add(commit)
                if commit.endswith("-dirty"):
                    errors.append(f"  {path.name}: stamp '{commit}' is from a dirty tree")
                elif not is_ancestor(commit):
                    errors.append(
                        f"  {path.name}: stamp '{commit}' is not an ancestor of HEAD"
                    )
    except Exception as exc:
        errors.append(f"  {path.name}: could not read ({exc})")
    return errors


def main() -> int:
    csvs = sorted(RESULTS_DIR.glob("*.csv"))
    if not csvs:
        print("check_csv_stamps: no CSVs in bench/results/ — nothing to check")
        return 0

    all_errors: list[str] = []
    for path in csvs:
        all_errors.extend(check_csv(path))

    if all_errors:
        print("check_csv_stamps: FAIL — the following stamps are not in this tree's history:")
        for line in all_errors:
            print(line)
        print()
        print("Re-run the relevant benchmark script on a clean checkout of a main commit,")
        print("then commit the new CSV so its stamp resolves in a fresh clone.")
        return 1

    print(f"check_csv_stamps: OK — all {len(csvs)} CSV(s) have stamps that are ancestors of HEAD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
