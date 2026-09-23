#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify that git_commit stamps in newly introduced bench/results/*.csv are ancestors of HEAD.

Only CSVs that are added or modified relative to the merge-base with the default branch are
checked.  Pre-existing CSVs whose stamps became orphaned through squash-merges are not
re-flagged here; the intent is to stop new violations from landing, not to block CI until
every old CSV is regenerated.

Exit 0 when all stamps pass.  Exit 1 and print the offending files when any fail.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
# Branches to measure "new" against, in preference order.
_BASE_CANDIDATES = ["upstream/main", "origin/main", "main"]


def _run(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=REPO_ROOT, capture_output=True, check=False, **kw)


def _merge_base() -> str | None:
    """Return the merge-base commit between HEAD and the default branch, or None."""
    for branch in _BASE_CANDIDATES:
        r = _run(["git", "merge-base", "HEAD", branch])
        if r.returncode == 0:
            return r.stdout.decode().strip()
    return None


def _changed_csvs(base: str) -> set[str]:
    """Return basenames of CSVs added or modified since base."""
    r = _run(["git", "diff", "--name-only", "--diff-filter=ACM", base, "HEAD"])
    if r.returncode != 0:
        return set()
    names: set[str] = set()
    for line in r.stdout.decode().splitlines():
        p = Path(line)
        if p.suffix == ".csv" and p.parts[0:2] == ("bench", "results"):
            names.add(p.name)
    return names


def is_ancestor(commit: str) -> bool:
    r = _run(["git", "merge-base", "--is-ancestor", commit, "HEAD"])
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
    base = _merge_base()
    if base is None:
        # No remote ref found — check all CSVs (e.g. a fresh detached clone).
        csvs = sorted(RESULTS_DIR.glob("*.csv"))
        scope_note = "all CSVs (no base branch found)"
    else:
        changed = _changed_csvs(base)
        if not changed:
            print("check_csv_stamps: OK — no bench/results CSV files changed in this branch")
            return 0
        csvs = sorted(p for p in RESULTS_DIR.glob("*.csv") if p.name in changed)
        scope_note = f"{len(csvs)} CSV(s) added/modified since {base[:12]}"

    if not csvs:
        print(f"check_csv_stamps: OK — {scope_note}, nothing to check")
        return 0

    all_errors: list[str] = []
    for path in csvs:
        all_errors.extend(check_csv(path))

    if all_errors:
        print(f"check_csv_stamps: FAIL — checking {scope_note}:")
        for line in all_errors:
            print(line)
        print()
        print("Re-run the relevant benchmark script on a clean checkout of a main commit,")
        print("then commit the new CSV so its stamp resolves in a fresh clone.")
        return 1

    print(f"check_csv_stamps: OK — {scope_note}, all stamps are ancestors of HEAD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
