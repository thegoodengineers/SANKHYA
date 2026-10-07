#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for make_scoreboard.py (#983).

Hermetic: a throwaway bench/results/ and README.md are built in a temporary directory with
a handful of synthetic CSVs whose pass counts are known by construction, so the generator's
numbers can be checked by hand rather than against whatever happens to be committed on
`main` today (those are exercised separately, by actually running the script against the
real repository - see the PR body for #983, which also says in which commands this was
run). `latest_result.latest()` is driven off REPO_ROOT and `git log`, so each fixture
repository gets its own git history with one commit per CSV.

    python bench/runners/test_make_scoreboard.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import latest_result  # noqa: E402
import make_scoreboard as scoreboard  # noqa: E402

FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES += 1


def git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return out.stdout.strip()


NETLIB_HEADER = ("instance,instance_sha256,rows,columns,nonzeros,status,message,"
                 "our_objective,published_objective,absolute_gap,relative_gap,"
                 "matches_published,independently_verified,verifier_message,passed,"
                 "objective_offset,differs_by_objective_constant,wall_seconds,"
                 "solver_seconds,iterations,algorithm,git_commit,machine,timestamp_utc,"
                 "solver_options")


def netlib_row(name: str, passed: int, commit_sha: str) -> str:
    return (f"{name},sha,10,10,10,optimal,,1.0,1.0,0.0,0.0,1,1,,{passed},0.0,0,0.1,0.1,1,"
           f"simplex,{commit_sha},test-machine,2026-01-0{passed + 1}T00:00:00+00:00,")


def build_fixture(repo: Path) -> dict:
    (repo / "bench" / "results").mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "t@t")
    git(repo, "config", "user.name", "t")

    results = repo / "bench" / "results"

    def commit_csv(stem: str, rows: list[tuple[str, int]]) -> tuple[Path, str]:
        """Two real commits: one that lands the CSV with a placeholder `git_commit`, whose
        hash is then known, and a second that rewrites the column to that (now real, now
        an ancestor of HEAD) hash - `commit_of()` reads the column, and it has to name a
        commit the fixture's own history actually produced for the ancestry check to mean
        anything."""
        path = results / f"netlib-full-{stem}.csv"
        text = NETLIB_HEADER + "\n" + "\n".join(netlib_row(n, p, "0000000")
                                                for n, p in rows) + "\n"
        path.write_text(text, encoding="utf-8")
        git(repo, "add", str(path.relative_to(repo)))
        git(repo, "commit", "-q", "-m", path.name)
        sha = git(repo, "rev-parse", "HEAD")
        path.write_text(text.replace("0000000", sha), encoding="utf-8")
        git(repo, "add", str(path.relative_to(repo)))
        git(repo, "commit", "-q", "-m", f"{path.name} stamp")
        return path, sha[:7]

    # Two netlib-full CSVs at two commits: the generator must pick the NEWER one (3 of 4
    # passed), not the older (2 of 4), and its own commit must be the one it reports.
    old_path, old_sha = commit_csv("0000000", [("a", 1), ("b", 1), ("c", 0), ("d", 0)])
    new_path, new_sha = commit_csv("1111111", [("a", 1), ("b", 1), ("c", 1), ("d", 0)])
    return {"old_sha": old_sha, "new_sha": new_sha, "old_path": old_path, "new_path": new_path}


def _patched(repo: Path):
    """Point both make_scoreboard's and latest_result's module globals at the fixture
    repository (latest_result.commit_order() runs `git log` in its own REPO_ROOT, a
    separate global from make_scoreboard's), and return the (restore) callback."""
    originals = (scoreboard.RESULTS_DIR, scoreboard.REPO_ROOT,
                latest_result.RESULTS_DIR, latest_result.REPO_ROOT)
    scoreboard.REPO_ROOT = latest_result.REPO_ROOT = repo
    scoreboard.RESULTS_DIR = latest_result.RESULTS_DIR = repo / "bench" / "results"

    def restore() -> None:
        (scoreboard.RESULTS_DIR, scoreboard.REPO_ROOT,
         latest_result.RESULTS_DIR, latest_result.REPO_ROOT) = originals
    return restore


def test_picks_the_newer_csv_and_counts_it_right() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        build_fixture(repo)
        restore = _patched(repo)
        try:
            row = scoreboard.netlib_row()
        finally:
            restore()
        check(row is not None, "a netlib row is produced")
        if row is not None:
            check(row.path.name == "netlib-full-1111111.csv",
                 "the newer (by git history) CSV is picked, not the older", row.path.name)
            check((row.passed, row.total) == (3, 4), "3 of 4 passed, read from that CSV",
                 str((row.passed, row.total)))


def test_render_has_no_fabricated_rows() -> None:
    """Every set this repository's real bench/results/ has no CSV for (here: everything
    but netlib) gets no row - the acceptance criterion that a missing set is left out
    rather than invented."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        build_fixture(repo)
        restore = _patched(repo)
        try:
            rows = scoreboard.all_rows()
        finally:
            restore()
        check(len(rows) == 1, "only the one set with a committed CSV gets a row",
             str([r.label for r in rows]))


def test_splice_is_idempotent() -> None:
    """Writing the table into a README that already has one replaces the old block and
    changes nothing outside it; running it twice produces the same text (the CI check's own
    assumption)."""
    readme = "# Title\n\nSome text.\n\n---\n\nmore text\n"
    table = f"{scoreboard.START_MARKER}\nrow\n{scoreboard.END_MARKER}\n"
    once = scoreboard.splice(readme, table)
    check("Some text." in once and "more text" in once, "surrounding text is kept")
    check(once.count("scoreboard:start") == 1,
         "exactly one scoreboard block after a first insert")
    twice = scoreboard.splice(once, table)
    check(twice == once, "splicing the same table again is a no-op")


def main() -> int:
    test_picks_the_newer_csv_and_counts_it_right()
    test_render_has_no_fabricated_rows()
    test_splice_is_idempotent()
    print(f"\n{FAILURES} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
