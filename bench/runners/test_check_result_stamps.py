#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for check_result_stamps.py (#433).

Hand-rolled rather than pytest, matching the repository convention. A throwaway git
repository is built in a temporary directory so the ancestry questions have known answers:
a main branch with two commits, a side branch with one commit that is squash-merged onto
main, and CSVs stamped with each of those. Run directly:

    python bench/runners/test_check_result_stamps.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_result_stamps as checker

FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES += 1


def git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def commit(repo: Path, name: str) -> str:
    (repo / name).write_text(name + "\n", encoding="utf-8")
    git(repo, "add", name)
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", name)
    return git(repo, "rev-parse", "--short=7", "HEAD")


def build_repository(repo: Path) -> dict[str, str]:
    git(repo, "init", "-q", "-b", "main")
    shas = {"main1": commit(repo, "one")}
    git(repo, "checkout", "-q", "-b", "side")
    shas["branch"] = commit(repo, "side-work")
    git(repo, "checkout", "-q", "main")
    shas["main2"] = commit(repo, "two")
    # The squash merge, done by hand so it needs no merge machinery on the runner (git
    # merge --squash exits 128 on the CI image): the branch's file arrives on main as a
    # fresh commit with no second parent, which is exactly what a squash merge leaves.
    git(repo, "checkout", "-q", "side", "--", "side-work")
    git(repo, "add", "side-work")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "squash of side")
    shas["squash"] = git(repo, "rev-parse", "--short=7", "HEAD")
    git(repo, "branch", "-q", "-D", "side")  # the branch commit is now orphaned, as after a PR
    return shas


def write_csv(results: Path, name: str, column: str | None) -> Path:
    path = results / name
    if column is None:
        path.write_text("instance,status\nx,optimal\n", encoding="utf-8")
    else:
        path.write_text(f"instance,status,git_commit\nx,optimal,{column}\n", encoding="utf-8")
    return path


def verdict_for(report: checker.Report, name: str):
    for v in report.verdicts:
        if v.path.name == name:
            return v
    return None


def test_stamps() -> None:
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td) / "repo"
        repo.mkdir()
        shas = build_repository(repo)
        results = Path(td) / "results"  # outside the repository: every CSV is looked at
        results.mkdir()
        write_csv(results, f"netlib-{shas['main1']}.csv", shas["main1"])
        write_csv(results, f"scale-{shas['main2']}.csv", None)
        write_csv(results, f"gpu-{shas['branch']}.csv", shas["branch"])
        write_csv(results, "miplib-cuts-off.csv", None)
        write_csv(results, f"compare-{shas['main1']}.csv", shas["main2"])
        write_csv(results, f"other-{shas['branch']}-second.csv", None)
        write_csv(results, f"old-{shas['branch']}.csv", None)

        g = checker.Git(repo)
        base = g.default_base()
        check(base == "main", "the base falls back to main without an origin", base)
        check(g.tracked_csvs(results) is None, "a directory outside the repository is globbed")

        # Everything cited: every unresolvable stamp fails.
        report = checker.check(results, g, base)
        v = verdict_for(report, f"netlib-{shas['main1']}.csv")
        check(v is not None and v.ok and not v.warning, "a column stamp on main passes",
              v.detail if v else "")
        v = verdict_for(report, f"scale-{shas['main2']}.csv")
        check(v is not None and v.ok, "a name-only stamp on main passes", v.detail if v else "")
        v = verdict_for(report, f"gpu-{shas['branch']}.csv")
        check(v is not None and not v.ok, "an orphaned branch stamp fails", v.detail if v else "")
        v = verdict_for(report, "miplib-cuts-off.csv")
        check(v is not None and v.ok and v.warning and "no stamp" in v.detail,
              "an unstamped CSV is a warning, not a failure", v.detail if v else "")
        v = verdict_for(report, f"compare-{shas['main1']}.csv")
        check(v is not None and v.ok and v.warning and "file name says" in v.detail,
              "a name that disagrees with the column is a warning; the column decides",
              v.detail if v else "")
        v = verdict_for(report, f"other-{shas['branch']}-second.csv")
        check(v is not None and not v.ok, "a suffixed name with an orphaned stamp fails",
              v.detail if v else "")

        # Cited against uncited: only the cited orphan fails, the other is a warning, and
        # --all turns the warning into a failure.
        cited = {f"gpu-{shas['branch']}.csv"}
        report = checker.check(results, g, base, cited)
        v = verdict_for(report, f"gpu-{shas['branch']}.csv")
        check(v is not None and not v.ok and v.cited, "a cited orphan fails", v.detail if v else "")
        v = verdict_for(report, f"old-{shas['branch']}.csv")
        check(v is not None and v.ok and v.warning and not v.cited,
              "an uncited orphan is a warning", v.detail if v else "")
        report = checker.check(results, g, base, cited, strict=True)
        v = verdict_for(report, f"old-{shas['branch']}.csv")
        check(v is not None and not v.ok, "--all fails the uncited orphan too",
              v.detail if v else "")

        # The squash list carries the branch stamp: listed against its squash commit it
        # passes; listed against a commit that is not on main either it still fails.
        (results / checker.SQUASH_LIST).write_text(
            f"# branch squash note\n{shas['branch']} {shas['squash']} gpu run on the side branch\n",
            encoding="utf-8")
        report = checker.check(results, g, base)
        v = verdict_for(report, f"gpu-{shas['branch']}.csv")
        check(v is not None and v.ok and "squash merge" in v.detail,
              "a branch stamp listed with its squash merge passes", v.detail if v else "")
        v = verdict_for(report, f"other-{shas['branch']}-second.csv")
        check(v is not None and v.ok, "the listing covers every CSV with that stamp",
              v.detail if v else "")
        (results / checker.SQUASH_LIST).write_text(
            f"{shas['branch']} {'0' * 7} a squash commit that does not exist\n", encoding="utf-8")
        report = checker.check(results, g, base)
        v = verdict_for(report, f"gpu-{shas['branch']}.csv")
        check(v is not None and not v.ok, "a listed squash commit that is not on main fails",
              v.detail if v else "")


def test_cited_names() -> None:
    with tempfile.TemporaryDirectory() as td:
        doc = Path(td) / "doc.md"
        doc.write_text("Source CSV: `bench/results/gpu-fb72ab4.csv`, beside miplib-078cb24.csv "
                       "and `netlib-full-adb37bb.csv`.\n", encoding="utf-8")
        names = checker.cited_names([doc, Path(td) / "missing.md"])
        check(names == {"gpu-fb72ab4.csv", "miplib-078cb24.csv", "netlib-full-adb37bb.csv"},
              "cited names are read out of the docs, missing files ignored", str(sorted(names)))


def test_the_committed_results_resolve() -> None:
    # The repository's own tracked CSVs, against whatever base this checkout has: every
    # stamp the docs cite is on main or listed with its squash merge. This is the check CI
    # runs, and the number of uncited warnings is printed so a reader sees it.
    g = checker.Git(checker.REPO_ROOT)
    base = g.default_base()
    report = checker.check(checker.RESULTS_DIR, g, base, checker.cited_names(checker.CITING_DOCS))
    check(len(report.verdicts) > 0, "the committed results were found", str(len(report.verdicts)))
    check(not report.failures, f"every cited stamp resolves against {base}",
          "; ".join(f"{v.path.name}: {v.detail}" for v in report.failures))
    print(f"  ({len(report.warnings)} warning(s) on uncited or unstamped CSVs)")


if __name__ == "__main__":
    print("check_result_stamps tests")
    test_stamps()
    test_cited_names()
    test_the_committed_results_resolve()
    if FAILURES:
        print(f"{FAILURES} failure(s)")
        sys.exit(1)
    print("all passed")
