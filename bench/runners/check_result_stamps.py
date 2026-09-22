#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check that every cited results CSV is stamped with a commit that main contains (#433).

The evidence claim behind docs/BENCHMARKS.md is that any published number can be recomputed
from a CSV that names the instance by sha256, the build by commit and the machine. A CSV
produced on a PR branch and carried to main by a squash merge is stamped with a commit that
dies at the merge: on a fresh clone the stamp does not resolve, and scripts/reproduce.sh
cannot tie the number back to a build. That is what happened to the GPU CSVs of #390 and
#424, and it happens to every benchmark run on a branch unless something checks.

This does. For every CSV git tracks under bench/results/ it reads the stamp - the
`git_commit` column of the first row when the runner recorded one, the `<sha>` in a
`PREFIX-<sha>[-suffix].csv` name otherwise - and asks git whether that commit is an ancestor
of the base ref (origin/main, or main, or HEAD, whichever exists). The column is the
authoritative stamp, since the runner wrote it; a name that disagrees with it is reported.

A stamp that is not on main can still be honest when the run happened on a branch and the
squash merge that carried it is: bench/results/squash-stamps.txt lists such pairs, one per
line, `<branch-sha> <squash-sha> <note>`, and a listed stamp passes when its squash commit
is an ancestor of the base. That is the issue's third option: record both commits rather
than re-stamp a CSV with a commit it was not produced on.

Severity follows the claim. A CSV that README.md or a docs/*.md file cites by name must
resolve, or the check fails; an uncited CSV that does not resolve is history rather than
evidence and is reported as a warning (`--all` makes those failures too). A CSV with no
stamp at all - the old miplib runner's shape - is reported and never fails.

Usage:

    python bench/runners/check_result_stamps.py [--base REF] [--results DIR] [--all]

Exit status 0 when every cited stamp resolves, 1 otherwise, with each failure named.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
SQUASH_LIST = "squash-stamps.txt"
CITING_DOCS = [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").glob("*.md"))]

_SHA_IN_NAME = re.compile(r"-([0-9a-f]{7,40})(?:-[a-z0-9]+)*\.csv$")
_CSV_NAME = re.compile(r"[A-Za-z0-9_.-]+\.csv")


@dataclass
class Stamp:
    path: Path
    sha: str  # empty when the CSV carries no stamp at all
    source: str  # "column", "name" or "none"
    note: str = ""  # a name that disagrees with the column


def first_row(path: Path) -> dict:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                return row
    except (OSError, ValueError):
        pass
    return {}


def stamp_of(path: Path) -> Stamp:
    """The commit a results CSV claims it was produced at, and where that claim came from."""
    column = (first_row(path).get("git_commit") or "").strip()
    named = _SHA_IN_NAME.search(path.name)
    name = named.group(1) if named else ""
    if column:
        note = ""
        if name and not (column.startswith(name) or name.startswith(column)):
            note = f"the file name says {name} but the git_commit column, which the runner wrote, says {column}"
        return Stamp(path, column, "column", note)
    if name:
        return Stamp(path, name, "name")
    return Stamp(path, "", "none")


def read_squash_list(results_dir: Path) -> dict[str, tuple[str, str]]:
    """branch sha -> (squash sha, note), from squash-stamps.txt; empty when absent."""
    pairs: dict[str, tuple[str, str]] = {}
    path = results_dir / SQUASH_LIST
    if not path.is_file():
        return pairs
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 2)
        if len(parts) < 2:
            continue
        pairs[parts[0]] = (parts[1], parts[2] if len(parts) > 2 else "")
    return pairs


def cited_names(docs: list[Path]) -> set[str]:
    """Every CSV file name a document mentions."""
    names: set[str] = set()
    for doc in docs:
        try:
            text = doc.read_text(encoding="utf-8")
        except OSError:
            continue
        names.update(_CSV_NAME.findall(text))
    return names


class Git:
    """The questions asked of git, so a test can answer them for a repository of its own."""

    def __init__(self, repo: Path) -> None:
        self.repo = repo

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=self.repo, capture_output=True, text=True)

    def ref_exists(self, ref: str) -> bool:
        return self._run("rev-parse", "--verify", "--quiet", ref + "^{commit}").returncode == 0

    def is_ancestor(self, sha: str, base: str) -> bool:
        return self._run("merge-base", "--is-ancestor", sha, base).returncode == 0

    def default_base(self) -> str:
        for ref in ("origin/main", "main", "HEAD"):
            if self.ref_exists(ref):
                return ref
        return "HEAD"

    def tracked_csvs(self, results_dir: Path) -> list[Path] | None:
        """The CSVs git tracks under results_dir, or None when it is not inside this repo."""
        try:
            relative = results_dir.resolve().relative_to(self.repo.resolve())
        except ValueError:
            return None
        out = self._run("ls-files", "--", str(relative / "*.csv").replace("\\", "/"))
        if out.returncode != 0:
            return None
        return sorted(self.repo / line for line in out.stdout.split("\n") if line.strip())


@dataclass
class Verdict:
    path: Path
    ok: bool
    detail: str
    cited: bool = True
    warning: bool = False


@dataclass
class Report:
    verdicts: list[Verdict] = field(default_factory=list)

    @property
    def failures(self) -> list[Verdict]:
        return [v for v in self.verdicts if not v.ok]

    @property
    def warnings(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.ok and v.warning]


def check(results_dir: Path, git: Git, base: str, cited: set[str] | None = None,
          strict: bool = False) -> Report:
    """`cited` is the set of CSV names the docs mention; None means every CSV is held to the
    cited standard. With `strict`, an uncited stamp that does not resolve fails as well."""
    squash = read_squash_list(results_dir)
    paths = git.tracked_csvs(results_dir)
    if paths is None:
        paths = sorted(results_dir.glob("*.csv"))
    report = Report()
    for path in paths:
        is_cited = cited is None or path.name in cited
        stamp = stamp_of(path)
        suffix = f"; {stamp.note}" if stamp.note else ""
        if not stamp.sha:
            report.verdicts.append(Verdict(path, True, "no stamp: neither a git_commit column nor "
                                                       "a sha in the name (an old runner's shape)",
                                           is_cited, warning=True))
            continue
        if git.is_ancestor(stamp.sha, base):
            report.verdicts.append(Verdict(path, True, f"{stamp.sha} is on {base} ({stamp.source})"
                                                       + suffix, is_cited, warning=bool(stamp.note)))
            continue
        carried = None
        for branch_sha, (squash_sha, note) in squash.items():
            if stamp.sha.startswith(branch_sha) or branch_sha.startswith(stamp.sha):
                carried = (squash_sha, note)
                break
        if carried is not None:
            squash_sha, note = carried
            if git.is_ancestor(squash_sha, base):
                report.verdicts.append(Verdict(path, True, f"{stamp.sha} was run on a branch; its "
                                                           f"squash merge {squash_sha} is on {base}"
                                                           + (f" ({note})" if note else "") + suffix,
                                               is_cited, warning=bool(stamp.note)))
                continue
            detail = (f"{stamp.sha} is listed in {SQUASH_LIST} with squash merge {squash_sha}, "
                      f"which is not on {base} either")
        else:
            detail = (f"{stamp.sha} ({stamp.source}) is not an ancestor of {base}: the run "
                      f"happened on a branch that was squash-merged, or on no branch of this "
                      f"repository; add the pair to {SQUASH_LIST} or re-run on a main commit")
        fails = is_cited or strict
        report.verdicts.append(Verdict(path, not fails, detail + suffix, is_cited,
                                       warning=not fails))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", default=None,
                        help="the ref every stamp must be an ancestor of (default: origin/main, "
                             "then main, then HEAD)")
    parser.add_argument("--results", default=str(RESULTS_DIR), help="the results directory")
    parser.add_argument("--all", action="store_true",
                        help="fail on any unresolvable stamp, cited by the docs or not")
    args = parser.parse_args(argv)
    results_dir = Path(args.results)
    git = Git(REPO_ROOT)
    base = args.base or git.default_base()
    report = check(results_dir, git, base, cited_names(CITING_DOCS), strict=args.all)
    for verdict in report.verdicts:
        mark = "FAIL" if not verdict.ok else ("warn" if verdict.warning else "ok  ")
        cited = "cited" if verdict.cited else "uncited"
        print(f"  [{mark}] {verdict.path.name} ({cited}): {verdict.detail}")
    print(f"{len(report.verdicts)} tracked CSV(s) checked against {base}: "
          f"{len(report.failures)} failure(s), {len(report.warnings)} warning(s)")
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
