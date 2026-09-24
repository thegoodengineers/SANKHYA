#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The commit a results CSV is stamped with (#433).

Every bench/runners/*.py writes a `git_commit` column, and docs/BENCHMARKS.md rests on the
claim that the column names the build that produced the numbers. Until #433 each runner
read `git rev-parse --short HEAD` when it started - the WORKING TREE at that moment, which
is the build only if nobody touches the repository during the run. On 23 Sep 2026 a three-
leg sweep against one binary recorded two different commits because two pull requests were
merged and pulled while it ran; nothing warned, and the odd row named code that was never
executed.

The binary knows what it was built from: `sankhya version` prints it in parentheses, the
same field scripts/binary_provenance.sh reads. So the stamp comes from the binary when one
is given, the tree is consulted only for the `-dirty` suffix (tracked files modified other
than the tier manifests, which the fetch scripts rewrite as part of a run), and HEAD is the
fallback when no binary is given or it does not answer. When the binary and the tree
disagree the runner says so on stderr and stamps the binary: the CSV records what ran.

bench/runners/check_result_stamps.py (#442) then asks whether that commit is on main.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# The manifests the fetch scripts rewrite as part of a run; a change there says nothing
# about what was measured. Untracked files are ignored for the same reason.
# Every fetcher that rewrites a tracked manifest belongs here. The four added with #529, #530
# and #491 were missing, and a Windows fetch left their manifests rewritten with CRLF endings,
# which stamped a whole night of otherwise clean runs `-dirty`.
MANIFEST_EXCLUSIONS = (
    ":!data/netlib/reference.json",
    ":!data/mittelmann/reference.json",
    ":!data/kennington/reference.json",
    ":!data/netlib-infeasible/reference.json",
    ":!data/maros-meszaros/reference.json",
    ":!data/miplib/reference.json",
)
_VERSION_SHA = re.compile(r"\(([0-9a-fA-F]{7,40})[,)]")


def parse_version_line(text: str) -> str:
    """The commit in `SANKHYA 0.1.0 (e4b7295, Release, ...)`, or an empty string."""
    match = _VERSION_SHA.search(text or "")
    return match.group(1) if match else ""


def binary_commit(binary: str | Path | None) -> str:
    """What the binary says it was built from, or an empty string when it cannot say."""
    if not binary:
        return ""
    try:
        result = subprocess.run([str(binary), "version"], capture_output=True, text=True,
                                check=False, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    return parse_version_line(result.stdout)


def repo_head() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=False)
    except OSError:
        return ""
    return result.stdout.strip()


def repo_dirty() -> bool:
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", ".",
             *MANIFEST_EXCLUSIONS],
            cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    except OSError:
        return False
    return bool(status.stdout.strip())


def stamp(binary: str | Path | None = None) -> str:
    """The commit to write into a results CSV: the binary's when it is known, HEAD otherwise,
    with `-dirty` appended when the tree has modified tracked files. Warns on stderr when
    the binary and the tree name different commits."""
    built = binary_commit(binary)
    head = repo_head()
    if binary and not built:
        # A binary that does not answer is the case this module exists for, silently
        # falling back to the tree would be the old behaviour with a new name. On the
        # Windows box it is Smart App Control refusing a freshly linked file for a few
        # minutes; the fix is to run the stamp again, not to trust HEAD.
        print(f"stamp: {binary} gave no version line; stamping the tree at "
              f"{head or 'unknown'} instead - re-run if the binary was blocked (#433)",
              file=sys.stderr)
    if built and head and not head.startswith(built) and not built.startswith(head):
        print(f"stamp: the binary was built at {built} but the tree is at {head}; the CSV "
              f"is stamped {built}, which is what ran (#433)", file=sys.stderr)
    commit = built or head or "unknown"
    if commit != "unknown" and repo_dirty():
        commit += "-dirty"
    return commit
