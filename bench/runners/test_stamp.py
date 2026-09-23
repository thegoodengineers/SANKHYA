#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for bench/runners/stamp.py (#433). Pure Python except for the one case that runs
the built binary when there is one.

    python bench/runners/test_stamp.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stamp  # noqa: E402

FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}  {detail}")
    if not condition:
        FAILURES += 1


def test_no_runner_stamps_on_its_own() -> None:
    """#589: the commit stamp is defined once, here. A runner that shells out to
    `git rev-parse` itself stamps HEAD rather than the binary and skips the manifest
    exclusion, which is how the first L4 run came back -dirty."""
    here = Path(__file__).resolve().parent
    allowed = {"stamp.py", "check_result_stamps.py"}
    offenders = sorted(p.name for p in here.glob("*.py")
                       if p.name not in allowed and not p.name.startswith("test_")
                       and "rev-parse" in p.read_text(encoding="utf-8"))
    check(not offenders, "no runner calls git rev-parse on its own", ", ".join(offenders))


def test_parse_version_line() -> None:
    check(stamp.parse_version_line("SANKHYA 0.1.0 (e4b7295, Release, GNU 16.1.0, CUDA off)")
          == "e4b7295", "a release version line yields its commit")
    check(stamp.parse_version_line("SANKHYA 0.1.0 (0018254)") == "0018254",
          "a bare parenthesised commit is read too")
    check(stamp.parse_version_line("SANKHYA 0.1.0 (unknown, Release)") == "",
          "a build with no commit yields nothing rather than a word")
    check(stamp.parse_version_line("") == "", "an empty line yields nothing")


def test_head_fallback_matches_git() -> None:
    head = stamp.repo_head()
    check(len(head) >= 7, "repo_head reads a short sha", head)
    s = stamp.stamp(None)
    check(s.startswith(head), "with no binary the stamp is HEAD", f"{s} vs {head}")
    check(s.endswith("-dirty") == stamp.repo_dirty(), "the -dirty suffix follows the tree", s)


def test_a_binary_that_does_not_answer_falls_back_to_head_and_says_so() -> None:
    import io, contextlib
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        s = stamp.stamp("no-such-sankhya-binary")
    check(s.startswith(stamp.repo_head()), "an unanswering binary falls back to HEAD", s)
    check("gave no version line" in err.getvalue(), "and the fallback is announced on stderr",
          err.getvalue().strip()[:90])


def test_built_binary_stamps_its_own_commit() -> None:
    root = Path(__file__).resolve().parents[2]
    binary = next((p for p in (root / "build" / "sankhya.exe", root / "build" / "sankhya")
                   if p.exists()), None)
    if binary is None:
        print("  [skip] no built binary in build/; the binary case is exercised in CI")
        return
    built = stamp.binary_commit(binary)
    if not built:
        print("  [skip] the built binary did not answer `version` here")
        return
    s = stamp.stamp(binary)
    check(s.startswith(built), "the stamp is the commit the binary reports", f"{s} vs {built}")


if __name__ == "__main__":
    print("stamp.py (#433)")
    test_parse_version_line()
    test_no_runner_stamps_on_its_own()
    test_head_fallback_matches_git()
    test_a_binary_that_does_not_answer_falls_back_to_head_and_says_so()
    test_built_binary_stamps_its_own_commit()
    print(f"{FAILURES} check(s) FAILED" if FAILURES else "all checks passed")
    sys.exit(1 if FAILURES else 0)
