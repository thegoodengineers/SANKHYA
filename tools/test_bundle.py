#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for tools/bundle.py and tools/replay_bundle.py (#526).

Needs a built sankhya-cli (tools/bundle.py drives it as a subprocess, the same way a
planner would run it by hand) - skips, rather than fails, if none is found, so this file
does not become a spurious failure on a checkout with no build tree.

    python tools/test_bundle.py
"""
from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bundle import build_bundle, default_binary  # noqa: E402
from replay_bundle import ReplayError, replay  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_MODEL = REPO_ROOT / "demo" / "crude_blend.mps"

FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    if condition:
        print(f"  [PASS] {name}" + (f"  {detail}" if detail else ""))
    else:
        FAILURES += 1
        print(f"  [FAIL] {name}" + (f"  {detail}" if detail else ""))


def _tamper(bundle_path: Path, tampered_path: Path, *, member: str, replace: bytes,
           with_: bytes) -> None:
    with zipfile.ZipFile(bundle_path) as source:
        names = source.namelist()
        contents = {name: source.read(name) for name in names}
    contents[member] = contents[member].replace(replace, with_)
    with zipfile.ZipFile(tampered_path, "w", zipfile.ZIP_DEFLATED) as out:
        for name in names:
            out.writestr(name, contents[name])


def test_a_clean_bundle_replays() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        bundle_path = directory / "run.zip"
        build_bundle(DEMO_MODEL, [], bundle_path)
        check(bundle_path.is_file(), "bundle.py wrote a zip file", str(bundle_path))

        raised = False
        try:
            replay(bundle_path, resolve_binary=None)
        except ReplayError as error:
            raised = True
            print(f"    (unexpected: {error})")
        check(not raised, "a clean bundle replays without raising")


def test_a_bundle_with_the_solution_edited_fails_replay() -> None:
    """#526's own acceptance criterion: a tampered bundle (solution changed) fails replay."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        bundle_path = directory / "run.zip"
        build_bundle(DEMO_MODEL, [], bundle_path)

        with zipfile.ZipFile(bundle_path) as source:
            original_sol = source.read("solution.sol")
        # Change one digit of the recorded objective - a plausible, subtle tamper, not a
        # gross corruption an unrelated check would have caught anyway.
        needle = b"214.145945945"
        check(needle in original_sol, "the fixture's objective is where the tamper expects it",
              "(if this fails, demo/crude_blend.mps's answer changed - update the needle)")

        tampered_path = directory / "run_tampered.zip"
        _tamper(bundle_path, tampered_path, member="solution.sol", replace=needle,
               with_=b"999999.999999")

        raised = False
        message = ""
        try:
            replay(tampered_path, resolve_binary=None)
        except ReplayError as error:
            raised = True
            message = str(error)
        check(raised, "a bundle with an edited solution.sol fails replay")
        check("solution.sol" in message and "sha256 mismatch" in message,
              "the failure names the tampered file and says why", message[:120])


def test_a_bundle_with_the_manifest_edited_fails_replay() -> None:
    """Not just the solution - ANY bundled file changing after the fact is caught."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        bundle_path = directory / "run.zip"
        build_bundle(DEMO_MODEL, [], bundle_path)

        tampered_path = directory / "run_tampered.zip"
        _tamper(bundle_path, tampered_path, member="manifest.json",
               replace=b'"options": []', with_=b'"options": ["presolve=false"]')

        raised = False
        try:
            replay(tampered_path, resolve_binary=None)
        except ReplayError:
            raised = True
        check(raised, "a bundle with an edited manifest.json fails replay")


def test_a_bundle_missing_its_checksums_file_fails_replay() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        bundle_path = directory / "run.zip"
        build_bundle(DEMO_MODEL, [], bundle_path)

        with zipfile.ZipFile(bundle_path) as source:
            names = [n for n in source.namelist() if n != "checksums.sha256.json"]
            contents = {n: source.read(n) for n in names}
        stripped_path = directory / "run_no_checksums.zip"
        with zipfile.ZipFile(stripped_path, "w", zipfile.ZIP_DEFLATED) as out:
            for name in names:
                out.writestr(name, contents[name])

        raised = False
        try:
            replay(stripped_path, resolve_binary=None)
        except ReplayError:
            raised = True
        check(raised, "a bundle with no checksums.sha256.json fails replay rather than "
              "silently skipping the integrity check")


def main() -> int:
    print("tools/bundle.py and tools/replay_bundle.py tests\n")
    try:
        default_binary()
    except FileNotFoundError:
        print("SKIPPED: no sankhya-cli binary under build/ or build-release/ - "
             "run scripts/configure.sh && cmake --build build first")
        return 0

    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            print(name)
            function()
    print()
    if FAILURES == 0:
        print("ALL TESTS PASSED")
        return 0
    print(f"{FAILURES} check(s) FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
