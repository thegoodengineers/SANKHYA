#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""fetch_data.install(): a tracked CRLF instance that emps re-decodes as LF on Linux is left
alone, so a clean checkout stays clean and the instance sha256 a CSV records does not
depend on the platform that ran the fetch. A real content change still replaces the file.
No network, no decoder.

    python bench/runners/test_fetch_line_endings.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fetch_data  # noqa: E402

FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}  {detail}")
    if not condition:
        FAILURES += 1


def main() -> int:
    tracked = (fetch_data.DATA_DIR / "afiro.mps").read_bytes()
    check(b"\r\n" in tracked, "the tracked afiro.mps is CRLF, as committed")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        target = work / "afiro.mps"
        fresh = work / "fresh.mps"
        target.write_bytes(tracked)
        fresh.write_bytes(tracked.replace(b"\r\n", b"\n"))  # what emps writes on Linux

        print("the same model, LF where the tracked copy is CRLF")
        check(not fetch_data.install(fresh, target), "install() reports nothing written")
        check(target.read_bytes() == tracked, "the tracked bytes, and so their sha256, stay")

        print("a real change to the model")
        changed = tracked.replace(b"-464.75", b"-464.76") + b"* edited\n"
        fresh.write_bytes(changed)
        check(fetch_data.install(fresh, target), "install() writes it")
        check(target.read_bytes() == changed, "and the target holds the new bytes")
        check(not (work / "afiro.mps.partial").exists(), "no .partial left behind")

        print("no tracked copy yet")
        target.unlink()
        check(fetch_data.install(fresh, target) and target.read_bytes() == changed,
              "the decoded file is installed")

    status = subprocess.run(["git", "status", "--porcelain", "--", "data/netlib/afiro.mps"],
                            cwd=fetch_data.REPO_ROOT, capture_output=True, text=True)
    check(status.stdout.strip() == "", "the repository copy was never touched")

    print("all passed" if FAILURES == 0 else f"{FAILURES} failure(s)")
    return 0 if FAILURES == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
