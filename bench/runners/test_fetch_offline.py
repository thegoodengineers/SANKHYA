#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""fetch_data.py --offline-fallback with netlib.org unreachable: the tracked readme and
instances are used, nothing tracked changes, and without the flag the same outage fails.
No network: the base URL points at a closed local port.

    python bench/runners/test_fetch_offline.py
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
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


def run(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    saved = sys.argv
    sys.argv = ["fetch_data.py", *argv]
    try:
        with redirect_stdout(out), redirect_stderr(out):
            try:
                code = fetch_data.main()
            except SystemExit as error:
                code = error.code if isinstance(error.code, int) else 1
                out.write(str(error))
    finally:
        sys.argv = saved
    return code, out.getvalue()


def snapshot() -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(fetch_data.DATA_DIR.iterdir())
            if p.is_file() and (p.suffix in (".mps", ".json") or p.name == "readme")}


def main() -> int:
    fetch_data.NETLIB_BASE = "http://127.0.0.1:9/lp/data"  # the discard port: refused at once
    fetch_data.OFFLINE_PROBE_TIMEOUT = 2
    manifest = json.loads((fetch_data.DATA_DIR / "reference.json").read_text())
    before = snapshot()

    print("netlib.org unreachable, --offline-fallback")
    code, text = run("--offline-fallback")
    check(code == 0, "the tracked copies are used and the fetch succeeds", text.strip()[-160:])
    check("WARNING" in text, "the fallback says so")
    after = snapshot()
    changed = sorted(n for n in before if before[n] != after.get(n) and n != "reference.json")
    check(not changed, "no tracked instance or readme changed", " ".join(changed))
    written = json.loads((fetch_data.DATA_DIR / "reference.json").read_text())
    same = all(written["instances"][n]["mps_sha256"] == e["mps_sha256"]
               and written["instances"][n]["published_optimal"] == e["published_optimal"]
               for n, e in manifest["instances"].items())
    check(same, "every instance keeps its sha256 and its published optimum")
    (fetch_data.DATA_DIR / "reference.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    print("netlib.org unreachable, no flag")
    code, text = run()
    check(code != 0, "without the flag an outage still fails", text.strip()[-120:])
    check(snapshot() == before, "and leaves the tracked files as they were")

    print("all passed" if FAILURES == 0 else f"{FAILURES} failure(s)")
    return 0 if FAILURES == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
