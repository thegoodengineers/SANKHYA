#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Re-verify an evidence bundle from tools/bundle.py (#526).

    python tools/replay_bundle.py run.zip
    python tools/replay_bundle.py run.zip --resolve build/sankhya   # also re-solve and compare

1. Every file in the bundle is re-hashed and checked against `checksums.sha256.json`,
   recorded when the bundle was made. A bundle where anything - the solution, the model, the
   manifest - was edited after the fact fails HERE, before anything downstream trusts it.
2. `tools/verify_solution.py` is run on the bundled model and solution: the same independent
   check (no code shared with the solver) a fresh solve gets.
3. With `--resolve <binary>`, the bundled model is re-solved with the bundled options and
   the objective/status compared against the bundled `stats.json` - the strongest replay,
   available whenever a matching binary exists to run.

Never links SANKHYA's C++: this reads files and shells out to `verify_solution.py` and,
only with `--resolve`, to a `sankhya-cli` binary the caller names - exactly like
`verify_solution.py` itself never links the solver it is checking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFY_SOLUTION = REPO_ROOT / "tools" / "verify_solution.py"


def _subprocess_env() -> dict:
    """See tools/bundle.py's function of the same name - the same Windows DLL-search-path
    gap applies to the `--resolve` re-solve here."""
    env = dict(os.environ)
    if sys.platform != "win32":
        return env
    extra = os.environ.get("SANKHYA_DLL_DIR") or r"C:\msys64\ucrt64\bin"
    if extra and Path(extra).is_dir():
        env["PATH"] = extra + os.pathsep + env.get("PATH", "")
    return env


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ReplayError(Exception):
    """A bundle failed replay - tampered, incomplete, or its answer did not check out."""


def check_integrity(staging: Path) -> dict:
    checksums_path = staging / "checksums.sha256.json"
    if not checksums_path.is_file():
        raise ReplayError("bundle carries no checksums.sha256.json - not a bundle "
                          "tools/bundle.py wrote, or that file was removed")
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))

    problems = []
    for name, recorded in checksums.items():
        path = staging / name
        if not path.is_file():
            problems.append(f"{name}: missing from the bundle")
            continue
        actual = sha256(path)
        if actual != recorded:
            problems.append(f"{name}: sha256 mismatch - recorded {recorded}, now {actual} "
                            "(the bundle was modified after it was made)")
    extra = {p.name for p in staging.iterdir() if p.is_file()} - set(checksums) - {
        "checksums.sha256.json"}
    if extra:
        problems.append(f"file(s) present but not in checksums.sha256.json: "
                        f"{sorted(extra)} (added after the bundle was made)")
    if problems:
        raise ReplayError("integrity check FAILED:\n  " + "\n  ".join(problems))
    return checksums


def replay(bundle_path: Path, resolve_binary: Path | None) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        with zipfile.ZipFile(bundle_path) as archive:
            archive.extractall(staging)

        check_integrity(staging)
        print("[PASS] integrity: every bundled file matches its recorded sha256")

        manifest = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
        model_path = staging / manifest["model_file"]
        sol_path = staging / "solution.sol"

        if not sol_path.is_file() or sol_path.stat().st_size == 0:
            raise ReplayError("no solution.sol in the bundle to verify")

        result = subprocess.run(
            [sys.executable, str(VERIFY_SOLUTION), str(model_path), str(sol_path), "--quiet"],
            capture_output=True, text=True)
        if result.returncode != 0:
            raise ReplayError(
                f"verify_solution.py rejected the bundled solution (exit {result.returncode}):\n"
                f"{result.stdout}\n{result.stderr}")
        print("[PASS] verify_solution.py accepts the bundled model/solution pair")

        if resolve_binary is not None:
            stats_before = json.loads((staging / "stats.json").read_text(encoding="utf-8"))
            cmd = [str(resolve_binary), "solve", str(model_path), "--stats",
                  str(staging / "resolve_stats.json")]
            for assignment in manifest.get("options", []):
                cmd += ["--option", assignment]
            if manifest.get("time_limit") is not None:
                cmd += ["--time-limit", str(manifest["time_limit"])]
            if manifest.get("ranging"):
                cmd.append("--ranging")
            run = subprocess.run(cmd, capture_output=True, text=True, env=_subprocess_env())
            if run.returncode not in (0, 1):  # the CLI's own exit codes for "ran"; see main.cpp
                raise ReplayError(f"re-solve failed to run (exit {run.returncode}): "
                                  f"{run.stdout}\n{run.stderr}")
            stats_after = json.loads((staging / "resolve_stats.json").read_text(encoding="utf-8"))
            before_result = stats_before.get("result", {})
            after_result = stats_after.get("result", {})
            if before_result.get("status") != after_result.get("status"):
                raise ReplayError(
                    f"re-solve status disagrees: bundled {before_result.get('status')!r}, "
                    f"replayed {after_result.get('status')!r}")
            before_obj = before_result.get("objective")
            after_obj = after_result.get("objective")
            if before_obj is not None and after_obj is not None:
                scale = max(1.0, abs(before_obj))
                if abs(before_obj - after_obj) > 1e-6 * scale:
                    raise ReplayError(
                        f"re-solve objective disagrees: bundled {before_obj}, "
                        f"replayed {after_obj}")
            print(f"[PASS] re-solve with {resolve_binary} agrees: "
                 f"{after_result.get('status')} {after_result.get('objective')}")

    print(f"\nREPLAYED: {bundle_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--resolve", type=Path, default=None, metavar="BINARY",
                        help="also re-solve with this sankhya-cli binary and compare")
    args = parser.parse_args()

    try:
        replay(args.bundle, args.resolve)
    except ReplayError as error:
        print(f"REPLAY FAILED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
