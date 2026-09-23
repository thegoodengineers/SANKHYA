#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A replayable evidence bundle for one solve: the model, the options, the binary's version
and commit, the SBOM's hash, the full log, the solution, and a manifest of sha256 sums, in
one zip (#526).

    python tools/bundle.py model.mps --bundle run.zip
    python tools/bundle.py model.mps --option time_limit=60 --ranging --bundle run.zip

Companion: `tools/replay_bundle.py run.zip` checks the manifest (every file's sha256 against
the hash recorded when the bundle was made - a tampered bundle fails here, not silently),
then re-runs `verify_solution.py` on the bundled model and solution. Neither script links
SANKHYA's C++: the bundle is built by driving `sankhya-cli` as a subprocess (the same way a
planner would run it by hand) and read back by parsing files, the same discipline
`verify_solution.py` itself keeps.
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
DEFAULT_SBOM = REPO_ROOT / "docs" / "sbom.spdx.json"

MANIFEST_NAME = "manifest.json"


def _subprocess_env() -> dict:
    """`os.environ`, with the MSYS2 UCRT64 toolchain's bin directory on `PATH`.

    `sankhya-cli` on Windows loads `libgomp-1.dll` (OpenMP) and other runtime DLLs from
    beside the compiler rather than anywhere Windows searches by default - the same gap
    `bindings/python/sankhya/_library.py` already works around for `libsankhya.dll`. A
    plain `subprocess.run` does not inherit whatever PATH a developer's interactive shell
    happened to have, so without this the binary fails to start at all
    (`STATUS_DLL_NOT_FOUND`) when this script is not run from inside such a shell - exactly
    the failure this function exists to prevent, found by running this script for real
    rather than assumed to work.
    """
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


def default_binary() -> Path:
    for candidate in ("build/sankhya", "build/sankhya.exe", "build-release/sankhya",
                      "build-release/sankhya.exe"):
        path = REPO_ROOT / candidate
        if path.is_file():
            return path
    raise FileNotFoundError(
        "no sankhya-cli binary found under build/ or build-release/; pass --binary")


def build_bundle(model_path: Path, options: list[str], bundle_path: Path, *,
                 binary: Path | None = None, ranging: bool = False,
                 time_limit: float | None = None) -> None:
    binary = binary or default_binary()
    model_path = model_path.resolve()

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        sol_path = staging / "solution.sol"
        stats_path = staging / "stats.json"

        cmd = [str(binary), "solve", str(model_path), "--write-sol", str(sol_path),
              "--stats", str(stats_path)]
        for assignment in options:
            cmd += ["--option", assignment]
        if time_limit is not None:
            cmd += ["--time-limit", str(time_limit)]
        if ranging:
            cmd.append("--ranging")

        env = _subprocess_env()
        run = subprocess.run(cmd, capture_output=True, text=True, env=env)
        (staging / "log.txt").write_text(
            f"$ {' '.join(cmd)}\n\n--- stdout ---\n{run.stdout}\n--- stderr ---\n{run.stderr}\n"
            f"--- exit code {run.returncode} ---\n", encoding="utf-8")

        version = subprocess.run([str(binary), "version"], capture_output=True, text=True,
                                 env=env).stdout.strip()

        # The model is EMBEDDED, not referenced by path: a bundle is evidence someone else
        # (or this machine, later) replays, and a path on the machine that made it is not
        # something a replay elsewhere can resolve. The hash is recorded too, so a
        # tampered/substituted embedded copy is still caught the same way every other
        # bundled file is.
        model_copy = staging / model_path.name
        model_copy.write_bytes(model_path.read_bytes())

        if sol_path.exists():
            # already staged as solution.sol
            pass
        else:
            (staging / "solution.sol").write_text(
                "; the solve did not write a solution file (see log.txt / stats.json)\n",
                encoding="utf-8")

        manifest = {
            "sankhya_version": version,
            "model_file": model_copy.name,
            "options": options,
            "time_limit": time_limit,
            "ranging": ranging,
            "command": cmd,
            "exit_code": run.returncode,
            "sbom_sha256": sha256(DEFAULT_SBOM) if DEFAULT_SBOM.is_file() else None,
        }
        (staging / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # The sha256 of every OTHER file, over the manifest as first written (so the
        # manifest cannot both name its own hash and be hashed itself - the file list
        # below is exactly what gets zipped).
        files = sorted(p for p in staging.iterdir() if p.is_file())
        checksums = {p.name: sha256(p) for p in files}

        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for p in files:
                archive.write(p, arcname=p.name)
            archive.writestr("checksums.sha256.json", json.dumps(checksums, indent=2))

    print(f"wrote {bundle_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("--option", action="append", default=[], dest="options",
                        metavar="NAME=VALUE")
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--ranging", action="store_true")
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()

    build_bundle(args.model, args.options, args.bundle, binary=args.binary,
                ranging=args.ranging, time_limit=args.time_limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
