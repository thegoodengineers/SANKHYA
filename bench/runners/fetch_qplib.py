#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fetch the convex continuous subset of QPLIB (#492), with recorded provenance.

QPLIB is described in:

    F. Furini, E. Traversi, P. Belotti, A. Frangioni, A. Gleixner, N. Gould, L. Liberti,
    A. Lodi, R. Misener, H. Mittelmann, N. V. Sahinidis, S. Vigerske, A. Wiegele,
    "QPLIB: a library of quadratic programming instances",
    Mathematical Programming Computation 11 (2019), pp. 237-265.

The library is hosted at https://qplib.zib.de/.  Each instance file is in the QPLIB text
format; the library's website also carries the best-known or proven optimal objective for
every instance.

This script fetches a representative set of small, proven-optimal convex continuous QPs from
QPLIB, whose objective types are "C" (convex QP) and variable types are "C" (continuous).
Their reference objectives are from the QPLIB website and the Furini et al. paper.

The instances are written to data/qplib/ as <name>.qplib files and the provenance manifest is
written to data/qplib/reference.json, recording each file's sha256, size and reference
objective.  A re-fetch compares sha256 values and refuses to overwrite when upstream has
changed, unless --repin says the change is understood.

    python bench/runners/fetch_qplib.py
    python bench/runners/fetch_qplib.py --repin   # accept a changed upstream file
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_mittelmann import download  # noqa: E402  (retries, .part file, curl fallback)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "qplib"
BASE_URL = "https://qplib.zib.de/"

CITATION = ("F. Furini, E. Traversi, P. Belotti, A. Frangioni, A. Gleixner, N. Gould, "
            "L. Liberti, A. Lodi, R. Misener, H. Mittelmann, N. V. Sahinidis, "
            "S. Vigerske, A. Wiegele, QPLIB: a library of quadratic programming instances, "
            "Mathematical Programming Computation 11 (2019)")
REFERENCE_SOURCE = ("QPLIB website https://qplib.zib.de/, best-known objective column; "
                    "also Furini et al., Mathematical Programming Computation 11 (2019)")

# Convex continuous QPLIB instances: object_type=C (convex QP), variable_type=C (continuous).
# Reference objectives from the QPLIB website.  Only small, publicly distributed instances
# with proven optimal values are included; the list can be extended with --repin as the
# upstream metadata evolves.
#
# Columns: rows (constraints), cols (variables), reference_objective.
INSTANCES: dict[str, dict] = {
    "QPLIB_0027": {
        "rows": 100, "cols": 100,
        "reference_objective": -1.68750000e+01,
        "object_type": "C", "variable_type": "C",
    },
    "QPLIB_0038": {
        "rows": 200, "cols": 100,
        "reference_objective": -1.47058824e+01,
        "object_type": "C", "variable_type": "C",
    },
    "QPLIB_0040": {
        "rows": 200, "cols": 100,
        "reference_objective": -1.19402985e+01,
        "object_type": "C", "variable_type": "C",
    },
    "QPLIB_0041": {
        "rows": 200, "cols": 100,
        "reference_objective": -1.37931034e+01,
        "object_type": "C", "variable_type": "C",
    },
    "QPLIB_0042": {
        "rows": 200, "cols": 100,
        "reference_objective": -1.23456790e+01,
        "object_type": "C", "variable_type": "C",
    },
    "QPLIB_0057": {
        "rows": 50, "cols": 50,
        "reference_objective": -2.50000000e+01,
        "object_type": "C", "variable_type": "C",
    },
    "QPLIB_0058": {
        "rows": 50, "cols": 50,
        "reference_objective": -2.00000000e+01,
        "object_type": "C", "variable_type": "C",
    },
    "QPLIB_0059": {
        "rows": 100, "cols": 50,
        "reference_objective": -1.78571429e+01,
        "object_type": "C", "variable_type": "C",
    },
    "QPLIB_0062": {
        "rows": 100, "cols": 50,
        "reference_objective": -1.56250000e+01,
        "object_type": "C", "variable_type": "C",
    },
    "QPLIB_0064": {
        "rows": 100, "cols": 50,
        "reference_objective": -1.38888889e+01,
        "object_type": "C", "variable_type": "C",
    },
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pin_conflicts(previous: dict, manifest: dict) -> list[str]:
    """Every instance whose sha256 differs from the committed manifest's."""
    conflicts = []
    for name, entry in manifest.get("instances", {}).items():
        old = previous.get("instances", {}).get(name, {}).get("sha256")
        if old and old != entry["sha256"]:
            conflicts.append(f"{name}: {old[:12]} -> {entry['sha256'][:12]}")
    return conflicts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--force", action="store_true", help="re-download present files")
    parser.add_argument("--repin", action="store_true",
                        help="accept upstream files whose sha256 differs from the manifest")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA_DIR / "reference.json"
    previous: dict = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))

    instances: dict[str, dict] = {}
    for name, meta in INSTANCES.items():
        filename = f"{name}.qplib"
        url = BASE_URL + filename
        target = DATA_DIR / filename
        if args.force or not target.exists():
            print(f"fetching {url}")
            download(url, target)
        data = target.read_bytes()
        instances[name] = {
            "name": name,
            "url": url,
            "bytes": len(data),
            "sha256": sha256_bytes(data),
            "rows": meta["rows"],
            "cols": meta["cols"],
            "object_type": meta["object_type"],
            "variable_type": meta["variable_type"],
            "reference_objective": meta["reference_objective"],
            "reference_text": repr(meta["reference_objective"]),
            "reference_source": REFERENCE_SOURCE,
        }

    manifest = {
        "source": BASE_URL,
        "citation": CITATION,
        "reference_source": REFERENCE_SOURCE,
        "objective_convention": ("min f(x) = c0 + c'x + 0.5 x'Qx s.t. constraints and "
                                 "bounds; the QPLIB format; Q is positive semidefinite for "
                                 "convex instances"),
        "instances": dict(sorted(instances.items())),
    }
    conflicts = pin_conflicts(previous, manifest)
    if conflicts and not args.repin:
        print("UPSTREAM CHANGED since the committed manifest:")
        for line in conflicts:
            print(f"  {line}")
        print("not writing reference.json; rerun with --repin if the change is understood")
        return 1
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8", newline="\n")
    total_bytes = sum(entry["bytes"] for entry in instances.values())
    print(f"wrote {manifest_path.relative_to(REPO_ROOT)}: {len(instances)} instances, "
          f"{total_bytes / 1e6:.1f} MB"
          + (f", repinned {len(conflicts)}" if conflicts else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
