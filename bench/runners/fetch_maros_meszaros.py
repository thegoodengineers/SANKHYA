#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fetch the Maros-Meszaros convex QP test set (#491), with recorded provenance.

The set is the one every convex QP paper reports: 138 problems in QPS format, collected in

    I. Maros and Cs. Meszaros, "A repository of convex quadratic programming problems",
    Optimization Methods and Software 11-12 (1999).

and distributed by the first author at https://www.doc.ic.ac.uk/~im/ as three zip archives
(QPDATA1: 76 problems from CUTE, QPDATA2: 46 from the Brunel group, QPDATA3: 16 from
miscellaneous sources) beside a readme, 00README.QP, whose table lists every problem's size
and its optimal objective. This script downloads all four files and records, in
data/maros-meszaros/reference.json:

*   the URL and sha256 of the readme and of each archive;
*   per instance: the archive and member it came from, the sha256 and size of the QPS file
    exactly as extracted (the files are DOS text and are not rewritten), and the readme's
    row for it - M, N, NZ, QN, QNZ and OPT.

THE REFERENCE OBJECTIVES ARE PARSED OUT OF THE README, NEVER TYPED. The readme defines its OPT
column as the "solution value obtained by the default settings of BPMPD solver", printed to
eight significant digits; the manifest carries that definition beside every value, and the
printed text as well as the float, so a reader can see exactly what the comparison is
against. BPMPD is an interior-point code of the second author's; its output is published
data here, as Netlib's readme values are for the LP set.

The archives are about 28 MB together and expand to about 235 MB, so they are NOT committed:
.gitignore excludes them and this manifest is what is committed. A re-fetch compares every
sha256 against the committed manifest and refuses to overwrite it when upstream has changed,
unless --repin says the change is understood.

    python bench/runners/fetch_maros_meszaros.py
    python bench/runners/fetch_maros_meszaros.py --repin   # accept a changed upstream file
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_mittelmann import download  # noqa: E402  (retries, a .part file, curl fallback)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "maros-meszaros"
BASE_URL = "https://www.doc.ic.ac.uk/%7Eim/"
README = "00README.QP"
ARCHIVES = ["QPDATA1.ZIP", "QPDATA2.ZIP", "QPDATA3.ZIP"]
EXPECTED_INSTANCES = 138  # "Presently, there are 138 problems available." - the readme

CITATION = ("I. Maros and Cs. Meszaros, A repository of convex quadratic programming "
            "problems, Optimization Methods and Software 11-12 (1999)")
REFERENCE_SOURCE = ("00README.QP, OPT column: 'solution value obtained by the default "
                    "settings of BPMPD solver', eight significant digits")

# One table row: NAME M N NZ QN QNZ OPT. Names are lower case with digits and hyphens
# (cont-050, hues-mod); the five counts are integers and OPT is a float in e-notation.
_ROW = re.compile(r"^([a-z0-9][a-z0-9-]*)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
                  r"([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s*$")


def parse_readme(text: str) -> dict[str, dict]:
    """The readme's problem table, keyed by the lower-case name it prints.

    Only lines that match a whole row are taken, so the prose around the table, the
    column legend and the acknowledgements (whose names are split across spaces, as in
    "cvxqp1 l") cannot contribute a row. A name seen twice is an error rather than a
    silent overwrite: two reference values for one instance means the table was misread.
    """
    table: dict[str, dict] = {}
    for line in text.splitlines():
        match = _ROW.match(line.strip())
        if not match:
            continue
        name, m, n, nz, qn, qnz, opt = match.groups()
        if name in table:
            raise ValueError(f"{README} lists {name} twice")
        table[name] = {"m": int(m), "n": int(n), "nz": int(nz), "qn": int(qn),
                       "qnz": int(qnz), "reference_text": opt,
                       "reference_objective": float(opt)}
    return table


def instance_name(member: str) -> str:
    """The readme's name for an archive member: CVXQP1_L.QPS is cvxqp1l, CONT-050.QPS is
    cont-050. The archives upper-case the names and put an underscore before the size
    letter of the cvxqp family; the readme does neither."""
    stem = member.rsplit("/", 1)[-1]
    if stem.upper().endswith(".QPS"):
        stem = stem[: -len(".QPS")]
    return stem.lower().replace("_", "")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pin_conflicts(previous: dict, manifest: dict) -> list[str]:
    """Every file whose sha256 differs from the committed manifest's. An upstream change is
    not an error in itself, but a benchmark row stamped with one file and compared against
    a manifest describing another would be, so it is surfaced rather than absorbed."""
    conflicts = []
    for name, entry in manifest.get("files", {}).items():
        old = previous.get("files", {}).get(name, {}).get("sha256")
        if old and old != entry["sha256"]:
            conflicts.append(f"{name}: {old[:12]} -> {entry['sha256'][:12]}")
    for name, entry in manifest.get("instances", {}).items():
        old = previous.get("instances", {}).get(name, {}).get("qps_sha256")
        if old and old != entry["qps_sha256"]:
            conflicts.append(f"{name}.qps: {old[:12]} -> {entry['qps_sha256'][:12]}")
    return conflicts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--force", action="store_true", help="re-download present files")
    parser.add_argument("--repin", action="store_true",
                        help="accept upstream files whose sha256 differs from the manifest")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA_DIR / "reference.json"
    previous = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))

    files: dict[str, dict] = {}
    for filename in [README, *ARCHIVES]:
        target = DATA_DIR / filename
        url = BASE_URL + filename
        if args.force or not target.exists():
            print(f"fetching {url}")
            download(url, target)
        files[filename] = {"url": url, "bytes": target.stat().st_size,
                           "sha256": sha256_file(target)}

    table = parse_readme((DATA_DIR / README).read_text(encoding="latin-1"))
    if len(table) != EXPECTED_INSTANCES:
        raise SystemExit(f"{README} yields {len(table)} table rows, expected "
                         f"{EXPECTED_INSTANCES}; refusing to write a partial manifest")

    instances: dict[str, dict] = {}
    for archive in ARCHIVES:
        with zipfile.ZipFile(DATA_DIR / archive) as bundle:
            for info in bundle.infolist():
                if info.is_dir():
                    continue
                name = instance_name(info.filename)
                if name not in table:
                    raise SystemExit(f"{archive} holds {info.filename}, which {README} "
                                     f"does not list; the name mapping is wrong")
                if name in instances:
                    raise SystemExit(f"{name} appears in {instances[name]['archive']} "
                                     f"and in {archive}")
                data = bundle.read(info)
                target = DATA_DIR / f"{name}.qps"
                # Written byte for byte: the sha256 recorded is that of the file as
                # distributed, CRLF line endings included, and both readers accept them.
                if args.force or not target.exists() or target.read_bytes() != data:
                    target.write_bytes(data)
                instances[name] = {
                    "name": name, "archive": archive, "member": info.filename,
                    "qps_bytes": len(data), "qps_sha256": sha256_bytes(data),
                    **table[name], "reference_source": REFERENCE_SOURCE,
                }
    missing = sorted(set(table) - set(instances))
    if missing:
        raise SystemExit(f"{README} lists {len(missing)} problem(s) no archive holds: "
                         + ", ".join(missing))

    manifest = {
        "source": "https://www.doc.ic.ac.uk/~im/",
        "citation": CITATION,
        "reference_source": REFERENCE_SOURCE,
        "objective_convention": ("min c_0 + c'x + 0.5 x'Qx s.t. constraints and bounds, "
                                 "the readme's form; OPT includes the constant c_0"),
        "files": files,
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
                             encoding="utf-8")
    total = sum(entry["qps_bytes"] for entry in instances.values())
    print(f"wrote {manifest_path.relative_to(REPO_ROOT)}: {len(instances)} instances, "
          f"{total / 1e6:.1f} MB of QPS" + (f", repinned {len(conflicts)}" if conflicts else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
