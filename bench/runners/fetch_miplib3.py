#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fetch the classic MIPLIB 3 set and the optima its own catalogue publishes (#761).

The MIPLIB 2017 subset (fetch_miplib.py) says where we stand; MIPLIB 3 is the set most of the
published record and most teaching material still reports on, so a reader comparing numbers
across sources needs this one too. It is kept apart from MIPLIB 2017 everywhere: its own
directory, its own manifest, its own CSV names and its own section of docs/BENCHMARKS.md.

    Bixby, Ceria, McZeal and Savelsbergh, "An updated mixed integer programming library:
    MIPLIB 3.0", Optima 58 (1998).

FOUR THINGS THIS IS DELIBERATE ABOUT.

1.  ONE PINNED ARCHIVE. ZIB still serves the 1998 set as one tarball. Its size and sha256
    as fetched on 3 Oct 2026 are written below, and any other bytes are refused: a
    reference optimum is only a reference for the model it was published with.

2.  ONLY OPTIMA THE CATALOGUE DOES NOT QUALIFY. INDEX PART A of `miplib.cat` gives each
    model's INT SOLN, and marks three of them "(not opt)": arki001, dano3mip and seymour.
    Those get no reference, for the reason fetch_miplib.py keeps only `=opt=`: a run could
    "beat" a value that is not an optimum and look like a triumph while being wrong. Six
    models in the archive have no catalogue row at all (markshare1, markshare2, mas74, mas76,
    mkc, swath, added after the catalogue was written) and get no reference either. Both
    kinds are listed in the manifest under `excluded`, each with its reason, and fetched
    anyway, so the files are there and the denominator is stated.

3.  THE TOLERANCE THE CATALOGUE'S OWN PRINTING ALLOWS. The catalogue prints optima to the
    precision its authors chose, and at least one is truncated, not rounded: rgn is printed
    82.1999 and its optimum is 82.19999924, which misses miplib.py's 1e-6 relative rule.
    So each reference carries `published_tolerance`, ONE UNIT IN THE LAST PRINTED DECIMAL
    PLACE once trailing zeros are stripped: 82.1999 gets 1e-4, 65.67 gets 1e-2, and 7350.0,
    21166.000 and 1201500 count as integers and get 0. An objective matches when it is
    within max(MATCH_RELATIVE_TOLERANCE * max(1, |published|), published_tolerance) of the
    published value, so the rule is never tighter than MIPLIB 2017's and is looser only by
    what the printed digits cannot say. MIPLIB 2017's manifest has no such field, and its
    matching is unchanged.

    One kept reference is contradicted elsewhere and carries a `caveat` (CAVEATS below):
    noswot, printed -43, the same model MIPLIB 2017 proves at -41. The value is not changed;
    the doc says why a run may miss it.

4.  THE BYTES WE SOLVE ARE THE BYTES IN THE ARCHIVE. Each model file opens with a block of
    `*` comment lines (name, sizes, source, the best solution) before the MPS. The solver's
    reader (src/io/mps_reader.cpp) and the verifier's (tools/verify_solution_mps.py) both
    skip a line that begins with `*`, wherever it is, and both stop at ENDATA (dcmulti has a
    non-standard IMPORTANCES block after it). So nothing is stripped: each model is written
    to data/miplib3/<name>.mps exactly as archived, and the manifest's sha256 is of those
    bytes, which are what every run reads.

`data/miplib3/` is gitignored except the manifest, which is committed (#761's acceptance).

    python bench/runners/fetch_miplib3.py
    python bench/runners/fetch_miplib3.py --archive path/to/miplib3.tar.gz   # no download
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import tarfile
from decimal import Decimal, InvalidOperation
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_miplib import get  # noqa: E402  (certifi's bundle when installed, #87's reasons)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "miplib3"
MANIFEST = DATA_DIR / "manifest.json"

ARCHIVE_URL = "https://miplib2010.zib.de/miplib3/miplib3.tar.gz"
# As fetched on 3 Oct 2026. A different archive is refused, not "probably fine".
ARCHIVE_BYTES = 6315559
ARCHIVE_SHA256 = "f07be4c2b3cad25b23b0c3085ec98412524989dbd24f4d385e5ffa1e3e4c118f"
CATALOGUE = "miplib.cat"
# The archive's three text files; every other regular file in it is a model.
NOT_MODELS = {"miplib.cat", "mps_format", "references"}
CITATION = ("Bixby, Ceria, McZeal and Savelsbergh, \"An updated mixed integer programming "
            "library: MIPLIB 3.0\", Optima 58 (1998)")
TOLERANCE_RULE = (
    "published_tolerance is one unit in the last decimal place of the catalogue's printed INT "
    "SOLN after trailing zeros are stripped (0 for a value that is then an integer); an "
    "objective matches when |ours - published| <= max(1e-6 * max(1, |published|), "
    "published_tolerance).")
NOT_OPTIMAL = "(not opt)"
# A reference the selection rule keeps but other published evidence contradicts. It is NOT
# overridden: the rule above is the rule, and the value stays the catalogue's. The note goes
# into the manifest entry as `caveat`, and the doc prints it beside every failure it explains.
CAVEATS = {
    "noswot": (
        "the catalogue's -43 equals its own LP SOLN, and MIPLIB 2017's noswot, the same model "
        "(every row, column, bound and right-hand side in the same order, renamed, with 50 "
        "coefficient lines differing by at most 6e-9 relative, compared on 3 Oct 2026), is "
        "marked =opt= -41.00000885 in miplib2017-v28.solu; a run that ends at -41 is "
        "expected to miss this reference"),
}


def parse_catalogue(text: str) -> dict[str, dict]:
    """INDEX PART A of miplib.cat: name -> {"int_soln": printed text, "not_opt": bool}.

    A row is `NAME ROWS COLS INT 0/1 CONT INT-SOLN [(not opt)] LP-SOLN`. The part ends at the
    first blank line after its rows; PART B and C repeat the names and are not read. Pure, so
    the parse is unit-tested on an excerpt (test_miplib3.py)."""
    rows: dict[str, dict] = {}
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines)
                  if line.strip().upper().startswith("INDEX - PART A")), None)
    if start is None:
        raise ValueError("no 'INDEX - PART A' in the catalogue")
    seen_rows = False
    for line in lines[start + 1:]:
        fields = line.split()
        if not fields:
            if seen_rows:
                break
            continue
        if fields[0].startswith("=") or fields[0].upper() == "NAME":
            continue
        not_opt = NOT_OPTIMAL in line
        values = line.replace(NOT_OPTIMAL, " ").split()
        if len(values) < 8:
            raise ValueError(f"catalogue row with {len(values)} fields: {line.strip()!r}")
        rows[values[0]] = {"int_soln": values[6], "not_opt": not_opt}
        seen_rows = True
    if not rows:
        raise ValueError("INDEX PART A has no rows")
    return rows


def published_tolerance(text: str) -> float:
    """One unit in the last printed decimal place, trailing zeros stripped; 0 for an integer.

    82.1999 -> 1e-4, 65.67 -> 1e-2, 7350.0 -> 0, 21166.000 -> 0, 1201500 -> 0. Only digits
    after the decimal point count, so a value printed with none is held to the 1e-6 rule
    alone."""
    try:
        Decimal(text)
    except InvalidOperation as error:
        raise ValueError(f"not a number: {text!r}") from error
    if "e" in text.lower():
        raise ValueError(f"exponent notation is not in this catalogue: {text!r}")
    decimals = text.split(".", 1)[1].rstrip("0") if "." in text else ""
    return float(Decimal(1).scaleb(-len(decimals))) if decimals else 0.0


def build_manifest(models: dict[str, bytes], catalogue: dict[str, dict],
                   catalogue_bytes: bytes) -> dict:
    """The manifest from the archive's models and its parsed catalogue. Pure.

    `instances` is what miplib.py and compare_suite.py run against: the models with a
    catalogue row not marked "(not opt)". `excluded` is every other model, with its reason.
    The shape follows data/miplib/reference.json (name, url, published_optimal), with the
    sha256 of the uncompressed bytes in place of the .gz digest."""
    missing = sorted(set(catalogue) - set(models))
    if missing:
        raise ValueError(f"catalogue rows with no model file: {', '.join(missing)}")
    instances: dict[str, dict] = {}
    excluded: dict[str, dict] = {}
    for name in sorted(models):
        blob = models[name]
        row = catalogue.get(name)
        record = {"name": name, "file": f"{name}.mps", "url": f"{ARCHIVE_URL}#miplib3/{name}",
                  "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
        if row is None:
            excluded[name] = {**record, "reason": "no row in miplib.cat INDEX PART A"}
        elif row["not_opt"]:
            excluded[name] = {**record, "reason": "INT SOLN marked (not opt) in miplib.cat",
                              "catalogue_int_soln": row["int_soln"]}
        else:
            instances[name] = {**record, "published_optimal": float(row["int_soln"]),
                               "published_text": row["int_soln"],
                               "published_tolerance": published_tolerance(row["int_soln"])}
            if name in CAVEATS:
                instances[name]["caveat"] = CAVEATS[name]
    return {
        "source": ARCHIVE_URL,
        "archive_bytes": ARCHIVE_BYTES,
        "archive_sha256": ARCHIVE_SHA256,
        "catalogue": CATALOGUE,
        "catalogue_sha256": hashlib.sha256(catalogue_bytes).hexdigest(),
        "citation": CITATION,
        "selection": "miplib3: every model whose INT SOLN in INDEX PART A is not marked "
                     "(not opt)",
        "tolerance_rule": TOLERANCE_RULE,
        "models_in_archive": len(models),
        "instances": instances,
        "excluded": excluded,
    }


def read_archive(blob: bytes) -> tuple[dict[str, bytes], bytes]:
    """The model files and the catalogue, by base name. Only regular files directly under
    `miplib3/` are read; nothing is extracted by the archive's own paths."""
    models: dict[str, bytes] = {}
    catalogue = None
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            parts = member.name.split("/")
            if len(parts) != 2 or parts[0] != "miplib3" or not re.fullmatch(r"[\w.-]+",
                                                                           parts[1]):
                raise ValueError(f"unexpected archive member {member.name!r}")
            data = archive.extractfile(member).read()
            if parts[1] == CATALOGUE:
                catalogue = data
            elif parts[1] not in NOT_MODELS:
                models[parts[1]] = data
    if catalogue is None:
        raise ValueError(f"no {CATALOGUE} in the archive")
    return models, catalogue


def check_archive(blob: bytes) -> None:
    digest = hashlib.sha256(blob).hexdigest()
    if len(blob) != ARCHIVE_BYTES or digest != ARCHIVE_SHA256:
        raise ValueError(f"archive is {len(blob)} bytes, sha256 {digest}; expected "
                         f"{ARCHIVE_BYTES} bytes, sha256 {ARCHIVE_SHA256}. Refusing it: the "
                         f"published optima belong to the pinned archive")


def write(path: Path, data: bytes) -> None:
    # A scratch path and a rename, so an interrupted fetch cannot leave a truncated model
    # behind that looks like a real one (#87).
    scratch = path.with_suffix(path.suffix + ".partial")
    try:
        scratch.write_bytes(data)
        scratch.replace(path)
    finally:
        scratch.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, default=None,
                        help="a local copy of miplib3.tar.gz instead of the download; it must "
                             "be the pinned archive")
    args = parser.parse_args()

    blob = args.archive.read_bytes() if args.archive else get(ARCHIVE_URL)
    try:
        check_archive(blob)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    print(f"  {ARCHIVE_URL}: {len(blob)} bytes, sha256 {ARCHIVE_SHA256} (pinned)")
    models, catalogue_bytes = read_archive(blob)
    catalogue = parse_catalogue(catalogue_bytes.decode("ascii", errors="replace"))
    manifest = build_manifest(models, catalogue, catalogue_bytes)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in models.items():
        write(DATA_DIR / f"{name}.mps", data)
    write(DATA_DIR / CATALOGUE, catalogue_bytes)

    print()
    print(f"  {'instance':<12}{'bytes':>10}  {'published':<16}{'tolerance':>10}")
    for name, entry in manifest["instances"].items():
        tolerance = entry["published_tolerance"]
        print(f"  {name:<12}{entry['bytes']:>10}  {entry['published_text']:<16}"
              f"{(f'{tolerance:g}' if tolerance else '-'):>10}")
    print()
    for name, entry in manifest["excluded"].items():
        print(f"  excluded {name:<12} {entry['reason']}")
    for name, entry in manifest["instances"].items():
        if "caveat" in entry:
            print(f"  caveat   {name:<12} {entry['caveat']}")
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8", newline="\n")
    print()
    print(f"wrote {len(models)} models and {CATALOGUE} to {DATA_DIR.relative_to(REPO_ROOT)}")
    print(f"wrote {MANIFEST.relative_to(REPO_ROOT)}: {len(manifest['instances'])} with a "
          f"published optimum, {len(manifest['excluded'])} excluded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
