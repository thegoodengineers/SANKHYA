#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fetch the classic MIPLIB 3 set and the optima its own catalogue publishes (#761).

The MIPLIB 2017 subset (fetch_miplib.py) says where we stand; MIPLIB 3 is the set most of the
published record and most teaching material still reports on, so a reader comparing numbers
across sources needs this one too. It is kept apart from MIPLIB 2017 everywhere: its own
directory, its own manifest, its own CSV names and its own section of docs/BENCHMARKS.md.

    Bixby, Ceria, McZeal and Savelsbergh, "An updated mixed integer programming library:
    MIPLIB 3.0", Optima 58 (1998).

FIVE THINGS THIS IS DELIBERATE ABOUT.

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

4.  A PRINTED VALUE IS REPLACED ONLY BY A PROVEN ONE, AND ONE THAT MAY BE ROUNDED PAST THE
    RULE IS NOT GRADED AGAINST. Two checks on top of the catalogue, both recorded per
    instance as `reference_source` (and `reference_note`) in the manifest:

    ERRATA (OVERRIDES below). Where MIPLIB 2017's solution file marks the same model `=opt=`
    at a value the catalogue contradicts, that value is the reference, pinned with the file
    (sha256 below), the line it was read from, and why. Two instances:
      noswot  printed -43, which is the catalogue's own LP SOLN; miplib2017-v28.solu line 23
              proves -41.00000885 (miplib2010_all.solu line 207 and the MIPLIB 2003 table
              say -41).
      gen     printed 112313; line 294 proves 112313.3627179998, 0.36 above the print and
              outside the 1e-6 rule (0.11). The objective can be fractional (continuous
              columns and fractional integer costs carry cost), so the print was rounded.

    ROUNDING. A value printed with decimals is covered by published_tolerance: one unit in
    its last place is more than any rounding or truncation to that place leaves. A value
    printed as an integer gets no extra tolerance, so it is AT RISK when the objective can
    be fractional (some column with a nonzero cost is continuous, or an integer column's
    cost is not integral; read with tools/verify_solution_mps.py) and one unit exceeds the
    1e-6 rule, i.e. |value| < 1e6. When every nonzero cost is an integer on an integer
    column the optimum is an integer and the print is exact (`objective_integral` in the
    manifest). Of the at-risk instances, 10teams, dcmulti, misc07 and pk1 are CONFIRMED by
    a proven value of the same model in miplib2017-v28.solu (CONFIRMED below) and keep the
    catalogue's value; fixnet6, misc03, pp08a and pp08aCUTS have no proven value with more
    digits in miplib2017-v28.solu or miplib2010_all.solu (the MIPLIB 2003 table prints six
    significant figures only) and are EXCLUDED rather than graded against a possibly
    rounded number. gen is at risk too, and overridden above.

    SAME MODEL, checked on 3 Oct 2026 for every override and confirmation: both files read
    in order with every row and column name replaced by the order of its first appearance,
    and every section, row type, marker, column entry, right-hand side and bound compared.
    gen, 10teams, dcmulti, misc07 and pk1 are identical up to names (780x870, 230x2025,
    290x548, 212x260 and 45x86); noswot (182x128, 100 integer) is identical up to names
    except 50 coefficient lines that differ by at most 6e-9 relative (-0.6667 in MIPLIB
    2017 against -0.666700006 here, and the like).

5.  THE BYTES WE SOLVE ARE THE BYTES IN THE ARCHIVE. Each model file opens with a block of
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
CATALOGUE_SOURCE = "miplib.cat"

# The later proven optima this file reads, as fetched on 3 Oct 2026 (point 4 above).
SOLU_2017 = "https://miplib.zib.de/downloads/miplib2017-v28.solu"
SOLU_2017_SHA256 = "61535544c02c300c583bcd58315039137c0f2d0d3d7206d132535a60e28c3814"
SOLU_2010 = "https://miplib2010.zib.de/download/miplib2010_all.solu"
SOLU_2010_SHA256 = "f874684e6bdeb4820b955e3030611839ed0face8cdf39c066b6c15cf3a43fd57"


def _pinned(name: str, text: str, line: int, same_model: str, why: str = "") -> dict:
    return {"text": text, "source": SOLU_2017, "source_sha256": SOLU_2017_SHA256,
            "line": line, "record": f"=opt= {name} {text}", "same_model": same_model,
            "why": why}


# Catalogue errata: the reference is the later proven optimum of the same model.
OVERRIDES = {
    "noswot": _pinned(
        "noswot", "-41.00000885", 23,
        "182x128, 100 integer, identical up to names except 50 coefficient lines differing "
        "by at most 6e-9 relative",
        "the catalogue prints -43, which is its own LP SOLN; MIPLIB 2017 proves -41 for the "
        f"same model ({SOLU_2010} line 207 also says =opt= -41)"),
    "gen": _pinned(
        "gen", "112313.3627179998", 294, "780x870, 150 integer, identical up to names",
        "the catalogue prints 112313, a rounding: the objective can be fractional and the "
        "proven optimum is 0.36 above it, outside the 1e-6 rule (0.11)"),
}
# Catalogue values at rounding risk that a proven optimum of the same model confirms within
# the 1e-6 rule; the catalogue's value is kept.
CONFIRMED = {
    "10teams": _pinned("10teams", "923.9999999999997", 210,
                       "230x2025, 1800 integer, identical up to names"),
    "dcmulti": _pinned("dcmulti", "188182", 248, "290x548, 75 integer, identical up to names"),
    "misc07": _pinned("misc07", "2810", 79, "212x260, 259 integer, identical up to names"),
    "pk1": _pinned("pk1", "11", 73, "45x86, 55 integer, identical up to names"),
}
ROUNDING_EXCLUSION = (
    "the catalogue prints an integer and the objective can be fractional, so a rounding of up "
    "to one unit is more than the 1e-6 rule allows at this size; no proven value with more "
    f"digits in {SOLU_2017} or {SOLU_2010}")


def objective_integral(costs, integer, offset: float = 0.0) -> bool:
    """True when every nonzero objective coefficient sits on an integer column and is itself
    an integer (and the constant is too): then every feasible point's objective, the optimum
    included, is an integer, and an integer print of it is exact."""
    return offset == round(offset) and all(
        c == 0 or (is_int and c == round(c)) for c, is_int in zip(costs, integer))


def rounding_risk(text: str, value: float, integral: bool) -> bool:
    """True when a correct optimum could miss the printed `text`: the objective can be
    fractional, the print is an integer (published_tolerance 0), and one unit, the most a
    rounding or truncation to an integer can be off, exceeds 1e-6 * max(1, |value|). A print
    with decimals is covered by published_tolerance, one unit in its last place."""
    return (not integral and published_tolerance(text) == 0.0
            and 1.0 > 1e-6 * max(1.0, abs(value)))


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
                   catalogue_bytes: bytes, integral: dict[str, bool]) -> dict:
    """The manifest from the archive's models, its parsed catalogue and, per model with a
    usable catalogue row, whether its objective is integral (objective_integral). Pure.

    `instances` is what miplib.py and compare_suite.py run against: the models with a
    catalogue row not marked "(not opt)", less those at rounding risk with no confirmation,
    with OVERRIDES applied. `excluded` is every other model, with its reason. The shape
    follows data/miplib/reference.json (name, url, published_optimal), with the sha256 of
    the uncompressed bytes in place of the .gz digest."""
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
        elif name in OVERRIDES:
            pin = OVERRIDES[name]
            instances[name] = {
                **record, "published_optimal": float(pin["text"]), "published_text": pin["text"],
                "published_tolerance": published_tolerance(pin["text"]),
                "objective_integral": integral[name], "catalogue_int_soln": row["int_soln"],
                "reference_source": "override",
                "reference_note": (f"{pin['why']}; {pin['source']} line {pin['line']}: "
                                   f"{pin['record']}; same model: {pin['same_model']}")}
        elif rounding_risk(row["int_soln"], float(row["int_soln"]), integral[name]):
            if name not in CONFIRMED:
                excluded[name] = {**record, "reason": ROUNDING_EXCLUSION,
                                  "catalogue_int_soln": row["int_soln"],
                                  "objective_integral": False}
                continue
            pin = CONFIRMED[name]
            printed, proven = float(row["int_soln"]), float(pin["text"])
            if abs(printed - proven) > 1e-6 * max(1.0, abs(proven)):
                raise ValueError(f"{name}: CONFIRMED {proven} does not confirm {printed}")
            instances[name] = {
                **record, "published_optimal": printed, "published_text": row["int_soln"],
                "published_tolerance": 0.0, "objective_integral": False,
                "reference_source": CATALOGUE_SOURCE,
                "reference_note": (f"printed as an integer with a possibly fractional "
                                   f"objective; confirmed by {pin['source']} line "
                                   f"{pin['line']}: {pin['record']}; same model: "
                                   f"{pin['same_model']}")}
        else:
            instances[name] = {**record, "published_optimal": float(row["int_soln"]),
                               "published_text": row["int_soln"],
                               "published_tolerance": published_tolerance(row["int_soln"]),
                               "objective_integral": integral[name],
                               "reference_source": CATALOGUE_SOURCE}
    return {
        "source": ARCHIVE_URL,
        "archive_bytes": ARCHIVE_BYTES,
        "archive_sha256": ARCHIVE_SHA256,
        "catalogue": CATALOGUE,
        "catalogue_sha256": hashlib.sha256(catalogue_bytes).hexdigest(),
        "citation": CITATION,
        "selection": "miplib3: every model whose INT SOLN in INDEX PART A is not marked "
                     "(not opt), less those whose integer print may be a rounding past the "
                     "match rule with no proven value to confirm it; catalogue errata replaced "
                     "by the proven optimum of the same model (fetch_miplib3.py, point 4)",
        "later_optima": {SOLU_2017: SOLU_2017_SHA256, SOLU_2010: SOLU_2010_SHA256},
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

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in models.items():
        write(DATA_DIR / f"{name}.mps", data)
    write(DATA_DIR / CATALOGUE, catalogue_bytes)
    # The objective's integrality, from the verifier's own reader on the files just written.
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from verify_solution_mps import parse_mps  # noqa: PLC0415
    integral = {}
    for name, row in catalogue.items():
        if not row["not_opt"]:
            model = parse_mps(DATA_DIR / f"{name}.mps")
            integral[name] = objective_integral(model.col_cost, model.col_integer,
                                                model.objective_offset)
    manifest = build_manifest(models, catalogue, catalogue_bytes, integral)

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
        if entry.get("reference_note"):
            print(f"  {entry['reference_source']:<8} {name:<12} {entry['reference_note']}")
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8", newline="\n")
    print()
    print(f"wrote {len(models)} models and {CATALOGUE} to {DATA_DIR.relative_to(REPO_ROOT)}")
    print(f"wrote {MANIFEST.relative_to(REPO_ROOT)}: {len(manifest['instances'])} with a "
          f"published optimum, {len(manifest['excluded'])} excluded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
