#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fetch the Kennington LP set (#530) and its PUBLISHED optimal values.

    https://netlib.org/lp/data/kennington/          (the sixteen problems)
    https://netlib.org/lp/data/kennington/readme    (the table this script parses)

Sixteen LPs from Carolan, Hill, Kennington, Niemi and Wichmann, "An Empirical Evaluation of
the KORBX Algorithms for Military Airlift Applications", Operations Research 38(2), 1990:
the CRE, KEN, OSA and PDS families. They are larger and sparser than the core Netlib set and
are the usual next rung above it.

The same two rules as fetch_data.py, for the same reason:

1.  The reference optima are PARSED from the directory's own readme, never typed in. The
    readme says they were computed by Vanderbei's ALPO in IEEE double precision and prints
    them to eight significant figures; both facts are recorded in the manifest, because a
    relative gap below 1e-8 against an eight-figure number says nothing.

2.  The files are "doubly compressed": gzip, then Netlib's packed encoding. The gzip layer is
    Python's own; the packed layer is expanded by Netlib's emps.c, downloaded and compiled at
    fetch time by fetch_data.build_emps(), never vendored. The sha256 of the .gz as
    downloaded, of the packed file and of the decoded MPS are all recorded in
    data/kennington/reference.json with the source URL.

Sets, both defined from the PUBLISHED table rather than a hand-kept list:

    full    all sixteen (the default; osa-60 alone decodes to 52 MB)
    small   published MPS size <= 1 MB: cre-a, cre-c, ken-07, pds-02 as of the current readme

    python bench/runners/fetch_kennington.py
    python bench/runners/fetch_kennington.py --set small
    python bench/runners/fetch_kennington.py pds-02 ken-07
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_data  # noqa: E402  - download(), sha256(), build_emps(), decompress()

KENNINGTON_BASE = "https://netlib.org/lp/data/kennington"
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "kennington"

SMALL_MAX_MPS_BYTES = 1_000_000
SET_NAMES = ("small", "full")

# Name  rows  columns  nonzeros  bounds  mpc  MPS  optimal value
#   CRE-A      3517    4067     19054        0    152726    659682   2.3595407e+07
SUMMARY_ROW = re.compile(
    r"^([A-Z]+-[A-Z0-9]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
    r"(-?\d+\.\d+[eE][+-]\d+)\s*$"
)


def significant_digits(text: str) -> int:
    """Significant figures in a published value such as `-6.7952044e+08` (eight)."""
    mantissa = text.lstrip("+-").split("e")[0].split("E")[0]
    return len(mantissa.replace(".", "").lstrip("0"))


def parse_summary_table(readme: str) -> dict[str, dict]:
    """The readme's statistics table, one entry per instance, keyed by lower-case name.

    Scans every line, like fetch_data.parse_summary_table(), rather than bracketing the
    table: the row shape (a FAMILY-NN name, six integers and a value in e-notation) is
    specific enough that prose cannot match it, and bracketing is what once truncated the
    feasible set's table at its first odd row.
    """
    entries: dict[str, dict] = {}
    for line in readme.splitlines():
        match = SUMMARY_ROW.match(line.strip())
        if not match:
            continue
        name, rows, cols, nonzeros, bounds, mpc, mps, optimal = match.groups()
        entries[name.lower()] = {
            "name": name.lower(),
            "published_rows": int(rows),
            "published_cols": int(cols),
            "published_nonzeros": int(nonzeros),
            "published_bounds": int(bounds),
            "published_mpc_bytes": int(mpc),
            "published_mps_bytes": int(mps),
            "published_optimal": float(optimal),
            "published_optimal_text": optimal,
            "published_significant_digits": significant_digits(optimal),
            # The readme's own words: "computed by Vanderbei's ALPO, running on an SGI
            # computer (with binary IEEE arithmetic)". An interior-point value, not a vertex.
            "optimal_source": "Vanderbei's ALPO (readme)",
        }
    return entries


def select(published: dict[str, dict], chosen_set: str) -> list[str]:
    if chosen_set == "small":
        return sorted(name for name, entry in published.items()
                      if entry["published_mps_bytes"] <= SMALL_MAX_MPS_BYTES)
    return sorted(published)


def lf_sha256(data: bytes) -> str:
    """sha256 with CRLF folded to LF: emps writes CRLF on Windows and LF elsewhere, so the
    hash of the file as written (`mps_sha256`, which the CSV records) differs by platform
    for the same instance, and this one does not."""
    return fetch_data.sha256(data.replace(b"\r\n", b"\n"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("instances", nargs="*", help="instance names, e.g. pds-02")
    parser.add_argument("--set", dest="instance_set", choices=SET_NAMES, default="full",
                        help=f"full (all sixteen) or small (published MPS size <= "
                             f"{SMALL_MAX_MPS_BYTES:,} bytes)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    readme_url = f"{KENNINGTON_BASE}/readme"
    print(f"reading the published table from {readme_url}")
    readme_bytes = fetch_data.download(readme_url)
    published = parse_summary_table(readme_bytes.decode("latin-1"))
    print(f"  {len(published)} instances listed with a published optimal value")
    if not published:
        raise SystemExit("could not parse the readme table; its format may have changed")

    if args.instances:
        wanted = [name.lower() for name in args.instances]
        chosen_set = "explicit"
    else:
        chosen_set = args.instance_set
        wanted = select(published, chosen_set)
    unknown = [name for name in wanted if name not in published]
    if unknown:
        raise SystemExit(f"not in the Kennington readme table: {', '.join(unknown)}")
    print(f"  set '{chosen_set}': {len(wanted)} of {len(published)}")

    manifest_path = DATA_DIR / "reference.json"
    manifest: dict = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"] = KENNINGTON_BASE
    manifest.setdefault("instances", {})
    # A named set replaces the instance list (fetch_data.py's rule): the runner runs what the
    # manifest lists, and a set name over leftover entries would mislabel the CSV.
    if chosen_set != "explicit":
        manifest["instances"] = {}
    manifest["instance_set"] = chosen_set
    manifest["available_instances"] = len(published)
    manifest["readme_sha256"] = fetch_data.sha256(readme_bytes)

    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        print("building Netlib's emps decoder")
        emps, emps_digest = fetch_data.build_emps(work)
        manifest["emps_sha256"] = emps_digest
        print(f"  emps.c sha256 {emps_digest}")

        for name in wanted:
            entry = dict(published[name])
            url = f"{KENNINGTON_BASE}/{name}.gz"
            try:
                gz_bytes = fetch_data.download(url)
            except Exception as error:  # noqa: BLE001 - report and continue
                print(f"  {name:<8} DOWNLOAD FAILED: {error}")
                failures.append(name)
                continue
            try:
                packed_bytes = gzip.decompress(gz_bytes)
            except (OSError, EOFError) as error:
                print(f"  {name:<8} GUNZIP FAILED: {error}")
                failures.append(name)
                continue
            packed = work / name
            packed.write_bytes(packed_bytes)
            target = DATA_DIR / f"{name}.mps"
            fetch_data.decompress(emps, packed, target)
            mps_bytes = target.read_bytes()
            entry.update({
                "url": url,
                "gz_sha256": fetch_data.sha256(gz_bytes),
                "gz_bytes": len(gz_bytes),
                "packed_sha256": fetch_data.sha256(packed_bytes),
                "mps_sha256": fetch_data.sha256(mps_bytes),
                "mps_lf_sha256": lf_sha256(mps_bytes),
                "mps_bytes": len(mps_bytes),
            })
            manifest["instances"][name] = entry
            print(f"  {name:<8} {entry['published_rows']:>7} rows "
                  f"{entry['published_cols']:>7} cols {entry['published_nonzeros']:>8} nz   "
                  f"optimal {entry['published_optimal_text']:>15}   "
                  f"{len(mps_bytes) / 1e6:>6.1f} MB MPS")
            del packed_bytes, mps_bytes, gz_bytes

    manifest["instances"] = dict(sorted(manifest["instances"].items()))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(f"\nwrote {manifest_path.relative_to(REPO_ROOT)}")
    print(f"{len(wanted) - len(failures)} instance(s) in {DATA_DIR.relative_to(REPO_ROOT)}")
    if failures:
        print(f"failed to fetch: {', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
