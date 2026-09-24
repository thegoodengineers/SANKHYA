#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fetch Netlib's INFEASIBLE LP collection (#529), with recorded provenance.

    https://netlib.org/lp/infeas/          (the collection, John W. Chinneck, 1993)
    https://netlib.org/lp/infeas/readme    (the SUMMARY OF INFEASIBLE LPs this script parses)

Every instance in this directory is infeasible by construction, so there is no optimal value
to publish and none is recorded. What the readme does publish is a PROBLEM SUMMARY TABLE of
names, row, column and nonzero counts and bound flags; the instance list is PARSED from that
table rather than typed in, for the same reason fetch_data.py parses the feasible set's
optima: if the collection and this script ever disagree, the disagreement has to be visible
rather than papered over by a hand-kept list.

The files are in Netlib's packed encoding, the same one lp/data uses, and are expanded with
Netlib's own emps.c. The decoder is downloaded and compiled at fetch time by
fetch_data.build_emps(), never vendored, exactly as for the feasible set; its sha256 is
recorded here too. Everything downloaded has its sha256 in data/netlib-infeasible/
reference.json, along with the URL it came from.

    python bench/runners/fetch_netlib_infeasible.py            # all of them
    python bench/runners/fetch_netlib_infeasible.py itest2     # named instances
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_data  # noqa: E402  - download(), sha256(), build_emps(), decompress()

INFEAS_BASE = "https://netlib.org/lp/infeas"
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "netlib-infeasible"

# The flags column of the table. Anything after the leading run of these is the free-text
# "Notes" column ("dense col (> 967)", "all cols are LO bounded"), kept verbatim.
FLAG_TOKENS = {"B", "FR", "FX", "MI", "PL", "UP"}

# name  rows  cols  nonzeros  [flags] [notes]
SUMMARY_ROW = re.compile(r"^([a-z0-9]+)\s+(\d+)\s+(\d+)\s+(\d+)(.*)$")


def parse_summary_table(readme: str) -> dict[str, dict]:
    """The PROBLEM SUMMARY TABLE of lp/infeas/readme, one entry per instance.

    Bracketed by its heading and by the REFERENCES heading that follows it. The readme is
    mostly prose, and prose can contain "name 12 34 56"-shaped lines (the contact block
    alone carries a telephone number); matching only inside the table keeps them out. The
    bracket is the heading, not the first non-matching line: fetch_data.py learned the hard
    way that stopping at the first odd line silently truncates a table.
    """
    entries: dict[str, dict] = {}
    inside = False
    for line in readme.splitlines():
        stripped = line.strip()
        if stripped.startswith("PROBLEM SUMMARY TABLE"):
            inside = True
            continue
        if inside and stripped.startswith("REFERENCES"):
            break
        if not inside:
            continue
        match = SUMMARY_ROW.match(stripped)
        if not match:
            continue
        name, rows, cols, nonzeros, rest = match.groups()
        tokens = rest.split()
        flags: list[str] = []
        while tokens and tokens[0] in FLAG_TOKENS:
            flags.append(tokens.pop(0))
        entries[name] = {
            "name": name,
            "published_rows": int(rows),
            "published_cols": int(cols),
            "published_nonzeros": int(nonzeros),
            "published_flags": " ".join(flags),
            "published_notes": " ".join(tokens),
            # Not a number anyone measured: the collection exists because every member is
            # infeasible, and the runner's pass criterion is a certificate of exactly that.
            "expected_status": "infeasible",
        }
    return entries


def lf_sha256(data: bytes) -> str:
    """sha256 of the bytes with CRLF folded to LF.

    emps writes CRLF on Windows and LF elsewhere (see .gitattributes), so `mps_sha256` - the
    hash of the file as written, which is what every CSV row records - differs by platform
    for the same instance. This one does not, so a manifest fetched on one machine can still
    say whether another machine decoded the same instance.
    """
    return fetch_data.sha256(data.replace(b"\r\n", b"\n"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("instances", nargs="*", help="instance names (default: all of them)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    readme_url = f"{INFEAS_BASE}/readme"
    print(f"reading the summary table from {readme_url}")
    readme_bytes = fetch_data.download(readme_url)
    published = parse_summary_table(readme_bytes.decode("latin-1"))
    print(f"  {len(published)} instances listed")
    if not published:
        raise SystemExit("could not parse the summary table; the readme format may have changed")

    wanted = [name.lower() for name in args.instances] or sorted(published)
    unknown = [name for name in wanted if name not in published]
    if unknown:
        raise SystemExit(f"not in the lp/infeas summary table: {', '.join(unknown)}")

    manifest_path = DATA_DIR / "reference.json"
    manifest: dict = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"] = INFEAS_BASE
    manifest.setdefault("instances", {})
    # A full fetch REPLACES the instance list, for the reason fetch_data.py's tiers do: a
    # manifest that accumulates entries the table no longer lists would be run and reported
    # as members of a set they are not in.
    if not args.instances:
        manifest["instances"] = {}
        manifest["instance_set"] = "all"
    else:
        manifest["instance_set"] = "explicit"
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
            url = f"{INFEAS_BASE}/{name}"
            try:
                packed_bytes = fetch_data.download(url)
            except Exception as error:  # noqa: BLE001 - report and continue
                print(f"  {name:<10} DOWNLOAD FAILED: {error}")
                failures.append(name)
                continue
            packed = work / name
            packed.write_bytes(packed_bytes)
            target = DATA_DIR / f"{name}.mps"
            fetch_data.decompress(emps, packed, target)
            mps_bytes = target.read_bytes()
            entry.update({
                "url": url,
                "packed_sha256": fetch_data.sha256(packed_bytes),
                "packed_bytes": len(packed_bytes),
                "mps_sha256": fetch_data.sha256(mps_bytes),
                "mps_lf_sha256": lf_sha256(mps_bytes),
                "mps_bytes": len(mps_bytes),
            })
            manifest["instances"][name] = entry
            print(f"  {name:<10} {entry['published_rows']:>6} rows "
                  f"{entry['published_cols']:>6} cols {entry['published_nonzeros']:>7} nz   "
                  f"{len(mps_bytes) / 1e3:>8.1f} kB MPS")

    manifest["instances"] = dict(sorted(manifest["instances"].items()))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8", newline="\n")
    print(f"\nwrote {manifest_path.relative_to(REPO_ROOT)}")
    print(f"{len(wanted) - len(failures)} instance(s) in {DATA_DIR.relative_to(REPO_ROOT)}")
    if failures:
        print(f"failed to fetch: {', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
