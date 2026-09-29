#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fetch the EXACT optimal objective of every Netlib LP, with recorded provenance (#747).

Netlib's own readme table is a list of floating-point values printed by the solvers of 1985
to 1995, and on eight instances it is wrong (README, "The eight verified answers that do not
match the readme match the exact optimum"). The exact values were computed once, in rational
arithmetic, by Koch's perPlex:

    T. Koch, "The final NETLIB-LP results", Operations Research Letters 32(2), 138-142, 2004
    (preprint ZIB-Report 03-05, 2003).

The paper prints each value to 32 digits. The same 32-digit values are in perPlex's own run
log, published beside the program at http://www.zib.de/koch/perplex/data/netlib/txt/
perplex.log. That page is no longer served by ZIB; the copy fetched here is the Internet
Archive's capture of it (the URL below, the `id_` form, which returns the file as archived
rather than wrapped in the archive's page). Its sha256 is pinned: a different file is a
different source and must be looked at, not silently parsed.

Nothing in the output is typed by hand. For each instance the log says "Solution is optimal"
and "Solution is feasible" after the rational check, then "objective = <32 digits>"; the
script refuses an instance that lacks any of the three. Three instance names differ from ours
only by punctuation (pilot-ja, pilot-we, vtp-base in the log; pilot.ja, pilot.we, vtp.base
in data/netlib), and are mapped explicitly.

The values EXCLUDE the objective-row constant, as Netlib's table does (e226 is -18.7519...,
the table's value, not the -11.6389... a solver reports with the constant included); the
runner compares the same way it compares against the table.

    python bench/runners/fetch_koch_exact.py      # writes data/netlib/koch_exact.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_data  # noqa: E402  - download(), sha256()

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "data" / "netlib" / "koch_exact.json"

ORIGINAL_URL = "http://www.zib.de/koch/perplex/data/netlib/txt/perplex.log"
ARCHIVED_URL = ("https://web.archive.org/web/20051220030941id_/"
                "http://www.zib.de/koch/perplex/data/netlib/txt/perplex.log")
LOG_SHA256 = "06af2176cd5faedd399b7e220c67dd5a593084825a65757a7bd4d29fd93682a2"
CITATION = ('T. Koch, "The final NETLIB-LP results", Operations Research Letters 32(2), '
            "138-142, 2004 (ZIB-Report 03-05, 2003)")

# The log's names that differ from data/netlib's only by punctuation.
RENAME = {"pilot-ja": "pilot.ja", "pilot-we": "pilot.we", "vtp-base": "vtp.base"}


def parse(log: str) -> dict[str, str]:
    """Instance name -> exact objective as the log prints it (a decimal string)."""
    values: dict[str, str] = {}
    for block in re.split(r"(?=^Name: )", log, flags=re.M):
        name = re.match(r"Name: (\S+)", block)
        if not name:
            continue
        objective = re.search(r"objective = (\S+)", block)
        if not ("Solution is optimal" in block and "Solution is feasible" in block
                and objective):
            raise SystemExit(f"{name.group(1)}: the log does not show a verified optimum")
        values[RENAME.get(name.group(1), name.group(1))] = objective.group(1)
    return values


def main() -> int:
    data = fetch_data.download(ARCHIVED_URL)
    digest = fetch_data.sha256(data)
    if digest != LOG_SHA256:
        raise SystemExit(f"perplex.log sha256 is {digest}, expected {LOG_SHA256}: a different "
                         "file, look at it before trusting it")
    values = parse(data.decode("ascii"))
    OUT.write_text(json.dumps({
        "citation": CITATION,
        "original_url": ORIGINAL_URL,
        "archived_url": ARCHIVED_URL,
        "log_sha256": digest,
        "note": ("exact optimal objective per instance, as perPlex printed it after its "
                 "rational feasibility and optimality check; the objective-row constant is "
                 "excluded, as in Netlib's readme table"),
        "instances": {name: {"exact_objective": values[name]} for name in sorted(values)},
    }, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(REPO_ROOT)}: {len(values)} instances, log sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
