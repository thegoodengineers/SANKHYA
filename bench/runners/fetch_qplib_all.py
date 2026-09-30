#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fetch EVERY QPLIB instance and convert it to QPS, for the whole-library run (#835).

fetch_qplib.py (#492) selects the convex continuous QPs with linear constraints and nothing
else. This takes every row of https://qplib.zib.de/instances.html - 453 when #835 was
opened - whatever its class: integer and binary columns, nonconvex objectives and quadratic
constraints included, because the point of #835 is to see what the reader and the solve
dispatcher do with each of them, refusals and all.

The site's pages, the listing, qplib.solu and the conversion check at QPLIB's published
point are fetch_qplib.py's, reused unchanged; the manifest is data/qplib/all.json (ignored by
git, like the instance files; the run's CSV records the sha256 of every instance it read).

A row per instance, never a silently shorter list: an instance whose .qplib file does not
parse, or whose conversion is refused (a constraint with no finite side, say), is recorded
with the reason as `conversion_error` and the runner reports it as not read.

    python bench/runners/fetch_qplib_all.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_qplib as fq  # noqa: E402
import qplib_format  # noqa: E402

MANIFEST = fq.DATA_DIR / "all.json"


def entry_for(row: dict, fields: dict, solu: dict) -> dict:
    tag, solu_value, solu_text = solu.get(row["name"], (None, None, None))
    return {
        "name": row["name"], "problem_type": row["o"] + row["v"] + row["c"],
        "listed_convex": row["convex"], "nvars": row["nvars"], "ncons": row["ncons"],
        "nz": row["nz"], "reference_objective": solu_value, "reference_text": solu_text,
        "reference_status": None if tag is None else f"qplib.solu tag ={tag}=",
        "page_solobjvalue_text": fields.get("solobjvalue_text"),
        "references_agree": fq.references_agree(fields.get("solobjvalue_text"), solu_value),
        "solinfeasibility": float(fields["solinfeasibility"])
        if fields.get("solinfeasibility") else None,
    }


def convert(entry: dict, qplib_path: Path, sol_path: Path | None) -> None:
    """Write data/qplib/<name>.qps and check it at the published point, into `entry`."""
    try:
        model = qplib_format.read(qplib_path)
        qps_text = qplib_format.to_qps(model)
    except ValueError as error:
        entry["conversion_error"] = str(error)[:300]
        return
    qps_path = fq.DATA_DIR / f"{entry['name']}.qps"
    qps_path.write_text(qps_text, encoding="utf-8", newline="\n")
    entry["qps"] = {"converted_by": "bench/runners/qplib_format.py",
                    "bytes": qps_path.stat().st_size, "sha256": fq.sha256_file(qps_path)}
    entry["maximize"] = model.maximize
    entry["integer_columns"] = sum(model.integer)
    entry["quadratic_row_entries"] = len(model.qi)
    if sol_path is None:
        entry["converter_check"] = {"passed": None, "note": "no published point to check at"}
        return
    try:
        entry["converter_check"] = fq.converter_check(
            qps_path, model, sol_path.read_text(encoding="latin-1"), entry["solinfeasibility"])
    except ValueError as error:
        entry["converter_check"] = {"passed": False, "note": str(error)[:300]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--force", action="store_true", help="re-download present files")
    parser.add_argument("--instances", nargs="*", help="only these (a partial manifest)")
    args = parser.parse_args()

    files = {name: fq.fetch(name, force=True)[1] for name in fq.SITE_FILES}
    home = fq.parse_home((fq.DATA_DIR / "index.html").read_text(encoding="utf-8"))
    rows = fq.parse_listing((fq.DATA_DIR / "instances.html").read_text(encoding="utf-8"))
    if len(rows) != home["discrete"] + home["continuous"]:
        raise SystemExit(f"instances.html lists {len(rows)} rows; index.html says "
                         f"{home['discrete']} discrete and {home['continuous']} continuous")
    solu = fq.parse_solu((fq.DATA_DIR / "qplib.solu").read_text(encoding="utf-8"))
    if args.instances:
        rows = [row for row in rows if row["name"] in set(args.instances)]

    instances = {}
    for row in rows:
        page_path, page_file = fq.fetch(f"{row['name']}.html", args.force)
        fields = fq.parse_instance_page(page_path.read_text(encoding="utf-8"))
        entry = entry_for(row, fields, solu)
        entry["page"] = page_file
        if fields.get("probtype") != entry["problem_type"]:
            entry["page_mismatch"] = f"probtype {fields.get('probtype')} on the page"
        if not row["qplib_path"]:
            entry["conversion_error"] = "the listing links no .qplib file"
            instances[row["name"]] = entry
            continue
        qplib_path, entry["qplib"] = fq.fetch(row["qplib_path"], args.force)
        sol_path = None
        if fields.get("sol_path"):
            sol_path, entry["sol"] = fq.fetch(fields["sol_path"], args.force)
        convert(entry, qplib_path, sol_path)
        instances[row["name"]] = entry
        check = entry.get("converter_check", {})
        print(f"  {row['name']} {entry['problem_type']}  "
              f"{entry.get('conversion_error') or 'check ' + str(check.get('passed'))}")

    manifest = {
        "source": fq.BASE_URL, "citation": fq.CITATION,
        "licence": {key: home[key] for key in ("licence", "licence_url", "licence_statement")},
        "listing_rows": len(rows), "partial": bool(args.instances), "files": files,
        "objective_convention": ("0.5 x'Q0 x + b0'x + q0, and 0.5 x'Qi x + bi'x in each "
                                 "constraint, the Q lower triangles as listed (doc.html)"),
        "instances": dict(sorted(instances.items())),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8", newline="\n")
    refused = [n for n, e in instances.items() if "conversion_error" in e]
    failed = [n for n, e in instances.items()
              if e.get("converter_check", {}).get("passed") is False]
    print(f"wrote {MANIFEST}: {len(instances)} instances; conversion refused on "
          f"{len(refused)}; check at the published point failed on {len(failed)}"
          + (f": {', '.join(failed)}" if failed else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
