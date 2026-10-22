#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run SANKHYA over the refinery LPs (data/refinery, kind "lp") and emit netlib.py's CSV.

The refinery set's own head-to-head (compare.py --suite refinery) runs every rival beside
SANKHYA; an option A/B needs SANKHYA alone, twice, in the same job (option_ab.py). This is
that runner: each LP of data/refinery/manifest.json solved and verified by netlib.run_one,
the reference objective the manifest's (an exact construction or a rational simplex, its
`reference_source` says which), the pass rule netlib.py's.

Usage:
    python bench/runners/refinery_lp.py --binary build/sankhya --time-limit 600
    python bench/runners/option_ab.py --runner refinery_lp.py --binary build/sankhya \\
        --option presolve_substitution=true --machine "..." --out-prefix ...
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kkt_crossings  # noqa: E402
import netlib  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "refinery"


def lp_instances() -> dict:
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    return {name: entry for name, entry in manifest["instances"].items()
            if entry.get("kind") == "lp"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--instances", nargs="*")
    parser.add_argument("--solver-option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--machine", default=None)
    args = parser.parse_args()

    reference = lp_instances()
    names = sorted(args.instances or reference)
    binary = args.binary or netlib.default_binary()
    commit, machine = netlib.git_commit(args.binary), args.machine or netlib.machine_tag()
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    options = " ".join(args.solver_option)
    rows = []
    for name in names:
        entry = reference.get(name)
        if entry is None:
            print(f"{name}: not an LP of data/refinery/manifest.json, skipped")
            continue
        mps = DATA_DIR / f"{name}.mps"
        published = float(entry["reference_objective"])
        blob = netlib.run_one(binary, mps, args.time_limit, True, args.solver_option)
        ours = blob.get("objective")
        gap = None if ours is None else abs(ours - published) / max(1.0, abs(published))
        verified = blob.get("verified")
        matches = bool(blob["status"] == "optimal" and gap is not None
                       and gap <= netlib.PASS_RELATIVE_TOLERANCE)
        rows.append({
            "instance": name, "instance_sha256": netlib.sha256_file(mps),
            "rows": blob.get("rows", ""), "columns": blob.get("columns", ""),
            "nonzeros": blob.get("nonzeros", ""), "status": blob["status"],
            "message": blob.get("message", ""),
            "verifier_message": blob.get("verifier_output", ""),
            "our_objective": "" if ours is None else repr(ours),
            "published_objective": repr(published),
            "absolute_gap": "" if ours is None else repr(abs(ours - published)),
            "relative_gap": "" if gap is None else repr(gap),
            "matches_published": int(matches),
            "independently_verified": "" if verified is None else int(verified),
            "passed": int(matches and verified is not False),
            "wall_seconds": round(blob.get("wall_seconds", 0.0), 6),
            "solver_seconds": blob.get("solver_seconds", ""),
            "iterations": blob.get("iterations", ""), "algorithm": blob.get("algorithm", ""),
            # #486: run_one flattened the crossings already; blank for an engine other than PDHG.
            **{k: blob.get(k, "") for k in kkt_crossings.ALL_COLUMNS},
            "git_commit": commit, "solver_options": options, "machine": machine,
            "timestamp_utc": timestamp,
        })
        print(f"{name:<26} {blob['status']:<14} {ours!s:>22} iters {blob.get('iterations', '-')}"
              f"  {blob.get('solver_seconds', '-')} s  verified {verified}")
    out_path = (REPO_ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=netlib.CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {netlib.display_path(out_path)}")
    return 0 if rows and all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
