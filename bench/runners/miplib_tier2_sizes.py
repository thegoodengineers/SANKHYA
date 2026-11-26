#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rows, columns, nonzeros and integer columns of every MIPLIB tier-2 instance (#508).

    python bench/runners/fetch_miplib.py --tier 2
    python bench/runners/miplib_tier2_sizes.py

#508 asks for the region where the device Feasibility Jump loses to be stated by instance
size. The instances themselves are gitignored (data/miplib-tier2/), so docs/BENCHMARKS.md
cannot read their sizes when it is regenerated; this writes them once, with the sha256 of
the file each count was read from, to bench/runners/miplib_tier2_sizes.json. The counts come
from tools/verify_solution_mps.py's reader, which shares no code with the solver.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

from verify_solution_mps import parse_mps  # noqa: E402

TIER2 = REPO / "bench" / "runners" / "miplib_tier2.json"
DATA = REPO / "data" / "miplib-tier2"
OUT = REPO / "bench" / "runners" / "miplib_tier2_sizes.json"


def main() -> int:
    names = [entry["name"] for entry in json.loads(TIER2.read_text())["instances"]]
    sizes = {}
    missing = []
    for name in names:
        path = DATA / f"{name}.mps.gz"
        if not path.exists():
            missing.append(name)
            continue
        model = parse_mps(path)
        sizes[name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(model.row_names),
            "cols": len(model.col_names),
            "nonzeros": sum(len(column) for column in model.entries),
            "integer_cols": sum(1 for flag in model.col_integer if flag),
        }
    if missing:
        print(f"not in {DATA}: {', '.join(missing)} (run fetch_miplib.py --tier 2)",
              file=sys.stderr)
        return 1
    record = {
        "rule": "Counted by tools/verify_solution_mps.py from data/miplib-tier2/<name>.mps.gz "
                "as fetch_miplib.py --tier 2 downloads it; the sha256 is of that file.",
        "instances": sizes,
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(REPO)}: {len(sizes)} instances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
