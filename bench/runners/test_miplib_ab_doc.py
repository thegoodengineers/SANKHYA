#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""miplib_ab_doc renders from synthetic CSVs: node-count mismatches and moved seeds are named.

    python bench/runners/test_miplib_ab_doc.py
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import miplib_ab_doc  # noqa: E402


def write(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def run(i, seed, matched, proved, first, commit="abc1234"):
    return {"instance": i, "seed": seed, "status": "optimal" if proved else "feasible",
            "matched_published": int(matched), "proved_optimal": int(proved),
            "wall_seconds": 5.0 if proved else 60.0, "solver_seconds": 4.0,
            "time_to_first_feasible": first, "primal_integral": 1.0, "nodes": 100,
            "our_objective": 1.0, "git_commit": commit, "machine": "test", "message": ""}


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        off = [run("a", s, True, True, 0.1) for s in range(3)] + \
              [run("b", s, False, False, "") for s in range(3)]
        on = [run("a", s, True, True, 0.1) for s in range(3)] + \
             [run("b", s, s == 0, False, 0.5) for s in range(3)]
        for name, rows in (("off.csv", off), ("on.csv", on)):
            write(d / name, rows)
            write(d / ("summary-" + name), [{"time_limit": 60}])
        text = miplib_ab_doc.seeds_ab(d / "off.csv", d / "on.csv", "opt")
        assert "`b` matched 0->1, proved 0->0" in text, text
        assert "| off | 6 | 3 | 3 |" in text and "| on | 6 | 4 | 3 |" in text, text

        slow = [dict(run("a", 0, True, True, 0.1), nodes=100),
                dict(run("b", 0, True, True, 0.1), nodes=50)]
        fast = [dict(run("a", 0, True, True, 0.1), nodes=100, solver_seconds=2.0),
                dict(run("b", 0, True, True, 0.1), nodes=51)]
        text = miplib_ab_doc.node_rate_ab(write(d / "c0.csv", slow), write(d / "c8.csv", fast),
                                          "cache")
        assert "**1 of 2**" in text and "differ: `b`" in text, text
        assert "**2.000x**" in text, text

        text = miplib_ab_doc.tier2_section(d / "on.csv")
        assert "`b` seed 1 (feasible)" in text and "`b` seed 2 (feasible)" in text, text
    print("test_miplib_ab_doc: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
