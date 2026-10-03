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
        assert "verifier rejected" not in text, text
        claimed = [dict(run("a", 0, True, True, 0.1), independently_verified=1, relative_gap=""),
                   dict(run("c", 1, True, True, 0.1), independently_verified=0,
                        relative_gap=1.2269e-4)]
        write(d / "t2.csv", claimed)
        write(d / "summary-t2.csv", [{"time_limit": 300}])
        text = miplib_ab_doc.tier2_section(d / "t2.csv")
        assert "verifier rejected" in text and "`c` seed 1 (relative gap 0.000123)" in text, text
        assert "`a` seed 0 (relative" not in text, text

        # #418: the plateau instances' bounds, seed by seed. rlp1 (min, optimum 15) moves its
        # bound from 14 to 14.5 on seed 1; noswot (min, optimum -41) moves AWAY on seed 0; a
        # maximisation case (b-ball read as max here, optimum 2) moves from 3 to 2.5, which is
        # closer although the number fell; opt1217 is absent from the on leg and named.
        def plateau(i, seed, bound, published, restarts="", branches=""):
            return dict(run(i, seed, True, False, 0.1), dual_bound=bound,
                        published_objective=published, restarts=restarts,
                        objective_branches=branches)
        p_off = [plateau("rlp1", s, 14.0, 15.0) for s in range(2)] + \
                [plateau("noswot", 0, -43.0, -41.0), plateau("b-ball", 0, 3.0, 2.0),
                 plateau("opt1217", 0, -20.0, -16.0)]
        p_on = [plateau("rlp1", 0, 14.0, 15.0, 1, 0), plateau("rlp1", 1, 14.5, 15.0, 2, 0),
                plateau("noswot", 0, -44.0, -41.0, 0, 0), plateau("b-ball", 0, 2.5, 2.0, 1, 3)]
        for name, rows in (("p_off.csv", p_off), ("p_on.csv", p_on)):
            write(d / name, rows)
            write(d / ("summary-" + name), [{"time_limit": 60}])
        text = miplib_ab_doc.plateau_ab(d / "p_off.csv", d / "p_on.csv", "mip_restarts")
        assert "Bound moved towards the published optimum: `b-ball` seed 0, `rlp1` seed 1." \
            in text, text
        assert "Moved away: `noswot` seed 0." in text, text
        assert "Not in both legs: `opt1217`." in text, text
        assert "is **met** by this run." in text, text
        assert "restarted at least once: **3 of 4**" in text, text
        assert "objective row: **1 of 4**" in text, text
        assert "| `rlp1` | 1 | feasible | feasible | 14 | 14.5 | **closer** |" in text, text
        assert "| `rlp1` | 0 | feasible | feasible | 14 | 14 | no |" in text, text
        assert "| on | 4 |" in text, text  # seeds_ab's table sits above

        # Nothing moved: the verdict says so.
        text = miplib_ab_doc.plateau_ab(d / "p_off.csv", d / "p_off.csv", "mip_restarts")
        assert "Bound moved towards the published optimum: none." in text, text
        assert "is **not met** by this run." in text, text
        assert "restarted at least once: **0 of 5**" in text, text

        # Legs from two commits, or a dirty one, are not compared at all.
        write(d / "other.csv", [dict(r, git_commit="def5678") for r in p_on])
        write(d / "summary-other.csv", [{"time_limit": 60}])
        text = miplib_ab_doc.plateau_ab(d / "p_off.csv", d / "other.csv", "mip_restarts")
        assert text.startswith("Not compared:") and "`abc1234`, `def5678`" in text, text
        write(d / "dirty.csv", [dict(r, git_commit="abc1234-dirty") for r in p_on])
        write(d / "summary-dirty.csv", [{"time_limit": 60}])
        assert miplib_ab_doc.plateau_ab(d / "dirty.csv", d / "dirty.csv", "x") \
            .startswith("Not compared:")
        assert miplib_ab_doc.plateau_ab(None, d / "p_on.csv", "x") == ""
    print("test_miplib_ab_doc: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
