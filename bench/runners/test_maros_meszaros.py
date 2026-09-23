#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Maros-Meszaros fetcher, runner, residuals and doc section (#491).

Hermetic: nothing here downloads a file or runs the solver. The one check that reads the
committed manifest reads only data/maros-meszaros/reference.json.

    python bench/runners/test_maros_meszaros.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_maros_meszaros as fetch  # noqa: E402
import maros_meszaros as runner  # noqa: E402
import maros_meszaros_doc as doc  # noqa: E402
import qp_residuals  # noqa: E402

FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}  {detail}")
    if not condition:
        FAILURES += 1


# ---- the readme table ------------------------------------------------------------------

README_EXCERPT = """\
OPT   solution value obtained by the default settings of BPMPD solver

The separable problems are easily recognized by a 0 entry in the QNZ column.

NAME           M       N      NZ      QN      QNZ       OPT

aug2d      10000   20200   40000   19800        0    1.6874118e+06
cont-050    2401    2597   12005    2597        0   -4.5638509e+00
cvxqp1l     5000   10000   14998   10000    29984    1.0870480e+08
hs51           3       5       7       5        2    8.8817842e-16
tame           1       2       2       2        1    0.0000000e+00

aug2d aug2dc aug2dcqp aug2dqp aug3d aug3dc aug3dcqp aug3dqp cvxqp1 l
cvxqp1 m cvxqp1 s cvxqp2 l cvxqp2 m cvxqp2 s cvxqp3 l cvxqp3 m cvxqp3 s
"""


def test_readme_table() -> None:
    table = fetch.parse_readme(README_EXCERPT)
    check(sorted(table) == ["aug2d", "cont-050", "cvxqp1l", "hs51", "tame"],
          "only whole table rows are read; prose and acknowledgements are not", str(sorted(table)))
    check(table["cont-050"]["reference_objective"] == -4.5638509,
          "a negative OPT keeps its sign", repr(table["cont-050"]))
    check(table["aug2d"]["reference_text"] == "1.6874118e+06",
          "the printed text is kept beside the float")
    check(table["cvxqp1l"]["qnz"] == 29984 and table["cvxqp1l"]["m"] == 5000,
          "the counts land in the right fields", repr(table["cvxqp1l"]))
    check(table["tame"]["reference_objective"] == 0.0, "a zero optimum parses")
    raised = False
    try:
        fetch.parse_readme(README_EXCERPT + "tame  1  2  2  2  1  5.0e+00\n")
    except ValueError:
        raised = True
    check(raised, "a name listed twice is an error, not an overwrite")


def test_member_names() -> None:
    cases = {"CVXQP1_L.QPS": "cvxqp1l", "CONT-050.QPS": "cont-050", "HUES-MOD.QPS": "hues-mod",
             "QAFIRO.QPS": "qafiro", "sub/HS21.QPS": "hs21"}
    for member, expected in cases.items():
        got = fetch.instance_name(member)
        check(got == expected, f"archive member {member} is {expected}", got)


def test_pin_conflicts() -> None:
    old = {"files": {"QPDATA1.ZIP": {"sha256": "a" * 64}},
           "instances": {"hs21": {"qps_sha256": "b" * 64}}}
    same = {"files": {"QPDATA1.ZIP": {"sha256": "a" * 64}},
            "instances": {"hs21": {"qps_sha256": "b" * 64}, "new": {"qps_sha256": "c" * 64}}}
    changed = {"files": {"QPDATA1.ZIP": {"sha256": "d" * 64}},
               "instances": {"hs21": {"qps_sha256": "e" * 64}}}
    check(fetch.pin_conflicts(old, same) == [], "unchanged files and a new instance are no conflict")
    check(len(fetch.pin_conflicts(old, changed)) == 2, "a changed archive and QPS are both named",
          str(fetch.pin_conflicts(old, changed)))
    check(fetch.pin_conflicts({}, changed) == [], "a first fetch has nothing to conflict with")


def test_committed_manifest() -> None:
    path = fetch.DATA_DIR / "reference.json"
    if not path.exists():
        check(True, "no committed manifest to check", "(skipped)")
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    instances = manifest["instances"]
    check(len(instances) == fetch.EXPECTED_INSTANCES, "the manifest holds all 138 instances",
          str(len(instances)))
    bad_sha = [n for n, e in instances.items() if not re.fullmatch(r"[0-9a-f]{64}", e["qps_sha256"])]
    check(not bad_sha, "every instance has a sha256", str(bad_sha[:5]))
    drift = [n for n, e in instances.items()
             if float(e["reference_text"]) != e["reference_objective"]]
    check(not drift, "every reference value is the float of the text the readme printed",
          str(drift[:5]))
    unsourced = [n for n, e in instances.items() if not e.get("reference_source")]
    check(not unsourced, "every reference value names its source", str(unsourced[:5]))
    check(all(e["url"].startswith("https://") for e in manifest["files"].values()),
          "every downloaded file has its URL")


# ---- the residuals ---------------------------------------------------------------------

# min -x - y + 0.5 (2x^2 + 2xy + 2y^2)  s.t.  x + y >= 0, x and y free. The optimum is
# x = y = 1/3, the row is slack there and its multiplier zero, and the objective -1/3.
QPS_TWO_FREE = """NAME          QCONV
ROWS
 N  COST
 G  R1
COLUMNS
    X         COST        -1.0   R1           1.0
    Y         COST        -1.0   R1           1.0
RHS
    RHS       R1           0.0
BOUNDS
 FR BND       X
 FR BND       Y
QUADOBJ
    X         X            2.0
    Y         X            1.0
    Y         Y            2.0
ENDATA
"""

# min (x - 3)^2 = x^2 - 6x + 9 with 0 <= x <= 1 and a row x <= 5: the column sits at its upper
# bound with a reduced cost of 2*1 - 6 = -4, which prices that upper bound.
QPS_AT_UPPER = """NAME          QBOUND
ROWS
 N  COST
 L  R1
COLUMNS
    X         COST        -6.0   R1           1.0
RHS
    RHS       COST        -9.0
    RHS       R1           5.0
BOUNDS
 UP BND       X            1.0
QUADOBJ
    X         X            2.0
ENDATA
"""

# max 4x - x^2 s.t. x <= 10, x free: optimum x = 2, objective 4.
QPS_MAXIMIZE = """NAME          QMAX
OBJSENSE
    MAX
ROWS
 N  COST
 L  R1
COLUMNS
    X         COST         4.0   R1           1.0
RHS
    RHS       R1          10.0
BOUNDS
 FR BND       X
QUADOBJ
    X         X           -2.0
ENDATA
"""


def residuals_of(qps: str, values: dict, row_duals: dict) -> qp_residuals.Residuals:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.qps"
        path.write_text(qps)
        model = qp_residuals.parse_mps(path)
    solution = qp_residuals.Solution()
    solution.header = {"status": "optimal"}
    solution.col_value = dict(values)
    solution.row_dual = dict(row_duals)
    return qp_residuals.compute(model, solution)


def test_residuals_at_the_optimum() -> None:
    third = 1.0 / 3.0
    r = residuals_of(QPS_TWO_FREE, {"X": third, "Y": third}, {"R1": 0.0})
    check(r.primal == 0.0 and r.dual < 1e-15 and r.gap < 1e-15,
          "the true optimum of the two-free-column QP has zero residuals",
          f"{r.primal:.1e} {r.dual:.1e} {r.gap:.1e}")
    check(abs(r.primal_objective + third) < 1e-15, "0.5 x'Qx with the off-diagonal counted twice",
          repr(r.primal_objective))
    check(r.meets(1e-9, True) and r.meets(1e-9, False), "and it succeeds at 1e-9 both ways")


def test_residuals_catch_a_feasible_non_stationary_point() -> None:
    # Feasible, objective honest, not stationary: gradient (-1 + 2 + 0.5, -1 + 1 + 1) = (1.5, 1).
    r = residuals_of(QPS_TWO_FREE, {"X": 1.0, "Y": 0.5}, {"R1": 0.0})
    check(r.primal == 0.0, "the near miss is primal feasible", f"{r.primal:.1e}")
    check(abs(r.dual - 1.5) < 1e-12, "its dual residual is the largest free-column gradient",
          f"{r.dual:.3e}")
    check(not r.meets(1e-6, True), "so it fails at 1e-6")


def test_residuals_catch_a_wrong_sign_row_multiplier() -> None:
    # A G row has no upper bound, so a negative multiplier prices a bound that is not there.
    third = 1.0 / 3.0
    r = residuals_of(QPS_TWO_FREE, {"X": third, "Y": third}, {"R1": -0.25})
    check(r.dual >= 0.25 - 1e-15, "a negative multiplier on a >= row is a dual violation",
          f"{r.dual:.3e}")


def test_residuals_catch_infeasibility() -> None:
    r = residuals_of(QPS_TWO_FREE, {"X": -1.0, "Y": -1.0}, {"R1": 0.0})
    check(abs(r.primal - 2.0) < 1e-15, "x + y = -2 against x + y >= 0 is a residual of 2",
          f"{r.primal:.3e}")
    check(r.primal_rel <= r.primal and r.dual_rel <= r.dual and r.gap_rel <= r.gap,
          "no relative measure exceeds its absolute one")


def test_residuals_price_an_active_bound() -> None:
    r = residuals_of(QPS_AT_UPPER, {"X": 1.0}, {"R1": 0.0})
    check(r.primal == 0.0 and r.dual == 0.0,
          "a reduced cost pricing the active upper bound is admissible",
          f"{r.primal:.1e} {r.dual:.1e}")
    check(abs(r.primal_objective - 4.0) < 1e-12 and r.gap < 1e-12,
          "and the objective constant and the bound close the gap",
          f"primal {r.primal_objective!r} dual {r.dual_objective!r}")
    off = residuals_of(QPS_AT_UPPER, {"X": 0.5}, {"R1": 0.0})
    check(off.dual == 0.0 and off.gap > 1.0,
          "off the bound the same sign is admissible but the gap exposes it", f"{off.gap:.3e}")


def test_residuals_on_a_maximize_model() -> None:
    r = residuals_of(QPS_MAXIMIZE, {"X": 2.0}, {"R1": 0.0})
    check(r.dual == 0.0 and r.gap < 1e-12 and abs(r.primal_objective - 4.0) < 1e-12,
          "the maximize optimum x = 2 has zero residuals",
          f"dual {r.dual:.1e} gap {r.gap:.1e} objective {r.primal_objective!r}")
    wrong = residuals_of(QPS_MAXIMIZE, {"X": 3.0}, {"R1": 0.0})
    check(abs(wrong.dual - 2.0) < 1e-12, "x = 3 leaves a gradient of 2 on a free column",
          f"{wrong.dual:.3e}")


# ---- the runner's verdicts -------------------------------------------------------------

def test_judge() -> None:
    good = qp_residuals.Residuals(1e-10, 1e-10, 1e-10, 1e-11, 1e-11, 1e-11, 1.0, 1.0)
    mid = qp_residuals.Residuals(1e-8, 1e-8, 1e-8, 1e-10, 1e-10, 1e-10, 1.0, 1.0)
    v = runner.judge("optimal", 1.0000001, 1.0, good)
    check(v["matches_reference"] and v["success_rel_1e-9"] and v["success_abs_1e-9"],
          "an optimal row within every tolerance succeeds everywhere", str(v))
    v = runner.judge("optimal", 1.0, 1.0, mid)
    check(v["success_rel_1e-9"] and not v["success_abs_1e-9"] and v["success_abs_1e-6"],
          "relative and absolute are judged separately", str(v))
    v = runner.judge("time_limit", 1.0, 1.0, good)
    check(not any(v[k] for k in v if k.startswith("success_")) and not v["matches_reference"],
          "a limit status never succeeds, whatever its residuals", str(v))
    v = runner.judge("optimal", 1.1, 1.0, good)
    check(not v["matches_reference"], "a wrong objective does not match the reference")
    v = runner.judge("optimal", 1e-7, 8.8817842e-16, good)
    check(v["matches_reference"], "a zero optimum is compared absolutely, not relatively",
          repr(v["relative_gap"]))
    v = runner.judge("optimal", None, 1.0, None)
    check(v["relative_gap"] is None and not v["success_rel_1e-6"], "no objective, no verdict")


def test_select() -> None:
    instances = {"b": {"qps_bytes": 10}, "a": {"qps_bytes": 10}, "c": {"qps_bytes": 5},
                 "d": {"qps_bytes": 99}}
    check(runner.select(instances, None, 3) == ["c", "a", "b"],
          "the smallest N by size, ties by name", str(runner.select(instances, None, 3)))
    check(runner.select(instances, None, None) == ["a", "b", "c", "d"], "no selection is all")
    check(runner.select(instances, ["d"], 2) == ["d"], "explicit names win over --smallest")


# ---- the doc section -------------------------------------------------------------------

def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=runner.CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in runner.CSV_COLUMNS})


def row(name: str, status: str, **flags) -> dict:
    base = {"instance": name, "status": status, "git_commit": "abc1234",
            "machine": "Linux-x86_64", "time_limit": "1000.0", "solver_seconds": "1.5",
            "our_objective": "1.0", "reference_objective": "1.0", "relative_gap": "0.0",
            "matches_reference": "1", "independently_verified": "1",
            "success_rel_1e-6": "1", "success_rel_1e-9": "1", "success_abs_1e-6": "1",
            "success_abs_1e-9": "1", "passed": "1"}
    base.update(flags)
    return base


def test_doc_section() -> None:
    check("Not yet run" in doc.section(None), "no CSV says so and says how to produce one")
    with tempfile.TemporaryDirectory() as tmp:
        dirty = Path(tmp) / "maros-meszaros-abc1234.csv"
        write_rows(dirty, [row("hs21", "optimal", git_commit="abc1234-dirty")])
        check("modified tree" in doc.section(dirty), "a -dirty CSV is refused")

        path = Path(tmp) / "maros-meszaros-abc1235.csv"
        write_rows(path, [
            row("hs21", "optimal"),
            row("qafiro", "optimal", **{"success_rel_1e-9": "0", "success_abs_1e-9": "0"}),
            row("boyd1", "time_limit", matches_reference="0", independently_verified="",
                **{"success_rel_1e-6": "0", "success_rel_1e-9": "0", "success_abs_1e-6": "0",
                   "success_abs_1e-9": "0", "passed": "0"}),
            row("qship08l", "optimal", matches_reference="0", independently_verified="0",
                passed="0"),
        ])
        text = doc.section(path)
    check("| 1e-6 | 3 / 4 |" in text and "| 1e-9 | 2 / 4 |" in text,
          "the success counts at both levels come from the rows")
    lines = {line.split("**")[1]: line for line in text.splitlines() if line.startswith("- **")}
    check("`boyd1`" in lines.get("Did not reach `optimal`", ""), "a time limit is named")
    check("`qship08l`" in lines.get("Rejected by `tools/verify_solution.py`", ""),
          "a verifier rejection is named")
    check("`qship08l`" in lines.get(
        "`optimal` but more than 1e-6 relative from the reference objective", ""),
        "a reference mismatch is named")
    check("`qafiro`" in lines.get("`optimal` but not successful at 1e-9 relative", "")
          and "`hs21`" not in text.split("Every failure, named:")[1],
          "a 1e-9 miss is named and a full success is not")
    check(chr(0x2014) not in text and chr(0x2013) not in text, "the section carries no em or en dash")


def main() -> int:
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            print(name)
            test()
    print(f"\n{'all passed' if FAILURES == 0 else f'{FAILURES} check(s) FAILED'}")
    return 0 if FAILURES == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
