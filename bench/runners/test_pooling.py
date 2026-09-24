#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the pooling data, model generator, runner and doc section (#516).

Hermetic: nothing here downloads a file or runs the solver. It reads only data/pooling/.

    python bench/runners/test_pooling.py
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_pooling as fetch  # noqa: E402
import pooling as runner  # noqa: E402
import pooling_doc as doc  # noqa: E402
import pooling_models as models  # noqa: E402

FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}  {detail}")
    if not condition:
        FAILURES += 1


# Alfaki and Haugland's Haverly1.gms, verbatim apart from the comment header.
HAVERLY1_GMS = """\
$ontext
    Haverly1 pooling problem data.
$offtext
$eolcom #
# Declare sets
    set i    / 1*6  /;
    set s(i) / 1*3  /;
    set t(i) / 5*6  /;
    set k    / 1*1  /;
alias (i,j);
# The arc unit cost c_{ij}
table c(i,j)
          4       5       6
  1    6.00    0.00    0.00
  2   16.00    0.00    0.00
  3    0.00    1.00   -5.00
  4    0.00   -9.00  -15.00 ;
table a(i,j)
      4   5   6
  1   1   0   0
  2   1   0   0
  3   0   1   1
  4   0   1   1 ;
table q(i,k)
          1
  1    3.00
  2    1.00
  3    2.00
  5    2.50
  6    1.50 ;
parameter bl(i) /  1     0.00
                   2     0.00
                   3     0.00
                   5     0.00
                   6     0.00 / ;
parameter bu(i) /  1   300.00
                   2   300.00
                   3   300.00
                   4   300.00
                   5   100.00
                   6   200.00 / ;
$include xmodel.gms
"""


def test_gams_parser() -> None:
    net = fetch.parse_gams_data(HAVERLY1_GMS)
    check(net["sources"] == [1, 2, 3] and net["pools"] == [4] and net["terminals"] == [5, 6],
          "sources, pools and terminals from the sets", repr(net["pools"]))
    check(net["arcs"] == [[1, 4, 6.0], [2, 4, 16.0], [3, 5, 1.0], [3, 6, -5.0], [4, 5, -9.0],
                          [4, 6, -15.0]], "arcs are exactly a(i,j) = 1, with c(i,j)")
    check(net["quality"]["2"] == [1.0] and net["quality"]["6"] == [1.5],
          "source quality and terminal bound read per node")
    check(net["upper"]["4"] == 300.0 and "4" not in net["lower"], "pool capacity read")
    committed = json.loads((models.INSTANCE_DIR / "haverly1.json").read_text())
    check(all(committed[key] == net[key] for key in net),
          "the committed haverly1.json is what the parser reads from the published file")
    try:
        fetch.parse_gams_data(HAVERLY1_GMS.replace("  4   0   1   1 ;", "  4   0   2   1 ;"))
        check(False, "an adjacency entry other than 0 or 1 is refused")
    except ValueError:
        check(True, "an adjacency entry other than 0 or 1 is refused")


def test_committed_models_match_generator() -> None:
    reference = json.loads((models.DATA_DIR / "reference.json").read_text())
    names = models.instance_names()
    check(len(names) == 13 and sorted(reference["instances"]) == names,
          "thirteen instances, each with a reference entry", str(names))
    for name in names:
        net = models.load(name)
        entry = reference["instances"][name]
        check(bool(entry.get("paper")) and isinstance(entry.get("reference_objective"), float),
              f"{name}: source paper and published optimum recorded")
        for formulation in models.FORMULATIONS:
            text = models.write_mps(models.build(net, formulation),
                                    models.preamble(net, formulation))
            path = models.DATA_DIR / f"{name}_{formulation}.mps"
            same = path.exists() and path.read_bytes().replace(b"\r\n", b"\n") == text.encode()
            if not same:
                check(False, f"{path.name} is what the generator writes")


# Observed on minlplib.org's instance pages for pooling_<name>pq (#Variables, #Constraints,
# #Quadratic Constraints), 2026-09-24. Their pq instance is the PQ-formulation with the RLT
# rows removed, i.e. our Q-formulation; equal counts on all thirteen say the network was
# transcribed with the same arcs, nodes and specifications.
MINLPLIB_Q_SIZES = {
    "adhya1": (33, 49, 20), "adhya2": (33, 57, 20), "adhya3": (52, 74, 32),
    "adhya4": (58, 77, 40), "bental4": (13, 16, 6), "bental5": (92, 86, 60),
    "foulds2": (36, 34, 16), "foulds3": (672, 571, 512), "foulds4": (672, 571, 512),
    "foulds5": (608, 563, 512), "haverly1": (10, 13, 4), "haverly2": (10, 13, 4),
    "haverly3": (10, 13, 4),
}


def test_sizes_match_minlplib() -> None:
    for name, expected in sorted(MINLPLIB_Q_SIZES.items()):
        model = models.build(models.load(name), "q")
        got = (len(model.columns), len(model.rows), sum(1 for r in model.rows if r.quadratic))
        check(got == expected, f"{name}: Q-formulation size equals MINLPLib's", f"{got}")


# Optimal flows derived by hand from Haverly's data (profit = minus cost):
#   1: B -> pool 100 -> Y, C -> Y 100            pool at 1 %, Y at 1.5 %      cost -400
#   2: A -> pool 300 -> X, C -> X 300            pool at 3 %, X at 2.5 %      cost -600
#   3: A 50 + B 150 -> pool 200 -> Y             pool at 1.5 %                cost -750
KNOWN_OPTIMA = {
    "haverly1": ({(2, 4): 100, (4, 6): 100, (3, 6): 100}, -400.0),
    "haverly2": ({(1, 4): 300, (4, 5): 300, (3, 5): 300}, -600.0),
    "haverly3": ({(1, 4): 50, (2, 4): 150, (4, 6): 200}, -750.0),
}


def test_known_optima_in_every_formulation() -> None:
    for name, (flows, cost) in KNOWN_OPTIMA.items():
        net = models.load(name)
        for formulation in models.FORMULATIONS:
            model = models.build(net, formulation)
            point = models.point_from_flows(net, formulation, flows)
            objective, violation = models.evaluate(model, point)
            check(abs(objective - cost) < 1e-9 and violation < 1e-9,
                  f"{name} {formulation.upper()}: the published optimum is a feasible point",
                  f"objective {objective:g}, worst violation {violation:.1e}")
    # The bilinear rows bite: all of A through the pool into Y is 3 % against a 1.5 % spec.
    net = models.load("haverly1")
    for formulation in models.FORMULATIONS:
        model = models.build(net, formulation)
        point = models.point_from_flows(net, formulation, {(1, 4): 100, (4, 6): 100})
        _, violation = models.evaluate(model, point)
        check(violation > 1.0, f"haverly1 {formulation.upper()}: an off-spec blend violates",
              f"worst violation {violation:g}")


def test_pq_adds_only_rlt_rows() -> None:
    for name in ("haverly1", "adhya4", "foulds5"):
        net = models.load(name)
        q = {row.name for row in models.build(net, "q").rows}
        pq = {row.name for row in models.build(net, "pq").rows}
        extra = pq - q
        check(q <= pq and extra and all(r.startswith(("rlt_", "rltc_")) for r in extra),
              f"{name}: PQ is Q plus the RLT rows", f"{len(extra)} extra rows")


def test_qcmatrix_convention() -> None:
    text = (models.DATA_DIR / "haverly1_p.mps").read_text()
    block = text.split("QCMATRIX tq_5_1\n", 1)[1].split("QCMATRIX", 1)[0]
    check(" f_4_5 p_4_1 0.5\n p_4_1 f_4_5 0.5" in block,
          "a term 1 * p*f is written as two symmetric entries of 1/2 (no 1/2 factor)")


def test_classify() -> None:
    ref = -400.0
    base = {"verifier_rc": 0}
    check(runner.classify({**base, "status": "refused", "message": "x: QCMATRIX section"}, ref)
          == "refused at read: quadratic constraints (QCMATRIX)", "a reader refusal is named")
    check(runner.classify({**base, "status": "optimal", "objective": -400.0,
                           "dual_bound": -400.0}, ref) == "", "optimal at the optimum passes")
    check(runner.classify({**base, "status": "optimal", "objective": -100.0}, ref)
          == "optimal claimed at a different objective", "a local optimum called optimal fails")
    check(runner.classify({**base, "status": "feasible", "objective": -400.0,
                           "dual_bound": -420.0}, ref) == "gap not closed",
          "the right point without a closed gap is not a pass")
    check(runner.classify({**base, "status": "optimal", "objective": -400.0,
                           "dual_bound": -350.0}, ref) == "proven bound above the published "
          "optimum", "a bound above the known optimum is flagged before anything else")
    check(runner.classify({"status": "optimal", "objective": -400.0, "verifier_rc": 1}, ref)
          == "rejected by the verifier", "the verifier has the last word")
    check(abs(runner.tolerance(-3500.0) - 0.35) < 1e-15 and runner.tolerance(0.0) == 1e-6,
          "pass tolerance is 1e-4 relative with a 1e-6 floor")


def test_doc_section() -> None:
    columns = runner.CSV_COLUMNS
    rows = [
        {"instance": "haverly1", "formulation": "p", "status": "optimal", "passed": "1",
         "our_objective": "-400", "reference_objective": "-400", "root_bound": "-500",
         "failure_reason": "", "git_commit": "abc1234", "machine": "m", "time_limit": "60"},
        {"instance": "haverly1", "formulation": "pq", "status": "optimal", "passed": "1",
         "our_objective": "-400", "reference_objective": "-400", "root_bound": "-450",
         "failure_reason": "", "git_commit": "abc1234", "machine": "m", "time_limit": "60"},
        {"instance": "foulds3", "formulation": "p", "status": "refused", "passed": "0",
         "reference_objective": "-8", "git_commit": "abc1234", "machine": "m",
         "failure_reason": "refused at read: quadratic constraints (QCMATRIX)"},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pooling-abc1234.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows([{key: row.get(key, "") for key in columns} for row in rows])
        text = doc.section(path)
        check("**2 of 3 runs**" in text, "the pass count is read from the CSV")
        check("`foulds3_p`" in text and "refused at read" in text, "a refused run is named")
        check("| `haverly1` | -500 | - | -450 | -400 | 50.0 % |" in text,
              "PQ closes half the P root gap in the synthetic row", text[-700:])
        dirty = Path(tmp) / "pooling-abc1234-dirty.csv"
        with dirty.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerow({key: {"git_commit": "abc1234-dirty"}.get(key, "x")
                             for key in columns})
        check("modified tree" in doc.section(dirty), "a -dirty CSV is refused")
    check("Not yet run" in doc.section(None), "no CSV says so and how to reproduce")


def main() -> int:
    for test in (test_gams_parser, test_committed_models_match_generator,
                 test_sizes_match_minlplib, test_known_optima_in_every_formulation,
                 test_pq_adds_only_rlt_rows, test_qcmatrix_convention, test_classify,
                 test_doc_section):
        print(test.__name__)
        test()
    print("all passed" if not FAILURES else f"{FAILURES} FAILED")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
