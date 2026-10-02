#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the MIPLIB 3 set (#761): the catalogue parse, the published_tolerance rule, the
manifest, the CSV names against every MIPLIB 2017 glob, the HiGHS worker on a MIP, the
grading and the doc section.

Hermetic: no solver runs and nothing is downloaded. The catalogue is an inline excerpt of
miplib.cat; HiGHS is a stand-in module that records what the worker asks of it; the one
subprocess is tools/verify_solution.py on a two-column MIP written here.

    python bench/runners/test_miplib3.py
"""
from __future__ import annotations

import csv
import fnmatch
import re
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))
import compare_suite  # noqa: E402
import fetch_miplib3  # noqa: E402
import latest_result  # noqa: E402
import miplib  # noqa: E402
import miplib3_doc  # noqa: E402
import rivals  # noqa: E402
from verify_solution_mps import parse_mps  # noqa: E402
from verify_solution_sol import parse_sol  # noqa: E402

FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}  {detail}")
    if not condition:
        FAILURES += 1


# Rows copied from miplib.cat as archived (3 Oct 2026), with its header and the start of
# PART B, so the parse is tested on the layout it reads: the three "(not opt)" rows, rgn's
# truncated value, and the integers printed with trailing zeros.
CATALOGUE = """\
INDEX - PART A : STATISTICS
===========================
NAME      ROWS  COLS   INT  0/1   CONT     INT SOLN                 LP SOLN
====      ====  ====  ====  ===   ====     ========                 =======
arki001   1048  1388  538   415    850     7580813.0459 (not opt)   7579599.80787
bell3a    123   133   71    39     62      878430.32                862578.64
dano3mip  3202  13873 552   ALL    13321   728.1111 (not opt)       576.23162474
danoint   664   521   56    ALL    465     65.67                    62.637280418
dsbmip    1182  1886  192   160    1694    -305.19817501            -305.19817501
flugpl    18    18    11    0      7       1201500                  1167185.73
gen       780   870   150   144    726     112313                   112130.0
gt2       29    188   188   24     0       21166.000                13460.233074
harp2     112   2993  2993  ALL    0       -73899798.00             -74353341.502
misc03    96    160   159   ALL    1       3360                     1910.0
noswot    182   128   100   75     25      -43                      -43.0
pk1       45    86    55    ALL    31      11.0                     0.0
pp08a     136   240   64    ALL    176     7350.0                   2748.3452381
rgn       24    180   100   ALL    80      82.1999                  48.7999
seymour   4944  1372  1372  ALL    0       423 (not opt)            403.84647413


Explanation of columns:
NAME          - name of the problem

INDEX - PART B : ORIGINS
========================
NAME     ORIGINATOR                          FORMULATOR                          DONATOR
rgn      Linus E. Schrage                    Laurence A. Wolsey
"""

print("catalogue")
cat = fetch_miplib3.parse_catalogue(CATALOGUE)
check(len(cat) == 15, "every PART A row and nothing from PART B", str(sorted(cat)))
check({n for n, r in cat.items() if r["not_opt"]} == {"arki001", "dano3mip", "seymour"},
      "the three (not opt) rows are marked")
check(cat["arki001"]["int_soln"] == "7580813.0459" and cat["seymour"]["int_soln"] == "423",
      "a (not opt) row keeps its printed value, not the qualifier")
check(cat["rgn"]["int_soln"] == "82.1999" and cat["gt2"]["int_soln"] == "21166.000",
      "values kept as printed text", f"{cat['rgn']} {cat['gt2']}")
try:
    fetch_miplib3.parse_catalogue("no index here")
    check(False, "a file with no PART A is refused")
except ValueError:
    check(True, "a file with no PART A is refused")

print("published_tolerance: one unit in the last printed decimal, trailing zeros stripped")
for text, expected in (("82.1999", 1e-4), ("65.67", 1e-2), ("878430.32", 1e-2),
                       ("-305.19817501", 1e-8), ("405935.18000", 1e-2), ("7350.0", 0.0),
                       ("21166.000", 0.0), ("-73899798.00", 0.0), ("1201500", 0.0),
                       ("0.0", 0.0), ("-43", 0.0)):
    got = fetch_miplib3.published_tolerance(text)
    check(abs(got - expected) <= 1e-18, f"{text} -> {expected:g}", f"got {got!r}")
for bad in ("1e-3", "n/a"):
    try:
        fetch_miplib3.published_tolerance(bad)
        check(False, f"{bad!r} refused")
    except ValueError:
        check(True, f"{bad!r} refused")

print("objective integrality and rounding risk")
check(fetch_miplib3.objective_integral([3.0, 0.0, -2.0], [True, False, True]),
      "integer costs on integer columns only: an integral objective")
check(not fetch_miplib3.objective_integral([3.0, 1.0], [True, False]),
      "a cost on a continuous column: possibly fractional")
check(not fetch_miplib3.objective_integral([0.5], [True]),
      "a fractional cost on an integer column: possibly fractional")
check(not fetch_miplib3.objective_integral([1.0], [True], offset=0.5),
      "a fractional objective constant: possibly fractional")
risk = fetch_miplib3.rounding_risk
check(risk("112313", 112313.0, False) and risk("7350.0", 7350.0, False),
      "an integer print of a possibly fractional objective below 1e6 is at risk (gen, pp08a)")
check(not risk("-43", -43.0, True), "not when the objective is integral: the print is exact")
check(not risk("1201500", 1201500.0, False) and not risk("106940226", 106940226.0, False),
      "not at 1e6 and above, where one unit is inside the 1e-6 rule (flugpl, khb05250)")
check(not risk("82.1999", 82.1999, False) and not risk("65.67", 65.67, False),
      "not with decimals: published_tolerance covers one unit in the last place")

print("manifest")
INTEGRAL = {"bell3a": False, "danoint": False, "dsbmip": False, "flugpl": False, "gen": False,
            "gt2": True, "harp2": True, "misc03": False, "noswot": True, "pk1": False,
            "pp08a": False, "rgn": False}
models = {n: f"* header\nNAME {n}\nENDATA\n".encode() for n in cat}
models["mas74"] = b"NAME mas74\nENDATA\n"
manifest = fetch_miplib3.build_manifest(models, cat, CATALOGUE.encode(), INTEGRAL)
inst, excl = manifest["instances"], manifest["excluded"]
check(set(inst) == set(cat) - {"arki001", "dano3mip", "seymour", "misc03", "pp08a"},
      "a reference for every row not marked (not opt) and not at unconfirmed rounding risk",
      str(sorted(inst)))
check(set(excl) == {"arki001", "dano3mip", "seymour", "mas74", "misc03", "pp08a"},
      "the rest are excluded")
check(excl["mas74"]["reason"].startswith("no row") and "(not opt)" in excl["seymour"]["reason"]
      and "no proven value" in excl["pp08a"]["reason"]
      and excl["pp08a"]["catalogue_int_soln"] == "7350.0", "each with its reason")
check(inst["rgn"]["published_optimal"] == 82.1999 and inst["rgn"]["published_tolerance"] == 1e-4
      and inst["gt2"]["published_tolerance"] == 0.0, "rgn's and gt2's tolerances recorded")
check(inst["rgn"]["sha256"] == fetch_miplib3.hashlib.sha256(models["rgn"]).hexdigest()
      and inst["rgn"]["file"] == "rgn.mps", "the sha256 of the bytes written and solved")
check(inst["noswot"]["published_optimal"] == -41.00000885
      and inst["noswot"]["reference_source"] == "override"
      and inst["noswot"]["catalogue_int_soln"] == "-43"
      and "line 23" in inst["noswot"]["reference_note"]
      and "same model" in inst["noswot"]["reference_note"],
      "noswot: the catalogue's -43 overridden by MIPLIB 2017's proven -41.00000885, with why")
check(inst["gen"]["published_optimal"] == 112313.3627179998
      and inst["gen"]["reference_source"] == "override"
      and inst["gen"]["catalogue_int_soln"] == "112313", "gen: the rounded print overridden")
check(inst["pk1"]["reference_source"] == "miplib.cat" and inst["pk1"]["published_optimal"] == 11.0
      and "confirmed by" in inst["pk1"]["reference_note"],
      "pk1: an at-risk integer print kept because a proven value confirms it")
check(inst["flugpl"]["reference_source"] == "miplib.cat"
      and "reference_note" not in inst["flugpl"], "flugpl: from the catalogue, no note")
check(all(e["objective_integral"] == INTEGRAL[n] for n, e in inst.items()),
      "objective_integral recorded per instance")
saved = dict(fetch_miplib3.CONFIRMED)
fetch_miplib3.CONFIRMED["misc03"] = fetch_miplib3._pinned("misc03", "3361", 1, "test")
try:
    fetch_miplib3.build_manifest(models, cat, b"", INTEGRAL)
    check(False, "a CONFIRMED value that does not confirm the print is refused")
except ValueError:
    check(True, "a CONFIRMED value that does not confirm the print is refused")
finally:
    fetch_miplib3.CONFIRMED.clear()
    fetch_miplib3.CONFIRMED.update(saved)
check(set(fetch_miplib3.OVERRIDES) == {"noswot", "gen"}
      and set(fetch_miplib3.CONFIRMED) == {"10teams", "dcmulti", "misc07", "pk1"},
      "the errata and confirmation tables are the ones the docstring names")

COMMITTED = REPO_ROOT / "data" / "miplib3" / "manifest.json"
if COMMITTED.exists():
    import json
    real = json.loads(COMMITTED.read_text(encoding="utf-8"))
    ri, rx = real["instances"], real["excluded"]
    check(len(ri) == 52 and len(rx) == 13, "the committed manifest: 52 references, 13 excluded",
          f"{len(ri)} and {len(rx)}")
    check({n for n, e in ri.items() if e["reference_source"] == "override"} == {"noswot", "gen"},
          "the committed overrides are noswot and gen")
    unsafe = [n for n, e in ri.items()
              if e["reference_source"] != "override" and "confirmed by" not in
              e.get("reference_note", "") and fetch_miplib3.rounding_risk(
                  e["published_text"], e["published_optimal"], e["objective_integral"])]
    check(not unsafe, "no committed reference is an unconfirmed integer print at risk", str(unsafe))
    check({n for n, e in rx.items() if "no proven value" in e["reason"]}
          == {"fixnet6", "misc03", "pp08a", "pp08aCUTS"}, "the rounding exclusions")
try:
    fetch_miplib3.build_manifest({"rgn": b""}, cat, b"", INTEGRAL)
    check(False, "a catalogue row with no model file is refused")
except ValueError:
    check(True, "a catalogue row with no model file is refused")
try:
    fetch_miplib3.check_archive(b"not the archive")
    check(False, "an archive with another sha256 is refused")
except ValueError:
    check(True, "an archive with another sha256 is refused")

print("matching")
rgn = inst["rgn"]
check(miplib.matches_published(82.19999924, 82.1999, rgn),
      "rgn's true optimum matches its printed value under published_tolerance")
check(not miplib.matches_published(82.19999924, 82.1999, {}),
      "and would not under the 1e-6 rule alone (the reason for the field)")
check(not miplib.matches_published(82.2002, 82.1999, rgn), "more than one unit off misses")
check(not miplib.matches_published(21166.05, 21166.0, inst["gt2"]),
      "an integer-printed value gets no extra tolerance")
check(miplib.match_tolerance(1201500.0, {}) == 1e-6 * 1201500.0
      and miplib.match_tolerance(1201500.0, inst["flugpl"]) == 1e-6 * 1201500.0,
      "never tighter than MATCH_RELATIVE_TOLERANCE")
check(miplib.matches_published(1201501.0, 1201500.0, {})
      and not miplib.matches_published(1201502.0, 1201500.0, {}),
      "a MIPLIB 2017 entry (no field) keeps the 1e-6 relative rule")
check(not miplib.matches_published(None, 1.0, rgn)
      and not miplib.matches_published(float("nan"), 1.0, rgn), "no objective never matches")

print("CSV names")
names = {
    "sankhya60": miplib.default_out_name("miplib3", 1, 3, 60.0, "abc1234"),
    "sankhya300": miplib.default_out_name("miplib3", 1, 3, 300.0, "abc1234"),
    "highs60": compare_suite.default_out_name("miplib3", True, 60.0, "abc1234"),
    "highs300": compare_suite.default_out_name("miplib3", True, 300.0, "abc1234"),
}
names["summary60"] = "summary-" + names["sankhya60"]
names["summary300"] = "summary-" + names["sankhya300"]
check(names["sankhya60"] == "miplib3-60s-seeds3-abc1234.csv"
      and names["sankhya300"] == "miplib3-300s-seeds3-abc1234.csv",
      "miplib.py --set miplib3 names the limit and the seeds", names["sankhya60"])
check(names["highs60"] == "highs-miplib3-60s-abc1234.csv", "the HiGHS CSV", names["highs60"])
check(miplib.default_out_name("miplib2017", 1, 1, 600.0, "abc1234") == "miplib-abc1234.csv"
      and miplib.default_out_name("miplib2017", 2, 3, 300.0, "abc1234")
      == "miplib-tier2-seeds3-abc1234.csv"
      and compare_suite.default_out_name("netlib", False, 120.0, "abc1234")
      == "head-to-head-netlib-partial-abc1234.csv", "MIPLIB 2017's and the suites' names unchanged")

# Every CSV glob in the generators and the demo, f-string fields widened to `*`. A MIPLIB 3
# name must match none of them except its own two in miplib3_doc.py.
SOURCES = [*sorted((REPO_ROOT / "bench" / "runners").glob("*.py")),
           *sorted((REPO_ROOT / "tools").glob("*.py")), REPO_ROOT / "demo" / "run_sih_demo.sh"]
globs = set()
for source in SOURCES:
    if source.name in ("miplib3_doc.py", "test_miplib3.py") or not source.exists():
        continue
    for literal in re.findall(r"[\"']([^\"'\s]*[*?\[][^\"'\s]*\.csv)[\"']",
                              source.read_text(encoding="utf-8", errors="replace")):
        globs.add(re.sub(r"\{[^}]*\}", "*", literal))
globs.discard("*.csv")  # check_result_stamps.py's stamp check reads every CSV on purpose
check({"miplib-*.csv", "miplib-tier2-seeds*-*.csv", "summary-miplib-seeds*-*.csv",
       "miplib-600s-*.csv"} <= globs, f"the scan finds the MIPLIB 2017 globs ({len(globs)})")
hits = sorted((g, n) for g in globs for n in names.values() if fnmatch.fnmatch(n, g))
check(not hits, "no other glob matches a MIPLIB 3 name", str(hits))
check(not latest_result.is_default_named(Path(names["sankhya60"]), "miplib"),
      "latest_result's tier-1 filter refuses it")
own = {limit: (miplib3_doc.sankhya_pattern(limit), miplib3_doc.highs_pattern(limit))
       for limit in miplib3_doc.LIMITS}
check(fnmatch.fnmatch(names["sankhya60"], own[60][0])
      and fnmatch.fnmatch(names["sankhya300"], own[300][0])
      and fnmatch.fnmatch(names["highs60"], own[60][1])
      and fnmatch.fnmatch(names["highs300"], own[300][1]), "its own globs find it")
check(not fnmatch.fnmatch("miplib3-600s-seeds3-abc1234.csv", own[60][0])
      and not fnmatch.fnmatch(names["sankhya300"], own[60][0])
      and not fnmatch.fnmatch("highs-miplib3-60s-partial-abc1234.csv", own[60][1])
      and not fnmatch.fnmatch(names["summary60"], own[60][0]),
      "60 s is not 300 s or 600 s, a partial HiGHS run is not the set, a summary is not a run")

print("HiGHS on a MIP")


class FakeHighs:
    """Records what the worker sets and reads; answers like highspy on a solved MIP."""
    seen: dict = {}

    def __init__(self):
        FakeHighs.seen = {"options": {}}

    def setOptionValue(self, key, value):
        FakeHighs.seen["options"][key] = value

    def readModel(self, path):
        FakeHighs.seen["path"] = path
        FakeHighs.seen["exists"] = Path(path).exists()

    def run(self):
        FakeHighs.seen["ran"] = True

    def getModelStatus(self):
        return "Optimal"

    def modelStatusToString(self, status):
        return status

    def getInfo(self):
        return types.SimpleNamespace(simplex_iteration_count=7, ipm_iteration_count=-1,
                                     qp_iteration_count=-1, objective_function_value=2.0)

    def getLp(self):
        return types.SimpleNamespace(col_names_=["x", "y"], row_names_=["c1"])

    def getSolution(self):
        return types.SimpleNamespace(value_valid=True, dual_valid=False, col_value=[1.0, 1.0],
                                     col_dual=[0.0, 0.0], row_value=[2.0], row_dual=[0.0])

    def getRunTime(self):
        return 0.01

    def version(self):
        return "fake"


sys.modules["highspy"] = types.SimpleNamespace(Highs=FakeHighs)
MIP = """NAME          TINYMIP
ROWS
 N  obj
 G  c1
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    x         obj       1.0          c1        1.0
    y         obj       1.0          c1        1.0
    MARKER                 'MARKER'                 'INTEND'
RHS
    RHS       c1        1.5
BOUNDS
 UP BND       x         10
 UP BND       y         10
ENDATA
"""
with tempfile.TemporaryDirectory() as tmp:
    model_path = Path(tmp) / "tiny.mps"
    model_path.write_text(MIP, encoding="utf-8")
    sol = Path(tmp) / "solution.sol"
    out = rivals._worker_highs(model_path, 60.0, sol)
    options = FakeHighs.seen["options"]
    check(FakeHighs.seen["path"] == str(model_path),
          "an .mps file is read as itself, with no link to make")
    check(options.get("threads") == 1 and options.get("time_limit") == 60.0
          and options.get("mip_feasibility_tolerance") == rivals.FEASIBILITY_TOLERANCE,
          "one thread, the limit, and the MIP tolerance at the project's 1e-7", str(options))
    check(not any("relax" in key for key in options),
          "nothing asks HiGHS to solve the relaxation instead of the MIP")
    check(out["status"] == "optimal" and parse_sol(sol).status == "feasible",
          "a MIP has no duals: written `feasible`, verified primal and integral only")
    qps = Path(tmp) / "model.qps"
    qps.write_text(MIP, encoding="utf-8")
    rivals._worker_highs(qps, 60.0, Path(tmp) / "q.sol")
    check(FakeHighs.seen["path"].endswith("model.mps") and FakeHighs.seen["exists"],
          "a .qps file is read through an .mps link, or a copy where links are refused")

    print("grading on MIPLIB 3")
    model = parse_mps(model_path)
    check(model.col_integer == [True, True], "the verifier's reader sees the integers")
    suite = compare_suite.SUITES["miplib3"]
    check(suite["kind"] == "mip" and suite["solvers"] == ("highs",)
          and suite["manifest"] == "manifest.json", "the suite: HiGHS only, its own manifest")
    out = rivals._worker_highs(model_path, 60.0, sol)
    loose = compare_suite.grade("highs", suite, model, model_path, sol, dict(out), 1.99995,
                                None, 60.0, published_tolerance=1e-4)
    out = rivals._worker_highs(model_path, 60.0, sol)
    strict = compare_suite.grade("highs", suite, model, model_path, sol, dict(out), 1.99995,
                                 None, 60.0)
    check(loose["independently_verified"] is True and loose["verification"] == "primal-only",
          "the integral point verifies", loose["verifier_message"])
    check(loose["matches_reference"] and not strict["matches_reference"],
          "published_tolerance widens the match exactly as in miplib.py, and only when given")

print("doc section")
with tempfile.TemporaryDirectory() as tmp:
    d = Path(tmp)
    empty = miplib3_doc.section({}, {"models_in_archive": 3, "instances": {}, "excluded": {}})
    check(empty.count("Not yet run. Reproduce with:") == 2
          and "python bench/runners/miplib.py --set miplib3 --seeds 3 --time-limit 60" in empty
          and "python bench/runners/compare.py --suite miplib3 --time-limit 300" in empty
          and "python bench/runners/fetch_miplib3.py" in empty,
          "with no CSVs: Not yet run, with the exact commands for both limits")

    def run(name, seed, matched, proved, verified, status=None):
        return {"instance": name, "seed": seed,
                "status": status or ("optimal" if proved else "feasible"),
                "matched_published": int(matched), "proved_optimal": int(proved),
                "independently_verified": verified, "wall_seconds": 2.0 if proved else 60.0,
                "time_to_first_feasible": 0.5, "primal_integral": 1.0, "threads": 1,
                "published_objective": "1.0", "git_commit": "abc1234", "machine": "box"}

    rows = ([run("rgn", s, True, True, 1) for s in range(3)]
            + [run("noswot", 0, False, False, 1), run("noswot", 1, False, False, 1, "optimal"),
               run("noswot", 2, False, False, "", "time_limit")]
            + [run("gt2", 0, True, True, 0), run("gt2", 1, True, True, 1),
               run("gt2", 2, True, False, 1)])
    sankhya = d / "miplib3-60s-seeds3-abc1234.csv"
    with sankhya.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    highs = d / "highs-miplib3-60s-abc1234.csv"
    hrows = [{"instance": "rgn", "status": "optimal", "matches_reference": "1",
              "independently_verified": "1", "solver_seconds": "0.1", "solver_version": "1.x",
              "machine": "box", "time_limit": "60.0"},
             {"instance": "noswot", "status": "optimal", "matches_reference": "0",
              "independently_verified": "1", "solver_seconds": "3.0", "solver_version": "1.x",
              "machine": "box", "time_limit": "60.0"}]
    with highs.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(hrows[0]))
        w.writeheader()
        w.writerows(hrows)
    man = {"models_in_archive": 4,
           "instances": {"rgn": {"published_text": "82.1999"},
                         "noswot": {"published_text": "-41.00000885",
                                    "catalogue_int_soln": "-43", "reference_source": "override",
                                    "reference_note": "the catalogue prints -43"},
                         "gt2": {"published_text": "21166.000"},
                         "p0033": {"published_text": "3089"}},
           "excluded": {"seymour": {"reason": "INT SOLN marked (not opt) in miplib.cat"}}}
    text = miplib3_doc.section({60: (sankhya, highs)}, man)
    check("**SANKHYA**, 9 runs (3 instances x 3 seeds): **6** reached the published optimum, "
          "**5** proved it, **7 of 8** points verified" in text, "the totals line")
    check("| `rgn` | 82.1999 | 3/3 (0 1 2) | 3/3 (0 1 2) | 3/3 |" in text,
          "a per-instance row: reached and proved seeds, verified")
    check("Reported optimal but rejected by the verifier: `gt2` seed 0." in text,
          "an optimal the verifier rejects is named on its own line")
    check("References that override the catalogue" in text
          and "`noswot` -43 -> -41.00000885: the catalogue prints -43" in text,
          "an override is stated with the printed value, the proven one and why")
    check("| `p0033` | 3089 | no run | - | - | - |" in text
          and "no SANKHYA run in the CSV: `p0033`" in text, "an instance with no run is named")
    check("`noswot` (optimal, NOT matched, verified, 3.0 s)" in text
          and "| `gt2` | 21166.000 |" in text and "not run |" in text,
          "HiGHS's verdicts, its misses named, an instance it did not run")
    check("Excluded, with no reference: `seymour` (marked \"(not opt)\")" in text,
          "exclusions named with their reason")
    check("#### At 300 s\n\nNot yet run." in text, "the other limit still says Not yet run")

print(f"\n{'all passed' if FAILURES == 0 else f'{FAILURES} FAILED'}")
sys.exit(1 if FAILURES else 0)
