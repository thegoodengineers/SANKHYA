#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SANKHYA against the four open-source solvers the problem statement names.

PS26119 names COIN-OR CBC, HiGHS, GLPK and SCIP. This renders one table from four CSVs,
each from a job in which all five solvers ran on the same machine, one thread each, with the
same time limit, every returned point checked by tools/verify_solution.py:

    head-to-head-netlib-<sha>.csv          compare.py --suite netlib        (LP, 120 s)
    head-to-head-miplib3-60s-<sha>.csv     compare.py --suite miplib3-all   (MILP, 60 s)
    stress-<sha>.csv                       stress.py --solvers ...          (LP, 60 s)
    head-to-head-maros-meszaros-partial-<sha>.csv   compare.py --suite maros-meszaros
                                           --solvers sankhya,scip           (QP, 60 s)

A cell is `proved / matched / verified / wrong`, counted from the rows:

*   Netlib, MIPLIB 3 and Maros-Meszaros. proved: the solver's own status is optimal.
    matched: proved, and the objective at its point is the published optimum (Koch's exact
    value on Netlib), by compare_suite.py's rule. verified: proved, and the verifier
    accepts the point. wrong: a claimed optimum further from the published value than the
    solver was asked to be (1e-6 relative on an LP or a QP, the 1e-4 relative gap on a
    MILP), or `infeasible` / `unbounded` on a model that has an optimum.
*   Stress set. proved: a verdict (optimal, infeasible or unbounded). matched: stress.py's
    verdict `correct`. verified: the verifier accepts the file. wrong: a verdict the
    instance contradicts, or an optimum off the known one by more than 1e-6 relative.

A point the verifier rejects whose verdict and objective are right is not a wrong answer:
it is the difference between proved and verified, and it is named beside the wrong ones.

The table goes into README.md (make_scoreboard.py) and docs/BENCHMARKS.md
(make_benchmarks_doc.py), with every wrong answer named in the latter.
"""
from __future__ import annotations

import csv
from pathlib import Path

import latest_result
import rivals

SOLVERS = ("sankhya", "highs", "scip", "cbc", "glpk")
SUITES = (("netlib", "Netlib LP", "head-to-head-netlib-*.csv", "head-to-head-netlib"),
          ("miplib3", "MIPLIB 3 MILP", "head-to-head-miplib3-60s-*.csv",
           "head-to-head-miplib3-60s"),
          ("stress", "Stress set LP", "stress-*.csv", "stress"),
          # SCIP is the one of the four that takes a quadratic objective and is not already
          # in section 4a.3's nine-solver run at a current commit: SANKHYA and SCIP alone.
          ("maros-meszaros", "Maros-Meszaros QP", "head-to-head-maros-meszaros-partial-*.csv",
           "head-to-head-maros-meszaros-partial"))
TITLE = "SANKHYA against the four open-source solvers the problem statement names"


def paths() -> dict[str, Path | None]:
    return {key: latest_result.latest(pattern, prefix=prefix)
            for key, _, pattern, prefix in SUITES}


def read_rows(path: Path) -> list[dict]:
    """The CSV's rows, the LP suites' `cbc-clp` (CBC's LP engine, the `clp` binary) under
    `cbc`, so the CBC column is one column."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["engine"] = row["solver"]
        if row["solver"] == "cbc-clp":
            row["solver"] = "cbc"
    return rows


def is_wrong(key: str, row: dict) -> bool:
    if key == "stress":
        if row["verdict"] != "wrong":
            return False
        return (row["status"] != row["expected_status"]
                or (row["status"] == "optimal" and float(row["relative_gap"] or 0.0) > 1e-6))
    if row["status"] in ("infeasible", "unbounded"):
        return True
    if row["status"] != "optimal":
        return False
    correct = row["matches_exact"] if row.get("matches_exact") else row["matches_reference"]
    if correct == "1" or not row["relative_gap"]:
        return False
    return key != "miplib3" or float(row["relative_gap"]) > rivals.MIP_RELATIVE_GAP


def counts(key: str, rows: list[dict]) -> tuple[int, int, int, int]:
    """(proved, matched, verified, wrong) for one solver's rows of one suite."""
    if key == "stress":
        return (sum(r["status"] in ("optimal", "infeasible", "unbounded") for r in rows),
                sum(r["verdict"] == "correct" for r in rows),
                sum(r["verified"] == "1" for r in rows),
                sum(is_wrong(key, r) for r in rows))
    proved = [r for r in rows if r["status"] == "optimal"]
    return (len(proved),
            sum((r["matches_exact"] if r.get("matches_exact") else r["matches_reference"]) == "1"
                for r in proved),
            sum(r["independently_verified"] == "1" for r in proved),
            sum(is_wrong(key, r) for r in rows))


def table(found: dict[str, Path | None]) -> list[str]:
    out = [f"**{TITLE}** (HiGHS, SCIP, COIN-OR CBC, GLPK). Each row is one job: all five "
           "solvers on the same machine, one thread each, the same time limit, each a "
           "separate process, every point checked by `tools/verify_solution.py`. A cell is "
           "**proved / matched / verified / wrong**: the solver's own claim of an optimum "
           "(on the stress set, of any verdict); that claim at the published optimum (on "
           "the stress set, graded correct); the point accepted by the verifier; and a "
           "wrong answer - an optimum off the published value by more than the solver was "
           "asked for, or a verdict the model contradicts. A rejected point with the right "
           "objective is not counted wrong: it is the gap between proved and verified. CBC is its `cbc` binary on MILPs and its LP engine, the "
           "`clp` binary, on LPs.", "",
           "| suite | instances | limit | " + " | ".join(rivals.LABELS[s] for s in SOLVERS)
           + " | CSV | commit |", "|---|---:|---:|" + "---|" * len(SOLVERS) + "---|---|"]
    versions: dict[str, set[str]] = {}
    ranking = []
    for key, title, pattern, _ in SUITES:
        path = found.get(key)
        if path is None:
            out.append(f"| {title} | - | - | " + "- | " * len(SOLVERS)
                       + f"no `{pattern}` committed | - |")
            continue
        rows = read_rows(path)
        cells, scores = [], []
        for solver in SOLVERS:
            mine = [r for r in rows if r["solver"] == solver]
            if not mine:
                cells.append("-")
                continue
            if solver != "sankhya":
                versions.setdefault(mine[0]["engine"], set()).update(
                    r["solver_version"] for r in mine if r["solver_version"])
            proved, matched, verified, wrong = counts(key, mine)
            cells.append(f"{proved} / {matched} / {verified} / {wrong}")
            both = (matched if key == "stress" else
                    sum(r["counted_for_time"] == "1" for r in mine))
            scores.append((both, rivals.LABELS[solver]))
        limit = rows[0].get("time_limit")
        out.append(f"| {title} | {len({r['instance'] for r in rows})} "
                   f"| {f'{float(limit):g} s' if limit else '60 s'} | " + " | ".join(cells)
                   + f" | `bench/results/{path.name}` | `{rows[0]['git_commit']}` |")
        what = "graded correct" if key == "stress" else "matched and verified"
        ranking.append(f"{title}, {what}: " + ", ".join(
            f"{label} {n}" for n, label in sorted(scores, key=lambda s: -s[0])) + ".")
    out += ["", " ".join(ranking)]
    if versions:
        out += ["", "Versions, as each solver reported them in those runs: " + ", ".join(
            f"{'Clp' if engine == 'cbc-clp' else rivals.LABELS[engine]} "
            f"{' and '.join(sorted(found_versions))}"
            for engine, found_versions in versions.items() if found_versions) + "."]
    return out


def is_rejected(key: str, row: dict) -> bool:
    """A verdict whose file the verifier rejected and which is_wrong() does not count."""
    if is_wrong(key, row):
        return False
    if key == "stress":
        return row["verified"] == "0"
    return row["status"] == "optimal" and row["independently_verified"] == "0"


def named(found: dict[str, Path | None], picked) -> list[str]:
    """Every row `picked(key, row)` takes, named with its reason, whichever solver's."""
    out = []
    for key, title, _, _ in SUITES:
        if found.get(key) is None or key == "stress":  # section 5b names the stress set's
            continue
        for row in read_rows(found[key]):
            if not picked(key, row):
                continue
            why = (row["reason"] if key == "stress" else
                   f"{row['status']}; " + (row["verifier_message"]
                                           or f"relative gap {row['relative_gap']}"))
            out.append(f"- {title}, {rivals.LABELS[row['engine']]} on `{row['instance']}`: "
                       f"{why.replace('|', '/')[:220]}")
    return out or ["- none, by any solver, on any of these suites."]


def miplib3_detail(path: Path, sgm) -> list[str]:
    """Per solver on MIPLIB 3: the counts, how every other run ended, and the time."""
    rows = read_rows(path)
    limit = float(rows[0]["time_limit"])
    out = [f"**MIPLIB 3 in detail** (`bench/results/{path.name}`, machine "
           f"`{rows[0]['machine']}`). Every solver stops at a relative gap of "
           f"{rivals.MIP_RELATIVE_GAP:g}, SANKHYA's default and HiGHS's, set explicitly for "
           "SCIP, CBC and GLPK; everything else is each solver's default. SANKHYA runs once, "
           "at its default seed, where section 2a reports the best of three seeds. The time "
           f"is the shifted geometric mean of each solver's own clock (shift 10 s), a run "
           f"that is not proved, matched and verified charged the {limit:g} s limit.", "",
           "| solver | version | proved | matched | proved, matched and verified | wrong "
           "| not proved, and how | SGM time |", "|---|---|---:|---:|---:|---:|---|---:|"]
    for solver in SOLVERS:
        mine = [r for r in rows if r["solver"] == solver]
        if not mine:
            continue
        proved, matched, _, wrong = counts("miplib3", mine)
        ended: dict[str, int] = {}
        for r in mine:
            if r["status"] != "optimal":
                ended[r["status"]] = ended.get(r["status"], 0) + 1
        charged = [float(r["solver_seconds"]) if r["counted_for_time"] == "1" else limit
                   for r in mine]
        out.append(f"| {rivals.LABELS[solver]} | {mine[0]['solver_version']} | {proved} "
                   f"| {matched} | {sum(r['counted_for_time'] == '1' for r in mine)} | {wrong} "
                   f"| {', '.join(f'{n} {s}' for s, n in sorted(ended.items())) or '-'} "
                   f"| {sgm(charged, 10.0):.2f} s |")
    names = lambda picked: ", ".join(f"`{n}`" for n in sorted(picked)) or "none"  # noqa: E731
    counted = {solver: {r["instance"] for r in rows
                        if r["solver"] == solver and r["counted_for_time"] == "1"}
               for solver in SOLVERS}
    ours = counted["sankhya"]
    out += ["", "Instance by instance, proved, matched and verified:", ""]
    for solver in SOLVERS[1:]:
        theirs = counted[solver]
        out.append(f"- {rivals.LABELS[solver]} and not SANKHYA ({len(theirs - ours)}): "
                   f"{names(theirs - ours)}. SANKHYA and not {rivals.LABELS[solver]} "
                   f"({len(ours - theirs)}): {names(ours - theirs)}.")
    return out + [""]


def section(found: dict[str, Path | None], sgm) -> str:
    out = [*table(found), "", "**Every wrong answer counted above** (the stress set's are "
           "named, with every rejection, in section 5b):", "",
           *named(found, is_wrong), "",
           "**Every other answer the verifier rejected** (proved, not verified):", "",
           *named(found, is_rejected), ""]
    if found.get("miplib3") is not None:
        out += miplib3_detail(found["miplib3"], sgm)
    return "\n".join(out)
