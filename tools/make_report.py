#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate docs/REPORT.md, the one document a jury reads, from the evidence on main.

docs/REPORT.md is NOT hand-written. Every number in it is computed here from a CSV in
bench/results/ - the same file docs/BENCHMARKS.md reads for that measurement, chosen by the
same bench/runners/latest_result.py - and printed beside that file's name and the commit
stamped in it. The passages on architecture, provenance and limits are copied verbatim from
the documents that own them. The prose around them, and the jury Q&A, live in this script
with `{name}` slots for the numbers, so a sentence cannot keep a figure its CSV has lost.

    python tools/make_report.py            # write docs/REPORT.md
    python tools/make_report.py --check    # exit 1 if the committed file is out of date (CI)
    python tools/make_report.py --pdf      # also docs/REPORT.pdf, when pandoc and a PDF
                                           # engine are installed; says so and skips if not

Run it after bench/runners/make_benchmarks_doc.py whenever a results CSV lands.
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "bench" / "runners"))
sys.path.insert(0, str(REPO_ROOT / "tools"))
import head_to_head_doc as h2h  # noqa: E402
import make_benchmarks_doc as mbd  # noqa: E402
from report_qa import CITES, QA  # noqa: E402

OUTPUT = REPO_ROOT / "docs" / "REPORT.md"
PDF_ENGINES = ("xelatex", "pdflatex", "lualatex", "tectonic", "typst", "wkhtmltopdf",
               "weasyprint")
EVIDENCE: list[tuple[str, str, str, str]] = []  # (measurement, value, csv, commit)
F: dict[str, object] = {}  # every number the prose and the Q&A may use


def load(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def count(rows: list[dict], key: str, *values: str) -> int:
    return sum(r.get(key) in values for r in rows)


def fnum(text: str | None) -> float | None:
    try:
        return float(text)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def evidence(key: str, path: Path | None, measure) -> None:
    """Run `measure(rows)` -> (headline text, {slot: value}) on one CSV and record both.
    A missing CSV leaves its slots saying so, which the report then prints instead of a
    number."""
    if path is None:
        F[key + "_csv"] = "no CSV on main"
        return
    rows = load(path)
    text, slots = measure(rows)
    F.update({f"{key}_{name}": value for name, value in slots.items()})
    F[key + "_csv"] = f"`bench/results/{path.name}`"
    F[key + "_commit"] = rows[0].get("git_commit", "?")
    EVIDENCE.append((key, text, path.name, rows[0].get("git_commit", "?")))


class Slots(dict):
    def __missing__(self, key: str) -> str:
        return "(no CSV on main)"


def fill(text: str) -> str:
    return text.format_map(Slots(F))


# --- one function per measurement -------------------------------------------------------

def m_netlib(rows):
    grade = mbd.exact_grade(rows)
    exact = grade[0] if grade else None
    slots = {"n": len(rows), "verified": count(rows, "independently_verified", "1"),
             "readme": count(rows, "passed", "1"), "exact": exact}
    return (f"Netlib LP, full set: {exact} of {len(rows)} within 1e-6 of Koch's exact optimum "
            f"and verified; {slots['readme']} match the readme; {slots['verified']} accepted "
            "by the independent verifier", slots)


def m_certified(rows):
    optimal = [r for r in rows if r["status"] == "optimal"]
    gaps = [fnum(r["certified_relative_gap"]) for r in optimal]
    tight = sum(g is not None and g <= 1e-6 for g in gaps)
    none = sum(r["safe_lower_bound"] in ("", "-inf") for r in optimal)
    slots = {"optimal": len(optimal), "tight": tight, "none": none,
             "loose": len(optimal) - tight - none}
    return (f"Safe bounds on Netlib: {tight} of {len(optimal)} optimal answers certified to "
            f"1e-6 by a rigorous bound, {none} with no finite bound", slots)


def m_passed(label):
    def measure(rows):
        passed = count(rows, "passed", "1")
        return f"{label}: {passed} of {len(rows)}", {"n": len(rows), "passed": passed}
    return measure


def m_miplib(rows):
    slots = {"n": len(rows), "matched": count(rows, "matched_published", "1"),
             "proved": count(rows, "proved_optimal", "1")}
    return (f"MIPLIB 2017 easy set: {slots['matched']} of {len(rows)} reach the published "
            f"optimum, {slots['proved']} prove it", slots)


def m_maros(rows):
    slots = {"n": len(rows), "optimal": count(rows, "status", "optimal"),
             "matched": count(rows, "matches_reference", "1"),
             "verified": count(rows, "independently_verified", "1"),
             "rejected": sum(r["status"] == "optimal" and r["independently_verified"] == "0"
                             for r in rows)}
    return (f"Maros-Meszaros convex QP: {slots['matched']} of {len(rows)} at the published "
            f"objective, {slots['verified']} verifier-accepted, {slots['rejected']} optimal "
            "answers rejected", slots)


def m_nlp(label):
    def measure(rows):
        slots = {"n": len(rows), "matched": count(rows, "match", "yes"),
                 "verified": count(rows, "verified", "yes")}
        return (f"{label}: {slots['matched']} of {len(rows)} at the published objective, "
                f"{slots['verified']} accepted by the checker", slots)
    return measure


def m_stress(rows):
    slots = {}
    for solver in ("sankhya", "highs"):
        mine = [r for r in rows if r["solver"] == solver]
        slots[solver + "_n"] = len(mine)
        slots[solver + "_version"] = mine[0]["solver_version"] if mine else "?"
        for verdict in ("correct", "wrong", "failed"):
            slots[f"{solver}_{verdict}"] = count(mine, "verdict", verdict)
    return (f"Stress set, {slots['sankhya_n']} badly scaled and adversarial LPs: SANKHYA "
            f"{slots['sankhya_correct']} correct / {slots['sankhya_wrong']} wrong / "
            f"{slots['sankhya_failed']} declined; HiGHS on its defaults "
            f"{slots['highs_correct']} / {slots['highs_wrong']} / {slots['highs_failed']}",
            slots)


def m_scenarios(rows):
    slots = {"n": len(rows), "verified": count(rows, "verified", "yes")}
    return f"`sankhya scenarios`: {slots['verified']} of {len(rows)} scenarios verified", slots


def m_million(rows):
    done = [r for r in rows if r["status"] == "optimal" and r["verified"] == "1"]
    best = "; ".join(f"{r['family']} {int(r['rows']):,} rows by `{r['arm']}` in "
                     f"{float(r['wall_seconds']):,.0f} s" for r in done)
    slots = {"n": len(rows), "done": len(done), "list": best or "none"}
    return (f"A million rows on the CPU: {len(done)} of {len(rows)} engine runs optimal and "
            f"verified ({best})", slots)


def m_compare(rows):
    slots = {"n": len(rows), "agree": count(rows, "objectives_agree", "1")}
    return (f"HiGHS agreement, medium Netlib tier: objectives agree on {slots['agree']} of "
            f"{len(rows)}", slots)


def m_exact(rows):
    done = [r for r in rows if r["exact_derivation"] == "done"]
    slots = {"n": len(rows), "done": len(done),
             "repaired": count(rows, "exact_verification", "verified"),
             "values": f"{sum(int(r['values_compared']) for r in done):,}",
             "corrected": f"{sum(int(r['values_disagreeing']) for r in done):,}"}
    return (f"Exact sensitivity: {len(done)} of {len(rows)} LPs with every dual, reduced cost "
            f"and range re-derived in rational arithmetic; {slots['corrected']} of "
            f"{slots['values']} floating-point values corrected", slots)


def m_head_to_head(rows):
    if rows[0]["suite"] == "netlib":
        rows = [r for r in rows if r["instance"] not in h2h.GENERATED_NETLIB]
    limit = float(rows[0]["time_limit"])
    slots: dict[str, object] = {"n": len({r["instance"] for r in rows}),
                                "machine": rows[0]["machine"], "limit": f"{limit:g}"}
    for solver in ("sankhya", "highs"):
        mine = [r for r in rows if r["solver"] == solver]
        charged = [float(r["solver_seconds"]) if r["counted_for_time"] == "1" else limit
                   for r in mine]
        slots[solver + "_sgm"] = f"{mbd.shifted_geometric_mean(charged, h2h.SHIFT_SECONDS):.3f}"
        slots[solver + "_counted"] = count(mine, "counted_for_time", "1")
        slots[solver + "_solved"] = count(mine, "status", "optimal")
        slots[solver + "_matched"] = count(mine, "matches_reference", "1")
    slots["rejected"] = sum(r["independently_verified"] == "0" and r["solver"] != "sankhya"
                            for r in rows)
    vs = h2h.per_instance(rows, "highs") or {"wins": 0, "ties": 0, "losses": 0, "compared": 0,
                                             "median": None}
    slots.update({k: vs[k] for k in ("wins", "ties", "losses", "compared")})
    slots["median"] = "-" if vs["median"] is None else f"{vs['median']:.2f}x"
    slots["table"] = "\n".join(h2h.solver_table(rows, mbd.shifted_geometric_mean) + [""]
                               + h2h.per_instance_table(rows))
    return (f"Head-to-head, {h2h.TITLES[rows[0]['suite']]}, {slots['n']} instances: SGM10 "
            f"{slots['sankhya_sgm']} s against HiGHS {slots['highs_sgm']} s; against HiGHS "
            f"faster on {vs['wins']}, tie on {vs['ties']}, slower on {vs['losses']}", slots)


def m_gpu(rows):
    """The card against the faster CPU arm, per model and tolerance, as a table."""
    lines = ["| model | rows | tolerance | card | faster CPU arm | card status | speed-up |",
             "|---|---:|---|---:|---:|---|---:|"]
    ratios = []
    cells = sorted({(r["instance"], r["tolerance"]) for r in rows})
    for instance, tolerance in cells:
        mine = [r for r in rows if (r["instance"], r["tolerance"]) == (instance, tolerance)]
        gpu = next((r for r in mine if r["arm"] == "gpu"), None)
        cpu = min((r for r in mine if r["arm"] != "gpu"), key=lambda r: float(r["seconds"]),
                  default=None)
        if gpu is None or cpu is None:
            continue
        ratio = float(cpu["seconds"]) / float(gpu["seconds"])
        ratios.append(ratio)
        lines.append(f"| `{instance}` | {int(gpu['rows']):,} | {tolerance} | "
                     f"{float(gpu['seconds']):.1f} s | {float(cpu['seconds']):.1f} s "
                     f"(`{cpu['arm']}`) | {gpu['status']} | {ratio:.2f}x |")
    slots = {"table": "\n".join(lines), "gpu": rows[0].get("gpu", "?").split(" (")[0],
             "best": f"{max(ratios):.2f}x", "worst": f"{min(ratios):.2f}x",
             "limits": sum(r["arm"] == "gpu" and r["status"] == "time_limit" for r in rows),
             "cells": len(ratios)}
    return (f"GPU PDHG on real models, {slots['gpu']}: {slots['worst']} to {slots['best']} "
            f"against the faster CPU arm over {len(ratios)} model-tolerance cells, "
            f"{slots['limits']} of them ending at the time limit", slots)


def gather() -> None:
    EVIDENCE.clear()
    F.clear()
    new = mbd.newest
    evidence("netlib", new("netlib-full-*.csv"), m_netlib)
    for suite in ("netlib", "kennington", "maros-meszaros"):
        evidence("h2h_" + suite.split("-")[0],
                 new(f"head-to-head-{suite}-*.csv", prefix=f"head-to-head-{suite}"),
                 m_head_to_head)
    evidence("certified", new("certified-gap-netlib-full-*.csv"), m_certified)
    evidence("infeasible", new("netlib-infeasible-*.csv"),
             m_passed("Netlib infeasible set, verdict with a verified Farkas certificate"))
    evidence("exact", new("exact-sensitivity-*.csv", prefix="exact-sensitivity"), m_exact)
    evidence("kennington", new("kennington-full-*.csv"),
             m_passed("Kennington LP, matched and verified"))
    evidence("mittelmann", new("mittelmann-*.csv"),
             m_passed("Mittelmann LP, solved and verified inside the limit"))
    evidence("miplib", new("miplib-*.csv", prefix="miplib"), m_miplib)
    evidence("maros", new("maros-meszaros-*.csv", prefix="maros-meszaros"), m_maros)
    evidence("hs", new("nlp-hs-*.csv"), m_nlp("Hock-Schittkowski NLP"))
    evidence("minlp", new("nlp-minlplib-*.csv"), m_nlp("Convex MINLPLib"))
    evidence("stress", new("stress-*.csv", prefix="stress"), m_stress)
    evidence("scenarios", new("scenarios-*.csv"), m_scenarios)
    evidence("million", mbd.million_doc.latest_csv(mbd.RESULTS_DIR), m_million)
    evidence("compare", new("compare-highs-medium-*.csv"), m_compare)
    evidence("gpu", new("gpu-real-a100-*.csv"), m_gpu)


# --- passages copied from the documents that own them -------------------------------------

def passage(doc: str, heading: str) -> str:
    """The section of `doc` whose heading starts with `heading`, verbatim, up to the next
    heading of the same or a higher level, its own sub-headings pushed down to level 4."""
    lines = (REPO_ROOT / doc).read_text(encoding="utf-8").splitlines()
    fence, start, level = False, None, 0
    out: list[str] = []
    for index, line in enumerate(lines):
        if line.startswith("```"):
            fence = not fence
        match = None if fence else re.match(r"(#+) +(.*)", line)
        if start is None:
            if match and match.group(2).startswith(heading):
                start, level = index, len(match.group(1))
            continue
        if match and len(match.group(1)) <= level:
            break
        out.append(f"#### {match.group(2)}" if match else line)
    if start is None:
        raise SystemExit(f"make_report: {doc} has no section starting with {heading!r}")
    body = "\n".join(out).strip().removesuffix("---").rstrip()
    return f"*From `{doc}`, section \"{heading}\", copied as it stands:*\n\n{body}\n"


def qa_section() -> str:
    assert len(QA) == 25, f"the jury Q&A must hold 25 questions, not {len(QA)}"
    out = []
    for number, (question, answer, cite) in enumerate(QA, start=1):
        assert CITES.search(cite), f"Q{number} cites no CSV or document"
        out.append(f"**Q{number}. {question}**\n\n{fill(answer)}\n\n*Evidence: {fill(cite)}*\n")
    return "\n".join(out)


def build() -> str:
    gather()
    table = "\n".join(f"| {text} | `bench/results/{name}` | `{commit}` |"
                      for _, text, name, commit in EVIDENCE)
    return (fill("""# SANKHYA — report for the jury

<!-- GENERATED FILE. Do not edit by hand. -->
<!-- Regenerate with: python tools/make_report.py -->

Smart India Hackathon 2026, problem statement SIH26119 (Mangalore Refinery and
Petrochemicals Limited): an indigenous GPU-accelerated optimization solver. This report is
generated by `tools/make_report.py`. Every number in it is computed from a CSV in
`bench/results/` and printed with that file's name and the commit stamped in it; the passages
marked "copied as it stands" come verbatim from the document named. CI regenerates it and
fails if the committed copy differs.

Contents: 1 What SANKHYA is · 2 Architecture · 3 Headline numbers · 4 Head-to-head ·
5 Verification · 6 Honest limits · 7 Provenance · 8 Jury Q&A

## 1. What SANKHYA is

A mathematical optimization solver written from scratch in C++20: linear programs, mixed
integer linear programs, convex quadratic programs, and local nonlinear programs, behind one
entry point, with a C API, a command line and Python bindings. It has two families of LP
engine: a revised simplex (primal and dual) that produces a basis and is the node solver for
branch and cut, and a restarted primal-dual first-order method (PDHG) that runs on the CPU
and, through CUDA, on the GPU. An interior point method sits between them.

What sets it apart is not speed. It is that every answer is checked by a program that shares
no code with the solver, infeasibility comes with a certificate, optimal values come with a
bound that survives floating-point error, and the sensitivity a planner prices from can be
re-derived in exact arithmetic. Section 5 gives the numbers and section 6 says where the
solver is behind.

## 2. Architecture

""") + passage("docs/ARCHITECTURE.md", "1. The shape in one paragraph") + "\n"
                + passage("docs/ARCHITECTURE.md", "2. Modules and their boundaries") + fill("""
## 3. Headline numbers, each with its CSV and commit

The file named is the one `docs/BENCHMARKS.md` reads for the same measurement; the commit is
the one stamped in the file's rows, the build that produced it.

| measurement | CSV | commit |
|---|---|---|
""") + table + fill("""

## 4. Head-to-head against HiGHS, SCIP, CBC/Clp and GLPK

Every solver is a separate process on the same machine, one thread, the same time limit;
every answer, a rival's included, goes through `tools/verify_solution.py`, and only a run
that is optimal, at the published optimum and verified counts for time. The method is in
`docs/BENCHMARKS.md` section 4a, with the performance profiles and every failure named.

### Netlib LP

{h2h_netlib_csv}, commit `{h2h_netlib_commit}`, {h2h_netlib_limit} s limit, machine
`{h2h_netlib_machine}`.

{h2h_netlib_table}

### Kennington LP

{h2h_kennington_csv}, commit `{h2h_kennington_commit}`, {h2h_kennington_limit} s limit,
machine `{h2h_kennington_machine}`.

{h2h_kennington_table}

### Maros-Meszaros QP

{h2h_maros_csv}, commit `{h2h_maros_commit}`, {h2h_maros_limit} s limit, machine
`{h2h_maros_machine}`.

{h2h_maros_table}

## 5. Verification: why an answer can be believed

- **An independent verifier.** `tools/verify_solution.py` reads only the model file and the
  written `.sol` file, with its own parsers, and never links the C++. On the full Netlib set
  it accepts {netlib_verified} of {netlib_n} answers, and {netlib_exact} of {netlib_n} are
  within 1e-6 of Koch's exact rational optimum ({netlib_csv}, `docs/BENCHMARKS.md`
  section 1c). In the head-to-head run the same verifier rejected {h2h_netlib_rejected}
  rival answers on Netlib ({h2h_netlib_csv}, section 4a.1).
- **Certificates of infeasibility.** {infeasible_passed} of {infeasible_n} of Netlib's
  infeasible LPs end with a Farkas certificate the verifier accepts ({infeasible_csv},
  section 3a). A verdict with no certificate is not counted.
- **Safe bounds.** {certified_tight} of {certified_optimal} optimal Netlib answers carry a
  Neumaier-Shcherbina bound within 1e-6 of the objective, re-derived exactly by the
  verifier; {certified_none} have no finite bound and say so ({certified_csv},
  section 1c.2).
- **Exact sensitivity.** On {exact_done} of {exact_n} LPs every dual, reduced cost and range
  was re-derived in rational arithmetic, correcting {exact_corrected} of {exact_values}
  floating-point values ({exact_csv}; `docs/PROVENANCE.md` section 3).
- **Agreement with an established solver.** HiGHS, as a separate process, agrees on the
  objective on {compare_agree} of {compare_n} medium-tier Netlib instances ({compare_csv},
  section 4).

## 6. Honest limits

- **Speed against HiGHS.** Netlib SGM {h2h_netlib_sankhya_sgm} s against
  {h2h_netlib_highs_sgm} s; Kennington {h2h_kennington_sankhya_sgm} s against
  {h2h_kennington_highs_sgm} s, slower on {h2h_kennington_losses} of
  {h2h_kennington_compared} instances ({h2h_netlib_csv}, {h2h_kennington_csv}).
- **Large LPs.** Mittelmann: {mittelmann_passed} of {mittelmann_n} inside the limit
  ({mittelmann_csv}, `docs/BENCHMARKS.md` section 1d).
- **MILP proofs.** {miplib_proved} of {miplib_n} MIPLIB instances proved, {miplib_matched}
  at the published optimum ({miplib_csv}, section 2).
- **Declined instances.** Stress set: {stress_sankhya_failed} of {stress_sankhya_n} declined,
  {stress_sankhya_wrong} wrong ({stress_csv}, section 5b). Maros-Meszaros: {maros_optimal}
  of {maros_n} optimal ({maros_csv}, section 2b). Safe bounds: {certified_none} of
  {certified_optimal} with no finite bound ({certified_csv}).
- **No real industrial model.** Every refinery number is from a generated model.
- **The GPU, against the faster CPU arm of the same engine** ({gpu_csv}, commit
  `{gpu_commit}`, {gpu_gpu}; `docs/BENCHMARKS.md` section 1g.3, where the small-model losses
  of section 1g sit beside it):

{gpu_table}

""") + passage("docs/BENCHMARKS.md", "6. What these numbers do not say") + "\n"
                + passage("docs/NEGATIVE-RESULTS.md", "4. Withdrawn claims") + """
## 7. Provenance

""" + passage("docs/PROVENANCE.md", "1. The red line") + fill("""
The dependency table, the algorithm citation table, the link line and the SBOM are in
`docs/PROVENANCE.md` sections 2 to 4; `docs/sbom.spdx.json` is the committed bill of
materials.

## 8. Jury Q&A

Twenty-five questions a refinery or optimization judge is likely to ask. Each answer ends
with the file or section it rests on; a number in an answer is the one computed for
section 3.

""") + qa_section())


def write_pdf() -> None:
    engine = next((e for e in PDF_ENGINES if shutil.which(e)), None)
    if not shutil.which("pandoc") or engine is None:
        print("make_report: no pandoc with a PDF engine here; Markdown only")
        return
    target = OUTPUT.with_suffix(".pdf")
    done = subprocess.run(["pandoc", OUTPUT.name, "-o", target.name, f"--pdf-engine={engine}",
                           "-V", "geometry:margin=2cm"], cwd=OUTPUT.parent)
    print(f"make_report: {'wrote docs/' + target.name if done.returncode == 0 else 'pandoc failed; Markdown only'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if docs/REPORT.md is not what this script generates")
    parser.add_argument("--pdf", action="store_true", help="also write docs/REPORT.pdf")
    args = parser.parse_args()
    document = build()
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8").replace("\r\n", "\n") if OUTPUT.exists() \
            else ""
        if current != document:
            print("FAIL: docs/REPORT.md is out of date; run python tools/make_report.py")
            return 1
        print(f"docs/REPORT.md is current: {len(EVIDENCE)} measurements, {len(QA)} questions")
        return 0
    OUTPUT.write_text(document, encoding="utf-8", newline="\n")
    print(f"wrote docs/REPORT.md from {len(EVIDENCE)} CSVs")
    if args.pdf:
        write_pdf()
    return 0


if __name__ == "__main__":
    sys.exit(main())
