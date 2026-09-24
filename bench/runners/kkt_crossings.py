#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The relative-KKT crossing columns of a PDHG run (#486), shared by the runners.

The engine records, in one run, the solver clock at which the relative KKT error first
came at or under 1e-4, 1e-6 and 1e-8 (src/pdhg/pdhg.cpp, stats JSON `effort`). The error is
the PDLP definition (Applegate et al., "Practical large-scale linear programming using
primal-dual hybrid gradient", NeurIPS 2021; the same definition in arXiv 2501.07018): the
largest of

    ||A x - proj_[l,u](A x)||_2 / (1 + ||b||_2)          relative primal residual
    ||c - A'y - z||_2 / (1 + ||c||_2)                    relative dual residual
    |c'x - b'y| / (1 + |c'x| + |b'y|)                    relative gap

on the unscaled model, b the finite row bounds, evaluated on the better of the current and
the averaged iterate (src/pdhg/pdhg_evaluate.cpp). A crossing is taken at the engine's
evaluation interval, so it is an upper bound within one interval, and it is taken in the
first-order phase, before any polish, so it is the polishing-off figure.

The stats JSON carries the fields for every engine (NaN where nothing was recorded). A
column that says "never reached" about a simplex run would be a claim about the wrong
engine, so a non-PDHG row gets blanks here, and a PDHG row keeps the writer's `nan` for a
level the run never reached.
"""

from __future__ import annotations

import math

COLUMNS = ("kkt_1e4_seconds", "kkt_1e6_seconds", "kkt_1e8_seconds")
# The same crossings as iteration counts (-1 where never reached). Seconds are what a
# published comparison quotes; the iteration is the part that does not move with the load
# on the machine, so a runner writes both.
ITERATION_COLUMNS = ("kkt_1e4_iterations", "kkt_1e6_iterations", "kkt_1e8_iterations")
ALL_COLUMNS = COLUMNS + ITERATION_COLUMNS
LEVELS = ("1e-4", "1e-6", "1e-8")


def is_pdhg(algorithm: str) -> bool:
    """`pdhg-cpu`, `pdhg-cuda`, `pdhg-cuda-multi`, `pdhg+ipm`: every first-order run."""
    return str(algorithm or "").startswith("pdhg")


def crossings(blob: dict) -> dict:
    """The six CSV cells (ALL_COLUMNS) for one stats JSON blob."""
    algorithm = blob.get("result", {}).get("algorithm", "")
    effort = blob.get("effort", {})
    if not is_pdhg(algorithm):
        return {k: "" for k in ALL_COLUMNS}
    cells = {k: effort.get(k, "nan") for k in COLUMNS}
    cells.update({k: effort.get(k, -1) for k in ITERATION_COLUMNS})
    return cells


def as_seconds(cell) -> float | None:
    """A crossing cell as a float: None when blank, NaN when never reached."""
    if cell is None or str(cell).strip() == "":
        return None
    try:
        return float(cell)
    except (TypeError, ValueError):
        return None


def consistent(row: dict) -> bool:
    """The three crossings in one run are ordered: a tighter level cannot be reached before
    a looser one, and once a level is missed every tighter one is missed too. Blank rows
    (other engines) are trivially consistent."""
    values = [as_seconds(row.get(k)) for k in COLUMNS]
    if all(v is None for v in values):
        return True
    if any(v is None for v in values):
        return False
    reached = [v for v in values if not math.isnan(v)]
    missed_after = any(math.isnan(v) for v in values[: len(reached)])
    return not missed_after and all(a <= b for a, b in zip(reached, reached[1:]))


def cell_text(cell) -> str:
    """How the doc prints one crossing."""
    value = as_seconds(cell)
    if value is None:
        return "-"
    if math.isnan(value):
        return "not reached"
    return f"{value:.3g}"


# ---- docs/BENCHMARKS.md, generated (make_benchmarks_doc.py calls these) ------------------

def has_crossings(rows) -> bool:
    """True when any row of a CSV carries a crossing: CSVs written before #486 have none,
    and their sections are rendered exactly as before."""
    return any(as_seconds(r.get(k)) is not None for r in rows for k in COLUMNS)


def verified_mark(row: dict) -> str:
    value = str(row.get("independently_verified", "")).strip()
    return {"1": "yes", "true": "yes", "True": "yes", "0": "**NO**", "false": "**NO**",
            "False": "**NO**"}.get(value, "-")


def pdhg_report_table(rows: list[dict]) -> list[str]:
    """The crossings of each instance's tightest PDHG run (restarts on) in a pdhg_report.py
    CSV, beside the status and the verifier's verdict on the returned point."""
    runs = [r for r in rows if r.get("algorithm") == "pdhg"
            and str(r.get("restarts_enabled")) == "1" and r.get("tolerance")]
    if not has_crossings(runs):
        return []
    tightest = min(float(r["tolerance"]) for r in runs)
    chosen: dict[str, dict] = {}
    for r in runs:
        if float(r["tolerance"]) == tightest:
            chosen.setdefault(r["instance"], r)
    out = ["", f"**Relative KKT crossings in the {tightest:g} run (#486).** Seconds on the "
               "solver clock at which the relative KKT error (defined below) first came at "
               "or under each level, in the same run, beside what that run reported and "
               "whether the independent verifier accepted the point:", "",
           "| instance | 1e-4 (s) | 1e-6 (s) | 1e-8 (s) | status | verified |",
           "|---|---:|---:|---:|---|---|"]
    for name in sorted(chosen):
        r = chosen[name]
        cells = " | ".join(cell_text(r.get(k)) for k in COLUMNS)
        note = "" if consistent(r) else " (crossings out of order)"
        out.append(f"| `{name}` | {cells} | {r.get('status', '')}{note} | {verified_mark(r)} |")
    return out


def feasibility_standard_table(pdhg_rows: dict[str, dict]) -> list[str]:
    """PDHG at the standard of Mittelmann's LP feasibility page: the time at which the
    relative KKT error first reached 1e-6, with no basis and no polish. Labelled as that
    standard and never as an optimum; the verdict beside it is on the point the run finally
    returned, which is a different and stricter claim."""
    if not has_crossings(pdhg_rows.values()):
        return []
    out = ["", "**PDHG at the feasibility-page standard (#486).** The same PDHG runs, read at "
               "the standard Mittelmann's LP feasibility page uses: the relative KKT error at "
               "or under 1e-6, no basis, no polish (the crossing is taken in the first-order "
               "phase, before any polish). This is a first-order tolerance, **not an optimal "
               "basis** and not what `optimal` means elsewhere in this document; the last two "
               "columns are what the full run went on to report and the verifier's verdict on "
               "that point.", "",
           "| instance | relative KKT <= 1e-6 at (s) | run status | verified |",
           "|---|---:|---|---|"]
    reached = 0
    for name in sorted(pdhg_rows):
        r = pdhg_rows[name]
        at = as_seconds(r.get("kkt_1e6_seconds"))
        if at is not None and not math.isnan(at):
            reached += 1
        out.append(f"| `{name}` | {cell_text(r.get('kkt_1e6_seconds'))} | "
                   f"{r.get('status', '')} | {verified_mark(r)} |")
    out += ["", f"Reached the feasibility-page standard: **{reached} of {len(pdhg_rows)}**."]
    return out
