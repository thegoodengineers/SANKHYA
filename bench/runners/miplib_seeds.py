#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Performance-variability control for the MIPLIB runner (#504): permuted copies of an
instance, and the per-instance numbers that make a multi-seed run readable.

WHY PERMUTE. A MIP solver's path depends on the order of the rows and columns: ties in
branching, in cut selection and in the simplex ratio test break by index, so the same model
in another order explores another tree. Lodi and Tramontani, "Performance variability in
mixed-integer programming", INFORMS TutORials in Operations Research 2013, recommend
measuring over several such permutations, because a single run confuses that variability
with the effect under test. The permutation is done HERE, on the file, so the solver is
untouched: it reads an ordinary MPS file that happens to list the same rows and columns in
another order.

HOW. Textually, line by line, so no number is re-printed and nothing can be rounded:

    ROWS      the N rows stay first in their own order (the first is the objective and must
              stay the first); every other row line is shuffled.
    COLUMNS   the lines are grouped by column name, each group keeps its integer marker
              state, the groups are shuffled, and INTORG/INTEND markers are re-emitted
              around the integer ones. The lines inside a group keep their order.
    others    RHS, RANGES, BOUNDS, OBJSENSE and the rest are copied unchanged: their meaning
              does not depend on where a row or column is declared.

Seed 0 is the file as published, never rewritten, so a one-seed run is the ordinary run.
A column whose lines are not contiguous is refused (ValueError) rather than guessed at.

THE NUMBERS. Per instance over its seeds: the seeds that matched the published optimum and
the seeds that proved it; the shifted geometric mean of the time with the unproved runs
charged the time limit (shift 10 s); the time to the first feasible point; and the primal
integral of Berthold, "Measuring the impact of primal heuristics", Operations Research
Letters 41 (2013) 611-614, from the incumbent trace the solver writes into its stats JSON.
"""
from __future__ import annotations

import gzip
import math
import random
from pathlib import Path

# Mittelmann's and MIPLIB's convention for running times: shift 10 s. The same shift for
# the time to first feasible would swamp it (most are well under a second), so that one is
# shifted by 1 s; both are stated where they are printed.
TIME_SHIFT_SECONDS = 10.0
FIRST_FEASIBLE_SHIFT_SECONDS = 1.0


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="")
    return path.open("r", encoding="utf-8", errors="replace", newline="")


def _is_marker(fields: list[str]) -> str | None:
    upper = [f.upper().strip("'\"") for f in fields]
    if "MARKER" not in upper:
        return None
    if "INTORG" in upper:
        return "INTORG"
    if "INTEND" in upper:
        return "INTEND"
    raise ValueError(f"unrecognised MARKER line: {' '.join(fields)}")


def permute_mps(source: Path, destination: Path, seed: int) -> dict:
    """Write `source` with its rows and columns in a seeded random order to `destination`.

    Returns counts of what was shuffled. Raises ValueError on a file this cannot permute
    without guessing (a column declared in two places)."""
    rng = random.Random(seed)
    head: list[str] = []          # everything before ROWS
    n_rows: list[str] = []
    rows: list[str] = []
    groups: dict[str, list[str]] = {}
    integer: dict[str, bool] = {}
    tail: list[str] = []          # RHS onwards, verbatim
    section = ""
    in_integer = False
    last_column = None
    with _open_text(source) as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if line and not line[0].isspace() and not line.startswith("*"):
                section = line.split()[0].upper()
                if section in ("ROWS", "COLUMNS"):
                    continue
            if section == "ROWS":
                fields = line.split()
                if not fields or line.startswith("*"):
                    continue
                (n_rows if fields[0].upper() == "N" else rows).append(line)
            elif section == "COLUMNS":
                fields = line.split()
                if not fields or line.startswith("*"):
                    continue
                marker = _is_marker(fields)
                if marker is not None:
                    in_integer = marker == "INTORG"
                    continue
                name = fields[0]
                if name != last_column:
                    if name in groups:
                        raise ValueError(f"{source}: column {name} is declared in two places")
                    groups[name] = []
                    integer[name] = in_integer
                    last_column = name
                elif integer[name] != in_integer:
                    raise ValueError(f"{source}: column {name} straddles an integer marker")
                groups[name].append(line)
            elif section in ("", "NAME", "OBJSENSE", "OBJSENS"):
                head.append(line)
            else:
                tail.append(line)

    rng.shuffle(rows)
    order = list(groups)
    rng.shuffle(order)

    out: list[str] = list(head)
    out.append("ROWS")
    out.extend(n_rows)
    out.extend(rows)
    out.append("COLUMNS")
    open_marker = False
    markers = 0
    for name in order:
        if integer[name] and not open_marker:
            out.append(f"    MARKER{markers}  'MARKER'  'INTORG'")
            open_marker = True
        elif not integer[name] and open_marker:
            out.append(f"    MARKER{markers}  'MARKER'  'INTEND'")
            open_marker = False
            markers += 1
        out.extend(groups[name])
    if open_marker:
        out.append(f"    MARKER{markers}  'MARKER'  'INTEND'")
    out.extend(tail)
    destination.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    return {"rows": len(rows), "columns": len(order),
            "integer_columns": sum(integer.values())}


# ---- The numbers --------------------------------------------------------------------------


def shifted_geometric_mean(values: list[float], shift: float) -> float | None:
    """exp(mean(log(v + shift))) - shift; None for an empty list."""
    if not values:
        return None
    return math.exp(sum(math.log(v + shift) for v in values) / len(values)) - shift


def primal_gap(objective: float | None, optimum: float) -> float:
    """Berthold (2013), definition 1: 0 when both are zero, 1 with no point or when the signs
    differ, |opt - obj| / max(|opt|, |obj|) otherwise. It lies in [0, 1]."""
    if objective is None or not math.isfinite(objective):
        return 1.0
    if objective == 0.0 and optimum == 0.0:
        return 0.0
    if objective * optimum < 0.0:
        return 1.0
    return abs(optimum - objective) / max(abs(optimum), abs(objective))


def primal_integral(trace: list[tuple[float, float]], optimum: float, end: float) -> float:
    """Berthold (2013), definition 2: the integral of the primal gap over [0, end], the gap
    a step function that changes only when a new incumbent arrives. Events after `end`
    (clock noise) are ignored. In seconds; an empty trace gives `end`, the gap being 1."""
    total = 0.0
    clock = 0.0
    gap = 1.0
    for seconds, objective in sorted(trace):
        if seconds >= end:
            break
        seconds = max(seconds, 0.0)
        total += gap * (seconds - clock)
        clock = seconds
        gap = primal_gap(objective, optimum)
    total += gap * max(0.0, end - clock)
    return total


def incumbent_metrics(trace: list, optimum: float, end: float,
                      has_point: bool) -> dict:
    """Time to first feasible and the primal integral of one run.

    `trace` is the stats JSON's effort.incumbent_trace, [[seconds, objective], ...]. When it
    is empty but the run reports a point (presolve settled the model, or the parallel search,
    which keeps no trace), the point is charged as found at `end`: an upper bound, and
    `trace_recorded` says it is one."""
    events = [(float(t), float(v)) for t, v in trace
              if t is not None and v is not None]
    recorded = bool(events)
    if not events and has_point:
        return {"time_to_first_feasible": end, "primal_integral": end,
                "incumbents": 0, "trace_recorded": False}
    first = min((t for t, _ in events), default=None)
    return {"time_to_first_feasible": first,
            "primal_integral": primal_integral(events, optimum, end),
            "incumbents": len(events), "trace_recorded": recorded}


def aggregate(rows: list[dict], time_limit: float) -> list[dict]:
    """One summary per instance over its seeds.

    Each row needs: instance, seed, matched (bool), proved (bool), seconds (the run's time),
    and optionally time_to_first_feasible (None when nothing was found) and primal_integral.
    A run counts as SOLVED only when it proved the published optimum; every other run is
    charged `time_limit` in the time mean, and a run that found nothing is charged
    `time_limit` in the first-feasible mean."""
    by_instance: dict[str, list[dict]] = {}
    for row in rows:
        by_instance.setdefault(row["instance"], []).append(row)
    summary = []
    for name in sorted(by_instance):
        runs = sorted(by_instance[name], key=lambda r: r["seed"])
        times = [min(r["seconds"], time_limit) if r["proved"] else time_limit for r in runs]
        firsts = [time_limit if r.get("time_to_first_feasible") is None
                  else min(r["time_to_first_feasible"], time_limit) for r in runs]
        integrals = [r["primal_integral"] for r in runs if r.get("primal_integral") is not None]
        summary.append({
            "instance": name,
            "seeds": len(runs),
            "seeds_matched": sum(1 for r in runs if r["matched"]),
            "seeds_proved": sum(1 for r in runs if r["proved"]),
            "matched_seeds": [r["seed"] for r in runs if r["matched"]],
            "proved_seeds": [r["seed"] for r in runs if r["proved"]],
            "sgm_seconds": shifted_geometric_mean(times, TIME_SHIFT_SECONDS),
            "sgm_first_feasible_seconds": shifted_geometric_mean(
                firsts, FIRST_FEASIBLE_SHIFT_SECONDS),
            "mean_primal_integral": (sum(integrals) / len(integrals)) if integrals else None,
        })
    return summary


def overall(rows: list[dict], time_limit: float) -> dict:
    """The whole set: every (instance, seed) run in one shifted geometric mean."""
    times = [min(r["seconds"], time_limit) if r["proved"] else time_limit for r in rows]
    return {
        "runs": len(rows),
        "matched": sum(1 for r in rows if r["matched"]),
        "proved": sum(1 for r in rows if r["proved"]),
        "sgm_seconds": shifted_geometric_mean(times, TIME_SHIFT_SECONDS),
    }
