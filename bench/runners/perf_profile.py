#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dolan-More performance profiles of the head-to-head CSVs, as hand-written SVG (#766).

E. D. Dolan and J. J. More, "Benchmarking optimization software with performance
profiles", Mathematical Programming 91 (2002) 201-213. For each instance p and solver s let
t(p,s) be the solver's time when its run counted (optimal, at the published optimum, and
accepted by the independent verifier) and infinity otherwise; the ratio r(p,s) is t(p,s)
over the best t(p,.) on that instance, and the profile of s is the fraction of ALL instances
with r(p,s) <= tau, for tau from 1 upward on a log2 axis. The height at tau = 1 is how often
a solver was fastest; the height at the right edge is the fraction it solved at all.

Times are each solver's own clock, floored at TIME_FLOOR_SECONDS before the ratio is taken:
GLPK reports its time to a tenth of a second and prints 0.0 for anything faster, and a ratio
against a clock that cannot see the difference would be noise. Below the floor, solvers tie.

No plotting library: like gpu_plot.py, the SVG is written by hand, so
make_benchmarks_doc.py regenerates the figure wherever it runs and it cannot disagree with
the CSV it is drawn from.
"""
from __future__ import annotations

import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
IMG_DIR = REPO_ROOT / "docs" / "img"
TIME_FLOOR_SECONDS = 0.1  # GLPK's clock resolution, the coarsest in the comparison

# Colour-blind-safe (Okabe-Ito); SANKHYA first and drawn thickest.
COLOURS = {"sankhya": "#D55E00", "highs": "#0072B2", "scip": "#009E73",
           "cbc-clp": "#CC79A7", "glpk": "#E69F00"}
LABELS = {"sankhya": "SANKHYA", "highs": "HiGHS", "scip": "SCIP", "cbc-clp": "CBC/Clp",
          "glpk": "GLPK"}


def ratios(rows: list[dict], solvers: list[str]) -> dict[str, list[float]]:
    """solver -> one performance ratio per instance (math.inf when the run did not count)."""
    times: dict[str, dict[str, float]] = {}
    for row in rows:
        t = math.inf
        if row.get("counted_for_time") == "1" and row.get("solver_seconds"):
            t = max(float(row["solver_seconds"]), TIME_FLOOR_SECONDS)
        times.setdefault(row["instance"], {})[row["solver"]] = t
    out: dict[str, list[float]] = {s: [] for s in solvers}
    for per in times.values():
        best = min((per.get(s, math.inf) for s in solvers), default=math.inf)
        for s in solvers:
            t = per.get(s, math.inf)
            out[s].append(t / best if math.isfinite(t) else math.inf)
    return out


def profile(values: list[float], tau: float) -> float:
    """rho_s(tau): the fraction of instances within a factor tau of the best."""
    return sum(r <= tau for r in values) / len(values) if values else 0.0


def svg(title: str, data: dict[str, list[float]]) -> str:
    width, height, left, right, top, bottom = 720, 440, 60, 150, 36, 56
    finite = [r for values in data.values() for r in values if math.isfinite(r)]
    k = max(1, math.ceil(math.log2(max(finite + [2.0]))))

    def px(tau: float) -> float:
        return left + math.log2(tau) / k * (width - left - right)

    def py(rho: float) -> float:
        return height - bottom - rho * (height - top - bottom)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" font-family="sans-serif" font-size="12">',
             f'<rect width="{width}" height="{height}" fill="white"/>',
             f'<text x="{(width - right + left) / 2:.0f}" y="20" text-anchor="middle" '
             f'font-size="14">{title}</text>']
    for e in range(0, k + 1):
        x = px(2.0 ** e)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - bottom}" '
                     'stroke="#ddd"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - bottom + 16}" text-anchor="middle">'
                     f'{2 ** e:g}</text>')
    for i in range(0, 11, 2):
        y = py(i / 10)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
                     'stroke="#ddd"/>')
        parts.append(f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end">{i / 10:.1f}'
                     '</text>')
    parts.append(f'<text x="{(width - right + left) / 2:.0f}" y="{height - 14}" '
                 'text-anchor="middle">tau: within this factor of the fastest (log2 scale)'
                 '</text>')
    parts.append(f'<text x="16" y="{(height - bottom + top) / 2:.0f}" text-anchor="middle" '
                 f'transform="rotate(-90 16 {(height - bottom + top) / 2:.0f})">'
                 'fraction of instances</text>')
    for index, (solver, values) in enumerate(data.items()):
        steps = sorted(r for r in values if math.isfinite(r))
        points = [(px(1.0), py(profile(values, 1.0)))]
        for r in steps:
            x = px(r)
            points.append((x, points[-1][1]))
            points.append((x, py(profile(values, r))))
        points.append((px(2.0 ** k), points[-1][1]))
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        colour = COLOURS.get(solver, "#555")
        thick = 3 if solver == "sankhya" else 1.8
        parts.append(f'<polyline points="{path}" fill="none" stroke="{colour}" '
                     f'stroke-width="{thick}"><title>{LABELS.get(solver, solver)}: fastest on '
                     f'{profile(values, 1.0):.0%}, solved {profile(values, math.inf):.0%}'
                     '</title></polyline>')
        ly = top + 12 + 20 * index
        parts.append(f'<line x1="{width - right + 14}" y1="{ly}" x2="{width - right + 40}" '
                     f'y2="{ly}" stroke="{colour}" stroke-width="{thick}"/>')
        parts.append(f'<text x="{width - right + 46}" y="{ly + 4}">'
                     f'{LABELS.get(solver, solver)}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_figure(suite: str, title: str, rows: list[dict], solvers: list[str],
                 out_dir: Path = IMG_DIR) -> Path | None:
    """docs/img/profile-<suite>.svg, or None when the CSV has no rows."""
    data = ratios(rows, solvers)
    if not any(data.values()):
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"profile-{suite}.svg"
    out.write_text(svg(title, data), encoding="utf-8", newline="\n")
    return out
