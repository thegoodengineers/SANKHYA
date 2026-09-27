#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GPU speedup against nonzeros, one figure per card, with the crossover marked (#488).

Reads the card's crossover CSV (gpu-<card>-<sha>.csv, the synthetic ladder) and its real-
instance CSV (gpu-real-<card>-<sha>.csv) and writes docs/img/gpu-speedup-<card>.svg: every
point is one instance at one tolerance, the y value the speedup of the card against the
FASTER CPU arm on the solver's own clock, so a loss cannot be hidden by choosing the arm.
The dashed line at 1x is the crossover; everything below it is a loss. The SVG is hand-
written, no plotting library, so make_benchmarks_doc.py can regenerate it wherever it runs
and the figure can never disagree with the CSVs it is drawn from.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_doc  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
IMG_DIR = REPO_ROOT / "docs" / "img"

TOL_COLOUR = {"1e-04": "#1f77b4", "1e-08": "#d62728"}


def _tol(row: dict) -> str:
    value = gpu_doc._float(row.get("tolerance"))
    return f"{value:.0e}" if value is not None else ""


def points(crossover: Path | None, real: Path | None) -> list[tuple[float, float, str, str, str]]:
    """(nnz, speedup, tolerance, label, series) for every measured GPU cell."""
    out = []
    if crossover is not None:
        rows = gpu_doc.read_csv(crossover)
        for r in rows:
            if r.get("algorithm") != "pdhg-cuda":
                continue
            cpu = next((c for c in rows if c.get("algorithm") == "pdhg-cpu"
                        and c.get("instance") == r.get("instance")
                        and _tol(c) == _tol(r)), None)
            s, g, nnz = (gpu_doc._float(cpu and cpu.get("seconds")),
                         gpu_doc._float(r.get("seconds")), gpu_doc._float(r.get("nnz")))
            if s and g and nnz:
                out.append((nnz, s / g, _tol(r), r.get("instance", ""), "synthetic"))
    if real is not None:
        rows = gpu_doc.read_csv(real)
        for r in rows:
            if r.get("arm") != "gpu":
                continue
            cpus = [gpu_doc._float(c.get("seconds")) for c in rows
                    if c.get("arm", "").startswith("cpu-")
                    and c.get("instance") == r.get("instance") and _tol(c) == _tol(r)]
            cpus = [c for c in cpus if c]
            g, nnz = gpu_doc._float(r.get("seconds")), gpu_doc._float(r.get("nnz"))
            if cpus and g and nnz:
                out.append((nnz, min(cpus) / g, _tol(r), r.get("instance", ""), "real"))
    return out


def svg(card: str, pts: list[tuple[float, float, str, str, str]]) -> str:
    width, height, left, right, top, bottom = 720, 420, 70, 20, 30, 60
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = math.floor(math.log10(min(xs))), math.ceil(math.log10(max(xs)))
    y0, y1 = math.floor(math.log10(min(ys + [1.0]))), math.ceil(math.log10(max(ys + [1.0])))
    if x1 == x0:
        x1 += 1
    if y1 == y0:
        y1 += 1

    def px(nnz: float) -> float:
        return left + (math.log10(nnz) - x0) / (x1 - x0) * (width - left - right)

    def py(speedup: float) -> float:
        return height - bottom - (math.log10(speedup) - y0) / (y1 - y0) * (height - top - bottom)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" font-family="sans-serif" font-size="12">',
             f'<rect width="{width}" height="{height}" fill="white"/>',
             f'<text x="{width / 2:.0f}" y="18" text-anchor="middle" font-size="14">'
             f'{card.upper()}: GPU speedup against the faster CPU arm, by nonzeros</text>']
    for e in range(x0, x1 + 1):
        x = px(10 ** e)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - bottom}" '
                     'stroke="#ddd"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - bottom + 16}" text-anchor="middle">'
                     f'1e{e}</text>')
    for e in range(y0, y1 + 1):
        y = py(10 ** e)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
                     'stroke="#ddd"/>')
        parts.append(f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end">'
                     f'{10 ** e:g}x</text>')
    one = py(1.0)
    parts.append(f'<line x1="{left}" y1="{one:.1f}" x2="{width - right}" y2="{one:.1f}" '
                 'stroke="#333" stroke-dasharray="6 4"/>')
    parts.append(f'<text x="{width - right - 4}" y="{one - 5:.1f}" text-anchor="end">'
                 'crossover, 1x: above is a win, below a loss</text>')
    parts.append(f'<text x="{width / 2:.0f}" y="{height - 8}" text-anchor="middle">'
                 'nonzeros of A</text>')
    for nnz, speedup, tol, label, series in sorted(pts):
        x, y = px(nnz), py(speedup)
        colour = TOL_COLOUR.get(tol, "#555")
        title = f'<title>{label} at {tol}: {speedup:.2f}x ({series})</title>'
        if series == "synthetic":
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{colour}">{title}'
                         '</circle>')
        else:
            parts.append(f'<rect x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" '
                         f'fill="{colour}">{title}</rect>')
            parts.append(f'<text x="{x + 8:.1f}" y="{y - 6:.1f}">{label}</text>')
    legend_y = top + 14
    for i, (tol, colour) in enumerate(TOL_COLOUR.items()):
        parts.append(f'<circle cx="{left + 12}" cy="{legend_y + 18 * i}" r="5" fill="{colour}"/>')
        parts.append(f'<text x="{left + 22}" y="{legend_y + 18 * i + 4}">{tol}, circle: '
                     'synthetic ladder; square: real instance</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_figure(card: str, crossover: Path | None, real: Path | None,
                 out_dir: Path = IMG_DIR) -> Path | None:
    """The figure's path, or None when neither CSV has a measured GPU cell."""
    pts = points(crossover, real)
    if not pts:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"gpu-speedup-{card}.svg"
    out.write_text(svg(card, pts), encoding="utf-8", newline="\n")
    return out


if __name__ == "__main__":
    import latest_result
    for name in ("l4", "a100"):
        path = write_figure(name, latest_result.latest(f"gpu-{name}-*.csv"),
                            latest_result.latest(f"gpu-real-{name}-*.csv"))
        print(name, path)
