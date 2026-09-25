#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The GPU real-instance and datacenter tables of docs/BENCHMARKS.md (#446, #488).

Kept out of make_benchmarks_doc.py, which imports both functions, so the fairness rules they
render can be tested alone (bench/runners/test_gpu_runners.py).

A GPU speedup is printed against BOTH CPU arms the runners now measure (bench/runners/
gpu_arms.py): one thread with the serial A x, the default, and N threads with
pdhg_parallel_spmv=true. The CSVs committed before #488 carry neither the arm columns nor a
reference objective; they still render, with '-' where a column is missing and a note that
says what their CPU arm actually was, so an old ratio is never read as a fair one.
"""
from __future__ import annotations

import csv
from pathlib import Path

CPU_ONE = "cpu-1t"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio(cpu: float | None, gpu: float | None) -> str:
    if not cpu or not gpu:
        return "-"
    return f"{cpu / gpu:.2f}x"


def _cell(value) -> str:
    return "-" if value in (None, "") else str(value)


def parallel_arm(rows: list[dict], arm_key: str) -> tuple[str | None, str]:
    """(label, thread count) of the N-thread CPU arm run with pdhg_parallel_spmv=true, or
    (None, "N") when the CSV has none - which every pre-#488 CSV is."""
    best: tuple[int, str, str] | None = None
    for r in rows:
        label = r.get(arm_key, "")
        if not label.startswith("cpu-") or label == CPU_ONE:
            continue
        if r.get("parallel_spmv", "") != "true":
            continue
        threads = r.get("cpu_threads", "") or label[4:].rstrip("t")
        count = int(_float(threads) or 0)
        if best is None or count > best[0]:
            best = (count, label, str(count))
    return (best[1], best[2]) if best else (None, "N")


def old_real_note(name: str) -> str:
    return (f"> `{name}` predates #488's fairness fix. Its runner passed no thread option, so "
            "its CPU arm is **one thread with the serial A x** - every speedup in it is "
            "against a single-threaded CPU - and it records no reference objective, so the "
            "N-thread and gap columns read '-'. Re-run `gpu_real_instances.py` on `main` for "
            "the three-arm table.")


def old_datacenter_note(name: str) -> str:
    return (f"> `{name}` predates #488's fairness fix. Its only CPU arm, `cpu-16t`, was run "
            "with `threads=16` and **without** `pdhg_parallel_spmv=true`: A^T y over 16 "
            "threads, A x serial. That is neither the one-thread default nor the parallel "
            "arm, so no speedup is computed from it here (its times are in the table above), "
            "and it records no reference objective or residuals, so those columns read '-'. "
            "Re-run `gpu_datacenter.py` on `main` for both CPU arms.")


def gpu_real_section(path: Path | None) -> str:
    """CPU vs GPU PDHG on non-synthetic instances (#446), both CPU arms (#488)."""
    if path is None:
        return chr(10).join([
            "Not yet run on this tier (the datacenter card's run is in 1g.3). Reproduce with:",
            "",
            "```",
            "python bench/runners/fetch_mittelmann.py",
            "python bench/runners/gpu_real_instances.py --binary build_gpu/sankhya",
            "```",
            "",
            "> **Needs a CUDA-capable card and a CUDA build** (`-DSANKHYA_ENABLE_CUDA=ON`, the "
            "CUDA runtime installed). On a build without CUDA, or a machine whose card fails the "
            "device checks, `gpu=true` warns and runs on the CPU, so both arms of the runner "
            "would be CPU solves and the GPU column would mean nothing: do not run it there.",
            "",
        ])
    rows = read_csv(path)
    if not rows:
        return "No GPU real-instance results yet." + chr(10)

    commit = rows[0].get("git_commit", "unknown")
    machine = rows[0].get("machine", "unknown")
    gpu = rows[0].get("gpu", "") or "not recorded"
    if "-dirty" in commit:
        return (f"`{path.name}` is stamped `{commit}`: produced from a modified tree. "
                "Re-run on a clean checkout of a main commit." + chr(10))
    has_arms = "arm" in rows[0]
    many_label, many = parallel_arm(rows, "arm") if has_arms else (None, "N")
    instances = list(dict.fromkeys(r["instance"] for r in rows))
    tolerances = sorted({_float(r.get("tolerance")) for r in rows} - {None}, reverse=True)

    def lookup(instance: str, tol: float, arm: str | None, algorithm: str) -> dict | None:
        if has_arms and arm is None:
            return None
        for r in rows:
            t = _float(r.get("tolerance"))
            if r.get("instance") != instance or t is None or abs(t - tol) > 1e-3 * tol:
                continue
            if (r.get("arm") == arm) if has_arms else (r.get("algorithm") == algorithm):
                return r
        return None

    def secs(r: dict | None) -> float | None:
        return _float(r.get("seconds")) if r else None

    def fmt_s(r: dict | None) -> str:
        s = secs(r)
        return "-" if s is None else f"{s:.3f}"

    def gap(r: dict | None) -> str:
        return _cell(r.get("rel_gap")) if r else "-"

    lines = [
        f"Source CSV: `bench/results/{path.name}`  ",
        f"Commit `{commit}` · machine `{machine}`  ",
        f"GPU: {gpu}",
        "",
        "Same protocol as §1g: PDHG alone, solver clock, warm-up GPU solve per instance. "
        "Report the result whichever way it goes. The card is compared with two CPU arms: one "
        "thread with the serial A x (the default configuration), and "
        f"{many} threads with `pdhg_parallel_spmv=true`, so A x is row-parallel as well as "
        "A^T y (#487, #488). The gap is |obj - ref| / max(1, |ref|) against the reference in "
        "the last column; `feasible` means the requested relative tolerance was met but not "
        "the project's absolute standard.",
        "",
        f"| instance | rows | tol | CPU 1 thread (s) | CPU {many} threads (s) | GPU (s) "
        f"| GPU vs 1 thread | GPU vs {many} threads | GPU status "
        f"| rel gap CPU 1t / CPU {many}t / GPU | reference |",
        "|----------|-----:|----:|-------------:|-------------:|--------:|--------:|--------:"
        "|--------|--------|--------|",
    ]
    for inst in instances:
        for tol in tolerances:
            one = lookup(inst, tol, CPU_ONE, "pdhg-cpu")
            par = lookup(inst, tol, many_label, "")
            card = lookup(inst, tol, "gpu", "pdhg-cuda")
            if not (one or par or card):
                continue
            first = one or par or card
            reference = "-"
            if first.get("reference_objective"):
                reference = f"{first['reference_objective']} ({first.get('reference_source', '')})"
            lines.append(
                f"| `{inst}` | {first.get('rows', '')} | {tol:.0e} | {fmt_s(one)} | {fmt_s(par)} "
                f"| {fmt_s(card)} | {_ratio(secs(one), secs(card))} "
                f"| {_ratio(secs(par), secs(card))} | {card.get('status', '-') if card else '-'} "
                f"| {gap(one)} / {gap(par)} / {gap(card)} | {reference} |")
    lines.append("")
    if not has_arms:
        lines += [old_real_note(path.name), ""]
    return chr(10).join(lines)


def gpu_datacenter_table(path: Path) -> str:
    """The datacenter runner's CSV (#488) as a table: one row per instance, mode and
    tolerance; the solver's own clock beside the median WALL of the repeats (process
    start-up and, on the card, context creation included) with their spread; and the
    forced-count pair (iteration_limit set) that isolates the per-iteration ratio from the
    iteration count. Below it, the GPU's speedup against each CPU arm."""
    rows = read_csv(path)
    if not rows:
        return "The CSV is empty.\n"
    first = rows[0]
    has_arms = "parallel_spmv" in first
    many_label, many = parallel_arm(rows, "mode") if has_arms else (None, "N")
    out = [f"`{path.name}` - {first.get('gpu', '?')}, solver at `{first.get('git_commit', '?')}`, "
           f"{first.get('machine', '?')}, {first.get('repeats', '?')} repeats per cell:\n",
           "| instance | mode | threads | parallel A x | tol | forced iterations | status "
           "| objective | rel gap | primal res | dual res | iterations | solver (s) "
           "| solver median (s) | median wall (s) | spread (s) |",
           "|---|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        out.append(f"| `{r.get('instance', '')}` | {r.get('mode', '')} "
                   f"| {_cell(r.get('cpu_threads'))} | {_cell(r.get('parallel_spmv'))} "
                   f"| {r.get('tol', '')} | {r.get('iteration_limit', '') or '-'} "
                   f"| {r.get('status', '')} | {r.get('objective', '')} | {_cell(r.get('rel_gap'))} "
                   f"| {_cell(r.get('primal_residual'))} | {_cell(r.get('dual_residual'))} "
                   f"| {r.get('iterations', '')} | {r.get('seconds', '')} "
                   f"| {_cell(r.get('solver_median_s'))} | {r.get('wall_median_s', '')} "
                   f"| {r.get('wall_spread_s', '')} |")

    def solver_time(r: dict | None) -> float | None:
        if r is None:
            return None
        return _float(r.get("solver_median_s")) or _float(r.get("seconds"))

    cells: list[tuple[str, str, str]] = list(dict.fromkeys(
        (r.get("instance", ""), r.get("tol", ""), r.get("iteration_limit", "")) for r in rows))

    def pick(cell: tuple[str, str, str], mode: str | None) -> dict | None:
        if mode is None:
            return None
        for r in rows:
            if (r.get("instance", ""), r.get("tol", ""), r.get("iteration_limit", "")) == cell \
                    and r.get("mode") == mode:
                return r
        return None

    if not has_arms:
        # Neither column has a CPU arm to divide by, so a table of dashes would say nothing.
        return "\n".join(out + ["", old_datacenter_note(path.name)]) + "\n"
    out += ["", "GPU speedup on the solver's own clock (median of the repeats):",
            "",
            f"| instance | tol | forced iterations | GPU vs 1 thread | GPU vs {many} threads, "
            "parallel A x | GPU rel gap |",
            "|---|---:|---:|---:|---:|---:|"]
    for cell in cells:
        card = pick(cell, "gpu")
        if card is None:
            continue
        one = pick(cell, CPU_ONE)
        par = pick(cell, many_label)
        out.append(f"| `{cell[0]}` | {cell[1]} | {cell[2] or '-'} "
                   f"| {_ratio(solver_time(one), solver_time(card))} "
                   f"| {_ratio(solver_time(par), solver_time(card))} "
                   f"| {_cell(card.get('rel_gap'))} |")
    return "\n".join(out) + "\n"
