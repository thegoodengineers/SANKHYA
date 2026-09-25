#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The arms, reference objective and gap arithmetic both GPU runners share (#488).

A GPU speedup is a ratio, and a ratio is only as fair as its denominator. Before #488 the
real-instance runner passed no thread option at all, so its CPU arm was one thread with the
serial A x, and the datacenter runner passed `threads=N` without `pdhg_parallel_spmv=true`,
so its "cpu-16t" rows parallelised A^T y and left A x serial. Neither CSV carried a
reference objective, so an answer's distance from the optimum was invisible.

Every GPU run now has two CPU denominators beside it:

  cpu-1t   threads=1, pdhg_parallel_spmv=false - the default configuration, serial A x
  cpu-<N>t threads=N, pdhg_parallel_spmv=true  - A x row-parallel over the same N workers
                                                  that already carry A^T y (#487)

and every row carries the instance's sha256, a reference objective with where it came from,
and the absolute and relative gap to it. The relative gap is |obj - ref| / max(1, |ref|),
the same normalisation as the MIP gap in include/sankhya/tolerances.hpp, so a reference
near zero does not inflate it.

Nothing here runs a solve, which is what lets test_gpu_runners.py pin it without a GPU.
"""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

# (arm label, algorithm, extra solver options, cpu_threads, parallel_spmv). The GPU arm's
# thread columns are blank: it passes neither option, and neither changes what the card does.
GPU_ARM = ("gpu", "pdhg-cuda", [], "", "")

# The columns every row of both CSVs now carries, in order.
FAIRNESS_COLUMNS = [
    "arm", "cpu_threads", "parallel_spmv", "instance_sha256", "reference_objective",
    "reference_source", "abs_gap", "rel_gap",
]


def cpu_arms(threads: int) -> list[tuple[str, str, list[str], int, bool]]:
    """The CPU arms for `--cpu-threads threads`: always the serial default, and the
    row-parallel N-thread arm unless N is 1 (it would be the same solve twice)."""
    arms = [("cpu-1t", "pdhg-cpu", ["threads=1", "pdhg_parallel_spmv=false"], 1, False)]
    if threads > 1:
        arms.append((f"cpu-{threads}t", "pdhg-cpu",
                     [f"threads={threads}", "pdhg_parallel_spmv=true"], threads, True))
    return arms


def all_arms(threads: int) -> list[tuple]:
    return [*cpu_arms(threads), GPU_ARM]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analytic_optimum(mps: Path) -> float | None:
    """The `* analytic optimum:` comment generate_refinery_lp.py writes (#211): the
    generator builds the optimal point and prints c^T x* from exact rationals."""
    try:
        with mps.open("r", encoding="utf-8", errors="replace") as handle:
            for _, line in zip(range(20), handle):
                match = re.match(r"\*\s*analytic optimum:\s*(\S+)", line)
                if match:
                    return float(match.group(1))
    except (OSError, ValueError):
        return None
    return None


def reference_objective(mps: Path, time_limit: float,
                        use_highs: bool = True) -> tuple[float | None, str, str]:
    """(value, source, note) for one instance, computed ONCE per instance.

    A generated model that states its own exact optimum (the refinery year) is its own
    reference, `construction`. Anything else asks HiGHS, run as a separate process over the
    same file by mittelmann.highs_objective (an accepted judgement call in docs/PROVENANCE.md:
    nothing of it is linked or read), under its own time limit. HiGHS missing, or not
    finishing optimal, leaves the reference blank and the source `none` - never a guess."""
    analytic = analytic_optimum(mps)
    if analytic is not None:
        return analytic, "construction", "analytic optimum stated by the generator"
    if not use_highs:
        return None, "none", "reference disabled (--no-reference)"
    import mittelmann  # lazy: the doc generator and the tests never need it
    value, status = mittelmann.highs_objective(mps, time_limit)
    if value is None:
        return None, "none", f"highs: {status}"
    return value, "highs", f"highs: {status}"


def gaps(objective: float | None, reference: float | None) -> tuple[float | None, float | None]:
    """(|obj - ref|, |obj - ref| / max(1, |ref|)); (None, None) when either is missing or
    not finite - a blank cell, never a zero that reads as agreement."""
    if objective is None or reference is None:
        return None, None
    if not (math.isfinite(objective) and math.isfinite(reference)):
        return None, None
    absolute = abs(objective - reference)
    return absolute, absolute / max(1.0, abs(reference))


def fmt_gap(value: float | None) -> str:
    return "" if value is None else f"{value:.3e}"


def fairness_cells(arm: tuple, digest: str, objective: float | None,
                   reference: float | None, reference_source: str) -> dict:
    """The FAIRNESS_COLUMNS of one row."""
    label, _, _, threads, parallel = arm
    absolute, relative = gaps(objective, reference)
    return {
        "arm": label,
        "cpu_threads": threads,
        "parallel_spmv": "" if parallel == "" else ("true" if parallel else "false"),
        "instance_sha256": digest,
        "reference_objective": "" if reference is None else repr(reference),
        "reference_source": reference_source if reference is not None else "none",
        "abs_gap": fmt_gap(absolute),
        "rel_gap": fmt_gap(relative),
    }


def speedup(cpu_seconds: float | None, gpu_seconds: float | None) -> str:
    if not cpu_seconds or not gpu_seconds:
        return "-"
    return f"{cpu_seconds / gpu_seconds:.2f}x"


def residuals(blob: dict) -> tuple[str, str]:
    """The absolute primal and dual residuals of the reported point, as the engine states
    them in its message (both engines word it `absolute primal X, dual Y` when optimal and
    `primal X vs T, dual Y vs T` when only the relative tolerance was met). When the message
    has neither - a time or iteration limit reports RELATIVE residuals, which are not the
    same quantity - the core's own recomputation in `quality` is used instead."""
    msg = blob.get("result", {}).get("message", "") or ""
    primal = re.search(r"absolute primal\s+([\d.e+\-]+)", msg) or \
        re.search(r"\bprimal\s+([\d.e+\-]+)\s+vs", msg)
    dual = re.search(r"absolute primal[\d.e+\-\s]+,\s*dual\s+([\d.e+\-]+)", msg) or \
        re.search(r",\s*dual\s+([\d.e+\-]+)\s+vs", msg)
    quality = blob.get("quality", {}) or {}

    def from_quality(key: str) -> str:
        value = quality.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError):
            return ""
        return f"{number:.3e}" if math.isfinite(number) else ""

    return (primal.group(1) if primal else from_quality("primal_infeasibility"),
            dual.group(1) if dual else from_quality("dual_infeasibility"))
