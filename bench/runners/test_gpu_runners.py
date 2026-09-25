#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the GPU runners' fairness columns (#488): the two CPU arms and the options each
one passes, the gap arithmetic, the residual parse, the row construction of both runners,
and the doc tables for the new CSVs and the pre-#488 ones. Pure Python: no solve, no GPU -
the solver call is replaced by a recorder that returns a synthetic result.

    python bench/runners/test_gpu_runners.py
"""
from __future__ import annotations

import csv
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_arms  # noqa: E402
import gpu_datacenter  # noqa: E402
import gpu_doc  # noqa: E402
import gpu_real_instances  # noqa: E402

FAILURES = 0

# brazil3's verified optimum (bench/results/mittelmann-72123ff.csv, HiGHS's objective) and
# the GPU's 1e-8 answer in gpu-datacenter-l4-fdc89c5.csv: the 1.5e-7 nobody reported.
BRAZIL3_REF = 2.0000000000012172
BRAZIL3_GPU = 1.9999996940528124


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}  {detail}")
    if not condition:
        FAILURES += 1


def test_arms() -> None:
    arms = gpu_arms.cpu_arms(16)
    check([a[0] for a in arms] == ["cpu-1t", "cpu-16t"], "two CPU arms at --cpu-threads 16")
    check(arms[0][2] == ["threads=1", "pdhg_parallel_spmv=false"] and arms[0][3:] == (1, False),
          "the 1-thread arm is the serial default, said explicitly", str(arms[0]))
    check(arms[1][2] == ["threads=16", "pdhg_parallel_spmv=true"] and arms[1][3:] == (16, True),
          "the N-thread arm turns the row-parallel A x on (the #488 bug was its absence)",
          str(arms[1]))
    check([a[0] for a in gpu_arms.cpu_arms(1)] == ["cpu-1t"],
          "--cpu-threads 1 does not run the same solve twice")
    check([a[0] for a in gpu_arms.all_arms(8)] == ["cpu-1t", "cpu-8t", "gpu"],
          "the card runs after both CPU arms")
    check(gpu_arms.GPU_ARM[2] == [], "the GPU arm passes no thread option")


def test_gaps() -> None:
    absolute, relative = gpu_arms.gaps(BRAZIL3_GPU, BRAZIL3_REF)
    check(abs(relative - 1.529742024e-7) < 1e-15 and abs(absolute - 2 * relative) < 1e-15,
          "brazil3's GPU 1e-8 answer is 1.53e-7 relative from the optimum", f"{relative:.6e}")
    absolute, relative = gpu_arms.gaps(1e-5, 0.0)
    check(absolute == 1e-5 and relative == 1e-5,
          "a reference of zero divides by 1, not by zero", f"{relative}")
    absolute, relative = gpu_arms.gaps(-3.0e6 + 3.0, -3.0e6)
    check(math.isclose(relative, 1e-6), "a large reference divides by |ref|", f"{relative}")
    check(gpu_arms.gaps(None, 1.0) == (None, None) and gpu_arms.gaps(1.0, None) == (None, None),
          "a missing objective or reference is a blank, not a zero")
    check(gpu_arms.gaps(float("nan"), 1.0) == (None, None)
          and gpu_arms.gaps(1.0, float("inf")) == (None, None), "non-finite is blank")
    check(gpu_arms.fmt_gap(None) == "" and gpu_arms.fmt_gap(1.5297420e-7) == "1.530e-07",
          "gap cells: blank or three significant decimals")
    check(gpu_arms.speedup(10.0, 4.0) == "2.50x" and gpu_arms.speedup(None, 4.0) == "-"
          and gpu_arms.speedup(10.0, 0.0) == "-", "speedup string and its blanks")


def test_fairness_cells() -> None:
    cells = gpu_arms.fairness_cells(gpu_arms.cpu_arms(16)[1], "ab" * 32, BRAZIL3_GPU,
                                    BRAZIL3_REF, "highs")
    check(cells["arm"] == "cpu-16t" and cells["cpu_threads"] == 16
          and cells["parallel_spmv"] == "true", "the N-thread row names its threads and A x")
    check(cells["reference_source"] == "highs" and cells["rel_gap"] == "1.530e-07"
          and cells["reference_objective"] == repr(BRAZIL3_REF), "reference and gap recorded")
    card = gpu_arms.fairness_cells(gpu_arms.GPU_ARM, "cd" * 32, 1.0, None, "highs")
    check(card["cpu_threads"] == "" and card["parallel_spmv"] == "",
          "the GPU row leaves the CPU columns blank")
    check(card["reference_source"] == "none" and card["abs_gap"] == "" and card["rel_gap"] == "",
          "no reference: source `none`, gaps blank")
    check(list(cells) == gpu_arms.FAIRNESS_COLUMNS, "the cells come in the declared order")


def test_residuals() -> None:
    optimal = {"result": {"message": "CUDA PDHG converged after 10 iterations and 2 restarts; "
                                     "absolute primal 1.000e-09, dual 2.000e-10, relative gap "
                                     "3.000e-11"}}
    check(gpu_arms.residuals(optimal) == ("1.000e-09", "2.000e-10"), "optimal message parsed")
    feasible = {"result": {"message": "met the requested relative tolerance 1.0e-08 after 5 "
                                      "iterations, but NOT the absolute standard this project "
                                      "verifies against (primal 3.000e-07 vs 1.0e-07, dual "
                                      "4.000e-08 vs 1.0e-07, relative gap 1.0e-09 vs 1.0e-09)"}}
    check(gpu_arms.residuals(feasible) == ("3.000e-07", "4.000e-08"),
          "feasible message parsed (absolute, against the project standard)")
    stopped = {"result": {"message": "stopped at relative primal 1.0e-03, dual 1.0e-02, gap "
                                     "1.0e-01 after 9 iterations and 0 restarts (target 1.0e-08)"},
               "quality": {"primal_infeasibility": 0.25, "dual_infeasibility": "nan"}}
    check(gpu_arms.residuals(stopped) == ("2.500e-01", ""),
          "a relative stop falls back to the core's absolute recomputation, nan left blank")


def test_reference() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        generated = Path(tmp) / "gen.mps"
        generated.write_text("NAME X\n* generator: g\n* analytic optimum: -38289180.25\nROWS\n")
        plain = Path(tmp) / "plain.mps"
        plain.write_text("NAME Y\nROWS\n N COST\nENDATA\n")
        check(gpu_arms.analytic_optimum(generated) == -38289180.25,
              "the refinery generator's analytic optimum is read from the file")
        check(gpu_arms.reference_objective(generated, 1.0) ==
              (-38289180.25, "construction", "analytic optimum stated by the generator"),
              "a file that states its optimum never calls HiGHS")
        value, source, _ = gpu_arms.reference_objective(plain, 1.0, use_highs=False)
        check(value is None and source == "none", "--no-reference leaves it blank")
        check(len(gpu_arms.sha256_file(plain)) == 64, "sha256 of the instance file")


class Recorder:
    """Stands in for run_solve: records each call's options, returns a synthetic answer."""

    def __init__(self):
        self.calls: list[tuple[str, list[str]]] = []

    def __call__(self, binary, mps, algorithm, tol, time_limit, extra_options=None):
        self.calls.append((algorithm, list(extra_options or [])))
        seconds = {"pdhg-cpu": 8.0 if "threads=1" in (extra_options or []) else 2.0,
                   "pdhg-cuda": 1.0}[algorithm]
        return {"status": "feasible", "objective": BRAZIL3_GPU, "iterations": 100,
                "seconds": seconds, "wall": seconds + 0.5, "primal_residual": "1.6e-09",
                "dual_residual": "7.5e-10"}


def test_real_runner_rows() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mps = Path(tmp) / "brazil3.mps"
        mps.write_text("NAME B\nROWS\n N COST\n L R1\nCOLUMNS\n X COST 1 R1 1\nENDATA\n")
        recorder = Recorder()
        saved = gpu_real_instances.run_solve
        gpu_real_instances.run_solve = recorder
        try:
            args = SimpleNamespace(binary=Path("sankhya"), time_limit=10.0, cpu_threads=16,
                                   reference_time_limit=1.0, no_reference=True,
                                   solver_option=["pdhg_two_matvec=true"])
            rows = gpu_real_instances.measure("brazil3", mps, args,
                                              {"git_commit": "abc1234", "machine": "m",
                                               "gpu": "g", "timestamp_utc": "t",
                                               "solver_options": "pdhg_two_matvec=true"})
        finally:
            gpu_real_instances.run_solve = saved
    measured = recorder.calls[1:]  # the first call is the untimed GPU warm-up
    check(len(rows) == 6 and len(measured) == 6, "3 arms x 2 tolerances", str(len(rows)))
    cpu_many = [opts for alg, opts in measured if "threads=16" in opts]
    check(len(cpu_many) == 2 and all("pdhg_parallel_spmv=true" in o for o in cpu_many),
          "the N-thread CPU solves really pass pdhg_parallel_spmv=true", str(cpu_many))
    check(all(o[-1] == "pdhg_two_matvec=true" for _, o in measured),
          "--solver-option still reaches every arm")
    check(all(set(r) == set(gpu_real_instances.CSV_COLUMNS) for r in rows),
          "every row has exactly the CSV's columns")
    check([r["arm"] for r in rows[:3]] == ["cpu-1t", "cpu-16t", "gpu"]
          and rows[0]["reference_source"] == "none" and rows[0]["rel_gap"] == "",
          "arms in order; --no-reference blanks the gap")
    row = gpu_real_instances.build_row("brazil3", (1, 1, 1), "ef" * 32, gpu_arms.GPU_ARM, 1e-8,
                                       recorder(None, None, "pdhg-cuda", 1e-8, 1.0),
                                       BRAZIL3_REF, "highs", {})
    check(row["rel_gap"] == "1.530e-07" and row["tolerance"] == "1e-08"
          and row["instance_sha256"] == "ef" * 32 and row["primal_residual"] == "1.6e-09",
          "a GPU row carries sha, gap and residuals")


def test_datacenter_rows() -> None:
    recorder = Recorder()
    saved = gpu_datacenter.run_solve
    gpu_datacenter.run_solve = recorder
    try:
        arm = gpu_arms.cpu_arms(16)[1]
        r = gpu_datacenter.repeated(Path("sankhya"), Path("x.mps"), arm, 1e-8, 10.0, 3, 2000)
    finally:
        gpu_datacenter.run_solve = saved
    check(all(opts == ["threads=16", "pdhg_parallel_spmv=true", "iteration_limit=2000"]
              for _, opts in recorder.calls) and len(recorder.calls) == 3,
          "the datacenter N-thread arm passes pdhg_parallel_spmv=true (the #488 bug)",
          str(recorder.calls[0]))
    check(r["solver_median_s"] == 2.0 and r["wall_median_s"] == 2.5 and r["wall_spread_s"] == 0.0,
          "median of the solver clock and of the wall")
    row = gpu_datacenter.build_row("brazil3", "ab" * 32, arm, 1e-8, 2000, r, BRAZIL3_REF,
                                   "highs", {"repeats": 3, "git_commit": "abc1234",
                                             "machine": "m", "card": "l4", "gpu": "g",
                                             "timestamp_utc": "t"})
    check(list(row) == gpu_datacenter.COLUMNS, "the row has the CSV's columns in order")
    check(row["mode"] == "cpu-16t" and row["parallel_spmv"] == "true" and row["cpu_threads"] == 16
          and row["rel_gap"] == "1.530e-07", "mode, threads, A x and gap in the row")


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_doc_real() -> None:
    new_rows = []
    for tol, times in (("1e-04", (38.0, 9.5, 6.0)), ("1e-08", (91.0, 20.0, 11.0))):
        for arm, seconds in zip(gpu_arms.all_arms(16), times):
            result = {"status": "feasible", "objective": BRAZIL3_GPU, "iterations": 1,
                      "seconds": seconds, "wall": seconds}
            new_rows.append(gpu_real_instances.build_row(
                "brazil3", (14646, 1, 1), "ab" * 32, arm, float(tol), result, BRAZIL3_REF,
                "highs", {"git_commit": "abc1234", "machine": "m", "gpu": "L4",
                          "timestamp_utc": "t", "solver_options": ""}))
    old_columns = ["instance", "rows", "cols", "nnz", "algorithm", "tolerance", "status",
                   "objective", "iterations", "seconds", "wall_seconds", "reached_tolerance",
                   "primal_residual", "dual_residual", "git_commit", "machine", "gpu",
                   "timestamp_utc"]
    old_rows = [{"instance": "brazil3", "rows": 14646, "algorithm": alg, "tolerance": "1e-08",
                 "status": "feasible", "seconds": s, "git_commit": "fdc89c5"}
                for alg, s in (("pdhg-cpu", 91.143214), ("pdhg-cuda", 10.987494))]
    with tempfile.TemporaryDirectory() as tmp:
        new = Path(tmp) / "gpu-real-l4-abc1234.csv"
        write_csv(new, gpu_real_instances.CSV_COLUMNS, new_rows)
        text = gpu_doc.gpu_real_section(new)
        old = Path(tmp) / "gpu-real-l4-fdc89c5.csv"
        write_csv(old, old_columns, old_rows)
        old_text = gpu_doc.gpu_real_section(old)
    line = next((ln for ln in text.splitlines() if ln.startswith("| `brazil3`") and "1e-08" in ln), "")
    check("| 91.000 | 20.000 | 11.000 | 8.27x | 1.82x | feasible |" in line,
          "new CSV: GPU against the 1-thread AND the 16-thread CPU", line)
    check("1.530e-07 / 1.530e-07 / 1.530e-07" in line and "(highs)" in line,
          "new CSV: the gap of every arm and the reference's source", line)
    check("CPU 16 threads (s)" in text and "predates #488" not in text,
          "new CSV: the header names the thread count, no old-file note")
    old_line = next((ln for ln in old_text.splitlines() if ln.startswith("| `brazil3`")), "")
    check("| 91.143 | - | 10.987 | 8.30x | - | feasible | - / - / - | - |" in old_line,
          "old CSV: the ratio it had, '-' where it has no column", old_line)
    check("predates #488" in old_text and "one thread with the serial A x" in old_text,
          "old CSV: the note says its CPU arm was single-threaded")


def test_doc_datacenter() -> None:
    fixed = {"repeats": 3, "git_commit": "abc1234", "machine": "m", "card": "l4", "gpu": "L4",
             "timestamp_utc": "t"}
    new_rows = []
    for arm, seconds in zip(gpu_arms.all_arms(16), (80.0, 16.0, 10.0)):
        r = {"status": "feasible", "objective": BRAZIL3_GPU, "iterations": 1, "seconds": seconds,
             "solver_median_s": seconds, "wall_median_s": seconds, "wall_spread_s": 0.1}
        new_rows.append(gpu_datacenter.build_row("brazil3", "ab" * 32, arm, 1e-8, None, r,
                                                 BRAZIL3_REF, "highs", fixed))
    old_columns = ["instance", "instance_sha256", "mode", "tol", "iteration_limit", "status",
                   "objective", "iterations", "seconds", "wall_median_s", "wall_spread_s",
                   "repeats", "git_commit", "machine", "card", "gpu", "timestamp_utc"]
    old_rows = [{"instance": "brazil3", "mode": m, "tol": "1e-08", "iteration_limit": "",
                 "status": "feasible", "seconds": s, "git_commit": "fdc89c5"}
                for m, s in (("cpu-16t", "80.629251"), ("gpu", "10.044387"))]
    with tempfile.TemporaryDirectory() as tmp:
        new = Path(tmp) / "gpu-datacenter-l4-abc1234.csv"
        write_csv(new, gpu_datacenter.COLUMNS, new_rows)
        text = gpu_doc.gpu_datacenter_table(new)
        old = Path(tmp) / "gpu-datacenter-l4-fdc89c5.csv"
        write_csv(old, old_columns, old_rows)
        old_text = gpu_doc.gpu_datacenter_table(old)
    check("| `brazil3` | 1e-08 | - | 8.00x | 1.60x | 1.530e-07 |" in text,
          "new CSV: speedup vs both arms and the GPU's gap")
    check("| cpu-16t | 16 | true |" in text, "new CSV: the N-thread row shows parallel A x")
    check("GPU speedup" not in old_text and "| `brazil3` | cpu-16t | - | - | 1e-08 |" in old_text,
          "old CSV: rows render with '-' for the new columns, and no speedup is claimed from "
          "its serial-A-x cpu-16t")
    check("without** `pdhg_parallel_spmv=true`" in old_text,
          "old CSV: the note names what its CPU arm was")


def test_committed_old_csvs_render() -> None:
    results = Path(__file__).resolve().parents[2] / "bench" / "results"
    for name, render in (("gpu-real-l4-fdc89c5.csv", gpu_doc.gpu_real_section),
                         ("gpu-datacenter-l4-fdc89c5.csv", gpu_doc.gpu_datacenter_table)):
        path = results / name
        if not path.exists():
            check(True, f"{name} not in this tree; skipped")
            continue
        text = render(path)
        check("predates #488" in text, f"the committed {name} renders with its note")


def main() -> int:
    for test in (test_arms, test_gaps, test_fairness_cells, test_residuals, test_reference,
                 test_real_runner_rows, test_datacenter_rows, test_doc_real,
                 test_doc_datacenter, test_committed_old_csvs_render):
        print(test.__name__)
        test()
    print(f"\n{FAILURES} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
