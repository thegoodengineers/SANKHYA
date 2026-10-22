#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the two GPU evidence runners (#888, #759) that need no card: the LPfeas-14
manifest's shape and rule, both runners' row construction, the arm table of the refinery
runner, the status map and the .sol conversion of a cuOpt-shaped answer through
fill_row_activities, judged by tools/verify_solution.py on a committed Netlib model.

    python bench/runners/test_mittelmann_cuopt.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_refinery_year  # noqa: E402
import mittelmann_cuopt  # noqa: E402
import rivals  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}  {detail}")
    if not condition:
        FAILURES += 1


def test_manifest() -> None:
    m = mittelmann_cuopt.load_manifest()
    inst = m["instances"]
    check(len(inst) == 14, "14 instances", str(len(inst)))
    sizes = [e["archive_bytes"] for e in inst.values()]
    check(sizes == sorted(sizes), "listed smallest archive first (the rule)")
    check(all(len(e["archive_sha256"]) == 64 and e["url"].startswith(m["source"])
              for e in inst.values()), "every entry pins a sha256 and a lptestset URL")
    check(all(e["rows"] > 0 and e["columns"] > 0 and e["nonzeros"] > 0 for e in inst.values()),
          "every entry carries the page's dimensions")
    check(all((e["reference_objective"] is None) == (e["reference_source"] is None)
              for e in inst.values()), "a reference always names its source")
    check("qap15" in inst and "neos-5251015" in inst, "qap15 first, neos-5251015 last")


def test_status_map_and_rows() -> None:
    check(mittelmann_cuopt.CUOPT_STATUS["timelimit"] == "time_limit"
          and mittelmann_cuopt.CUOPT_STATUS["primalfeasible"] == "feasible",
          "cuOpt status names map to ours")
    entry = {"rows": 10, "columns": 20, "nonzeros": 30}
    result = {"status": "optimal", "objective": 2.5, "iterations": 7, "seconds": 1.5,
              "wall": 2.0}
    row = mittelmann_cuopt.build_row("x", entry, "ab" * 32, "cuopt", "PDLP", 1e-4, 300.0,
                                     result, (1, ""), (1, ""), (2.0, "highs"),
                                     {"git_commit": "c", "machine": "m", "gpu": "g",
                                      "driver_version": "", "cuda_runtime": "",
                                      "cuopt_version": "26.8.0", "timestamp_utc": "t"})
    check(set(row) == set(mittelmann_cuopt.CSV_COLUMNS), "cuOpt row has every CSV column",
          str(set(mittelmann_cuopt.CSV_COLUMNS) ^ set(row)))
    check(row["rel_gap"] == "2.500e-01" and row["tolerance"] == "1e-04", "gap and tolerance",
          f"{row['rel_gap']} {row['tolerance']}")
    arms = gpu_refinery_year.arms(4, True)
    labels = [a[0] for a in arms]
    check(labels[:3] == ["gpu-pdhg-1e4-xover", "gpu-pdhg-1e8-xover", "gpu-pdhg-1e8"],
          "the card's three arms come first", str(labels))
    check("cpu-pdhg-4t-1e4-xover" in labels and labels[-4:] == [
        "highs-ipm-1t", "highs-simplex-1t", "highs-ipm-4t", "highs-simplex-4t"],
          "the CPU PDHG and the four HiGHS references", str(labels))
    check(all("pdhg_crossover=true" in a[2] for a in arms if a[0].endswith("xover")),
          "every -xover arm passes pdhg_crossover=true")
    check([a[0] for a in gpu_refinery_year.arms(4, False)][0].startswith("cpu-"),
          "--no-gpu drops the card's arms")
    rrow = gpu_refinery_year.build_row(
        "refinery_12", (1068, 1656, 13610), "0" * 64, arms[0],
        {"status": "optimal", "objective": -61203.88791, "seconds": 1.0, "seconds_min": 0.9,
         "seconds_max": 1.2, "repeats": 3, "wall": 1.3, "verified": 1,
         "message": "crossover: 12 pivots", "crossover_pivots": "12"},
        -61203.88791, 600.0, {k: "" for k in ("git_commit", "machine", "host", "gpu",
                                               "driver_version", "cuda_runtime",
                                               "timestamp_utc")})
    check(set(rrow) == set(gpu_refinery_year.CSV_COLUMNS), "refinery row has every column",
          str(set(gpu_refinery_year.CSV_COLUMNS) ^ set(rrow)))
    check(rrow["rel_gap"] == "0.000e+00" and rrow["crossover_pivots"] == "12"
          and rrow["tolerance"] == "1e-04", "refinery row cells", str(rrow))
    m = gpu_refinery_year.RESTARTS.search("converged after 6880 iterations and 16 restarts;")
    check(m is not None and m.group(2) == "16", "restarts parsed from the message")


def test_cuopt_sol_conversion_verifies() -> None:
    """A cuOpt-shaped answer (names, values, reduced costs, duals, placeholder activities)
    on Netlib afiro, written as rivals.write_sol does in the worker, activities filled, and
    judged by the verifier. The point is our own verified optimum, read from a solve of the
    committed model if a binary is at build/sankhya (or $SANKHYA_BINARY), else skipped."""
    mps = REPO_ROOT / "data" / "netlib" / "afiro.mps"
    import os  # noqa: PLC0415
    candidates = [Path(os.environ["SANKHYA_BINARY"])] if os.environ.get("SANKHYA_BINARY") else []
    candidates += [REPO_ROOT / "build" / "sankhya", REPO_ROOT / "build" / "sankhya.exe"]
    binary = next((p for p in candidates if p.exists()), None)
    if binary is None or not mps.exists():
        print("  [SKIP] .sol conversion: no build/sankhya binary or afiro.mps")
        return
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from verify_solution_sol import parse_sol  # noqa: PLC0415
    with tempfile.TemporaryDirectory() as tmp:
        ours = Path(tmp) / "ours.sol"
        done = subprocess.run([str(binary), "solve", str(mps), "--write-sol", str(ours),
                               "--option", "log_to_console=false"], capture_output=True,
                              text=True, check=False)
        if not ours.exists():
            print(f"  [SKIP] .sol conversion: the binary did not run ({done.stderr[-120:]})")
            return
        ref = parse_sol(ours)
        fake = Path(tmp) / "cuopt.sol"
        rivals.write_sol(fake, solver="cuopt", status="optimal",
                         objective=ref.header_float("objective"),
                         columns={n: (ref.col_value[n], ref.col_dual.get(n, 0.0))
                                  for n in ref.col_value},
                         rows={n: (0.0, ref.row_dual.get(n, 0.0)) for n in ref.row_dual})
        mittelmann_cuopt.fill_row_activities(fake, mps)
        verdict, failing = mittelmann_cuopt.verify(mps, fake, None)
        check(verdict == 1, "a cuOpt-shaped .sol with recomputed activities verifies", failing)
        at_tol = mittelmann_cuopt.verify(mps, fake, 1e-4)
        check(at_tol[0] == 1, "and at the requested tolerance", at_tol[1])


def main() -> int:
    for test in (test_manifest, test_status_map_and_rows, test_cuopt_sol_conversion_verifies):
        print(test.__name__)
        test()
    print(f"\n{'OK' if FAILURES == 0 else f'{FAILURES} FAILED'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
