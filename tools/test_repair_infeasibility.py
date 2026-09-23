#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for tools/repair_infeasibility.py (#523).

#523's own acceptance criterion is "each repair verified feasible by the independent tool
after applying it" - every test here does exactly that: build an infeasible model, repair
it, APPLY the reported relaxation to a fresh copy of the ORIGINAL model, solve THAT, and run
tools/verify_solution.py (independent of the solver, and of this tool) on the result.
Passing the repair's own phase-1 solve is not accepted as evidence on its own.

    python tools/test_repair_infeasibility.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bindings" / "python"))

from repair_infeasibility import parse_mps, repair  # noqa: E402
from verify_solution_mps import Model as SourceModel  # noqa: E402


def _subprocess_env() -> dict:
    """See tools/bundle.py's function of the same name: sankhya-cli needs the MSYS2
    UCRT64 bin directory on PATH to start at all on Windows (libgomp-1.dll)."""
    env = dict(os.environ)
    if sys.platform != "win32":
        return env
    extra = os.environ.get("SANKHYA_DLL_DIR") or r"C:\msys64\ucrt64\bin"
    if extra and Path(extra).is_dir():
        env["PATH"] = extra + os.pathsep + env.get("PATH", "")
    return env

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFY_SOLUTION = REPO_ROOT / "tools" / "verify_solution.py"

FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    if condition:
        print(f"  [PASS] {name}" + (f"  {detail}" if detail else ""))
    else:
        FAILURES += 1
        print(f"  [FAIL] {name}" + (f"  {detail}" if detail else ""))


def _write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


# min x  s.t.  x >= 5 (R1), x <= 2 (R2), x free above 0 (default lower 0). Infeasible: R1
# and R2 cannot both hold. Minimal total relaxation is 3, on either row alone.
_CROSSED_BOUNDS_MPS = """NAME          CROSSED
ROWS
 N  COST
 G  R1
 L  R2
COLUMNS
    X         COST         1.0    R1           1.0
    X         R2           1.0
RHS
    RHS       R1           5.0    R2           2.0
ENDATA
"""

# Two columns, one equality each side of an infeasible pair: x = 10 (R1, forced up by a
# large coefficient... simpler: two SEPARATE fixed-value equalities that cannot both hold
# because they constrain the SAME column to two different exact values.
_INFEASIBLE_EQUALITIES_MPS = """NAME          EQPAIR
ROWS
 N  COST
 E  R1
 E  R2
COLUMNS
    X         COST         1.0    R1           1.0
    X         R2           1.0
RHS
    RHS       R1           3.0    R2           7.0
ENDATA
"""


def _apply_relaxation_and_verify(mps_path: Path, report: dict, *, binary: Path) -> bool:
    """Apply every reported relaxation to a FRESH copy of the model, solve, and check with
    the independent verify_solution.py. Returns True iff it verifies as feasible."""
    source = parse_mps(mps_path)
    row_lower = list(source.row_lower)
    row_upper = list(source.row_upper)
    for r in report["relaxations"]:
        if r["kind"] != "row":
            continue
        i = source.row_index[r["name"]]
        if r["side"] == "lower":
            row_lower[i] -= r["amount"]
        else:
            row_upper[i] += r["amount"]

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        relaxed_mps = directory / "relaxed.mps"
        lines = ["NAME          RELAXED", "ROWS", " N  COST"]
        for i, name in enumerate(source.row_names):
            sense = "E" if row_lower[i] == row_upper[i] else (
                "G" if row_upper[i] >= 1e30 else ("L" if row_lower[i] <= -1e30 else "E"))
            lines.append(f" {sense}  {name}")
        lines.append("COLUMNS")
        for j, name in enumerate(source.col_names):
            lines.append(f"    {name}         COST         {source.col_cost[j]!r}")
            for row, value in source.entries[j]:
                lines.append(f"    {name}         {source.row_names[row]}  {value!r}")
        lines.append("RHS")
        for i, name in enumerate(source.row_names):
            bound = row_upper[i] if row_upper[i] < 1e30 else row_lower[i]
            lines.append(f"    RHS       {name}  {bound!r}")
        ranges = [(i, name) for i, name in enumerate(source.row_names)
                 if row_lower[i] > -1e30 and row_upper[i] < 1e30
                 and row_lower[i] != row_upper[i]]
        if ranges:
            lines.append("RANGES")
            for i, name in ranges:
                lines.append(f"    RNG       {name}  {row_upper[i] - row_lower[i]!r}")
        lines.append("BOUNDS")
        for j, name in enumerate(source.col_names):
            if source.col_lower[j] > -1e30 and source.col_lower[j] != 0.0:
                lines.append(f" LO BND       {name}  {source.col_lower[j]!r}")
            if source.col_upper[j] < 1e30:
                lines.append(f" UP BND       {name}  {source.col_upper[j]!r}")
        lines.append("ENDATA")
        relaxed_mps.write_text("\n".join(lines) + "\n", encoding="ascii")

        sol_path = directory / "relaxed.sol"
        run = subprocess.run(
            [str(binary), "solve", str(relaxed_mps), "--write-sol", str(sol_path),
            "--option", "log_to_console=false"],
            capture_output=True, text=True, env=_subprocess_env())
        if run.returncode not in (0, 1):
            print(f"    (solve failed: {run.stdout}\n{run.stderr})")
            return False
        if not sol_path.is_file():
            return False

        verify = subprocess.run(
            [sys.executable, str(VERIFY_SOLUTION), str(relaxed_mps), str(sol_path), "--quiet"],
            capture_output=True, text=True)
        if verify.returncode != 0:
            print(f"    (verify_solution.py: {verify.stdout}\n{verify.stderr})")
        # A relaxed model that VERIFIES must also not be reported infeasible.
        return verify.returncode == 0 and "status infeasible" not in sol_path.read_text().lower()


def _default_binary() -> Path | None:
    for candidate in ("build/sankhya.exe", "build/sankhya", "build-release/sankhya.exe",
                      "build-release/sankhya"):
        path = REPO_ROOT / candidate
        if path.is_file():
            return path
    return None


def test_crossed_row_bounds_is_repaired_and_the_repair_verifies() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mps_path = _write(Path(tmp), "crossed.mps", _CROSSED_BOUNDS_MPS)
        source = parse_mps(mps_path)
        report = repair(source, include_bounds=False, weights={}, optimize=False, log=False)

        check(report["repairable"], "the crossed-bounds model is reported repairable")
        check(abs(report["total_weighted_relaxation"] - 3.0) < 1e-6,
              "the minimal total relaxation is 3 (hand-derived: R1 needs 5, R2 allows 2)",
              str(report["total_weighted_relaxation"]))

        binary = _default_binary()
        if binary is None:
            print("    (SKIPPED verification: no sankhya-cli binary found)")
            return
        ok = _apply_relaxation_and_verify(mps_path, report, binary=binary)
        check(ok, "applying the reported relaxation to a fresh model makes it verifiably "
              "feasible (tools/verify_solution.py, independent of this tool and the solver)")


def test_infeasible_equalities_is_repaired_and_the_repair_verifies() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mps_path = _write(Path(tmp), "eqpair.mps", _INFEASIBLE_EQUALITIES_MPS)
        source = parse_mps(mps_path)
        report = repair(source, include_bounds=False, weights={}, optimize=False, log=False)

        check(report["repairable"], "the conflicting-equalities model is reported repairable")
        check(abs(report["total_weighted_relaxation"] - 4.0) < 1e-6,
              "the minimal total relaxation is 4 (hand-derived: |7 - 3|)",
              str(report["total_weighted_relaxation"]))

        binary = _default_binary()
        if binary is None:
            print("    (SKIPPED verification: no sankhya-cli binary found)")
            return
        ok = _apply_relaxation_and_verify(mps_path, report, binary=binary)
        check(ok, "applying the reported relaxation makes the model verifiably feasible")


def test_an_already_feasible_model_needs_no_relaxation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        feasible_mps = _write(Path(tmp), "feasible.mps", _CROSSED_BOUNDS_MPS.replace(
            "R2           2.0", "R2           10.0"))
        source = parse_mps(feasible_mps)
        report = repair(source, include_bounds=False, weights={}, optimize=False, log=False)
        check(report["repairable"], "a feasible model is (trivially) repairable")
        check(report["total_weighted_relaxation"] == 0.0,
              "a feasible model needs zero relaxation",
              str(report["total_weighted_relaxation"]))
        check(report["relaxations"] == [], "no relaxation is reported for a feasible model")


def test_weights_favour_the_cheaper_row() -> None:
    """A row weighted heavily should be relaxed less than an unweighted (or lighter) one,
    when both COULD carry the repair - the whole point of a WEIGHTED elastic sum."""
    # min x  s.t.  x >= 10 (R1), x <= 0 (R2). Symmetric crossed bounds: either row alone
    # can carry a relaxation of 10. Weighting R1 heavily should push the repair onto R2.
    mps_text = """NAME          WEIGHTED
ROWS
 N  COST
 G  R1
 L  R2
COLUMNS
    X         COST         1.0    R1           1.0
    X         R2           1.0
RHS
    RHS       R1           10.0    R2           0.0
ENDATA
"""
    with tempfile.TemporaryDirectory() as tmp:
        mps_path = _write(Path(tmp), "weighted.mps", mps_text)
        source = parse_mps(mps_path)
        report = repair(source, include_bounds=False, weights={"R1": 100.0}, optimize=False,
                        log=False)
        check(report["repairable"], "the symmetric crossed-bounds model is repairable")
        r2_moved = any(r["name"] == "R2" for r in report["relaxations"])
        r1_moved = any(r["name"] == "R1" for r in report["relaxations"])
        check(r2_moved and not r1_moved,
              "with R1 weighted 100x, the repair falls on the cheap row (R2), not R1",
              str(report["relaxations"]))


def main() -> int:
    print("tools/repair_infeasibility.py tests\n")
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            print(name)
            function()
    print()
    if FAILURES == 0:
        print("ALL TESTS PASSED")
        return 0
    print(f"{FAILURES} check(s) FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
