#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for bench/case_studies/refinery/generator.py (#517).

Checks that the generator:
  - Runs without error at small size for LP and MILP.
  - Writes a valid free-format MPS file (ROWS / COLUMNS / RHS / BOUNDS / ENDATA).
  - With a built solver (build/sankhya, or SANKHYA_BIN): both files solve, and
    the MILP's optimum is strictly above the LP's, so the binaries matter.
  - The LP KKT verifier passes (already enforced inside build_lp / verify_lp,
    but we check it raises on a deliberately corrupted plan).

    python bench/case_studies/refinery/test_generator.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

# Make the generator importable without installing anything
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import generator  # noqa: E402

FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    label = "PASS" if condition else "FAIL"
    suffix = f"  {detail}" if detail else ""
    print(f"  [{label}] {name}{suffix}")
    if not condition:
        FAILURES += 1


# ---------------------------------------------------------------------------
# Helper: build an LP instance at small size and write it to a temp file
# ---------------------------------------------------------------------------

def _make_lp(seed: int = 1, milp: bool = False) -> tuple[generator.Instance, Path]:
    dims = dict(generator.SIZE_PRESETS["small"])
    dims["seed"] = seed
    inst = generator.build_lp(**dims)
    generator.verify_lp(inst)
    if milp:
        generator.extend_to_milp(inst, dims["periods"], dims["crudes"])
    tmp = Path(tempfile.mktemp(suffix=".mps"))
    generator.write_mps(
        inst, tmp, milp=milp,
        periods=dims["periods"], crudes=dims["crudes"],
        products=dims["products"], units=dims["units"],
        seed=seed)
    return inst, tmp


# ---------------------------------------------------------------------------
# Test: LP generation and KKT verification pass at small size
# ---------------------------------------------------------------------------

def test_lp_small_builds_and_verifies() -> None:
    try:
        inst, tmp = _make_lp(seed=1, milp=False)
        check(len(inst.cols) > 0 and len(inst.rows) > 0, "LP small build and KKT verify",
              f"{len(inst.rows)} rows, {len(inst.cols)} cols")
        check(tmp.exists() and tmp.stat().st_size > 0, "LP MPS file written",
              str(tmp))
        tmp.unlink(missing_ok=True)
    except Exception as exc:
        check(False, "LP small build and KKT verify", str(exc))


# ---------------------------------------------------------------------------
# Test: MILP generation at small size
# ---------------------------------------------------------------------------

def test_milp_small_builds() -> None:
    try:
        inst, tmp = _make_lp(seed=1, milp=True)
        binary_count = sum(1 for b in inst.integer if b)
        check(binary_count > 0, "MILP has binary variables", f"binary={binary_count}")
        check(tmp.exists() and tmp.stat().st_size > 0, "MILP MPS file written")
        tmp.unlink(missing_ok=True)
    except Exception as exc:
        check(False, "MILP small build", str(exc))


# ---------------------------------------------------------------------------
# Test: MPS file has required sections in the right order
# ---------------------------------------------------------------------------

def test_mps_sections_present() -> None:
    _, tmp = _make_lp(seed=2, milp=False)
    try:
        text = tmp.read_text(encoding="utf-8")
        sections = ["NAME", "ROWS", "COLUMNS", "RHS", "BOUNDS", "ENDATA"]
        positions = [text.find(s) for s in sections]
        for i, (sec, pos) in enumerate(zip(sections, positions)):
            check(pos >= 0, f"section {sec!r} present")
        check(all(positions[i] < positions[i + 1] for i in range(len(positions) - 1)),
              "sections appear in correct order")
        # Objective row
        check(" N  COST" in text, "objective row declared as N  COST")
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Test: MPS contains the synthetic-data disclaimer comment
# ---------------------------------------------------------------------------

def test_mps_synthetic_notice_present() -> None:
    _, tmp = _make_lp(seed=3, milp=False)
    try:
        text = tmp.read_text(encoding="utf-8")
        check("SYNTHETIC" in text.upper(), "synthetic data notice in MPS file")
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Test: MPS file can be read by sankhya (if binary is available)
# ---------------------------------------------------------------------------

def _find_binary() -> str | None:
    import os
    env = os.environ.get("SANKHYA_BIN")
    if env and Path(env).exists():
        return env
    for p in (REPO_ROOT / "build" / "sankhya", REPO_ROOT / "build" / "sankhya.exe",
              REPO_ROOT / "build" / "Release" / "sankhya.exe"):
        if p.exists():
            return str(p)
    return shutil.which("sankhya")


def _solve(binary: str, mps: Path) -> tuple[str, float | None]:
    """The status and objective the CLI reports for `mps`."""
    import re
    result = subprocess.run([binary, "solve", str(mps), "--time-limit", "60"],
                            capture_output=True, text=True, timeout=120)
    text = result.stdout + result.stderr
    # The summary table the CLI prints for every model class.
    status = re.search(r"^status\s+(\S+)", text, flags=re.M)
    objective = re.search(r"^objective\s+(\S+)", text, flags=re.M)
    return (status.group(1) if status else f"rc={result.returncode}",
            float(objective.group(1)) if objective else None)


def test_solves_and_the_milp_is_not_its_lp() -> None:
    binary = _find_binary()
    if binary is None:
        print("  [skip] no sankhya binary found; solve checks skipped")
        return
    inst_lp, lp = _make_lp(seed=1, milp=False)
    _, milp = _make_lp(seed=1, milp=True)
    try:
        lp_status, lp_obj = _solve(binary, lp)
        milp_status, milp_obj = _solve(binary, milp)
        check(lp_status == "optimal", "the LP solves to optimal", lp_status)
        check(milp_status == "optimal", "the MILP solves to optimal", milp_status)
        analytic = float(sum(inst_lp.cost[j] * inst_lp.x[j] for j in range(len(inst_lp.cols))))
        check(lp_obj is not None and abs(lp_obj - analytic) <= 1e-6 * max(1.0, abs(analytic)),
              "the LP optimum is the generator's KKT-verified one", f"{lp_obj} vs {analytic}")
        # The fixed ordering costs make the MILP strictly dearer than its LP: every period
        # buys crude, and each order costs a share of the largest purchase on top.
        check(lp_obj is not None and milp_obj is not None
              and milp_obj > lp_obj + 1e-6 * max(1.0, abs(lp_obj)),
              "the MILP optimum is strictly above the LP's", f"MILP {milp_obj}, LP {lp_obj}")
    except subprocess.TimeoutExpired:
        check(False, "solve", "timed out")
    finally:
        lp.unlink(missing_ok=True)
        milp.unlink(missing_ok=True)


def test_milp_binaries_carry_a_fixed_cost() -> None:
    inst, tmp = _make_lp(seed=1, milp=True)
    tmp.unlink(missing_ok=True)
    costs = [inst.cost[j] for j, name in enumerate(inst.cols) if name.startswith("ZDEC_")]
    check(len(costs) > 0 and all(c > 0 for c in costs),
          "every order binary has a positive fixed cost", f"{len(costs)} binaries")


# ---------------------------------------------------------------------------
# Test: verify_lp raises on a corrupted plan
# ---------------------------------------------------------------------------

def test_verify_lp_catches_infeasibility() -> None:
    dims = dict(generator.SIZE_PRESETS["small"])
    inst = generator.build_lp(**dims, seed=99)
    # Corrupt the first primal value so a balance row is violated
    inst.x[0] = inst.x[0] + Fraction(1000)
    raised = False
    try:
        generator.verify_lp(inst)
    except AssertionError:
        raised = True
    check(raised, "verify_lp raises AssertionError on infeasible plan")


# ---------------------------------------------------------------------------
# Test: size presets produce different row counts
# ---------------------------------------------------------------------------

def test_size_presets_produce_different_sizes() -> None:
    sizes = {}
    for label, preset in generator.SIZE_PRESETS.items():
        inst = generator.build_lp(**preset, seed=7)
        sizes[label] = len(inst.rows)
    check(sizes["small"] < sizes["medium"] < sizes["large"],
          "size presets produce strictly increasing row counts",
          str(sizes))


# ---------------------------------------------------------------------------
# Test: CLI --size and --milp flags produce a file
# ---------------------------------------------------------------------------

def test_cli_flags() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "test_out.mps"
        rc = generator.main(["--size", "small", "--seed", "5", "--out", str(out)])
        check(rc == 0 and out.exists(), "CLI --size small produces a file")

        out_milp = Path(tmpdir) / "test_milp.mps"
        rc = generator.main(["--size", "small", "--seed", "5", "--milp",
                             "--out", str(out_milp)])
        check(rc == 0 and out_milp.exists(), "CLI --size small --milp produces a file")
        # MILP file should be larger (extra columns and rows)
        if out.exists() and out_milp.exists():
            check(out_milp.stat().st_size > out.stat().st_size,
                  "MILP file is larger than LP file")


# ---------------------------------------------------------------------------
# Test: deterministic — same seed gives same file
# ---------------------------------------------------------------------------

def test_deterministic() -> None:
    results = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "det.mps"
            generator.main(["--size", "small", "--seed", "42", "--out", str(out)])
            results.append(out.read_text(encoding="utf-8"))
    check(results[0] == results[1], "same seed produces identical MPS file")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("test_generator.py (#517)")
    test_lp_small_builds_and_verifies()
    test_milp_small_builds()
    test_mps_sections_present()
    test_mps_synthetic_notice_present()
    test_milp_binaries_carry_a_fixed_cost()
    test_solves_and_the_milp_is_not_its_lp()
    test_verify_lp_catches_infeasibility()
    test_size_presets_produce_different_sizes()
    test_cli_flags()
    test_deterministic()
    print()
    if FAILURES:
        print(f"{FAILURES} check(s) FAILED")
        sys.exit(1)
    else:
        print("all checks passed")
