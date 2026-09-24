#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for bench/case_studies/refinery/generator.py (#517).

Checks that the generator:
  - Runs without error at small size for LP and MILP.
  - Writes a valid free-format MPS file (ROWS / COLUMNS / RHS / BOUNDS / ENDATA).
  - The MPS file can be parsed by sankhya's MPS reader when a built binary is
    present (verified via `sankhya info`).
  - The LP KKT verifier passes (already enforced inside build_lp / verify_lp,
    but we check it raises on a deliberately corrupted plan).

    python tests/unit/test_refinery_generator.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

# Make the generator importable without installing anything
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "bench" / "case_studies" / "refinery"))
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
        check(True, "LP small build and KKT verify")
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

def test_mps_readable_by_sankhya() -> None:
    # Locate sankhya binary
    candidates = [
        REPO_ROOT / "build" / "sankhya.exe",
        REPO_ROOT / "build" / "sankhya",
        REPO_ROOT / "build" / "Release" / "sankhya.exe",
    ]
    binary = next((str(p) for p in candidates if p.exists()), None)
    if binary is None:
        binary = shutil.which("sankhya")
    if binary is None:
        print("  [skip] no sankhya binary found; MPS readability check skipped")
        return

    _, tmp = _make_lp(seed=1, milp=False)
    try:
        result = subprocess.run(
            [binary, "info", str(tmp)],
            capture_output=True, text=True, timeout=30)
        check(result.returncode == 0,
              "sankhya info reads LP MPS without error",
              f"rc={result.returncode}  stderr={result.stderr.strip()[:120]}")
    except subprocess.TimeoutExpired:
        check(False, "sankhya info LP", "timed out after 30s")
    finally:
        tmp.unlink(missing_ok=True)

    _, tmp = _make_lp(seed=1, milp=True)
    try:
        result = subprocess.run(
            [binary, "info", str(tmp)],
            capture_output=True, text=True, timeout=30)
        check(result.returncode == 0,
              "sankhya info reads MILP MPS without error",
              f"rc={result.returncode}  stderr={result.stderr.strip()[:120]}")
    except subprocess.TimeoutExpired:
        check(False, "sankhya info MILP", "timed out after 30s")
    finally:
        tmp.unlink(missing_ok=True)


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
    print("test_refinery_generator.py (#517)")
    test_lp_small_builds_and_verifies()
    test_milp_small_builds()
    test_mps_sections_present()
    test_mps_synthetic_notice_present()
    test_mps_readable_by_sankhya()
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
