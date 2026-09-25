#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for verify_certificate.py (#518).

Standalone, no test framework dependency - consistent with the rest of tools/. Run directly:

    python tools/test_verify_certificate.py

Hand-written certificates exercise every reason and every way a step can be wrong; when a
built solver is found (build/sankhya, or SANKHYA_BIN), certificates it writes for small MILPs
are checked end to end, and then tampered with: one leaf bound raised, one multiplier
changed. Each tampered file must be rejected.

Exit codes: 0 all tests passed, 1 at least one failed.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_certificate as vc  # noqa: E402

FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES += 1


def run_checker(text: str, *extra: str) -> tuple[int, str]:
    # The hand-written certificates have no MPS file: their model is the one they state.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "c.vipr"
        path.write_text(text)
        return run_checker_on(path, "--model-as-written", *extra)


def run_checker_on(path: Path, *extra: str) -> tuple[int, str]:
    result = subprocess.run([sys.executable, str(HERE / "verify_certificate.py"), str(path),
                             *extra], capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


# min x + y  s.t.  x + y >= 3/2 (c0), x, y integer in [0, 3]: optimum 2.
MODEL = """VER 1.0
VAR 2
x y
INT 2
0 1
OBJ min
2 0 1 1 1
CON 5 4
c0 G 3/2 2 0 1 1 1
lx G 0 1 0 1
ux L 3 1 0 1
ly G 0 1 1 1
uy L 3 1 1 1
RTP range 2 2
SOL 1
best 2 0 1 1 1
"""

# Branch on x: x <= 0 | x >= 1. Under each side c0 alone gives x + y >= 3/2; the split
# keeps it; rounding (integer coefficients on integer variables) gives x + y >= 2.
PROOF = """DER 6
a0 L 0 1 0 1 { asm } -1
n1 G 3/2 OBJ { lin 2 0 1 5 0 } -1
a1 G 1 1 0 1 { asm } -1
n2 G 3/2 OBJ { lin 2 0 1 7 0 } -1
u0 G 3/2 OBJ { uns 6 5 8 7 } -1
r0 G 2 OBJ { rnd 1 9 1 } -1
"""


def hand_written() -> None:
    print("hand-written certificates")
    code, out = run_checker(MODEL + PROOF)
    check(code == 0 and "VERIFIED" in out, "a branch, a split and a rounding verify", out.strip()
          .splitlines()[-2] if out.strip() else "")
    # A leaf bound raised: 3/2 -> 8/5 is more than c0 proves.
    code, out = run_checker(MODEL + PROOF.replace("n1 G 3/2", "n1 G 8/5"))
    check(code == 1 and "REJECTED" in out, "a raised leaf bound is rejected", out.strip())
    # A multiplier changed: 1 -> 1/2 on c0 proves only 3/4.
    code, out = run_checker(MODEL + PROOF.replace("{ lin 2 0 1 5 0 }", "{ lin 2 0 1/2 5 0 }"))
    check(code == 1, "a changed multiplier is rejected", out.strip())
    # A multiplier of the wrong sign on a >= row, for a >= derivation.
    code, out = run_checker(MODEL + PROOF.replace("{ lin 2 0 1 7 0 }", "{ lin 2 0 -1 7 0 }"))
    check(code == 1 and "wrong sign" in out, "a wrong-signed multiplier is rejected")
    # Rounding a bound that is not integral on its variables: impossible with a continuous y.
    code, out = run_checker(MODEL.replace("INT 2\n0 1", "INT 1\n0") + PROOF)
    check(code == 1 and "rounding" in out, "rounding over a continuous variable is rejected")
    # The two assumptions of a split must be x <= d and x >= d + 1.
    code, out = run_checker(MODEL + PROOF.replace("a1 G 1 1 0 1", "a1 G 2 1 0 1"))
    check(code == 1 and "split" in out, "a split that leaves a gap is rejected")
    # A claim stronger than the proof.
    code, out = run_checker(MODEL.replace("RTP range 2 2", "RTP range 3 3") + PROOF)
    check(code == 1, "an RTP lower bound above the proof is rejected")
    # An infeasible solution: (0, 1) violates c0.
    code, out = run_checker(MODEL.replace("best 2 0 1 1 1", "best 1 1 1") + PROOF)
    check(code == 2 and "violates" in out, "an infeasible SOL point does not verify")
    # Forward references are refused.
    code, out = run_checker(MODEL + PROOF.replace("{ uns 6 5 8 7 }", "{ uns 6 5 10 7 }"))
    check(code == 1, "a forward reference is rejected")
    # Completion: without the assumption referenced, the path bound is unavailable, and a
    # combination that needs it fails; a lin with c0 and no bound it lacks still verifies.
    completed = PROOF.replace("n1 G 3/2 OBJ { lin 2 0 1 5 0 }",
                              "n1 G 2 2 0 2 1 1 { lin 2 0 1 5 0 }")
    code, out = run_checker(MODEL + completed)
    check(code == 1, "a completion needing more than the box gives is rejected")
    # Without --mps and without --model-as-written, a proof that checks still does not exit 0.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "c.vipr"
        path.write_text(MODEL + PROOF)
        code, out = run_checker_on(path)
    check(code == 2 and "NOT checked" in out, "no source model: never exit 0", out.strip())
    # A sol step from an INFEASIBLE point: (0, 1) is worth 1, below the true optimum 2, so
    # "x + y <= 1" contradicts r0, "no" derives 0 >= 1 from the two, and with an absurd step
    # the old checker skipped the proof side and let the claim 3 stand.
    lying = (MODEL.replace("RTP range 2 2", "RTP range 3 3").replace("best 2 0 1 1 1",
                                                                     "best 1 1 1")
             + PROOF.replace("DER 6", "DER 8") + "s0 L 1 OBJ { sol } -1\n"
             + "no G 1 0 { lin 2 10 1 11 -1 } -1\n")
    code, out = run_checker(lying)
    check(code == 1 and "sol" in out, "a sol step from an infeasible point is rejected",
          out.strip())
    # A final bound that still carries an assumption: n1 holds only under x <= 0.
    code, out = run_checker(MODEL.replace("RTP range 2 2", "RTP range 3/2 3/2")
                            + "DER 2\n" + "\n".join(PROOF.splitlines()[1:3]) + "\n")
    check(code == 1, "a bound under an undischarged assumption is rejected", out.strip())
    # uns naming a model constraint (c0, index 0) where an assumption must be.
    code, out = run_checker(MODEL + PROOF.replace("{ uns 6 5 8 7 }", "{ uns 6 0 8 7 }"))
    check(code == 1, "uns on a non-assumption is rejected", out.strip())


# ---- End to end, with the solver --------------------------------------------------------

KNAPSACK = """NAME          KNAP
OBJSENSE
    {sense}
ROWS
 N  obj
 L  c1
 L  c2
 L  c3
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    x1        obj       {c1}   c1        2
    x1        c2        4      c3        3
    x2        obj       {c2}   c1        3
    x2        c2        1      c3        4
    x3        obj       {c3}   c1        1
    x3        c2        2      c3        2
    MARKER                 'MARKER'                 'INTEND'
    z         obj       {cz}   c1        0.1
RHS
    rhs       c1        5      c2        11
    rhs       c3        8
BOUNDS
 UP bnd       x1        10
 UP bnd       x2        10
 UP bnd       x3        10
 UP bnd       z         7.5
ENDATA
"""


ODD = """NAME          ODD
ROWS
 N  obj
 E  c1
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    x         obj       1      c1        2
    MARKER                 'MARKER'                 'INTEND'
RHS
    rhs       c1        1
BOUNDS
 UP bnd       x         5
ENDATA
"""


def find_binary() -> Path | None:
    env = os.environ.get("SANKHYA_BIN")
    if env and Path(env).exists():
        return Path(env)
    root = HERE.parent
    for candidate in ("build/sankhya", "build/sankhya.exe", "build-release/sankhya",
                      "build-release/sankhya.exe"):
        if (root / candidate).exists():
            return root / candidate
    return None


def solve(binary: Path, mps: Path, cert: Path, *options: str) -> str:
    args = [str(binary), "solve", str(mps), "--option", f"write_certificate={cert}"]
    for option in options:
        args += ["--option", option]
    result = subprocess.run(args, capture_output=True, text=True)
    return result.stdout + result.stderr


def tamper_leaf_bound(text: str) -> str:
    """Raise the right-hand side of the first leaf's objective bound by 1000."""
    lines = text.splitlines()
    for k, line in enumerate(lines):
        fields = line.split()
        if fields and fields[0].startswith("n") and len(fields) > 3 and fields[3] == "OBJ":
            value = vc.Fraction(fields[2])
            shift = 1000 if fields[1] == "G" else -1000
            fields[2] = str(value + shift)
            lines[k] = " ".join(fields)
            return "\n".join(lines) + "\n"
    raise AssertionError("no leaf bound to tamper with")


def tamper_multiplier(text: str) -> str:
    """Scale the first row multiplier of the first leaf's lin by 1000."""
    lines = text.splitlines()
    for k, line in enumerate(lines):
        match = re.match(r"(n\S+ [GL] \S+ OBJ \{ lin \d+ \d+ )(\S+)(.*)", line)
        if match and vc.Fraction(match.group(2)) != 0:
            scaled = vc.Fraction(match.group(2)) * 1000
            lines[k] = match.group(1) + str(scaled) + match.group(3)
            return "\n".join(lines) + "\n"
    raise AssertionError("no multiplier to tamper with")


def end_to_end() -> None:
    binary = find_binary()
    if binary is None:
        print("end to end: SKIPPED, no built solver (set SANKHYA_BIN)")
        return
    print(f"end to end with {binary}")
    cases = [
        ("min pure integer", dict(sense="MIN", c1=-5, c2=-4, c3=-3, cz=0)),
        ("max pure integer", dict(sense="MAX", c1=5, c2=4, c3=3, cz=0)),
        ("min with a continuous column", dict(sense="MIN", c1=-5, c2=-4, c3=-3, cz=-0.7)),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        for name, fields in cases:
            mps = Path(tmp) / "knap.mps"
            mps.write_text(KNAPSACK.format(**fields))
            cert = Path(tmp) / "knap.vipr"
            log = solve(binary, mps, cert, "enable_root_cuts=false")
            check(cert.exists(), f"{name}: the solver writes a certificate",
                  "" if cert.exists() else log[-400:])
            if not cert.exists():
                continue
            code, out = run_checker_on(cert, "--mps", str(mps), "--feas-tol", "1e-9")
            check(code == 0 and "VERIFIED" in out, f"{name}: it verifies", out.strip()
                  .splitlines()[-2] if out.strip() else "")
            text = cert.read_text()
            bad = Path(tmp) / "bad.vipr"
            bad.write_text(tamper_leaf_bound(text))
            code, out = run_checker_on(bad)
            check(code == 1, f"{name}: a leaf bound raised is rejected", out.strip()[:200])
            bad.write_text(tamper_multiplier(text))
            code, out = run_checker_on(bad)
            check(code == 1, f"{name}: a multiplier changed is rejected", out.strip()[:200])
        # An infeasible MILP: 2 x = 1 with x integer. Both children of the root are infeasible
        # LPs, so the proof is two Farkas contradictions and a split.
        mps = Path(tmp) / "odd.mps"
        mps.write_text(ODD)
        cert = Path(tmp) / "odd.vipr"
        log = solve(binary, mps, cert, "enable_root_cuts=false")
        code, out = run_checker_on(cert, "--mps", str(mps)) if cert.exists() else (-1, log)
        check(code == 0 and "proved infeasible" in out, "an infeasible MILP: proved infeasible",
              out.strip()[-200:])
        # With cuts on, the cut rows are derived in the certificate: see
        # test_verify_certificate_cuts.py.


def main() -> int:
    global FAILURES
    hand_written()
    end_to_end()
    # Cut rows (#518): their own file, which imports this one as a module, so its count of
    # failures lives there.
    import test_verify_certificate_cuts as cuts
    cuts.main()
    FAILURES += cuts.base.FAILURES
    print("all passed" if FAILURES == 0 else f"{FAILURES} failure(s)")
    return 0 if FAILURES == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
