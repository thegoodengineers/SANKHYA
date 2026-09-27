#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for cut rows in a MILP certificate (#518), run from test_verify_certificate.py.

A cut the node LPs carried is a derived constraint of the certificate: a Chvatal-Gomory
`rnd` step, or a split (two assumptions, a `lin` step under each, and the `uns` step that
discharges them), as in Cheung, Gleixner and Steffy, "Verifying integer programming
results", IPCO 2017. The checker re-derives every one in exact arithmetic, so a cut with one
coefficient changed must be rejected. Hand-written certificates hold both forms; when a built
solver is found, a certificate written with root cuts on is checked end to end.

    python tools/test_verify_certificate_cuts.py
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_verify_certificate as base  # noqa: E402

# min -x - y - z  s.t.  2x + 2y + 2z <= 3, binaries: optimum -1. The CG cut x + y + z <= 1
# is c0 times 1/2 rounded down; the root bound -1 is that cut with multiplier -1.
ROUNDING = """VER 1.0
VAR 3
x y z
INT 3
0 1 2
OBJ min
3 0 -1 1 -1 2 -1
CON 7 6
c0 L 3 3 0 2 1 2 2 2
lx G 0 1 0 1
ux L 1 1 0 1
ly G 0 1 1 1
uy L 1 1 1 1
lz G 0 1 2 1
uz L 1 1 2 1
RTP range -1 -1
SOL 1
best 1 0 1
DER 2
cut0 L 1 3 0 1 1 1 2 1 { rnd 1 0 1/2 } -1
n0 G -1 OBJ { lin 1 7 -1 } -1
"""

# min -x + 3y  s.t.  x - y <= 3/2, x integer in [0, 3], y >= 0 continuous: optimum -1 at
# (1, 0), LP bound -3/2. The MIR cut x - 2y <= 1 holds under x <= 1 (with y >= 0) and under
# x >= 2 (2 c0 - (x >= 2)); with it the root bound is -1.
SPLIT = """VER 1.0
VAR 2
x y
INT 1
0
OBJ min
2 0 -1 1 3
CON 4 3
c0 L 3/2 2 0 1 1 -1
lx G 0 1 0 1
ux L 3 1 0 1
ly G 0 1 1 1
RTP range -1 -1
SOL 1
best 1 0 1
DER 6
cut0_a0 L 1 1 0 1 { asm } -1
cut0_a1 G 2 1 0 1 { asm } -1
cut0_d0 L 1 2 0 1 1 -2 { lin 1 4 1 } -1
cut0_d1 L 1 2 0 1 1 -2 { lin 2 0 2 5 -1 } -1
cut0 L 1 2 0 1 1 -2 { uns 6 4 7 5 } -1
n0 G -1 OBJ { lin 1 8 -1 } -1
"""


# Gomory cuts at the root: min -x0 - 2 x1 on 2 x0 + 2 x1 <= 3, -2 x0 + 2 x1 <= 1 (LP vertex
# (1/2, 1)), a continuous x2, and a binary knapsack.
GOMORY = """NAME          CUTS1
ROWS
 N  obj
 L  c1
 L  c2
 L  c3
 L  k1
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    x0        obj       -1     c1        2
    x0        c2        -2     c3        3
    x1        obj       -2     c1        2
    x1        c2        2      c3        1
    b1        obj       -5     k1        5
    b2        obj       -4     k1        4
    b3        obj       -3     k1        3
    MARKER                 'MARKER'                 'INTEND'
    x2        c3        -1
RHS
    rhs       c1        3      c2        1
    rhs       c3        4      k1        7
BOUNDS
 UP bnd       x0        10
 UP bnd       x1        10
 UP bnd       x2        5
 UP bnd       b1        1
 UP bnd       b2        1
 UP bnd       b3        1
ENDATA
"""


def hand_written() -> None:
    print("cut rows, hand-written")
    code, out = base.run_checker(ROUNDING)
    base.check(code == 0 and "VERIFIED" in out, "a Chvatal-Gomory cut row verifies",
               out.strip()[-120:])
    # x's coefficient 1 -> 2: 1/2 c0 completed over x <= 1 proves only 5/2, floor 2 > 1.
    code, out = base.run_checker(ROUNDING.replace("cut0 L 1 3 0 1 1", "cut0 L 1 3 0 2 1"))
    base.check(code == 1 and "cut0" in out, "a rounded cut with one coefficient changed is "
               "rejected", out.strip()[:160])
    # Rounding needs the combination's value: 1/2 -> 1 proves x + y + z <= 3 only.
    code, out = base.run_checker(ROUNDING.replace("{ rnd 1 0 1/2 }", "{ rnd 1 0 1 }"))
    base.check(code == 1, "a rounding with the wrong multiplier is rejected")
    code, out = base.run_checker(SPLIT)
    base.check(code == 0 and "VERIFIED" in out, "a split cut row verifies", out.strip()[-120:])
    # y's coefficient -2 -> -1 in every line of the cut: under x >= 2 the combination is
    # x - 2y <= 1, and x - y needs an upper bound on y that does not exist.
    code, out = base.run_checker(SPLIT.replace("1 -2 {", "1 -1 {"))
    base.check(code == 1 and "cut0_d1" in out, "a split cut with one coefficient changed is "
               "rejected", out.strip()[:160])
    # Only the final line changed: the two sides no longer imply it.
    code, out = base.run_checker(SPLIT.replace("cut0 L 1 2 0 1 1 -2", "cut0 L 1 2 0 1 1 -1"))
    base.check(code == 1, "a split whose conclusion differs from its sides is rejected",
               out.strip()[:160])
    # A split on a continuous column is no disjunction.
    code, out = base.run_checker(SPLIT.replace("cut0_a0 L 1 1 0 1", "cut0_a0 L 1 1 1 1")
                                 .replace("cut0_a1 G 2 1 0 1", "cut0_a1 G 2 1 1 1"))
    base.check(code == 1, "a split on a continuous column is rejected", out.strip()[:160])


def cut_indices(text: str) -> list[int]:
    return sorted({int(m.group(1)) for m in re.finditer(r"^cut(\d+) ", text, re.MULTILINE)})


def tamper_cut(text: str, k: int) -> str:
    """Raise the first coefficient of cut k by 1 in every line that states it."""
    lines = text.splitlines()
    pattern = re.compile(rf"^(cut{k}(?:_d\d)? L \S+ \d+ \d+ )(\S+)(.*)$")
    changed = 0
    for i, line in enumerate(lines):
        match = pattern.match(line)
        if match:
            value = base.vc.Fraction(match.group(2)) + 1
            lines[i] = match.group(1) + str(value) + match.group(3)
            changed += 1
    assert changed > 0, f"cut{k} not found"
    return "\n".join(lines) + "\n"


def end_to_end() -> None:
    binary = base.find_binary()
    if binary is None:
        print("cut rows end to end: SKIPPED, no built solver (set SANKHYA_BIN)")
        return
    print(f"cut rows end to end with {binary}")
    with tempfile.TemporaryDirectory() as tmp:
        written = 0
        models = [(f"knapsack {sense} {costs}",
                   base.KNAPSACK.format(sense=sense, c1=costs[0], c2=costs[1], c3=costs[2],
                                        cz=costs[3]))
                  for sense, costs in (("MIN", (-5, -4, -3, 0)), ("MAX", (5, 4, 3, 0)),
                                       ("MIN", (-5, -4, -3, -0.7)))]
        models.append(("gomory", GOMORY))
        for name, text_of_model in models:
            mps = Path(tmp) / "model.mps"
            mps.write_text(text_of_model)
            cert = Path(tmp) / "cuts.vipr"
            if cert.exists():
                cert.unlink()
            log = base.solve(binary, mps, cert, "enable_root_cuts=true")
            base.check(cert.exists(), f"{name}: written with root cuts on",
                       "" if cert.exists() else log[-300:])
            if not cert.exists():
                continue
            code, out = base.run_checker_on(cert, "--mps", str(mps), "--feas-tol", "1e-9")
            base.check(code == 0 and "VERIFIED" in out, f"{name}: it verifies",
                       out.strip()[-160:])
            text = cert.read_text()
            cuts = cut_indices(text)
            if not cuts:
                continue
            written += 1
            bad = Path(tmp) / "bad.vipr"
            for k in cuts:
                bad.write_text(tamper_cut(text, k))
                code, out = base.run_checker_on(bad, "--mps", str(mps))
                base.check(code == 1 and f"cut{k}" in out,
                           f"{name}: cut{k} with one coefficient changed is rejected",
                           out.strip()[:160])
        base.check(written > 0, "at least one certificate carries derived cut rows")
        # Dense cuts (#496): on 6 columns every Gomory cut over one nonzero is above the
        # density cap, so the certificate verified above carries a cut only the root's
        # dense admission let in; without it there are fewer cut rows.
        mps = Path(tmp) / "model.mps"
        mps.write_text(GOMORY)
        counts = []
        for dense in ("cut_dense_max=0", "cut_dense_max=10"):
            cert = Path(tmp) / "dense.vipr"
            if cert.exists():
                cert.unlink()
            base.solve(binary, mps, cert, "enable_root_cuts=true", dense)
            counts.append(len(cut_indices(cert.read_text())) if cert.exists() else -1)
        base.check(counts[1] > counts[0] >= 0, "gomory: the dense admission writes more cut "
                   "rows, and they verified above", f"cut rows without/with: {counts}")


def main() -> int:
    hand_written()
    end_to_end()
    return 0 if base.FAILURES == 0 else 1


if __name__ == "__main__":
    code = main()
    print("all passed" if code == 0 else f"{base.FAILURES} failure(s)")
    sys.exit(code)
