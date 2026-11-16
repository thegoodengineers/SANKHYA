#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Two questions about an LP our engines do not finish, answered without our solver.

1. IS HiGHS'S POINT FEASIBLE BY OUR STANDARD? HiGHS runs as a separate process
   (rivals.run, the head-to-head's own driver), its answer is written into our .sol layout
   and judged by tools/verify_solution.py, so a disagreement between HiGHS's objective and
   ours can be put down to one of the two points, not to a guess.

2. DOES THE FEASIBLE SET HAVE A STRICT INTERIOR? An interior point method needs one (the
   Slater condition); when some inequalities hold with equality at every feasible point
   (implicit equalities), the iterates crowd a face they cannot reach from inside. The
   phase-1 LP that maximises the smallest slack answers it (the standard construction, e.g.
   Boyd & Vandenberghe, "Convex Optimization" (2004), section 11.4.1):

       max t   s.t.  a_i x + s_i t <= u_i   (each finite row upper bound, row not an equality)
                     a_i x - s_i t >= l_i   (each finite row lower bound, likewise)
                     x_j + t <= u_j, x_j - t >= l_j   (each finite bound of a free-ranging
                                                       column, l_j < u_j)
                     equality rows and fixed columns as they are, 0 <= t <= 1

   with s_i = max(1, max_j |a_ij|), so t is a slack in the row's own units. t* > 0 means a
   strict interior exists; t* = 0 means at least one inequality is an implicit equality, and
   the rows and bounds whose dual in that LP is nonzero are the ones that cannot be slack.
   HiGHS solves it too: this is a diagnosis of the model, not a claim about our solver, and
   no CSV is written.

    python bench/runners/interior_check.py data/mittelmann/irish-electricity.mps
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))
import rivals  # noqa: E402
from verify_solution_mps import parse_mps  # noqa: E402

VERIFIER = REPO_ROOT / "tools" / "verify_solution.py"
DUAL_FLOOR = 1e-9  # a dual above this in the slack LP marks a constraint that cannot be slack


def highs_point(model: Path, time_limit: float) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        sol = Path(tmp) / "highs.sol"
        out = rivals.run("highs", model, "lp", time_limit, sol)
        print(f"HiGHS: status {out.get('status')}, objective {out.get('objective')!r}, "
              f"{out.get('seconds')} s, version {out.get('version')}")
        if not sol.exists():
            print("HiGHS wrote no point; nothing to verify")
            return
        check = subprocess.run([sys.executable, str(VERIFIER), str(model), str(sol)],
                               capture_output=True, text=True)
        print(f"tools/verify_solution.py on HiGHS's point: exit {check.returncode}")
        print(check.stdout.strip())


def slack_lp(model: Path, time_limit: float) -> None:
    import highspy  # noqa: PLC0415 - installed where this runs (bench.yml)
    import numpy as np  # noqa: PLC0415

    m = parse_mps(model)
    inf = highspy.kHighsInf
    n, rows = m.num_cols, m.num_rows
    t = n  # the slack variable's column
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    lower = [(-inf if not math.isfinite(v) else v) for v in m.col_lower] + [0.0]
    upper = [(inf if not math.isfinite(v) else v) for v in m.col_upper] + [1.0]
    cost = [0.0] * n + [-1.0]  # maximise t
    h.addVars(n + 1, np.array(lower), np.array(upper))
    h.changeColsCost(n + 1, np.arange(n + 1, dtype=np.int32), np.array(cost))
    by_row: list[list[tuple[int, float]]] = [[] for _ in range(rows)]
    for j, column in enumerate(m.entries):
        for i, v in column:
            if v != 0.0:
                by_row[i].append((j, v))
    kinds: list[str] = []
    def add(index: list[int], value: list[float], lo: float, hi: float, kind: str) -> None:
        h.addRow(lo, hi, len(index), np.array(index, dtype=np.int32), np.array(value))
        kinds.append(kind)
    for i in range(rows):
        lo, hi = m.row_lower[i], m.row_upper[i]
        idx = [j for j, _ in by_row[i]]
        val = [v for _, v in by_row[i]]
        scale = max([1.0] + [abs(v) for v in val])
        if lo == hi:
            add(idx, val, lo, hi, f"row {m.row_names[i]} (equality)")
            continue
        if math.isfinite(hi):
            add(idx + [t], val + [scale], -inf, hi, f"row {m.row_names[i]} upper")
        if math.isfinite(lo):
            add(idx + [t], val + [-scale], lo, inf, f"row {m.row_names[i]} lower")
    for j in range(n):
        lo, hi = m.col_lower[j], m.col_upper[j]
        if not lo < hi:
            continue
        if math.isfinite(hi):
            add([j, t], [1.0, 1.0], -inf, hi, f"column {m.col_names[j]} upper")
        if math.isfinite(lo):
            add([j, t], [1.0, -1.0], lo, inf, f"column {m.col_names[j]} lower")
    h.run()
    status = h.modelStatusToString(h.getModelStatus())
    value = h.getSolution().col_value[t] if status == "Optimal" else float("nan")
    print(f"max-min-slack LP: {len(kinds)} rows, status {status}, t* = {value!r} "
          f"({h.getRunTime():.1f} s)")
    if status != "Optimal":
        return
    duals = h.getSolution().row_dual
    tight = [(abs(d), k) for d, k in zip(duals, kinds)
             if abs(d) > DUAL_FLOOR and not k.endswith("(equality)")]
    print(f"inequalities with a dual above {DUAL_FLOOR:g} in it (cannot be slack when "
          f"t* = 0): {len(tight)}")
    for d, k in sorted(tight, reverse=True)[:25]:
        print(f"  {k}: dual {d:.3e}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("models", nargs="+", type=Path)
    parser.add_argument("--time-limit", type=float, default=600.0)
    args = parser.parse_args()
    for model in args.models:
        print(f"== {model}")
        highs_point(model, args.time_limit)
        slack_lp(model, args.time_limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
