"""Scratch, never merged: HiGHS's own KKT measures on Maros-Meszaros QPs beside what
tools/verify_solution.py says of the file the head-to-head harness writes for it."""
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench" / "runners"))
import rivals  # noqa: E402

import highspy  # noqa: E402

INF = 1e29


def own(model: Path) -> None:
    h = highspy.Highs()
    for key, value in (("output_flag", False), ("time_limit", 60.0), ("threads", 1),
                       ("primal_feasibility_tolerance", 1e-7),
                       ("dual_feasibility_tolerance", 1e-7)):
        h.setOptionValue(key, value)
    with tempfile.TemporaryDirectory() as tmp:
        h.readModel(str(rivals._mps_path(model, Path(tmp))))
        h.run()
    info = h.getInfo()
    print("  highs status:", h.modelStatusToString(h.getModelStatus()), "| version", h.version())
    for name in ("objective_function_value", "primal_solution_status", "dual_solution_status",
                 "num_primal_infeasibilities", "max_primal_infeasibility",
                 "sum_primal_infeasibilities", "num_dual_infeasibilities",
                 "max_dual_infeasibility", "sum_dual_infeasibilities",
                 "num_complementarity_violations", "max_complementarity_violation",
                 "primal_dual_objective_error", "qp_iteration_count", "basis_validity"):
        print(f"  highs info {name} = {getattr(info, name, 'n/a')}")
    for name in ("solver", "kkt_tolerance", "optimality_tolerance", "qp_regularization_value",
                 "primal_residual_tolerance", "dual_residual_tolerance"):
        try:
            print(f"  highs option {name} = {h.getOptionValue(name)[1]}")
        except Exception as error:  # noqa: BLE001
            print(f"  highs option {name}: {error}")
    # By index, from HiGHS's own arrays: d = c + Q x - A' y, and its sign against the bounds.
    try:
        lp, sol = h.getLp(), h.getSolution()
        x, d, y = list(sol.col_value), list(sol.col_dual), list(sol.row_dual)
        cost, lo, up = list(lp.col_cost_), list(lp.col_lower_), list(lp.col_upper_)
        n = lp.num_col_
        start, index, value = (list(lp.a_matrix_.start_), list(lp.a_matrix_.index_),
                               list(lp.a_matrix_.value_))
        print("  a_matrix format:", lp.a_matrix_.format_, "| sense:", lp.sense_)
        grad = cost[:]
        hess = h.getModel().hessian_
        hs, hi, hv = list(hess.start_), list(hess.index_), list(hess.value_)
        print("  hessian format:", hess.format_, "dim", hess.dim_, "nnz", len(hv))
        for j in range(hess.dim_):
            for p in range(hs[j], hs[j + 1]):
                i = hi[p]
                grad[i] += hv[p] * x[j]
                if i != j:
                    grad[j] += hv[p] * x[i]
        recomputed = grad[:]
        for j in range(n):
            for p in range(start[j], start[j + 1]):
                recomputed[j] -= value[p] * y[index[p]]
        stationarity = max(abs(recomputed[j] - d[j]) for j in range(n))
        worst, worst_j, comp, comp_j = 0.0, -1, 0.0, -1
        for j in range(n):
            below = d[j] < 0 and (up[j] >= INF or up[j] - x[j] > 1e-7 * max(1.0, abs(up[j])))
            above = d[j] > 0 and (lo[j] <= -INF or x[j] - lo[j] > 1e-7 * max(1.0, abs(lo[j])))
            if (below or above) and abs(d[j]) > worst:
                worst, worst_j = abs(d[j]), j
            slack = min(x[j] - lo[j] if lo[j] > -INF else INF, up[j] - x[j] if up[j] < INF else INF)
            if slack < INF and abs(d[j]) * slack > comp:
                comp, comp_j = abs(d[j]) * slack, j
        name = lambda j: lp.col_names_[j] if j >= 0 else "-"  # noqa: E731
        print(f"  by index: max |c + Qx - A'y - col_dual| = {stationarity:.3e}")
        print(f"  by index: worst reduced cost of the wrong sign off its bound = {worst:.3e} on "
              f"{name(worst_j)} (x {x[worst_j]!r}, [{lo[worst_j]!r}, {up[worst_j]!r}], d {d[worst_j]!r})")
        print(f"  by index: worst |d| * distance to the nearer bound = {comp:.3e} on {name(comp_j)}")
    except Exception as error:  # noqa: BLE001
        print("  by-index check not available:", repr(error))


def harness(model: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        sol = Path(tmp) / "h.sol"
        got = rivals.run("highs", model, "qp", 60.0, sol)
        print("  harness:", {k: got.get(k) for k in ("status", "objective", "duals", "iterations")})
        if sol.exists():
            check = subprocess.run([sys.executable, str(ROOT / "tools" / "verify_solution.py"),
                                    str(model), str(sol)], capture_output=True, text=True)
            for line in check.stdout.splitlines():
                if "[" in line or "VERIFIED" in line or "REJECT" in line:
                    print("  verifier:", line.strip()[:210])


for instance in sys.argv[1:]:
    path = ROOT / "data" / "maros-meszaros" / f"{instance}.qps"
    print(f"== {instance}", flush=True)
    own(path)
    harness(path)
    sys.stdout.flush()
