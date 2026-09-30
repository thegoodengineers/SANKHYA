#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One solve by a commercial edition's published Python package, in its own process (#533).

    python commercial_child.py cplex|gurobi|xpress <model file> <time limit s>

Prints one JSON object on stdout: status, objective, nodes, iterations, version and the
model's size as the package read it. It is started by commercial_agreement.py with the
interpreter of the virtualenv that holds the package (`--python`), so no commercial code is
ever imported into the runner's process, let alone linked into SANKHYA. Only each package's
documented public API is called; nothing of any solver's source is read.
"""
from __future__ import annotations

import json
import sys


def cplex_solve(path: str, limit: float) -> dict:
    import cplex

    c = cplex.Cplex()
    for stream in (c.set_results_stream, c.set_log_stream, c.set_warning_stream,
                   c.set_error_stream):
        stream(None)
    c.parameters.timelimit.set(limit)
    c.parameters.threads.set(1)
    c.read(path, "mps")
    out = {"version": c.get_version(), "columns": c.variables.get_num(),
           "rows": c.linear_constraints.get_num()}
    c.solve()
    s = c.solution
    code = s.get_status()
    mip = c.problem_type[c.get_problem_type()] in ("MILP", "MIQP", "MIQCP")
    optimal = {s.status.optimal, s.status.MIP_optimal, s.status.optimal_tolerance}
    infeasible = {s.status.infeasible, s.status.MIP_infeasible}
    unbounded = {s.status.unbounded, s.status.MIP_unbounded}
    status = ("optimal" if code in optimal else "infeasible" if code in infeasible
              else "unbounded" if code in unbounded else s.get_status_string(code))
    out["status"] = status
    try:
        out["objective"] = s.get_objective_value()
    except cplex.exceptions.CplexError:
        out["objective"] = None
    out["nodes"] = s.progress.get_num_nodes_processed() if mip else 0
    out["iterations"] = s.progress.get_num_iterations()
    return out


def gurobi_solve(path: str, limit: float) -> dict:
    import gurobipy as gp

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.read(path, env=env)
    m.Params.TimeLimit = limit
    m.Params.Threads = 1
    out = {"version": ".".join(map(str, gp.gurobi.version())), "columns": m.NumVars,
           "rows": m.NumConstrs}
    m.optimize()
    names = {gp.GRB.OPTIMAL: "optimal", gp.GRB.INFEASIBLE: "infeasible",
             gp.GRB.UNBOUNDED: "unbounded", gp.GRB.INF_OR_UNBD: "infeasible_or_unbounded",
             gp.GRB.TIME_LIMIT: "time_limit"}
    out["status"] = names.get(m.Status, f"status_{m.Status}")
    out["objective"] = m.ObjVal if m.SolCount > 0 else None
    out["nodes"] = int(m.NodeCount) if m.IsMIP else 0
    out["iterations"] = int(m.IterCount)
    return out


def xpress_solve(path: str, limit: float) -> dict:
    import xpress as xp

    p = xp.problem()
    p.setControl({"outputlog": 0, "timelimit": limit, "threads": 1})
    p.readProb(path)
    out = {"version": xp.getversion(), "columns": p.attributes.cols,
           "rows": p.attributes.rows}
    p.optimize()
    status = str(p.attributes.solstatus).split(".")[-1].lower()
    out["status"] = status
    has_point = status in ("optimal", "feasible")
    out["objective"] = p.attributes.objval if has_point else None
    out["nodes"] = int(p.attributes.nodes)
    out["iterations"] = int(p.attributes.simplexiter)
    return out


SOLVERS = {"cplex": cplex_solve, "gurobi": gurobi_solve, "xpress": xpress_solve}


def main() -> int:
    solver, path, limit = sys.argv[1], sys.argv[2], float(sys.argv[3])
    try:
        out = SOLVERS[solver](path, limit)
    except ImportError as error:
        out = {"status": "not_installed", "message": str(error)}
    except Exception as error:  # the package's own error, e.g. a size-limit refusal
        out = {"status": "error", "message": f"{type(error).__name__}: {error}"[:300]}
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
