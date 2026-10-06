#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The other solvers of the head-to-head (#766): HiGHS, SCIP, CBC/Clp and GLPK.

Every rival runs as a SEPARATE PROCESS on the file we solve. HiGHS and SCIP are pip packages
(`highspy`, `pyscipopt`) driven by this same script re-invoked as a child (`--worker`), and
Clp and GLPK are the distribution's binaries (`clp`, `glpsol`). Nothing here is linked into
SANKHYA, no rival's source was read to write it, and every flag used is documented in the
solver's own help output or user manual. See the judgement call in docs/PROVENANCE.md.

What this module adds to compare.py is the step that makes a rival's answer checkable: each
solver's output is converted into the SANKHYA .sol layout (the one tools/verify_solution.py
reads), so the same independent verifier judges every solver. Two rules keep it fair:

*   A rival's own numbers go into the file, never ours: its objective, its activities, its
    duals. A row the rival did not report (a free row its reader drops) is filled with the
    activity A x recomputed from its point and a zero multiplier, and the count is recorded.
*   Duals are claimed only when the solver produced usable ones. SCIP's LP duals are not
    meaningful after its presolve, and it has none at all for a QP it solves as a nonlinear
    program, so an optimal SCIP answer is written with status `feasible`: the verifier then
    checks the point's feasibility and objective and skips the dual conditions, and the
    CSV's `verification` column says `primal-only`.

Every solver gets one thread, the same time limit, and primal and dual feasibility
tolerances of 1e-7, the project's own and the verifier's. That is the default for HiGHS,
Clp and GLPK; SCIP's defaults are 1e-6 and are set to 1e-7 here.
"""
from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SOLVERS = ("highs", "scip", "cbc-clp", "glpk", "osqp", "clarabel", "piqp", "scs")
LABELS = {"sankhya": "SANKHYA", "highs": "HiGHS", "scip": "SCIP", "cbc-clp": "CBC/Clp",
          "glpk": "GLPK", "osqp": "OSQP", "clarabel": "Clarabel", "piqp": "PIQP",
          "scs": "SCS"}
# osqp, clarabel, piqp and scs (#983) are the convex QP comparators Maros-Meszaros is
# published against. Each is a pip package, imported only inside its own worker process
# (QP_PYTHON_SOLVERS), exactly like highspy and pyscipopt above: never a build or test
# dependency of the library. All four are set to the project's own relative tolerance,
# 1e-6, and polishing/refinement left at each package's default (stated per row in
# `solver_options`, #983's acceptance criterion 3) rather than silently retuned to flatter
# one of them.
QP_PYTHON_SOLVERS = ("osqp", "clarabel", "piqp", "scs")
QP_RELATIVE_TOLERANCE = 1e-6
FEASIBILITY_TOLERANCE = 1e-7
# The child is killed this long after its own limit should have stopped it.
HANG_MARGIN_SECONDS = 60.0
# One thread everywhere, including any BLAS or OpenMP a library might start.
SINGLE_THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                     "MKL_NUM_THREADS": "1"}


def supports(solver: str, kind: str) -> bool:
    """GLPK has no quadratic objective. The four QP comparators (#983) are run only where
    the issue asks for them, beside Maros-Meszaros: they are convex-QP solvers, not general
    LP/MIP engines, so they are excluded from the LP and MIP suites rather than run there
    on Q=0 and reported as if that were a fair comparison."""
    if solver == "glpk" and kind == "qp":
        return False
    if solver in QP_PYTHON_SOLVERS and kind != "qp":
        return False
    return True


# =========================================================================================
# The .sol writer: the layout tools/verify_solution_sol.py reads
# =========================================================================================

def _name(name: str) -> str:
    """Quote a name with whitespace, as the .sol layout requires (forplan's `DEDO3 11`)."""
    if any(c.isspace() for c in name) or name.startswith('"'):
        return '"' + name.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return name


def write_sol(path: Path, *, solver: str, status: str, objective: float | None,
              columns: dict[str, tuple[float, float]], rows: dict[str, tuple[float, float]],
              message: str = "") -> None:
    """columns: name -> (value, reduced cost); rows: name -> (activity, multiplier)."""
    lines = [f"# converted from {LABELS.get(solver, solver)} output by bench/runners/rivals.py",
             f"status {status}"]
    if objective is not None:
        lines.append(f"objective {objective!r}")
    if message:
        lines.append(f"message {message}")
    lines.append(f"begin columns {len(columns)}")
    lines += [f"{_name(n)} {v!r} {d!r}" for n, (v, d) in columns.items()]
    lines.append("end columns")
    lines.append(f"begin rows {len(rows)}")
    lines += [f"{_name(n)} {a!r} {y!r}" for n, (a, y) in rows.items()]
    lines.append("end rows")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reconcile(sol_path: Path, model, solver: str) -> list[str]:
    """Bring a converted .sol onto the model's names, and say what was changed.

    *   Clp and GLPK print names with their spaces removed (forplan's `DEDO3 1R` comes back
        `DEDO31R`) and SCIP with underscores (`DEDO3_1R`); a name the model lacks is mapped
        back when exactly one spaced model name has that form.
    *   A row the solver did not report is added at A x from the solver's own point with a
        zero multiplier. SCIP reports no multipliers at all, so this changes nothing a
        verifier checks there; for the others it is counted in the returned notes.
    *   GLPK's MPS reader takes the objective row's right-hand side as the objective
        constant, where the rest of the field (ours, HiGHS, Clp, the verifier) takes its
        negation, so on a model with a constant GLPK states c'x - offset. Its objective is
        restated in the common convention, c'x + offset, and the note says so. Its point
        and multipliers are untouched, and the constant cannot change either.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    from verify_solution_sol import parse_sol  # noqa: PLC0415
    solution = parse_sol(sol_path)
    notes: list[str] = []

    def rename(values: dict, known: dict) -> dict:
        stripped: dict[str, list[str]] = {}
        for name in known:
            if " " in name:
                for form in {name.replace(" ", ""), name.replace(" ", "_")}:
                    stripped.setdefault(form, []).append(name)
        out, mapped = {}, 0
        for name, value in values.items():
            if name not in known and len(stripped.get(name, [])) == 1:
                name, mapped = stripped[name][0], mapped + 1
            out[name] = value
        if mapped:
            notes.append(f"{mapped} names mapped back to their spaced form")
        return out

    columns = rename({n: (solution.col_value[n], solution.col_dual[n])
                      for n in solution.col_value}, model.col_index)
    rows = rename({n: (solution.row_activity[n], solution.row_dual[n])
                   for n in solution.row_activity}, model.row_index)
    missing = [n for n in model.row_names if n not in rows]
    if missing and all(n in columns for n in model.col_names):
        activity = [0.0] * model.num_rows
        for j, name in enumerate(model.col_names):
            x = columns[name][0]
            for i, value in model.entries[j]:
                activity[i] += value * x
        for name in missing:
            rows[name] = (activity[model.row_index[name]], 0.0)
        if solver != "scip":
            notes.append(f"{len(missing)} unreported rows filled at A x")
    objective = solution.header_float("objective")
    if solver == "glpk" and objective is not None and model.objective_offset:
        objective += 2.0 * model.objective_offset
        notes.append(f"objective restated from GLPK's constant convention "
                     f"(constant {-model.objective_offset:g} there, {model.objective_offset:g} "
                     "here)")
    write_sol(sol_path, solver=solver, status=solution.status, objective=objective,
              columns=columns, rows=rows, message=solution.header.get("message", ""))
    return notes


# =========================================================================================
# HiGHS and SCIP: pip packages, each run in a child process of this script
# =========================================================================================

def _mps_path(model: Path, tmp: Path) -> Path:
    """`model` itself when it already ends `.mps`; otherwise a link (a copy where the system
    will not make one, as Windows without developer mode will not) named `.mps`, because
    HiGHS picks its reader by extension and reads nothing from a `.qps` file."""
    if model.suffix.lower() == ".mps":
        return model
    link = tmp / (model.stem + ".mps")
    try:
        link.symlink_to(model.resolve())
    except OSError:
        import shutil  # noqa: PLC0415
        shutil.copyfile(model, link)
    return link


def _worker_highs(model: Path, time_limit: float, sol: Path) -> dict:
    """HiGHS on `model` as the file states it: an LP, a QP, or a MIP when the file has
    INTORG markers (MIPLIB 3, #761), which readModel reads as integrality, so the MIP is
    solved and not its relaxation. mip_feasibility_tolerance, the row and integrality
    tolerance of HiGHS's MIP solver (1e-6 by default), is set to the same 1e-7 as the LP
    tolerances; it does nothing on a model without integers. A MIP has no duals, so its
    point is written `feasible` and verified primal and integral only."""
    import highspy  # noqa: PLC0415 - external solver, never a build dependency
    h = highspy.Highs()
    for key, value in (("output_flag", False), ("time_limit", float(time_limit)),
                       ("threads", 1), ("primal_feasibility_tolerance", FEASIBILITY_TOLERANCE),
                       ("dual_feasibility_tolerance", FEASIBILITY_TOLERANCE),
                       ("mip_feasibility_tolerance", FEASIBILITY_TOLERANCE)):
        h.setOptionValue(key, value)
    with tempfile.TemporaryDirectory() as tmp:
        h.readModel(str(_mps_path(model, Path(tmp))))
        h.run()
    status = h.modelStatusToString(h.getModelStatus()).strip().lower()
    info, lp, solution = h.getInfo(), h.getLp(), h.getSolution()
    mapped = {"optimal": "optimal", "infeasible": "infeasible", "unbounded": "unbounded",
              "time limit reached": "time_limit", "iteration limit reached": "iteration_limit",
              "primal infeasible or unbounded": "infeasible_or_unbounded"}.get(status, status)
    out = {"status": mapped, "message": status, "seconds": h.getRunTime(),
           "iterations": (info.simplex_iteration_count + max(info.ipm_iteration_count, 0)
                          + max(info.qp_iteration_count, 0)),
           "objective": info.objective_function_value if mapped == "optimal" else None,
           "duals": bool(solution.dual_valid), "version": h.version()}
    if mapped == "optimal" and solution.value_valid:
        duals = bool(solution.dual_valid)
        # One copy of each vector: highspy converts the whole vector on every attribute
        # access, so indexing `solution.col_value[j]` in the loop is quadratic - 88 s of
        # wall time around a 1.25 s solve on osa-14, and past the hang margin on osa-30.
        col_value, col_dual = list(solution.col_value), list(solution.col_dual)
        row_value, row_dual = list(solution.row_value), list(solution.row_dual)
        write_sol(sol, solver="highs", status="optimal" if duals else "feasible",
                  objective=out["objective"],
                  columns={n: (col_value[j], col_dual[j] if duals else 0.0)
                           for j, n in enumerate(lp.col_names_)},
                  rows={n: (row_value[i], row_dual[i] if duals else 0.0)
                        for i, n in enumerate(lp.row_names_)})
    return out


def _worker_scip(model: Path, time_limit: float, sol: Path) -> dict:
    import pyscipopt  # noqa: PLC0415 - external solver, never a build dependency
    m = pyscipopt.Model()
    m.hideOutput()
    # SCIP picks its reader by extension and has no `qps` reader; the MPS reader reads the
    # QUADOBJ section.
    m.readProblem(str(model), extension="mps")
    for key, value in (("limits/time", float(time_limit)), ("lp/threads", 1),
                       ("parallel/maxnthreads", 1),
                       ("numerics/feastol", FEASIBILITY_TOLERANCE),
                       ("numerics/dualfeastol", FEASIBILITY_TOLERANCE)):
        m.setParam(key, value)
    m.optimize()
    status = m.getStatus()
    mapped = {"optimal": "optimal", "infeasible": "infeasible", "unbounded": "unbounded",
              "inforunbd": "infeasible_or_unbounded", "timelimit": "time_limit"}.get(status,
                                                                                    status)
    out = {"status": mapped, "message": status, "seconds": m.getSolvingTime(),
           "iterations": m.getNLPIterations(), "duals": False,
           "objective": m.getObjVal() if mapped == "optimal" else None,
           "version": f"{m.getMajorVersion()}.{m.getMinorVersion()}.{m.getTechVersion()}"}
    if mapped == "optimal":
        best = m.getBestSol()
        columns = {v.name: (m.getSolVal(best, v), 0.0) for v in m.getVars()}
        rows = {}
        # The ORIGINAL constraints: presolve upgrades linear rows into other handlers.
        for cons in m.getConss(False):
            if cons.getConshdlrName() == "linear":
                rows[cons.name] = (m.getActivity(cons, best), 0.0)
        # Primal only: `feasible` makes the verifier check the point and skip the duals.
        write_sol(sol, solver="scip", status="feasible", objective=out["objective"],
                  columns=columns, rows=rows, message="primal only; SCIP reports no usable duals")
    return out


def _python_rival(solver: str, model: Path, time_limit: float, sol: Path) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", solver,
                   str(model), repr(float(time_limit)), str(sol), str(report)]
        started = time.perf_counter()
        try:
            done = subprocess.run(command, capture_output=True, text=True,
                                  timeout=time_limit + HANG_MARGIN_SECONDS,
                                  env={**os.environ, **SINGLE_THREAD_ENV})
        except subprocess.TimeoutExpired:
            return {"status": "hung", "wall_seconds": time.perf_counter() - started,
                    "message": f"no exit within {time_limit + HANG_MARGIN_SECONDS:g} s"}
        wall = time.perf_counter() - started
        if not report.exists():
            return {"status": "crashed", "wall_seconds": wall,
                    "message": (done.stderr.strip().splitlines() or ["no output"])[-1][:300]}
        out = json.loads(report.read_text(encoding="utf-8"))
    out["wall_seconds"] = wall
    return out


# =========================================================================================
# Clp: the COIN-OR binary, which is also CBC's LP engine
# =========================================================================================

CLP_STATUS = re.compile(r"^(Optimal|Primal infeasible|Dual infeasible|Stopped on time|"
                        r"Stopped on iterations|Stopped on difficulties|Stopped on ctrl-c)"
                        r".*?(?:objective value\s+(\S+))?\s*$", re.M)
CLP_TIME = re.compile(r"Total time \(CPU seconds\):\s*([\d.eE+-]+)")
CLP_ITERATIONS = re.compile(r"-\s*(\d+) iterations")
CLP_VERSION = re.compile(r"Coin LP version (\S+?),")


def _clp_names(text: str) -> tuple[list[str], list[str]]:
    """Row then column names, in Clp's order, from `-printingOptions all -solu`.

    Each record is `index name value dual`, a leading `**` marking an infeasible entry. The
    index restarts at 0 for the columns. A name can contain a space, so it is everything
    between the index and the last two fields."""
    rows: list[str] = []
    cols: list[str] = []
    target, last = rows, -1
    for line in text.splitlines()[1:]:
        fields = line.replace("**", " ").split()
        if len(fields) < 4 or not fields[0].isdigit():
            continue
        index = int(fields[0])
        if index <= last and target is rows:
            target = cols
        last = index
        target.append(" ".join(fields[1:-2]))
    return rows, cols


def _clp(model: Path, kind: str, time_limit: float, sol: Path) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        names, binary = Path(tmp) / "names.txt", Path(tmp) / "solution.bin"
        # -solve lets Clp choose its algorithm on an LP; a quadratic objective is solved by
        # its barrier, the method its manual names for QPs.
        method = "-barrier" if kind == "qp" else "-solve"
        command = ["clp", str(model.resolve()), "-sec", repr(float(time_limit)),
                   "-primalT", repr(FEASIBILITY_TOLERANCE),
                   "-dualT", repr(FEASIBILITY_TOLERANCE), method,
                   "-printingOptions", "all", "-solu", str(names), "-saveSolution", str(binary)]
        started = time.perf_counter()
        try:
            done = subprocess.run(command, capture_output=True, text=True, cwd=tmp,
                                  timeout=time_limit + HANG_MARGIN_SECONDS,
                                  env={**os.environ, **SINGLE_THREAD_ENV})
        except subprocess.TimeoutExpired:
            return {"status": "hung", "wall_seconds": time.perf_counter() - started,
                    "message": f"no exit within {time_limit + HANG_MARGIN_SECONDS:g} s"}
        wall = time.perf_counter() - started
        text = done.stdout + done.stderr
        matches = CLP_STATUS.findall(text)
        word = matches[-1][0] if matches else ""
        status = {"Optimal": "optimal", "Primal infeasible": "infeasible",
                  "Dual infeasible": "unbounded", "Stopped on time": "time_limit",
                  "Stopped on iterations": "iteration_limit"}.get(word, "numerical_error"
                                                                  if word else "unparsed")
        version = CLP_VERSION.search(text)
        clock = CLP_TIME.search(text)
        iterations = CLP_ITERATIONS.findall(text)
        out = {"status": status, "message": word or text.strip()[-200:], "duals": True,
               "seconds": float(clock.group(1)) if clock else wall, "wall_seconds": wall,
               "iterations": int(iterations[-1]) if iterations else "",
               "version": version.group(1) if version else "", "objective": None}
        if status == "optimal" and binary.exists() and names.exists():
            # saveSolution, as `clp -saveSolution??` documents it: the row and column counts,
            # then doubles - the objective, row activities, row duals, column activities and
            # reduced costs.
            data = binary.read_bytes()
            m, n = struct.unpack_from("ii", data, 0)
            values = struct.unpack_from(f"{1 + 2 * m + 2 * n}d", data, 8)
            row_names, col_names = _clp_names(names.read_text(errors="replace"))
            if len(row_names) != m or len(col_names) != n:
                out["status"] = "unparsed"
                out["message"] = f"names for {len(row_names)}x{len(col_names)}, data {m}x{n}"
                return out
            out["objective"] = values[0]
            act, dual = values[1:1 + m], values[1 + m:1 + 2 * m]
            x, d = values[1 + 2 * m:1 + 2 * m + n], values[1 + 2 * m + n:]
            write_sol(sol, solver="cbc-clp", status="optimal", objective=values[0],
                      columns={c: (x[j], d[j]) for j, c in enumerate(col_names)},
                      rows={r: (act[i], dual[i]) for i, r in enumerate(row_names)})
        return out


# =========================================================================================
# GLPK: glpsol
# =========================================================================================

GLPK_TIME = re.compile(r"Time used:\s*([\d.]+)\s*secs")
GLPK_ITERATIONS = re.compile(r"^[ *]\s*(\d+):", re.M)
GLPK_VERSION = re.compile(r"GLPK LP/MIP Solver\s+(\S+)")


def _glpk_names(report: str) -> tuple[list[str], list[str]]:
    """Row and column names from glpsol's printable report (`-o`), in its order.

    A record starts `%6d %-12s`; a name longer than twelve characters is printed alone and
    its values continue on the next line. Twelve fixed columns keep a name with a space in
    it whole."""
    rows: list[str] = []
    cols: list[str] = []
    target = None
    for line in report.splitlines():
        if line.lstrip().startswith("No.") and "Row name" in line:
            target = rows
            continue
        if line.lstrip().startswith("No.") and "Column name" in line:
            target = cols
            continue
        if target is None or len(line) < 8 or not line[:6].strip().isdigit():
            continue
        if line[6] != " ":
            continue
        if len(line) > 19 and line[19] == " ":
            target.append(line[7:19].rstrip())
        else:
            target.append(line[7:].rstrip())
    return rows, cols


def _glpk(model: Path, time_limit: float, sol: Path) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        raw, report = Path(tmp) / "solution.txt", Path(tmp) / "report.txt"
        command = ["glpsol", "--mps", str(model.resolve()), "--tmlim", str(max(1, int(time_limit))),
                   "-w", str(raw), "-o", str(report)]
        started = time.perf_counter()
        try:
            done = subprocess.run(command, capture_output=True, text=True, cwd=tmp,
                                  timeout=time_limit + HANG_MARGIN_SECONDS)
        except subprocess.TimeoutExpired:
            return {"status": "hung", "wall_seconds": time.perf_counter() - started,
                    "message": f"no exit within {time_limit + HANG_MARGIN_SECONDS:g} s"}
        wall = time.perf_counter() - started
        text = done.stdout + done.stderr
        clock, version = GLPK_TIME.search(text), GLPK_VERSION.search(text)
        iterations = GLPK_ITERATIONS.findall(text)
        out = {"status": "unparsed", "message": text.strip().splitlines()[-1][:200]
               if text.strip() else "", "duals": True, "objective": None,
               "seconds": float(clock.group(1)) if clock else wall, "wall_seconds": wall,
               "iterations": int(iterations[-1]) if iterations else "",
               "version": version.group(1) if version else ""}
        if "TIME LIMIT EXCEEDED" in text:
            out["status"] = "time_limit"
        elif "PROBLEM HAS NO PRIMAL FEASIBLE SOLUTION" in text:
            out["status"] = "infeasible"
        elif "PROBLEM HAS UNBOUNDED SOLUTION" in text:
            out["status"] = "unbounded"
        if not raw.exists():
            if out["status"] == "unparsed":
                out["status"] = "numerical_error" if done.returncode == 0 else "crashed"
            return out
        # glpsol -w, basic solution: `s bas m n pstat dstat obj`, then `i row stat prim dual`
        # and `j col stat prim dual`, rows and columns in the order of the report.
        head, row_vals, col_vals = None, [], []
        for line in raw.read_text().splitlines():
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "s":
                head = fields
            elif fields[0] == "i":
                row_vals.append((float(fields[3]), float(fields[4])))
            elif fields[0] == "j":
                col_vals.append((float(fields[3]), float(fields[4])))
        if head is None:
            return out
        if head[4] == "f" and head[5] == "f":
            out["status"] = "optimal"
        elif out["status"] == "unparsed":
            out["status"] = {"n": "infeasible", "i": "numerical_error"}.get(head[4],
                                                                           "numerical_error")
        if out["status"] != "optimal":
            return out
        out["objective"] = float(head[6])
        row_names, col_names = _glpk_names(report.read_text(errors="replace"))
        if len(row_names) != len(row_vals) or len(col_names) != len(col_vals):
            out["status"] = "unparsed"
            out["message"] = (f"names for {len(row_names)}x{len(col_names)}, "
                              f"data {len(row_vals)}x{len(col_vals)}")
            return out
        write_sol(sol, solver="glpk", status="optimal", objective=out["objective"],
                  columns={c: col_vals[j] for j, c in enumerate(col_names)},
                  rows={r: row_vals[i] for i, r in enumerate(row_names)})
        return out


# =========================================================================================
# osqp, clarabel, piqp and scs: pip packages, each run in a child process of this script
# (#983). None of them is pyscipopt-shaped (they take raw matrices, not a model file), so
# each worker reads the .qps itself with tools/verify_solution_mps.parse_mps - the same
# reader the verifier and compare_suite.py's own grading use, so nothing here can read the
# model differently from the number it is graded against.
# =========================================================================================

def _qp_standard_form(model) -> tuple:
    """(P, q, A, l, u): the general convex-QP form every one of the four comparators
    accepts in some corner of its API - minimize 0.5 x'Px + q'x subject to l <= Ax <= u.

    P is the FULL symmetric Hessian (model.hessian is the lower triangle, #514's QUADOBJ
    convention, mirrored here, not doubled: model.hessian_times already contributes an
    off-diagonal entry to both rows once each, so mirroring is the whole job).
    `A` stacks the model's own rows first, then one identity row per column for its bounds,
    so every one of the four APIs - whether it wants a single general A or a separate box -
    can be fed from the same two matrices; the per-solver code below slices the identity
    block back out where an API asks for box constraints instead of extra rows."""
    import numpy as np  # noqa: PLC0415
    import scipy.sparse as sp  # noqa: PLC0415
    n = model.num_cols
    rows_i, cols_i, vals = [], [], []
    for (r, c), value in model.hessian.items():
        rows_i.append(r); cols_i.append(c); vals.append(value)
        if r != c:
            rows_i.append(c); cols_i.append(r); vals.append(value)
    P = sp.csc_matrix((vals, (rows_i, cols_i)), shape=(n, n)) if vals else sp.csc_matrix((n, n))
    q = np.array(model.col_cost, dtype=float)
    m = model.num_rows
    a_rows, a_cols, a_vals = [], [], []
    for j in range(n):
        for i, value in model.entries[j]:
            a_rows.append(i); a_cols.append(j); a_vals.append(value)
    A_rows = sp.csc_matrix((a_vals, (a_rows, a_cols)), shape=(m, n))
    A_box = sp.identity(n, format="csc")
    A = sp.vstack([A_rows, A_box], format="csc")
    l = np.concatenate([np.array(model.row_lower, dtype=float),
                        np.array(model.col_lower, dtype=float)])
    u = np.concatenate([np.array(model.row_upper, dtype=float),
                        np.array(model.col_upper, dtype=float)])
    return P, q, A, l, u


def _qp_point(model, x) -> tuple[dict, dict]:
    """A solver's own `x` turned into the (activity, 0.0) / (value, 0.0) maps write_sol
    wants. None of the four comparators used here hands back row or column duals in a form
    worth carrying through rivals.reconcile's name games, so the point is written primal
    only, the same choice already made for SCIP above - `verification` is then
    `primal-only` and the duality-gap conditions are skipped rather than claimed."""
    columns = {name: (float(x[j]), 0.0) for j, name in enumerate(model.col_names)}
    activity = [0.0] * model.num_rows
    for j, name in enumerate(model.col_names):
        for i, value in model.entries[j]:
            activity[i] += value * float(x[j])
    rows = {name: (activity[i], 0.0) for i, name in enumerate(model.row_names)}
    return columns, rows


def _read_qp_model(model_path: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    from verify_solution_mps import parse_mps  # noqa: PLC0415
    return parse_mps(model_path)


def _objective(model, x) -> float:
    # float(...): x and model.col_cost can be numpy arrays, and write_sol's `!r` would
    # otherwise print `np.float64(0.25)` into the .sol file - a token the verifier's own
    # parser (and every other solver's row in the same file) does not expect.
    return float(sum(c * v for c, v in zip(model.col_cost, x)) + model.quadratic_objective(x)
                + model.objective_offset)


def _worker_osqp(model_path: Path, time_limit: float, sol: Path) -> dict:
    import osqp  # noqa: PLC0415 - external solver, never a build dependency
    model = _read_qp_model(model_path)
    P, q, A, l, u = _qp_standard_form(model)
    solver = osqp.OSQP()
    settings = {"eps_abs": QP_RELATIVE_TOLERANCE, "eps_rel": QP_RELATIVE_TOLERANCE,
                "time_limit": float(time_limit), "verbose": False, "polish": True}
    solver.setup(P.tocsc(), q, A.tocsc(), l, u, **settings)
    result = solver.solve()
    status = result.info.status
    mapped = {"solved": "optimal", "solved inaccurate": "feasible",
              "primal infeasible": "infeasible", "dual infeasible": "unbounded",
              "maximum iterations reached": "iteration_limit",
              "run time limit reached": "time_limit"}.get(status, status)
    out = {"status": mapped, "message": status, "seconds": result.info.run_time,
           "iterations": result.info.iter, "duals": False, "version": osqp.__version__,
           "solver_options": (f"eps_abs={settings['eps_abs']:g},eps_rel="
                              f"{settings['eps_rel']:g},polish=true"),
           "objective": None}
    if mapped == "optimal" and result.x is not None:
        out["objective"] = _objective(model, result.x)
        columns, rows = _qp_point(model, result.x)
        write_sol(sol, solver="osqp", status="feasible", objective=out["objective"],
                 columns=columns, rows=rows, message="primal only; see rivals.py _qp_point")
    return out


def _worker_clarabel(model_path: Path, time_limit: float, sol: Path) -> dict:
    import clarabel  # noqa: PLC0415 - external solver, never a build dependency
    model = _read_qp_model(model_path)
    P, q, A_box, l, u = _qp_standard_form(model)
    # Clarabel's cone form is A x + s = b, s in the nonnegative cone split in two halves:
    # Ax <= u becomes (A)x + s = u, and -Ax <= -l becomes (-A)x + s = -l; a row with
    # l == u (an equality) is folded into the ZeroConeT half instead of two inequalities.
    import numpy as np  # noqa: PLC0415
    import scipy.sparse as sp  # noqa: PLC0415
    eq = l == u
    A_eq, b_eq = A_box[eq], l[eq]
    A_ineq = sp.vstack([A_box[~eq], -A_box[~eq]], format="csc")
    b_ineq = np.concatenate([u[~eq], -l[~eq]])
    big = 1e20
    keep = np.isfinite(b_ineq) & (np.abs(b_ineq) < big)
    A_stack = sp.vstack([A_eq, A_ineq[keep]], format="csc")
    b_stack = np.concatenate([b_eq, b_ineq[keep]])
    cones = [clarabel.ZeroConeT(int(eq.sum())), clarabel.NonnegativeConeT(int(keep.sum()))]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.time_limit = float(time_limit)
    settings.tol_feas = QP_RELATIVE_TOLERANCE
    settings.tol_gap_abs = QP_RELATIVE_TOLERANCE
    settings.tol_gap_rel = QP_RELATIVE_TOLERANCE
    solver = clarabel.DefaultSolver(P.tocsc(), q, A_stack, b_stack, cones, settings)
    solution = solver.solve()
    status = str(solution.status)
    mapped = {"Solved": "optimal", "PrimalInfeasible": "infeasible",
              "DualInfeasible": "unbounded", "MaxIterations": "iteration_limit",
              "MaxTime": "time_limit"}.get(status, status)
    out = {"status": mapped, "message": status, "seconds": solution.solve_time,
           "iterations": solution.iterations, "duals": False,
           "version": clarabel.__version__,
           "solver_options": f"tol_feas={QP_RELATIVE_TOLERANCE:g}", "objective": None}
    if mapped == "optimal" and solution.x is not None:
        out["objective"] = _objective(model, solution.x)
        columns, rows = _qp_point(model, solution.x)
        write_sol(sol, solver="clarabel", status="feasible", objective=out["objective"],
                 columns=columns, rows=rows, message="primal only; see rivals.py _qp_point")
    return out


def _worker_piqp(model_path: Path, time_limit: float, sol: Path) -> dict:
    import piqp  # noqa: PLC0415 - external solver, never a build dependency
    model = _read_qp_model(model_path)
    P, q, A_box, l, u = _qp_standard_form(model)
    import numpy as np  # noqa: PLC0415
    n = model.num_cols
    A_rows, l_rows, u_rows = A_box[:model.num_rows], l[:model.num_rows], u[:model.num_rows]
    x_l, x_u = l[model.num_rows:], u[model.num_rows:]
    solver = piqp.SparseSolver()
    solver.settings.eps_abs = QP_RELATIVE_TOLERANCE
    solver.settings.eps_rel = QP_RELATIVE_TOLERANCE
    solver.settings.verbose = False
    solver.setup(P.tocsc(), q, G=A_rows.tocsc(), h_l=l_rows, h_u=u_rows, x_l=x_l, x_u=x_u)
    status = solver.solve()
    mapped = {piqp.PIQP_SOLVED: "optimal", piqp.PIQP_PRIMAL_INFEASIBLE: "infeasible",
              piqp.PIQP_DUAL_INFEASIBLE: "unbounded",
              piqp.PIQP_MAX_ITER_REACHED: "iteration_limit"}.get(status, str(status))
    info = solver.result.info
    out = {"status": mapped, "message": str(status), "seconds": info.run_time,
           "iterations": info.iter, "duals": False, "version": piqp.__version__,
           "solver_options": f"eps_abs={QP_RELATIVE_TOLERANCE:g}", "objective": None}
    if mapped == "optimal":
        x = np.asarray(solver.result.x).ravel()
        out["objective"] = _objective(model, x)
        columns, rows = _qp_point(model, x)
        write_sol(sol, solver="piqp", status="feasible", objective=out["objective"],
                 columns=columns, rows=rows, message="primal only; see rivals.py _qp_point")
    return out


def _worker_scs(model_path: Path, time_limit: float, sol: Path) -> dict:
    import scs  # noqa: PLC0415 - external solver, never a build dependency
    model = _read_qp_model(model_path)
    P, q, A_box, l, u = _qp_standard_form(model)
    import numpy as np  # noqa: PLC0415
    import scipy.sparse as sp  # noqa: PLC0415
    # SCS's box cone (>= 3.0) is {(t, s): t*l <= s <= t*u}, one dimension larger than the
    # number of rows it bounds: `t` is its own extra row, fixed to 1 by a zero row of A and
    # b=1 (SCS's own box-cone example fixes it the same way). Equalities go in the zero
    # cone first, same split clarabel's worker above makes.
    eq = l == u
    box = ~eq
    n_box = int(box.sum())
    zero_row = sp.csc_matrix((1, A_box.shape[1]))
    # s = b - Ax must land in [l, u] for the box rows; A x + s = b with b = 0 gives
    # s = -Ax, so the box block's A is negated here (s = -(-Ax) = Ax then lands in [l, u]
    # directly) rather than flipping l and u, which is easier to misread.
    A_stack = sp.vstack([A_box[eq], zero_row, -A_box[box]], format="csc")
    b_stack = np.concatenate([l[eq], [1.0], np.zeros(n_box)])
    data = {"P": P.tocsc(), "A": A_stack, "b": b_stack, "c": q}
    cone = {"z": int(eq.sum()), "bl": l[box], "bu": u[box]}
    solver = scs.SCS(data, cone, eps_abs=QP_RELATIVE_TOLERANCE, eps_rel=QP_RELATIVE_TOLERANCE,
                     time_limit_secs=float(time_limit), verbose=False)
    result = solver.solve()
    status = result["info"]["status"]
    mapped = {"solved": "optimal", "solved/inaccurate": "feasible",
              "infeasible": "infeasible", "unbounded": "unbounded",
              "time limit reached": "time_limit"}.get(status, status)
    out = {"status": mapped, "message": status, "seconds": result["info"]["solve_time"] / 1000.0,
           "iterations": result["info"]["iter"], "duals": False, "version": scs.__version__,
           "solver_options": f"eps_abs={QP_RELATIVE_TOLERANCE:g}", "objective": None}
    if mapped == "optimal":
        x = result["x"]
        out["objective"] = _objective(model, x)
        columns, rows = _qp_point(model, x)
        write_sol(sol, solver="scs", status="feasible", objective=out["objective"],
                 columns=columns, rows=rows, message="primal only; see rivals.py _qp_point")
    return out


QP_WORKERS = {"osqp": _worker_osqp, "clarabel": _worker_clarabel, "piqp": _worker_piqp,
             "scs": _worker_scs}


def _python_qp_rival(solver: str, model: Path, time_limit: float, sol: Path) -> dict:
    """Same subprocess/timeout harness as _python_rival above, with one addition the issue
    asks for explicitly: a package that is not installed is reported as `not_installed`,
    named, in the row the CSV gets - never silently dropped from the suite."""
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--qp-worker", solver,
                   str(model), repr(float(time_limit)), str(sol), str(report)]
        started = time.perf_counter()
        try:
            done = subprocess.run(command, capture_output=True, text=True,
                                  timeout=time_limit + HANG_MARGIN_SECONDS,
                                  env={**os.environ, **SINGLE_THREAD_ENV})
        except subprocess.TimeoutExpired:
            return {"status": "hung", "wall_seconds": time.perf_counter() - started,
                    "message": f"no exit within {time_limit + HANG_MARGIN_SECONDS:g} s"}
        wall = time.perf_counter() - started
        if not report.exists():
            stderr = done.stderr.strip()
            if f"No module named '{solver}'" in stderr:
                return {"status": "not_installed", "wall_seconds": wall,
                        "message": f"{LABELS[solver]} ({solver}) is not installed; "
                                   f"skipped by name, not silently (pip install {solver})"}
            return {"status": "crashed", "wall_seconds": wall,
                    "message": (stderr.splitlines() or ["no output"])[-1][:300]}
        out = json.loads(report.read_text(encoding="utf-8"))
    out["wall_seconds"] = wall
    return out


def _qp_worker(argv: list[str]) -> int:
    solver, model, time_limit, sol, report = argv
    out = QP_WORKERS[solver](Path(model), float(time_limit), Path(sol))
    Path(report).write_text(json.dumps(out), encoding="utf-8")
    return 0


# =========================================================================================
# The entry point compare.py calls
# =========================================================================================

def run(solver: str, model: Path, kind: str, time_limit: float, sol: Path) -> dict:
    """Solve `model` with `solver` in a separate process and write its answer to `sol`.

    Returns status, objective, seconds (the solver's own clock), wall_seconds (the child
    process), iterations, duals, version, message. `sol` exists only when the status is
    optimal and the answer could be converted."""
    if not supports(solver, kind):
        return {"status": "unsupported", "message": "no quadratic objective in GLPK",
                "seconds": None, "wall_seconds": None, "duals": False}
    if solver in ("highs", "scip"):
        return _python_rival(solver, model, time_limit, sol)
    if solver == "cbc-clp":
        return _clp(model, kind, time_limit, sol)
    if solver == "glpk":
        return _glpk(model, time_limit, sol)
    if solver in QP_PYTHON_SOLVERS:
        return _python_qp_rival(solver, model, time_limit, sol)
    raise ValueError(f"unknown solver {solver}")


def _worker(argv: list[str]) -> int:
    solver, model, time_limit, sol, report = argv
    runner = {"highs": _worker_highs, "scip": _worker_scip}[solver]
    out = runner(Path(model), float(time_limit), Path(sol))
    Path(report).write_text(json.dumps(out), encoding="utf-8")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 7 and sys.argv[1] == "--worker":
        sys.exit(_worker(sys.argv[2:]))
    if len(sys.argv) == 7 and sys.argv[1] == "--qp-worker":
        sys.exit(_qp_worker(sys.argv[2:]))
    print(__doc__)
    sys.exit(2)
