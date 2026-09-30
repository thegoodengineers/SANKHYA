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

SOLVERS = ("highs", "scip", "cbc-clp", "glpk")
LABELS = {"sankhya": "SANKHYA", "highs": "HiGHS", "scip": "SCIP", "cbc-clp": "CBC/Clp",
          "glpk": "GLPK"}
FEASIBILITY_TOLERANCE = 1e-7
# The child is killed this long after its own limit should have stopped it.
HANG_MARGIN_SECONDS = 60.0
# One thread everywhere, including any BLAS or OpenMP a library might start.
SINGLE_THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                     "MKL_NUM_THREADS": "1"}


def supports(solver: str, kind: str) -> bool:
    """GLPK has no quadratic objective; everything else takes both LPs and QPs."""
    return not (solver == "glpk" and kind == "qp")


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

def _worker_highs(model: Path, time_limit: float, sol: Path) -> dict:
    import highspy  # noqa: PLC0415 - external solver, never a build dependency
    h = highspy.Highs()
    for key, value in (("output_flag", False), ("time_limit", float(time_limit)),
                       ("threads", 1), ("primal_feasibility_tolerance", FEASIBILITY_TOLERANCE),
                       ("dual_feasibility_tolerance", FEASIBILITY_TOLERANCE)):
        h.setOptionValue(key, value)
    with tempfile.TemporaryDirectory() as tmp:
        # HiGHS picks its reader by extension and reads nothing from a `.qps` file.
        link = Path(tmp) / (model.stem + ".mps")
        link.symlink_to(model.resolve())
        h.readModel(str(link))
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
    print(__doc__)
    sys.exit(2)
