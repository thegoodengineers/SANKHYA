# SPDX-License-Identifier: Apache-2.0
"""SANKHYA web board: a thin HTTP layer over the command-line solver.

Nothing here solves anything. Every number the page shows comes from one of four places,
each of which exists independently of this file:

  * the solver's own log (``sankhya solve --option log_level=verbose``), streamed line by
    line as it is written;
  * the JSON result blob the solver writes with ``--stats``;
  * the ``.sol`` file it writes with ``--write-sol``, which is what the independent checker
    ``tools/verify_solution.py`` reads and recomputes;
  * HiGHS, through ``highspy``, as the reference the problem statement asks us to compare
    against.

The page is a window onto those, not a second implementation of any of them. Run it with

    python apps/web/server.py            # http://127.0.0.1:8010

The solver binary is found the same way demo/run_sih_demo.sh finds it, or named by
``SANKHYA_BIN``.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STATIC = HERE / "public"
ON_VERCEL = bool(os.environ.get("VERCEL")) or not (REPO / "tools").is_dir()
# On Vercel the function bundle holds a static Linux binary plus vendored copies of the
# checker and the demo models (see vendor.py); the only writable directory is /tmp.
TOOLS = HERE / "vendor" / "tools" if ON_VERCEL else REPO / "tools"
DEMO = HERE / "vendor" / "demo" if ON_VERCEL else REPO / "demo"
UPLOADS = Path(tempfile.gettempdir()) / "sankhya-web-uploads"
UPLOADS.mkdir(exist_ok=True)

app = FastAPI(title="SANKHYA board")
# The page may be served from elsewhere (a static host) and call this server by URL.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"],
                   allow_headers=["*"])


def find_binary() -> Path:
    env = os.environ.get("SANKHYA_BIN")
    if env:
        return Path(env)
    if ON_VERCEL:
        # The bundle is read-only and may drop the execute bit; run a copy from /tmp.
        src = HERE / "api" / "bin" / "sankhya"
        dst = Path(tempfile.gettempdir()) / "sankhya"
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copyfile(src, dst)
            dst.chmod(0o755)
        return dst
    for candidate in ("build/sankhya", "build/sankhya.exe", "build/Release/sankhya.exe",
                      "build-dbg/sankhya", "build-dbg/sankhya.exe"):
        p = REPO / candidate
        if p.is_file():
            return p
    raise RuntimeError("no solver binary: build first (scripts/configure.sh build Release && "
                       "cmake --build build -j) or set SANKHYA_BIN")


# The built-in models are the demo's three case studies plus the two that show a proof and
# a quadratic objective. Options listed here are what the model NEEDS to be read at all,
# not tuning: the pooling model has bilinear constraints and is refused without the global
# method, and that refusal is deliberate (see the reader's error message).
BUILTIN = [
    {"id": "crude_blend", "file": "crude_blend.mps", "title": "Crude blend LP",
     "blurb": "Three crudes, a throughput cap, a diesel floor, a sulphur ceiling. "
              "Maximise margin. The model every algorithm should agree on.", "options": []},
    {"id": "blend_milp", "file": "blend_milp.mps", "title": "Blend MILP",
     "blurb": "The same blend with a whole-cargo decision. Branch and bound closes the gap.",
     "options": []},
    {"id": "crude_blend_qp", "file": "crude_blend_qp.mps", "title": "Blend QP",
     "blurb": "A quadratic penalty on the mix. Solved by the convex QP engine.", "options": []},
    {"id": "crude_blend_infeasible", "file": "crude_blend_infeasible.mps",
     "title": "Infeasible blend",
     "blurb": "Demands that cannot be met. The solver proves it with a certificate and names "
              "the smallest set of constraints that conflict.", "options": []},
    {"id": "pooling_haverly", "file": "pooling_haverly.mps", "title": "Haverly pooling",
     "blurb": "The classic nonconvex pooling problem, bilinear in the pool quality. Solved to "
              "global optimality.", "options": ["nonconvex=global"]},
]

ALGORITHMS = [
    {"id": "auto", "title": "Automatic", "option": None},
    {"id": "dual-simplex", "title": "Dual simplex", "option": "algorithm=dual-simplex"},
    {"id": "simplex", "title": "Primal simplex", "option": "algorithm=simplex"},
    {"id": "ipm", "title": "Interior point", "option": "algorithm=ipm"},
    {"id": "pdhg", "title": "PDHG (first order)", "option": "algorithm=pdhg"},
]


def resolve_model(model_id: str) -> tuple[Path, list[str], str]:
    for m in BUILTIN:
        if m["id"] == model_id:
            return DEMO / m["file"], list(m["options"]), m["title"]
    if re.fullmatch(r"upload-[0-9a-f]{12}", model_id):
        matches = list(UPLOADS.glob(model_id + ".*"))
        if matches:
            return matches[0], [], matches[0].name.split(".", 1)[1]
    raise HTTPException(404, f"unknown model {model_id!r}")


def run_info(path: Path, options: list[str]) -> dict:
    cmd = [str(find_binary()), "info", str(path)]
    for o in options:
        cmd += ["--option", o]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(HERE), timeout=60)
    text = out.stdout + out.stderr
    info = {"raw": text.strip(), "ok": out.returncode == 0}
    # `sankhya info` prints a key-value table: "rows   3136  (equality 3136, ...)".
    for key, field in (("rows", "rows"), ("columns", "columns"), ("nonzeros", "nonzeros")):
        m = re.search(rf"^{key}\s+(\d+)", text, re.M)
        if m:
            info[field] = int(m[1])
    m = re.search(r"integer (\d+)", text)
    if m:
        info["integer_columns"] = int(m[1])
    m = re.search(r"^problem class\s+(\S+)", text, re.M)
    if m:
        info["problem_class"] = m[1]
    m = re.search(r"^name\s+(\S+)", text, re.M)
    if m:
        info["name"] = m[1]
    return info


@app.get("/")
def index():
    # The page is edited often while the demo is being prepared; never serve a stale copy.
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/meta")
def meta():
    version = subprocess.run([str(find_binary()), "version"], capture_output=True, text=True,
                             cwd=str(HERE)).stdout.strip()
    try:
        import highspy  # noqa: F401
        highs = True
    except ImportError:
        highs = False
    return {"version": version, "models": BUILTIN, "algorithms": ALGORITHMS,
            "highs_available": highs}


@app.get("/api/info/{model_id}")
def info(model_id: str):
    path, options, title = resolve_model(model_id)
    d = run_info(path, options)
    d["title"] = title
    d["source"] = "demo/" + path.name if path.parent == DEMO else path.name
    return d


@app.post("/api/upload")
async def upload(file: UploadFile):
    name = Path(file.filename or "model.mps").name
    if not name.lower().endswith((".mps", ".lp")):
        raise HTTPException(400, "upload an .mps or .lp file")
    model_id = "upload-" + uuid.uuid4().hex[:12]
    dest = UPLOADS / f"{model_id}.{name}"
    data = await file.read()
    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(413, "50 MB limit")
    dest.write_bytes(data)
    return {"id": model_id, "title": name}


ITER_LINE = re.compile(r"^\s*(\d+)\s+(-?[\d.]+e[+-]\d+)\s+(-?[\d.]+e[+-]\d+|-)"
                       r"\s+(-?[\d.]+e[+-]\d+|-)\s+([\d.]+)s\s*$")


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def parse_sol(path: Path) -> dict:
    """The .sol file's column and row blocks, by name. Same layout the verifier reads."""
    cols, rows, header, farkas, iis = [], [], {}, [], []
    section = None
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("begin columns"):
            section = "columns"; continue
        if line.startswith("begin rows"):
            section = "rows"; continue
        if line.startswith("begin farkas"):
            section = "farkas"; continue
        if line.startswith("begin iis "):
            section = "iis"; continue
        if line.startswith("begin "):
            section = "other"; continue
        if line.startswith("end "):
            section = None; continue
        parts = line.split()
        if section == "columns" and len(parts) >= 3:
            cols.append({"name": parts[0], "value": float(parts[1]),
                         "reduced_cost": float(parts[2]),
                         "basis": parts[3] if len(parts) > 3 else ""})
        elif section == "rows" and len(parts) >= 3:
            rows.append({"name": parts[0], "activity": float(parts[1]),
                         "dual": float(parts[2]), "basis": parts[3] if len(parts) > 3 else ""})
        elif section == "farkas" and len(parts) == 2:
            farkas.append({"row": parts[0], "multiplier": float(parts[1])})
        elif section == "iis" and len(parts) == 2:
            iis.append({"kind": parts[0], "name": parts[1]})
        elif section is None and len(parts) >= 2:
            header[parts[0]] = " ".join(parts[1:])
    return {"columns": cols, "rows": rows, "header": header, "farkas": farkas, "iis": iis}


@app.get("/api/solve/{model_id}")
async def solve(model_id: str, algorithm: str = "auto", time_limit: float = 60.0):
    path, options, _ = resolve_model(model_id)
    alg = next((a for a in ALGORITHMS if a["id"] == algorithm), None)
    if alg is None:
        raise HTTPException(400, f"unknown algorithm {algorithm!r}")
    if alg["option"]:
        options.append(alg["option"])

    work = Path(tempfile.mkdtemp(prefix="sankhya-solve-"))
    stats_path, sol_path, prog_path = work / "stats.json", work / "out.sol", work / "prog.jsonl"
    cmd = [str(find_binary()), "solve", str(path), "--option", "log_level=verbose",
           "--time-limit", str(time_limit), "--stats", str(stats_path),
           "--write-sol", str(sol_path), "--progress-out", str(prog_path)]
    for o in options:
        cmd += ["--option", o]

    async def stream():
        yield sse("start", {"command": " ".join(Path(c).name if i == 0 else c
                                                for i, c in enumerate(cmd)),
                            "model": path.name, "algorithm": algorithm})
        t0 = time.perf_counter()
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, cwd=str(HERE))
        prog_seen = 0
        assert proc.stdout is not None
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            yield sse("log", {"t": round(time.perf_counter() - t0, 3), "line": line})
            m = ITER_LINE.match(line)
            if m:
                yield sse("iter", {"iteration": int(m[1]), "objective": float(m[2]),
                                   "primal_inf": None if m[3] == "-" else float(m[3]),
                                   "dual_inf": None if m[4] == "-" else float(m[4]),
                                   "seconds": float(m[5])})
            # Branch and bound progress is written as JSON lines by the solver itself.
            if prog_path.exists():
                lines = prog_path.read_text().splitlines()
                for p in lines[prog_seen:]:
                    try:
                        yield sse("progress", json.loads(p))
                    except json.JSONDecodeError:
                        pass
                prog_seen = len(lines)
        rc = await proc.wait()
        if prog_path.exists():
            for p in prog_path.read_text().splitlines()[prog_seen:]:
                try:
                    yield sse("progress", json.loads(p))
                except json.JSONDecodeError:
                    pass

        result = {"exit_code": rc, "wall_seconds": round(time.perf_counter() - t0, 4)}
        if stats_path.exists():
            result["stats"] = json.loads(stats_path.read_text())
        if sol_path.exists():
            result["solution"] = parse_sol(sol_path)
        yield sse("result", result)

        # The independent check. It links none of our C++ and parses the model itself.
        if sol_path.exists() and path.suffix.lower() == ".mps":
            vcmd = [sys.executable, str(TOOLS / "verify_solution.py"), str(path),
                    str(sol_path)]
            vproc = await asyncio.create_subprocess_exec(
                *vcmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                cwd=str(HERE))
            checks = []
            assert vproc.stdout is not None
            while True:
                raw = await vproc.stdout.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                yield sse("verify_log", {"line": line})
                m = re.match(r"\s*\[(PASS|FAIL|SKIP|WARN)\]\s+(.+?)\s{2,}(.*)$", line)
                if m:
                    checks.append({"status": m[1], "check": m[2].strip(), "detail": m[3].strip()})
            vrc = await vproc.wait()
            yield sse("verify", {"exit_code": vrc, "verified": vrc == 0, "checks": checks})
        else:
            yield sse("verify", {"exit_code": None, "verified": None, "checks": [],
                                 "note": "the independent checker reads MPS only"})
        yield sse("end", {})
        shutil.rmtree(work, ignore_errors=True)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/highs/{model_id}")
def highs(model_id: str, time_limit: float = 60.0):
    """HiGHS on the same file, the way bench/runners/cross_check_highs.py calls it."""
    path, options, _ = resolve_model(model_id)
    try:
        import highspy
    except ImportError:
        return {"available": False, "reason": "highspy is not installed"}
    if "nonconvex=global" in options:
        return {"available": False,
                "reason": "HiGHS does not read bilinear constraints; no reference for pooling"}
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    t0 = time.perf_counter()
    status = h.readModel(str(path))
    if str(status).endswith("kError"):
        return {"available": False, "reason": "HiGHS could not read this model"}
    h.run()
    seconds = time.perf_counter() - t0
    info = h.getInfo()
    status = str(h.getModelStatus()).split(".")[-1]
    status = re.sub(r"^k", "", status).lower()          # highspy spells it HighsModelStatus.kOptimal
    counts = {"simplex": info.simplex_iteration_count, "ipm": info.ipm_iteration_count,
              "nodes": info.mip_node_count}
    counts = {k: v for k, v in counts.items() if v is not None and v >= 0}
    return {"available": True, "status": status,
            "objective": info.objective_function_value, "seconds": round(seconds, 4),
            "iterations": max(counts.values(), default=None), "counts": counts}


if __name__ == "__main__":
    import uvicorn
    find_binary()
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8010")))
