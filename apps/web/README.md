# SANKHYA board

A web page that runs the solver in front of an audience. It is a window, not a solver:
every number on it comes from the command-line binary, from the independent checker, or
from HiGHS, and the page shows what they wrote.

Live: https://sankhya-board.vercel.app

## Run it locally

```bash
scripts/configure.sh build Release && cmake --build build -j     # once
pip install fastapi uvicorn python-multipart highspy              # highspy is optional
python apps/web/server.py                                         # http://127.0.0.1:8010
```

`SANKHYA_BIN` names a specific binary; otherwise the same search as `demo/run_sih_demo.sh`.
`PORT` and `HOST` change where it listens.

## What the page shows

Three panels, left to right.

**Model.** The five demo models (crude blend LP, blend MILP, blend QP, the infeasible
blend, Haverly pooling) or an uploaded `.mps` / `.lp`. Dimensions from `sankhya info`.
An algorithm picker for pure LPs: automatic, dual simplex, primal simplex, interior point,
PDHG. MILP, QP and pooling go to their own engines, so the picker is disabled for them.

**Solve.** The solver's log, streamed line by line as `sankhya solve --option
log_level=verbose` prints it. The iteration table in that log is parsed into two charts:
objective by iteration, and primal and dual infeasibility on a log scale. For a branch and
bound the first chart switches to incumbent and best bound by time, read from the JSON lines
the solver writes with `--progress-out`.

**Result.** One row per solver: SANKHYA's engine, then HiGHS through `highspy` on the same
file, the way `bench/runners/cross_check_highs.py` calls it. A line says whether the
objectives agree. Below it, `tools/verify_solution.py` run on the `.sol` file the solver
wrote, one line per check, and a badge that is green only when its exit code is 0. Then the
Farkas certificate and the irreducible infeasible subsystem for an infeasible model, the
variable values and reduced costs, and the row activities and duals (shadow prices, for a
refinery model). A download button gives the whole run as JSON.

## What it deliberately does not do

- No tuning sliders. A setting that breaks a model in front of a jury is not a feature.
- No GPU toggle. The CUDA path is measured on a card or it is not shown.
- No solver logic in JavaScript. The chart is drawn from the log lines; if the log and the
  chart ever disagree, the log is right.
- No claim of speed. Where HiGHS is faster, the table says so. The problem statement asks
  for a comparison.

## Layout

| | |
|---|---|
| `server.py` | the FastAPI app; also the local entry point |
| `public/index.html` | the page, one file, no build step, no CDN |
| `api/index.py` | Vercel's entry point; imports the same app |
| `api/bin/sankhya` | a fully static Linux build of the solver, for the Vercel function; not committed, built by CI or by hand |
| `vendor/` | copies of `tools/verify_solution*.py` and `demo/*.mps`, made by `vendor.py` |
| `Dockerfile`, `fly.toml` | the same backend as a container, for a host that runs one |

## Endpoints

| | |
|---|---|
| `GET /api/meta` | version, built-in models, algorithms, whether highspy is present |
| `GET /api/info/{model}` | dimensions and class from `sankhya info` |
| `POST /api/upload` | an `.mps` or `.lp` file; returns an id usable as `{model}` |
| `GET /api/solve/{model}?algorithm=` | server-sent events: `start`, `log`, `iter`, `progress`, `result`, `verify_log`, `verify`, `end` |
| `GET /api/highs/{model}` | HiGHS on the same file |

A page link with `?model=<id>` preselects a model. `?api=<url>` points the page at a
backend on another origin and is remembered in that browser.

## Deploying to Vercel

The whole thing, solver included, runs on Vercel as one Python serverless function.
`.github/workflows/web-board.yml` builds the binary on every push that touches the solver
or the board and uploads it as the `sankhya-linux-static` artifact; on pushes to main it
also deploys, if the repository has `VERCEL_TOKEN`, `VERCEL_ORG_ID` and `VERCEL_PROJECT_ID`
as secrets (the two ids are in `.vercel/project.json` after `vercel link`). Without the
token the deploy job is skipped and CI stays green.

By hand, the binary is built in WSL or on any Linux with GCC 10 or newer:

```bash
cmake -B build-linux -DCMAKE_BUILD_TYPE=Release -DSANKHYA_BUILD_TESTS=OFF \
      -DSANKHYA_WITH_OPENMP=OFF -DCMAKE_EXE_LINKER_FLAGS=-static \
      -DZLIB_LIBRARY_RELEASE=/usr/lib/x86_64-linux-gnu/libz.a
cmake --build build-linux -j --target sankhya-cli
cp build-linux/sankhya apps/web/api/bin/sankhya
```

Static linking is what lets one file run on Vercel's Amazon Linux without matching its
libstdc++; OpenMP is off because its runtime is only shipped as a shared object. Then:

```bash
python apps/web/vendor.py
python apps/web/test_server.py
cd apps/web && vercel --prod
```

Limits worth knowing: uploads are capped by Vercel's request body size (about 4.5 MB), a
solve has 300 seconds, and the function has no GPU and one or two cores. The Netlib
instances in `data/netlib/` all fit; MIPLIB does not, and the board is not the place to
claim it does.
