# SANKHYA — architecture

How the solver is put together, where each algorithm lives, and where the next engine plugs
in. PS26119 asks for a "transparent, extensible foundation"; this document is the map that
claim is checked against. Every citation for an algorithm named here is in
`docs/PROVENANCE.md`; every number is in `docs/BENCHMARKS.md`, generated from CSVs.

## 1. The shape in one paragraph

A `Model` goes in, a `Solution` comes out, through one function:

```
solve(const Model&, const Options&) -> Solution        (src/core/solve.cpp)
```

`solve()` classifies the model by what it contains — integrality, a quadratic objective,
neither — and dispatches to an engine. Every engine consumes the same `Model`, produces the
same `Solution`, and is judged by the same code afterwards: `Solution::recompute_quality()`
re-measures feasibility, the reduced costs and the duality gap against the original model,
and the dispatcher's status guard downgrades any claim the measurement does not support.
That seam is what makes a new engine a bounded piece of work: it has to produce a
`Solution`, and it gets the same audit as the ones that exist.

```
                MPS / LP file            C API            Python (pybind11)
                       │                    │                    │
                       └──────── readers (src/io) ───────────────┘
                                          │
                                        Model
                                          │
                              presolve (src/presolve)          ┐
                                          │                    │  LP path
                   ┌──────────────┬───────┴────────┬────────┐  │
              dual simplex   primal simplex      PDHG      IPM (algorithm=ipm)
              (default LP)   (algorithm=simplex) (pdhg)     │
                   └──────────────┴───────┬────────┴────────┘  │
                                       postsolve               ┘
                                          │
                     branch and bound (src/mip) ── node LP: warm-started dual simplex
                     convex QP (src/qp)        ── Condat–Vũ, MIQP nodes
                                          │
                              Solution ── recompute_quality ── status guard
                                          │
                        .sol / --stats JSON (src/io)   tools/verify_solution.py
```

## 2. Modules and their boundaries

| directory | what lives there | depends on | lines |
|---|---|---|---|
| `include/sankhya/` | the public headers: `model.hpp` (Model, Solution, BasisStatus), `options.hpp`, `tolerances.hpp` (every numerical constant, cited), `sankhya.h` (the C API), `io.hpp`, `mip.hpp`, `qp.hpp`, `pdhg.hpp` | nothing | 1.6k |
| `src/core/` | `Model`/`Solution` implementation, `recompute_quality()`, the `solve()` dispatcher and its status guard | everything below | 0.7k |
| `src/util/` | logging, the options registry (every option has a description and a default, checked for duplicates by a test), the version stamp | – | 0.8k |
| `src/la/` | CSC/CSR sparse matrix with row views, the sparse LU with Markowitz threshold pivoting and the product-form update (FTRAN/BTRAN, eta file), Ruiz + Pock–Chambolle equilibration | – | 1.6k |
| `src/io/` | MPS (fixed and free, RANGES, negative-UP convention, MARKER blocks, gzip) and LP readers, the `.sol` writer and the `--stats` JSON writer | core | 2.2k |
| `src/presolve/` | reductions (empty/fixed/singleton rows and columns, redundant rows, free-column singletons, doubleton equations, integer bound rounding) and the postsolve stack that reconstructs the primal and the DUAL of the original model | core, la | 1.8k |
| `src/simplex/` | `simplex_core.hpp` — the state the two simplex loops share (basis, factors, pricing weights, perturbation, warm start); `primal_simplex.cpp` — bounded-variable revised primal simplex, composite phase 1, Devex pricing, textbook and Harris ratio tests, bound perturbation, basis repair; `dual_simplex.cpp` — bounded dual simplex, bound-flipping ratio test, dual Devex, artificial bounds, cost perturbation, hand-over to the primal loop; `dense_lu` — a dense reference used by tests | core, la | 3.2k |
| `src/pdhg/` | restarted PDHG (PDLP-style), CPU; the CUDA backend hangs off this path (`src/gpu/`, behind `SANKHYA_ENABLE_CUDA`, on `main` since #329, measured on an RTX 5050 by #390: the GPU overtakes the CPU at 10,000 rows, `docs/BENCHMARKS.md` section 1g) | core, la | 0.7k |
| `src/ipm/` | Mehrotra predictor-corrector interior-point method on the normal equations, over the sparse LDLᵀ in `src/la/ldl.cpp`; no basis | core, la | 0.5k |
| `src/qp/` | convexity check (Cholesky of the Hessian), Condat–Vũ first-order convex QP | core, la | 0.5k |
| `src/mip/` | branch and bound: propagation, root diving, reliability branching with strong branching, warm-started dual node LPs, MIQP nodes through the QP engine; root cuts are PR #159 | core, simplex, qp | 1.3k |
| `src/api/` | the C API over `solve()` (including `sankhya_set_callback` and `sankhya_model_interrupt`); the Python bindings (`bindings/python/`, including `Model.interrupt()`) wrap this, not the C++ | core | 0.5k |
| `apps/sankhya-cli/` | `sankhya solve|info|options|version`, `--stats`, `--write-sol`, `--option k=v`. Uses `sankhya_set_callback` for graceful SIGINT interrupted exit | api, io | – |
| `tools/` | `verify_solution.py`: re-parses the model with its own reader and checks the `.sol` file's primal feasibility, reduced costs, dual feasibility, complementary slackness and strong duality. Shares no code with the solver, deliberately | – | 1.5k |
| `tests/` | unit tests per module; `oracles/` — a rational-arithmetic simplex and exact MILP branch and bound that the float engines are fuzzed against; `robustness/` — the sweeps that find where the solver stops working | – | 10k |
| `bench/runners/` | Netlib, MIPLIB and Mittelmann runners and fetchers, the HiGHS comparison (a separate process over the same files), the robustness sweep, `make_benchmarks_doc.py` which generates `docs/BENCHMARKS.md` from the CSVs | – | 4k |

The dependency direction is strictly downward in that table: `la` knows nothing about
models, `simplex` knows nothing about integrality, `mip` knows nothing about file formats.
No file in `src/` reads or links anything from another optimization solver; `docs/PROVENANCE.md`
records the dependency table, the link line and the algorithm citations.

## 3. The two invariants everything else rests on

**The frozen interface.** `Model`, `Solution` and `solve()` do not change without an explicit
note in the PR that changes them. Readers produce a `Model`; every engine consumes one and
produces a `Solution`; the writers, the verifier, the C API and the bindings consume a
`Solution`. A new engine touches `src/<engine>/` and one branch of the dispatcher.

**Nothing is reported that was not measured.** `recompute_quality()` recomputes the row
activities, the objective, the reduced costs and the KKT residuals against the ORIGINAL
model after postsolve, and the status guard turns an engine's "optimal" into "feasible" when
those numbers disagree with the claim. Outside the process, `tools/verify_solution.py`
repeats the audit with its own reader, and the exact oracle in `tests/oracles/` is the
standard the float engines are compared against on instances nobody chose. A wrong answer
in numerical code prints and looks correct; these three layers are how it gets caught.

### 3a. The size this build supports

`Index` is `std::int32_t`, so a sparse structure addresses at most **2,147,483,647 nonzeros**
and the same number of rows or columns (`kMaxNonzeros` in `include/sankhya/types.hpp`). The
choice is deliberate: every offset, loop bound and allocation size in a pattern is an `Index`,
and widening it to 64 bits doubles the memory of every pattern array on every model to buy a
size no benchmark here reaches. The largest instance the scale runners generate is about
5,000,000 nonzeros, three orders of magnitude below the ceiling.

What the ceiling costs is a guard rather than a risk (#305). `SparseMatrix` counts entries as
they arrive, refuses the ones past its limit and flags itself; `Model::validate()` turns the
flag into a model error before an engine reads the pattern; the sparse LDL^T checks the size
of the factor its ordering implies, since fill-in can make the factor far denser than the
matrix. A model that exceeds the limit is therefore **refused with a diagnostic**, not
assembled from a prefix sum that wrapped into negative offsets.

`SparseMatrix::set_nonzero_limit()` lowers the ceiling per instance. It exists so the refusal
path is testable at a handful of entries instead of 2^31, and doubles as a per-matrix memory
cap for a caller who wants one.

## 4. How a solve flows

1. **Read.** `src/io` produces a `Model` with column-major storage, bounds, integrality and
   an optional lower-triangular Hessian. Coefficients below `kZeroDrop` are dropped here and
   nowhere later, and `docs/BENCHMARKS.md` §5 records what that costs.
2. **Classify.** Integrality → branch and bound; a Hessian → convex QP (or MIQP nodes);
   otherwise an LP engine chosen by `algorithm`: `auto` is a rule table on the model's shape
   (`src/core/engine_selection.cpp`, #284): the dual simplex below 20,000 rows and 100,000
   nonzeros, the interior point with crossover above either, PDHG from 100,000 rows, and a
   starting basis always the dual simplex; every threshold names its CSV, the answer carries
   the rule and the reason, and an interior point that declines - a factor beyond its budget, or a
   set-up past `ipm_setup_share` of the time limit (#357) - falls back to PDHG at or above the
   row limit and to the dual simplex below it, on the time that is left.
3. **Presolve** (LP path). Reductions are recorded on a stack. The reduced model is scaled
   (Ruiz then Pock–Chambolle) inside the simplex entry point; the scaled and unscaled
   attempts share one time budget.
4. **Solve.** The dual simplex starts from the slack basis (or a warm start), boxes any
   column that is dual infeasible, runs the bound-flipping ratio test with dual Devex
   pricing, perturbs costs on a degenerate stall, and hands the basis to the primal loop
   whenever it cannot finish honestly (artificial bound active, pivot disagreement on fresh
   factors, marginal infeasibility). Optimality is declared only on fresh factors.
5. **Postsolve.** Primal values are reconstructed in reverse record order; the duals are
   reconstructed to a fixed point, then every reduced cost of a column presolve could have
   touched is recomputed as `c − Aᵀy`.
6. **Audit and report.** `recompute_quality()`, the status guard, then the `.sol` and JSON
   writers. The `.sol` file carries 17 significant digits so the verifier sees what the
   solver saw.

For a MILP, step 4 becomes the tree: propagation at each node, a root dive for an incumbent,
reliability branching (pseudocosts once observed enough times, strong branching before that),
and every child solved by the dual simplex from its parent's basis, which is dual feasible
there by construction.

### 4a. What presolve reports

Presolve is the one stage that rewrites the model a user handed over, so it accounts for
itself (#286). Every solve carries a `Solution::PresolveReport`: rows, columns and nonzeros
before and after with the reduction percentages, the count of each reduction that fired, the
count of what presolve DECLINED to do and why (a column carrying curvature, an integer column
whose substitution would come back fractional), the number of passes, why it stopped (fixed
point, pass limit, or an infeasibility it proved), and how long it took.

The same numbers appear in three places, from one source: a summary line in the log and in
the CLI's result block, a per-reduction breakdown at `--option log_level=verbose`, and a
`presolve` object in the `--stats` JSON for the benchmark runners. The counts are derived
from the records postsolve replays, so the report cannot drift away from the transformation
it describes.

## 5. Where the next engines plug in

**The engine registry** (#297, `src/solver_engine/`). Every engine is a `SolverEngine`: a
name, the classes it accepts, what its answer carries (a basis, row duals, a certificate,
a warm start taken, an interrupt honoured, a deterministic mode kept) and a sentence on
what it is, registered once in `builtin_engines.cpp` as a thin wrapper over the existing
`solve_*()` function. `solve()` asks the registry which LP engine runs (`engine::select()`,
the rule table of #284) and whether it takes a starting basis, and the model's class is
decided in one place, `engine::classify`, for the dispatcher and the registry alike.
`sankhya engines` prints the table, `--format json` for scripts, with every flag read from
the engine rather than typed beside it. The names `algorithm` accepts are the registry's
`algorithm_names()`; the option table's own list is held to them by a test, since the
option layer cannot ask the engines without depending on them. When `algorithm` names an
LP engine for a MILP, QP or MIQP, the class's engine runs and the log and the answer's
message say so rather than the request being dropped silently. Adding an engine is a
wrapper and a registration line in `builtin_engines.cpp` and a branch in `solve()` for how
to run it; `solve()` refuses a registered engine it has no branch for rather than running
another under its name (#402).

- **Interior-point method** (#56) — built: `src/ipm/` is one branch of the LP dispatcher
  over the sparse LDLᵀ in `src/la/ldl.cpp` (#70). On its own it produces a `Solution` without
  a basis, which the status guard and the verifier handle as they do for PDHG; by default
  (`crossover=true`, #219) `src/simplex/crossover.cpp` then pushes that answer to a vertex
  with the dual simplex warm-started from a basis guessed off the interior point, so the
  reported solution carries a basis. The dual simplex stays the node engine for branch and
  bound.
- **MIQP** — already present: the branch-and-bound node relaxation is a QP when the model
  has a Hessian, and the node bound is quadratic (`src/mip/branch_and_bound.cpp`).
- **NLP / MINLP** — the frozen interface is the constraint: a `Model` today is linear
  constraints with an optional quadratic objective. A nonlinear engine would extend `Model`
  with constraint functions and gradients (an explicit interface change, per the rule in
  §3) and plug in at the same dispatcher seam; the MILP tree needs no change to search over
  it, since it only reads `col_value` and the bound. Nothing of this exists yet, and
  `docs/PS26119_COVERAGE.md` says so.
- **Cutting planes** — already present, and off by default. Root GMI and lifted cover cuts
  landed in #159 (`src/mip/cuts.cpp`) and single-row MIR cuts in #221
  (`src/mip/mir_cuts.cpp`), appended as rows of the working model before the search
  starts; the answer reports the root bound before and after the round. `enable_root_cuts`
  is false by measurement: the three-way A/B at `078cb24` (`docs/BENCHMARKS.md` section 2)
  has the off leg at 14 of 30 reached and 9 proved, the root round at 13 and 9 (one match
  lost at the time limit, nodes 1.049x) and root plus tree rounds at 14 and 10
  (`neos-3611689-kaihu` proved, nodes 0.920x). The tree leg is the first to gain a proof
  without losing one; whether it becomes the default is #221's open decision.
- **Parallelism** — the column loops in pricing and in the sparse products are
  embarrassingly parallel and deterministic (no cross-thread reductions); the tree search is
  the larger prize and the harder one, because a race on the incumbent can fathom a node
  that should have been explored (#57). It is `mip_threads` since #222; section 12.

## 6. Toolchain, as tested

- C++20, CMake ≥ 3.20, Ninja. CI is Ubuntu (GCC), Release and Debug + ASan/UBSan; the
  Windows development boxes use MSYS2 UCRT64 GCC 16.1.0 through `scripts/configure.sh`,
  which refuses a compiler older than GCC 10.
- Dependencies (all non-solver, table in `docs/PROVENANCE.md`): fmt, CLI11, nlohmann/json,
  GoogleTest (tests only), zlib (optional), pybind11 (bindings). No BLAS, no LAPACK.
- `clang-format` 22.1.8 from pip, pinned, because its output differs between majors.
- Python 3.10+ for the runners, `highspy` optional for the comparison.
- `scripts/reproduce.sh` is the one command from a fresh clone: configure, build, test,
  fetch, benchmark, robustness sweep, comparison, and regenerate `docs/BENCHMARKS.md`. It
  says which steps need the network and skips them, named, when it is absent.

## 7. Reproducibility, and what is actually promised

`--option deterministic=true` turns on the reproducible mode (#288). It is off by default:
it refuses a wall-clock `time_limit`, and a caller who asked for one has to keep getting it.

Four different things get called reproducibility, and only some of them are ours to give:

| Level | What it means | SANKHYA today |
|---|---|---|
| Mathematical | The same optimal value, whatever route is taken to it | Yes, and the exact rational oracle in `tests/oracles/` is what checks it |
| Numerical | The same answer inside the documented tolerances | Yes, within `tol::kPrimalFeasibility` and friends; this is what the verifier enforces |
| Execution determinism | The same decisions in the same order: same algorithm, same iteration count, same nodes, same branches | Yes in deterministic mode, on one build and one machine. This is what `tests/unit/test_deterministic.cpp` measures |
| Bit for bit | Identical bit patterns in every reported number | Yes for repeated runs of one build on one machine; NOT claimed across compilers, optimization levels, CPUs or GPUs, and nothing here tests that |

What deterministic mode changes, all of it in one function, `apply_deterministic_mode()` in
`src/core/solve.cpp`, before any engine sees the options:

- a `time_limit` is refused with a warning naming the value, and the search is expected to
  be bounded with `iteration_limit` or `node_limit` instead, which count the same on every
  machine;
- `polish_max_seconds` stops bounding the interior-point polish; `polish_max_factor_nonzeros`
  does, which is a property of the model rather than of the machine;
- PDHG's 70 percent share of a finite time limit is not taken, because there is no finite
  time limit left to share;
- `threads` becomes 1 unless the caller set it. Set explicitly, it is honoured and the log
  says that reproducibility then rests on #57's measurement that the column loops are
  order-independent rather than on anything this mode re-checks.

What it does not touch, because the audit in the PR for #288 found it already deterministic:
every randomized component in the solver is a power iteration started from a fixed seed
(`src/pdhg/pdhg.cpp`, `src/la/scaling.cpp`, `src/qp/qp_condat_vu.cpp`), all three now derived
from `random_seed` rather than from a literal, and nothing anywhere seeds from the clock or
from an address. The ratio tests break ties on the larger pivot and then on the first
candidate reached, scanning in index order. Node selection keeps the first node achieving the
best bound while scanning the open list in order, and the open list is built in index order.
Presolve walks rows and columns in index order; its `unordered_map`s are lookup tables that
are never iterated, so no reduction depends on a bucket order. Cover-cut variables are sorted
by coefficient descending and then by column index (`src/mip/cuts.cpp`).

What it cannot remove: `solve_seconds` and every timing in the log, which are measurements
of this run and are meant to differ; and a progress callback, whose window is wall-clock, so
how often it fires varies between runs. A callback that only reports is harmless, one that
interrupts decides the answer on the clock, and deterministic mode warns when one is
attached.

Two runs that want to be compared should first check they were handed the same model:
`Model::fingerprint()` is a 64-bit FNV-1a over the canonical numbers, reported in the log in
deterministic mode and written to every stats blob as `model.fingerprint`. It hashes bit
patterns, so -0.0 and 0.0 are different inputs, and it ignores names, because two models
that differ only in what their columns are called solve identically.

The GPU path refuses `deterministic=true` (#383): the fused `atomicAdd` reductions in the
CUDA kernels are order-dependent and cannot satisfy the bit-for-bit promise. When
`deterministic=true` and the GPU path would otherwise be selected, `solve()` logs a warning
and falls back to CPU PDHG. `docs/PS26119_COVERAGE.md` says what exists.

## 8. Resource limits, and what each one means

One place decides what a limit means: `ResourceLimits` in `src/core/resource_limits.hpp`,
read from the options once per solve and consulted by every engine through `StopController`
(#289). Before it, each engine read the options for itself and they did not agree.

| Option | No limit | Zero | N |
|---|---|---|---|
| `time_limit` | the default sentinel, or any non-finite value | a budget of ZERO SECONDS: the solve stops at its first safe point | at most N seconds of wall clock |
| `iteration_limit` | `-1` | no iterations are performed | at most N iterations |
| `node_limit` | `-1` | no nodes are processed | at most N nodes |

A negative duration or a count below `-1` is a configuration error: it is refused with a
warning naming the value, and the solve runs without that limit rather than with a nonsense
one.

**What the clock covers.** It starts at the top of `solve()`, so it covers presolve, the
engine and postsolve. It does not cover reading the model or writing the answer, which happen
in the CLI around the call. An engine reached after presolve is given what is LEFT of the
budget, not the whole of it.

**Precedence**, when more than one limit is exhausted at the same check:

    user interrupt  >  time  >  iterations  >  nodes

The interrupt wins because it is the only one a person is waiting on; time beats the counters
because it is the limit protecting a caller's own deadline.

**Safe points.** A limit is observed at a boundary the engine chooses: between simplex
iterations, between PDHG iterations, between branch-and-bound nodes, and inside the interior
point's ordering and factorization through the same predicate (#197). A factorization or a
kernel already running finishes first, so overrunning a deadline by one such step is expected
and is not a violated limit.

**A limit is never a numerical failure.** `kTimeLimit`, `kIterationLimit`, `kNodeLimit` and
`kInterrupted` are distinct from `kOptimal`, `kInfeasible`, `kUnbounded` and
`kNumericalError`, and the solve reports which one stopped it in `Solution::stopped_by`,
because a MILP that hits a limit holding an incumbent reports `kFeasible` and the status can
no longer say. Three cases that used to come back `numerical_error` and no longer do:
`node_limit=0`, `time_limit=0` on a MILP, and a node LP that runs out of iterations.

**What a stopped solve still reports.** The incumbent, the best bound and the gap where a MILP
has them. A search stopped before it found any integer point reports no point, the worst
representable objective and an INFINITE gap, because an unknown gap is not a closed one, and
a bound only if a node proved one: an unevaluated root proves nothing.

**Not implemented, and not pretended.** There is no memory limit and no GPU memory limit. Peak
resident memory is not portably queryable from this binary, and `docs/PS26119_COVERAGE.md`
says so rather than the option table carrying a knob that does nothing. The CUDA backend
estimates whether the model fits in the device's memory before it starts and falls back to
the CPU when it does not (#370); that estimate is the one device-memory claim made, and
kernel termination is not a limit this section enforces.

## 9. Nonlinear models: the representation, before any engine

`src/nlp/` (#296) is the model layer a nonlinear objective or constraint needs: an expression
graph over the model's columns, evaluated without ever returning NaN as a value,
differentiated exactly, and classified for convexity by written rules. It solves nothing, and
no LP, MILP, QP or MIQP engine reads it.

- **`NonlinearModel` holds a frozen `Model`** for the columns, bounds, integrality, linear rows
  and the linear and quadratic objective, and adds an `ExpressionGraph` over the same columns
  for the objective term and the constraints the existing engines cannot take. There is no
  second variable model, and an LP never passes through an expression tree.
- **The graph is an arena of immutable nodes named by index.** A child is always created before
  its parent, so the graph cannot hold a cycle, a reference cannot dangle, and index order is an
  evaluation order. Construction is hash-consed: the same operation built twice is one node,
  commutative operations sort their children, and a shared subexpression is referenced rather
  than copied. Misuse is sticky, not thrown (nothing in `src/` throws): an unknown node, an
  out-of-range column or a non-finite constant yields `kNoExpr` and one recorded message, as
  `SparseMatrix` does for an overflowed count.
- **Simplification never changes a domain.** `x + 0`, `x * 1`, `-(-x)`, constant folding and
  sum flattening happen; `0 * log(x)` stays a product, because at x = -1 it is an error, not a
  zero, and `exp(log(x))` stays a composition for the same reason.
- **Evaluation separates three failures.** A domain error (log of a non-positive, sqrt of a
  negative, division by zero, a non-integer power of a negative) is not an overflow, and neither
  is a malformed expression. `NonlinearModel::evaluate` keeps a domain failure apart from a
  constraint violation: a point the model does not define is not a point that violates it.
- **Derivatives are exact.** Gradients by reverse-mode AD; Hessians by forward-over-reverse, one
  sweep per column the expression contains, returned as a sparse lower triangle. A derivative
  that does not exist (sqrt at 0) is refused, not returned as infinity. Finite differences
  exist only as `check_derivatives`, a test facility.
- **Convexity is conservative.** The composition rules of disciplined convex programming, with
  each atom's monotonicity read off the interval range of its argument, so x^3 is convex on
  x >= 0 and 1/x convex on x > 0. Anything the rules cannot establish is unknown, never guessed:
  `log(exp(x))` is affine in truth and unknown here. The soundness test puts every random
  expression the rules call convex or concave through the definition at random point pairs.

What is not here: an engine (#226), a file format (nothing reads or writes a nonlinear model
yet), nonlinear presolve, and spatial branch and bound for the non-convex case.

## 10. Profiling: where a solve's time goes

`--option profile=basic|detailed` (#285) records a tree of named regions with inclusive time,
exclusive time and call counts, plus counters, and prints it in the log;
`--option profile_out=<path>` also writes it as JSON. `src/util/profiler.hpp` holds the
profiler and `ProfileScope`, its RAII timer.

- **basic**: `solve`, then `presolve`, `engine`, `postsolve` and `verification` (the status
  guard and the certificate check), with `ranging` and `iis` when they run, and the counters
  every engine already keeps (iterations, nodes, polish iterations, cuts).
- **detailed**: also what happens inside an engine. The dual simplex's pricing, pivot row,
  ratio test, FTRAN, update, refactorization, basic values and reduced costs come from the
  accumulators #210 already keeps and are not timed a second time. The interior point adds
  normal-equation assembly, ordering and factorization, PDHG its primal and dual steps, and
  the branch and bound its node LPs, heuristics and branching (strong branching's probes
  included). Counters add refactorizations, nodes pruned, warm and cold node LPs, and PDHG
  restarts.

The profiler rides on the `Logger` every engine is already handed, so no engine signature
changed. With profiling off it is never attached, and a scope costs a null-pointer test; a
disabled scope does not read the clock. `bench/results/profiler-overhead-3630cba.csv` measures
off, basic and detailed against `main` at the same commit, and finds no overhead above the
machine's noise, with identical answers in every mode.

There is no GPU timing and no memory statistic. The CUDA path has no profiler hook yet, and
peak resident memory is not portably measurable from this binary; neither is reported rather
than guessed.

## 11. Conflict analysis: learning from infeasible nodes

`conflict_analysis` (#292, `src/mip/conflict.hpp`, `src/mip/branch_and_bound_conflicts.cpp`)
learns, from each node proved infeasible, which of its branching decisions were to blame, and
uses that in every later node's propagation. It is OFF by default, and the second A/B run
(#405) says why: the 30-instance MIPLIB set at 60 s, one commit, one machine,
`conflict_analysis` the only difference. It was run on the closeout branch at `6c405f8`, a
commit that is not on `main`, so its CSVs are not committed (the #256 rule, as in #340) and
what follows is an observation until the same runner is repeated on `main` as a bench PR.

`conflict_use` (#406) separates learning from use for that measurement: `none` learns and
uses nothing (the analysis's cost alone), `prune` only prunes, `propagate` (the default) also
fixes bounds; with `conflict_minimize` these are the five arms of the ablation #292 asks for.

- **Nothing moved that must not move.** 14 of 30 reach the published optimum and 9 of 30 prove
  it, in BOTH legs. No instance changed status, no matched or proved verdict moved, and no
  feasible point was lost: the same 28 solutions pass independent verification either way
  (`enlight8` and `enlight_hard` end at the limit with no incumbent in both legs).
- **Where the search finishes, conflicts make it smaller.** Over the nine proved instances the
  node count goes from 2,285 to 2,204 (0.965x) in 6.13 s against 5.67 s. Six are identical and
  the whole movement is three instances: `supportcase16` 87 nodes against 123, `supportcase14`
  102 against 115, `flugpl` 437 against 469.
- **Where it does not finish, conflicts cost throughput.** The 21 instances stopped by the time
  limit explore 1,900,912 nodes with conflicts on against 2,026,885 without (0.938x) - in the
  SAME wall clock, so that is nodes not reached, not search saved. `enlight8` is the extreme at
  0.373x: 46,704 nodes against 125,200, with 37.4 s of the 60 s budget spent inside the
  analysis itself. Of the three incumbents that moved at the limit, two improved
  (`timtab1` 1,149,285 against 1,217,806, `neos-5140963-mincio` 14,900 against 15,226) and one
  worsened (`gen-ip054` 6,870.87 against 6,859.87).

This CONFIRMS the pre-merge observation rather than revising it. That observation, made on a
working tree before #381 and never citable, named four instances and this run reproduces every
one in the same direction: `supportcase16` 91 against 127 then, 87 against 123 now;
`supportcase14` 102 against 124 then, 102 against 115 now; `flugpl` 437 against 469 in both;
`enlight8` fewer nodes in the same budget, 23,040 against 49,918 then and 46,704 against
125,200 now. The default stays OFF: the measured benefit is 81 nodes over the
nine instances that finish and no verdict either way, against up to 62 percent of a time
budget on an instance whose infeasible nodes are cheap and many. The option is there for the
instances that behave like `supportcase16`, and the default follows the measurement, as it does
for the cut families.

Conflict quality on three of them, from `conflict_out` at the same limit: `supportcase16`
analysed 37 infeasible nodes, learned 35 (2 not proved from the global bounds), mean size 1.5,
450 bounds tightened, 0.007 s; `flugpl` analysed and learned 146, mean size 3.7, 15 nodes
pruned and 713 bounds tightened, 0.005 s; `enlight8` learned 6,783, mean size 7.3, 3,212 nodes
pruned and 615,775 bounds tightened, 37.4 s. Short conflicts that tighten bounds are what pays;
long ones on an instance that produces thousands of them are what does not.

- **What is learned.** A set of bound literals `x_j <= v` / `x_j >= v` on integer columns,
  taken from the node's branching decisions, that the rows and the GLOBAL column bounds cannot
  satisfy together. The learned constraint is the bound disjunction "one of them is false".
  Because it is proved from the global bounds it is globally valid; there are no node-local
  conflicts.
- **When a set counts as proved.** Only after it is checked again from scratch: the literals
  applied to the global bounds, then node propagation (which may empty the box), then, for a
  node whose LP was infeasible, the LP's Farkas multipliers re-evaluated conservatively on the
  propagated box (every coefficient counted at its true bound, no coefficient rounded to
  zero, a margin that scales with the aggregation). Neither the LP's status nor its
  multipliers are taken on trust; a set that does not pass is counted as rejected and dropped.
  Numerical failures, time limits and interrupts never reach the analysis at all.
- **Minimisation.** The literals the Farkas proof leans on are tried first, then a deletion
  filter drops each decision the check still passes without, at most 32 checks per conflict.
  The node's own decision is kept without a check, since its parent's solved LP shows it is
  needed.
- **Use.** At the start of each propagation sweep: a conflict whose literals all hold prunes
  the node without an LP, and one with a single undecided literal fixes that literal false,
  which is an integer bound one step past it.
- **Store.** Canonical sorted literals, exact duplicates rejected, at most `conflict_max`
  (10,000) held and `conflict_max_size` (32) literals each. A full store forgets the least
  used tenth, ordered by uses, then last use, then age: a total order, so a rerun keeps the
  same conflicts. Forgetting weakens pruning and never changes the feasible region.
- **Budget.** Work, not seconds: verification calls may not run more than 8 per explored node
  ahead (plus a start-up allowance), so the analysis is deterministic under
  `deterministic=true` and cannot eat a search that finds infeasible nodes cheaply.

Conflicts live in the indices of the model the search runs on, which is the presolved model
when presolve ran; they are solver metadata and are not mapped back. `conflict_out=<path>`
writes them and their statistics as JSON for diagnostics, with the `columns` field naming the
column count of the model the indices belong to, so a reader cannot quietly read them as the
original model's. The log's `Conflicts:` line and the profiler's `conflict analysis` region
report their cost.

## 12. Parallel tree search

`mip_threads=N` (#222, `src/mip/parallel_search.hpp`, `src/mip/branch_and_bound_parallel.cpp`)
runs the branch and bound on N worker threads. Each worker runs the ordinary sequential
search on one subtree at a time, with its own working model and node LPs. A subtree is a
chain of bound changes from the root, so giving one away copies a few dozen numbers. The
workers share the incumbent (every worker prunes against the best point anyone found), the
node count (so `node_limit` covers the whole search), one solution pool, the pseudocosts
(exchanged every 20 nodes), the node scaling (computed once), a queue of subtrees and a stop
flag. A worker with at least eight open nodes gives the smallest-bound ones away whenever
another worker is idle and the queue is empty, never its two newest nodes, which its dive is
about to take. Each worker sets OpenMP's thread count to one when it starts: the count is per
thread, and a new thread would otherwise fork a full team in every column loop.

The answer does not depend on the thread count; the tree explored does. A subtree that stops
early (a limit, or its own gap test) hands back the smallest bound among its open nodes, the
reported bound is the smallest of everything left open anywhere, and optimality is claimed only
when every subtree closed or met the gap target. The caller's progress callback is called from
the calling thread only, and its interrupt is forwarded to the workers.

Not taken, with a note in the log: an MIQP (QP node relaxations), `pool_complete` (its pruning
reads the pool at every node), a checkpoint or resume (the file holds one search's tree) and
`deterministic=true` (the tree varies with timing). `node_limit` can be overshot by at most one
node per worker, because each counts a node before it checks. Root cuts, if enabled, are the
root worker's own rows; other workers do not see them.
