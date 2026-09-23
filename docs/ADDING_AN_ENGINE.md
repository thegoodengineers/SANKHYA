# How to add a new solver engine to SANKHYA

The steps #297 lists, in order, with the file each one touches. The contract the engine
inherits - ownership, thread safety, status mapping, how failures surface - is
`docs/ARCHITECTURE.md` section 13; this page is the work. An engine is a wrapper over a free
function that does the algorithm, in the shape the existing six already have
(`src/solver_engine/builtin_engines.cpp`). Nothing below needs a change to another engine.

Where a step touches more than the engine's own files, the page says so: an LP engine needs a
line in `solve()`'s dispatch and in the option table, and a second engine for a class that has
one today (QP, MILP, MIQP) needs `solve()`'s branch for that class to ask the selector, which it
does not yet, because there has been nothing to choose between.

## 1. Define the engine

Write the algorithm as a free function under its own directory, taking what every engine takes
and returning a `Solution`:

```cpp
// src/myengine/myengine.hpp
namespace sankhya::myengine {
[[nodiscard]] Solution solve_myengine(const Model& model, const Options& options,
                                      Logger& logger, SolveControl* control);
}
```

Read the model through `const Model&` and do not keep it; keep the working state on the stack
of the call. Print nothing: write to `logger` (`logger.info(...)`, the iteration table), which
the caller owns and may have silenced. Draw every random number from the `random_seed` option,
never from the clock or an address (`docs/ARCHITECTURE.md` section 7). Do not throw and do not
exit; every ending is a status (step 6).

Then the wrapper, in `src/solver_engine/builtin_engines.cpp`:

```cpp
class MyEngine final : public SolverEngine {
 public:
  [[nodiscard]] std::string name() const override { return "myengine"; }
  [[nodiscard]] std::string summary() const override { return "One sentence for `sankhya engines`"; }
  [[nodiscard]] std::string source() const override { return "src/myengine/myengine.cpp"; }
  [[nodiscard]] EngineCapabilities capabilities() const override;  // step 2
  [[nodiscard]] Solution solve_verified(const Model& model, const Options& options,
                                        Logger& logger, SolveControl* control) const override {
    const Options effective = apply_deterministic_mode(options, logger);
    return myengine::solve_myengine(model, effective, logger, control);
  }
};
```

`solve_verified` is the algorithm's entry; the public `solve()` is not virtual and does the
class gate and the `stopped_by` mapping for every engine, so a wrapper never repeats them.

## 2. Declare capabilities

Set only what the wrapper's own `solve_verified` delivers, on the answer it returns:

```cpp
EngineCapabilities caps;
caps.lp = true;                          // the classes it accepts
caps.supports_duals = true;              // row_dual and col_dual are on the answer
caps.supports_interrupt = true;          // it checks the SolveControl at its safe points
caps.supports_deterministic_mode = true; // it calls apply_deterministic_mode itself
return caps;
```

A flag that is true only when the engine is reached through `solve()` is not the engine's:
the interior point's `supports_basis` is honest because its wrapper runs the crossover itself.
`supports_warm_start` means the wrapper reads `SolveControl::start_col_status` and
`start_row_status` (#218); `supports_checkpoint` means it writes the `checkpoint` option's file
and reads `resume` (#287); `supports_certificates` means an infeasible or unbounded answer
carries `farkas_dual` or `primal_ray` that `verify_and_keep_certificate` accepts. Step 8's
tests fail an engine that claims one of these and does not deliver it.

## 3. Register the engine

One line in `register_builtin_engines()`, same file:

```cpp
registry.register_engine(std::make_shared<MyEngine>());
```

`SolverRegistry::builtin()` then has it, `sankhya engines` lists it with the flags from step 2,
and `registry.candidates(model)` offers it for every class it accepts.

For an LP engine, two more places, both held by tests:

- `SolverRegistry::algorithm_names()` now includes it, so `algorithm=myengine` is accepted by
  `solve()`. The option table in `src/util/options.cpp` keeps its own `choices` list for
  `algorithm`, because `src/util` cannot depend on the engines;
  `OptionsAndRegistry.AlgorithmChoicesAreAutoPlusTheRegistrysAlgorithmNames` fails until the
  name is added there.
- `solve()`'s LP branch (`src/core/solve.cpp`) runs the chosen engine by name, and refuses a
  registered engine it has no branch for rather than running another under its name. Add the
  branch beside the simplex, PDHG and interior-point ones.

For `algorithm=auto` to choose it, add a rule to the table in `src/core/engine_selection.cpp`
(#284), with the CSV that justifies its threshold named in the rule's reason, as every existing
rule does. Without a rule it is reached only by name.

## 4. Implement configuration

Engine options go in the one option table, `src/util/options.cpp`: a name, a type, a default,
a description, and a range or a list of choices. The table checks all of them when a value is
set from text - the CLI's `--option`, the C API - so a bad value is refused before any solve
starts, and `sankhya options` documents it with no further work. Read them inside the engine
with `options.get_int(...)`, `get_double`, `get_bool`, `get_string`. Name them after the engine
(`pdhg_*`, `ipm_*`, `mip_*`) so a reader can tell whose they are.

A combination the engine cannot honour is said, not ignored: a warning naming the value, as
`--gpu` without PDHG is (`src/core/solve.cpp`), or a `kNotSolved` answer naming why.

## 5. Implement the solve lifecycle

There is no configure, initialize or finalize to write: construction is registration, and a
solve is one call. Inside the call, obey the shared limits through the shared checker rather
than reading `time_limit` yourself:

```cpp
const Timer timer;
const ResourceLimits limits(options, logger);       // src/core/resource_limits.hpp
StopController stop(control, timer, limits);       // src/core/stop_controller.hpp
while (...) {
  SolveStatus why;
  if (stop.should_stop([&] { return progress_now(); }, &why)) { status = why; break; }
  if (limits.iterations_exhausted(iterations)) { status = SolveStatus::kIterationLimit; break; }
  ...
}
```

Check at a safe point - between iterations, between nodes - so that stopping leaves a point to
report. The precedence between the limits is section 8's: interrupt, then time, then
iterations, then nodes.

## 6. Produce SolverResult

The result is a `Solution` (`include/sankhya/model.hpp`); the engine layer has no other result
type. Allocate it for the model, fill what the engine has, and leave the rest at its default,
which means "not produced":

- `status`: one `SolveStatus`, mapped from the algorithm's own ending as the table in
  `docs/ARCHITECTURE.md` section 13 does. A limit is a limit status, never `kNumericalError`;
  `kOptimal` is claimed only against the project's tolerances (`include/sankhya/tolerances.hpp`),
  and a first-order method that met its own request but not those reports `kFeasible`.
- `col_value`, and `objective` at that point; `row_dual` and `col_dual` if step 2 declared
  duals; `col_status` and `row_status` if it declared a basis; `iterations`, `nodes`;
  `algorithm`, the name the answer carries into the stats JSON; `message`, the reason in words.
- Call `solution.recompute_quality(model)` last, so the feasibility and duality numbers on the
  answer are measured from the vectors rather than asserted by the engine.

`stopped_by` is filled from the status for you (step 1). Postsolve, the status guards, ranging
and the independent verifier run on the answer after the engine returns; none of them is the
engine's job.

## 7. Integrate statistics and profiling

Counts that every engine has go on the `Solution` (`iterations`, `nodes`). What is particular to
the engine goes to the profiler riding on the logger, which costs a null-pointer test when
profiling is off:

```cpp
{
  ProfileScope timed(logger.profiler(), "myengine step", ProfileMode::kDetailed);
  ...
}
if (Profiler* p = logger.profiler()) p->count("myengine restarts", restarts);
```

`--option profile=basic|detailed` prints the tree and `profile_out` writes it as JSON
(`docs/ARCHITECTURE.md` section 10). `bench/runners/engine_dispatch.py` reads the `engine`
region beside the dispatch regions; add a case for the new engine there.

## 8. Add tests

- Nothing to write for the common contract: `tests/unit/test_solver_engine_conformance.cpp`
  runs over every registered engine and every class it declares, and checks the answer against
  its own measurement of the point, each declared capability against the answer, the limit
  statuses and `stopped_by` under a zero budget, a zero time limit and an interrupt, and two
  identical runs under `deterministic=true`. Run it first; it is the quickest way to find a
  capability the wrapper claims and does not deliver.
- The algorithm's own tests go in `tests/unit/test_myengine.cpp`, added to `tests/CMakeLists.txt`:
  known optima, the edge cases the method is known to have, and a comparison against the exact
  oracle in `tests/oracles/` where the class allows it.
- Solve the committed instances through the benchmark runner for the class, with
  `--solver-option algorithm=myengine`, and let `tools/verify_solution.py` check every answer.

## 9. Add documentation

- A row in `docs/ARCHITECTURE.md` section 2 for the new directory, and the engine in section 5.
- The algorithm's citations in `docs/PROVENANCE.md`: every method named in the code is cited.
- Any number you quote comes from a CSV in `bench/results/`, measured on `main`, and is
  reported in `docs/BENCHMARKS.md` through `bench/runners/make_benchmarks_doc.py`, never typed.
- The coverage tracker, `docs/PS26119_COVERAGE.md`, says what the engine does and, just as
  plainly, what it does not.
