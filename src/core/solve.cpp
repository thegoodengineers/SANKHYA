// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the solve() dispatcher.
//
// THIS FILE IS THE SEAM. Every engine is dispatched from here and nowhere else; since #297
// the engines are REGISTERED in src/solver_engine/builtin_engines.cpp and the registry's
// selector picks the LP engine, while this file decides how to run it:
//   Phase 2  primal revised simplex  -> LP
//   Phase 4  restarted PDHG          -> LP, large and sparse
//   Phase 5  branch and cut          -> MILP
//   Phase 8  Mehrotra IPM, convex QP -> LP and QP
// A class with no engine returns kNotSolved and says so. Per ENGINEERING_RULES.md an
// unimplemented path reports the truth rather than a plausible zero - and in particular a MILP
// is NOT quietly handed to the simplex and its fractional relaxation reported as optimal, which
// is the single most damaging thing this dispatcher could do.
//
// THE PIPELINE, AND WHERE #297'S REGISTRY PARTICIPATES IN IT (full integration). One
// authoritative pipeline runs every solve, in this order: validate (Model::validate()) ->
// apply_deterministic_mode -> classify -> [WHICH ENGINE: engine::select() /
// SolverRegistry::builtin().candidates(), src/solver_engine/solver_selector.cpp - the ONE
// place that decision is made; this file no longer re-derives it] -> presolve -> [HOW TO RUN
// IT: the chosen engine's underlying solve_*() free function, called directly, with the
// polish/fallback/warm-start/node-scaling choreography below - orchestration that belongs
// above any one engine's own contract, not inside it] -> postsolve ->
// verify_and_keep_certificate -> reconcile_status_with_measurement / ranging / IIS ->
// Solution. Presolve, postsolve, certificate verification, resource limits, profiling and
// logging stay centralized HERE, exactly as before #297 - the registry supplies WHICH engine,
// this file still owns everything else, and there is one pipeline, not two.
//
// A SolverEngine's OWN solve() (src/solver_engine/builtin_engines.cpp) is a second, narrower
// entry point for a caller who wants the engine directly rather than through this pipeline
// (src/solver_engine/solver_engine_dispatch.hpp: the same presolve pipeline, resource limits,
// out-of-memory guard and status guards; no polish, fallback, ranging or IIS). It re-applies
// apply_deterministic_mode and certificate verification itself (the same shared functions
// this file calls, not a re-derived copy) precisely because it does NOT go through this
// pipeline and so cannot assume this file already did.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <new>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#ifdef SANKHYA_ENABLE_CUDA
#include "gpu/device.hpp"
#include "gpu/gpu_memory.hpp"
#include "gpu/multi_device.hpp"
#include "gpu/pdhg_gpu.hpp"
#include "gpu/pdhg_gpu_guard.hpp"
#include "gpu/pdhg_multi_gpu.hpp"
#endif

#include <fmt/format.h>

#include "core/deterministic_mode.hpp"
#include "core/engine_race.hpp"
#include "core/engine_selection.hpp"
#include "core/iis.hpp"
#include "core/kkt_check.hpp"
#include "core/presolve_pipeline.hpp"
#include "core/resource_limits.hpp"
#include "core/status_guard.hpp"
#include "mip/components.hpp"
#include "sankhya/certificate.hpp"
#include "sankhya/ipm.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/mip.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/pdhg.hpp"
#include "sankhya/qp.hpp"
#include "sankhya/solve_control.hpp"
#include "sankhya/timer.hpp"
#include "sankhya/version.hpp"
#include "simplex/crossover.hpp"
#include "simplex/ranging.hpp"
#include "solver_engine/solver_registry.hpp"
#include "solver_engine/solver_selector.hpp"
#include "util/profiler.hpp"
#include "util/threads.hpp"

#include "../simplex/primal_simplex.hpp"

namespace sankhya {
namespace {

/// The problem class is decided from the model by the engine layer
/// (src/solver_engine/solver_engine.hpp), once, for this dispatcher and the registry alike.
using engine::classify;
using engine::ProblemClass;

/// The `algorithm` values solve() itself accepts for an LP, besides "auto" - every registered
/// engine (src/solver_engine/builtin_engines.cpp) that takes an LP and is not a GPU variant
/// (#297 full integration). GPU PDHG is reached only through "auto" (when eligible) or
/// algorithm=pdhg combined with --gpu, never by its own registered name "pdhg-gpu": #297's
/// own selector example has the user say algorithm=pdhg and hardware eligibility decide the
/// CPU/CUDA backend underneath, not a second user-facing name for it - so "pdhg-gpu" is
/// deliberately excluded here even though it IS a real, separately registered, separately
/// testable engine (src/solver_engine/solver_selector.cpp reaches it directly). This list is
/// DERIVED from the registry - adding a new LP engine there extends what solve() accepts
/// without an edit here - rather than a second, hand-maintained copy of engine names.; the
/// predicate is SolverRegistry::algorithm_names(), which `sankhya engines` and the option
/// table's test read too. An engine accepted here still needs a branch below that knows
/// how to run it, or solve() refuses it by name rather than running another in its place.
std::vector<std::string> registered_lp_algorithms() {
  return engine::SolverRegistry::builtin().algorithm_names();
}

}  // namespace

/// Force the reported status to agree with the measured quality of the point.
///
/// Every engine calls Solution::recompute_quality() before returning, so primal_infeasibility
/// and dual_infeasibility are MEASURED from the returned vectors rather than asserted by the
/// engine about itself. Nothing was checking that the status agreed with them, and the two
/// drifted apart: PDHG terminates on a RELATIVE KKT criterion at pdhg_tolerance (1e-4 by
/// default), and that was being translated straight into kOptimal. A relative KKT residual of
/// 1e-4 is not the same claim as "primal feasible to 1e-7", and on all eight Netlib instances
/// the gap between those two statements was three to five orders of magnitude. sc50b, whose
/// published optimum is exactly -70, was returned as -70.0139 and labelled optimal - a value
/// better than the optimum, which is only reachable from outside the feasible region.
///
/// The check lives here, in the dispatcher, rather than inside any one engine. It is a
/// property of the Solution contract, not of an algorithm, so the interior-point and QP
/// engines inherit it in Phase 8 instead of having to re-derive it. An engine remains free to
/// report kFeasible, kIterationLimit or anything else; what it cannot do is claim a proof
/// whose evidence is on the same object and disagrees.
///
/// Statuses that make no claim about the point are left alone.
///
/// `check_dual` is false for a MILP. A branch-and-bound incumbent is produced by a NODE LP
/// whose bounds were tightened by branching, so its reduced costs are dual feasible for that
/// node and generally are not for the original model. Optimality of a MILP is proved by the
/// bound closing against the incumbent, not by the reduced costs of the last LP solved, so
/// applying the dual test there would reject correct answers. Integrality is checked instead:
/// it is the condition that actually distinguishes a MILP solution from its relaxation.
/// Never publish an answer whose numbers are not numbers (#194).
///
/// This is the belt to the interior-point method's braces, and it is deliberately engine
/// agnostic: any solve that claims a point and then reports a non-finite objective is
/// reporting something no consumer can use and no reader can check. The benchmark runners
/// write whatever comes back into a CSV, so a NaN here does not stay here - it becomes an
/// entry in the project's evidence, in the column that exists to say whether the answer was
/// right. Downgrading is the honest outcome: the solve failed numerically, and saying so is
/// worth more than a plausible-looking row.
void refuse_a_non_finite_answer(Solution* solution, Logger& logger) {
  if (!claims_a_point(solution->status)) return;

  // A LIMITED SEARCH THAT FOUND NOTHING IS NOT A NUMERICAL FAILURE (#289). A MILP stopped by
  // a limit before it had an incumbent reports no point and the worst representable
  // objective, deliberately, so that a reader cannot take a gap of zero for a closed one
  // (see branch_and_bound.cpp). That infinity used to arrive here and be downgraded:
  // `node_limit=0` and `time_limit=0` both came back numerical_error, which says the solver
  // broke when what happened is that it was told to stop. The guard below is for an engine
  // that produces a point full of NaN, and an empty point with an infinite objective under a
  // resource status is neither.
  const bool stopped_on_a_limit = solution->status == SolveStatus::kTimeLimit ||
                                  solution->status == SolveStatus::kIterationLimit ||
                                  solution->status == SolveStatus::kNodeLimit ||
                                  solution->status == SolveStatus::kInterrupted;
  const bool values_are_numbers =
      std::all_of(solution->col_value.begin(), solution->col_value.end(),
                  [](double v) { return std::isfinite(v); });
  if (stopped_on_a_limit && values_are_numbers && std::isinf(solution->objective)) {
    // The values that came with it are whatever the vectors were allocated with, and a
    // reader must not take them for a solution, so they go - which is what this function
    // does to a rejected point too. What stays is the status, the reason and the bound.
    solution->col_value.clear();
    solution->row_activity.clear();
    // ... and so do the numbers measured on them (#505). Postsolve rebuilds a full vector
    // from what presolve fixed plus zeros and measures it, so an `ej` stopped with nothing
    // reported primal_infeasibility 1 - the violation of a point nobody claimed - into the
    // log, the .sol header and the stats JSON.
    solution->primal_infeasibility = 0.0;
    solution->primal_infeasibility_scaled = 0.0;
    solution->dual_infeasibility = 0.0;
    solution->dual_infeasibility_scaled = 0.0;
    solution->integrality_violation = 0.0;
    solution->message +=
        "; no feasible point had been found when the limit stopped the search, so none is "
        "reported (the objective is the worst representable value, not a solution)";
    return;
  }

  const bool finite = std::isfinite(solution->objective) &&
                      std::all_of(solution->col_value.begin(), solution->col_value.end(),
                                  [](double v) { return std::isfinite(v); });
  if (finite) return;

  logger.warning(
      "the solve returned a non-finite answer under status {}; reporting it as a numerical "
      "failure rather than as a point",
      to_string(solution->status));
  solution->message +=
      "; the answer contained a value that is not a number, so it is reported as a numerical "
      "failure rather than as a point (#194)";
  solution->status = SolveStatus::kNumericalError;
  solution->col_value.clear();
  solution->row_activity.clear();
  solution->objective = 0.0;
  solution->dual_bound = 0.0;
}

/// Name the resource that ended the solve, for an engine that reported the status but not
/// the reason (#289). The branch and bound sets it directly, because a limit it hits while
/// holding an incumbent is reported as kFeasible and the status can no longer say which
/// limit it was; everywhere else the status determines it.
void record_why_it_stopped(Solution* solution) {
  if (solution->stopped_by != LimitReason::kNone) return;
  switch (solution->status) {
    case SolveStatus::kTimeLimit: solution->stopped_by = LimitReason::kTime; break;
    case SolveStatus::kIterationLimit: solution->stopped_by = LimitReason::kIterations; break;
    case SolveStatus::kNodeLimit: solution->stopped_by = LimitReason::kNodes; break;
    case SolveStatus::kInterrupted: solution->stopped_by = LimitReason::kInterrupt; break;
    default: break;
  }
}

// verify_and_keep_certificate (#191) now lives in src/core/certificate.{hpp,cpp} as public
// API, shared with the SolverEngine wrappers that can produce a certificate (#297 review,
// certificate support) so a caller reaching simplex through either path gets the same check.

/// PDHG's answer, finished by the interior point (#229).
///
/// A first-order method converges linearly, with a rate that flattens as the iterate nears
/// the optimum and a floor set by floating-point noise in the step: on the scale families it
/// stops between 1e-5 and 1e-7 relative, and a million iterations do not move it (#198). A
/// second-order method started from that point converges quadratically. So when PDHG stops
/// short of the standard - at a limit, or at its own tolerance without meeting the absolute
/// one - its point, row duals and reduced costs are handed to the interior point as a
/// starting point, with a small iteration budget and whatever time the caller has left.
///
/// The polish is not free and does not pretend to be: the merged answer's iteration count is
/// the SUM of both phases, its algorithm reads "pdhg+ipm", and the polish's own count is kept
/// in polish_iterations so a benchmark row can say which phase did what. The factor is the
/// cost that can be prohibitive - the random scale family fills 17% of n^2 (#193) - and the
/// interior point measures it from the ordering before building it: above
/// polish_max_factor_nonzeros it declines, PDHG's answer stands, and the message says why.
///
/// A polished answer replaces PDHG's only when it is better - optimal, or feasible with
/// smaller scaled violations - so the polish cannot make the answer worse.
void polish_with_the_interior_point(Solution* first, const Model& model, const Options& options,
                                    Logger& logger, SolveControl* control, const Timer& timer) {
  if (!options.get_bool("pdhg_polish")) return;
  if (first->status == SolveStatus::kOptimal || !claims_a_point(*first)) return;
  const auto n = static_cast<std::size_t>(model.num_cols());
  const auto m = static_cast<std::size_t>(model.num_rows());
  if (first->col_value.size() != n || first->row_dual.size() != m ||
      first->col_dual.size() != n) {
    return;
  }

  Options polish = options;
  // THE POLISH HAS AN ITERATION BUDGET OF ITS OWN, and that is deliberate (#229): the whole
  // point of the route is that a first-order pass stopped at ITS limit is finished by a
  // second engine, and the answer's iteration count is the sum of the two, which the message
  // and `polish_iterations` both say. So iteration_limit bounds the first-order phase and
  // polish_iteration_limit bounds the finish; a two-phase route spends two budgets. It is
  // the one place where the total can exceed iteration_limit, and #289 documents it rather
  // than quietly changing what --option pdhg_polish=true was measured to do.
  Logger quiet(nullptr);
  const ResourceLimits limits(options, quiet);
  polish.set_int("iteration_limit", options.get_int("polish_iteration_limit"));
  // THE POLISH HAS A CLOCK OF ITS OWN. The factor cap above catches a factor the ordering
  // has already sized, but on the random scale family at 20,000 rows the ORDERING is the
  // cost: it ran for the whole 1,200 s limit before it could report a factor too large to
  // build, and PDHG's 1,000 iterations had taken 1.8 s. A polish that costs a thousand
  // times the solve it finishes is not a polish. So the interior point gets the smaller of
  // the time the limit has left and polish_max_seconds, and #197's deadline - which reaches
  // inside the ordering - is what enforces it. A model whose factor is affordable (the
  // staircase family) finishes in seconds; one whose factor is not is declined in
  // polish_max_seconds and the first-order answer stands, which is the honest outcome.
  double budget = options.get_double("polish_max_seconds");
  // In deterministic mode the polish is bounded by polish_max_factor_nonzeros alone (#288):
  // what the clock has left is exactly the kind of decision that mode exists to remove.
  if (!options.get_bool("deterministic") && limits.has_time_limit()) {
    const double remaining = limits.remaining_seconds(timer.elapsed_seconds());
    if (remaining <= 0.0) {
      first->message += "; no time left for the interior-point polish";
      return;
    }
    budget = std::min(budget, remaining);
  }
  polish.set_double("time_limit", budget);
  logger.info("Polish: handing PDHG's point to the interior point, {} iterations at most",
              polish.get_int("iteration_limit"));
  const ipm::WarmStart warm{first->col_value, first->row_dual, first->col_dual};
  // A polish that runs out of memory declines like one past its factor cap (#437): PDHG's
  // answer stands, and the message below says what the interior point reported.
  Solution polished = run_declining_on_out_of_memory(
      [&] {
        if (interior_point_out_of_memory_for_testing()) throw std::bad_alloc();
        return ipm::solve_ipm(model, polish, logger, control, &warm);
      },
      "ipm", timer, logger);

  const auto worst = [](const Solution& s) {
    return std::max(s.primal_infeasibility_scaled, s.dual_infeasibility_scaled);
  };
  const bool better =
      polished.status == SolveStatus::kOptimal ||
      (polished.status == SolveStatus::kFeasible && worst(polished) < worst(*first));
  if (!better) {
    first->polish_iterations = polished.iterations;
    first->message += fmt::format(
        "; the interior-point polish did not improve it ({} after {} iterations: {})",
        to_string(polished.status), polished.iterations, polished.message);
    return;
  }
  polished.polish_iterations = polished.iterations;
  polished.iterations += first->iterations;
  // The first-order phase's KKT crossings (#486) belong to the merged answer.
  polished.kkt_1e4_seconds = first->kkt_1e4_seconds;
  polished.kkt_1e6_seconds = first->kkt_1e6_seconds;
  polished.kkt_1e8_seconds = first->kkt_1e8_seconds;
  polished.kkt_1e4_iterations = first->kkt_1e4_iterations;
  polished.kkt_1e6_iterations = first->kkt_1e6_iterations;
  polished.kkt_1e8_iterations = first->kkt_1e8_iterations;
  polished.algorithm = first->algorithm + "+ipm";  // "pdhg-cpu+ipm"
  polished.message =
      fmt::format("{}; polished by the interior point in {} iterations{}", first->message,
                  polished.polish_iterations,
                  polished.message.empty() ? std::string() : " (" + polished.message + ")");
  *first = std::move(polished);
}

/// PDHG's share of a finite time limit when a polish is to follow; the rest is the polish's.
constexpr double kPdhgShareOfTheTimeLimit = 0.7;

void reconcile_status_with_measurement(Solution* solution, const Options& options,
                                       Logger& logger, bool check_dual) {
  ProfileScope timed(logger.profiler(), "verification");  // #285
  if (!claims_a_point(*solution)) return;

  const double primal_tolerance = options.get_double("primal_feasibility_tolerance");
  const double dual_tolerance = options.get_double("dual_feasibility_tolerance");
  const double integrality_tolerance = options.get_double("integrality_tolerance");

  // A point that violates its own constraints is not feasible, so neither kOptimal nor
  // kFeasible is available. The engine stopped believing it had converged, so this is a
  // numerical failure and is reported as one, with the number that contradicts it.
  //
  // The engine stopped before reaching a conclusion, so the point it returns is an
  // interim one. It is not expected to be feasible, so do not downgrade the status
  // if it is not. Only claims of optimality or feasibility are subject to measurement.
  if (solution->status != SolveStatus::kOptimal && solution->status != SolveStatus::kFeasible) {
    return;
  }

  // THE TEST IS ON THE SCALED VIOLATION, and the absolute one is still what gets printed.
  // An absolute tolerance asks a badly scaled model for accuracy it cannot have: on Netlib
  // grow7, whose largest solution value is 4.8e+07, 1e-7 absolute is 2.1e-15 relative, which
  // is below double precision's reach after three hundred iterations of arithmetic. Judging
  // that point infeasible says nothing about the point and everything about the units the
  // question was asked in. See Solution::primal_infeasibility_scaled for the derivation.
  if (solution->primal_infeasibility_scaled > primal_tolerance) {
    const std::string detail = fmt::format(
        "engine reported {} but the returned point violates primal feasibility by {:.3e} "
        "({:.3e} relative to the scale it was measured on), above the {:.1e} tolerance; it is "
        "not a feasible point",
        to_string(solution->status), solution->primal_infeasibility,
        solution->primal_infeasibility_scaled, primal_tolerance);
    solution->status = SolveStatus::kNumericalError;
    solution->message = solution->message.empty() ? detail : solution->message + "; " + detail;
    logger.warning("{}", detail);
    return;
  }

  // Integrality, for the same reason and with the same force. A branch-and-bound run that
  // reports optimal while holding a fractional integer variable has reported the relaxation,
  // which ENGINEERING_RULES.md names as the single most damaging thing this dispatcher could
  // do.
  if (solution->integrality_violation > integrality_tolerance) {
    const std::string detail = fmt::format(
        "engine reported {} but an integer column is fractional by {:.3e}, above the {:.1e} "
        "tolerance; this is a relaxation, not an integer solution",
        to_string(solution->status), solution->integrality_violation, integrality_tolerance);
    solution->status = SolveStatus::kNumericalError;
    solution->message = solution->message.empty() ? detail : solution->message + "; " + detail;
    logger.warning("{}", detail);
    return;
  }

  // Primal feasible but dual infeasible: the point is usable, the optimality claim is not
  // supported. kFeasible says exactly that and already exists for the purpose.
  // The SCALED violation decides, the absolute one is still printed - the same split as
  // the primal check above and for the same reason (#152). On grow7 the absolute dual
  // infeasibility can be 6.1 against prices of order 1e+07; judged absolutely that is a
  // failed optimality claim, judged against its own terms it is 6e-07 and the claim stands.
  if (check_dual && solution->status == SolveStatus::kOptimal &&
      solution->dual_infeasibility_scaled > dual_tolerance) {
    const std::string detail = fmt::format(
        "engine reported optimal but the reduced costs violate dual feasibility by {:.3e} "
        "({:.3e} relative to the terms they are computed from), above the {:.1e} tolerance; "
        "reporting a feasible point rather than a proof",
        solution->dual_infeasibility, solution->dual_infeasibility_scaled, dual_tolerance);
    solution->status = SolveStatus::kFeasible;
    solution->message = solution->message.empty() ? detail : solution->message + "; " + detail;
    logger.warning("{}", detail);
  }

  // Complementary slackness, judged the way the verifier judges it: absolutely (#209). The
  // relative measure below lets a row priced in the thousands sit a few 1e-10 inside its
  // bound; the verifier's absolute product does not, and a claim the verifier rejects must
  // not leave here as one. The point is usable, so it is reported feasible, not wrong.
  if (check_dual && solution->status == SolveStatus::kOptimal &&
      solution->complementarity_violation > tol::kComplementarity) {
    const std::string detail = fmt::format(
        "engine reported optimal but the largest |multiplier| * slack is {:.3e}, above the "
        "{:.1e} the independent verifier accepts; reporting a feasible point rather than a "
        "proof",
        solution->complementarity_violation, tol::kComplementarity);
    solution->status = SolveStatus::kFeasible;
    solution->message = solution->message.empty() ? detail : solution->message + "; " + detail;
    logger.warning("{}", detail);
  }
}

// apply_deterministic_mode (#288) now lives in src/core/deterministic_mode.{hpp,cpp}, shared
// with the SolverEngine wrappers (#297 review, deterministic execution context) so a caller
// reaching an engine through either solve() or the registry gets the same guarantee.

namespace {
Solution solve_unguarded(const Model& model, const Options& options, SolveControl* control,
                         Logger& logger, const Timer& timer, std::string* engine_ran);

/// The profile of a finished solve: the table into the log, and the JSON to profile_out when
/// one is named (#285). A profile that cannot be written is a warning - the solve it
/// describes has already succeeded or failed on its own terms, and must not change status
/// because a diagnostic file could not be opened.
void report_profile(Profiler& profiler, const Options& options, const Solution& solution,
                    Logger& logger) {
  // The counters every engine already keeps on the Solution, so the report carries them
  // without each engine having to call into the profiler for its headline numbers.
  profiler.count("iterations", solution.iterations);
  if (solution.nodes > 0) profiler.count("nodes", solution.nodes);
  if (solution.polish_iterations > 0)
    profiler.count("polish iterations", solution.polish_iterations);
  if (solution.cuts_applied > 0) profiler.count("cuts applied", solution.cuts_applied);
  logger.info("{}", profiler.format_text());
  const std::string path = options.get_string("profile_out");
  if (path.empty()) return;
  std::FILE* out = std::fopen(path.c_str(), "wb");
  if (out == nullptr) {
    logger.warning("profile_out: cannot open {} for writing; the profile is in the log only",
                   path);
    return;
  }
  const std::string text = profiler.format_json();
  const bool written = std::fwrite(text.data(), 1, text.size(), out) == text.size();
  if (std::fclose(out) != 0 || !written) {
    logger.warning("profile_out: writing {} failed; the profile is in the log only", path);
  }
}
}  // namespace

Solution solve(const Model& model, const Options& requested_options, SolveControl* control) {
  Timer timer;
  {
    Solution solution;
    solution.allocate_for(model);
    const std::string problem = model.validate();
    if (!problem.empty()) {
      solution.status = SolveStatus::kModelError;
      solution.message = problem;
      solution.solve_seconds = timer.elapsed_seconds();
      return solution;
    }
  }
  Logger logger(requested_options.get_bool("log_to_console") ? stdout : nullptr);
  // Rewritten once, here, so no engine below has to know about the mode - and so the log
  // says what was changed before anything runs (#288).
  const Options options = apply_deterministic_mode(requested_options, logger);
  if (options.get_bool("deterministic")) {
    logger.info("Deterministic mode: no decision depends on the clock");
    // The reproducibility fingerprint (#288): what a second run has to match for the same
    // numbers to be expected of it. No CPU model or driver version here - this binary does
    // not detect them, and a field filled with a guess is worse than a field that is absent.
    logger.info("Reproducibility: model {:016x}, seed {}, threads {}", model.fingerprint(),
                options.get_int("random_seed"), options.get_int("threads"));
    logger.info("Reproducibility: {} {} ({}), {}, CUDA {}", version_string(), git_commit(),
                build_type(), compiler_string(), cuda_enabled() ? "built in" : "not built in");
    if (control != nullptr && control->progress_callback) {
      // The one clock this mode cannot take away. The callback is due on a wall-clock
      // window, so HOW OFTEN it fires differs between runs; a callback that only reports is
      // harmless, one that interrupts decides the answer on the clock after all.
      logger.warning(
          "deterministic: a progress callback is attached and its window is wall-clock, so "
          "it fires a different number of times each run; interrupting from it makes the "
          "result non-reproducible");
    }
  }
  // The profiler lives here, for the whole solve, and rides on the logger every engine is
  // already handed (#285). Off - the default - it is never attached, and every scope in the
  // engines reduces to a null-pointer test.
  ProfileMode profile_mode = ProfileMode::kOff;
  (void)parse_profile_mode(options.get_string("profile"), &profile_mode);
  Profiler profiler(profile_mode);
  if (profile_mode != ProfileMode::kOff) logger.set_profiler(&profiler);

  // The whole dispatch runs under the out-of-memory guard (#246): an engine that exhausts
  // the machine comes back as a status with the engine named, not as an aborted process.
  // The name is read when the guard fires (#437): under `auto` the dispatch below records
  // which engine it chose, and later which one it fell back to, so the answer names the
  // engine that actually ran out rather than "solver".
  const std::string engine = options.get_string("algorithm");
  std::string engine_ran = engine == "auto" ? "solver" : engine;
  Solution solved;
  {
    ProfileScope whole(logger.profiler(), "solve");
    solved = run_engine_guarded(
        [&] { return solve_unguarded(model, options, control, logger, timer, &engine_ran); },
        [&] { return engine_ran; }, timer, logger);
  }
  if (profile_mode != ProfileMode::kOff) {
    report_profile(profiler, options, solved, logger);
    logger.set_profiler(nullptr);
  }
  return solved;
}

bool& interior_point_out_of_memory_for_testing() {
  static bool on = false;
  return on;
}

double& interior_point_extra_seconds_for_testing() {
  static double seconds = 0.0;
  return seconds;
}

namespace {
Solution solve_unguarded(const Model& model, const Options& options, SolveControl* control,
                         Logger& logger, const Timer& timer, std::string* engine_ran) {
  Solution solution;
  solution.allocate_for(model);

  apply_thread_option(options, logger);
  LogLevel level = LogLevel::kInfo;
  if (parse_log_level(options.get_string("log_level"), &level)) logger.set_level(level);
  const std::string progress_out = options.get_string("progress_out");
  if (!progress_out.empty()) logger.enable_progress_output(progress_out);

  // Every limit this solve runs under, read once and interpreted in one place (#289).
  const ResourceLimits limits(options, logger);
  // The options an engine is given when part of the budget is already spent. A no-limit
  // solve gets the options unchanged, so nothing is copied on the common path.
  const auto with_the_time_that_is_left = [&](const Options& base) -> Options {
    if (!limits.has_time_limit()) return base;
    Options narrowed = base;
    narrowed.set_double("time_limit", limits.remaining_seconds(timer.elapsed_seconds()));
    return narrowed;
  };

  // Classification and engine selection are timed as their own regions (#297), so the
  // dispatch layer's cost is measured beside the engine's rather than assumed negligible;
  // bench/runners/engine_dispatch.py reads them from profile_out.
  const ProblemClass problem_class = [&] {
    const ProfileScope timed(logger.profiler(), "classification");
    return classify(model);
  }();
  // Discovery for the classes with one engine each: is a registered engine there at all.
  const auto no_registered_engine = [&] {
    const ProfileScope timed(logger.profiler(), "engine selection");
    return engine::SolverRegistry::builtin().candidates(model).empty();
  };
  logger.info("Model {}: {} rows, {} columns, {} nonzeros, {} integer columns",
              model.name.empty() ? std::string("(unnamed)") : model.name, model.num_rows(),
              model.num_cols(), model.num_nonzeros(), model.num_integer_columns());
  logger.info("Problem class: {}", engine::to_string(problem_class));

  // AN LP ENGINE ASKED FOR ON ANOTHER CLASS IS SAID, NOT DROPPED (#297). The MILP, QP and
  // MIQP branches have never read `algorithm`: the class has one engine, so there is nothing
  // to select. A caller who set it anyway used to get no sign of that. The note goes to the
  // log now and onto the answer's message once the class's own engine has run; the engine
  // choice itself is unchanged.
  std::string engine_note;
  if (problem_class != ProblemClass::kLp) {
    const std::string requested = options.get_string("algorithm");
    if (requested != "auto") {
      const engine::SolverEngine* named = engine::SolverRegistry::builtin().find(requested);
      const char* runs = problem_class == ProblemClass::kQp ? "convex-qp" : "branch-and-bound";
      if (named == nullptr) {
        engine_note =
            fmt::format("algorithm={} is not an engine; {} ran, as the {} class decides",
                        requested, runs, engine::to_string(problem_class));
      } else if (!named->capabilities().accepts(problem_class)) {
        engine_note = fmt::format(
            "algorithm={} names an engine that does not take a {} model; {} ran, as the class "
            "decides, and the option chose nothing",
            requested, engine::to_string(problem_class), runs);
      }
      if (!engine_note.empty()) logger.warning("{}", engine_note);
    }
  }
  const auto say_which_engine_ran = [&engine_note](Solution* answer) {
    if (engine_note.empty()) return;
    answer->message =
        answer->message.empty() ? engine_note : answer->message + "; " + engine_note;
  };

  // PRESOLVE RUNS HERE, not inside an engine, and for EVERY class (#301). The reductions are
  // properties of the model, so every engine gets them, and - more importantly - postsolve
  // then re-measures the recovered point against the ORIGINAL model before the status guard
  // sees it. A reduction or postsolve bug therefore surfaces as a feasibility violation on a
  // model no engine ever touched, and the guard downgrades the status rather than letting a
  // confident answer to a different problem out of the door.
  //
  // Until #301 this wrapper existed only inside the LP branch, so a MILP, QP or MIQP was
  // handed straight to its engine with `presolve` silently ignored. What made that more than
  // an oversight is that presolve used to empty the reduced model's Hessian: running it on a
  // QP would have dropped the curvature. Columns carrying Hessian entries are now protected
  // from removal and the Hessian travels with the reduced model, so the same pipeline is
  // correct for all four classes.
  //
  // `proved` comes back true when presolve settled the model on its own; the caller then has
  // a complete Solution and returns it. The actual pipeline (#301) is shared with
  // engine::solve() (src/solver_engine/solver_engine_dispatch.cpp) through
  // run_with_presolve (src/core/presolve_pipeline.cpp, #297 full integration) - this is a
  // thin adapter that keeps the call sites below unchanged, not a second copy of the logic.
  const auto with_presolve = [&](auto&& run_engine, bool* proved) -> Solution {
    const std::function<Solution(const Model&)> run_engine_fn = run_engine;
    PresolveOutcome outcome =
        run_with_presolve(model, options, logger, timer, problem_class, run_engine_fn);
    *proved = outcome.proved;
    return std::move(outcome.solution);
  };
  bool presolve_proved_it = false;

  if (problem_class == ProblemClass::kLp) {
    const std::string requested = options.get_string("algorithm");

    // "auto" means the DUAL simplex (#65), measured rather than assumed: on the Netlib full
    // set at 120 s it passes 78/89 against the primal's 74/89, solves d6cube, modszk1 and
    // fit2p where the primal hits the limit, leaves no verifier rejection, and takes 0.37x
    // the primal's time on the 83 instances both solve (bench/results/netlib-full-dual-
    // a947a1e.csv against netlib-full-default-a947a1e.csv). The primal stays selectable as
    // "simplex". PDHG is a first-order method: it converges to a tolerance rather than to a
    // vertex, produces no basis, and on the small instances we benchmark today the simplex
    // is both faster and exact. It is selected explicitly, and it becomes the automatic
    // choice only once there is evidence for a crossover point to switch on.
    // ENGINE SELECTION (#284, #297 full integration): "auto" is a rule-based decision from
    // the model's shape, made in engine_selection.cpp and carried on the answer as
    // engine_rule/engine_reason. The paragraph above is the measurement behind the default
    // rule; the other rules name theirs in the reason. WHICH engine to run - "auto"'s rule
    // table, an explicit name validated against the registry, and the hardware-aware CPU/GPU
    // PDHG choice (#281, #282, #383) - is decided in exactly one place now,
    // engine::select() (src/solver_engine/solver_selector.cpp), which this dispatcher and a
    // caller going through the registry directly both call; there is no second copy of that
    // decision here. HOW to run the chosen engine - the PDHG-to-IPM polish, the
    // IPM-to-dual-simplex fallback, warm-start basis handling and node-scaling reuse below -
    // stays here, in solve()'s own orchestration: those are decisions ABOVE any one engine,
    // not part of any engine's own contract (#297 review: "solve() retains orchestration
    // responsibilities that legitimately belong above an engine").
    if (requested != "auto") {
      const std::vector<std::string> accepted = registered_lp_algorithms();
      if (std::find(accepted.begin(), accepted.end(), requested) == accepted.end()) {
        solution.status = SolveStatus::kNotSolved;
        solution.algorithm = "none";
        std::string list = "auto";
        for (std::size_t k = 0; k < accepted.size(); ++k) {
          list += (k + 1 == accepted.size() ? " and " : ", ") + accepted[k];
        }
        solution.message =
            fmt::format("algorithm '{}' is not an engine; {} are available", requested, list);
        logger.warning("{}", solution.message);
        solution.solve_seconds = timer.elapsed_seconds();
        return solution;
      }
    }
    const bool warm_given = control != nullptr && control->has_starting_basis();
    const engine::EngineChoice chosen = [&] {
      const ProfileScope timed(logger.profiler(), "engine selection");
      return engine::select(engine::SolverRegistry::builtin(), model, options, logger,
                            warm_given);
    }();
    if (chosen.engine == nullptr) {
      // Not reachable for a `requested` value the check above already accepted; kept as an
      // honest report rather than an assumption for "auto", per ENGINEERING_RULES.md - an
      // unimplemented path reports the truth rather than a plausible answer.
      solution.status = SolveStatus::kNotSolved;
      solution.algorithm = "none";
      solution.message = chosen.reason;
      logger.warning("{}", solution.message);
      solution.solve_seconds = timer.elapsed_seconds();
      return solution;
    }
    if (requested == "auto")
      logger.info("Engine selection: {} - {}", chosen.engine->name(), chosen.reason);
    *engine_ran = chosen.engine->name();
    const bool want_pdhg =
        chosen.engine->name() == "pdhg" || chosen.engine->name() == "pdhg-gpu";
    const bool want_ipm = chosen.engine->name() == "ipm";
    const bool want_dual = chosen.engine->name() == "dual-simplex";
    const bool want_primal = chosen.engine->name() == "simplex";
    if (!want_pdhg && !want_ipm && !want_dual && !want_primal) {
      // registered_lp_algorithms() admits every registered CPU LP engine, and the dispatch
      // below runs four of them by name with the primal simplex as its last branch. An engine
      // registered without a branch here must not fall through to the primal under its own
      // name: that would be a different engine, reported as the one asked for.
      solution.status = SolveStatus::kNotSolved;
      solution.algorithm = "none";
      solution.message = fmt::format(
          "engine '{}' is registered but solve() has no dispatch for it; it needs a branch "
          "beside the simplex, pdhg and ipm ones",
          chosen.engine->name());
      logger.warning("{}", solution.message);
      solution.solve_seconds = timer.elapsed_seconds();
      return solution;
    }
#ifdef SANKHYA_ENABLE_CUDA
    // The compute-capability, VRAM and deterministic-mode gates (#281, #282, #383) already
    // ran inside engine::select() (gpu::gpu_pdhg_is_safe, src/gpu/pdhg_gpu_guard.cu, #297
    // review A1) before it could choose "pdhg-gpu"; this is just that choice, named, so the
    // CPU-only build does not see it as unused.
    const bool use_gpu_pdhg = chosen.engine->name() == "pdhg-gpu";
#endif

    // One place runs the engine on whichever model - reduced or original - is being solved,
    // so the polish of a PDHG answer happens before postsolve in both cases.
    const auto run_lp_engine = [&](const Model& target) -> Solution {
      // THE CLOCK STARTED WHEN THE SOLVE DID (#289). Presolve has already spent some of the
      // budget by the time an engine is reached, and handing the engine the full time_limit
      // gave a solve with presolve on strictly more time than the caller allowed. The engine
      // is given what is left.
      const Options engine_options = with_the_time_that_is_left(options);
      if (want_pdhg) {
        Options first_pass = engine_options;
        const double time_limit = engine_options.get_double("time_limit");
        if (engine_options.get_bool("pdhg_polish") &&
            !engine_options.get_bool("deterministic") && time_limit > 0.0 &&
            std::isfinite(time_limit)) {
          first_pass.set_double("time_limit", time_limit * kPdhgShareOfTheTimeLimit);
        }
#ifdef SANKHYA_ENABLE_CUDA
        if (use_gpu_pdhg) {
          // GPU path: auto-routed by size:pdhg-gpu, or explicit --gpu flag (both gated by the
          // VRAM check above). Multi-GPU when gpu_devices names more than one device (#295).
          const std::vector<int> gpu_dev_ids =
              gpu::parse_device_ids(options.get_string("gpu_devices"));
          Solution first;
          if (gpu_dev_ids.size() > 1) {
            first = gpu::solve_pdhg_multi_gpu(target, first_pass, gpu_dev_ids, logger, control);
          } else {
            first = gpu::solve_pdhg_gpu(target, first_pass, logger, control);
          }
          polish_with_the_interior_point(&first, target, options, logger, control, timer);
          return first;
        }
#endif
        Solution first = pdhg::solve_pdhg(target, first_pass, logger, control);
        polish_with_the_interior_point(&first, target, engine_options, logger, control, timer);
        return first;
      }
      if (want_ipm) {
        // Under its own memory guard (#437), so an exhausted factor comes back as the
        // declined status the fallback below tests instead of unwinding past it.
        // With crossover_from_nonoptimal the interior point stops a little short of the
        // limit so a time-limited answer can still be crossed over (#474); otherwise these are
        // engine_options unchanged.
        const Options interior_options =
            interior_point_options_before_crossover(engine_options);
        Solution interior = run_declining_on_out_of_memory(
            [&] {
              if (interior_point_out_of_memory_for_testing()) throw std::bad_alloc();
              Solution answer = ipm::solve_ipm(target, interior_options, logger, control);
              if (interior_point_extra_seconds_for_testing() > 0.0) {
                std::this_thread::sleep_for(
                    std::chrono::duration<double>(interior_point_extra_seconds_for_testing()));
              }
              return answer;
            },
            "ipm", timer, logger);
        // From the interior point's answer to a vertex (#219), when asked: the basis the
        // rest of the pipeline wants, at the cost of a few pivots from an optimal point -
        // and, under crossover_from_nonoptimal, from a feasible or stopped one (#474).
        // The crossover runs on what the budget has left too (#289), and it measures that
        // itself: crossover_to_vertex() gives the pivots time_limit minus `timer`'s elapsed
        // seconds (simplex/crossover.cpp). It is therefore handed the CALLER'S options and
        // the solve's clock, as the registry's ipm engine hands them
        // (solver_engine/builtin_engines.cpp). It used to be handed
        // with_the_time_that_is_left(options), whose time_limit is already net of the
        // elapsed seconds, so they were subtracted twice and an interior point that finished
        // past half the limit left the crossover nothing: irish-electricity finished at 170 s
        // of 300 and was told "no time left for crossover" with 130 s left
        // (bench/results/mittelmann-ipm-5c7efbc.csv, #576).
        interior =
            crossover_when_wanted(target, std::move(interior), options, logger, control, timer);
        // A SELECTED interior point that declines - a factor beyond its budget, a
        // numerical failure, no answer at all - is not the end of the solve: the selector
        // chose it from the model's shape, and the shape can lie (a dense model can be
        // cheap to pivot on). The dual simplex then runs from scratch on the time that is
        // left, and the message records the fallback. A limit is not retried: the time
        // is gone either way. An explicit algorithm=ipm is reported as it came back.
        const bool declined = interior.status == SolveStatus::kNumericalError ||
                              interior.status == SolveStatus::kNotSolved;
        if (requested == "auto" && declined) {
          // Which engine takes over follows the same rule table (#356): below the row limit
          // the dual simplex is the measured default; at or above it the dual simplex is
          // the engine that already lost at that size, and the first-order method is the
          // one that reaches the optimum there (scale-e134aeb.csv, 20,000 and 100,000 rows).
          // Judged on the model as given (model.num_rows(), which is what engine::select()
          // handed to select_engine() too), not on the presolved target: the rule table was
          // applied to the original shape and the fallback follows it.
          const bool large = model.num_rows() >= kDualSimplexRowLimit;
          const char* engine_name = large ? "PDHG" : "the dual simplex";
          *engine_ran = large ? "pdhg" : "dual-simplex";
          logger.warning("the interior point declined ({}); falling back to {}",
                         interior.message, engine_name);
          Solution fallback;
          if (large) {
            Options remaining = with_the_time_that_is_left(options);
            fallback = pdhg::solve_pdhg(target, remaining, logger, control);
            polish_with_the_interior_point(&fallback, target, remaining, logger, control,
                                           timer);
          } else {
            fallback = solve_dual_simplex(target, with_the_time_that_is_left(options), logger,
                                          control);
          }
          const std::string note =
              fmt::format("the interior point declined ({}) and the solve fell back to {}",
                          interior.message.empty() ? std::string(to_string(interior.status))
                                                   : interior.message,
                          engine_name);
          fallback.message = fallback.message.empty() ? note : fallback.message + "; " + note;
          return fallback;
        }
        return interior;
      }
      return want_dual ? solve_dual_simplex(target, engine_options, logger, control)
                       : solve_primal_simplex(target, engine_options, logger, control);
    };

    if (options.get_bool("gpu") && !want_pdhg) {
      // --gpu is only supported with --algorithm pdhg (issue #17). For all other engines
      // the flag is silently ignored and the CPU path runs; the GPU path is routed above.
#ifndef SANKHYA_ENABLE_CUDA
      logger.warning(
          "--gpu requested but this build has no CUDA backend compiled in; running on CPU");
#else
      logger.warning("--gpu is only supported with --algorithm pdhg; running on CPU");
#endif
    }

    // A STARTING BASIS (#218) names the caller's rows and columns, so the engine runs on
    // the model as given: presolve is bypassed and the message says so. The dual simplex
    // is the default restart (bound and right-hand-side edits keep the old basis dual
    // feasible); `algorithm=simplex` restarts the primal (cost edits keep it primal
    // feasible). A basis the engine cannot seed is reported by it and the solve runs cold.
    const bool warm_requested = control != nullptr && control->has_starting_basis();
    // Whether the CHOSEN engine can start from a basis at all is a capability, not a
    // hand-maintained "every engine except these two" list (#297 full integration, central
    // configuration validation): today this is exactly !want_pdhg && !want_ipm since only
    // the two simplex engines declare supports_warm_start, but it now reads that from the
    // engine itself and stays correct as engines are added or removed without an edit here.
    // THE ENGINE RACE (#476), when asked for under auto: every LP engine at once, and the first
    // answer that passes an in-process check against this model wins. It runs its own
    // presolve and postsolve per engine, so what comes back is already in the model's terms.
    const bool race = requested == "auto" && engine_race_applies(options, warm_requested);
    bool race_accepted = false;
    if (race) {
      solution = run_engine_race(model, options, control, logger, timer, chosen.engine->name(),
                                 engine_ran, &race_accepted);
    } else if (warm_requested && chosen.engine->capabilities().supports_warm_start) {
      WarmStart warm;
      warm.col_status = control->start_col_status;
      warm.row_status = control->start_row_status;
      const bool lengths_fit = static_cast<Index>(warm.col_status.size()) == model.num_cols() &&
                               static_cast<Index>(warm.row_status.size()) == model.num_rows();
      if (!lengths_fit) {
        logger.warning(
            "the starting basis has {} column and {} row statuses for a model with {} "
            "columns and {} rows; ignored, solving cold",
            warm.col_status.size(), warm.row_status.size(), model.num_cols(), model.num_rows());
        solution = with_presolve(run_lp_engine, &presolve_proved_it);
      } else {
        logger.info("Warm start from the given basis; presolve bypassed");
        solution = want_dual ? solve_dual_simplex(model, options, logger, control, &warm)
                             : solve_primal_simplex(model, options, logger,
                                                    build_node_scaling(model, options), control,
                                                    &warm);
        const std::string note = "warm start from the given basis, presolve bypassed";
        solution.message = solution.message.empty() ? note : solution.message + "; " + note;
      }
    } else {
      if (warm_requested) {
        logger.warning("a starting basis was given but {} produces no basis; ignored",
                       requested);
      }
      solution = with_presolve(run_lp_engine, &presolve_proved_it);
    }
    // The decision travels on the answer, whichever path produced it (postsolve builds a
    // fresh Solution, so this is set after the engine ran, not before).
    solution.engine_rule = race ? "race" : chosen.rule;
    solution.engine_reason =
        race
            ? fmt::format("engine_race: {}; the rule table would have chosen {} ({})",
                          race_accepted
                              ? *engine_ran + "'s answer passed its check first"
                              : "no answer passed its check, " + *engine_ran + "'s is reported",
                          chosen.engine->name(), chosen.reason)
            : chosen.reason;
    if (presolve_proved_it) {
      logger.info("Result: {} (proved during presolve)  {:.3f}s", to_string(solution.status),
                  solution.solve_seconds);
      return solution;
    }
    solution.solve_seconds = timer.elapsed_seconds();
    reconcile_status_with_measurement(&solution, options, logger, /*check_dual=*/true);
    refuse_a_non_finite_answer(&solution, logger);
    record_why_it_stopped(&solution);
    say_which_engine_ran(&solution);
    verify_and_keep_certificate(&solution, model, logger);
    // Sensitivity ranging runs on the ORIGINAL model after postsolve so the vectors are
    // full-size and the basis is expressed in terms of original column and row indices.
    detail::compute_ranging(model, options, logger, solution);
    compute_iis(model, &solution, options, logger);
    logger.info("Result: {}  objective {:.10g}  {} iterations  {:.3f}s",
                to_string(solution.status), solution.objective, solution.iterations,
                solution.solve_seconds);
    logger.info("Measured primal infeasibility {:.3e}, dual infeasibility {:.3e}",
                solution.primal_infeasibility, solution.dual_infeasibility);
    return solution;
  }

  if (problem_class == ProblemClass::kMilp) {
    // Discovery through the registry (#297 full integration), NOT selection by `algorithm`:
    // this class has always ignored that option (there being one engine to consult it for),
    // and still does (#297 review, B2) - registry.candidates() answers only "does a
    // registered engine exist for this class", the same question engine::select()'s
    // "only-candidate" rule answers for a caller going through the registry directly
    // (src/solver_engine/solver_selector.cpp), without engine::select()'s algorithm-name
    // check, which WOULD wrongly refuse a MILP whenever `algorithm` happens to be set to an
    // LP-only engine's name (harmless today since solve() never read it for this class, but
    // exactly the silent-change B2 forbids).
    if (no_registered_engine()) {
      solution.status = SolveStatus::kNotSolved;
      solution.algorithm = "none";
      solution.message = "no registered engine supports MILP models";
      logger.warning("{}", solution.message);
      solution.solve_seconds = timer.elapsed_seconds();
      return solution;
    }
    {
      const mip::MilpComponents parts = mip::describe_components(options);
      logger.info(
          "MILP composition: node selection {}, branching {}, relaxation engine {}, cuts {}, "
          "heuristics {}, conflict analysis {}",
          parts.node_selection, parts.branching, parts.relaxation_engine,
          parts.cuts_enabled ? "on" : "off", parts.heuristics_enabled ? "on" : "off",
          parts.conflict_analysis_enabled ? "on" : "off");
    }
    *engine_ran = "branch-and-bound";
    solution = with_presolve(
        [&](const Model& target) {
          return mip::solve_branch_and_bound(target, with_the_time_that_is_left(options),
                                             logger, control);
        },
        &presolve_proved_it);
    if (presolve_proved_it) {
      logger.info("Result: {} (proved during presolve)  {:.3f}s", to_string(solution.status),
                  solution.solve_seconds);
      return solution;
    }
    reconcile_status_with_measurement(&solution, options, logger, /*check_dual=*/false);
    // NOT for the branch and bound's "nothing found" convention, which deliberately reports
    // the worst representable objective - an infinity there is a considered statement that
    // no point exists, not a broken number. Only a claimed POINT is checked, and that
    // convention comes with kInfeasible or a limit and no values.
    refuse_a_non_finite_answer(&solution, logger);
    record_why_it_stopped(&solution);
    say_which_engine_ran(&solution);
    logger.info("Result: {}  objective {:.10g}  bound {:.10g}  {} nodes  {:.3f}s",
                to_string(solution.status), solution.objective, solution.dual_bound,
                solution.nodes, solution.solve_seconds);
    logger.info("Measured integrality violation {:.3e}, primal infeasibility {:.3e}",
                solution.integrality_violation, solution.primal_infeasibility);
    return solution;
  }

  if (problem_class == ProblemClass::kQp) {
    // Discovery, not selection - see the identical comment in the MILP branch above.
    if (no_registered_engine()) {
      solution.status = SolveStatus::kNotSolved;
      solution.algorithm = "none";
      solution.message = "no registered engine supports QP models";
      logger.warning("{}", solution.message);
      solution.solve_seconds = timer.elapsed_seconds();
      return solution;
    }
    *engine_ran = "convex-qp";
    solution = with_presolve(
        [&](const Model& target) {
          return qp::solve_convex_qp(target, with_the_time_that_is_left(options), logger,
                                     control);
        },
        &presolve_proved_it);
    if (presolve_proved_it) {
      logger.info("Result: {} (proved during presolve)  {:.3f}s", to_string(solution.status),
                  solution.solve_seconds);
      return solution;
    }
    // check_dual is false: the QP's reduced costs are c + Qx - A'y, which is not the
    // quantity Solution::recompute_quality() tests, and applying the LP dual rule here
    // would reject correct answers. Primal feasibility and the status still have to agree.
    reconcile_status_with_measurement(&solution, options, logger, /*check_dual=*/false);
    // THE SAME IN-PROCESS KKT GATE THE LP PATH HAS (#590, as #157 for LP): a QP `optimal`
    // the check cannot back is withdrawn to `feasible` with the failing check in the
    // message, before it is written. The first Maros-Meszaros run had six answers labelled
    // optimal that the independent verifier rejected (dpklo1 with a row violated by 77,
    // stcqp1 and stcqp2 with the duals wrong); this is what stops that label leaving the
    // solver, whatever the cause behind each one.
    if (solution.status == SolveStatus::kOptimal) {
      KktTolerances tolerances;
      tolerances.primal = options.get_double("primal_feasibility_tolerance");
      tolerances.dual = options.get_double("dual_feasibility_tolerance");
      const KktVerdict verdict = check_qp_optimality(model, solution, tolerances);
      if (!verdict.passed) {
        solution.status = SolveStatus::kFeasible;
        solution.message += fmt::format(
            "{}the in-process KKT check does not back the optimal claim ({}: {}); reported "
            "as feasible, not optimal (#590)",
            solution.message.empty() ? "" : "; ", verdict.check, verdict.detail);
        logger.warning("QP: optimal withdrawn to feasible, {}: {}", verdict.check,
                       verdict.detail);
      }
    }
    refuse_a_non_finite_answer(&solution, logger);
    record_why_it_stopped(&solution);
    say_which_engine_ran(&solution);
    logger.info("Result: {}  objective {:.10g}  {} iterations  {:.3f}s",
                to_string(solution.status), solution.objective, solution.iterations,
                solution.solve_seconds);
    logger.info("Measured primal infeasibility {:.3e}", solution.primal_infeasibility);
    return solution;
  }

  if (problem_class == ProblemClass::kMiqp) {
    // MIQP is branch and bound over QP node relaxations - the two engines joined, which is
    // exactly what the message this replaces said was missing. The QP engine refuses a
    // non-convex Hessian before any arithmetic starts, so a non-convex MIQP is still refused
    // rather than solved to a local point; that check now happens at the first node.
    //
    // check_dual stays false for the same reason it is false for a MILP: the reduced costs
    // belong to a node whose bounds branching tightened, and for a QP they are c + Qx - A'y
    // rather than the quantity recompute_quality() measures. Integrality and primal
    // feasibility are what distinguish an MIQP answer from its relaxation, and both are
    // checked.
    //
    // Discovery, not selection - see the identical comment in the MILP branch above.
    if (no_registered_engine()) {
      solution.status = SolveStatus::kNotSolved;
      solution.algorithm = "none";
      solution.message = "no registered engine supports MIQP models";
      logger.warning("{}", solution.message);
      solution.solve_seconds = timer.elapsed_seconds();
      return solution;
    }
    {
      const mip::MilpComponents parts = mip::describe_components(options);
      logger.info(
          "MILP composition: node selection {}, branching {}, relaxation engine {}, cuts {}, "
          "heuristics {}, conflict analysis {}",
          parts.node_selection, parts.branching, parts.relaxation_engine,
          parts.cuts_enabled ? "on" : "off", parts.heuristics_enabled ? "on" : "off",
          parts.conflict_analysis_enabled ? "on" : "off");
    }
    *engine_ran = "branch-and-bound";
    solution = with_presolve(
        [&](const Model& target) {
          return mip::solve_branch_and_bound(target, with_the_time_that_is_left(options),
                                             logger, control);
        },
        &presolve_proved_it);
    if (presolve_proved_it) {
      logger.info("Result: {} (proved during presolve)  {:.3f}s", to_string(solution.status),
                  solution.solve_seconds);
      return solution;
    }
    reconcile_status_with_measurement(&solution, options, logger, /*check_dual=*/false);
    refuse_a_non_finite_answer(&solution, logger);
    record_why_it_stopped(&solution);
    say_which_engine_ran(&solution);
    logger.info("Result: {}  objective {:.10g}  bound {:.10g}  {} nodes  {:.3f}s",
                to_string(solution.status), solution.objective, solution.dual_bound,
                solution.nodes, solution.solve_seconds);
    logger.info("Measured integrality violation {:.3e}, primal infeasibility {:.3e}",
                solution.integrality_violation, solution.primal_infeasibility);
    return solution;
  }

  solution.status = SolveStatus::kNotSolved;
  solution.algorithm = "none";
  solution.message =
      fmt::format("no engine is implemented for {} yet", engine::to_string(problem_class));
  logger.warning("{}", solution.message);

  solution.solve_seconds = timer.elapsed_seconds();
  return solution;
}
}  // namespace

}  // namespace sankhya
