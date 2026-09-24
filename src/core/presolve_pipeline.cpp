// SPDX-License-Identifier: Apache-2.0
#include "core/presolve_pipeline.hpp"

#include <algorithm>
#include <string>
#include <utility>

#include <fmt/format.h>

#include "mip/debug_solution.hpp"
#include "presolve/presolve.hpp"
#include "sankhya/certificate.hpp"
#include "sankhya/io.hpp"
#include "util/profiler.hpp"

namespace sankhya {

PresolveOutcome run_with_presolve(const Model& model, const Options& options, Logger& logger,
                                  const Timer& timer, engine::ProblemClass problem_class,
                                  const std::function<Solution(const Model&)>& run_engine) {
  PresolveOutcome outcome;

  // A COMPLETE SOLUTION POOL AND PRESOLVE ASK FOR DIFFERENT THINGS (#301 meeting #225).
  // pool_complete promises the k best integer assignments OF THE MODEL THE CALLER HANDED
  // OVER. Presolve's column removals are optimality arguments as much as feasibility ones -
  // an empty column is parked at the bound its cost prefers, a free singleton is
  // substituted at the end of its interval - so the assignments they settle are exactly the
  // alternatives the pool was asked to enumerate. Keeping both would report "the four best
  // plans" for a model with columns already spent. The complete pool wins, and says so.
  const bool mixed_integer = problem_class == engine::ProblemClass::kMilp ||
                             problem_class == engine::ProblemClass::kMiqp;
  // THE DEBUG-SOLUTION CHECK (#500) reads its point by column NAME, which is what lets it
  // follow presolve's renumbering. An unnamed column is written as C<index>, and presolve
  // would carry that name to a different index, so the columns are named first.
  const bool debug_check = mixed_integer && !options.get_string("debug_solution").empty();
  const bool unnamed = model.col_names.size() != static_cast<std::size_t>(model.num_cols()) ||
                       std::any_of(model.col_names.begin(), model.col_names.end(),
                                   [](const std::string& name) { return name.empty(); });
  if (debug_check && unnamed) {
    return run_with_presolve(mip::with_column_names(model), options, logger, timer,
                             problem_class, run_engine);
  }
  if (mixed_integer && options.get_bool("presolve") && options.get_bool("pool_complete")) {
    const char* why =
        "pool_complete enumerates the best assignments of the model as given, and presolve "
        "settles some of those columns before the search sees them";
    logger.info("Presolve skipped: {}", why);
    outcome.solution = run_engine(model);
    outcome.solution.presolve_report.skipped_because = why;
    return outcome;
  }
  if (mixed_integer && options.get_bool("presolve") &&
      !options.get_string("write_certificate").empty()) {
    const char* why =
        "write_certificate proves the answer for the model as given, and presolve would hand "
        "the search a different one (#518)";
    logger.info("Presolve skipped: {}", why);
    outcome.solution = run_engine(model);
    outcome.solution.presolve_report.skipped_because = why;
    return outcome;
  }
  if (!options.get_bool("presolve")) {
    if (!options.get_string("write_presolved").empty()) {
      logger.warning(
          "write_presolved: presolve is off, so there is no presolved model to write; "
          "nothing was written");
    }
    // ran stays false, and the reason is the option rather than a decision made here, so
    // skipped_because is left empty (#286).
    ProfileScope timed(logger.profiler(), "engine");
    outcome.solution = run_engine(model);
    return outcome;
  }

  const presolve::Result reduced = [&] {
    ProfileScope timed(logger.profiler(), "presolve");
    return presolve::presolve(model, options, logger);
  }();
  // Every reduction against the known point (#500), before anything acts on the result.
  if (debug_check) mip::check_presolve_against_debug_solution(model, reduced, options, logger);
  if (reduced.proved_infeasible) {
    Solution proof;
    proof.allocate_for(model);
    proof.status = SolveStatus::kInfeasible;
    proof.algorithm = "presolve";
    // The same bound convention the engines use for an infeasible verdict (#299): the
    // worst value the objective can take, on the model's own sense.
    proof.dual_bound = model.sense == ObjSense::kMaximize ? -kInfinity : kInfinity;
    // Presolve's proof is a small Farkas argument over the rows it used (#253), handed
    // over as a candidate and checked here against the ORIGINAL model exactly as an
    // engine's certificate is; one that does not hold (a contradiction that needed
    // integrality rounding, say) is dropped and the message says so.
    proof.message = reduced.message + "; proved by presolve";
    if (mixed_integer) {
      // The branch and bound's convention for "nothing was found", which presolve's proof
      // has to match now that it can settle a MILP: the worst representable objective and
      // infinite gaps. Leaving them at zero would print `gap 0.00e+00` beside a model that
      // has no point at all, and a gap of zero reads as a closed search.
      const double nothing_found = model.sense == ObjSense::kMaximize ? -kInfinity : kInfinity;
      proof.objective = nothing_found;
      proof.absolute_gap = kInfinity;
      proof.relative_gap = kInfinity;
    }
    proof.farkas_dual = reduced.farkas_dual;
    proof.presolve_report = reduced.report;
    verify_and_keep_certificate(&proof, model, logger);
    proof.solve_seconds = timer.elapsed_seconds();
    outcome.proved = true;
    outcome.solution = std::move(proof);
    return outcome;
  }

  // Dump the presolved model when --option write_presolved=<path> is set.
  const std::string presolved_path = options.get_string("write_presolved");
  if (!presolved_path.empty()) {
    std::string write_error;
    if (!io::write_model(presolved_path, reduced.model, &write_error)) {
      logger.warning("write_presolved: {}", write_error);
    } else {
      logger.info("Presolved model written to {}", presolved_path);
    }
  }

  Solution inner = [&] {
    ProfileScope timed(logger.profiler(), "engine");
    return run_engine(reduced.model);
  }();
  ProfileScope timed(logger.profiler(), "postsolve");
  outcome.solution = presolve::postsolve(reduced, model, inner);
  return outcome;
}

}  // namespace sankhya
