// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the reduced-cost polish of a simplex optimum (#548).
//
// A basis the simplex calls optimal may carry reduced costs of the wrong sign up to the dual
// feasibility tolerance (1e-7). Each such entry is an improving direction left untaken, and
// on a badly scaled model the objective pays for them: pilot87 ended at 301.71069277 with
// its worst reduced cost 6.9e-8 on the wrong side, 1.14e-6 above Koch's exact optimum
// 301.71034733 (The final NETLIB-LP results, Oper. Res. Lett. 32, 2004). Nothing was loose
// but the tolerance; the same solve at 1e-9 lands on the exact optimum.
//
// So the optimum is finished by the primal simplex, restarted from its own basis on the
// same model, unscaled, with the dual tolerance tightened to tol::kReducedCostPolish. This
// is the standard finish of a simplex code under a relaxed pricing tolerance (Koberstein,
// "The dual simplex method, techniques for a fast and stable implementation", PhD thesis,
// Paderborn 2005, sec. 6.2: remove the relaxation and continue with the primal simplex);
// the basis stays primal feasible, so only improving pivots remain. The polished answer is
// kept only when it is optimal, primal feasible to the caller's tolerance and its dual
// infeasibility is smaller; otherwise the original stands.

#include <string>

#include <fmt/format.h>

#include "primal_simplex.hpp"
#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya {

Solution polish_reduced_costs(const Model& model, Solution solution, const Options& options,
                              Logger& logger, SolveControl* control) {
  if (solution.status != SolveStatus::kOptimal) return solution;
  if (solution.dual_infeasibility <= tol::kReducedCostPolish) return solution;
  if (options.get_double("dual_feasibility_tolerance") <= tol::kReducedCostPolish) {
    return solution;
  }
  if (solution.col_status.size() != static_cast<std::size_t>(model.num_cols()) ||
      solution.row_status.size() != static_cast<std::size_t>(model.num_rows())) {
    return solution;
  }
  Options tight = options;
  tight.set_double("dual_feasibility_tolerance", tol::kReducedCostPolish);
  tight.set_bool("scaling", false);
  const double time_limit = options.get_double("time_limit");
  if (time_limit < 1e300) {
    // The caller's options already carry the time that is left for the solve; a polish that
    // runs out of it is discarded below, so the answer is never worse than the one given.
    tight.set_double("time_limit", time_limit);
  }
  WarmStart warm;
  warm.col_status = solution.col_status;
  warm.row_status = solution.row_status;
  logger.info(
      "Reduced-cost polish (#548): dual infeasibility {:.3e}, restarting the primal "
      "simplex from the optimal basis at {:.0e}",
      solution.dual_infeasibility, tol::kReducedCostPolish);
  Timer clock;
  Solution polished = solve_primal_simplex(model, tight, logger, NodeScaling{}, control, &warm);
  const double primal_tolerance = options.get_double("primal_feasibility_tolerance");
  const bool better = polished.status == SolveStatus::kOptimal &&
                      polished.primal_infeasibility <= primal_tolerance &&
                      polished.dual_infeasibility < solution.dual_infeasibility;
  if (!better) {
    logger.info("Reduced-cost polish: {} after {} pivots, kept the unpolished optimum",
                to_string(polished.status), polished.iterations);
    return solution;
  }
  const std::string note = fmt::format(
      "reduced-cost polish: {} pivot(s) in {:.2f}s took the dual infeasibility from {:.1e} "
      "to {:.1e} and the objective from {:.12g} to {:.12g} (#548)",
      polished.iterations, clock.elapsed_seconds(), solution.dual_infeasibility,
      polished.dual_infeasibility, solution.objective, polished.objective);
  logger.info("{}", note);
  polished.iterations += solution.iterations;
  polished.algorithm = solution.algorithm;
  polished.message = solution.message.empty() ? note : solution.message + "; " + note;
  return polished;
}

}  // namespace sankhya
