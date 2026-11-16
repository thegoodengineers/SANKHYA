// SPDX-License-Identifier: Apache-2.0
// SANKHYA - optimality-based bound tightening for the global search (#514, `global_obbt`).
//
// THE METHOD. A product's McCormick envelope is only as tight as the box of its factors. FBBT
// (fbbt.cpp) tightens that box one row at a time; OBBT asks the whole relaxation instead: for
// each column x_j of a product, minimize and maximize x_j over the McCormick relaxation of the
// box plus the row objective <= incumbent, and move x_j's bounds there. Zamora and Grossmann
// (J. Global Optim. 14, 1999) and Tawarmalani and Sahinidis (Math. Programming 99, 2004) for
// the method; Gleixner, Berthold, Muller and Weltge (J. Global Optim. 67, 2017) for skipping a
// bound when the last LP's point already sits on it, since no LP over the same rows can then
// move it.
//
// EVERY BOUND IS PROVED. The bound applied is not the LP's objective but the Lagrangian at its
// multipliers (dual_bound_from_multipliers, relaxation.cpp), valid for any multipliers, relaxed
// outward by kGlobalObbtSafety for the rounding in that sum. The cutoff row removes only points
// whose objective is above the incumbent, which cannot beat it; the search's proven bound is
// the least of its boxes and the incumbent, so no claim rests on the removed part.
//
// NOT UNDER A CERTIFICATE. The bound certificate's checker (tools/verify_global_certificate.py)
// splits the model's own box; it does not re-derive a tightening, so a certified search uses
// neither FBBT nor this.

#include <algorithm>
#include <cmath>
#include <vector>

#include "global/global_internal.hpp"
#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::global {

ObbtOutcome obbt(const Problem& problem, double cutoff, const Options& options, double seconds,
                 Box* box) {
  ObbtOutcome out;
  Timer timer;
  const auto n = static_cast<std::size_t>(problem.n);
  const auto near = [](double value, double bound) {
    return std::fabs(value - bound) <=
           tol::kGlobalObbtAtBound * std::max(1.0, std::fabs(bound));
  };
  // Settled sides: not a product column, already too narrow to matter, or shown not to move.
  std::vector<char> lower_done(n, 1);
  std::vector<char> upper_done(n, 1);
  for (std::size_t j = 0; j < n; ++j) {
    if (problem.in_product[j] == 0) continue;
    const double width = box->upper[j] - box->lower[j];
    if (width <= tol::kGlobalMinBranchWidth * std::max(1.0, std::fabs(box->upper[j]))) continue;
    lower_done[j] = 0;
    upper_done[j] = 0;
  }

  Model lp = build_relaxation(problem, *box, cutoff);
  bool stale = false;  // the box moved since `lp` was built
  std::vector<BasisStatus> col_status;
  std::vector<BasisStatus> row_status;
  for (std::size_t j = 0; j < n; ++j) {
    for (const bool lower_side : {true, false}) {
      std::vector<char>& done = lower_side ? lower_done : upper_done;
      if (done[j] != 0) continue;
      done[j] = 1;
      const double left = seconds - timer.elapsed_seconds();
      if (left <= 0.0) return out;
      if (stale) {
        // Same shape, so the last basis still starts the next LP.
        lp = build_relaxation(problem, *box, cutoff);
        stale = false;
      }
      std::fill(lp.col_cost.begin(), lp.col_cost.end(), 0.0);
      lp.objective_offset = 0.0;
      lp.col_cost[j] = lower_side ? 1.0 : -1.0;
      const Solution solved =
          solve_lp(lp, inner_lp_options(options, left), &col_status, &row_status);
      ++out.lps;
      out.iterations += solved.iterations;
      if (solved.status == SolveStatus::kInfeasible) {  // kept only with a checked Farkas proof
        out.infeasible = true;
        return out;
      }
      if (solved.status != SolveStatus::kOptimal) continue;
      col_status = solved.col_status;
      row_status = solved.row_status;
      if (solved.col_value.size() == static_cast<std::size_t>(lp.num_cols())) {
        for (std::size_t k = 0; k < n; ++k) {
          if (near(solved.col_value[k], box->lower[k])) lower_done[k] = 1;
          if (near(solved.col_value[k], box->upper[k])) upper_done[k] = 1;
        }
      }
      // A lower bound on +x_j (minimizing) or on -x_j (maximizing).
      const double proved = dual_bound_from_multipliers(lp, solved.row_dual);
      if (!std::isfinite(proved)) continue;
      const double safe = proved - tol::kGlobalObbtSafety * std::max(1.0, std::fabs(proved));
      double& lower = box->lower[j];
      double& upper = box->upper[j];
      const double candidate = lower_side ? safe : -safe;
      const double gain = lower_side ? candidate - lower : upper - candidate;
      const double scale = std::max(1.0, std::fabs(lower_side ? lower : upper));
      if (!(gain > tol::kFbbtMinImprovement * scale)) continue;
      // The LP's own point is in the box, so a proved bound past the other end is rounding.
      if (lower_side) {
        lower = std::min(candidate, upper);
      } else {
        upper = std::max(candidate, lower);
      }
      ++out.tightened;
      stale = true;
    }
  }
  return out;
}

}  // namespace sankhya::global
