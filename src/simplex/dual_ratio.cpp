// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the dual simplex's ratio tests and its cost perturbation (#65, #465).
//
// Split out of dual_simplex.cpp, which keeps the iteration loop.
//
// References
//   Maros, "Computational Techniques of the Simplex Method" (Kluwer, 2003), ch. 9 (cost
//     perturbation against degeneracy) and ch. 10 (the bound-flipping ratio test).
//   Koberstein, A. (2005), "The dual simplex method, techniques for a fast and stable
//     implementation", PhD thesis, Universitaet Paderborn, ch. 6 - the bound-flipping ratio
//     test with Harris's tolerance and cost shifting, and the cost perturbation applied
//     before the first iteration. Taken from the thesis text only: docs/PROVENANCE.md,
//     judgement call 12.
//   Harris, P.M.J. (1973), "Pivot selection methods of the Devex LP code", Mathematical
//     Programming 5, 1-28 - the two-pass ratio test.
//
// HARRIS INSIDE THE BOUND-FLIPPING TEST (#465). The textbook test sorts the breakpoints
// t_j = d_j / |alpha_rj| and passes them one at a time while the dual objective's slope stays
// positive; the column it stops at enters, ties going to the largest |alpha_rj|. A tie there
// is exact to 1e-9, so on a degenerate row the pivot is whatever is smallest by rounding,
// and a small pivot is how a basis decays. Harris widens the tie: pass one bounds the step by
// the tightest RELAXED breakpoint, (d_j + delta) / |alpha_rj| with delta =
// kDualHarrisRelaxation, and pass two takes every breakpoint whose exact ratio is under that
// bound as one group. A group of boxed columns the slope can pay for is flipped whole and the
// test moves on to the next group; the first group it cannot pass supplies the entering
// column, the largest |alpha_rj| in it. The other columns of that group are passed by at most
// delta / |alpha_rj| of step, so their reduced costs end at most delta on the wrong side,
// which is inside the dual tolerance. The entering column's own reduced cost can already be
// on the wrong side by up to delta, which would make the dual step negative and undo
// progress; instead the loop SHIFTS its cost by exactly that amount (shift_cost()), the step
// is zero, and the shifted costs are carried like a perturbation: restored before any answer
// and finished by the primal simplex, never reported.
//
// PERTURBATION AT THE START (#465). The stall perturbation in perturb_costs() waits for
// 500 consecutive degenerate pivots. A model whose costs take only a handful of distinct
// values - brazil3's are mostly zero - is dual degenerate from the first pivot, so
// perturb_costs_at_start() moves every nonbasic structural cost before the loop begins.

#include "simplex_core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "sankhya/tolerances.hpp"

namespace sankhya::detail {

namespace {

/// COST PERTURBATION, the dual's analogue of the primal loop's bound perturbation. A dual
/// degenerate vertex has many reduced costs at exactly zero, so the dual ratio test ties
/// and the dual step is zero; dfl001 spent 1001 consecutive iterations that way and was
/// handed to the primal, which then needed 15,000 phase-1 iterations. Each NONBASIC cost is
/// shifted by this fraction of max(1, |c_j|), times the same deterministic per-variable
/// factor the bound perturbation uses, in the direction that keeps its reduced cost dual
/// feasible. Nonbasic only: y = B^-T c_B is untouched, so every other reduced cost is too.
///
/// 1e-6 and not kPerturbationSize's 1e-9: the shift has to exceed the dual tolerance the
/// ratio test judges ties by (1e-7), or it changes nothing the test can see.
constexpr double kDualCostPerturbation = 1e-6;

}  // namespace

DualRatioResult Simplex::dual_ratio_test(Index leaving_slot, bool leaving_to_upper) const {
  return dual_harris_ ? dual_ratio_test_harris(leaving_slot, leaving_to_upper)
                      : dual_ratio_test_textbook(leaving_slot, leaving_to_upper);
}

DualRatioResult Simplex::dual_ratio_test_textbook(Index leaving_slot,
                                                  bool leaving_to_upper) const {
  // The leaving variable x_r is outside its bounds. Moving the duals by t along rho changes
  // every nonbasic reduced cost by d_j <- d_j - s t alpha_rj, where s = +1 when x_r is above
  // its upper bound and -1 when below its lower one (so that the leaving variable's own new
  // reduced cost, -s t, has the sign its destination bound demands). A nonbasic column
  // blocks when its reduced cost would cross zero: at its lower bound when s alpha_rj > 0,
  // at its upper bound when s alpha_rj < 0, and a free column for either sign. The ratio is
  // d_j / (s alpha_rj), never negative - a reduced cost a hair on the wrong side of zero is
  // treated as zero rather than as a step backwards.
  DualRatioResult result;
  const double s = leaving_to_upper ? 1.0 : -1.0;

  struct Candidate {
    Index column;
    double ratio;
    double alpha_abs;
    double range;  ///< upper - lower when both are finite, else infinite
  };
  std::vector<Candidate> candidates;
  candidates.reserve(64);

  // THE PIVOT FLOOR IS RELATIVE TO THE ROW (#244). An absolute 1e-9 floor is a statement
  // about scaled rows; on an unscaled one with entries of order 1e+3 it admits a pivot five
  // orders below its neighbours, and the basis the next factorization sees is singular. So
  // the floor is the larger of the absolute tolerance and a fraction of the row's largest
  // entry among the columns that could enter.
  double alpha_max = 0.0;
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (basis_position_[u] >= 0 || status_[u] == BasisStatus::kFixed) continue;
    alpha_max = std::max(alpha_max, std::fabs(pivot_row_[u]));
  }
  const double pivot_floor =
      std::max(tol::kPivotTolerance, tol::kDualPivotRelativeFloor * alpha_max);

  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (basis_position_[u] >= 0) continue;
    if (status_[u] == BasisStatus::kFixed) continue;
    const double a = s * pivot_row_[u];
    if (std::fabs(a) <= pivot_floor) continue;
    bool eligible = false;
    switch (status_[u]) {
      case BasisStatus::kAtLower: eligible = a > 0.0; break;
      case BasisStatus::kAtUpper: eligible = a < 0.0; break;
      case BasisStatus::kNonbasicFree: eligible = true; break;
      case BasisStatus::kFixed:
      case BasisStatus::kBasic:
      case BasisStatus::kUnknown: break;
    }
    if (!eligible) continue;
    double ratio = reduced_cost_[u] / a;
    if (ratio < 0.0) ratio = 0.0;
    const double lo = lower_[u];
    const double hi = upper_[u];
    const double range = is_finite_bound(lo) && is_finite_bound(hi)
                             ? hi - lo
                             : std::numeric_limits<double>::infinity();
    candidates.push_back({k, ratio, std::fabs(a), range});
  }

  if (candidates.empty()) {
    result.dual_unbounded = true;
    return result;
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& x, const Candidate& y) { return x.ratio < y.ratio; });

  // BOUND FLIPPING (Maros ch. 10, Koberstein sec. 3.3). The dual objective along the step
  // is piecewise linear and concave: its slope starts at the leaving row's primal
  // infeasibility and drops by |alpha_rj| * (upper_j - lower_j) each time a BOXED column's
  // reduced cost crosses zero - because that column can simply be moved to its other bound
  // and stay dual feasible. So the step runs past every boxed breakpoint while the slope
  // stays positive, flipping those columns, and stops at the first breakpoint it cannot
  // pass: a column with only one bound, or one whose flip would turn the slope negative.
  // That column enters. On a MILP relaxation nearly every column is boxed, and one dual
  // pivot then does the work of many.
  const Index leaving = basis_[static_cast<std::size_t>(leaving_slot)];
  const double x_r = x_basic_[static_cast<std::size_t>(leaving_slot)];
  double slope = leaving_to_upper ? x_r - upper_[static_cast<std::size_t>(leaving)]
                                  : lower_[static_cast<std::size_t>(leaving)] - x_r;

  std::size_t stop = 0;
  while (stop < candidates.size()) {
    const Candidate& candidate = candidates[stop];
    if (!std::isfinite(candidate.range)) break;
    const double drop = candidate.alpha_abs * candidate.range;
    // Flip only while the row stays infeasible beyond tolerance afterwards; a flip that
    // lands the row inside its bounds is the whole step, and the pivot is not needed.
    if (slope - drop <= primal_tolerance_) break;
    slope -= drop;
    result.flips.push_back(candidate.column);
    ++stop;
  }
  if (stop == candidates.size()) {
    // Every candidate was boxed and passed, and the dual objective still climbs: the dual
    // ray is unbounded, so the primal is infeasible. The flips are moot.
    result.flips.clear();
    result.dual_unbounded = true;
    return result;
  }

  // Among the candidates tied at the stopping ratio, the largest pivot - the same tie-break
  // the primal ratio test uses, for the same reason: a tiny pivot element is how a basis
  // decays. (A Harris two-pass window here was measured for #244 and set aside: without
  // cost shifting its tolerance-sized wrong-signed reduced costs accumulate across pivots,
  // 6.8e-6 on etamacro, and the status guard downgrades the claim. #465 brings it back WITH
  // the cost shift, as dual_ratio_test_harris() below.)
  const double stop_ratio = candidates[stop].ratio;
  std::size_t best = stop;
  for (std::size_t i = stop; i < candidates.size(); ++i) {
    if (candidates[i].ratio > stop_ratio + tol::kRatioTestFeasibility) break;
    if (candidates[i].alpha_abs > candidates[best].alpha_abs) best = i;
  }
  result.entering = candidates[best].column;
  return result;
}

DualRatioResult Simplex::dual_ratio_test_harris(Index leaving_slot,
                                                bool leaving_to_upper) const {
  // The same breakpoints as the textbook test above: moving the duals by t along rho moves
  // d_j by -s t alpha_rj, and column j blocks when that drives d_j through zero. Writing
  // a_j = s alpha_rj and g_j = d_j sign(a_j) - the reduced cost's distance from zero on the
  // side its status admits - the exact breakpoint is max(g_j, 0) / |a_j| and the relaxed one
  // max(g_j + delta, 0) / |a_j|, never below the exact one.
  DualRatioResult result;
  const double s = leaving_to_upper ? 1.0 : -1.0;

  struct Candidate {
    Index column;
    double ratio;    ///< exact breakpoint
    double relaxed;  ///< breakpoint with the reduced cost relaxed by kDualHarrisRelaxation
    double alpha_abs;
    double range;  ///< upper - lower when both are finite, else infinite
  };
  std::vector<Candidate> candidates;
  candidates.reserve(64);

  // The pivot floor relative to the row (#244), exactly as the textbook test applies it.
  double alpha_max = 0.0;
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (basis_position_[u] >= 0 || status_[u] == BasisStatus::kFixed) continue;
    alpha_max = std::max(alpha_max, std::fabs(pivot_row_[u]));
  }
  const double pivot_floor =
      std::max(tol::kPivotTolerance, tol::kDualPivotRelativeFloor * alpha_max);

  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (basis_position_[u] >= 0) continue;
    if (status_[u] == BasisStatus::kFixed) continue;
    const double a = s * pivot_row_[u];
    if (std::fabs(a) <= pivot_floor) continue;
    bool eligible = false;
    switch (status_[u]) {
      case BasisStatus::kAtLower: eligible = a > 0.0; break;
      case BasisStatus::kAtUpper: eligible = a < 0.0; break;
      case BasisStatus::kNonbasicFree: eligible = true; break;
      case BasisStatus::kFixed:
      case BasisStatus::kBasic:
      case BasisStatus::kUnknown: break;
    }
    if (!eligible) continue;
    const double alpha_abs = std::fabs(a);
    const double g = a > 0.0 ? reduced_cost_[u] : -reduced_cost_[u];
    const double ratio = std::max(g, 0.0) / alpha_abs;
    const double relaxed = std::max(g + tol::kDualHarrisRelaxation, 0.0) / alpha_abs;
    const double lo = lower_[u];
    const double hi = upper_[u];
    const double range = is_finite_bound(lo) && is_finite_bound(hi)
                             ? hi - lo
                             : std::numeric_limits<double>::infinity();
    candidates.push_back({k, ratio, relaxed, alpha_abs, range});
  }

  if (candidates.empty()) {
    result.dual_unbounded = true;
    return result;
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& x, const Candidate& y) { return x.ratio < y.ratio; });

  // PASS ONE, for every group at once: the Harris bound of the candidates still in play from
  // position i on is the smallest relaxed breakpoint among them, a suffix minimum. The group
  // starting at i is then every candidate whose EXACT breakpoint is under that bound - never
  // empty, since the first one's exact breakpoint is at most the minimiser's exact one, which
  // is at most its relaxed one.
  std::vector<double> bound(candidates.size());
  double running = std::numeric_limits<double>::infinity();
  for (std::size_t i = candidates.size(); i-- > 0;) {
    running = std::min(running, candidates[i].relaxed);
    bound[i] = running;
  }

  const Index leaving = basis_[static_cast<std::size_t>(leaving_slot)];
  const double x_r = x_basic_[static_cast<std::size_t>(leaving_slot)];
  double slope = leaving_to_upper ? x_r - upper_[static_cast<std::size_t>(leaving)]
                                  : lower_[static_cast<std::size_t>(leaving)] - x_r;

  std::size_t first = 0;
  while (first < candidates.size()) {
    std::size_t end = first + 1;
    while (end < candidates.size() && candidates[end].ratio <= bound[first]) ++end;
    // A group of boxed columns is passed whole, by flipping each, when the slope stays
    // positive beyond the primal tolerance afterwards - the textbook rule applied to the
    // group instead of the single breakpoint.
    double drop = 0.0;
    bool all_boxed = true;
    for (std::size_t i = first; i < end; ++i) {
      if (!std::isfinite(candidates[i].range)) {
        all_boxed = false;
        break;
      }
      drop += candidates[i].alpha_abs * candidates[i].range;
    }
    if (all_boxed && slope - drop > primal_tolerance_) {
      slope -= drop;
      for (std::size_t i = first; i < end; ++i) result.flips.push_back(candidates[i].column);
      first = end;
      continue;
    }
    // PASS TWO: the largest pivot of the group the step cannot pass.
    std::size_t best = first;
    for (std::size_t i = first + 1; i < end; ++i) {
      if (candidates[i].alpha_abs > candidates[best].alpha_abs) best = i;
    }
    result.entering = candidates[best].column;
    return result;
  }
  // Every group was boxed and passed and the dual objective still climbs: the dual ray is
  // unbounded, so the primal is infeasible. The flips are moot.
  result.flips.clear();
  result.dual_unbounded = true;
  return result;
}

void Simplex::shift_cost(Index k, double amount) {
  if (!cost_perturbed_ && !cost_shifted_) unperturbed_cost_ = cost_;
  const auto u = static_cast<std::size_t>(k);
  cost_[u] += amount;
  reduced_cost_[u] += amount;
  cost_shifted_ = true;
  ++cost_shifts_;
}

// -----------------------------------------------------------------------------------------
// Cost perturbation
// -----------------------------------------------------------------------------------------

void Simplex::perturb_costs() {
  if (cost_perturbed_) return;
  if (!cost_shifted_) unperturbed_cost_ = cost_;
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (basis_position_[u] >= 0) continue;
    // The per-variable factor in (0.25, 1] that perturbation_for() encodes, rescaled from
    // the bound perturbation's size to the cost one's, times the cost's own magnitude.
    const double factor = perturbation_for(k) / kPerturbationSize;
    const double shift = kDualCostPerturbation * factor * std::max(1.0, std::fabs(cost_[u]));
    if (status_[u] == BasisStatus::kAtLower) {
      cost_[u] += shift;  // d_j = c_j - a_j^T y grows: further inside dual feasibility
    } else if (status_[u] == BasisStatus::kAtUpper) {
      cost_[u] -= shift;  // d_j shrinks: further inside on the upper side
    }
    // Fixed and free nonbasic columns are left alone: a fixed one has no sign condition
    // to protect, and a free one must keep its reduced cost at zero.
  }
  cost_perturbed_ = true;
  ++cost_perturbations_;
  compute_reduced_costs(false);
}

bool Simplex::perturb_costs_at_start() {
  if (cost_perturbed_ || n_ == 0) return false;
  // FEW DISTINCT COSTS IS THE SIGNAL. Every tie in the dual ratio test is a tie between
  // reduced costs, and on the slack basis those are the costs themselves; a model whose n
  // costs take fewer than n/4 values ties from the first pivot. Counted on the caller's
  // costs when they are known: scaling multiplies each cost by its own column factor, but
  // it multiplies that column's pivot-row entry by the same factor, so the breakpoints
  // d_j / |alpha_rj| - where the ties are - do not change, while the distinct count would.
  const bool original =
      original_costs_ != nullptr && original_costs_->size() == static_cast<std::size_t>(n_);
  std::vector<double> costs =
      original ? *original_costs_ : std::vector<double>(cost_.begin(), cost_.begin() + n_);
  std::sort(costs.begin(), costs.end());
  const auto distinct =
      static_cast<Index>(std::unique(costs.begin(), costs.end()) - costs.begin());
  if (static_cast<double>(distinct) >=
      tol::kDualStartPerturbationDistinctFraction * static_cast<double>(n_)) {
    logger_.verbose(
        "dual simplex: {} distinct costs over {} columns; no perturbation at the start",
        distinct, n_);
    return false;
  }
  if (!cost_shifted_) unperturbed_cost_ = cost_;
  Count moved = 0;
  for (Index k = 0; k < n_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (basis_position_[u] >= 0) continue;
    // A deterministic factor in [0.5, 1], from the same per-column hash the bound
    // perturbation uses (reproducible runs; see perturbation_for()).
    const double unit = (perturbation_for(k) / kPerturbationSize - 0.25) / 0.75;
    const double xi = (tol::kDualStartPerturbationAbsolute +
                       tol::kDualStartPerturbationRelative * std::fabs(cost_[u])) *
                      (0.5 + 0.5 * unit);
    if (status_[u] == BasisStatus::kAtLower) {
      cost_[u] += xi;
    } else if (status_[u] == BasisStatus::kAtUpper) {
      cost_[u] -= xi;
    } else {
      continue;  // fixed and free columns, as in perturb_costs()
    }
    ++moved;
  }
  cost_perturbed_ = true;
  ++cost_perturbations_;
  compute_reduced_costs(false);
  logger_.verbose(
      "dual simplex: {} distinct costs over {} columns; {} nonbasic cost(s) perturbed at "
      "the start",
      distinct, n_, moved);
  return true;
}

void Simplex::remove_cost_perturbation() {
  if (!cost_perturbed_ && !cost_shifted_) return;
  cost_ = unperturbed_cost_;
  cost_perturbed_ = false;
  cost_shifted_ = false;
  compute_reduced_costs(false);
}

}  // namespace sankhya::detail
