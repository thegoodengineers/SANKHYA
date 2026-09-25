// SPDX-License-Identifier: Apache-2.0
// SANKHYA - bounded-variable revised dual simplex (issue #65).
//
// References
//   Lemke, C.E. (1954), "The dual method of solving the linear programming problem", Naval
//     Research Logistics Quarterly 1, 36-47 - the method.
//   Maros, "Computational Techniques of the Simplex Method" (Kluwer, 2003), ch. 10 - the
//     bounded-variable dual, and the bound-flipping ratio test.
//   Koberstein, A. (2005), "The dual simplex method, techniques for a fast and stable
//     implementation", PhD thesis, Universitaet Paderborn - the artificial-bound treatment
//     of dual infeasibility (sec. 4.5) and the practical shape of the whole algorithm.
//   Forrest, J.J. and Goldfarb, D. (1992), "Steepest-edge simplex algorithms for linear
//     programming", Mathematical Programming 57, 341-374 - the dual devex weights, and the
//     exact dual steepest-edge update below (#411; Koberstein 2005, ch. 5, for the
//     practical form).
//
// DUAL STEEPEST EDGE (#411). The leaving row r is the basic variable furthest outside its
// bounds, measured against how far the duals must move to fix it: the dual step along
// rho_r = e_r^T B^-1 changes the duals by t rho_r, so the honest measure of row r's
// infeasibility is delta_r^2 / ||rho_r||^2. Devex approximates ||rho_r||^2 with a reference
// framework and resets it when it drifts; steepest edge maintains it exactly. After a pivot
// on (r, q) with alpha = B^-1 a_q the new inverse's rows are
//     rho_r'  =  rho_r / alpha_r,            rho_i'  =  rho_i - (alpha_i / alpha_r) rho_r,
// so, writing w_i = ||rho_i||^2 and tau = B^-1 rho_r (one extra FTRAN, the cross term
// rho_i . rho_r = e_i^T B^-1 rho_r = tau_i),
//     w_r'  =  w_r / alpha_r^2,
//     w_i'  =  w_i  -  2 (alpha_i / alpha_r) tau_i  +  (alpha_i / alpha_r)^2 w_r,
// with w_r taken as the norm the BTRAN of this very iteration produced, rho_r . rho_r, so a
// drift in the stored weight is corrected at the row that just left rather than carried.
// The weights start exact: 1 on the slack basis, and one BTRAN per row on a warm basis up to
// kDualSteepestEdgeExactInitRows rows (beyond that, 1, which is Devex's start too). A
// weight that comes back non-positive or non-finite sends every weight back to the exact
// start rather than pricing on a lie.
//
// WHY A SECOND METHOD. The primal simplex keeps a primal feasible point and improves the
// objective; the dual keeps every reduced cost sign-admissible and reduces the primal
// infeasibility. Branch and bound is where that difference is worth a second implementation:
// a child node is the parent with ONE bound moved, so the parent's optimal basis is still
// dual feasible at the child but no longer primal feasible - exactly the state the dual
// starts from, and typically a few pivots from the child's optimum. The primal has to
// re-solve from the slack basis, or run a phase 1 from a basis that is now infeasible.
// Measured on the MIPLIB subset before this landed: every node was a cold primal solve.
//
// THE SHAPE. Everything below runs over the state simplex_core.hpp declares: the same
// working problem in [A | -I] form, the same LU factors and product-form update, the same
// refactorization ladder and basis repair, the same reporting. What is specific to the
// dual is the choice of leaving ROW first (dual pricing, by primal infeasibility weighted
// with dual devex), the pivot row r of B^-1 [A | -I] from one BTRAN, the ratio test over
// the REDUCED COSTS with bound flipping, and what to do about a starting basis that is not
// dual feasible.
//
// DUAL INFEASIBILITY AT THE START. A nonbasic column at its lower bound with a negative
// reduced cost is dual infeasible; if it has an upper bound it is simply moved there, and if
// it has none an ARTIFICIAL upper bound is invented so that it can be (Koberstein sec. 4.5).
// The dual then solves that boxed problem. If at its optimum no column sits on an artificial
// bound, the boxes changed nothing and the answer stands; if one does, the boxes are removed
// and the PRIMAL loop finishes from that basis, which is primal feasible for the true bounds
// on every row but the ones the moved columns touch. No answer is ever reported about the
// boxed problem: the boxes are gone before finish() runs, on every exit.
//
// THE RATIO TEST'S PIVOT FLOOR IS RELATIVE TO THE ROW (#244), and the optimal exit sums
// what its wrong-signed reduced costs price before it claims anything (see the loop). The
// ratio tests - Harris's two passes with cost shifting (#465; dual_ratio_test=harris, the
// default) and the textbook one - and the cost perturbation against dual degeneracy live in
// dual_ratio.cpp: a stall perturbs the nonbasic costs and iterates on (or, behind
// dual_perturb_costs_at_start, the costs are perturbed before the first pivot), and only a
// stall that survives that hands the basis to the primal loop, which has its own
// anti-degeneracy machinery - so a stall costs iterations and never a wrong answer.

#include "core/safe_bound.hpp"
#include "primal_simplex.hpp"
#include "simplex_core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include <fmt/format.h>

#include "../core/stop_controller.hpp"
#include "../util/profiler.hpp"
#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::detail {

namespace {

/// Width of an artificial bound, in the units of the (scaled) working problem.
///
/// Large enough that a column the true optimum leaves interior is not pinned by it, small
/// enough that the boxed problem is not itself ill-conditioned: 1e6 is Koberstein's value.
/// Its exact size does not affect the answer - a box that binds at the dual's optimum is
/// removed and the primal loop finishes - only how often that hand-over happens.
constexpr double kArtificialBound = 1e6;

/// Restart the dual devex reference framework once any row weight passes this. Same role
/// and same order as kDevexResetThreshold for the primal weights.
constexpr double kDualWeightResetThreshold = 1e6;

/// Relative disagreement between the pivot element as computed along the pivot ROW (from
/// the BTRAN of e_r) and along the entering COLUMN (from the FTRAN of a_q) beyond which the
/// factors are not trusted for this pivot. The two are the same number computed through
/// different solves; on faithful factors they agree to rounding.
///
/// 1e-8, MEASURED. pilot87 hands over on "pivot -5.304e+01 along the column against
/// -5.304e+01 along the row on fresh factors": seven digits of agreement, on a basis whose
/// smallest factorization pivot is 1.6e-7. Loosening this to 1e-6 to let that pivot through
/// was tried: the dual then took it, and the bases it produced from there went singular past
/// the repair limit in 6 s, on both the scaled and the unscaled attempt - a numerical error
/// where the hand-over reaches a verified feasible point. The disagreement is a symptom of
/// the basis, not of the tolerance, and the primal loop is the right place to be on it.
constexpr double kPivotAgreement = 1e-8;

}  // namespace

// -----------------------------------------------------------------------------------------
// Dual feasibility and the artificial bounds
// -----------------------------------------------------------------------------------------

void Simplex::make_dual_feasible() {
  artificial_lower_.assign(static_cast<std::size_t>(total_), 0);
  artificial_upper_.assign(static_cast<std::size_t>(total_), 0);
  Count boxed = 0;
  Count flipped = 0;
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (basis_position_[u] >= 0) continue;
    const double d = reduced_cost_[u];
    switch (status_[u]) {
      case BasisStatus::kAtLower:
        if (d < -dual_tolerance_) {
          if (!is_finite_bound(upper_[u])) {
            upper_[u] = lower_[u] + kArtificialBound;
            artificial_upper_[u] = 1;
            ++boxed;
          }
          status_[u] = BasisStatus::kAtUpper;
          nonbasic_value_[u] = upper_[u];
          ++flipped;
        }
        break;
      case BasisStatus::kAtUpper:
        if (d > dual_tolerance_) {
          if (!is_finite_bound(lower_[u])) {
            lower_[u] = upper_[u] - kArtificialBound;
            artificial_lower_[u] = 1;
            ++boxed;
          }
          status_[u] = BasisStatus::kAtLower;
          nonbasic_value_[u] = lower_[u];
          ++flipped;
        }
        break;
      case BasisStatus::kNonbasicFree:
        // A free column's reduced cost must be zero. One that is not is boxed on both
        // sides and parked on the side its sign admits; it enters the basis at the first
        // pivot that can use it, and stays there as a rule.
        if (std::fabs(d) > dual_tolerance_) {
          lower_[u] = -kArtificialBound;
          upper_[u] = kArtificialBound;
          artificial_lower_[u] = 1;
          artificial_upper_[u] = 1;
          ++boxed;
          if (d > 0.0) {
            status_[u] = BasisStatus::kAtLower;
            nonbasic_value_[u] = lower_[u];
          } else {
            status_[u] = BasisStatus::kAtUpper;
            nonbasic_value_[u] = upper_[u];
          }
          ++flipped;
        }
        break;
      case BasisStatus::kFixed:
      case BasisStatus::kBasic:
      case BasisStatus::kUnknown: break;
    }
  }
  if (flipped > 0) {
    logger_.verbose(
        "dual simplex: {} column(s) moved to their other bound and {} given an "
        "artificial bound to make the starting basis dual feasible",
        flipped, boxed);
    compute_basic_values();
  }
}

bool Simplex::any_artificial_bound() const {
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (artificial_lower_[u] != 0 || artificial_upper_[u] != 0) return true;
  }
  return false;
}

bool Simplex::any_artificial_bound_active() const {
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (basis_position_[u] >= 0) continue;
    if (status_[u] == BasisStatus::kAtLower && artificial_lower_[u] != 0) return true;
    if (status_[u] == BasisStatus::kAtUpper && artificial_upper_[u] != 0) return true;
  }
  return false;
}

void Simplex::remove_artificial_bounds() {
  bool moved = false;
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (artificial_lower_[u] == 0 && artificial_upper_[u] == 0) continue;
    if (k < n_) {
      lower_[u] = model_.col_lower[u];
      upper_[u] = model_.col_upper[u];
    } else {
      const auto r = static_cast<std::size_t>(k - n_);
      lower_[u] = model_.row_lower[r];
      upper_[u] = model_.row_upper[r];
    }
    artificial_lower_[u] = 0;
    artificial_upper_[u] = 0;
    if (basis_position_[u] >= 0) continue;
    // Parked on a bound that no longer exists: the nearest real one, or free at zero.
    // make_nonbasic() is the same choice a basis repair makes.
    make_nonbasic(k, nonbasic_value_[u]);
    moved = true;
  }
  if (moved) compute_basic_values();
}

// -----------------------------------------------------------------------------------------
// Dual pricing
// -----------------------------------------------------------------------------------------

void Simplex::reset_dual_weights() {
  dual_weight_.assign(static_cast<std::size_t>(m_), 1.0);
  ++dual_weight_resets_;
  // Steepest edge starts from the truth where that is affordable (#411): the slack basis
  // has every row norm at 1 already, and a warm basis gets one BTRAN per row up to the cap.
  if (dual_steepest_edge_ && warm_started_ && m_ <= tol::kDualSteepestEdgeExactInitRows) {
    compute_exact_dual_weights();
  }
}

void Simplex::compute_exact_dual_weights() {
  dual_weight_ = exact_dual_weights_for_testing();
}

std::vector<double> Simplex::exact_dual_weights_for_testing() {
  std::vector<double> weights(static_cast<std::size_t>(m_), 1.0);
  std::vector<double> row(static_cast<std::size_t>(m_), 0.0);
  for (Index slot = 0; slot < m_; ++slot) {
    std::fill(row.begin(), row.end(), 0.0);
    row[static_cast<std::size_t>(slot)] = 1.0;
    lu_.solve_transpose(row.data());  // e_slot^T B^-1
    double norm2 = 0.0;
    for (const double v : row) norm2 += v * v;
    weights[static_cast<std::size_t>(slot)] = norm2;
  }
  return weights;
}

Index Simplex::choose_leaving_row() const {
  // The basic variable furthest outside its bounds, in the dual devex metric: infeasibility
  // squared over the row's reference weight. The weight approximates the norm of row r of
  // B^-1, which is what turns a primal infeasibility into a dual step length - the dual
  // analogue of why the primal prices on d^2 / w rather than on |d|.
  Index best = -1;
  double best_score = 0.0;
  for (Index slot = 0; slot < m_; ++slot) {
    const auto s = static_cast<std::size_t>(slot);
    const Index k = basis_[s];
    const double x = x_basic_[s];
    const double lo = lower_[static_cast<std::size_t>(k)];
    const double hi = upper_[static_cast<std::size_t>(k)];
    double infeasibility = 0.0;
    if (is_finite_bound(lo) && x < lo - primal_tolerance_) {
      infeasibility = lo - x;
    } else if (is_finite_bound(hi) && x > hi + primal_tolerance_) {
      infeasibility = x - hi;
    } else {
      continue;
    }
    const double score = infeasibility * infeasibility / dual_weight_[s];
    if (score > best_score) {
      best_score = score;
      best = slot;
    }
  }
  return best;
}

/// Below this fraction of the rows, rho is sparse enough that scattering its support
/// through the row-wise copy of A beats gathering every column (#243). Measured on the
/// four scale models: see the PR.
constexpr double kSparsePivotRowFraction = 0.1;

void Simplex::compute_pivot_row(Index leaving_slot) {
  Timer clock;
  std::fill(rho_.begin(), rho_.end(), 0.0);
  rho_[static_cast<std::size_t>(leaving_slot)] = 1.0;
  lu_.solve_transpose(rho_.data());
  pivot_row_btran_seconds_ += clock.elapsed_seconds();
  clock.reset();

  // rho's support, and what a row-wise pass over it would cost: one entry per nonzero of
  // each row it touches, plus the row's logical column.
  Index rho_nonzeros = 0;
  double row_work = 0.0;
  const std::vector<Index>& row_starts = by_row_.row_starts();
  for (Index i = 0; i < m_; ++i) {
    if (rho_[static_cast<std::size_t>(i)] == 0.0) continue;
    ++rho_nonzeros;
    row_work += 1.0 + static_cast<double>(row_starts[static_cast<std::size_t>(i) + 1] -
                                          row_starts[static_cast<std::size_t>(i)]);
  }
  rho_nonzeros_total_ += static_cast<double>(rho_nonzeros);
  ++pivot_rows_computed_;

  const bool sparse =
      static_cast<double>(rho_nonzeros) <= kSparsePivotRowFraction * static_cast<double>(m_);
  if (sparse) {
    // pivot_row_[k] = rho . a_k, from the rows of A that rho touches. Every entry the
    // previous pass wrote is zeroed first: the touched list when that pass was sparse, the
    // whole row when it was dense and left values everywhere.
    ++pivot_rows_sparse_;
    if (pivot_row_held_sparse_) {
      for (const Index k : pivot_row_touched_) {
        pivot_row_[static_cast<std::size_t>(k)] = 0.0;
        pivot_row_marked_[static_cast<std::size_t>(k)] = 0;
      }
    } else {
      std::fill(pivot_row_.begin(), pivot_row_.end(), 0.0);
    }
    pivot_row_touched_.clear();
    pivot_row_held_sparse_ = true;
    const auto touch = [&](Index k, double value) {
      const auto u = static_cast<std::size_t>(k);
      if (pivot_row_marked_[u] == 0) {
        pivot_row_marked_[u] = 1;
        pivot_row_touched_.push_back(k);
        pivot_row_[u] = value;
      } else {
        pivot_row_[u] += value;
      }
    };
    for (Index i = 0; i < m_; ++i) {
      const double rho_i = rho_[static_cast<std::size_t>(i)];
      if (rho_i == 0.0) continue;
      // The logical column of row i is -e_i, so its entry is -rho_i.
      if (basis_position_[static_cast<std::size_t>(n_ + i)] < 0) touch(n_ + i, -rho_i);
      const ColumnView row = by_row_.row(i);
      for (Index q = 0; q < row.size; ++q) {
        const Index j = row.rows[q];  // a column index: CsrView reuses the ColumnView layout
        if (basis_position_[static_cast<std::size_t>(j)] >= 0) continue;
        touch(j, rho_i * row.values[q]);
      }
    }
    pivot_row_gather_seconds_ += clock.elapsed_seconds();
    return;
  }

  // Dense rho from here on. Every entry of pivot_row_ is written by either pass below, so
  // the sparse pass's bookkeeping is cleared rather than trusted.
  for (const Index k : pivot_row_touched_) pivot_row_marked_[static_cast<std::size_t>(k)] = 0;
  pivot_row_touched_.clear();
  pivot_row_held_sparse_ = false;

  // ROW-WISE INTO A DENSE ROW, WHEN THAT READS LESS OF A. The gather below reads every
  // nonbasic column whole, O(nnz(A)) whatever rho looks like; the rows rho touches hold
  // row_work entries, and at a third of the rows nonzero - the medium tier's usual density,
  // above the sparse pass's 10% - that is well under nnz(A). The rows are scattered into a
  // zeroed dense row with no touched-list bookkeeping, and the basic columns zeroed after.
  // The numbers are the gather's to the bit: column j's entry is summed over ascending rows
  // in both (the columns are stored sorted by row, SparseMatrix::finalize), the terms are
  // the same products, and the gather's extra terms are the rho_i = 0 ones, which add a
  // zero and change nothing. So the pivot path does not move; only the time does.
  // Single-threaded only: the gather is the loop OpenMP splits (#57).
  if (pivot_row_single_thread_ &&
      row_work < static_cast<double>(model_.num_nonzeros()) + static_cast<double>(m_)) {
    ++pivot_rows_row_wise_;
    std::fill(pivot_row_.begin(), pivot_row_.end(), 0.0);
    for (Index i = 0; i < m_; ++i) {
      const double rho_i = rho_[static_cast<std::size_t>(i)];
      if (rho_i == 0.0) continue;
      // The logical column of row i is -e_i, so its entry is -rho_i.
      pivot_row_[static_cast<std::size_t>(n_ + i)] = -rho_i;
      const ColumnView row = by_row_.row(i);
      for (Index q = 0; q < row.size; ++q) {
        pivot_row_[static_cast<std::size_t>(row.rows[q])] += rho_i * row.values[q];
      }
    }
    for (Index slot = 0; slot < m_; ++slot) {
      pivot_row_[static_cast<std::size_t>(basis_[static_cast<std::size_t>(slot)])] = 0.0;
    }
    pivot_row_gather_seconds_ += clock.elapsed_seconds();
    return;
  }

  // Otherwise a gather per column, deterministic at any thread count (#57).
#ifdef SANKHYA_HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (basis_position_[u] >= 0) {
      pivot_row_[u] = 0.0;
      continue;
    }
    double dot = 0.0;
    for_each_entry(k, [&](Index row, double coefficient) {
      dot += rho_[static_cast<std::size_t>(row)] * coefficient;
    });
    pivot_row_[u] = dot;
  }
  pivot_row_gather_seconds_ += clock.elapsed_seconds();
}

void Simplex::update_dual_weights(Index leaving_slot, double pivot) {
  if (std::fabs(pivot) < tol::kZeroDrop) return;
  const auto r = static_cast<std::size_t>(leaving_slot);
  if (dual_steepest_edge_) {
    // EXACT STEEPEST EDGE (#411); the derivation is in the file comment. rho_ is still
    // e_r^T B^-1 from this iteration's BTRAN and alpha_ is B^-1 a_q from its FTRAN, both
    // off the same factors, and the basis has not changed yet. w_r is read off rho itself
    // so the row that leaves corrects any drift in its stored weight.
    double weight_r = 0.0;
    for (const double v : rho_) weight_r += v * v;
    tau_ = rho_;
    lu_.solve(tau_.data());  // tau = B^-1 rho_r: the cross terms rho_i . rho_r
    const double inverse_pivot = 1.0 / pivot;
    bool healthy = std::isfinite(weight_r) && weight_r > 0.0;
    for (Index slot = 0; slot < m_ && healthy; ++slot) {
      const auto s = static_cast<std::size_t>(slot);
      if (s == r) continue;
      const double a = alpha_[s];
      if (a == 0.0) continue;
      const double ratio = a * inverse_pivot;
      const double w = dual_weight_[s] - 2.0 * ratio * tau_[s] + ratio * ratio * weight_r;
      if (!std::isfinite(w)) {
        healthy = false;
        break;
      }
      dual_weight_[s] = std::max(w, tol::kDualSteepestEdgeWeightFloor);
    }
    if (healthy) {
      const double w = weight_r * inverse_pivot * inverse_pivot;
      healthy = std::isfinite(w);
      if (healthy) dual_weight_[r] = std::max(w, tol::kDualSteepestEdgeWeightFloor);
    }
    if (!healthy) {
      // A weight that is not a number is not a weight: back to Devex's start of 1 for
      // every row, counted as a reset, rather than pricing on it.
      dual_weight_.assign(static_cast<std::size_t>(m_), 1.0);
      ++dual_weight_resets_;
    }
    return;
  }
  const double weight_r = dual_weight_[r];
  const double inverse_pivot = 1.0 / pivot;
  double largest = 1.0;
  // w_i <- max(w_i, (alpha_iq / alpha_rq)^2 w_r) for every other row; the row the entering
  // column takes over gets w_r / alpha_rq^2, floored at 1. Exact steepest edge subtracts a
  // cross term that needs a second FTRAN; devex drops it and keeps the max, so the weights
  // only grow inside a reference framework and are reset once they drift too far.
  for (Index slot = 0; slot < m_; ++slot) {
    const auto s = static_cast<std::size_t>(slot);
    if (s == r) continue;
    const double a = alpha_[s];
    if (a == 0.0) continue;
    const double ratio = a * inverse_pivot;
    const double candidate = ratio * ratio * weight_r;
    if (candidate > dual_weight_[s]) dual_weight_[s] = candidate;
    if (dual_weight_[s] > largest) largest = dual_weight_[s];
  }
  dual_weight_[r] = std::max(weight_r * inverse_pivot * inverse_pivot, 1.0);
  if (dual_weight_[r] > largest) largest = dual_weight_[r];
  if (largest > kDualWeightResetThreshold) reset_dual_weights();
}

// -----------------------------------------------------------------------------------------
// The iteration loop
// -----------------------------------------------------------------------------------------

std::optional<Solution> Simplex::dual_loop(Timer& timer, Count* iterations_io) {
  Count& iterations = *iterations_io;
  pivot_row_.assign(static_cast<std::size_t>(total_), 0.0);
  pivot_row_marked_.assign(static_cast<std::size_t>(total_), 0);
  pivot_row_touched_.clear();
  pivot_row_held_sparse_ = false;
  by_row_.build(model_.matrix);  // once per solve; the pattern never changes (#243)
  pivot_row_single_thread_ = options_.get_int("threads") == 1;
#ifndef SANKHYA_HAVE_OPENMP
  pivot_row_single_thread_ = true;  // no thread pool: the gather is serial whatever was asked
#endif
  compute_reduced_costs(false);
  make_dual_feasible();
  reset_dual_weights();
  // #465: the Harris ratio test is the LP default since its A/B ("auto"; branch and bound
  // resolves "auto" to textbook for its own LPs, mip::with_node_lp_defaults); the start
  // perturbation is still off by default until its own.
  const std::string ratio_rule = options_.get_string("dual_ratio_test");
  dual_harris_ = ratio_rule == "harris" || ratio_rule == "auto";
  // Which rule each dual solve ran under, for whoever reads the counters: the test that
  // pins LP solves to Harris and branch-and-bound LPs to textbook reads exactly these.
  if (Profiler* profiler = logger_.profiler();
      profiler != nullptr && profiler->records(ProfileMode::kBasic)) {
    profiler->count(dual_harris_ ? "dual solves, harris ratio test"
                                 : "dual solves, textbook ratio test");
  }
  if (options_.get_bool("dual_perturb_costs_at_start")) perturb_costs_at_start();

  int degenerate_run = 0;
  const auto hand_over = [&](const std::string& why) -> std::optional<Solution> {
    logger_.info(
        "Dual simplex: {} at iteration {}; the primal simplex continues from this "
        "basis",
        why, iterations);
    remove_cost_perturbation();
    remove_artificial_bounds();
    return std::nullopt;
  };
  const auto singular = [&]() {
    // Or abandoned on the deadline (#208): factorization_failed() tells the two apart. The
    // two removals below do not need the factors; the reduced costs they would refresh stay
    // as last computed, which is what an abandoned factorization leaves.
    remove_cost_perturbation();
    remove_artificial_bounds();
    return factorization_failed(iterations, timer);
  };
  // Refactorize and recompute everything the loop reads. Used wherever a claim reached
  // through the eta file must be re-examined on fresh factors before it is acted on.
  const auto refresh = [&]() {
    if (!refactorize()) return false;
    ++refactorizations_;
    compute_basic_values();
    compute_reduced_costs(false);
    return true;
  };

  StopController stop(control_, timer, limits_);
  // Zero iterations allowed means none are performed (#289); the count below happens after
  // a pivot, which is the right place for every larger budget and the wrong one for this.
  if (const LimitReason why = limits_.exhausted(timer.elapsed_seconds(), iterations, 0);
      why != LimitReason::kNone) {
    // The start perturbation (#465) ran above; the duals this exit reports must be for the
    // model's own costs (review of #653).
    remove_cost_perturbation();
    compute_reduced_costs(false);
    return finish(status_for(why),
                  limits_.describe(why, timer.elapsed_seconds(), iterations, 0), iterations,
                  timer.elapsed_seconds());
  }

  for (;;) {
    iterations_seen_ = iterations;
    if (iterations % 20 == 0) {
      logger_.iteration(iterations, minimization_objective(), max_infeasibility(), -1.0,
                        timer.elapsed_seconds());
    }

    Timer phase_clock;
    const auto charge = [&](std::size_t phase) {
      dual_phase_seconds_[phase] += phase_clock.elapsed_seconds();
      phase_clock.reset();
    };
    const Index leaving_slot = choose_leaving_row();
    charge(0);
    if (leaving_slot < 0) {
      // PRIMAL FEASIBLE ON FRESH FACTORS OR NOT AT ALL - the same rule the primal loop
      // applies to its optimality claim, for the same reason: the basic values were
      // computed through whatever eta file is in play.
      if (m_ > 0 && lu_.eta_count() > 0) {
        if (!refresh()) return singular();
        logger_.verbose(
            "iteration {}: primal feasible through the eta file; re-checking on "
            "fresh factors",
            iterations);
        continue;
      }
      if (any_artificial_bound_active()) {
        return hand_over("optimal for the boxed problem with an artificial bound active");
      }
      // OPTIMAL FOR THE PERTURBED COSTS IS NOT OPTIMAL, exactly as the primal loop says of
      // its perturbed bounds. The point is primal feasible, and with the exact costs back
      // it is a phase-2 start for the primal loop: usually a handful of pivots.
      if (cost_perturbed_ || cost_shifted_) {
        return hand_over("optimal under cost perturbation; exact costs restored");
      }
      // OPTIMAL WITHIN TOLERANCE IS NOT OPTIMAL WHEN THE TOLERANCE IS WORTH MONEY (#244).
      // A nonbasic column whose reduced cost drifted a few 1e-6 onto the wrong side - below
      // the dual tolerance at the scale of its own terms, so every per-column check accepts
      // it - still prices its whole range: on pilot4 one such column at its lower bound with
      // d = -4.8e-6 and a range of 3,128 puts 1.5e-2 into the duality gap of a 2,581
      // objective, the verifier rejects the claim, and the published optimum is indeed
      // 3.5e-3 lower. The gap is an identity - primal minus dual is the sum over the
      // nonbasic columns of d_j times the distance from x_j to the bound d_j prices - so it
      // is summed here against the verifier's own relative tolerance (kDualityGap), and
      // when it exceeds it the basis goes to the primal loop, which prices exactly those
      // columns as improving and finishes in a handful of pivots.
      {
        double gap = 0.0;
        for (Index k = 0; k < total_; ++k) {
          const auto u = static_cast<std::size_t>(k);
          if (basis_position_[u] >= 0 || status_[u] == BasisStatus::kFixed) continue;
          const double d = reduced_cost_[u];
          // A reduced cost inside the dual tolerance is one the verifier excuses in full
          // (its share of the gap is "accounted"); only a larger one prices its range.
          if (std::fabs(d) <= dual_tolerance_) continue;
          double distance = 0.0;
          if (status_[u] == BasisStatus::kAtLower && d < 0.0) {
            distance =
                is_finite_bound(upper_[u]) ? upper_[u] - lower_[u] : std::fabs(lower_[u]);
          } else if (status_[u] == BasisStatus::kAtUpper && d > 0.0) {
            distance =
                is_finite_bound(lower_[u]) ? upper_[u] - lower_[u] : std::fabs(upper_[u]);
          } else if (status_[u] == BasisStatus::kNonbasicFree && d != 0.0) {
            distance = 1.0;
          }
          gap += std::fabs(d) * distance;
        }
        const double objective = minimization_objective();
        if (gap > tol::kDualityGap * std::max(1.0, std::fabs(objective))) {
          return hand_over(fmt::format(
              "primal feasible, but the wrong-signed reduced costs price {:.3e} of objective "
              "against a {:.1e} relative duality tolerance",
              gap, tol::kDualityGap));
        }
      }
      remove_artificial_bounds();
      return finish(SolveStatus::kOptimal, {}, iterations, timer.elapsed_seconds());
    }

    const Index leaving = basis_[static_cast<std::size_t>(leaving_slot)];
    const auto l = static_cast<std::size_t>(leaving);
    const bool to_upper = is_finite_bound(upper_[l]) &&
                          x_basic_[static_cast<std::size_t>(leaving_slot)] > upper_[l];

    compute_pivot_row(leaving_slot);
    charge(1);
    const DualRatioResult ratio = dual_ratio_test(leaving_slot, to_upper);
    charge(2);

    if (ratio.dual_unbounded) {
      // INFEASIBILITY IS CLAIMED ON FRESH FACTORS, and only about the caller's model. With
      // an artificial bound anywhere the problem being solved is not the caller's, and a
      // box can be what makes it infeasible; the primal loop settles it instead.
      if (lu_.eta_count() > 0) {
        if (!refresh()) return singular();
        continue;
      }
      if (any_artificial_bound()) {
        return hand_over("no dual ratio-test candidate while artificial bounds are in play");
      }
      remove_cost_perturbation();
      // THE SAME CAUTION THE PRIMAL APPLIES TO ITS PHASE-1 STALL. "No candidate" is a Farkas
      // certificate in exact arithmetic; in floating point it ignores every column whose
      // pivot-row entry is below kPivotTolerance, and on a violation within a few orders of
      // the feasibility tolerance one of those could still close it. A marginal violation
      // is therefore handed to the primal loop, whose phase 1 decides between a numerical
      // stall and a proof on its own terms; only a violation far above tolerance is claimed
      // here, and the message says so.
      const double violation =
          to_upper ? x_basic_[static_cast<std::size_t>(leaving_slot)] - upper_[l]
                   : lower_[l] - x_basic_[static_cast<std::size_t>(leaving_slot)];
      if (violation <= kInfeasibilityProofFactor * primal_tolerance_) {
        return hand_over(fmt::format(
            "no dual ratio-test candidate on a marginal violation of {:.3e}", violation));
      }
      // THE PROOF, KEPT (#191). The comment above already names it: rho_ = B^-T e_r is a
      // Farkas vector for the rows, computed by compute_pivot_row(leaving_slot) a few lines
      // up, on fresh factors (the eta refresh above forces a `continue`, so the claiming
      // path never runs on an updated basis). Its SIGN depends on which bound the leaving
      // variable crossed, so solve() tries the vector both ways and keeps whichever proves
      // the original model infeasible - the check is cheap and settles the convention
      // without a derivation that could be silently wrong.
      pending_farkas_ = rho_;
      return finish(SolveStatus::kInfeasible,
                    fmt::format("dual simplex: basic variable {} is outside its bounds by "
                                "{:.3e}, far above the {:.1e} feasibility tolerance, and no "
                                "nonbasic column can move it (the dual is unbounded), at "
                                "iteration {}",
                                leaving, violation, primal_tolerance_, iterations),
                    iterations, timer.elapsed_seconds());
    }

    // UPDATE, DO NOT RECOMPUTE (#210). B x_B = -N x_N, so a set of flips moves the basic
    // values by B^-1 of the flipped columns times their changes: one FTRAN of a vector with
    // a few columns' worth of entries, against a pass over every nonzero of the model plus
    // the same FTRAN. The recomputation returns at the next refactorization, below.
    if (!ratio.flips.empty()) flip_rhs_.assign(static_cast<std::size_t>(m_), 0.0);
    for (const Index k : ratio.flips) {
      const auto u = static_cast<std::size_t>(k);
      const double before = nonbasic_value_[u];
      if (status_[u] == BasisStatus::kAtLower) {
        status_[u] = BasisStatus::kAtUpper;
        nonbasic_value_[u] = upper_[u];
      } else {
        status_[u] = BasisStatus::kAtLower;
        nonbasic_value_[u] = lower_[u];
      }
      const double change = nonbasic_value_[u] - before;
      for_each_entry(k, [&](Index row, double coefficient) {
        flip_rhs_[static_cast<std::size_t>(row)] -= coefficient * change;
      });
      ++bound_flips_;
    }
    if (!ratio.flips.empty()) {
      lu_.solve(flip_rhs_.data());
      for (Index i = 0; i < m_; ++i) {
        x_basic_[static_cast<std::size_t>(i)] += flip_rhs_[static_cast<std::size_t>(i)];
      }
      charge(6);
    }

    const double x_now = x_basic_[static_cast<std::size_t>(leaving_slot)];
    const double target = to_upper ? upper_[l] : lower_[l];
    const double residual = to_upper ? x_now - target : target - x_now;
    // A LIMIT EXIT STILL CARRIES A BOUND. The basis is dual feasible at every iteration of
    // this loop, so the objective of its (primal infeasible) basic solution is the dual
    // objective: a valid bound on the LP optimum - as long as no artificial bound is in
    // play, since those bound a different problem. Strong branching (#69) caps its probes
    // at a few dozen iterations and reads exactly this; finish() itself only ever states a
    // bound for an optimal exit.
    const auto stop_at_limit = [&](SolveStatus status, const std::string& why) {
      // Exact costs first: the bound below is the objective of the basic solution, and a
      // perturbed cost vector would put a perturbation-sized error into a number that
      // branch and bound prunes against.
      // A basis that was dual feasible for PERTURBED or SHIFTED costs is not, in general, dual
      // feasible for the model's own: restoring them can leave nonbasic reduced costs of the
      // wrong sign (by up to the perturbation, ~1e-5 with dual_perturb_costs_at_start), and
      // then the objective of the basic solution is no bound at all - a wrong-signed column
      // with no bound on its far side makes the true bound -inf. Such an exit reports the
      // Neumaier-Shcherbina bound from its duals instead (#519, src/core/safe_bound.hpp),
      // which charges every wrong-signed reduced cost against the column's range and says
      // -inf when it cannot (review of #653).
      const bool costs_moved = cost_perturbed_ || cost_shifted_;
      remove_cost_perturbation();
      const bool bound_is_valid = !any_artificial_bound();
      const double bound = minimization_objective();
      remove_artificial_bounds();
      compute_reduced_costs(false);
      Solution stopped = finish(status, why, iterations, timer.elapsed_seconds());
      if (bound_is_valid && !costs_moved) {
        stopped.dual_bound = bound;
      } else if (bound_is_valid &&
                 stopped.row_dual.size() == static_cast<std::size_t>(model_.num_rows())) {
        const SafeBound safe = safe_dual_bound(model_, stopped.row_dual);
        if (std::isfinite(safe.value)) {
          stopped.dual_bound = model_.sense_multiplier() * safe.value + model_.objective_offset;
        }
      }
      return stopped;
    };
    const auto count_iteration = [&]() -> std::optional<Solution> {
      ++iterations;
      ++dual_iterations_;
      if (limits_.iterations_exhausted(iterations)) {
        return stop_at_limit(
            SolveStatus::kIterationLimit,
            limits_.describe(LimitReason::kIterations, timer.elapsed_seconds(), iterations, 0));
      }

      SolveStatus stop_status;
      if (stop.should_stop(
              [&]() {
                Progress p;
                p.phase = Progress::Phase::kLp;
                p.iterations = iterations;
                p.objective = kInfinity;  // not valid during dual loop
                p.best_bound = minimization_objective();
                return p;
              },
              &stop_status)) {
        return stop_at_limit(
            stop_status,
            stop_status == SolveStatus::kTimeLimit
                ? limits_.describe(LimitReason::kTime, timer.elapsed_seconds(), iterations, 0)
                : limits_.describe(LimitReason::kInterrupt, timer.elapsed_seconds(), iterations,
                                   0));
      }
      return std::nullopt;
    };
    if (residual <= primal_tolerance_ || ratio.entering < 0) {
      // The flips alone brought the row inside its bounds; no basis change this iteration.
      if (std::optional<Solution> stopped = count_iteration()) return stopped;
      continue;
    }

    const Index entering = ratio.entering;
    const auto e = static_cast<std::size_t>(entering);
    ftran_entering_column(entering);
    charge(3);
    const double pivot = alpha_[static_cast<std::size_t>(leaving_slot)];
    const double pivot_by_row = pivot_row_[e];

    // THE PIVOT IS COMPUTED TWICE AND MUST AGREE. alpha_rq along the row came through the
    // BTRAN of e_r, along the column through the FTRAN of a_q; they are the same entry of
    // B^-1 a_q. A disagreement is the factors talking, not the model: refactorize and take
    // the iteration again. If fresh factors still disagree, or the pivot is negligible, the
    // basis is beyond what this loop can do with it and the primal loop takes over.
    const bool agree = std::fabs(pivot - pivot_by_row) <=
                       kPivotAgreement * std::max(1.0, std::fabs(pivot_by_row));
    if (!agree || std::fabs(pivot) <= tol::kPivotTolerance) {
      if (lu_.eta_count() > 0) {
        if (!refresh()) return singular();
        logger_.verbose(
            "iteration {}: pivot row/column disagreement {:.3e} vs {:.3e} through "
            "the eta file; refactorized",
            iterations, pivot_by_row, pivot);
        continue;
      }
      return hand_over(
          fmt::format("pivot {:.3e} along the column against {:.3e} along the "
                      "row on fresh factors",
                      pivot, pivot_by_row));
    }

    // HARRIS'S WRONG-SIGNED ENTERING COLUMN (#465). The Harris test may pick a column whose
    // reduced cost sits up to kDualHarrisRelaxation on the wrong side of zero; stepping on
    // it would move the duals backwards. Its cost is shifted so the reduced cost is exactly
    // zero instead, and the step is degenerate (Koberstein 2005, ch. 6). The shift is
    // undone with every other cost change before any answer leaves this loop.
    if (dual_harris_) {
      const double d = reduced_cost_[e];
      const double a = (to_upper ? 1.0 : -1.0) * pivot_by_row;
      if (a > 0.0 ? d < 0.0 : d > 0.0) shift_cost(entering, -d);
    }

    // Dual step length: the duals move by this much along rho below, and the stall counter
    // reads it here.
    const double dual_step = reduced_cost_[e] / pivot_by_row;
    if (std::fabs(dual_step) <= tol::kRatioTestFeasibility) {
      ++degenerate_run;
      // PERTURB BEFORE HANDING OVER. The primal loop is the last resort, and on dfl001 it
      // cost 15,000 phase-1 iterations; breaking the ties costs one pass over the costs.
      // The pivot in hand is discarded and the iteration taken again from the perturbed
      // reduced costs, which the ratio test now sees as distinct.
      if (!cost_perturbed_ && degenerate_run > kPerturbationTrigger) {
        logger_.verbose(
            "{} consecutive dual-degenerate iterations at iteration {}: perturbing costs",
            degenerate_run, iterations);
        perturb_costs();
        degenerate_run = 0;
        continue;
      }
      if (degenerate_run > kStallLimit) {
        return hand_over(
            fmt::format("{} consecutive dual-degenerate iterations", degenerate_run));
      }
    } else {
      degenerate_run = 0;
    }

    // Before the basis changes: the weight update reads alpha_ (already computed) and the
    // leaving slot's weight.
    update_dual_weights(leaving_slot, pivot);

    // The leaving variable lands exactly on the bound it violated; the entering variable's
    // value follows from the row: x_r - delta * alpha_rq = bound, so delta = (x_r - bound) /
    // alpha_rq is the change in x_q.
    basis_position_[l] = -1;
    nonbasic_value_[l] = target;
    status_[l] = to_upper ? BasisStatus::kAtUpper : BasisStatus::kAtLower;
    if (lower_[l] == upper_[l]) status_[l] = BasisStatus::kFixed;

    const double delta = (x_now - target) / pivot;
    basis_[static_cast<std::size_t>(leaving_slot)] = entering;
    basis_position_[e] = leaving_slot;
    status_[e] = BasisStatus::kBasic;
    nonbasic_value_[e] += delta;

    // UPDATE, DO NOT RECOMPUTE (#210). Every iteration used to end with a fresh FTRAN and a
    // pass over the nonzeros for x_B, and a fresh BTRAN and a pass over every column for d
    // - two of the four solves and three of the three matrix passes an iteration paid for.
    // Measured on the 20,000-row random model of the scale tables, with the
    // factorization already fixed, 34% of the time. The
    // textbook update needs the vectors already computed: x_B moves by -delta along alpha
    // (the entering column through the basis) with the entering variable taking the
    // vacated slot, and the duals move by the dual step along rho (the leaving row through
    // the basis), which moves every reduced cost by that step times its pivot-row entry.
    // The leaving column's pivot-row entry is 1 - it is the unit column of its own slot -
    // and pivot_row_ holds 0 there because it was basic, so its reduced cost is set by hand;
    // the entering column's is zero by construction. The recomputation still happens
    // wherever the factors are fresh: at every refactorization, and at every refresh().
    for (Index i = 0; i < m_; ++i) {
      x_basic_[static_cast<std::size_t>(i)] -= delta * alpha_[static_cast<std::size_t>(i)];
    }
    x_basic_[static_cast<std::size_t>(leaving_slot)] = nonbasic_value_[e];
    charge(6);
    for (Index k = 0; k < total_; ++k) {
      reduced_cost_[static_cast<std::size_t>(k)] -=
          dual_step * pivot_row_[static_cast<std::size_t>(k)];
    }
    reduced_cost_[e] = 0.0;
    reduced_cost_[l] = -dual_step;
    for (Index i = 0; i < m_; ++i) {
      y_[static_cast<std::size_t>(i)] += dual_step * rho_[static_cast<std::size_t>(i)];
    }
    charge(7);

    // The same update-or-refactorize policy as the primal loop, for the same reasons; see
    // the primal loop for the evidence behind each trigger.
    const bool trust_update = !basis_needed_stricter_threshold_;
    const bool updated = trust_update && lu_.update(leaving_slot, alpha_.data());
    charge(4);
    if (trust_update && !updated) ++rejected_updates_;
    eta_work_since_refactor_ += static_cast<double>(lu_.eta_nonzeros());
    const bool past_break_even =
        eta_work_since_refactor_ >
        refactor_work_ratio_ * std::max(1.0, static_cast<double>(lu_.factor_nonzeros()));
    if (!updated || past_break_even || lu_.should_refactorize()) {
      if (!refactorize()) return singular();
      ++refactorizations_;
      charge(5);
      // Fresh factors: whatever the updates above accumulated is replaced by the truth.
      compute_basic_values();
      charge(6);
      compute_reduced_costs(false);
      charge(7);
    }

    if (std::optional<Solution> stopped = count_iteration()) return stopped;
  }
}

Solution Simplex::run_dual(const WarmStart* warm) {
  Timer timer;
  limits_ = ResourceLimits(options_, logger_);
  time_limit_ = options_.get_double("time_limit");
  arm_deadline(timer);
  algorithm_name_ = "simplex-dual";
  if (std::optional<Solution> early = prepare(warm, timer)) return *early;

  logger_.info("Dual simplex: {} rows, {} columns, {} nonzeros{}", m_, n_,
               model_.num_nonzeros(), warm_started_ ? ", warm start" : "");
  logger_.begin_iteration_table();

  Count iterations = 0;
  std::optional<Solution> done = dual_loop(timer, &iterations);
  if (bound_flips_ > 0 || dual_iterations_ > 0) {
    logger_.verbose(
        "dual simplex: {} iterations, {} bound flips, {} weight resets, {} cost "
        "perturbation(s), {} Harris cost shift(s)",
        dual_iterations_, bound_flips_, dual_weight_resets_, cost_perturbations_, cost_shifts_);
  }
  if (done) return *done;
  algorithm_name_ = "simplex-dual+primal";
  return primal_loop(timer, &iterations);
}

}  // namespace sankhya::detail
