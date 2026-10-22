// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the primal push of a crossover (#343): from an interior point on the optimal
// face to a vertex of it, one superbasic variable at a time.
//
// THE PROBLEM THE PUSH SOLVES. A basis guessed off an interior point (crossover.cpp) sends
// every nonbasic variable to a bound at once, and a point that had thousands of variables
// strictly inside their bounds lands far from feasibility; the dual simplex then rebuilds
// feasibility with as many pivots as a cold start (41,872 against 34,071 on the 5,000-row
// staircase model). The push keeps the point FEASIBLE throughout instead: every nonbasic
// variable starts at the interior point's value - "superbasic", nonbasic but not at a bound
// - and is moved to its nearer bound along a direction that keeps A x = b, with a ratio test
// against the basic variables. When a basic variable would leave its bounds first, it leaves
// the basis and the pushed variable enters: one pivot. When nothing blocks, the variable
// simply arrives at its bound and the basis is unchanged. After the last push every nonbasic
// variable is at a bound, the point is a primal feasible vertex on (or near) the optimal face,
// and the ordinary primal loop finishes from there in the few pivots the residual
// degeneracy needs.
//
// Bixby, "Solving real-world linear programs: a decade and more of progress", Operations
// Research 50 (2002), sec. 4; Andersen & Ye, "Combining interior-point and pivoting
// algorithms for constrained linear programs", Management Science 42 (1996); Megiddo, "On
// finding primal- and dual-optimal bases", ORSA J. Computing 3 (1991) for the argument that
// the push terminates at a vertex of the optimal face.

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "../util/profiler.hpp"
#include "primal_simplex.hpp"
#include "sankhya/tolerances.hpp"
#include "simplex_core.hpp"

namespace sankhya::detail {

Solution Simplex::run_push(const WarmStart& warm, const std::vector<double>& interior_x,
                           const std::vector<double>& interior_activity) {
  Timer timer;
  // The limits BEFORE the deadline is armed, as run() and run_dual() do: arm_deadline() reads
  // limits_, and prepare() sets it only after. Armed first, the push saw a default
  // ResourceLimits with no time limit and ran with no deadline at all - on rmine15 after the
  // cuDSS interior point (#417) it took 244 s of the 143 s left, 14,353 pivots and 143
  // refactorizations of a 358,395-row basis, and the solve ended at 400 s of 300.
  limits_ = ResourceLimits(options_, logger_);
  time_limit_ = options_.get_double("time_limit");
  arm_deadline(timer);
  if (std::optional<Solution> early = prepare(&warm, timer)) return *early;
  logger_.info("Crossover push: {} rows, {} columns, {} nonzeros{}", m_, n_,
               model_.num_nonzeros(),
               warm_started_ ? ", basis guess installed" : ", slack basis");

  // Every nonbasic variable that the interior point held strictly inside its bounds starts
  // there. The working problem's logical for row i carries the row's activity, so the same
  // rule applies to rows. A fixed variable has nowhere to go.
  std::vector<Index> superbasic;
  const bool have_point = static_cast<Index>(interior_x.size()) == n_ &&
                          static_cast<Index>(interior_activity.size()) == m_;
  if (have_point) {
    for (Index k = 0; k < total_; ++k) {
      const auto u = static_cast<std::size_t>(k);
      if (basis_position_[u] >= 0 || status_[u] == BasisStatus::kFixed) continue;
      const double v =
          k < n_ ? interior_x[u] : interior_activity[static_cast<std::size_t>(k - n_)];
      if (!std::isfinite(v)) continue;
      const double scale = std::max(1.0, std::fabs(v));
      const bool inside_lower =
          !is_finite_bound(lower_[u]) || v - lower_[u] > primal_tolerance_ * scale;
      const bool inside_upper =
          !is_finite_bound(upper_[u]) || upper_[u] - v > primal_tolerance_ * scale;
      if (!inside_lower || !inside_upper) {
        // On a bound: the one the POINT is on, not the one the warm start named or
        // set_initial_basis() picked nearest to zero (#888). A slack-basis start sent every
        // column the point held at its upper bound to its lower one, and x_B then missed
        // the point by 1.7e+02 on the refinery at T = 365.
        if (!inside_lower) {
          status_[u] = BasisStatus::kAtLower;
          nonbasic_value_[u] = lower_[u];
        } else {
          status_[u] = BasisStatus::kAtUpper;
          nonbasic_value_[u] = upper_[u];
        }
        continue;
      }
      nonbasic_value_[u] = v;
      superbasic.push_back(k);
    }
  }
  compute_basic_values();
  // THE PUSH RESTS ON THE POINT BEING FEASIBLE. x_B is recomputed from the superbasic
  // values through the guessed basis, and on an ill-conditioned model (wood1p) that basis
  // can amplify the interior point's residual into a point that is not feasible at all;
  // pushing from there, and then asking the primal loop to repair 1e-2 of infeasibility on
  // a massively degenerate model, ran 683,000 pivots to a time limit. A point that is not
  // feasible to a loose multiple of the tolerance is handed back as "not solved", and the
  // caller keeps the interior point's answer.
  const double loose = 1e3 * primal_tolerance_;
  // FACTORS THAT FACTORIZED ARE NOT YET FACTORS THAT SOLVE (#1023). Threshold pivoting
  // bounds the growth of U only by (1 + 1/tau) per step, so a basis can factorize at
  // tau = 0.01 with every pivot above the tolerance and still solve to nothing: on
  // maros-r7 the guess above (3,136 interior entries for 3,136 rows, none evicted, smallest
  // pivot 9.5e-3) gave x_B 2.9e3 from a point snapped to its bounds by 1e-7, and the decline
  // below kept the interior point's answer with no basis at all. The same basis at
  // tau = 0.1 reproduced the point to 6e-7. Duff, Erisman & Reid, "Direct Methods for
  // Sparse Matrices" (2nd ed., 2017), sec. 5.4: check the result, raise tau when it fails.
  // The result checked is the one the push needs, x_B against the point, and the stricter
  // rung stays for the rest of this crossover, whose pivots refactorize the same kind of
  // basis. The simplex loops elsewhere are untouched: a check on every fresh factorization
  // fired on 1,389 of pilot87's 1,835 and the dual simplex took 158.6 s against 22.4 s on
  // the same box, where the loops' own accuracy check had been coping all along.
  while (max_infeasibility() > loose) {
    const double before = max_infeasibility();
    if (!refactorize_stricter()) break;
    compute_basic_values();
    logger_.info(
        "Crossover push: the basis guess reproduced the interior point to {:.3e}; "
        "refactorized at a stricter Markowitz threshold, now {:.3e}",
        before, max_infeasibility());
  }
  if (max_infeasibility() > loose) {
    return finish(SolveStatus::kNotSolved,
                  fmt::format("crossover push declined: the basis guess reproduces the "
                              "interior point to an infeasibility of {:.3e}",
                              max_infeasibility()),
                  0, timer.elapsed_seconds());
  }

  Count pivots = 0;
  Count arrived = 0;
  Count refactorizations = 0;
  bool deadline_hit = false;
  for (const Index k : superbasic) {
    if (deadline_ && (pivots & 63) == 0 && deadline_()) {
      deadline_hit = true;
      break;
    }
    const auto u = static_cast<std::size_t>(k);
    if (basis_position_[u] >= 0) continue;  // entered the basis while another was pushed
    const double value = nonbasic_value_[u];
    // The bound to push towards: the nearer finite one; a free variable goes to zero, the
    // value the simplex holds a nonbasic free variable at.
    double target;
    BasisStatus target_status;
    const bool has_lower = is_finite_bound(lower_[u]);
    const bool has_upper = is_finite_bound(upper_[u]);
    if (has_lower && has_upper) {
      const bool nearer_upper = upper_[u] - value < value - lower_[u];
      target = nearer_upper ? upper_[u] : lower_[u];
      target_status = nearer_upper ? BasisStatus::kAtUpper : BasisStatus::kAtLower;
    } else if (has_lower) {
      target = lower_[u];
      target_status = BasisStatus::kAtLower;
    } else if (has_upper) {
      target = upper_[u];
      target_status = BasisStatus::kAtUpper;
    } else {
      target = 0.0;
      target_status = BasisStatus::kNonbasicFree;
    }
    const int direction = target >= value ? 1 : -1;
    const double distance = std::fabs(target - value);
    if (distance == 0.0) {
      status_[u] = target_status;
      ++arrived;
      continue;
    }

    // alpha = B^-1 a_k: how the basic variables move per unit of the push.
    {
      const ProfileScope timed(logger_.profiler(), "push ftran", ProfileMode::kDetailed);
      ftran_entering_column(k);
    }
    const ProfileScope ratio_timed(logger_.profiler(), "push ratio test and update",
                                   ProfileMode::kDetailed);
    double step = distance;
    Index leaving_slot = -1;
    bool leaving_to_upper = false;
    for (Index slot = 0; slot < m_; ++slot) {
      const double a = alpha_[static_cast<std::size_t>(slot)];
      if (std::fabs(a) <= tol::kPivotTolerance) continue;
      // x_B changes by -direction * a per unit step.
      const double rate = -static_cast<double>(direction) * a;
      const auto i = static_cast<std::size_t>(basis_[static_cast<std::size_t>(slot)]);
      const double xb = x_basic_[static_cast<std::size_t>(slot)];
      double room;
      if (rate < 0.0) {
        if (!is_finite_bound(lower_[i])) continue;
        room = xb - lower_[i];
      } else {
        if (!is_finite_bound(upper_[i])) continue;
        room = upper_[i] - xb;
      }
      const double t = std::max(0.0, room) / std::fabs(rate);
      if (t < step) {
        step = t;
        leaving_slot = slot;
        leaving_to_upper = rate > 0.0;
      }
    }

    for (Index slot = 0; slot < m_; ++slot) {
      x_basic_[static_cast<std::size_t>(slot)] -=
          static_cast<double>(direction) * step * alpha_[static_cast<std::size_t>(slot)];
    }
    if (leaving_slot < 0) {
      nonbasic_value_[u] = target;
      status_[u] = target_status;
      ++arrived;
      continue;
    }

    // A basic variable blocks: it leaves at the bound it reached, the pushed variable
    // enters at the value the step carried it to. The same swap the primal loop makes.
    const auto slot = static_cast<std::size_t>(leaving_slot);
    const Index leaving = basis_[slot];
    const auto l = static_cast<std::size_t>(leaving);
    basis_position_[l] = -1;
    if (leaving_to_upper) {
      nonbasic_value_[l] = upper_[l];
      status_[l] = BasisStatus::kAtUpper;
    } else {
      nonbasic_value_[l] = lower_[l];
      status_[l] = BasisStatus::kAtLower;
    }
    if (lower_[l] == upper_[l]) status_[l] = BasisStatus::kFixed;
    basis_[slot] = k;
    basis_position_[u] = leaving_slot;
    status_[u] = BasisStatus::kBasic;
    x_basic_[slot] = value + static_cast<double>(direction) * step;
    nonbasic_value_[u] = x_basic_[slot];
    ++pivots;
    const bool updated = [&] {
      const ProfileScope timed(logger_.profiler(), "push basis update", ProfileMode::kDetailed);
      return lu_.update(leaving_slot, alpha_.data());
    }();
    // The same refactorization rule as the primal and dual loops (primal_simplex.cpp): in
    // operation counts, when the eta file's solves have cost refactor_work_ratio_ times the
    // base factors, or when the LU itself asks. A flat 100 pivots refactorized the T = 365
    // refinery's near-slack basis 409 times in the push, 25.3 of its 44 s (profile=detailed,
    // #888); the break-even rule is what the loops were calibrated on and is deterministic
    // for the same reason they are.
    eta_work_since_refactor_ += static_cast<double>(lu_.eta_nonzeros());
    const bool past_break_even =
        eta_work_since_refactor_ >
        refactor_work_ratio_ * std::max(1.0, static_cast<double>(lu_.factor_nonzeros()));
    if (!updated || past_break_even || lu_.should_refactorize()) {
      const ProfileScope timed(logger_.profiler(), "push refactorization",
                               ProfileMode::kDetailed);
      if (!refactorize()) {
        if (factors_abandoned_) return factorization_failed(pivots, timer);
        return finish(SolveStatus::kNumericalError,
                      "the basis went singular during the crossover push", pivots,
                      timer.elapsed_seconds());
      }
      ++refactorizations;
      eta_work_since_refactor_ = 0.0;
      compute_basic_values();
    }
  }
  if (!refactorize()) {
    if (factors_abandoned_) return factorization_failed(pivots, timer);
    return finish(SolveStatus::kNumericalError,
                  "the basis went singular after the crossover push", pivots,
                  timer.elapsed_seconds());
  }
  compute_basic_values();
  reset_devex();
  logger_.info(
      "Crossover push: {} superbasic variable(s), {} arrived at a bound without a pivot, {} "
      "basis changes, {} refactorizations, max infeasibility {:.3e}{}",
      superbasic.size(), arrived, pivots, refactorizations, max_infeasibility(),
      deadline_hit ? " (time limit reached inside the push)" : "");
  if (max_infeasibility() > loose) {
    return finish(SolveStatus::kNotSolved,
                  fmt::format("crossover push declined: the pushed point is {:.3e} infeasible, "
                              "beyond what the primal loop should be asked to repair",
                              max_infeasibility()),
                  pivots, timer.elapsed_seconds());
  }

  // The vertex is primal feasible (up to the push's arithmetic); the primal loop prices the
  // reduced costs and finishes. Its iteration count continues from the push's pivots.
  Count iterations = pivots;
  Solution solution = primal_loop(timer, &iterations);
  const std::string note = fmt::format("crossover push: {} superbasic, {} pivots in the push",
                                       superbasic.size(), pivots);
  solution.message = solution.message.empty() ? note : solution.message + "; " + note;
  return solution;
}

}  // namespace sankhya::detail
