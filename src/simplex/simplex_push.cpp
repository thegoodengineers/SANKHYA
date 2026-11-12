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

double Simplex::push_residual() const {
  // max_i |(A x - s)_i| over the working problem [A | -I][x ; s] = 0, every variable at the
  // value the simplex holds for it (#888 diagnostics).
  std::vector<double> rows(static_cast<std::size_t>(m_), 0.0);
  for (Index k = 0; k < total_; ++k) {
    const double v = variable_value(k);
    if (v == 0.0) continue;
    for_each_entry(k, [&](Index row, double coefficient) {
      rows[static_cast<std::size_t>(row)] += coefficient * v;
    });
  }
  double worst = 0.0;
  for (const double r : rows) worst = std::max(worst, std::fabs(r));
  return worst;
}

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
  const double asked_ceiling = options_.get_double("crossover_push_ceiling");
  const double loose = asked_ceiling > 0.0
                           ? asked_ceiling
                           : tol::kCrossoverPushCeilingFactor * primal_tolerance_;
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
  // Per refactorization window (#888 diagnostics): the smallest eta pivot relative to its
  // column, and the largest step times that column's largest entry.
  double window_min_relative_pivot = 1.0;
  double window_max_move = 0.0;
  const double refactor_below = options_.get_double("crossover_push_refactor_below");
  const auto refresh_every =
      static_cast<Count>(options_.get_int("crossover_push_refresh_every"));
  Count since_refresh = 0;
  bool deadline_hit = false;
  // A LEAVING VARIABLE LEAVES WHERE IT IS, NOT WHERE ITS BOUND IS (#888). The ratio test
  // lets a basic variable pass its bound - by the Harris tolerance, and by more when its
  // |alpha| is under the pivot floor and it cannot block - and when it later leaves, its
  // step is clamped at zero and it used to be SET to the bound: the difference went
  // straight into A x = b. On the 709,556-row refinery year that residual left 1e-10 at
  // pivot 28,000 and reached 1e15 by 55,000, under every update scheme and refactorization
  // cadence tried, with fresh factors accurate to 1e-12 throughout. So the variable leaves
  // at the value it has (the equations stay exact), and one off its bound by more than the
  // primal tolerance goes back on the queue to be pushed there by a ratio test like any
  // superbasic; Gill, Murray, Saunders & Wright (1989) keep the same invariant with their
  // expanding bounds, and reset at the end as the cleanup below does. The requeue count is
  // bounded by the queue's starting length, past which a variable just waits for cleanup.
  in_push_ = true;
  std::vector<Index> off_bound;
  // A variable is requeued ONCE: with a budget of the queue's starting length the refinery
  // year spent its last 294,000 pivots (of 1,252,000) on steps of length zero, a handful of
  // infeasible variables evicting one another in turn, and never reached the cleanup. One
  // that leaves off its bound a second time waits for the cleanup and the primal loop.
  std::vector<char> was_requeued(static_cast<std::size_t>(total_), 0);
  std::size_t requeued = 0;
  for (std::size_t next = 0; next < superbasic.size(); ++next) {
    const Index k = superbasic[next];
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
    // THE RATIO TEST IS HARRIS'S TWO PASSES WITH A RELATIVE PIVOT FLOOR (#888; Harris 1973,
    // Gill, Murray, Saunders & Wright 1989; the dual loop's rule, #244). A single pass that
    // takes the first blocking variable and refuses only |alpha| <= 1e-9 pivoted, on the
    // refinery year, on entries down to 1.3e-9 of a column whose largest entry is O(1),
    // and the basis went singular after 47,520 pivots from a primal-feasible point. Pass
    // one finds the longest step every basic variable allows when its bound is relaxed by
    // the primal tolerance; pass two takes, among the variables that block at or before
    // that step, the one with the LARGEST |alpha|, and no entry below the relative floor
    // (tol::kCrossoverPushPivotRelativeFloor of alpha's largest) can block at all.
    double alpha_max = 0.0;
    for (Index slot = 0; slot < m_; ++slot) {
      alpha_max = std::max(alpha_max, std::fabs(alpha_[static_cast<std::size_t>(slot)]));
    }
    const double asked_floor = options_.get_double("crossover_push_pivot_floor");
    const double pivot_floor = std::max(
        tol::kPivotTolerance,
        (asked_floor > 0.0 ? asked_floor : tol::kCrossoverPushPivotRelativeFloor) * alpha_max);
    const auto blocking = [&](Index slot, double tolerance, double* t, bool* to_upper) {
      const double a = alpha_[static_cast<std::size_t>(slot)];
      if (std::fabs(a) <= pivot_floor) return false;
      // x_B changes by -direction * a per unit step.
      const double rate = -static_cast<double>(direction) * a;
      const auto i = static_cast<std::size_t>(basis_[static_cast<std::size_t>(slot)]);
      const double xb = x_basic_[static_cast<std::size_t>(slot)];
      double room;
      if (rate < 0.0) {
        if (!is_finite_bound(lower_[i])) return false;
        room = xb - lower_[i] + tolerance;
      } else {
        if (!is_finite_bound(upper_[i])) return false;
        room = upper_[i] - xb + tolerance;
      }
      *t = std::max(0.0, room) / std::fabs(rate);
      *to_upper = rate > 0.0;
      return true;
    };
    double theta_max = distance;
    for (Index slot = 0; slot < m_; ++slot) {
      double t = 0.0;
      bool to_upper = false;
      if (blocking(slot, primal_tolerance_, &t, &to_upper)) theta_max = std::min(theta_max, t);
    }
    double step = distance;
    Index leaving_slot = -1;
    bool leaving_to_upper = false;
    double leaving_alpha = 0.0;
    for (Index slot = 0; slot < m_; ++slot) {
      double t = 0.0;
      bool to_upper = false;
      if (!blocking(slot, 0.0, &t, &to_upper) || t > theta_max) continue;
      const double a = std::fabs(alpha_[static_cast<std::size_t>(slot)]);
      if (a > leaving_alpha) {
        leaving_alpha = a;
        leaving_slot = slot;
        leaving_to_upper = to_upper;
        step = t;
      }
    }
    if (leaving_slot >= 0) step = std::min(step, distance);
    const double relative_pivot = alpha_max > 0.0 ? leaving_alpha / alpha_max : 1.0;
    window_max_move = std::max(window_max_move, step * alpha_max);

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
    const double leaving_bound = leaving_to_upper ? upper_[l] : lower_[l];
    const double leaving_now = x_basic_[slot];
    status_[l] = leaving_to_upper ? BasisStatus::kAtUpper : BasisStatus::kAtLower;
    if (lower_[l] == upper_[l]) status_[l] = BasisStatus::kFixed;
    // Within the push's own ceiling on the start's infeasibility (`loose`) it is set onto
    // the bound as before, and the next x_B solve absorbs the difference: a start the push
    // accepted 5e-5 infeasible (the refinery at T = 365) has basic slacks that far outside,
    // and requeueing every one of them doubled that push (45,027 requeues, 85,781 pivots
    // for 40,671) at a threshold of one tolerance and of ten alike.
    const double leaving_scale = std::max(1.0, std::fabs(leaving_bound));
    if (std::fabs(leaving_now - leaving_bound) <= loose * leaving_scale) {
      nonbasic_value_[l] = leaving_bound;
    } else {
      nonbasic_value_[l] = leaving_now;
      off_bound.push_back(leaving);
      if (was_requeued[l] == 0) {
        was_requeued[l] = 1;
        superbasic.push_back(leaving);
        ++requeued;
      }
    }
    basis_[slot] = k;
    basis_position_[u] = leaving_slot;
    status_[u] = BasisStatus::kBasic;
    x_basic_[slot] = value + static_cast<double>(direction) * step;
    nonbasic_value_[u] = x_basic_[slot];
    ++pivots;
    // The basis repair's stall guard (repair_basis) counts repairs that arrive "at the same
    // iteration"; the push never advanced that clock, so its whole run was one iteration
    // and the fourth singular refactorization in 173,093 pivots on the refinery year was
    // refused as a stall (#888). A push pivot is progress.
    iterations_seen_ = pivots;
    window_min_relative_pivot = std::min(window_min_relative_pivot, relative_pivot);
    const bool small_eta = refactor_below > 0.0 && relative_pivot < refactor_below;
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
    const bool refactor_now =
        !updated || small_eta || past_break_even || lu_.should_refactorize();
    if (!refactor_now && refresh_every > 0 && ++since_refresh >= refresh_every) {
      // x_B again from the factors, as the primal loop does every iteration: the push's
      // incrementally moved x_B carries every eta's error forward (#888).
      compute_basic_values();
      since_refresh = 0;
    }
    if (refactor_now) {
      since_refresh = 0;
      const ProfileScope timed(logger_.profiler(), "push refactorization",
                               ProfileMode::kDetailed);
      // WHERE THE PUSH'S FACTORS DEGRADE (#888), under profile=detailed only: one line per
      // refactorization with why it fired, the etas it discarded, the fresh factors' pivot
      // range, and the residual of [A | -I] x = 0 through the eta-updated x_B and then
      // through fresh factors. A residual that grows between refactorizations is the update;
      // one that stays large after a fresh factorization is the basis itself.
      const bool diagnose =
          logger_.profiler() != nullptr && logger_.profiler()->records(ProfileMode::kDetailed);
      const Index etas_before = lu_.eta_count();
      const char* why = !updated          ? "update refused"
                        : small_eta       ? "small eta pivot"
                        : past_break_even ? "break-even"
                                          : "eta cap";
      const double residual_before = diagnose ? push_residual() : 0.0;
      if (!refactorize()) {
        if (factors_abandoned_) return factorization_failed(pivots, timer);
        return finish(SolveStatus::kNumericalError,
                      "the basis went singular during the crossover push", pivots,
                      timer.elapsed_seconds());
      }
      ++refactorizations;
      eta_work_since_refactor_ = 0.0;
      compute_basic_values();
      if (diagnose) {
        logger_.info(
            "Push refactorization {} at pivot {}: {} ({} etas), pivots {:.3e} .. {:.3e} "
            "(spread {:.2e}), {} nonzeros, residual {:.3e} before, {:.3e} after; window: "
            "smallest eta pivot {:.2e} of its column, largest move {:.2e}",
            refactorizations, pivots, why, etas_before, lu_.smallest_pivot(),
            lu_.largest_pivot(), lu_.largest_pivot() / std::max(lu_.smallest_pivot(), 1e-300),
            lu_.factor_nonzeros(), residual_before, push_residual(), window_min_relative_pivot,
            window_max_move);
      }
      window_min_relative_pivot = 1.0;
      window_max_move = 0.0;
    }
  }
  in_push_ = false;
  // The cleanup: every variable that left off its bound and is still nonbasic goes onto it
  // now, and x_B is solved for afresh below; what that moves is the infeasibility the
  // check after it measures and the primal loop's phase 1 repairs.
  for (const Index k : off_bound) {
    const auto u = static_cast<std::size_t>(k);
    if (basis_position_[u] >= 0) continue;
    if (lower_[u] == upper_[u]) {
      status_[u] = BasisStatus::kFixed;
      nonbasic_value_[u] = lower_[u];
    } else if (status_[u] == BasisStatus::kAtUpper) {
      nonbasic_value_[u] = upper_[u];
    } else if (status_[u] == BasisStatus::kAtLower) {
      nonbasic_value_[u] = lower_[u];
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
      "Crossover push: {} superbasic variable(s) ({} of them requeued after leaving off a "
      "bound), {} arrived at a bound without a pivot, {} basis changes, {} refactorizations, "
      "max infeasibility {:.3e}{}",
      superbasic.size(), requeued, arrived, pivots, refactorizations, max_infeasibility(),
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
