// SPDX-License-Identifier: Apache-2.0
// SANKHYA - convex QP by a proximal (primal-dual regularized) interior point (#490).
//
// References
//   Friedlander & Orban, "A primal-dual regularized interior-point method for convex quadratic
//     programs", Mathematical Programming Computation 4 (2012) - the regularized Newton
//     system and why it is quasi-definite.
//   Schwan, Jiang, Kuhn & Jones, "PIQP: a proximal interior-point quadratic programming
//     solver", arXiv 2304.00290 (2023), the paper only - proximal centres moved to the
//     current iterate every iteration, so the regularization changes the matrix and never the
//     residual being driven to zero.
//   Mehrotra, "On the implementation of a primal-dual interior point method", SIAM J. Optim.
//     2(4) (1992) - the predictor-corrector, sigma = (mu_aff / mu)^3.
//   Vanderbei, "Symmetric quasidefinite matrices", SIAM J. Optim. 5(1) (1995) - a
//     quasi-definite matrix has an LDL^T with D of known signs under EVERY symmetric
//     permutation, so the fill-reducing ordering needs no numerical pivoting.
//
// THE PROBLEM AS THE ITERATION SEES IT. Every row becomes an equality: an equality row stays
// as it is, and a ranged or one-sided row a'x gets a slack column w, a'x - w = 0, carrying the
// row's bounds as its own. Fixed columns are substituted out. What is left is
//
//     min  g'v + 0.5 v'Hv   s.t.  M v = b,   l <= v <= u     (some bounds infinite)
//
// with H = sense * Q on the original columns and zero on the slacks: equality constraints and
// bounds, nothing else, which is the first slice #490 allows. The multipliers are y (free, on
// M v = b) and z_l, z_u >= 0 on the finite bounds, and the KKT conditions are
//
//     H v + g - M'y - z_l + z_u = 0,   M v = b,   (v - l) z_l = mu,   (u - v) z_u = mu.
//
// THE NEWTON SYSTEM. Eliminating dz_l and dz_u leaves, with Theta^{-1} = Z_l S_l^{-1} +
// Z_u S_u^{-1} and the proximal parameters rho (on v) and delta (on y):
//
//     [ -(H + Theta^{-1} + rho I)   M' ] [dv]   [ r_d - rc_l ./ s_l + rc_u ./ s_u ]
//     [  M                      delta I ] [dy] = [ b - M v                        ]
//
// Negative definite over positive definite: quasi-definite for any rho, delta > 0, even when
// H is singular and a column is free (Theta^{-1} = 0). It is factored by the project's own
// sparse LDL^T (src/la/ldl.cpp) with the pivot signs enforced (factorize_quasidefinite), the
// AMD ordering unchanged. A pivot that comes out with the wrong sign or too small in
// magnitude means rho and delta are too small for the arithmetic; they are raised and the
// matrix refactored, the "raised on small pivots" rule of the issue.
//
// NOT IN THIS SLICE (#490's "Not done"): infeasibility and unboundedness detection from the
// proximal iterates (an infeasible model runs to the iteration ceiling and says so), warm
// starts, and scaling. The answer is judged by the same in-process KKT gate as every QP
// answer before it may be called optimal.

#include "sankhya/qp.hpp"
#include "sankhya/solve_control.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "../core/stop_controller.hpp"
#include "la/ldl.hpp"
#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

#include "convexity.hpp"
#include "qp_ipm_system.hpp"

namespace sankhya::qp {
namespace {

using ipm_detail::hessian_times;
using ipm_detail::inf_norm;
using ipm_detail::KktMatrix;
using ipm_detail::Standard;

constexpr double kFractionToBoundary = 0.995;  // Mehrotra (1992), and #490
constexpr Count kIterationCeiling = 200;       // an IPM that has not converged by here won't
constexpr Count kIterationsForProducts = 5;    // extra iterations to close the products

/// Solve K x = rhs with the factors, then two steps of iterative refinement against the
/// regularized K itself: the factors belong to K up to rounding and any lifted pivot, and the
/// refinement recovers what that costs without changing the system being solved.
void solve_refined(const SparseLdl& ldl, const SparseMatrix& k, const std::vector<double>& rhs,
                   std::vector<double>* x) {
  *x = rhs;
  ldl.solve(x->data());
  std::vector<double> kx(rhs.size()), correction(rhs.size());
  for (int step = 0; step < 2; ++step) {
    KktMatrix::multiply(k, *x, &kx);
    for (std::size_t i = 0; i < rhs.size(); ++i) correction[i] = rhs[i] - kx[i];
    ldl.solve(correction.data());
    for (std::size_t i = 0; i < rhs.size(); ++i) (*x)[i] += correction[i];
  }
}

}  // namespace

Solution solve_convex_qp_ipm(const Model& model, const Options& options, Logger& logger,
                             SolveControl* control) {
  Timer timer;
  Solution solution;
  solution.allocate_for(model);
  solution.algorithm = "qp-ipm";

  // ---- convexity, before any arithmetic: the refusal is the Condat-Vu engine's, unchanged --
  const ConvexityResult convexity = check_convexity(model);
  if (convexity.verdict != Convexity::kConvex) {
    solution.status = SolveStatus::kModelError;
    solution.message =
        convexity.verdict == Convexity::kIndefinite
            ? fmt::format(
                  "the quadratic objective is not convex: {}. A non-convex QP has "
                  "local minima and saddle points, and this engine would return one "
                  "of them labelled optimal, so it is refused instead",
                  convexity.detail)
            : fmt::format("convexity could not be established: {}", convexity.detail);
    logger.warning("{}", solution.message);
    solution.solve_seconds = timer.elapsed_seconds();
    return solution;
  }
  for (Index j = 0; j < model.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (model.col_lower[u] > model.col_upper[u]) {
      solution.status = SolveStatus::kNotSolved;
      solution.message = fmt::format(
          "column {} has crossed bounds; the interior point needs "
          "an interior",
          j);
      solution.solve_seconds = timer.elapsed_seconds();
      return solution;
    }
  }

  const Standard s = ipm_detail::standardize(model);
  const auto nc = static_cast<std::size_t>(s.cols);
  const auto nr = static_cast<std::size_t>(s.rows);
  const double tolerance = options.get_double("qp_ipm_tolerance");
  const ResourceLimits limits(options, logger);
  StopController stop(control, timer, limits);

  // ---- starting point: strictly inside every finite bound, unit bound multipliers ---------
  std::vector<double> v(nc, 0.0), y(nr, 0.0), zl(nc, 0.0), zu(nc, 0.0);
  std::vector<bool> has_lower(nc), has_upper(nc);
  Index bound_count = 0;
  for (std::size_t j = 0; j < nc; ++j) {
    has_lower[j] = is_finite_bound(s.lower[j]);
    has_upper[j] = is_finite_bound(s.upper[j]);
    const double width = (has_lower[j] && has_upper[j]) ? s.upper[j] - s.lower[j] : kInfinity;
    const double inset = std::min(1.0, 0.5 * width);
    double value = 0.0;
    if (has_lower[j] && value < s.lower[j] + inset) value = s.lower[j] + inset;
    if (has_upper[j] && value > s.upper[j] - inset) value = s.upper[j] - inset;
    v[j] = value;
    if (has_lower[j]) {
      zl[j] = 1.0;
      ++bound_count;
    }
    if (has_upper[j]) {
      zu[j] = 1.0;
      ++bound_count;
    }
  }

  KktMatrix kkt(s);
  SparseLdl ldl;
  const Index dim = s.cols + s.rows;
  std::vector<signed char> signs(static_cast<std::size_t>(dim), 1);
  for (std::size_t j = 0; j < nc; ++j) signs[j] = -1;

  const auto should_stop = [&]() { return limits.time_exhausted(timer.elapsed_seconds()); };
  {
    const SparseMatrix pattern = kkt.build(std::vector<double>(nc, 1.0), 1.0, 1.0);
    if (!ldl.analyze(pattern, should_stop)) {
      solution.status = ldl.stopped_early() ? SolveStatus::kTimeLimit : SolveStatus::kNotSolved;
      solution.message =
          ldl.stopped_early()
              ? limits.describe(LimitReason::kTime, timer.elapsed_seconds(), 0, 0)
              : "the KKT system's ordering could not be computed";
      solution.solve_seconds = timer.elapsed_seconds();
      return solution;
    }
  }

  std::vector<double> hv(nc), mv(nr), mty(nc), rd(nc), rp(nr), theta_inverse(nc);
  std::vector<double> rhs(static_cast<std::size_t>(dim)), step(static_cast<std::size_t>(dim));
  std::vector<double> dv(nc), dy(nr), dzl(nc), dzu(nc), rcl(nc), rcu(nc), dv_aff(nc);
  std::vector<double> dzl_aff(nc), dzu_aff(nc);
  double rho = options.get_double("qp_ipm_regularization");
  double delta = rho;
  const double regularization_floor = rho;

  SolveStatus status = SolveStatus::kIterationLimit;
  std::string message;
  Count iterations = 0;
  Count iterations_past_relative = 0;
  std::vector<double> kept_v, kept_y;
  double primal_rel = kInfinity, dual_rel = kInfinity, gap_rel = kInfinity;

  const auto residuals = [&]() {
    hessian_times(s.h, v, &hv);
    std::fill(mv.begin(), mv.end(), 0.0);
    s.m.multiply_add(v.data(), mv.data());
    std::fill(mty.begin(), mty.end(), 0.0);
    if (s.rows > 0) s.m.transpose_multiply_add(y.data(), mty.data());
    for (std::size_t j = 0; j < nc; ++j) rd[j] = hv[j] + s.g[j] - mty[j] - zl[j] + zu[j];
    for (std::size_t i = 0; i < nr; ++i) rp[i] = s.b[i] - mv[i];
  };

  // The step for a given complementarity target: rcl and rcu hold sigma mu - s z (- the
  // Mehrotra cross term), and the result lands in dv, dy, dzl, dzu.
  const auto newton = [&](const SparseMatrix& k) {
    for (std::size_t j = 0; j < nc; ++j) {
      double r = rd[j];
      if (has_lower[j]) r -= rcl[j] / (v[j] - s.lower[j]);
      if (has_upper[j]) r += rcu[j] / (s.upper[j] - v[j]);
      rhs[j] = r;
    }
    for (std::size_t i = 0; i < nr; ++i) rhs[nc + i] = rp[i];
    solve_refined(ldl, k, rhs, &step);
    for (std::size_t j = 0; j < nc; ++j) {
      dv[j] = step[j];
      dzl[j] = has_lower[j] ? (rcl[j] - zl[j] * dv[j]) / (v[j] - s.lower[j]) : 0.0;
      dzu[j] = has_upper[j] ? (rcu[j] + zu[j] * dv[j]) / (s.upper[j] - v[j]) : 0.0;
    }
    for (std::size_t i = 0; i < nr; ++i) dy[i] = step[nc + i];
  };

  // The largest alpha in (0, 1] keeping every slack and bound multiplier non-negative.
  const auto longest_step = [&]() {
    double alpha = 1.0;
    for (std::size_t j = 0; j < nc; ++j) {
      if (has_lower[j]) {
        if (dv[j] < 0.0) alpha = std::min(alpha, -(v[j] - s.lower[j]) / dv[j]);
        if (dzl[j] < 0.0) alpha = std::min(alpha, -zl[j] / dzl[j]);
      }
      if (has_upper[j]) {
        if (dv[j] > 0.0) alpha = std::min(alpha, (s.upper[j] - v[j]) / dv[j]);
        if (dzu[j] < 0.0) alpha = std::min(alpha, -zu[j] / dzu[j]);
      }
    }
    return alpha;
  };

  // ---- starting point, after Mehrotra (1992, sec. 7), adapted to bounds ------------------
  // v and y from min g'v + v'(H + I)v/2 s.t. M v = b: one factorization of the same pattern
  // with Theta^{-1} = I, so the start already nearly satisfies the rows (from the unit start
  // the primal residual stayed at 1.0 for thirty iterations on qpcboei2). v is then moved
  // inside its bounds by a margin that grows with how far outside the least-squares point
  // was, and the bound multipliers take the sign-split stationarity residual plus a shift
  // that balances them against the slacks.
  {
    std::fill(theta_inverse.begin(), theta_inverse.end(), 1.0);
    const SparseMatrix k0 = kkt.build(theta_inverse, rho, delta);
    if (ldl.factorize_quasidefinite(k0, signs, 0.1 * std::min(rho, delta), should_stop) &&
        ldl.regularized_pivots() == 0) {
      for (std::size_t j = 0; j < nc; ++j) rhs[j] = s.g[j];
      for (std::size_t i = 0; i < nr; ++i) rhs[nc + i] = s.b[i];
      solve_refined(ldl, k0, rhs, &step);
      if (std::all_of(step.begin(), step.end(), [](double x) { return std::isfinite(x); })) {
        std::copy(step.begin(), step.begin() + static_cast<std::ptrdiff_t>(nc), v.begin());
        std::copy(step.begin() + static_cast<std::ptrdiff_t>(nc), step.end(), y.begin());
      }
    }
    double worst = 0.0;
    for (std::size_t j = 0; j < nc; ++j) {
      if (has_lower[j]) worst = std::min(worst, v[j] - s.lower[j]);
      if (has_upper[j]) worst = std::min(worst, s.upper[j] - v[j]);
    }
    const double margin = std::max(1.0, -1.5 * worst);
    for (std::size_t j = 0; j < nc; ++j) {
      const double width = (has_lower[j] && has_upper[j]) ? s.upper[j] - s.lower[j] : kInfinity;
      const double inset = std::min(margin, 0.5 * width);
      if (has_lower[j] && v[j] < s.lower[j] + inset) v[j] = s.lower[j] + inset;
      if (has_upper[j] && v[j] > s.upper[j] - inset) v[j] = s.upper[j] - inset;
      zl[j] = 0.0;
      zu[j] = 0.0;
    }
    residuals();  // with z = 0, rd is the stationarity residual H v + g - M'y
    double slack_times_z = 0.0, slack_sum = 0.0;
    for (std::size_t j = 0; j < nc; ++j) {
      if (has_lower[j]) {
        zl[j] = std::max(rd[j], 0.0);
        slack_times_z += (v[j] - s.lower[j]) * zl[j];
        slack_sum += v[j] - s.lower[j];
      }
      if (has_upper[j]) {
        zu[j] = std::max(-rd[j], 0.0);
        slack_times_z += (s.upper[j] - v[j]) * zu[j];
        slack_sum += s.upper[j] - v[j];
      }
    }
    const double shift = std::max(1.0, slack_sum > 0.0 ? 0.5 * slack_times_z / slack_sum : 0.0);
    for (std::size_t j = 0; j < nc; ++j) {
      if (has_lower[j]) zl[j] += shift;
      if (has_upper[j]) zu[j] += shift;
    }
  }

  while (true) {
    residuals();
    double complementarity = 0.0;
    double largest_product = 0.0;
    for (std::size_t j = 0; j < nc; ++j) {
      if (has_lower[j]) {
        const double product = (v[j] - s.lower[j]) * zl[j];
        complementarity += product;
        largest_product = std::max(largest_product, product);
      }
      if (has_upper[j]) {
        const double product = (s.upper[j] - v[j]) * zu[j];
        complementarity += product;
        largest_product = std::max(largest_product, product);
      }
    }
    // A row's product as the KKT check will measure it: its multiplier against the distance
    // from the row ACTIVITY a'x = w - r_p to the nearer bound, not from the slack w, which
    // differs by the primal residual (primalc1: 3.9e-06 on a row whose w-product was tiny).
    for (Index k = 0; k < s.slacks; ++k) {
      const auto u = static_cast<std::size_t>(s.free_n + k);
      const auto r = static_cast<std::size_t>(s.slack_row[static_cast<std::size_t>(k)]);
      const double activity = v[u] - rp[r];
      double distance = kInfinity;
      if (has_lower[u]) distance = std::min(distance, std::fabs(activity - s.lower[u]));
      if (has_upper[u]) distance = std::min(distance, std::fabs(s.upper[u] - activity));
      if (std::isfinite(distance)) {
        largest_product = std::max(largest_product, std::fabs(y[r]) * distance);
      }
    }
    const double mu =
        bound_count > 0 ? complementarity / static_cast<double>(bound_count) : 0.0;

    // Relative residuals, and the Dorn duality gap: primal g'v + v'Hv/2 against
    // b'y + l'z_l - u'z_u - v'Hv/2.
    double vhv = 0.0, gv = 0.0, dual = 0.0;
    for (std::size_t j = 0; j < nc; ++j) {
      vhv += v[j] * hv[j];
      gv += s.g[j] * v[j];
      if (has_lower[j]) dual += s.lower[j] * zl[j];
      if (has_upper[j]) dual -= s.upper[j] * zu[j];
    }
    for (std::size_t i = 0; i < nr; ++i) dual += s.b[i] * y[i];
    const double primal_objective = gv + 0.5 * vhv;
    const double dual_objective = dual - 0.5 * vhv;
    primal_rel = inf_norm(rp) / (1.0 + std::max(inf_norm(mv), inf_norm(s.b)));
    dual_rel = inf_norm(rd) / (1.0 + std::max({inf_norm(hv), inf_norm(s.g), inf_norm(mty)}));
    gap_rel = std::fabs(primal_objective - dual_objective) /
              (1.0 + std::max(std::fabs(primal_objective), std::fabs(dual_objective)));

    logger.iteration(iterations, primal_objective, primal_rel, dual_rel,
                     timer.elapsed_seconds());
    // The relative gap alone is not enough: an optimality claim is also held to an ABSOLUTE
    // complementarity product per item (tol::kComplementarity, the verifier's test), which a
    // relative gap of 1e-9 on an objective of 5e+05 does not imply (qadlittl: 1.1e-06 on one
    // row). A tenth of it leaves room for the mapping back to the model's rows and columns.
    //
    // When the relative measures are met and the products are not, a few more iterations
    // usually close them; when they cannot (a row whose multiplier is 1e+03 needs its
    // activity on the bound to 1e-10, below what the residual reaches in double precision),
    // the point is handed back as an optimality claim and the in-process KKT gate in solve()
    // decides, withdrawing it to feasible with the product named if it is not backed. The
    // last point that met the relative measures is kept, so an iterate that stops being
    // finite while mu is pushed to zero cannot cost the answer already found.
    const bool relative_met =
        primal_rel <= tolerance && dual_rel <= tolerance && gap_rel <= tolerance;
    if (relative_met && (largest_product <= 0.1 * tol::kComplementarity ||
                         iterations_past_relative >= kIterationsForProducts)) {
      status = SolveStatus::kOptimal;
      break;
    }
    if (relative_met) {
      ++iterations_past_relative;
      kept_v = v;
      kept_y = y;
    }
    if (limits.time_exhausted(timer.elapsed_seconds())) {
      status = SolveStatus::kTimeLimit;
      message = limits.describe(LimitReason::kTime, timer.elapsed_seconds(), iterations, 0);
      break;
    }
    if (limits.iterations_exhausted(iterations) ||
        (limits.iteration_limit() < 0 && iterations >= kIterationCeiling)) {
      status = SolveStatus::kIterationLimit;
      message = limits.iteration_limit() >= 0
                    ? limits.describe(LimitReason::kIterations, timer.elapsed_seconds(),
                                      iterations, 0)
                    : fmt::format(
                          "stopped at this engine's own ceiling of {} iterations with "
                          "relative residuals {:.1e} primal, {:.1e} dual, {:.1e} gap",
                          kIterationCeiling, primal_rel, dual_rel, gap_rel);
      break;
    }
    SolveStatus stop_status;
    if (stop.should_stop(
            [&]() {
              Progress p;
              p.phase = Progress::Phase::kLp;
              p.iterations = iterations;
              p.objective = kInfinity;
              p.best_bound = -kInfinity;
              return p;
            },
            &stop_status)) {
      status = stop_status;
      message = limits.describe(
          stop_status == SolveStatus::kTimeLimit ? LimitReason::kTime : LimitReason::kInterrupt,
          timer.elapsed_seconds(), iterations, 0);
      break;
    }
    ++iterations;

    // ---- factor, raising the proximal parameters while a pivot comes out wrong ----------
    for (std::size_t j = 0; j < nc; ++j) {
      double t = 0.0;
      if (has_lower[j]) t += zl[j] / (v[j] - s.lower[j]);
      if (has_upper[j]) t += zu[j] / (s.upper[j] - v[j]);
      theta_inverse[j] = t;
    }
    SparseMatrix k;
    bool factored = false;
    for (int attempt = 0; attempt < 8 && !factored; ++attempt) {
      k = kkt.build(theta_inverse, rho, delta);
      if (!ldl.factorize_quasidefinite(k, signs, 0.1 * std::min(rho, delta), should_stop)) {
        break;
      }
      if (ldl.regularized_pivots() == 0) {
        factored = true;
      } else {
        rho *= 100.0;
        delta *= 100.0;
      }
    }
    if (!factored) {
      status = ldl.stopped_early() ? SolveStatus::kTimeLimit : SolveStatus::kNumericalError;
      message =
          ldl.stopped_early()
              ? limits.describe(LimitReason::kTime, timer.elapsed_seconds(), iterations, 0)
              : fmt::format(
                    "the regularized KKT system lost its quasi-definite sign "
                    "pattern at rho = {:.1e}",
                    rho);
      break;
    }

    // ---- predictor ------------------------------------------------------------------------
    for (std::size_t j = 0; j < nc; ++j) {
      rcl[j] = has_lower[j] ? -(v[j] - s.lower[j]) * zl[j] : 0.0;
      rcu[j] = has_upper[j] ? -(s.upper[j] - v[j]) * zu[j] : 0.0;
    }
    newton(k);
    double sigma = 0.0;
    if (bound_count > 0) {
      const double alpha_aff = longest_step();
      double affine = 0.0;
      for (std::size_t j = 0; j < nc; ++j) {
        if (has_lower[j]) {
          affine += (v[j] - s.lower[j] + alpha_aff * dv[j]) * (zl[j] + alpha_aff * dzl[j]);
        }
        if (has_upper[j]) {
          affine += (s.upper[j] - v[j] - alpha_aff * dv[j]) * (zu[j] + alpha_aff * dzu[j]);
        }
      }
      const double mu_aff = affine / static_cast<double>(bound_count);
      sigma = mu > 0.0 ? std::pow(mu_aff / mu, 3.0) : 0.0;
      sigma = std::min(1.0, std::max(0.0, sigma));

      // ---- corrector: the centring target and Mehrotra's second-order term ---------------
      dv_aff = dv;
      dzl_aff = dzl;
      dzu_aff = dzu;
      for (std::size_t j = 0; j < nc; ++j) {
        if (has_lower[j]) {
          rcl[j] = sigma * mu - (v[j] - s.lower[j]) * zl[j] - dv_aff[j] * dzl_aff[j];
        }
        if (has_upper[j]) {
          rcu[j] = sigma * mu - (s.upper[j] - v[j]) * zu[j] + dv_aff[j] * dzu_aff[j];
        }
      }
      newton(k);
    }

    const double alpha =
        bound_count > 0 ? std::min(1.0, kFractionToBoundary * longest_step()) : 1.0;
    for (std::size_t j = 0; j < nc; ++j) {
      v[j] += alpha * dv[j];
      zl[j] += alpha * dzl[j];
      zu[j] += alpha * dzu[j];
    }
    for (std::size_t i = 0; i < nr; ++i) y[i] += alpha * dy[i];
    const auto finite = [](const std::vector<double>& x) {
      return std::all_of(x.begin(), x.end(), [](double e) { return std::isfinite(e); });
    };
    if (!finite(v) || !finite(y) || !finite(zl) || !finite(zu)) {
      if (!kept_v.empty()) {
        v = kept_v;  // the last point that met the relative measures; the gate judges it
        y = kept_y;
        status = SolveStatus::kOptimal;
      } else {
        status = SolveStatus::kNumericalError;
        message = "the interior point's iterate stopped being finite";
      }
      break;
    }

    // The proximal parameters follow mu down, never below the configured floor (the
    // centres move with the iterate, so a large rho slows the method and never biases it).
    rho = std::max(regularization_floor, std::min(rho, 0.1 * mu));
    delta = rho;
  }

  // ---- back to the caller's model -------------------------------------------------------
  const double sense = model.sense_multiplier();
  for (Index j = 0; j < s.n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const Index kcol = s.column_of[u];
    solution.col_value[u] = kcol < 0 ? s.fixed_value[u] : v[static_cast<std::size_t>(kcol)];
  }
  for (Index i = 0; i < model.num_rows(); ++i) {
    const auto u = static_cast<std::size_t>(i);
    const Index r = s.row_of[u];
    solution.row_dual[u] = r < 0 ? 0.0 : sense * y[static_cast<std::size_t>(r)];
  }
  // Reduced costs in the model's own sense, c + Q x - A' row_dual, the quantity the Condat-Vu
  // engine reports and the KKT check derives.
  std::vector<double> qx(static_cast<std::size_t>(s.n), 0.0);
  hessian_times(model.hessian, solution.col_value, &qx);
  for (Index j = 0; j < s.n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    double d = model.col_cost[u] + qx[u];
    const ColumnView column = model.matrix.column(j);
    for (Index p = 0; p < column.size; ++p) {
      d -= column.values[p] * solution.row_dual[static_cast<std::size_t>(column.rows[p])];
    }
    solution.col_dual[u] = d;
  }
  solution.status = status;
  solution.message = message;
  solution.iterations = iterations;
  solution.solve_seconds = timer.elapsed_seconds();
  solution.dual_bound = status == SolveStatus::kOptimal
                            ? model.evaluate_objective(solution.col_value.data())
                            : (model.sense == ObjSense::kMaximize ? kInfinity : -kInfinity);
  solution.recompute_quality(model);
  logger.verbose(
      "QP interior point: {} iterations, relative residuals {:.2e} primal, {:.2e} "
      "dual, {:.2e} gap",
      iterations, primal_rel, dual_rel, gap_rel);
  return solution;
}

}  // namespace sankhya::qp
