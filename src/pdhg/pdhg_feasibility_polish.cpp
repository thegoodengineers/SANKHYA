// SPDX-License-Identifier: Apache-2.0
// SANKHYA - PDHG feasibility polishing (#483).
//
//   [AHLL25] Applegate, Hinder, Lu & Lubin, "PDLP: A Practical First-Order Method for
//            Large-Scale Linear Programming", arXiv:2501.07018 (2025), the feasibility
//            polishing section: when the main run's relative gap is small, solve the primal
//            feasibility problem (objective zero) from (x_k, 0) and the dual feasibility
//            problem (bounds zero) from (0, y_k) with PDHG for a fraction of the iterations
//            spent so far; first-order methods reach feasibility on such problems far faster
//            than optimality on the original, and starting from the current iterates keeps
//            the polished points close to them, so the gap of the polished pair stays near
//            the gap of the iterate it came from.
//   [PDLP]   Applegate, Diaz, Hinder, Lu, Lubin, O'Donoghue, Schudy, NeurIPS 2021, sections
//            3.1 (adaptive step), 3.2 (primal weight), 4.3 (restarts): the rules each phase
//            runs with, the same ones src/pdhg/pdhg.cpp applies to the main run.
//
// Written from the papers; per ENGINEERING_RULES.md the source of PDLP, OR-Tools, cuPDLP and
// every other LP solver was NOT consulted.
//
// WHAT IS NOT CLAIMED. The polished pair is measured, not assumed: its residuals and gap come
// from the same evaluate() that judges the main run, on the ORIGINAL problem. A phase that
// does not reach its target within its budget is discarded and the input half stands.

#include "pdhg_feasibility_polish.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "sankhya/tolerances.hpp"

namespace sankhya::pdhg {
namespace {

/// The phases test their target this often, and on their last step.
constexpr Count kPolishEvaluationInterval = 20;

double project(double value, double lower, double upper) {
  if (is_finite_bound(lower) && value < lower) return lower;
  if (is_finite_bound(upper) && value > upper) return upper;
  return value;
}

/// A finite bound moved to zero; an infinite one kept.
double homogenize(double bound) {
  return is_finite_bound(bound) ? 0.0 : bound;
}

}  // namespace

FeasibilityPolisher::FeasibilityPolisher(const Problem& problem, const Scaling& scaling,
                                         double spectral_norm)
    : problem_(problem),
      scaling_(scaling),
      spectral_norm_(spectral_norm),
      homogenized_(*problem.model) {
  const Model& model = *problem.model;
  const auto n = static_cast<std::size_t>(model.num_cols());
  const auto m = static_cast<std::size_t>(model.num_rows());

  zero_cost_.assign(n, 0.0);
  primal_problem_.model = &model;
  primal_problem_.cost = zero_cost_;
  primal_problem_.bound_norm = problem.bound_norm;
  primal_problem_.cost_norm = 0.0;

  for (std::size_t j = 0; j < n; ++j) {
    homogenized_.col_lower[j] = homogenize(homogenized_.col_lower[j]);
    homogenized_.col_upper[j] = homogenize(homogenized_.col_upper[j]);
  }
  for (std::size_t i = 0; i < m; ++i) {
    homogenized_.row_lower[i] = homogenize(homogenized_.row_lower[i]);
    homogenized_.row_upper[i] = homogenize(homogenized_.row_upper[i]);
  }
  dual_problem_.model = &homogenized_;
  dual_problem_.cost = problem.cost;
  dual_problem_.bound_norm = 0.0;
  dual_problem_.cost_norm = problem.cost_norm;

  // Scaling multiplies a bound by a positive factor, so zero stays zero and infinity stays
  // infinity: the scaled homogenized bounds are the homogenized scaled ones.
  h_col_lower_.resize(n);
  h_col_upper_.resize(n);
  h_row_lower_.resize(m);
  h_row_upper_.resize(m);
  for (std::size_t j = 0; j < n; ++j) {
    h_col_lower_[j] = homogenize(scaling.col_lower[j]);
    h_col_upper_[j] = homogenize(scaling.col_upper[j]);
  }
  for (std::size_t i = 0; i < m; ++i) {
    h_row_lower_[i] = homogenize(scaling.row_lower[i]);
    h_row_upper_[i] = homogenize(scaling.row_upper[i]);
  }
}

void FeasibilityPolisher::unscale(const std::vector<double>& xs, const std::vector<double>& ys,
                                  std::vector<double>& x, std::vector<double>& y) const {
  for (std::size_t j = 0; j < xs.size(); ++j) x[j] = xs[j] * scaling_.column[j];
  for (std::size_t i = 0; i < ys.size(); ++i) y[i] = ys[i] * scaling_.row[i];
}

FeasibilityPolisher::PhaseResult FeasibilityPolisher::run(
    const std::vector<double>& cost, const std::vector<double>& col_lower,
    const std::vector<double>& col_upper, const std::vector<double>& row_lower,
    const std::vector<double>& row_upper, const Problem& judged_on, std::vector<double> x,
    std::vector<double> y, Count budget, double omega, double eta,
    const std::function<bool(const Residuals&)>& done, const std::function<bool()>& stop) {
  const SparseMatrix& a = scaling_.matrix;
  const std::size_t n = x.size();
  const std::size_t m = y.size();

  PhaseResult result;
  for (std::size_t j = 0; j < n; ++j) x[j] = project(x[j], col_lower[j], col_upper[j]);

  std::vector<double> at_y(n, 0.0);
  std::vector<double> x_next(n, 0.0);
  std::vector<double> y_next(m, 0.0);
  std::vector<double> a_x(m, 0.0);
  std::vector<double> a_x_next(m, 0.0);
  std::vector<double> x_sum(n, 0.0);
  std::vector<double> y_sum(m, 0.0);
  std::vector<double> x_restart = x;
  std::vector<double> y_restart = y;
  std::vector<double> xu(n, 0.0);
  std::vector<double> x_avg(n, 0.0);
  std::vector<double> y_avg(m, 0.0);
  std::vector<double> yu(m, 0.0);
  if (m > 0) a.multiply(x.data(), a_x.data());

  const double eta_ceiling = 1.0e3 / std::max(spectral_norm_, 1e-12);
  Count averaged = 0;
  Count steps = 0;
  Count last_restart = 0;
  double restart_kkt = std::numeric_limits<double>::infinity();
  // A bound on attempted steps as well as accepted ones, so a run of rejected steps cannot
  // spin: each rejection shrinks eta, and the main loop's argument for termination applies,
  // but the budget is a promise about work and is kept as one.
  Count attempts = 0;
  const Count attempt_limit = 4 * budget + 16;

  const auto judge = [&](const std::vector<double>& xs, const std::vector<double>& ys) {
    unscale(xs, ys, xu, yu);
    return evaluate(judged_on, xu, yu, activity_, reduced_);
  };

  while (steps < budget && attempts < attempt_limit) {
    if (stop()) break;
    ++attempts;
    const double tau = eta / omega;
    const double sigma = eta * omega;

    if (m > 0) {
      a.transpose_multiply(y.data(), at_y.data());
    } else {
      std::fill(at_y.begin(), at_y.end(), 0.0);
    }
    for (std::size_t j = 0; j < n; ++j) {
      x_next[j] = project(x[j] - tau * (cost[j] + at_y[j]), col_lower[j], col_upper[j]);
    }
    if (m > 0) a.multiply(x_next.data(), a_x_next.data());
    for (std::size_t i = 0; i < m; ++i) {
      // A xbar = 2 A x' - A x by linearity, then the prox of sigma_C through Moreau.
      const double v = y[i] + sigma * (2.0 * a_x_next[i] - a_x[i]);
      y_next[i] = v - sigma * project(v / sigma, row_lower[i], row_upper[i]);
    }

    double movement = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      const double d = x_next[j] - x[j];
      movement += 0.5 * omega * d * d;
    }
    double interaction = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
      const double d = y_next[i] - y[i];
      movement += 0.5 * d * d / omega;
      interaction += d * (a_x_next[i] - a_x[i]);
    }
    interaction = std::fabs(interaction);

    // [PDLP] section 3.1, as in the main loop, including its two guards: zero interaction
    // accepts and leaves eta where it is, and the exponent starts at 2.
    const bool no_information = interaction <= 0.0;
    const double limit =
        no_information ? std::numeric_limits<double>::infinity() : movement / interaction;
    const double exponent = static_cast<double>(std::max<Count>(2, steps + 1));
    const double proposed = std::min((1.0 - std::pow(exponent, -0.3)) * limit,
                                     (1.0 + std::pow(exponent, -0.6)) * eta);
    if (eta <= limit) {
      x.swap(x_next);
      y.swap(y_next);
      a_x.swap(a_x_next);
      for (std::size_t j = 0; j < n; ++j) x_sum[j] += x[j];
      for (std::size_t i = 0; i < m; ++i) y_sum[i] += y[i];
      ++averaged;
      ++steps;
    } else {
      if (!no_information) eta = std::clamp(proposed, 1e-12, eta_ceiling);
      continue;
    }
    if (!no_information) eta = std::clamp(proposed, 1e-12, eta_ceiling);

    const bool last = steps >= budget;
    if (steps % kPolishEvaluationInterval != 0 && !last) continue;

    // The better of the current iterate and the average since the last restart, on this
    // phase's own KKT error; the phase ends the moment either has what it came for.
    const Residuals current = judge(x, y);
    Residuals average;
    bool average_better = false;
    if (averaged > 0) {
      const auto count = static_cast<double>(averaged);
      for (std::size_t j = 0; j < n; ++j) x_avg[j] = x_sum[j] / count;
      for (std::size_t i = 0; i < m; ++i) y_avg[i] = y_sum[i] / count;
      average = judge(x_avg, y_avg);
      average_better = average.worst() < current.worst();
    }
    if (done(current)) {
      result.reached = true;
      result.x = x;
      result.y = y;
      break;
    }
    if (averaged > 0 && done(average)) {
      result.reached = true;
      result.x = x_avg;
      result.y = y_avg;
      break;
    }
    if (last) break;

    // [PDLP] section 4.3 restart to the better candidate, and 3.2 primal weight update.
    const double kkt = average_better ? average.worst() : current.worst();
    const Count since = steps - last_restart;
    const bool artificial =
        since >= std::max<Count>(kPolishEvaluationInterval,
                                 static_cast<Count>(0.36 * static_cast<double>(steps)));
    if (kkt <= 0.2 * restart_kkt || artificial) {
      if (average_better) {
        x = x_avg;
        y = y_avg;
        if (m > 0) a.multiply(x.data(), a_x.data());
      }
      double dx2 = 0.0;
      double dy2 = 0.0;
      for (std::size_t j = 0; j < n; ++j) dx2 += (x[j] - x_restart[j]) * (x[j] - x_restart[j]);
      for (std::size_t i = 0; i < m; ++i) dy2 += (y[i] - y_restart[i]) * (y[i] - y_restart[i]);
      const double dx_norm = std::sqrt(dx2);
      const double dy_norm = std::sqrt(dy2);
      if (dx_norm > 1e-12 && dy_norm > 1e-12) {
        omega = std::exp(0.5 * std::log(dy_norm / dx_norm) + 0.5 * std::log(omega));
        omega = std::clamp(omega, 1e-6, 1e6);
      }
      std::fill(x_sum.begin(), x_sum.end(), 0.0);
      std::fill(y_sum.begin(), y_sum.end(), 0.0);
      averaged = 0;
      x_restart = x;
      y_restart = y;
      restart_kkt = kkt;
      last_restart = steps;
    }
  }
  result.iterations = attempts;
  return result;
}

FeasibilityPolisher::Outcome FeasibilityPolisher::polish(const std::vector<double>& x,
                                                         const std::vector<double>& y,
                                                         Count budget, double omega, double eta,
                                                         const std::function<bool()>& stop) {
  const std::size_t n = x.size();
  const std::size_t m = y.size();
  Outcome out;
  out.x = x;
  out.y = y;

  std::vector<double> xs(n);
  std::vector<double> ys(m);
  for (std::size_t j = 0; j < n; ++j) xs[j] = x[j] / scaling_.column[j];
  for (std::size_t i = 0; i < m; ++i) ys[i] = y[i] / scaling_.row[i];

  // Primal phase, from (x_k, 0) on the zero-objective problem. Done when x is feasible to
  // kPdhgTight relative AND the project's absolute primal tolerance, so the claim made of
  // the result is the one tools/verify_solution.py checks.
  const auto primal_done = [](const Residuals& r) {
    return r.primal <= tol::kPdhgTight && r.absolute_primal <= tol::kPrimalFeasibility;
  };
  PhaseResult primal = run(zero_cost_, scaling_.col_lower, scaling_.col_upper,
                           scaling_.row_lower, scaling_.row_upper, primal_problem_, xs,
                           std::vector<double>(m, 0.0), budget, omega, eta, primal_done, stop);
  out.iterations += primal.iterations;
  std::vector<double> scratch_x(n, 0.0);
  std::vector<double> scratch_y(m, 0.0);
  if (primal.reached) {
    out.primal_reached = true;
    unscale(primal.x, primal.y, out.x, scratch_y);
  }

  // Dual phase, from (0, y_k) on the homogenized problem. Its dual residual is measured
  // exactly as the original's would be: same matrix, same cost, same finite/infinite
  // pattern of bounds.
  const auto dual_done = [](const Residuals& r) {
    return r.dual <= tol::kPdhgTight && r.absolute_dual <= tol::kDualFeasibility;
  };
  PhaseResult dual =
      run(scaling_.cost, h_col_lower_, h_col_upper_, h_row_lower_, h_row_upper_, dual_problem_,
          std::vector<double>(n, 0.0), ys, budget, omega, eta, dual_done, stop);
  out.iterations += dual.iterations;
  if (dual.reached) {
    out.dual_reached = true;
    unscale(dual.x, dual.y, scratch_x, out.y);
  }

  out.residuals = evaluate(problem_, out.x, out.y, activity_, reduced_);
  return out;
}

}  // namespace sankhya::pdhg
