// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the KKT conditions of a nonlinear model at a point (NLP stage 2). See
// nlp_kkt_check.hpp for every measure and its scaling.

#include "nlp/nlp_kkt_check.hpp"

#include <algorithm>
#include <cmath>

#include <fmt/format.h>

namespace sankhya::nlp {
namespace {

/// How far a multiplier's sign contradicts the bounds it can price (verify_solution.py's
/// sign_violation): positive needs a finite lower bound, negative a finite upper bound; an
/// equality admits any sign.
double sign_violation(double multiplier, double lower, double upper) {
  if (lower == upper) return 0.0;
  if (multiplier > 0.0 && !std::isfinite(lower)) return multiplier;
  if (multiplier < 0.0 && !std::isfinite(upper)) return -multiplier;
  return 0.0;
}

/// |multiplier| * slack, or |multiplier| where no bound is finite (verify_solution.py's
/// complementarity).
double complementarity_of(double multiplier, double value, double lower, double upper) {
  const double below = std::isfinite(lower) ? value - lower : kInfinity;
  const double above = std::isfinite(upper) ? upper - value : kInfinity;
  const double slack = std::min(below, above);
  return std::isfinite(slack) ? std::fabs(multiplier) * std::max(slack, 0.0)
                              : std::fabs(multiplier);
}

}  // namespace

NlpKktReport check_nlp_kkt(const NlpProblem& problem, const Vec& x, const Vec& y, const Vec& d,
                           double primal, double dual, double complementarity) {
  NlpKktReport r;
  const Index n = problem.num_variables();
  const Index m = problem.num_constraints();
  Evaluation error;
  double f = 0.0;
  Vec grad, g, jac;
  if (!problem.objective_gradient(x, &f, &grad, &error) ||
      !problem.constraints(x, &g, &error) || !problem.jacobian(x, &jac, &error)) {
    r.failed = "the functions are undefined at the point: " + error.message;
    return r;
  }
  r.evaluated = true;
  const auto& xl = problem.x_lower();
  const auto& xu = problem.x_upper();
  const auto& gl = problem.g_lower();
  const auto& gu = problem.g_upper();
  std::string where;
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double v = std::max({xl[u] - x[u], x[u] - xu[u], 0.0});
    const double s = v / std::max(1.0, std::fabs(x[u]));
    if (s > r.primal) {
      r.primal = s;
      r.primal_absolute = v;
      where = fmt::format("column {}", j);
    }
  }
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double v = std::max({gl[u] - g[u], g[u] - gu[u], 0.0});
    const double s = v / std::max(1.0, std::fabs(g[u]));
    if (s > r.primal) {
      r.primal = s;
      r.primal_absolute = v;
      where = fmt::format("row {}", i);
    }
  }
  r.primal_ok = r.primal <= primal;
  if (!r.primal_ok) {
    r.failed = fmt::format("primal feasibility: {:.3e} relative on {}", r.primal, where);
  }

  // Stationarity and the column signs, each scaled by the terms of its own column.
  Vec jt_y(static_cast<std::size_t>(n), 0.0), scale(static_cast<std::size_t>(n), 1.0);
  for (Index j = 0; j < n; ++j) {
    scale[static_cast<std::size_t>(j)] =
        std::max(1.0, std::fabs(grad[static_cast<std::size_t>(j)]));
  }
  const auto& starts = problem.jacobian_starts();
  const auto& cols = problem.jacobian_columns();
  for (Index i = 0; i < m; ++i) {
    for (Index k = starts[static_cast<std::size_t>(i)];
         k < starts[static_cast<std::size_t>(i) + 1]; ++k) {
      const auto j = static_cast<std::size_t>(cols[static_cast<std::size_t>(k)]);
      const double term = y[static_cast<std::size_t>(i)] * jac[static_cast<std::size_t>(k)];
      jt_y[j] += term;
      scale[j] = std::max(scale[j], std::fabs(term));
    }
  }
  std::string dual_where, sign_where, compl_where;
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double s = std::fabs(grad[u] - jt_y[u] - d[u]) / scale[u];
    if (s > r.stationarity) {
      r.stationarity = s;
      dual_where = fmt::format("column {}", j);
    }
    const double sv = sign_violation(d[u], xl[u], xu[u]) / scale[u];
    if (sv > r.sign) {
      r.sign = sv;
      sign_where = fmt::format("column {}", j);
    }
    if (xl[u] != xu[u]) {
      const double c = complementarity_of(d[u], x[u], xl[u], xu[u]);
      if (c > r.complementarity) {
        r.complementarity = c;
        compl_where = fmt::format("column {}", j);
      }
    }
  }
  double y_norm = 1.0;
  for (const double v : y) y_norm = std::max(y_norm, std::fabs(v));
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double sv = sign_violation(y[u], gl[u], gu[u]) / y_norm;
    if (sv > r.sign) {
      r.sign = sv;
      sign_where = fmt::format("row {}", i);
    }
    if (gl[u] != gu[u]) {
      const double c = complementarity_of(y[u], g[u], gl[u], gu[u]);
      if (c > r.complementarity) {
        r.complementarity = c;
        compl_where = fmt::format("row {}", i);
      }
    }
  }
  if (r.failed.empty() && r.stationarity > dual) {
    r.failed = fmt::format("stationarity: {:.3e} relative on {}", r.stationarity, dual_where);
  }
  if (r.failed.empty() && r.sign > dual) {
    r.failed = fmt::format("multiplier sign: {:.3e} relative on {}", r.sign, sign_where);
  }
  if (r.failed.empty() && r.complementarity > complementarity) {
    r.failed = fmt::format("complementarity: {:.3e} on {}", r.complementarity, compl_where);
  }
  r.passed = r.failed.empty();
  return r;
}

}  // namespace sankhya::nlp
