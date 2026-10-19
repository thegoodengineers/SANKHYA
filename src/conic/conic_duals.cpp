// SPDX-License-Identifier: Apache-2.0
// SANKHYA - dual recovery at the conic optimum (#755).
//
// WHY. An interior point on second-order cones converges to the optimum with its point within
// the gap, but its multipliers row by row only within about the square root of it: the
// second-order cone's error bound is Holderian with exponent 1/2, where a polyhedral one is
// linear (Sturm, "Error bounds for linear matrix inequalities", SIAM J. Optim. 10 (2000);
// for SOCP, Alizadeh & Goldfarb, "Second-order cone programming", Math. Programming 95 (2003),
// section 7). Measured on the unit disc, the multiplier off by 7.8e-6 at a gap of 1.2e-9 and by
// 3.4e-7 at 2.3e-13. Only their combination A'y + s = c is accurate to the tolerance, and a
// model's KKT conditions are judged row by row (tools/verify_solution.py), so a reduced cost of
// 1e-5 on a column strictly inside its bounds fails complementarity at any reachable gap.
//
// WHAT. The point is accurate, so the active set can be read off it: every equality row, every
// row and column bound within tol::kConicActiveShare of its slack scale, and every one whose
// interior-point multiplier outweighs its slack, both scaled. The second test is Tapia's
// indicator (El-Bakry, Tapia & Zhang, "A study of indicators for identifying zero variables in
// interior-point methods", SIAM Review 36 (1994)): at an interior-point iterate an active
// constraint has slack near mu / y and an inactive one a multiplier near mu / slack, so the
// ratio separates them whatever the multiplier's size - where a fixed window misses an active
// row whose multiplier is small and whose slack therefore sits above it. Stationarity at the
// point, grad f(x) = sum_i y_i grad g_i(x) + d with d supported on the active bounds, is then
// a linear least-squares problem in the active multipliers, min || r - M lambda ||, with r the
// objective's gradient and M's columns the active rows' gradients a_i + 2 Q_i x and the unit
// vectors of the active bounds - the classical way to obtain multipliers from a primal
// solution (Nocedal & Wright, "Numerical Optimization", 2nd ed. (2006), section 18.3,
// least-squares multiplier estimates). It is solved by the normal equations
// (M'M + eps I) lambda = M'r with the project's sparse LDL^T, positive definite for eps > 0
// even when the active gradients are dependent, and refined on the residual r - M lambda.
//
// SAFE BY CONSTRUCTION. The recovered multipliers replace the interior point's only when their
// measured KKT error - sign violations of every multiplier and |multiplier| * slack - is
// smaller. A misread active set costs the recovery, never the answer, and the verifier judges
// whichever is kept.

#include "conic/conic_duals.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "la/ldl.hpp"
#include "qp/qp_ipm_system.hpp"
#include "sankhya/sparse.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::conic {
namespace {

using Gradient = std::vector<std::pair<Index, double>>;

/// Row i's gradient a_i + 2 Q_i x over the model's columns, merged per column.
std::vector<Gradient> row_gradients(const QcqpModel& model, const std::vector<double>& x) {
  const Model& lin = model.linear;
  std::vector<Gradient> out(static_cast<std::size_t>(lin.num_rows()));
  const CsrView rows(lin.matrix);
  for (Index i = 0; i < lin.num_rows(); ++i) {
    const ColumnView row = rows.row(i);
    for (Index p = 0; p < row.size; ++p) {
      out[static_cast<std::size_t>(i)].emplace_back(row.rows[p], row.values[p]);
    }
  }
  for (const QuadraticTerm& t : model.quadratic) {
    Gradient& g = out[static_cast<std::size_t>(t.row)];
    g.emplace_back(t.first, t.value * x[static_cast<std::size_t>(t.second)]);
    g.emplace_back(t.second, t.value * x[static_cast<std::size_t>(t.first)]);
  }
  for (Gradient& g : out) {
    std::sort(g.begin(), g.end());
    Gradient merged;
    for (const auto& [j, v] : g) {
      if (!merged.empty() && merged.back().first == j) {
        merged.back().second += v;
      } else {
        merged.emplace_back(j, v);
      }
    }
    g = std::move(merged);
  }
  return out;
}

/// The worst KKT violation of row multipliers `y` (minimization space): the reduced costs
/// d = r - sum_i y_i g_i follow from them, and every multiplier is held to its sign against
/// the bounds that exist and to |multiplier| * slack. Relative to max(1, |r|).
double kkt_error(const QcqpModel& model, const std::vector<double>& x,
                 const std::vector<double>& activity, const std::vector<double>& r,
                 const std::vector<Gradient>& gradients, const std::vector<double>& y,
                 std::vector<double>* d_out) {
  const Model& lin = model.linear;
  std::vector<double> d = r;
  for (std::size_t i = 0; i < y.size(); ++i) {
    if (y[i] == 0.0) continue;
    for (const auto& [j, v] : gradients[i]) d[static_cast<std::size_t>(j)] -= y[i] * v;
  }
  const auto violation = [](double multiplier, double value, double lower, double upper) {
    if (lower == upper) return 0.0;
    double worst = 0.0;
    if (multiplier > 0.0 && !is_finite_bound(lower)) worst = multiplier;
    if (multiplier < 0.0 && !is_finite_bound(upper)) worst = -multiplier;
    const double slack = std::min(is_finite_bound(lower) ? value - lower : kInfinity,
                                  is_finite_bound(upper) ? upper - value : kInfinity);
    if (std::isfinite(slack)) worst = std::max(worst, std::fabs(multiplier) * slack);
    return worst;
  };
  double worst = 0.0;
  for (std::size_t j = 0; j < d.size(); ++j) {
    worst = std::max(worst, violation(d[j], x[j], lin.col_lower[j], lin.col_upper[j]));
  }
  for (std::size_t i = 0; i < y.size(); ++i) {
    worst = std::max(worst, violation(y[i], activity[i], lin.row_lower[i], lin.row_upper[i]));
  }
  double scale = 1.0;
  for (const double e : r) scale = std::max(scale, std::fabs(e));
  if (d_out != nullptr) *d_out = std::move(d);
  return worst / scale;
}

}  // namespace

bool recover_duals(const QcqpModel& model, Solution* solution) {
  const Model& lin = model.linear;
  const Index n = lin.num_cols();
  const Index m = lin.num_rows();
  if (!model.cones.empty() || static_cast<Index>(solution->col_value.size()) != n ||
      static_cast<Index>(solution->row_dual.size()) != m) {
    return false;
  }
  const double sigma = lin.sense_multiplier();
  const std::vector<double>& x = solution->col_value;
  const std::vector<double> activity = model.row_activity(x);
  const std::vector<Gradient> gradients = row_gradients(model, x);

  // r = sigma (c + H x), the objective's gradient in minimization space.
  std::vector<double> r(static_cast<std::size_t>(n), 0.0);
  if (lin.has_quadratic_objective()) qp::ipm_detail::hessian_times(lin.hessian, x, &r);
  for (std::size_t j = 0; j < r.size(); ++j) r[j] = sigma * (r[j] + lin.col_cost[j]);

  // ---- the active set at the point --------------------------------------------------------
  // Multipliers are scaled by the objective gradient's size, slacks by the constraint's.
  double multiplier_scale = 1.0;
  for (const double e : r) multiplier_scale = std::max(multiplier_scale, std::fabs(e));
  const auto active = [&](double value, double lower, double upper, double magnitude,
                          double multiplier) {
    const double scale = std::max(1.0, magnitude);
    const double below = is_finite_bound(lower) ? value - lower : kInfinity;
    const double above = is_finite_bound(upper) ? upper - value : kInfinity;
    if (std::min(below, above) <= tol::kConicActiveShare * scale) return true;
    // Tapia's indicator, on the side the multiplier prices (positive: the lower bound).
    const double slack = multiplier > 0.0 ? below : multiplier < 0.0 ? above : kInfinity;
    return std::isfinite(slack) && std::fabs(multiplier) / multiplier_scale > slack / scale;
  };
  std::vector<Index> active_rows;
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    double magnitude = std::fabs(activity[u]);
    for (const auto& [j, v] : gradients[u]) {
      magnitude = std::max(magnitude, std::fabs(v * x[static_cast<std::size_t>(j)]));
    }
    if (active(activity[u], lin.row_lower[u], lin.row_upper[u], magnitude,
               sigma * solution->row_dual[u])) {
      active_rows.push_back(i);
    }
  }
  const bool have_reduced_costs = static_cast<Index>(solution->col_dual.size()) == n;
  std::vector<Index> active_cols;
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (active(x[u], lin.col_lower[u], lin.col_upper[u], std::fabs(x[u]),
               have_reduced_costs ? sigma * solution->col_dual[u] : 0.0)) {
      active_cols.push_back(j);
    }
  }
  const auto k = static_cast<Index>(active_rows.size() + active_cols.size());

  // ---- (M'M + eps I) lambda = M'r, refined on the least-squares residual -------------------
  // M by model column: column j of the model meets active constraint t with coefficient v.
  std::vector<std::vector<std::pair<Index, double>>> by_column(static_cast<std::size_t>(n));
  for (std::size_t t = 0; t < active_rows.size(); ++t) {
    for (const auto& [j, v] : gradients[static_cast<std::size_t>(active_rows[t])]) {
      by_column[static_cast<std::size_t>(j)].emplace_back(static_cast<Index>(t), v);
    }
  }
  for (std::size_t t = 0; t < active_cols.size(); ++t) {
    by_column[static_cast<std::size_t>(active_cols[t])].emplace_back(
        static_cast<Index>(active_rows.size() + t), 1.0);
  }
  const auto m_times = [&](const std::vector<double>& lambda, std::vector<double>* out) {
    out->assign(static_cast<std::size_t>(n), 0.0);
    for (std::size_t j = 0; j < by_column.size(); ++j) {
      for (const auto& [t, v] : by_column[j])
        (*out)[j] += v * lambda[static_cast<std::size_t>(t)];
    }
  };
  const auto mt_times = [&](const std::vector<double>& v_n, std::vector<double>* out) {
    out->assign(static_cast<std::size_t>(k), 0.0);
    for (std::size_t j = 0; j < by_column.size(); ++j) {
      for (const auto& [t, v] : by_column[j]) (*out)[static_cast<std::size_t>(t)] += v * v_n[j];
    }
  };
  std::vector<double> lambda(static_cast<std::size_t>(k), 0.0);
  if (k > 0) {
    SparseMatrix normal(k, k);
    double largest = 1.0;
    for (const auto& column : by_column) {
      for (const auto& [t1, v1] : column) {
        for (const auto& [t2, v2] : column) {
          if (t2 > t1) continue;  // lower triangle: row t1 >= column t2
          normal.add_entry(t1, t2, v1 * v2);
          if (t1 == t2) largest = std::max(largest, v1 * v1);
        }
      }
    }
    const double eps = tol::kConicRecoveryRegularization * largest;
    for (Index t = 0; t < k; ++t) normal.add_entry(t, t, eps);
    normal.finalize(0.0);
    SparseLdl ldl;
    if (!ldl.analyze(normal) || !ldl.factorize(normal, eps)) return false;
    // Positive definite, so any ordering is stable (an augmented system with an eps block
    // is not: an ordering that pivots on -eps first swells the rest to 1/eps). The normal
    // equations square the condition number; refining on the residual r - M lambda takes
    // most of that back (Bjorck, "Numerical Methods for Least Squares Problems", SIAM 1996,
    // section 2.9, the corrected seminormal equations).
    std::vector<double> residual = r;
    std::vector<double> mlambda, step;
    for (int pass = 0; pass < 3; ++pass) {
      mt_times(residual, &step);
      ldl.solve(step.data());
      for (std::size_t t = 0; t < lambda.size(); ++t) lambda[t] += step[t];
      m_times(lambda, &mlambda);
      for (std::size_t j = 0; j < residual.size(); ++j) residual[j] = r[j] - mlambda[j];
    }
  }
  std::vector<double> y(static_cast<std::size_t>(m), 0.0);
  for (std::size_t t = 0; t < active_rows.size(); ++t) {
    if (!std::isfinite(lambda[t])) return false;
    y[static_cast<std::size_t>(active_rows[t])] = lambda[t];
  }

  // ---- keep whichever measures better -----------------------------------------------------
  std::vector<double> y_ipm(static_cast<std::size_t>(m));
  for (std::size_t i = 0; i < y_ipm.size(); ++i) y_ipm[i] = sigma * solution->row_dual[i];
  const double before = kkt_error(model, x, activity, r, gradients, y_ipm, nullptr);
  std::vector<double> d;
  const double after = kkt_error(model, x, activity, r, gradients, y, &d);
  if (!(after < before)) return false;
  for (std::size_t i = 0; i < y.size(); ++i) solution->row_dual[i] = sigma * y[i];
  solution->col_dual.assign(d.size(), 0.0);
  for (std::size_t j = 0; j < d.size(); ++j) solution->col_dual[j] = sigma * d[j];
  return true;
}

}  // namespace sankhya::conic
