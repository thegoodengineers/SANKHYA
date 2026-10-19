// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the active-set finish at the conic optimum (#755).
//
// WHY. An interior point on second-order cones stops within the gap of the optimum's VALUE,
// but its point and its multipliers only within about the square root of it: the
// second-order cone's error bound is Holderian with exponent 1/2, where a polyhedral one is
// linear (Sturm, "Error bounds for linear matrix inequalities", SIAM J. Optim. 10 (2000);
// Alizadeh & Goldfarb, "Second-order cone programming", Math. Programming 95 (2003), sec. 7).
// Measured: on the unit disc the row multiplier was off by 7.8e-6 at a gap of 1.2e-9; on the
// blending model of test_conic.cpp, whose optimum moves along a curved face, the point was off
// by about 1e-3 at a gap of 3e-9 - an objective error of y Q dx^2, consistent with the gap -
// and the reduced costs of its interior columns, the true stationarity error there, were near
// 1e-5. A model's KKT conditions are judged constraint by constraint
// (tools/verify_solution.py), so neither passes them, at any gap the arithmetic can reach.
//
// WHAT. The point is close, so the active set can be read off it: every equality row, every
// row and column bound within tol::kConicActiveShare of its slack scale, and every one whose
// interior-point multiplier outweighs its slack, both scaled (Tapia's indicator: El-Bakry,
// Tapia & Zhang, "A study of indicators for identifying zero variables in interior-point
// methods", SIAM Review 36 (1994) - at an interior-point iterate an active constraint has
// slack near mu / y and an inactive one a multiplier near mu / slack). On it:
//
//   1. Least-squares multipliers at the point, min || grad f - M lambda || with M's columns the
//      active rows' gradients a_i + 2 Q_i x and the active bounds' unit vectors (Nocedal &
//      Wright, "Numerical Optimization", 2nd ed. (2006), sec. 18.3), by the normal equations
//      (M'M + eps I) lambda = M'r refined on the residual (Bjorck, "Numerical Methods for
//      Least Squares Problems" (1996), sec. 2.9). Positive definite, so any ordering is stable.
//   2. From there, Newton's method on the KKT system of the active set taken as equalities -
//      grad f(x) - M(x) lambda = 0, g_A(x) = b_A - whose Jacobian
//
//          [ H_L   -M     ]       H_L = sigma H - sum_i lambda_i 2 Q_i, positive semidefinite
//          [ -M'   -delta ]       when the multipliers have their signs (the problem is convex)
//
//      is quasi-definite with the regularization and factored by the signed LDL^T, the
//      regularization refined back out as in the QP interior point. Quadratic convergence
//      from a point this close (Nocedal & Wright, sec. 18.1, Newton on the KKT system; the
//      same finish as the QP engine's #980).
//
// SAFE BY CONSTRUCTION. A result replaces the interior point's only when its measured model
// KKT error - every multiplier's sign against the bounds that exist and |multiplier| * slack -
// is smaller AND its primal violation is no worse. A misread active set costs the finish,
// never the answer, and the verifier judges whichever is kept.

#include "conic/conic_duals.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
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

/// sigma (c + H x), the objective's gradient in minimization space.
std::vector<double> objective_gradient(const Model& lin, const std::vector<double>& x) {
  std::vector<double> r(static_cast<std::size_t>(lin.num_cols()), 0.0);
  if (lin.has_quadratic_objective()) qp::ipm_detail::hessian_times(lin.hessian, x, &r);
  const double sigma = lin.sense_multiplier();
  for (std::size_t j = 0; j < r.size(); ++j) r[j] = sigma * (r[j] + lin.col_cost[j]);
  return r;
}

/// The largest bound violation of x and of its row activities, each relative to max(1, its
/// own magnitude).
double primal_violation(const Model& lin, const std::vector<double>& x,
                        const std::vector<double>& activity) {
  const auto over = [](double value, double lower, double upper) {
    return std::max({0.0, lower - value, value - upper}) / std::max(1.0, std::fabs(value));
  };
  double worst = 0.0;
  for (std::size_t j = 0; j < x.size(); ++j) {
    worst = std::max(worst, over(x[j], lin.col_lower[j], lin.col_upper[j]));
  }
  for (std::size_t i = 0; i < activity.size(); ++i) {
    worst = std::max(worst, over(activity[i], lin.row_lower[i], lin.row_upper[i]));
  }
  return worst;
}

/// The worst KKT violation of row multipliers `y` (minimization space): the reduced costs
/// d = r - sum_i y_i g_i follow from them, and every multiplier is held to its sign against
/// the bounds that exist and to |multiplier| * slack. Relative to max(1, |r|).
double kkt_error(const Model& lin, const std::vector<double>& x,
                 const std::vector<double>& activity, const std::vector<double>& r,
                 const std::vector<Gradient>& gradients, const std::vector<double>& y,
                 std::vector<double>* d_out) {
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

/// One active constraint: a row (column < 0) at `side`, or a column bound at `side`.
struct Constraint {
  Index row = -1;
  Index column = -1;
  double side = 0.0;
};

/// The active set at x, from the slacks and the interior point's multipliers (`y` on the
/// rows and `d` on the columns, minimization space).
std::vector<Constraint> active_set(const Model& lin, const std::vector<double>& x,
                                   const std::vector<double>& activity,
                                   const std::vector<Gradient>& gradients,
                                   const std::vector<double>& r, const std::vector<double>& y,
                                   const std::vector<double>& d) {
  double multiplier_scale = 1.0;
  for (const double e : r) multiplier_scale = std::max(multiplier_scale, std::fabs(e));
  // The side a constraint is active at, or NaN when it is not active.
  const auto side_of = [&](double value, double lower, double upper, double magnitude,
                           double multiplier) {
    const double inactive = std::numeric_limits<double>::quiet_NaN();
    if (is_finite_bound(lower) && lower == upper) return lower;
    const double scale = std::max(1.0, magnitude);
    const double below = is_finite_bound(lower) ? value - lower : kInfinity;
    const double above = is_finite_bound(upper) ? upper - value : kInfinity;
    // Tapia's indicator, on the side the multiplier prices (positive: the lower bound).
    const double weight = std::fabs(multiplier) / multiplier_scale;
    if (multiplier > 0.0 && std::isfinite(below) && weight > below / scale) return lower;
    if (multiplier < 0.0 && std::isfinite(above) && weight > above / scale) return upper;
    if (std::min(below, above) <= tol::kConicActiveShare * scale) {
      return below <= above ? lower : upper;
    }
    return inactive;
  };
  std::vector<Constraint> out;
  for (Index i = 0; i < lin.num_rows(); ++i) {
    const auto u = static_cast<std::size_t>(i);
    double magnitude = std::fabs(activity[u]);
    for (const auto& [j, v] : gradients[u]) {
      magnitude = std::max(magnitude, std::fabs(v * x[static_cast<std::size_t>(j)]));
    }
    const double side =
        side_of(activity[u], lin.row_lower[u], lin.row_upper[u], magnitude, y[u]);
    if (!std::isnan(side)) out.push_back({i, -1, side});
  }
  for (Index j = 0; j < lin.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double side = side_of(x[u], lin.col_lower[u], lin.col_upper[u], std::fabs(x[u]),
                                d.empty() ? 0.0 : d[u]);
    if (!std::isnan(side)) out.push_back({-1, j, side});
  }
  return out;
}

/// M by model column: column j meets active constraint t with coefficient v.
std::vector<Gradient> columns_of(Index n, const std::vector<Constraint>& active,
                                 const std::vector<Gradient>& gradients) {
  std::vector<Gradient> by_column(static_cast<std::size_t>(n));
  for (std::size_t t = 0; t < active.size(); ++t) {
    if (active[t].row >= 0) {
      for (const auto& [j, v] : gradients[static_cast<std::size_t>(active[t].row)]) {
        by_column[static_cast<std::size_t>(j)].emplace_back(static_cast<Index>(t), v);
      }
    } else {
      by_column[static_cast<std::size_t>(active[t].column)].emplace_back(static_cast<Index>(t),
                                                                         1.0);
    }
  }
  return by_column;
}

/// min || r - M lambda || by the refined normal equations; false when they cannot be factored.
bool least_squares(const std::vector<Gradient>& by_column, Index k,
                   const std::vector<double>& r, std::vector<double>* lambda) {
  lambda->assign(static_cast<std::size_t>(k), 0.0);
  if (k == 0) return true;
  SparseMatrix normal(k, k);
  double largest = 1.0;
  for (const Gradient& column : by_column) {
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
  std::vector<double> residual = r;
  std::vector<double> step(static_cast<std::size_t>(k));
  for (int pass = 0; pass < tol::kConicRecoveryRefinement; ++pass) {
    std::fill(step.begin(), step.end(), 0.0);
    for (std::size_t j = 0; j < by_column.size(); ++j) {
      for (const auto& [t, v] : by_column[j])
        step[static_cast<std::size_t>(t)] += v * residual[j];
    }
    ldl.solve(step.data());
    for (std::size_t t = 0; t < step.size(); ++t) (*lambda)[t] += step[t];
    residual = r;
    for (std::size_t j = 0; j < by_column.size(); ++j) {
      for (const auto& [t, v] : by_column[j]) {
        residual[j] -= v * (*lambda)[static_cast<std::size_t>(t)];
      }
    }
  }
  return std::all_of(lambda->begin(), lambda->end(), [](double e) { return std::isfinite(e); });
}

/// Newton's method on the KKT system of `active` taken as equalities, from (x, lambda).
/// False when a step cannot be computed; x and lambda are then left where the last step was.
bool newton(const QcqpModel& model, const std::vector<Constraint>& active,
            std::vector<double>* x, std::vector<double>* lambda) {
  const Model& lin = model.linear;
  const Index n = lin.num_cols();
  const auto k = static_cast<Index>(active.size());
  const Index dim = n + k;
  const double sigma = lin.sense_multiplier();
  std::vector<signed char> signs(static_cast<std::size_t>(dim), 1);
  for (Index t = 0; t < k; ++t) signs[static_cast<std::size_t>(n + t)] = -1;
  double previous = kInfinity;
  for (int iteration = 0; iteration < tol::kConicPolishIterations; ++iteration) {
    const std::vector<double> activity = model.row_activity(*x);
    const std::vector<Gradient> gradients = row_gradients(model, *x);
    const std::vector<Gradient> by_column = columns_of(n, active, gradients);
    std::vector<double> rhs = objective_gradient(lin, *x);  // becomes -F1 below
    for (std::size_t j = 0; j < by_column.size(); ++j) {
      for (const auto& [t, v] : by_column[j])
        rhs[j] -= v * (*lambda)[static_cast<std::size_t>(t)];
    }
    double residual = 0.0;
    for (std::size_t j = 0; j < by_column.size(); ++j) {
      residual = std::max(residual, std::fabs(rhs[j]));
      rhs[j] = -rhs[j];
    }
    rhs.resize(static_cast<std::size_t>(dim));
    for (Index t = 0; t < k; ++t) {
      const Constraint& c = active[static_cast<std::size_t>(t)];
      const double value = c.row >= 0 ? activity[static_cast<std::size_t>(c.row)]
                                      : (*x)[static_cast<std::size_t>(c.column)];
      rhs[static_cast<std::size_t>(n + t)] = value - c.side;  // F2
      residual =
          std::max(residual, std::fabs(value - c.side) / std::max(1.0, std::fabs(c.side)));
    }
    if (residual <= tol::kConicPolishResidual || !(residual < previous)) break;
    previous = residual;

    // [ H_L + rho I   -M ; -M'   -delta I ], lower triangle; K0 the same with rho = delta = 0.
    double largest = 1.0;
    const auto assemble = [&](double regularization) {
      SparseMatrix k_matrix(dim, dim);
      const bool hessian = lin.has_quadratic_objective();
      for (Index j = 0; j < n; ++j) {
        if (hessian) {
          const ColumnView h = lin.hessian.column(j);
          for (Index p = 0; p < h.size; ++p) {
            k_matrix.add_entry(h.rows[p], j, sigma * h.values[p]);
          }
        }
        k_matrix.add_entry(j, j, regularization);
      }
      for (Index t = 0; t < k; ++t) {
        const Constraint& c = active[static_cast<std::size_t>(t)];
        if (c.row < 0) continue;
        const double l = (*lambda)[static_cast<std::size_t>(t)];
        for (const QuadraticTerm& q : model.quadratic) {
          if (q.row != c.row) continue;
          // -lambda * Hessian of v x_a x_b: v off the diagonal (lower), 2 v on it.
          k_matrix.add_entry(q.second, q.first,
                             (q.first == q.second ? -2.0 : -1.0) * l * q.value);
        }
      }
      for (std::size_t j = 0; j < by_column.size(); ++j) {
        for (const auto& [t, v] : by_column[j]) {
          k_matrix.add_entry(n + t, static_cast<Index>(j), -v);
          largest = std::max(largest, std::fabs(v));
        }
      }
      for (Index t = 0; t < k; ++t) k_matrix.add_entry(n + t, n + t, -regularization);
      k_matrix.finalize(0.0);
      return k_matrix;
    };
    const SparseMatrix k0 = assemble(0.0);
    const double rho = tol::kConicPolishRegularization * largest;
    const SparseMatrix k_reg = assemble(rho);
    SparseLdl ldl;
    if (!ldl.analyze(k_reg) || !ldl.factorize_quasidefinite(k_reg, signs, rho)) return false;
    std::vector<double> step;
    qp::ipm_detail::solve_refined(ldl, k_reg, rhs, &step);
    qp::ipm_detail::refine_unregularized(ldl, k0, rhs, &step);
    if (!std::all_of(step.begin(), step.end(), [](double e) { return std::isfinite(e); })) {
      return false;
    }
    for (Index j = 0; j < n; ++j)
      (*x)[static_cast<std::size_t>(j)] += step[static_cast<std::size_t>(j)];
    for (Index t = 0; t < k; ++t) {
      (*lambda)[static_cast<std::size_t>(t)] += step[static_cast<std::size_t>(n + t)];
    }
  }
  return true;
}

}  // namespace

Finish finish_on_active_set(const QcqpModel& model, Solution* solution) {
  const Model& lin = model.linear;
  const Index n = lin.num_cols();
  const Index m = lin.num_rows();
  if (!model.cones.empty() || static_cast<Index>(solution->col_value.size()) != n ||
      static_cast<Index>(solution->row_dual.size()) != m) {
    return Finish::kNone;
  }
  const double sigma = lin.sense_multiplier();
  const std::vector<double> x0 = solution->col_value;
  const std::vector<double> activity0 = model.row_activity(x0);
  const std::vector<Gradient> gradients0 = row_gradients(model, x0);
  const std::vector<double> r0 = objective_gradient(lin, x0);
  std::vector<double> y0(static_cast<std::size_t>(m));
  for (std::size_t i = 0; i < y0.size(); ++i) y0[i] = sigma * solution->row_dual[i];
  std::vector<double> d0;
  if (static_cast<Index>(solution->col_dual.size()) == n) {
    d0.resize(static_cast<std::size_t>(n));
    for (std::size_t j = 0; j < d0.size(); ++j) d0[j] = sigma * solution->col_dual[j];
  }
  const double primal0 = primal_violation(lin, x0, activity0);
  const double kkt0 = kkt_error(lin, x0, activity0, r0, gradients0, y0, nullptr);

  const std::vector<Constraint> active = active_set(lin, x0, activity0, gradients0, r0, y0, d0);
  const auto k = static_cast<Index>(active.size());
  std::vector<double> lambda;
  if (!least_squares(columns_of(n, active, gradients0), k, r0, &lambda)) return Finish::kNone;

  // A candidate (x, lambda) measured on the model; kept when it beats the best so far.
  struct Candidate {
    std::vector<double> x, activity, y, d;
    double kkt = kInfinity;
    double primal = kInfinity;
  };
  const auto measure = [&](const std::vector<double>& x, const std::vector<double>& l) {
    Candidate c;
    c.x = x;
    c.activity = model.row_activity(x);
    c.y.assign(static_cast<std::size_t>(m), 0.0);
    for (std::size_t t = 0; t < active.size(); ++t) {
      if (active[t].row >= 0) c.y[static_cast<std::size_t>(active[t].row)] = l[t];
    }
    c.primal = primal_violation(lin, x, c.activity);
    c.kkt = kkt_error(lin, x, c.activity, objective_gradient(lin, x), row_gradients(model, x),
                      c.y, &c.d);
    return c;
  };
  const double primal_allowed = std::max(primal0, tol::kPrimalFeasibility);
  const auto better = [&](const Candidate& c, double kkt_to_beat) {
    return c.primal <= primal_allowed && c.kkt < kkt_to_beat;
  };

  Finish result = Finish::kNone;
  Candidate best;
  best.kkt = kkt0;
  const Candidate duals_only = measure(x0, lambda);
  if (better(duals_only, best.kkt)) {
    best = duals_only;
    result = Finish::kDuals;
  }
  std::vector<double> x = x0;
  std::vector<double> l = lambda;
  if (newton(model, active, &x, &l)) {
    const Candidate polished = measure(x, l);
    if (better(polished, best.kkt)) {
      best = polished;
      result = Finish::kPolished;
    }
  }
  if (result == Finish::kNone) return result;

  solution->col_value = best.x;
  solution->row_activity = best.activity;
  for (std::size_t i = 0; i < best.y.size(); ++i) solution->row_dual[i] = sigma * best.y[i];
  solution->col_dual.assign(best.d.size(), 0.0);
  for (std::size_t j = 0; j < best.d.size(); ++j) solution->col_dual[j] = sigma * best.d[j];
  solution->objective = model.objective(best.x);
  solution->dual_bound = solution->objective;
  return result;
}

}  // namespace sankhya::conic
