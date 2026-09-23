// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the in-process KKT check (#476). See kkt_check.hpp for what it is and why it is a
// reimplementation of tools/verify_solution.py's checks rather than a call into it.

#include "core/kkt_check.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "sankhya/types.hpp"

namespace sankhya {
namespace {

constexpr double kUnbounded = std::numeric_limits<double>::infinity();

KktVerdict fail(const char* check, std::string detail) {
  KktVerdict verdict;
  verdict.passed = false;
  verdict.check = check;
  verdict.detail = std::move(detail);
  return verdict;
}

/// A multiplier pushing against a bound that does not exist, in minimize space: a positive
/// one prices the lower bound, a negative one the upper. Any multiplier is admissible on an
/// equality or a fixed variable. (verify_solution.py, `sign_violation`.)
double sign_violation(double multiplier, double lower, double upper) {
  if (lower == upper) return 0.0;
  if (multiplier > 0.0 && !is_finite_bound(lower)) return multiplier;
  if (multiplier < 0.0 && !is_finite_bound(upper)) return -multiplier;
  return 0.0;
}

/// |multiplier| * slack, where a slack that cannot close (no finite bound on either side)
/// makes the condition "the multiplier is zero". (verify_solution.py, `complementarity`.)
double complementarity(double multiplier, double slack) {
  if (!std::isfinite(slack)) return std::fabs(multiplier);
  return std::fabs(multiplier) * slack;
}

/// The Lagrangian term a bound contributes to the dual objective, in minimize space.
/// (verify_solution_checks.py, `bound_contribution`.)
double bound_contribution(double multiplier, double lower, double upper) {
  if (multiplier > 0.0 && is_finite_bound(lower)) return multiplier * lower;
  if (multiplier < 0.0 && is_finite_bound(upper)) return multiplier * upper;
  return 0.0;
}

/// The part of the duality gap one item accounts for, given the per-item checks already
/// accepted it: all of it when the multiplier is within the dual tolerance at its own scale,
/// otherwise at most |multiplier| * nearest slack. (verify_solution.py, `share`.)
double share(double multiplier, double value, double lower, double upper, double scale,
             double dual_tolerance) {
  if (multiplier == 0.0) return 0.0;
  const double priced = multiplier > 0.0 ? lower : upper;
  const double term = std::fabs(multiplier) *
                      (is_finite_bound(priced) ? std::fabs(value - priced) : std::fabs(value));
  if (std::fabs(multiplier) <= dual_tolerance * scale) return term;
  const double nearest =
      std::min(is_finite_bound(lower) ? std::fabs(value - lower) : kUnbounded,
               is_finite_bound(upper) ? std::fabs(value - upper) : kUnbounded);
  return std::isfinite(nearest) ? std::min(term, std::fabs(multiplier) * nearest) : 0.0;
}

double nearer_slack(double value, double lower, double upper) {
  const double below = is_finite_bound(lower) ? value - lower : kUnbounded;
  const double above = is_finite_bound(upper) ? upper - value : kUnbounded;
  return std::min(below, above);
}

}  // namespace

KktVerdict check_optimality(const Model& model, const Solution& s,
                            const KktTolerances& tolerances, bool allow_hessian);

KktVerdict check_lp_optimality(const Model& model, const Solution& s,
                               const KktTolerances& tolerances) {
  return check_optimality(model, s, tolerances, /*allow_hessian=*/false);
}

KktVerdict check_qp_optimality(const Model& model, const Solution& s,
                               const KktTolerances& tolerances) {
  return check_optimality(model, s, tolerances, /*allow_hessian=*/true);
}

KktVerdict check_optimality(const Model& model, const Solution& s,
                            const KktTolerances& tolerances, bool allow_hessian) {
  const Index n = model.num_cols();
  const Index m = model.num_rows();
  const auto un = static_cast<std::size_t>(n);
  const auto um = static_cast<std::size_t>(m);
  const bool quadratic = model.hessian.num_nonzeros() > 0;
  if (model.num_integer_columns() > 0 || (quadratic && !allow_hessian)) {
    return fail("class", allow_hessian ? "integer columns: the check covers LP and convex QP"
                                       : "not an LP: the check covers LP optimality only");
  }
  if (s.status != SolveStatus::kOptimal) {
    return fail("verdict",
                fmt::format("status is {}, not an optimality claim", to_string(s.status)));
  }

  // ---- Structure: a value for every column and row, and every value a number ----------
  if (s.col_value.size() != un || s.row_activity.size() != um || s.col_dual.size() != un ||
      s.row_dual.size() != um) {
    return fail("structure", "the point, the activities or the duals are not the model's size");
  }
  const auto finite = [](const std::vector<double>& v) {
    return std::all_of(v.begin(), v.end(), [](double x) { return std::isfinite(x); });
  };
  if (!finite(s.col_value) || !finite(s.row_activity) || !finite(s.col_dual) ||
      !finite(s.row_dual) || !std::isfinite(s.objective)) {
    return fail("structure", "a value in the answer is not a finite number");
  }
  const std::vector<double>& x = s.col_value;

  // ---- Column bounds, relative to each value's own magnitude ---------------------------
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double violation =
        std::max({model.col_lower[u] - x[u], x[u] - model.col_upper[u], 0.0});
    const double scaled = violation / std::max(1.0, std::fabs(x[u]));
    if (scaled > tolerances.primal) {
      return fail(
          "column bounds",
          fmt::format("violation {:.3e} ({:.3e} relative) on column {}", violation, scaled, j));
    }
  }

  // ---- Row activity, recomputed, relative to the largest term of each row --------------
  std::vector<double> activity(um, 0.0);
  std::vector<double> row_scale(um, 1.0);
  for (Index j = 0; j < n; ++j) {
    const double xj = x[static_cast<std::size_t>(j)];
    if (xj == 0.0) continue;
    const ColumnView column = model.matrix.column(j);
    for (Index p = 0; p < column.size; ++p) {
      const auto i = static_cast<std::size_t>(column.rows[p]);
      const double term = column.values[p] * xj;
      activity[i] += term;
      row_scale[i] = std::max(row_scale[i], std::fabs(term));
    }
  }
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double violation =
        std::max({model.row_lower[u] - activity[u], activity[u] - model.row_upper[u], 0.0});
    if (violation / row_scale[u] > tolerances.primal) {
      return fail("row activity",
                  fmt::format("violation {:.3e} ({:.3e} relative to the row's terms) on row {}",
                              violation, violation / row_scale[u], i));
    }
    const double disagreement = std::fabs(activity[u] - s.row_activity[u]);
    if (disagreement > tol::kVerifierConsistency) {
      return fail("activity agreement",
                  fmt::format("|recomputed - reported| = {:.3e} on row {}", disagreement, i));
    }
  }

  // ---- Objective, recomputed ---------------------------------------------------------
  double objective = model.objective_offset;
  for (Index j = 0; j < n; ++j) {
    objective += model.col_cost[static_cast<std::size_t>(j)] * x[static_cast<std::size_t>(j)];
  }
  // Q x from the stored lower triangle (each off-diagonal entry stands for two of the
  // symmetric matrix), and x'Qx/2 into the objective, the convention of Model::hessian.
  std::vector<double> qx(un, 0.0);
  if (quadratic) {
    for (Index j = 0; j < n; ++j) {
      const ColumnView c = model.hessian.column(j);
      for (Index k = 0; k < c.size; ++k) {
        const auto i = static_cast<std::size_t>(c.rows[k]);
        const auto uj = static_cast<std::size_t>(j);
        qx[i] += c.values[k] * x[uj];
        if (i != uj) qx[uj] += c.values[k] * x[i];
      }
    }
    double half_xqx = 0.0;
    for (Index j = 0; j < n; ++j) {
      half_xqx += 0.5 * qx[static_cast<std::size_t>(j)] * x[static_cast<std::size_t>(j)];
    }
    objective += half_xqx;
  }
  const double objective_scale = std::max(1.0, std::fabs(objective));
  if (std::fabs(objective - s.objective) > tol::kVerifierObjective * objective_scale) {
    return fail("objective",
                fmt::format("recomputed {:.12e}, reported {:.12e}", objective, s.objective));
  }

  // ---- Duals, in minimize space ------------------------------------------------------
  const double sigma = model.sense_multiplier();
  std::vector<double> y(um);
  for (Index i = 0; i < m; ++i) {
    y[static_cast<std::size_t>(i)] = sigma * s.row_dual[static_cast<std::size_t>(i)];
  }
  std::vector<double> column_scale(un, 1.0);
  std::vector<double> derived_dual(un, 0.0);
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    // For a QP the gradient of the objective is c + Q x, not c; every condition below is
    // stated in terms of the gradient, so this one substitution carries them over.
    const double cost = sigma * (model.col_cost[u] + qx[u]);
    const ColumnView column = model.matrix.column(j);
    double expected = cost;
    double scale = std::max(1.0, std::fabs(cost));
    for (Index p = 0; p < column.size; ++p) {
      const double term = column.values[p] * y[static_cast<std::size_t>(column.rows[p])];
      expected -= term;
      scale = std::max(scale, std::fabs(term));
    }
    column_scale[u] = scale;
    // The QP engine carries no basis and reports no reduced costs, so for a QP d is DERIVED
    // from (model, x, y), the stronger test: the sign and complementarity checks then price
    // against a vector the solver never chose (the verifier does the same).
    const double d = quadratic ? expected : sigma * s.col_dual[u];
    if (quadratic) derived_dual[u] = d;
    const double difference = std::fabs(expected - d);
    if (difference / scale > tol::kVerifierConsistency) {
      return fail("reduced costs",
                  fmt::format("|c - A^T y - d| = {:.3e} ({:.3e} relative to its terms) on "
                              "column {}",
                              difference, difference / scale, j));
    }
    const double violation = sign_violation(d, model.col_lower[u], model.col_upper[u]);
    if (violation / scale > tolerances.dual) {
      return fail("dual feasibility (columns)",
                  fmt::format("{:.3e} ({:.3e} relative) on column {}", violation,
                              violation / scale, j));
    }
  }
  double dual_norm = 1.0;
  for (const double v : y) dual_norm = std::max(dual_norm, std::fabs(v));
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double violation = sign_violation(y[u], model.row_lower[u], model.row_upper[u]);
    if (violation / dual_norm > tolerances.dual) {
      return fail("dual feasibility (rows)",
                  fmt::format("{:.3e} ({:.3e} relative to |y|) on row {}", violation,
                              violation / dual_norm, i));
    }
  }

  // ---- Complementary slackness, absolutely, as the verifier judges it -----------------
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    if (model.row_lower[u] == model.row_upper[u]) continue;
    const double product = complementarity(
        y[u], nearer_slack(activity[u], model.row_lower[u], model.row_upper[u]));
    if (product > tol::kComplementarity) {
      return fail("complementary slackness",
                  fmt::format("|multiplier| * slack = {:.3e} on row {}", product, i));
    }
  }
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (model.col_lower[u] == model.col_upper[u]) continue;
    // For a QP the multiplier priced here is the derived one, as in the sign check above.
    const double column_dual = quadratic ? derived_dual[u] : sigma * s.col_dual[u];
    const double product = complementarity(
        column_dual, nearer_slack(x[u], model.col_lower[u], model.col_upper[u]));
    if (product > tol::kComplementarity) {
      return fail("complementary slackness",
                  fmt::format("|multiplier| * slack = {:.3e} on column {}", product, j));
    }
  }

  // ---- The basis, when every status is known ------------------------------------------
  if (!quadratic) {
    const auto known = [](const std::vector<BasisStatus>& v) {
      return !v.empty() && std::none_of(v.begin(), v.end(), [](BasisStatus b) {
        return b == BasisStatus::kUnknown;
      });
    };
    if (s.col_status.size() == un && s.row_status.size() == um && known(s.col_status) &&
        (m == 0 || known(s.row_status))) {
      Index basic = 0;
      const auto off_bound = [&](BasisStatus status, double value, double lower, double upper) {
        const double scale = std::max(1.0, std::fabs(value));
        if (status == BasisStatus::kAtLower && is_finite_bound(lower)) {
          return std::fabs(value - lower) / scale;
        }
        if (status == BasisStatus::kAtUpper && is_finite_bound(upper)) {
          return std::fabs(value - upper) / scale;
        }
        return 0.0;
      };
      double worst = 0.0;
      for (Index j = 0; j < n; ++j) {
        const auto u = static_cast<std::size_t>(j);
        if (s.col_status[u] == BasisStatus::kBasic) ++basic;
        worst = std::max(
            worst, off_bound(s.col_status[u], x[u], model.col_lower[u], model.col_upper[u]));
      }
      for (Index i = 0; i < m; ++i) {
        const auto u = static_cast<std::size_t>(i);
        if (s.row_status[u] == BasisStatus::kBasic) ++basic;
        worst = std::max(worst, off_bound(s.row_status[u], activity[u], model.row_lower[u],
                                          model.row_upper[u]));
      }
      if (basic != m) {
        return fail("basis", fmt::format("{} basic entries for {} rows", basic, m));
      }
      if (worst > tolerances.primal) {
        return fail("nonbasic entries on their bounds",
                    fmt::format("worst distance {:.3e}", worst));
      }
    }
  }

  // ---- Strong duality, with the verifier's per-item accounting ------------------------
  double dual_objective = 0.0;
  double accounted = 0.0;
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    dual_objective += bound_contribution(y[u], model.row_lower[u], model.row_upper[u]);
    accounted += share(y[u], activity[u], model.row_lower[u], model.row_upper[u], dual_norm,
                       tolerances.dual);
  }
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double d = quadratic ? derived_dual[u] : sigma * s.col_dual[u];
    dual_objective += bound_contribution(d, model.col_lower[u], model.col_upper[u]);
    accounted += share(d, x[u], model.col_lower[u], model.col_upper[u], column_scale[u],
                       tolerances.dual);
  }
  dual_objective = sigma * dual_objective + model.objective_offset;
  if (quadratic) {
    // At a KKT point the bound contributions sum to (c + Qx)'x = c'x + x'Qx, which overshoots
    // the primal objective c'x + x'Qx/2 by exactly x'Qx/2: subtracting it is the Dorn dual
    // of a convex QP, and it makes the gap a real optimality test rather than an identity
    // that would fail by a fixed amount on every quadratic instance.
    double half_xqx = 0.0;
    for (Index j = 0; j < n; ++j) {
      half_xqx += 0.5 * qx[static_cast<std::size_t>(j)] * x[static_cast<std::size_t>(j)];
    }
    dual_objective -= half_xqx;
  }
  const double gap = std::fabs(objective - dual_objective);
  if (gap > tolerances.duality_gap * objective_scale + accounted) {
    return fail("strong duality",
                fmt::format("primal {:.12e}, dual {:.12e}, gap {:.3e}, {:.3e} of it accounted "
                            "for by per-item violations",
                            objective, dual_objective, gap, accounted));
  }

  KktVerdict verdict;
  verdict.passed = true;
  return verdict;
}

}  // namespace sankhya
