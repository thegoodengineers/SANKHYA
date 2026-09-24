// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the interior point's model-space measurement against the status guard (#582).
//
// ipm/model_space.cpp claims to compute, from the scaled iterate alone, exactly what
// Solution::recompute_quality computes on the point once it is mapped back to the model.
// These tests hold it to that: on randomly scaled models with every bound shape, both
// senses and random points, the three numbers must agree to rounding. Then one constructed
// instance shows the disagreement #582 is about: a dual residual of 1e-9 in scaled units -
// 5e-10 relative, what the loop's own test reads, inside its 1e-8 - that the guard measures
// as 1e-4 in the model's units, because the column carrying it was scaled by 1e-5.

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "ipm/model_space.hpp"
#include "la/scaling.hpp"
#include "sankhya/model.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::ipm {
namespace {

/// A model, its hand-made scaling (Ahat = Dr A Dc and the scaled bounds and min-sense
/// cost), and a scaled iterate over total = n + m as the engine holds it.
struct Case {
  Model model;
  Scaling scaling;
  std::vector<double> cost, lower, upper, x, y, zl, zu;
};

void build_scaled(Case* c) {
  const Model& model = c->model;
  const Index n = model.num_cols();
  const Index m = model.num_rows();
  const double sense = model.sense_multiplier();
  c->scaling.matrix.reset(m, n);
  for (Index j = 0; j < n; ++j) {
    const ColumnView column = model.matrix.column(j);
    for (Index q = 0; q < column.size; ++q) {
      const Index i = column.rows[q];
      c->scaling.matrix.add_entry(i, j,
                                  column.values[q] *
                                      c->scaling.row[static_cast<std::size_t>(i)] *
                                      c->scaling.column[static_cast<std::size_t>(j)]);
    }
  }
  c->scaling.matrix.finalize(0.0);
  c->cost.assign(static_cast<std::size_t>(n), 0.0);
  c->lower.assign(static_cast<std::size_t>(n + m), 0.0);
  c->upper.assign(static_cast<std::size_t>(n + m), 0.0);
  const auto scale_bound = [](double b, double f) { return is_finite_bound(b) ? b * f : b; };
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double dc = c->scaling.column[u];
    c->cost[u] = sense * model.col_cost[u] * dc;
    c->lower[u] = scale_bound(model.col_lower[u], 1.0 / dc);
    c->upper[u] = scale_bound(model.col_upper[u], 1.0 / dc);
  }
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double dr = c->scaling.row[u];
    c->lower[static_cast<std::size_t>(n + i)] = scale_bound(model.row_lower[u], dr);
    c->upper[static_cast<std::size_t>(n + i)] = scale_bound(model.row_upper[u], dr);
  }
}

ScaledIterate view(const Case& c) {
  ScaledIterate it;
  it.matrix = &c.scaling.matrix;
  it.cost = &c.cost;
  it.lower = &c.lower;
  it.upper = &c.upper;
  it.x = &c.x;
  it.y = &c.y;
  it.zl = &c.zl;
  it.zu = &c.zu;
  it.scaling = &c.scaling;
  return it;
}

/// The point as solve_scaled() and InteriorPoint::finish() would hand it to the guard:
/// x = Dc xhat, row dual sense * Dr yhat, reduced cost sense * (zl - zu) / Dc, and for a
/// fixed column c - a^T y, which is what the engine reports there.
Solution mapped_back(const Case& c) {
  const Model& model = c.model;
  const Index n = model.num_cols();
  const Index m = model.num_rows();
  const double sense = model.sense_multiplier();
  Solution s;
  s.allocate_for(model);
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    s.row_dual[u] = sense * c.scaling.row[u] * c.y[u];
  }
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double dc = c.scaling.column[u];
    s.col_value[u] = c.x[u] * dc;
    if (model.col_lower[u] == model.col_upper[u]) {
      double d = model.col_cost[u];
      const ColumnView column = model.matrix.column(j);
      for (Index q = 0; q < column.size; ++q) {
        d -= column.values[q] * s.row_dual[static_cast<std::size_t>(column.rows[q])];
      }
      s.col_dual[u] = d;
    } else {
      s.col_dual[u] = sense * (c.zl[u] - c.zu[u]) / dc;
    }
  }
  s.recompute_quality(model);
  return s;
}

void expect_agrees(double ours, double guard, const char* what, int trial) {
  EXPECT_NEAR(ours, guard, 1e-12 + 1e-8 * std::fabs(guard)) << what << ", trial " << trial;
}

Case random_case(std::mt19937_64& rng, bool maximize) {
  std::uniform_real_distribution<double> entry(-3.0, 3.0);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::uniform_real_distribution<double> decades(-4.0, 4.0);
  std::uniform_int_distribution<int> shape(0, 4);
  const Index m = 6;
  const Index n = 9;
  Case c;
  Model& model = c.model;
  model.sense = maximize ? ObjSense::kMaximize : ObjSense::kMinimize;
  model.matrix.reset(m, n);
  model.col_cost.resize(static_cast<std::size_t>(n));
  model.col_lower.resize(static_cast<std::size_t>(n));
  model.col_upper.resize(static_cast<std::size_t>(n));
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kContinuous);
  model.row_lower.resize(static_cast<std::size_t>(m));
  model.row_upper.resize(static_cast<std::size_t>(m));
  for (Index j = 0; j < n; ++j) {
    for (Index i = 0; i < m; ++i) {
      if (unit(rng) < 0.5) model.matrix.add_entry(i, j, entry(rng));
    }
  }
  model.matrix.finalize(0.0);
  // Every bound shape: free, lower only, upper only, boxed, fixed.
  const auto bounds = [&](double* lo, double* hi) {
    const double a = 2.0 * entry(rng);
    switch (shape(rng)) {
      case 0:
        *lo = -kInfinity;
        *hi = kInfinity;
        break;
      case 1:
        *lo = a;
        *hi = kInfinity;
        break;
      case 2:
        *lo = -kInfinity;
        *hi = a;
        break;
      case 3:
        *lo = a;
        *hi = a + 1.0 + 3.0 * unit(rng);
        break;
      default:
        *lo = a;
        *hi = a;
        break;
    }
  };
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    model.col_cost[u] = 10.0 * entry(rng);
    bounds(&model.col_lower[u], &model.col_upper[u]);
  }
  for (Index i = 0; i < m; ++i) {
    bounds(&model.row_lower[static_cast<std::size_t>(i)],
           &model.row_upper[static_cast<std::size_t>(i)]);
  }
  c.scaling.row.resize(static_cast<std::size_t>(m));
  c.scaling.column.resize(static_cast<std::size_t>(n));
  for (double& f : c.scaling.row) f = std::pow(10.0, decades(rng));
  for (double& f : c.scaling.column) f = std::pow(10.0, decades(rng));
  build_scaled(&c);
  // A random scaled iterate: x anywhere near its box (so some bounds are violated), y of
  // either sign, and multipliers only where the engine keeps them - on a finite bound of an
  // unfixed variable.
  c.x.assign(static_cast<std::size_t>(n + m), 0.0);
  c.zl.assign(static_cast<std::size_t>(n + m), 0.0);
  c.zu.assign(static_cast<std::size_t>(n + m), 0.0);
  c.y.resize(static_cast<std::size_t>(m));
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    c.x[u] = 3.0 * entry(rng) / c.scaling.column[u];
    const bool fixed = model.col_lower[u] == model.col_upper[u];
    if (!fixed && is_finite_bound(c.lower[u])) c.zl[u] = unit(rng) * c.scaling.column[u];
    if (!fixed && is_finite_bound(c.upper[u])) c.zu[u] = unit(rng) * c.scaling.column[u];
  }
  for (Index i = 0; i < m; ++i) {
    c.y[static_cast<std::size_t>(i)] = entry(rng) / c.scaling.row[static_cast<std::size_t>(i)];
  }
  return c;
}

TEST(InteriorPointModelSpace, AgreesWithTheStatusGuardOnRandomlyScaledModels) {
  std::mt19937_64 rng(582);
  for (int trial = 0; trial < 200; ++trial) {
    const Case c = random_case(rng, trial % 2 == 1);
    const ModelSpaceMeasure ours = measure_in_model_space(view(c));
    const Solution guard = mapped_back(c);
    expect_agrees(ours.primal, guard.primal_infeasibility_scaled, "primal", trial);
    expect_agrees(ours.dual, guard.dual_infeasibility_scaled, "dual", trial);
    expect_agrees(ours.complementarity, guard.complementarity_violation, "complementarity",
                  trial);
  }
}

TEST(InteriorPointModelSpace, AnUnscaledModelIsMeasuredTheSameWithOrWithoutFactors) {
  std::mt19937_64 rng(1);
  Case c = random_case(rng, false);
  std::fill(c.scaling.row.begin(), c.scaling.row.end(), 1.0);
  std::fill(c.scaling.column.begin(), c.scaling.column.end(), 1.0);
  build_scaled(&c);
  ScaledIterate with = view(c);
  ScaledIterate without = with;
  without.scaling = nullptr;
  const ModelSpaceMeasure a = measure_in_model_space(with);
  const ModelSpaceMeasure b = measure_in_model_space(without);
  EXPECT_EQ(a.primal, b.primal);
  EXPECT_EQ(a.dual, b.dual);
  EXPECT_EQ(a.complementarity, b.complementarity);
}

// THE DISCREPANCY, CONSTRUCTED. min x1 + x2 s.t. x1 + x2 = 2, x1 - x2 = 0, x >= 0, whose
// optimum is x = (1, 1), y = (1, 0), reduced costs zero. Column 1 is scaled by 1e-5 and row
// 2 by 1e-4, as Ruiz does to a column of large entries and a row of large coefficients.
// The scaled iterate is the optimum with two perturbations of the size the loop accepts:
// z_l on column 1 carries 1e-9 that c - A^T y does not, and the point sits 1e-8 off row 2
// in scaled units. The loop's own measures - the largest residual over 1 + the norm of the
// cost or of x, in scaled units - read 5e-10 and 1e-13, both inside kIpmTolerance = 1e-8.
// In the model the dual residual is 1e-9 / 1e-5 = 1e-4 and the row misses by 1e-8 / 1e-4 =
// 1e-4, three decades above the guard's 1e-7: an optimal claimed on the scaled numbers
// would be downgraded (dual) or rejected (primal) by the guard.
TEST(InteriorPointModelSpace, AResidualInsideTheLoopsToleranceIsFourDecadesOutInTheModel) {
  Case c;
  Model& model = c.model;
  model.sense = ObjSense::kMinimize;
  model.matrix.reset(2, 2);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.add_entry(1, 0, 1.0);
  model.matrix.add_entry(1, 1, -1.0);
  model.matrix.finalize();
  model.col_cost = {1.0, 1.0};
  model.col_lower = {0.0, 0.0};
  model.col_upper = {kInfinity, kInfinity};
  model.col_type.assign(2, VarType::kContinuous);
  model.row_lower = {2.0, 0.0};
  model.row_upper = {2.0, 0.0};
  c.scaling.row = {1.0, 1e-4};
  c.scaling.column = {1e-5, 1.0};
  build_scaled(&c);
  // Model x = (1 - 5e-5, 1 + 5e-5): row 1 exact, row 2 off by 1e-4 (1e-8 scaled).
  c.x = {(1.0 - 5e-5) / 1e-5, 1.0 + 5e-5, 0.0, 0.0};
  c.y = {1.0, 0.0};
  c.zl = {1e-9, 0.0, 0.0, 0.0};
  c.zu = {0.0, 0.0, 0.0, 0.0};

  // What the loop reads: the scaled residuals over 1 + the scaled norms.
  const double chat_norm = std::max(std::fabs(c.cost[0]), std::fabs(c.cost[1]));
  const double scaled_dual = 1e-9 / (1.0 + chat_norm);
  const double xhat_norm = std::max(std::fabs(c.x[0]), std::fabs(c.x[1]));
  const double row2_scaled =
      std::fabs(c.scaling.matrix.at(1, 0) * c.x[0] + c.scaling.matrix.at(1, 1) * c.x[1]);
  const double scaled_primal = row2_scaled / (1.0 + xhat_norm);
  EXPECT_LT(scaled_dual, 1e-8);
  EXPECT_LT(scaled_primal, 1e-8);

  const ModelSpaceMeasure ours = measure_in_model_space(view(c));
  const Solution guard = mapped_back(c);
  EXPECT_GT(ours.dual, 1e2 * tol::kDualFeasibility);
  EXPECT_GT(ours.primal, 1e2 * tol::kPrimalFeasibility);
  EXPECT_NEAR(ours.dual, 1e-4, 1e-6);
  EXPECT_NEAR(ours.primal, 1e-4, 1e-6);
  expect_agrees(ours.primal, guard.primal_infeasibility_scaled, "primal", 0);
  expect_agrees(ours.dual, guard.dual_infeasibility_scaled, "dual", 0);
  // The worst entries point at the column and the row that carry it.
  EXPECT_EQ(std::max_element(ours.dual_violation.begin(), ours.dual_violation.end()) -
                ours.dual_violation.begin(),
            0);
  EXPECT_EQ(std::max_element(ours.row_violation.begin(), ours.row_violation.end()) -
                ours.row_violation.begin(),
            1);
}

}  // namespace
}  // namespace sankhya::ipm
