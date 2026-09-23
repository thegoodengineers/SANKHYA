// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the interior point's answer is judged in the model's units (#582).
//
// The engine iterates in Ruiz-scaled space and its own residual test lives there. On
// irish-electricity the scaled primal residual read 1.2e-8 while the status guard, which
// measures the point in the model's units, read 1.0e-4: four decades, the row factors of
// the worst rows. The engine now makes the guard's measurement itself and claims optimal
// only when both hold. These cases scale a small model's rows and columns by up to twelve
// decades and require that an `optimal` from the interior point is one the guard accepts,
// at the simplex's objective, and that the engine never reports the guard's own number
// above the tolerance under an optimal status.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya {
namespace {

Options with_algorithm(const char* algorithm) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", false);
  options.set_string("algorithm", algorithm);
  return options;
}

/// A dense-row LP, then every row i multiplied by row_factor[i] and every column j by
/// column_factor[j] (the column's bounds and cost divided, so the feasible set and the
/// optimal value are unchanged up to the change of variables).
Model scaled_lp(const std::vector<std::vector<double>>& rows,
                const std::vector<double>& row_lower, const std::vector<double>& row_upper,
                const std::vector<double>& cost, const std::vector<double>& col_upper,
                const std::vector<double>& row_factor,
                const std::vector<double>& column_factor) {
  Model model;
  const auto n = static_cast<Index>(cost.size());
  const auto m = static_cast<Index>(rows.size());
  model.sense = ObjSense::kMinimize;
  model.col_cost.resize(static_cast<std::size_t>(n));
  model.col_lower.assign(static_cast<std::size_t>(n), 0.0);
  model.col_upper.resize(static_cast<std::size_t>(n));
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kContinuous);
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    // x_model = x / column_factor: the cost and the bound follow the change of variables.
    model.col_cost[u] = cost[u] * column_factor[u];
    model.col_upper[u] = col_upper[u] / column_factor[u];
  }
  model.row_lower.resize(static_cast<std::size_t>(m));
  model.row_upper.resize(static_cast<std::size_t>(m));
  model.matrix.reset(m, n);
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    model.row_lower[u] =
        std::isfinite(row_lower[u]) ? row_lower[u] * row_factor[u] : row_lower[u];
    model.row_upper[u] =
        std::isfinite(row_upper[u]) ? row_upper[u] * row_factor[u] : row_upper[u];
  }
  for (Index j = 0; j < n; ++j) {
    for (Index i = 0; i < m; ++i) {
      const double a = rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
      if (a != 0.0) {
        model.matrix.add_entry(i, j,
                               a * row_factor[static_cast<std::size_t>(i)] *
                                   column_factor[static_cast<std::size_t>(j)]);
      }
    }
  }
  model.matrix.finalize();
  return model;
}

constexpr double kInf = kInfinity;

void expect_the_guard_accepts(const Model& model, const char* what) {
  const Solution simplex = solve(model, with_algorithm("dual-simplex"));
  ASSERT_EQ(simplex.status, SolveStatus::kOptimal) << what << ": " << simplex.message;
  const Solution ipm = solve(model, with_algorithm("ipm"));
  ASSERT_TRUE(ipm.status == SolveStatus::kOptimal || ipm.status == SolveStatus::kFeasible)
      << what << ": " << ipm.message;
  // An optimal from the interior point is one the guard measured and accepted: the
  // model-space relative primal infeasibility is inside the project tolerance.
  if (ipm.status == SolveStatus::kOptimal) {
    EXPECT_LE(ipm.primal_infeasibility_scaled, tol::kPrimalFeasibility)
        << what << ": " << ipm.message;
    EXPECT_NEAR(ipm.objective, simplex.objective,
                1e-6 * std::max(1.0, std::fabs(simplex.objective)))
        << what << ": " << ipm.message;
  }
}

TEST(InteriorPointScaling, AnUnscaledTextbookModelIsOptimal) {
  const Model model = scaled_lp({{1.0, 1.0, 1.0}, {1.0, 2.0, 3.0}, {2.0, 1.0, 0.0}},
                                {-kInf, -kInf, -kInf}, {10.0, 15.0, 8.0}, {-1.0, -2.0, -3.0},
                                {kInf, kInf, kInf}, {1.0, 1.0, 1.0}, {1.0, 1.0, 1.0});
  expect_the_guard_accepts(model, "unscaled");
}

TEST(InteriorPointScaling, RowsTwelveDecadesApartAreOptimalInTheModelsUnits) {
  const Model model = scaled_lp({{1.0, 1.0, 1.0}, {1.0, 2.0, 3.0}, {2.0, 1.0, 0.0}},
                                {-kInf, -kInf, -kInf}, {10.0, 15.0, 8.0}, {-1.0, -2.0, -3.0},
                                {kInf, kInf, kInf}, {1e-6, 1.0, 1e6}, {1.0, 1.0, 1.0});
  expect_the_guard_accepts(model, "rows 1e-6 .. 1e6");
}

TEST(InteriorPointScaling, RowsAndColumnsBothBadlyScaledAreOptimalInTheModelsUnits) {
  const Model model = scaled_lp(
      {{1.0, 1.0, 1.0, 1.0}, {1.0, 2.0, 3.0, 0.0}, {2.0, 1.0, 0.0, 4.0}, {0.0, 1.0, 1.0, 1.0}},
      {-kInf, -kInf, -kInf, 1.0}, {10.0, 15.0, 8.0, 6.0}, {-1.0, -2.0, -3.0, -1.5},
      {kInf, 5.0, kInf, 3.0}, {1e-5, 1.0, 1e4, 1e2}, {1e-3, 1.0, 1e3, 1e-2});
  expect_the_guard_accepts(model, "rows 1e-5 .. 1e4, columns 1e-3 .. 1e3");
}

TEST(InteriorPointScaling, AnEqualityRowAtAMillionIsOptimalInTheModelsUnits) {
  const Model model =
      scaled_lp({{1.0, 1.0, 1.0}, {1.0, -1.0, 0.0}}, {6.0, -kInf}, {6.0, 1.0}, {1.0, 2.0, 3.0},
                {kInf, kInf, kInf}, {1e6, 1e-6}, {1.0, 1e3, 1e-3});
  expect_the_guard_accepts(model, "equality row at 1e6 beside a row at 1e-6");
}

}  // namespace
}  // namespace sankhya
