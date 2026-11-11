// SPDX-License-Identifier: Apache-2.0
// SANKHYA - presolve keeps every coefficient the model states (#792).
//
// A row and a column scaled by 2^-20 put a 2^-40 = 9.1e-13 coefficient into the matrix. It
// is data, and an exact change of variables away from 1.0. Presolve used to finalize its
// reduced model with the default absolute drop of 1e-11 and solved a different model (Netlib
// adlittle under the stress set of #762 lost five coefficients that way). These tests hold
// the reduced model to every coefficient, and the solve to the unscaled optimum.

#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "presolve/presolve.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kScale = 0x1p-20;  // exact: only the exponent changes

/// min -x - y  s.t.  x + 2y <= 4,  3x + y <= 6,  x, y >= 0  (optimum -2.8 at (1.6, 1.2)),
/// with x replaced by kScale * x' and the first row multiplied by kScale.
Model scaled_lp() {
  Model model;
  model.col_cost = {-kScale, -1.0};
  model.col_lower = {0.0, 0.0};
  model.col_upper = {kInf, kInf};
  model.col_type.assign(2, VarType::kContinuous);
  model.row_lower = {-kInf, -kInf};
  model.row_upper = {4.0 * kScale, 6.0};
  model.matrix.reset(2, 2);
  model.matrix.add_entry(0, 0, kScale * kScale);  // 9.1e-13
  model.matrix.add_entry(0, 1, 2.0 * kScale);
  model.matrix.add_entry(1, 0, 3.0 * kScale);
  model.matrix.add_entry(1, 1, 1.0);
  model.matrix.finalize(0.0);
  model.hessian.reset(2, 2);
  model.hessian.finalize();
  return model;
}

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", true);
  return options;
}

TEST(PresolveScaled, TheReducedModelKeepsACoefficientBelowTheOldAbsoluteDrop) {
  const Model model = scaled_lp();
  ASSERT_EQ(model.matrix.num_nonzeros(), 4);
  Logger logger(nullptr);
  const presolve::Result reduced = presolve::presolve(model, quiet(), logger);
  EXPECT_EQ(reduced.model.matrix.num_nonzeros(), 4) << "a 9.1e-13 coefficient is data";
}

TEST(PresolveScaled, TheScaledModelSolvesToTheUnscaledOptimum) {
  const Solution solution = solve(scaled_lp(), quiet());
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NEAR(solution.objective, -2.8, 1e-9);
  EXPECT_NEAR(solution.col_value[0] * kScale, 1.6, 1e-9);
  EXPECT_NEAR(solution.col_value[1], 1.2, 1e-9);
}

/// min sum c_j x_j over 0 <= x <= upper subject to rows[i] x >= rhs[i], every row multiplied
/// by scale[i].
Model box_lp(const std::vector<double>& cost, const std::vector<double>& upper,
             const std::vector<std::vector<double>>& rows, const std::vector<double>& rhs,
             const std::vector<double>& scale) {
  Model model;
  const auto n = static_cast<Index>(cost.size());
  const auto m = static_cast<Index>(rows.size());
  model.col_cost = cost;
  model.col_lower.assign(cost.size(), 0.0);
  model.col_upper = upper;
  model.col_type.assign(cost.size(), VarType::kContinuous);
  model.row_lower.resize(rows.size());
  model.row_upper.assign(rows.size(), kInf);
  model.matrix.reset(m, n);
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    model.row_lower[u] = rhs[u] * scale[u];
    for (Index j = 0; j < n; ++j) {
      const double a = rows[u][static_cast<std::size_t>(j)];
      if (a != 0.0) model.matrix.add_entry(i, j, a * scale[u]);
    }
  }
  model.matrix.finalize(0.0);
  model.hessian.reset(n, n);
  model.hessian.finalize();
  return model;
}

TEST(PresolveScaled, ARowBelowTheFeasibilityToleranceIsNotRedundant) {
  // 1e-8 x0 + 1e-8 x1 >= 2e-8 with x in [0, 1] is x0 + x1 >= 2: min x0 + x1 is 2. At x = 0
  // the row is short by 2e-8, inside the absolute 1e-7 the redundant-row test used, so the
  // row was dropped and the answer came back 0, `optimal`.
  const Model model = box_lp({1.0, 1.0}, {1.0, 1.0}, {{1.0, 1.0}}, {2.0}, {1e-8});
  Logger logger(nullptr);
  const presolve::Result reduced = presolve::presolve(model, quiet(), logger);
  EXPECT_EQ(reduced.model.num_rows(), 1) << "the row binds";
  const Solution solution = solve(model, quiet());
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NEAR(solution.objective, 2.0, 1e-9);
}

TEST(PresolveScaled, RowScaledInstancesKeepTheirOptimum) {
  // Random integer LPs over a box, feasible by construction, every row multiplied by 2^-k
  // with k in 24..34 - exact, and it leaves the feasible set and the optimum alone, while
  // every right-hand side lands below the feasibility tolerance. The reference is the
  // unscaled model solved without presolve. Before the redundant-row tests of presolve.cpp
  // and dual_reduce.cpp took the row's scale, 116 of 200 such instances came back `optimal`
  // at another objective, and none did with presolve off.
  std::mt19937 rng(20261111);
  const auto draw = [&rng](int lo, int hi) {
    return std::uniform_int_distribution<int>(lo, hi)(rng);
  };
  Options plain = quiet();
  plain.set_bool("presolve", false);
  int compared = 0, wrong = 0;
  std::vector<std::string> failures;
  for (int trial = 0; trial < 300; ++trial) {
    const auto n = static_cast<std::size_t>(draw(3, 6));
    const auto m = static_cast<std::size_t>(draw(2, 5));
    std::vector<double> cost(n), upper(n), point(n), rhs(m), scale(m), ones(m, 1.0);
    std::vector<std::vector<double>> rows(m, std::vector<double>(n));
    for (std::size_t j = 0; j < n; ++j) {
      upper[j] = draw(1, 4);
      point[j] = draw(0, static_cast<int>(upper[j]));
      cost[j] = draw(-3, 3);
    }
    for (std::size_t i = 0; i < m; ++i) {
      double activity = 0.0;
      for (std::size_t j = 0; j < n; ++j) {
        rows[i][j] = draw(0, 3) == 0 ? 0.0 : draw(-3, 3);
        activity += rows[i][j] * point[j];
      }
      rhs[i] = activity - draw(0, 1);
      scale[i] = std::ldexp(1.0, -draw(24, 34));
    }
    const Solution reference = solve(box_lp(cost, upper, rows, rhs, ones), plain);
    if (reference.status != SolveStatus::kOptimal) continue;
    ++compared;
    const Solution scaled = solve(box_lp(cost, upper, rows, rhs, scale), quiet());
    if (scaled.status == SolveStatus::kOptimal &&
        std::fabs(scaled.objective - reference.objective) <=
            1e-6 * std::max(1.0, std::fabs(reference.objective))) {
      continue;
    }
    ++wrong;
    if (failures.size() < 5) {
      failures.push_back("trial " + std::to_string(trial) + ": " + to_string(scaled.status) +
                         " objective " + std::to_string(scaled.objective) + ", reference " +
                         std::to_string(reference.objective));
    }
  }
  std::cout << "\n=== row-scaled instances, presolve on ===\n  compared " << compared
            << "\n  WRONG " << wrong << "\n";
  for (const std::string& f : failures) std::cout << "  " << f << "\n";
  EXPECT_EQ(wrong, 0);
  EXPECT_GT(compared, 250);
}

}  // namespace
}  // namespace sankhya
