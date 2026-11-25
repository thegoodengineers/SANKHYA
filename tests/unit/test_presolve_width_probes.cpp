// SPDX-License-Identifier: Apache-2.0
// SANKHYA - probes beside the narrow-box and near-equation tests. Not for merge: each one
// asks a question of main and prints the answer.

#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

Model build(ObjSense sense, const std::vector<std::vector<double>>& rows,
            const std::vector<double>& row_lower, const std::vector<double>& row_upper,
            const std::vector<double>& cost, const std::vector<double>& col_lower,
            const std::vector<double>& col_upper, const std::vector<bool>& integer) {
  Model model;
  const auto n = static_cast<Index>(cost.size());
  const auto m = static_cast<Index>(rows.size());
  model.sense = sense;
  model.col_cost = cost;
  model.col_lower = col_lower;
  model.col_upper = col_upper;
  model.col_type.assign(cost.size(), VarType::kContinuous);
  for (std::size_t j = 0; j < integer.size(); ++j) {
    if (integer[j]) model.col_type[j] = VarType::kInteger;
  }
  model.row_lower = row_lower;
  model.row_upper = row_upper;
  model.matrix.reset(m, n);
  for (Index i = 0; i < m; ++i) {
    for (Index j = 0; j < n; ++j) {
      const double v = rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
      if (v != 0.0) model.matrix.add_entry(i, j, v);
    }
  }
  model.matrix.finalize(0.0);
  model.hessian.reset(n, n);
  model.hessian.finalize();
  EXPECT_EQ(model.validate(), "");
  return model;
}

Solution report(const Model& model, const Options& options, const std::string& what) {
  const Solution s = solve(model, options);
  std::cout.precision(17);
  std::cout << "  PROBE " << what << ": " << to_string(s.status) << ", objective "
            << s.objective << ", x =";
  for (const double v : s.col_value) std::cout << ' ' << v;
  std::cout << "\n    message: " << s.message << "\n";
  return s;
}

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  return options;
}

TEST(WidthProbe, TheEngineSolvesWhatTheNarrowBoxFixLeavesInTheReducedModel) {
  // What presolve hands on once z's carried box is kept: max 2^30 z - w - v,
  // 2^30 z - 10 w - 10 v <= 0, w + v >= 1, z in [0, 10 / 2^30], w and v integer in [0, 5].
  // Optimum 9. Presolve off, so the box is not fixed on the way.
  const Model model =
      build(ObjSense::kMaximize, {{0x1p30, -10, -10}, {0, 1, 1}}, {-kInf, 1}, {0, kInf},
            {0x1p30, -1, -1}, {0, 0, 0}, {10 / 0x1p30, 5, 5}, {false, true, true});
  Options options = quiet();
  options.set_bool("presolve", false);
  const Solution s = report(model, options, "reduced model of the carried box, presolve off");
  ASSERT_EQ(s.status, SolveStatus::kOptimal);
  EXPECT_NEAR(s.objective, 9.0, 1e-6);
}

TEST(WidthProbe, ParallelRowsWithSmallCoefficientsAreNotMerged) {
  // max r1 + 1.25 r2 + z:  s r1 + 1.5 s r2 <= 7,  s r1 + s r2 <= 5,  s = 2^-30,
  // r1, r2 >= 0, z binary. Optimum r1 = 2^30, r2 = 4 * 2^30, z = 1: 6 * 2^30 + 1.
  // The rows are not parallel; their coefficients differ by 4.7e-10, under the absolute
  // 1e-9 the proportionality test falls back to below one.
  const double s = 0x1p-30;
  const Model model =
      build(ObjSense::kMaximize, {{s, 1.5 * s, 0}, {s, s, 0}}, {-kInf, -kInf}, {7, 5},
            {1, 1.25, 1}, {0, 0, 0}, {kInf, kInf, 1}, {false, false, true});
  const double exact = 6 * 0x1p30 + 1;
  for (const bool parallel : {false, true}) {
    Options options = quiet();
    options.set_bool("presolve_parallel_rows", parallel);
    const Solution s2 =
        report(model, options, std::string("parallel rows ") + (parallel ? "on" : "off"));
    EXPECT_EQ(s2.status, SolveStatus::kOptimal) << "parallel rows " << parallel;
    EXPECT_NEAR(s2.objective, exact, 1e-6 * exact) << "parallel rows " << parallel;
  }
}

TEST(WidthProbe, ANearEquationDoesNotMakeAColumnAnImpliedInteger) {
  // max 2^30 (x - y1 - y2):  0 <= x - y1 - y2 <= 2^-24,  x in [0, 10],  y integer in [0, 3].
  // Optimum 64. presolve_implied_integer takes a row at most 1e-7 wide for an equation.
  const Model model =
      build(ObjSense::kMaximize, {{1, -1, -1}}, {0}, {0x1p-24}, {0x1p30, -0x1p30, -0x1p30},
            {0, 0, 0}, {10, 3, 3}, {false, true, true});
  for (const bool implied : {false, true}) {
    Options options = quiet();
    options.set_bool("presolve_implied_integer", implied);
    const Solution s =
        report(model, options, std::string("implied integer ") + (implied ? "on" : "off"));
    EXPECT_EQ(s.status, SolveStatus::kOptimal) << "implied integer " << implied;
    EXPECT_NEAR(s.objective, 64.0, 1e-6) << "implied integer " << implied;
  }
}

}  // namespace
}  // namespace sankhya
