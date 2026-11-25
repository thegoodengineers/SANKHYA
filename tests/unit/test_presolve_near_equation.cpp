// SPDX-License-Identifier: Apache-2.0
// SANKHYA - presolve substitutes a two-entry row only when the row is an equation.
//
// A ranged row whose range is inside the feasibility tolerance is still a range: on a row
// scaled by 2^-27 a range of 3.7e-8 is five units of its columns. The doubleton reduction
// used to take every two-entry row at most 1e-7 wide for an equation at its lower bound,
// in the row's own units, and so removed the rest of the range from the model: a MILP came
// back `optimal` 6 below its optimum and a feasible LP came back `infeasible`.
//
// Every number below is a power of two or a small integer, so the stated optimum is exact.

#include <iostream>
#include <limits>
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

Options quiet(bool presolve = true) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", presolve);
  return options;
}

/// Solved with presolve on, and off as well unless `presolve_only`, each printed and held to
/// `objective`.
void expect_optimum(const Model& model, double objective, const std::string& what,
                    bool presolve_only = false) {
  for (const bool presolve : {true, false}) {
    if (!presolve && presolve_only) continue;
    const Solution s = solve(model, quiet(presolve));
    std::cout.precision(12);
    std::cout << "  " << what << ", presolve " << (presolve ? "on" : "off") << ": "
              << to_string(s.status) << ", objective " << s.objective << ", x =";
    for (const double v : s.col_value) std::cout << ' ' << v;
    std::cout << "\n";
    EXPECT_EQ(s.status, SolveStatus::kOptimal)
        << what << ", presolve " << presolve << ": " << s.message;
    if (s.status != SolveStatus::kOptimal) continue;
    EXPECT_NEAR(s.objective, objective, 1e-6) << what << ", presolve " << presolve;
  }
}

Count doubleton_equations(const Model& model) {
  Logger logger(nullptr);
  return presolve::presolve(model, quiet(), logger).report.doubleton_equations;
}

/// max z - w:  lower <= s x + s z <= upper,  z + w >= 3,
/// x in [0, 10],  z integer in [0, 7],  w integer in [0, 10].
Model scaled_pair(double s, double lower, double upper) {
  return build(ObjSense::kMaximize, {{s, s, 0}, {0, 1, 1}}, {lower, 3}, {upper, kInf},
               {0, 1, -1}, {0, 0, 0}, {10, 7, 10}, {false, true, true});
}

TEST(PresolveNearEquation, ARangedRowInsideTheToleranceIsNotAnEquation) {
  // The first row is 2 <= x + z <= 7 multiplied by 2^-27: its range is 3.7e-8. z = 7,
  // w = 0, x = 0 is feasible and z - w <= z <= 7, so the optimum is 7. Taken for the
  // equation x + z = 2 the row gave z <= 2, and z = 2, w = 1, objective 1, came back
  // `optimal`.
  const double s = 0x1p-27;
  const Model model = scaled_pair(s, 2 * s, 7 * s);
  EXPECT_EQ(doubleton_equations(model), 0) << "2 <= x + z <= 7 is not x + z = 2";
  expect_optimum(model, 7.0, "a ranged row scaled by 2^-27");
}

TEST(PresolveNearEquation, ARangedRowInsideTheToleranceKeepsAnLpFeasible) {
  // min x + y:  0 <= 2^-20 (x + y) <= 2^-20 * 3/32,  x, y >= 1/32.
  // x = y = 1/32 is feasible (x + y = 1/16 against 3/32) and optimal. The range is 8.9e-8;
  // taken for x + y = 0 the row gave y <= -1/32 against y >= 1/32, and presolve called the
  // LP infeasible.
  const double s = 0x1p-20;
  const Model model = build(ObjSense::kMinimize, {{s, s}}, {0}, {s * 3 / 32}, {1, 1},
                            {0.03125, 0.03125}, {kInf, kInf}, {false, false});
  EXPECT_EQ(doubleton_equations(model), 0);
  expect_optimum(model, 0.0625, "a ranged LP row scaled by 2^-20");
}

TEST(PresolveNearEquation, AnEquationStillFoldsWhateverItsScale) {
  // x + z = 2, as written and multiplied by 2^-27: z <= 2, so z = 2, w = 1, objective 1.
  // With presolve only: the scaled equation is 1.5e-8 wide of nothing, and an engine that
  // is handed it meets it to the feasibility tolerance, which is not what is under test.
  for (const double s : {1.0, 0x1p-27}) {
    const Model model = scaled_pair(s, 2 * s, 2 * s);
    EXPECT_EQ(doubleton_equations(model), 1) << "scale " << s;
    expect_optimum(model, 1.0, "an equation scaled by " + std::to_string(s), true);
  }
}

}  // namespace
}  // namespace sankhya
