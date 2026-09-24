// SPDX-License-Identifier: Apache-2.0
// SANKHYA - implied-integer detection in presolve (#513).
//
// A continuous variable is implied integer when it appears in an equality row where the RHS is
// integer, every other live variable has an integer coefficient and is already integer, and the
// remaining variable's coefficient divides all of those values. The pass runs to a fixpoint:
// a variable promoted in one row can unlock further detections in another.

#include <gtest/gtest.h>

#include <limits>
#include <vector>

#include "presolve/presolve.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

Options with_implied_integer(bool on) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", true);
  options.set_bool("presolve_implied_integer", on);
  return options;
}

Model build(const std::vector<std::vector<double>>& rows, const std::vector<double>& row_lower,
            const std::vector<double>& row_upper, const std::vector<double>& cost,
            const std::vector<double>& col_lower, const std::vector<double>& col_upper,
            const std::vector<bool>& integer = {}) {
  Model model;
  const auto n = static_cast<Index>(cost.size());
  const auto m = static_cast<Index>(rows.size());
  model.sense = ObjSense::kMinimize;
  model.col_cost = cost;
  model.col_lower = col_lower;
  model.col_upper = col_upper;
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kContinuous);
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
  model.matrix.finalize();
  model.hessian.reset(n, n);
  model.hessian.finalize();
  EXPECT_EQ(model.validate(), "");
  return model;
}

// x0 + x1 = 3, x1 integer in [0, 3], x0 continuous in [0, 3].
// x0 = 3 - x1 must be integer. With coefficient +1 the detection fires.
TEST(PresolveImpliedInteger, ContinuousVariablePromotedByEqualityRow) {
  Model model =
      build({{1.0, 1.0}},                            // rows
            {3.0}, {3.0},                            // equality: x0 + x1 = 3
            {1.0, 0.0},                              // minimize x0
            {0.0, 0.0}, {3.0, 3.0}, {false, true});  // x1 is integer, x0 is continuous

  Logger logger(nullptr);
  const presolve::Result result_on =
      presolve::presolve(model, with_implied_integer(true), logger);
  EXPECT_GE(result_on.report.implied_integers, 1);

  const presolve::Result result_off =
      presolve::presolve(model, with_implied_integer(false), logger);
  EXPECT_EQ(result_off.report.implied_integers, 0);
}

// Two variables, both continuous, one equality row. Neither can be detected on its own because
// neither is integer when the other is not: detection should NOT fire.
TEST(PresolveImpliedInteger, TwoContinuousVariablesNotDetected) {
  Model model =
      build({{1.0, 1.0}}, {3.0}, {3.0}, {1.0, 1.0}, {0.0, 0.0}, {3.0, 3.0}, {false, false});

  Logger logger(nullptr);
  const presolve::Result result = presolve::presolve(model, with_implied_integer(true), logger);
  EXPECT_EQ(result.report.implied_integers, 0);
}

// Chain: x0 + x1 = 3 (x1 integer) implies x0 integer.
// Then x0 + x2 = 5 (x0 now implied integer) implies x2 integer.
TEST(PresolveImpliedInteger, FixpointChainPropagatesToSecondRow) {
  // 2 rows, 3 variables: x0 cont, x1 int, x2 cont.
  // Row 0: x0 + x1 = 3
  // Row 1: x0 + x2 = 5
  Model model = build({{1.0, 1.0, 0.0}, {1.0, 0.0, 1.0}}, {3.0, 5.0}, {3.0, 5.0},
                      {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}, {5.0, 3.0, 5.0}, {false, true, false});

  Logger logger(nullptr);
  const presolve::Result result = presolve::presolve(model, with_implied_integer(true), logger);
  EXPECT_GE(result.report.implied_integers, 2);
}

// Non-unit coefficient: x0 appears in 2*x0 + x1 = 4, x1 integer in [0,4], x0 continuous.
// 2 divides 4 and 1? No: 1/2 is not integer, so x0 is NOT implied integer.
TEST(PresolveImpliedInteger, NonUnitCoefficientNotDivisible) {
  Model model =
      build({{2.0, 1.0}}, {4.0}, {4.0}, {1.0, 0.0}, {0.0, 0.0}, {2.0, 4.0}, {false, true});

  Logger logger(nullptr);
  const presolve::Result result = presolve::presolve(model, with_implied_integer(true), logger);
  EXPECT_EQ(result.report.implied_integers, 0);
}

// x0 appears in 2*x0 + 2*x1 = 6 with x1 integer. 2 divides 6 and 2, so x0 IS implied integer.
TEST(PresolveImpliedInteger, NonUnitCoefficientDivisible) {
  Model model =
      build({{2.0, 2.0}}, {6.0}, {6.0}, {1.0, 0.0}, {0.0, 0.0}, {3.0, 3.0}, {false, true});

  Logger logger(nullptr);
  const presolve::Result result = presolve::presolve(model, with_implied_integer(true), logger);
  EXPECT_GE(result.report.implied_integers, 1);
}

}  // namespace
}  // namespace sankhya
