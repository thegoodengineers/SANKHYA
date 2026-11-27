// SPDX-License-Identifier: Apache-2.0
// SANKHYA - presolve fixes a column only when its box is a point (#1362).
//
// A box narrower than the feasibility tolerance is still a box: what it is worth is its
// width times the coefficients and the cost the column carries, and with a coefficient of
// 2^30 a width of 6e-8 is 64 units of row activity. Presolve used to fix every column whose
// box was at most 1e-7 wide at its lower bound, in the column's own units, and so removed
// that range from the model: a MILP came back `optimal` 10 below its optimum and a feasible
// LP came back `infeasible`. The box need not be narrow in the file: a doubleton equation
// y = 2^30 z carries y's box [0, 10] onto z as [0, 9.3e-9].
//
// Every number below is a power of two or a small integer, so the stated optimum is exact.

#include <algorithm>
#include <cmath>
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
constexpr double kBig = 0x1p30;      // 1.07e9
constexpr double kNarrow = 0x1p-24;  // 5.96e-8, inside the feasibility tolerance
constexpr double kSmall = 0x1p-27;   // 7.45e-9

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

/// Solved with presolve on and off, each printed, each held to `objective`.
void expect_optimum(const Model& model, double objective, double tolerance,
                    const std::string& what) {
  for (const bool presolve : {true, false}) {
    const Solution s = solve(model, quiet(presolve));
    std::cout.precision(12);
    std::cout << "  " << what << ", presolve " << (presolve ? "on" : "off") << ": "
              << to_string(s.status) << ", objective " << s.objective << ", x =";
    for (const double v : s.col_value) std::cout << ' ' << v;
    std::cout << "\n";
    EXPECT_EQ(s.status, SolveStatus::kOptimal)
        << what << ", presolve " << presolve << ": " << s.message;
    if (s.status != SolveStatus::kOptimal) continue;
    EXPECT_NEAR(s.objective, objective, tolerance) << what << ", presolve " << presolve;
  }
}

Count fixed_columns(const Model& model) {
  Logger logger(nullptr);
  return presolve::presolve(model, quiet(), logger).report.fixed_columns;
}

TEST(PresolveNarrowBox, ABoxADoubletonCarriedOverIsNotFixed) {
  // max y - w - v:  y - 2^30 z = 0,  y - 10 w - 10 v <= 0,  w + v >= 1,
  // y in [0, 10], z >= 0, w and v integer in [0, 5].
  // With s = w + v >= 1 the rows give y <= min(10, 10 s) = 10, so y - s <= 9, and y = 10,
  // z = 10 / 2^30, s = 1 attains it. The doubleton eliminates the first of two continuous
  // columns: y before z puts y's box onto z as [0, 9.3e-9], which was then fixed at 0 with
  // its 2^30 in the second row, and y = 0, objective -1, came back `optimal`. z before y
  // eliminates z and never made the narrow box; both orders must agree.
  const Model y_first = build(
      ObjSense::kMaximize, {{1, 0, 0, -kBig}, {1, -10, -10, 0}, {0, 1, 1, 0}}, {0, -kInf, 1},
      {0, 0, kInf}, {1, -1, -1, 0}, {0, 0, 0, 0}, {10, 5, 5, kInf}, {false, true, true, false});
  const Model z_first = build(
      ObjSense::kMaximize, {{-kBig, 0, 0, 1}, {0, -10, -10, 1}, {0, 1, 1, 0}}, {0, -kInf, 1},
      {0, 0, kInf}, {0, -1, -1, 1}, {0, 0, 0, 0}, {kInf, 5, 5, 10}, {false, true, true, false});
  EXPECT_EQ(fixed_columns(y_first), 0) << "z's box [0, 10 / 2^30] is worth 10 in its row";
  expect_optimum(y_first, 9.0, 1e-6, "y before z");
  expect_optimum(z_first, 9.0, 1e-6, "z before y");
}

TEST(PresolveNarrowBox, ANarrowBoxInTheModelIsNotFixed) {
  // max y:  y - 2^30 x <= 0,  y integer in [0, 100],  x in [0, 2^-24].  y <= 2^30 x <= 64.
  // x was fixed at 0, the row became y <= 0, and 0 came back `optimal`.
  const Model model = build(ObjSense::kMaximize, {{1, -kBig}}, {-kInf}, {0}, {1, 0}, {0, 0},
                            {100, kNarrow}, {true, false});
  EXPECT_EQ(fixed_columns(model), 0);
  expect_optimum(model, 64.0, 1e-6, "a narrow box under 2^30");
}

TEST(PresolveNarrowBox, ANarrowBoxKeepsWhatItsCostIsWorth) {
  // max 2^30 x + z:  x + z <= 2 (never binding),  x in [0, 2^-24],  z binary.  64 + 1.
  // x was fixed at 0 and 1 came back `optimal`: the cost is something the box feeds too.
  const Model model = build(ObjSense::kMaximize, {{1, 1}}, {-kInf}, {2}, {kBig, 1}, {0, 0},
                            {kNarrow, 1}, {false, true});
  EXPECT_EQ(fixed_columns(model), 0);
  expect_optimum(model, 65.0, 1e-6, "a narrow box with a cost of 2^30");
}

TEST(PresolveNarrowBox, ANarrowBoxDoesNotEmptyTheRowThatNeedsIt) {
  // min x:  2^30 x >= 32,  x in [0, 2^-24].  Feasible for x in [2^-25, 2^-24]; the optimum
  // is x = 2^-25. x was fixed at 0, the row was left with no entries and a lower bound of
  // 32, and presolve called the LP infeasible.
  const Model model =
      build(ObjSense::kMinimize, {{kBig}}, {32}, {kInf}, {1}, {0}, {kNarrow}, {false});
  EXPECT_EQ(fixed_columns(model), 0);
  expect_optimum(model, 0x1p-25, 1e-10, "a row that needs the narrow box");
  const Solution s = solve(model, quiet());
  ASSERT_EQ(s.col_value.size(), 1u);
  EXPECT_GE(kBig * s.col_value[0], 32.0 - 1e-6);
}

TEST(PresolveNarrowBox, ANarrowBoxIsNotFixedBesideASmallerCoefficientEither) {
  // min x0 + x1:  x0 + 2^-27 x1 >= 2^-27,  x0 in [0, 2^-25],  x1 in [0, 1/2].
  // x0 = 2^-27, x1 = 0 is feasible and optimal. No coefficient of x0 is above one, so a
  // width weighed by the column's own coefficients would still call it fixed: at x0 = 0 the
  // row is 2^-27 x1 >= 2^-27, a singleton that says x1 >= 1 against x1 <= 1/2, and presolve
  // called the LP infeasible. The optimum is 7.45e-9; the solve is held to the feasibility
  // tolerance around it, since the row may be met to that tolerance.
  const Model model = build(ObjSense::kMinimize, {{1, kSmall}}, {kSmall}, {kInf}, {1, 1},
                            {0, 0}, {0x1p-25, 0.5}, {false, false});
  EXPECT_EQ(fixed_columns(model), 0);
  expect_optimum(model, kSmall, 1e-7, "a narrow box beside a smaller coefficient");
}

TEST(PresolveNarrowBox, ABoxThatIsAPointOrCrossedByRoundingIsStillFixed) {
  // A point: x in [3, 3]. Crossed by rounding: 0.1 x <= 0.3 against x >= 3 gives the upper
  // bound 0.3 / 0.1 = 2.9999999999999996, one ulp under the lower. Neither box holds two
  // values, so both columns are fixed, at 3.
  const Model point = build(ObjSense::kMinimize, {{1, 1}}, {-kInf}, {10}, {1, 1}, {3, 0},
                            {3, 5}, {false, false});
  EXPECT_EQ(fixed_columns(point), 1);
  const Model crossed =
      build(ObjSense::kMinimize, {{0.1}}, {-kInf}, {0.3}, {1}, {3}, {10}, {false});
  EXPECT_EQ(fixed_columns(crossed), 1);
  expect_optimum(crossed, 3.0, 1e-9, "a box crossed by rounding");
}

TEST(PresolveNarrowBox, ANarrowBoxLeftByADoubletonComesBackBasicInsideItsOriginalBox) {
  // min 2k - y:  k - e = 1,  y - k <= 0,  e >= 0,  k in [0, 1 + 2^-24],  y in [0, 10].
  // y = k at the optimum, so the objective is k, and k = 1 + e >= 1: k = y = 1, e = 0,
  // objective 1. The doubleton eliminates e and gives k the box [1, 1 + 2^-24], 6e-8 wide,
  // which since #1362 stays in the reduced model; the engine leaves k at its lower end, 1.
  // In the original model 1 is not a bound of k: k is inside [0, 1 + 2^-24], held there by
  // the equation, and basic, with e on its lower bound. 1 is within the tolerance of k's
  // UPPER bound, and postsolve took that for "on a bound": k came back at_lower, 1 away
  // from 0, the shape of scaled Netlib finnis's 4e-5 on 2E51CAP (#1365).
  const Model model =
      build(ObjSense::kMinimize, {{-1, 1, 0}, {0, -1, 1}}, {1, -kInf}, {1, 0}, {0, 2, -1},
            {0, 0, 0}, {kInf, 1 + kNarrow, 10}, {false, false, false});
  const Solution s = solve(model, quiet());
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_NEAR(s.objective, 1.0, 1e-9);
  ASSERT_EQ(s.col_status.size(), 3u);
  ASSERT_EQ(s.row_status.size(), 2u);
  // The basis the .sol file carries, held as tools/verify_solution.py holds it: one basic
  // entry per row, and every nonbasic entry on the bound its status names.
  int basic = 0;
  const auto on_named = [&](BasisStatus status, double value, double lower, double upper,
                            const std::string& what) {
    std::cout << "  " << what << " = " << value << ", " << to_string(status) << "\n";
    if (status == BasisStatus::kBasic) ++basic;
    const double scale = std::max(1.0, std::fabs(value));
    if (status == BasisStatus::kAtLower) {
      EXPECT_LE(std::fabs(value - lower), 1e-7 * scale) << what << " is not on its lower bound";
    }
    if (status == BasisStatus::kAtUpper) {
      EXPECT_LE(std::fabs(value - upper), 1e-7 * scale) << what << " is not on its upper bound";
    }
  };
  const char* names[] = {"e", "k", "y"};
  for (Index j = 0; j < 3; ++j) {
    const auto u = static_cast<std::size_t>(j);
    on_named(s.col_status[u], s.col_value[u], model.col_lower[u], model.col_upper[u], names[u]);
  }
  for (Index i = 0; i < 2; ++i) {
    const auto u = static_cast<std::size_t>(i);
    double activity = 0.0;
    for (Index j = 0; j < 3; ++j) {
      const ColumnView column = model.matrix.column(j);
      for (Index t = 0; t < column.size; ++t) {
        if (column.rows[t] == i)
          activity += column.values[t] * s.col_value[static_cast<std::size_t>(j)];
      }
    }
    on_named(s.row_status[u], activity, model.row_lower[u], model.row_upper[u],
             "row " + std::to_string(i));
  }
  EXPECT_EQ(basic, 2);
  EXPECT_EQ(s.col_status[1], BasisStatus::kBasic) << "k is inside its original box";
}

}  // namespace
}  // namespace sankhya
