// SPDX-License-Identifier: Apache-2.0
// SANKHYA - #1343: a MILP whose LP relaxation is unbounded is unbounded only if it has an
// integer feasible point.
//
// The search stopped at an unbounded root relaxation and reported the MILP unbounded. Over
// integers 2 y - 2 z = 1 has no point, so a model with that row is infeasible whatever a
// column outside it does to the objective. With rational data the two cases are told apart
// by whether an integer point exists (Meyer 1974), and solve_branch_and_bound now finds
// out. The first three models are infeasible and came back `unbounded`; the general-form
// oracle fuzz (#1334) drew the third.

#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

Model build(const std::vector<std::vector<double>>& rows, const std::vector<double>& row_lower,
            const std::vector<double>& row_upper, const std::vector<double>& cost,
            const std::vector<double>& col_lower, const std::vector<double>& col_upper,
            const std::vector<bool>& integral) {
  Model model;
  const auto cols = static_cast<Index>(cost.size());
  const auto num_rows = static_cast<Index>(rows.size());
  model.resize_columns(cols);
  model.resize_rows(num_rows);
  model.col_cost = cost;
  model.col_lower = col_lower;
  model.col_upper = col_upper;
  model.row_lower = row_lower;
  model.row_upper = row_upper;
  model.matrix.reset(num_rows, cols);
  for (Index j = 0; j < cols; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (integral[u]) model.col_type[u] = VarType::kInteger;
    for (Index i = 0; i < num_rows; ++i) {
      const double a = rows[static_cast<std::size_t>(i)][u];
      if (a != 0.0) model.matrix.add_entry(i, j, a);
    }
  }
  model.matrix.finalize();
  return model;
}

/// The status under the defaults, with presolve off, and on one thread.
void expect_status(const Model& model, SolveStatus wanted) {
  ASSERT_EQ(model.validate(), "");
  for (const bool presolve : {true, false}) {
    for (const int threads : {0, 1}) {  // 0: one worker per core
      Options options;
      options.set_bool("log_to_console", false);
      options.set_bool("presolve", presolve);
      options.set_int("mip_threads", threads);
      const Solution solved = solve(model, options);
      EXPECT_EQ(solved.status, wanted)
          << "presolve " << presolve << ", mip_threads " << threads << ": " << solved.message;
    }
  }
}

const double inf = kInfinity;

TEST(MilpUnboundedRelaxation, NoIntegerPointIsInfeasibleNotUnbounded) {
  // min -x: 2 y - 2 z = 1, y and z integer in [0, 5], x >= 0 in no row.
  expect_status(
      build({{0, 2, -2}}, {1}, {1}, {-1, 0, 0}, {0, 0, 0}, {inf, 5, 5}, {false, true, true}),
      SolveStatus::kInfeasible);
  // The same with x in a row of its own, x - w >= 0, both free to grow.
  expect_status(build({{0, 2, -2, 0}, {1, 0, 0, -1}}, {1, 0}, {1, inf}, {-1, 0, 0, 0},
                      {0, 0, 0, 0}, {inf, 5, 5, inf}, {false, true, true, false}),
                SolveStatus::kInfeasible);
  // And maximized: max x.
  Model maximized =
      build({{0, 2, -2}}, {1}, {1}, {1, 0, 0}, {0, 0, 0}, {inf, 5, 5}, {false, true, true});
  maximized.sense = ObjSense::kMaximize;
  expect_status(maximized, SolveStatus::kInfeasible);
}

TEST(MilpUnboundedRelaxation, TheFuzzModelIsInfeasible) {
  // min 6 x0: -3 x1 - 3/2 x2 = -5, x0 free, x1 integer at most 199/3, x2 integer in
  // [-193/4, -11/2]. The row is 3 (2 x1 + x2) = 10.
  expect_status(build({{0, -3, -1.5}}, {-5}, {-5}, {6, 0, 0}, {-inf, -inf, -48.25},
                      {inf, 199.0 / 3.0, -5.5}, {false, true, true}),
                SolveStatus::kInfeasible);
}

TEST(MilpUnboundedRelaxation, WithAnIntegerPointItIsStillUnbounded) {
  // The control: 2 y - 2 z = 2 has integer points, and x grows without limit beside them.
  expect_status(
      build({{0, 2, -2}}, {2}, {2}, {-1, 0, 0}, {0, 0, 0}, {inf, 5, 5}, {false, true, true}),
      SolveStatus::kUnbounded);
  // A general-integer column that is itself the unbounded one: min -y, y >= 0 integer.
  expect_status(build({{1, -1}}, {0}, {inf}, {-1, 0}, {0, 0}, {inf, inf}, {true, false}),
                SolveStatus::kUnbounded);
}

}  // namespace
}  // namespace sankhya
