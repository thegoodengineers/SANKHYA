// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the exact repair of a basis optimal only to tolerance (#757).
//
// Each model is two variables wide so its exact optimum can be worked out by hand, and each
// hands the repair the basis a floating-point simplex would stop at: infeasible or of the
// wrong sign by 2^-40, far inside the 1e-7 tolerances. The repair must reach the exactly
// optimal basis stated in the comment, with the pivots stated, and the exact check must then
// verify it.

#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "exact/exact_repair.hpp"
#include "exact/exact_verify.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::exact {
namespace {

const double kTiny = std::ldexp(1.0, -40);

Model make_lp(const std::vector<std::vector<double>>& rows,
              const std::vector<double>& row_lower, const std::vector<double>& row_upper,
              const std::vector<double>& cost, const std::vector<double>& col_lower,
              const std::vector<double>& col_upper) {
  Model model;
  const auto n = static_cast<Index>(cost.size());
  const auto m = static_cast<Index>(rows.size());
  model.col_cost = cost;
  model.col_lower = col_lower;
  model.col_upper = col_upper;
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kContinuous);
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
  return model;
}

Solution reported(const Model& model, std::vector<BasisStatus> cols,
                  std::vector<BasisStatus> rows, std::vector<double> values) {
  Solution solution;
  solution.status = SolveStatus::kOptimal;
  solution.col_status = std::move(cols);
  solution.row_status = std::move(rows);
  solution.col_value = std::move(values);
  solution.col_dual.assign(solution.col_value.size(), 0.0);
  solution.row_dual.assign(solution.row_status.size(), 0.0);
  solution.recompute_quality(model);
  return solution;
}

// min y  s.t.  r: x + y = 1,  0 <= x <= 1 - 2^-40,  y >= 0.
// Optimum y = 2^-40 (x at its upper bound). The float basis {x} with y at 0 puts x at 1,
// 2^-40 above its bound: primal infeasible, dual feasible (d_y = 1). One dual simplex pivot
// (x leaves to its upper bound, y enters) reaches the exact optimum.
Model primal_off_by_tiny() {
  return make_lp({{1.0, 1.0}}, {1.0}, {1.0}, {0.0, 1.0}, {0.0, 0.0}, {1.0 - kTiny, kInfinity});
}

TEST(ExactRepair, APrimalInfeasibilityOfTwoToTheMinusFortyIsRepairedByADualPivot) {
  const Model model = primal_off_by_tiny();
  const Solution solution = reported(model, {BasisStatus::kBasic, BasisStatus::kAtLower},
                                     {BasisStatus::kFixed}, {1.0, 0.0});
  EXPECT_EQ(verify_basis_exact(model, solution).verdict, ExactVerdict::kFailed);
  const RepairResult repair = repair_basis_exact(model, solution);
  ASSERT_EQ(repair.verdict, ExactVerdict::kVerified) << repair.message;
  EXPECT_EQ(repair.dual_pivots, 1);
  EXPECT_EQ(repair.primal_pivots, 0);
  EXPECT_EQ(repair.col_status[0], BasisStatus::kAtUpper);
  EXPECT_EQ(repair.col_status[1], BasisStatus::kBasic);
  EXPECT_EQ(repair.col_value[0], 1.0 - kTiny);
  EXPECT_EQ(repair.col_value[1], kTiny);
  EXPECT_EQ(repair.row_dual[0], 1.0);
  EXPECT_EQ(repair.col_dual[0], -1.0);
}

// max x + (1 + 2^-40) y  s.t.  r: x + y <= 1,  x, y >= 0.
// Optimum y = 1. The float basis {x} leaves y's reduced cost at -2^-40 in minimise space:
// primal feasible, dual infeasible. One primal simplex pivot (y enters, x leaves at 0).
Model dual_off_by_tiny() {
  Model model = make_lp({{1.0, 1.0}}, {-kInfinity}, {1.0}, {1.0, 1.0 + kTiny}, {0.0, 0.0},
                        {kInfinity, kInfinity});
  model.sense = ObjSense::kMaximize;
  return model;
}

TEST(ExactRepair, AReducedCostOfTheWrongSignByTwoToTheMinusFortyIsRepairedByAPrimalPivot) {
  const Model model = dual_off_by_tiny();
  const Solution solution = reported(model, {BasisStatus::kBasic, BasisStatus::kAtLower},
                                     {BasisStatus::kAtUpper}, {1.0, 0.0});
  EXPECT_EQ(verify_basis_exact(model, solution).verdict, ExactVerdict::kFailed);
  const RepairResult repair = repair_basis_exact(model, solution);
  ASSERT_EQ(repair.verdict, ExactVerdict::kVerified) << repair.message;
  EXPECT_EQ(repair.dual_pivots, 0);
  EXPECT_EQ(repair.primal_pivots, 1);
  EXPECT_EQ(repair.col_status[0], BasisStatus::kAtLower);
  EXPECT_EQ(repair.col_status[1], BasisStatus::kBasic);
  EXPECT_EQ(repair.col_value[1], 1.0);
  EXPECT_EQ(repair.row_dual[0], 1.0 + kTiny);  // the model's sense: a maximise dual
}

// Both at once, one block each: r1 and x1, y1 as primal_off_by_tiny, r2 and x2, y2 as
// dual_off_by_tiny with the costs negated to minimise. The basis is primal AND dual
// infeasible, so the repair shifts y2's cost, runs the dual simplex, restores the cost and
// runs the primal simplex: one pivot of each kind.
TEST(ExactRepair, BothInfeasibilitiesAtOnceAreRepairedThroughACostShift) {
  const Model model =
      make_lp({{1.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 1.0}}, {1.0, -kInfinity}, {1.0, 1.0},
              {0.0, 1.0, -1.0, -(1.0 + kTiny)}, {0.0, 0.0, 0.0, 0.0},
              {1.0 - kTiny, kInfinity, kInfinity, kInfinity});
  const Solution solution = reported(
      model,
      {BasisStatus::kBasic, BasisStatus::kAtLower, BasisStatus::kBasic, BasisStatus::kAtLower},
      {BasisStatus::kFixed, BasisStatus::kAtUpper}, {1.0, 0.0, 1.0, 0.0});
  const RepairResult repair = repair_basis_exact(model, solution);
  ASSERT_EQ(repair.verdict, ExactVerdict::kVerified) << repair.message;
  EXPECT_EQ(repair.shifted_costs, 1);
  EXPECT_EQ(repair.dual_pivots, 1);
  EXPECT_EQ(repair.primal_pivots, 1);
  EXPECT_EQ(repair.col_status[1], BasisStatus::kBasic);
  EXPECT_EQ(repair.col_status[3], BasisStatus::kBasic);
  EXPECT_EQ(repair.col_value[1], kTiny);
  EXPECT_EQ(repair.col_value[3], 1.0);
}

// A boxed column whose reduced cost has the wrong sign moves to its other bound when that
// comes first: min -2^-40 y  s.t.  r: x - y <= 1,  x >= 0,  0 <= y <= 1. The float basis {x}
// with y at 0 has y1 = 0 and d_y = -2^-40; raising y raises x, which has no upper bound, so
// y runs to its own bound: a bound flip, no pivot, and x = 2.
TEST(ExactRepair, ABoxedColumnOfTheWrongSignFlipsToItsOtherBound) {
  const Model model =
      make_lp({{1.0, -1.0}}, {-kInfinity}, {1.0}, {0.0, -kTiny}, {0.0, 0.0}, {kInfinity, 1.0});
  const Solution solution = reported(model, {BasisStatus::kBasic, BasisStatus::kAtLower},
                                     {BasisStatus::kAtUpper}, {1.0, 0.0});
  const RepairResult repair = repair_basis_exact(model, solution);
  ASSERT_EQ(repair.verdict, ExactVerdict::kVerified) << repair.message;
  EXPECT_EQ(repair.bound_flips, 1);
  EXPECT_EQ(repair.dual_pivots + repair.primal_pivots, 0);
  EXPECT_EQ(repair.col_status[1], BasisStatus::kAtUpper);
  EXPECT_EQ(repair.col_value[0], 2.0);
  EXPECT_EQ(repair.col_value[1], 1.0);
}

TEST(ExactRepair, AnExactlyOptimalBasisIsLeftAlone) {
  // min x + y  s.t.  x + 2y >= 4, 2x + y >= 4: optimum x = y = 4/3 on both rows.
  const Model model = make_lp({{1.0, 2.0}, {2.0, 1.0}}, {4.0, 4.0}, {kInfinity, kInfinity},
                              {1.0, 1.0}, {0.0, 0.0}, {kInfinity, kInfinity});
  const Solution solution =
      reported(model, {BasisStatus::kBasic, BasisStatus::kBasic},
               {BasisStatus::kAtLower, BasisStatus::kAtLower}, {4.0 / 3.0, 4.0 / 3.0});
  const RepairResult repair = repair_basis_exact(model, solution);
  ASSERT_EQ(repair.verdict, ExactVerdict::kVerified) << repair.message;
  EXPECT_FALSE(repair.changed());
}

TEST(ExactRepair, AnInfeasibleModelFailsInsteadOfBeingRepaired) {
  // x + y = 1 with x <= 1 - 2^-40 and y <= 0: exactly infeasible, though the float basis {x}
  // at x = 1 misses by 2^-40 only. The dual simplex finds no entering variable.
  const Model model =
      make_lp({{1.0, 1.0}}, {1.0}, {1.0}, {0.0, 1.0}, {0.0, 0.0}, {1.0 - kTiny, 0.0});
  const Solution solution = reported(model, {BasisStatus::kBasic, BasisStatus::kAtLower},
                                     {BasisStatus::kFixed}, {1.0, 0.0});
  const RepairResult repair = repair_basis_exact(model, solution);
  EXPECT_EQ(repair.verdict, ExactVerdict::kFailed) << repair.message;
  EXPECT_FALSE(repair.changed());
  // The proof: one multiplier on r. Its combined row x + y - s = 0 has, over x <= 1 - 2^-40,
  // y <= 0 and s = 1, the greatest value -2^-40 < 0.
  ASSERT_EQ(repair.farkas_row.size(), 1U);
  EXPECT_EQ(repair.farkas_row[0].first, 0);
}

TEST(ExactRepair, NoBasisAtAllIsDeclinedNotFailed) {
  // An interior point answer without crossover labels every variable kUnknown (Netlib
  // maros-r7 under algorithm=auto): there is no basis to repair, which is a decline. It was
  // reported FAILED, "variable 0 is nonbasic at no finite bound".
  const Model model = primal_off_by_tiny();
  const Solution solution = reported(model, {BasisStatus::kUnknown, BasisStatus::kUnknown},
                                     {BasisStatus::kUnknown}, {1.0 - kTiny, kTiny});
  const RepairResult repair = repair_basis_exact(model, solution);
  EXPECT_EQ(repair.verdict, ExactVerdict::kDeclined) << repair.message;
  EXPECT_NE(repair.message.find("no basis status"), std::string::npos) << repair.message;
}

TEST(ExactRepair, SolveWithExactReportsTheRepairedBasisAndVerifiesIt) {
  // The same model through solve(): whatever basis the engine stops at, the reported one
  // must be exactly optimal, and the .sol fields must say how it got there. Without
  // presolve: postsolve hands back y labelled `fixed` at its interior value 2^-40, which no
  // exact module can read as a basis (a label question, not a tolerance one).
  const Model model = primal_off_by_tiny();
  Options options;
  options.set_bool("presolve", false);
  options.set_bool("log_to_console", false);
  options.set_bool("exact", true);
  options.set_bool("ranging", true);
  const Solution solution = solve(model, options);
  ASSERT_EQ(solution.status, SolveStatus::kOptimal);
  EXPECT_EQ(solution.exact_repair_status, Solution::ExactVerification::kVerified)
      << solution.exact_repair_message;
  EXPECT_EQ(solution.exact_status, Solution::ExactVerification::kVerified);
  EXPECT_EQ(solution.sensitivity_status, Solution::ExactVerification::kVerified);
  EXPECT_EQ(solution.col_value[1], kTiny);

  options.set_bool("exact_repair", false);
  const Solution unrepaired = solve(model, options);
  EXPECT_EQ(unrepaired.exact_repair_status, Solution::ExactVerification::kNotAttempted);
}

}  // namespace
}  // namespace sankhya::exact
