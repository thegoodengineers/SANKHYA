// SPDX-License-Identifier: Apache-2.0
// SANKHYA - one round of primal iterative refinement (#1082, src/core/lp_primal_refinement.hpp;
// Gleixner, Steffy and Wolter, INFORMS J. Computing 28(3), 2016, sec. 3).

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "core/lp_primal_refinement.hpp"
#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"

namespace sankhya {
namespace {

Model netlib(const std::string& name) {
  Model model;
  const std::string path =
      (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
       "data/netlib" / (name + ".mps"))
          .string();
  EXPECT_TRUE(io::read_model(path, &model).ok) << path;
  return model;
}

Options dual_simplex() {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("certified_gap", false);
  options.set_string("algorithm", "dual-simplex");
  return options;
}

TEST(PrimalRefinement, TheRefinedLpTakenBackIsTheLpsOptimumFromAPointOffIt) {
  const Model model = netlib("sc50a");
  const Solution optimum = solve(model, dual_simplex());
  ASSERT_EQ(optimum.status, SolveStatus::kOptimal) << optimum.message;

  // A point 1e-9 off the optimum, the way a float solve leaves one: Delta is set by it.
  std::vector<double> point = optimum.col_value;
  for (std::size_t j = 0; j < point.size(); ++j) {
    point[j] += 1e-9 * static_cast<double>(static_cast<int>(j % 3) - 1);
  }
  const PrimalRefinement refinement = refine_primal(model, point, 40);
  EXPECT_GT(refinement.max_violation, 0.0);
  EXPECT_LE(refinement.max_violation, 1e-6);
  EXPECT_GE(std::ldexp(refinement.max_violation, refinement.scale_bits), 1.0);
  EXPECT_LT(std::ldexp(refinement.max_violation, refinement.scale_bits), 2.0);

  // From the optimal basis the refined LP is solved; back in the LP it is the optimum.
  SolveControl control;
  control.start_col_status = optimum.col_status;
  control.start_row_status = optimum.row_status;
  Solution refined = solve(refinement.model, dual_simplex(), &control);
  ASSERT_EQ(refined.status, SolveStatus::kOptimal) << refined.message;
  const std::vector<double> duals = refined.row_dual;
  const Solution back = unrefine_primal(model, refinement, std::move(refined));
  EXPECT_EQ(back.row_dual, duals);  // the costs were kept, so the duals are the LP's
  double objective = model.objective_offset;
  for (std::size_t j = 0; j < back.col_value.size(); ++j) {
    objective += model.col_cost[j] * back.col_value[j];
  }
  EXPECT_NEAR(objective, optimum.objective, 1e-9 * std::fabs(optimum.objective));
  EXPECT_LE(back.primal_infeasibility, 1e-11);
}

TEST(PrimalRefinement, BoundsAreMeasuredFromThePointAndScaled) {
  // min x0 + x1, x0 + x1 >= 1, 0 <= x0 <= 4, x1 >= 0, about the point (0.25, 0.5): the row
  // is violated by 0.25, so Delta = 4.
  Model model;
  model.col_cost = {1.0, 1.0};
  model.col_lower = {0.0, 0.0};
  model.col_upper = {4.0, kInfinity};
  model.col_type.assign(2, VarType::kContinuous);
  model.row_lower = {1.0};
  model.row_upper = {kInfinity};
  model.matrix.reset(1, 2);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.finalize();
  const PrimalRefinement r = refine_primal(model, {0.25, 0.5}, 40);
  EXPECT_EQ(r.max_violation, 0.25);
  EXPECT_EQ(r.scale_bits, 2);
  EXPECT_EQ(r.model.col_lower, (std::vector<double>{-1.0, -2.0}));
  EXPECT_EQ(r.model.col_upper[0], 15.0);
  EXPECT_FALSE(is_finite_bound(r.model.col_upper[1]));
  EXPECT_EQ(r.model.row_lower, (std::vector<double>{1.0}));
  EXPECT_FALSE(is_finite_bound(r.model.row_upper[0]));
  EXPECT_EQ(r.model.col_cost, model.col_cost);
  // A point with no violation at all: Delta is the cap.
  EXPECT_EQ(refine_primal(model, {1.0, 0.0}, 40).scale_bits, 40);
}

}  // namespace
}  // namespace sankhya
