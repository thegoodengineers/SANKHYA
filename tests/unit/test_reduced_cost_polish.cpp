// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the reduced-cost polish of a simplex optimum (#548).
//
// pilot87 ended "optimal" with a reduced cost 6.9e-8 on the wrong side, inside the 1e-7 dual
// tolerance, and 1.14e-6 above its exact optimum. The polish restarts the primal simplex
// from that basis at tol::kReducedCostPolish. Pinned here on a two-column LP whose optimum
// under a loose dual tolerance leaves a 1e-5 improvement untaken.
#include <limits>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

#include "simplex/primal_simplex.hpp"

namespace sankhya {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kEps = 1e-5;

// min -x - (1 + eps) y  s.t.  x + y <= 1,  0 <= x, y <= 1. Optimum y = 1, objective -(1+eps).
Model two_columns() {
  Model model;
  model.name = "polish";
  model.col_cost = {-1.0, -(1.0 + kEps)};
  model.col_lower = {0.0, 0.0};
  model.col_upper = {1.0, 1.0};
  model.col_type.assign(2, VarType::kContinuous);
  model.row_lower = {-kInf};
  model.row_upper = {1.0};
  model.matrix.reset(1, 2);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.finalize();
  model.hessian.reset(2, 2);
  model.hessian.finalize();
  return model;
}

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  return options;
}

// The vertex x = 1: optimal to a dual tolerance of 1e-4, with y's reduced cost -1e-5.
Solution loose_optimum(const Model& model, Logger& logger) {
  Options loose = quiet();
  loose.set_double("dual_feasibility_tolerance", 1e-4);
  loose.set_bool("scaling", false);
  WarmStart warm;
  warm.col_status = {BasisStatus::kBasic, BasisStatus::kAtLower};
  warm.row_status = {BasisStatus::kAtUpper};
  return solve_primal_simplex(model, loose, logger, NodeScaling{}, nullptr, &warm);
}

TEST(ReducedCostPolish, AnImprovementInsideTheDualToleranceIsTaken) {
  const Model model = two_columns();
  Logger logger(nullptr);
  const Solution loose = loose_optimum(model, logger);
  ASSERT_EQ(loose.status, SolveStatus::kOptimal) << loose.message;
  ASSERT_NEAR(loose.objective, -1.0, 1e-12);
  ASSERT_GT(loose.dual_infeasibility, tol::kReducedCostPolish);

  const Solution polished = polish_reduced_costs(model, loose, quiet(), logger);
  EXPECT_EQ(polished.status, SolveStatus::kOptimal) << polished.message;
  EXPECT_NEAR(polished.objective, -(1.0 + kEps), 1e-12);
  EXPECT_LE(polished.dual_infeasibility, tol::kReducedCostPolish);
  EXPECT_NE(polished.message.find("reduced-cost polish"), std::string::npos);
}

TEST(ReducedCostPolish, AnExactOptimumIsLeftAlone) {
  const Model model = two_columns();
  Logger logger(nullptr);
  const Solution exact = solve_primal_simplex(model, quiet(), logger);
  ASSERT_EQ(exact.status, SolveStatus::kOptimal) << exact.message;
  const Solution again = polish_reduced_costs(model, exact, quiet(), logger);
  EXPECT_EQ(again.iterations, exact.iterations);
  EXPECT_EQ(again.message, exact.message);
  EXPECT_EQ(again.objective, exact.objective);
}

TEST(ReducedCostPolish, ANonOptimalAnswerIsLeftAlone) {
  const Model model = two_columns();
  Logger logger(nullptr);
  Solution stopped = loose_optimum(model, logger);
  stopped.status = SolveStatus::kFeasible;
  const Solution again = polish_reduced_costs(model, stopped, quiet(), logger);
  EXPECT_EQ(again.status, SolveStatus::kFeasible);
  EXPECT_EQ(again.objective, stopped.objective);
}

}  // namespace
}  // namespace sankhya
