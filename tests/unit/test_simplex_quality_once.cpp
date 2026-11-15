// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a scaled simplex solve measures its answer once, on the caller's model (#1158).
//
// Simplex::finish() used to measure the quality of every answer on the model it solved, and
// solve_with_scaling() then unscaled the point and measured it again on the caller's model,
// overwriting all of it. The first measurement is now skipped for the scaled attempt. What
// must hold afterwards is what held before: every quality field of the Solution a scaled
// solve returns is the one Solution::recompute_quality() computes on the CALLER'S model
// from the returned point and duals. So each solve here is re-measured by the test and
// compared field for field, to the bit: a path that came back unmeasured, or measured on
// the scaled model, differs.
//
// The paths: optimal (cold and warm, dual and primal), infeasible (no point), stopped at an
// iteration limit, costs below one (the rescaled-cost recursion), and scaling off (the
// solve that still measures in finish()).

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#include "simplex/primal_simplex.hpp"

namespace sankhya {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  return options;
}

/// A bounded LP, min c'x s.t. A x <= b, 0 <= x <= u, with entries spread over three decades
/// so the equilibration does something.
Model random_lp(std::uint32_t seed, double cost_scale) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> entry(-9, 9);
  std::uniform_int_distribution<int> decade(0, 2);
  const Index rows = 6;
  const Index cols = 9;
  Model m;
  m.sense = ObjSense::kMinimize;
  m.col_cost.resize(static_cast<std::size_t>(cols));
  for (auto& c : m.col_cost) c = cost_scale * static_cast<double>(entry(rng));
  m.col_lower.assign(static_cast<std::size_t>(cols), 0.0);
  m.col_upper.assign(static_cast<std::size_t>(cols), 7.0);
  m.col_type.assign(static_cast<std::size_t>(cols), VarType::kContinuous);
  m.row_lower.assign(static_cast<std::size_t>(rows), -kInf);
  m.row_upper.resize(static_cast<std::size_t>(rows));
  m.matrix.reset(rows, cols);
  for (Index i = 0; i < rows; ++i) {
    const double scale = std::pow(10.0, decade(rng));
    for (Index j = 0; j < cols; ++j) {
      const int a = entry(rng);
      if (a != 0) m.matrix.add_entry(i, j, scale * static_cast<double>(a));
    }
    m.row_upper[static_cast<std::size_t>(i)] = scale * 11.0;
  }
  m.matrix.finalize();
  return m;
}

void expect_measured_on(const Model& model, const Solution& solved, const char* what) {
  Solution again = solved;
  again.recompute_quality(model);
  EXPECT_EQ(solved.objective, again.objective) << what;
  EXPECT_EQ(solved.row_activity, again.row_activity) << what;
  EXPECT_EQ(solved.primal_infeasibility, again.primal_infeasibility) << what;
  EXPECT_EQ(solved.primal_infeasibility_scaled, again.primal_infeasibility_scaled) << what;
  EXPECT_EQ(solved.dual_infeasibility, again.dual_infeasibility) << what;
  EXPECT_EQ(solved.dual_infeasibility_scaled, again.dual_infeasibility_scaled) << what;
  EXPECT_EQ(solved.complementarity_violation, again.complementarity_violation) << what;
  EXPECT_EQ(solved.integrality_violation, again.integrality_violation) << what;
  // The gaps are infinite when the bound is: equal, or both not finite.
  EXPECT_TRUE(solved.absolute_gap == again.absolute_gap ||
              (!std::isfinite(solved.absolute_gap) && !std::isfinite(again.absolute_gap)))
      << what;
  EXPECT_TRUE(solved.relative_gap == again.relative_gap ||
              (!std::isfinite(solved.relative_gap) && !std::isfinite(again.relative_gap)))
      << what;
}

WarmStart basis_of(const Solution& solved) {
  WarmStart warm;
  warm.col_status = solved.col_status;
  warm.row_status = solved.row_status;
  return warm;
}

TEST(SimplexQualityOnce, AScaledSolveReturnsTheQualityOfTheCallersModel) {
  Logger logger(nullptr);
  int optimal = 0;
  for (std::uint32_t seed = 1; seed <= 30; ++seed) {
    for (const double cost_scale : {1.0, 1e-3}) {  // 1e-3: every cost below one in size
      const Model model = random_lp(seed, cost_scale);
      const Options options = quiet();
      const NodeScaling scaling = build_node_scaling(model, options);
      ASSERT_TRUE(scaling.valid);

      const Solution dual = solve_dual_simplex(model, options, logger, scaling);
      expect_measured_on(model, dual, "dual, cold");
      const Solution primal = solve_primal_simplex(model, options, logger, scaling);
      expect_measured_on(model, primal, "primal, cold");
      if (dual.status != SolveStatus::kOptimal) continue;
      ++optimal;

      // A child of a tree: one bound tightened, warm from the optimal basis.
      Model child = model;
      child.col_upper[0] = std::floor(dual.col_value[0]);
      const WarmStart warm = basis_of(dual);
      const Solution warm_dual =
          solve_dual_simplex(child, options, logger, scaling, nullptr, &warm);
      expect_measured_on(child, warm_dual, "dual, warm");

      // Stopped short: the point in hand is still measured.
      Options limited = quiet();
      limited.set_int("iteration_limit", 1);
      const Solution stopped = solve_dual_simplex(model, limited, logger, scaling);
      expect_measured_on(model, stopped, "dual, iteration limit");
    }
  }
  EXPECT_GT(optimal, 20);  // the paths above were exercised
}

TEST(SimplexQualityOnce, AnInfeasibleScaledSolveIsMeasuredToo) {
  Logger logger(nullptr);
  Model model = random_lp(5, 1.0);
  // x0 >= 100 against x0 <= 7: no point.
  model.col_lower[0] = 100.0;
  model.col_upper[0] = 100.0;
  model.row_lower[0] = -kInf;
  model.row_upper[0] = -1e6;  // and a row no point in the box satisfies
  const Options options = quiet();
  const NodeScaling scaling = build_node_scaling(model, options);
  const Solution dual = solve_dual_simplex(model, options, logger, scaling);
  EXPECT_NE(dual.status, SolveStatus::kOptimal);
  expect_measured_on(model, dual, "dual, infeasible");
}

TEST(SimplexQualityOnce, WithScalingOffTheSolveMeasuresItself) {
  Logger logger(nullptr);
  Options options = quiet();
  options.set_bool("scaling", false);
  for (std::uint32_t seed = 1; seed <= 10; ++seed) {
    const Model model = random_lp(seed, 1.0);
    const NodeScaling none = build_node_scaling(model, options);
    ASSERT_FALSE(none.valid);
    const Solution dual = solve_dual_simplex(model, options, logger, none);
    expect_measured_on(model, dual, "dual, unscaled");
  }
}

}  // namespace
}  // namespace sankhya
