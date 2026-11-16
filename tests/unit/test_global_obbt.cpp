// SPDX-License-Identifier: Apache-2.0
// SANKHYA - optimality-based bound tightening at the root of the global search (#514,
// `global_obbt`).
//
// What must hold: a tightened box still holds every feasible point whose objective is at most
// the cutoff, the bounds do move where the relaxation says they can, and the search with the
// option on proves the same optima.

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "global/global_internal.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/qcqp.hpp"

namespace sankhya {
namespace {

struct RowSpec {
  std::vector<std::pair<Index, double>> linear;
  std::vector<QuadraticTerm> products;  ///< `row` is filled in by make_model
  double lower;
  double upper;
};

QcqpModel make_model(Index n, double lower, double upper, const std::vector<double>& cost,
                     const std::vector<RowSpec>& rows) {
  QcqpModel model;
  Model& lin = model.linear;
  const auto m = static_cast<Index>(rows.size());
  lin.resize_columns(n);
  lin.resize_rows(m);
  for (std::size_t j = 0; j < static_cast<std::size_t>(n); ++j) {
    lin.col_lower[j] = lower;
    lin.col_upper[j] = upper;
    lin.col_cost[j] = cost[j];
  }
  lin.matrix.reset(m, n);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    lin.row_lower[i] = rows[i].lower;
    lin.row_upper[i] = rows[i].upper;
    for (const auto& [column, value] : rows[i].linear) {
      lin.matrix.add_entry(static_cast<Index>(i), column, value);
    }
    for (QuadraticTerm term : rows[i].products) {
      term.row = static_cast<Index>(i);
      model.quadratic.push_back(term);
    }
  }
  lin.matrix.finalize(0.0);
  lin.hessian.reset(n, n);
  lin.hessian.finalize(0.0);
  return model;
}

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  return options;
}

TEST(GlobalObbt, MovesTheBoundsTheRelaxationProvesAndKeepsTheFeasiblePoint) {
  // x * y >= 4 and x + y <= 4 over [0, 10]^2: the only feasible point is (2, 2). Over the box
  // the envelope gives w <= 10 x and w <= 10 y, so the first LPs already show x, y in
  // [0.4, 3.6]; later ones, over the tighter box, may show more.
  const QcqpModel model = make_model(
      2, 0.0, 10.0, {0.0, 0.0},
      {{{}, {{0, 0, 1, 1.0}}, 4.0, kInfinity}, {{{0, 1.0}, {1, 1.0}}, {}, -kInfinity, 4.0}});
  const global::Problem problem = global::build_problem(model);
  global::Box box{problem.col_lower, problem.col_upper};
  const global::ObbtOutcome outcome =
      global::obbt(problem, kInfinity, quiet(), kInfinity, &box);
  EXPECT_FALSE(outcome.infeasible);
  EXPECT_GT(outcome.tightened, 0);
  for (std::size_t j = 0; j < 2; ++j) {
    EXPECT_GE(box.lower[j], 0.4 - 1e-6) << j;
    EXPECT_LE(box.upper[j], 3.6 + 1e-6) << j;
    EXPECT_LE(box.lower[j], 2.0);  // the feasible point is never cut off
    EXPECT_GE(box.upper[j], 2.0);
  }
}

TEST(GlobalObbt, NoFeasibleGridPointAtOrBelowTheCutoffLeavesTheBox) {
  // Random small non-convex models over [-1, 2]^3. The cutoff is a little above the best
  // feasible grid point, so several feasible points are at or below it; every one of them
  // must still be inside the box OBBT returns.
  std::mt19937 generator(1999);
  std::uniform_real_distribution<double> coefficient(-2.0, 2.0);
  std::uniform_real_distribution<double> inside(-1.0, 2.0);
  constexpr int kSteps = 18;
  int kept = 0;
  int tightened = 0;
  for (int trial = 0; trial < 30; ++trial) {
    std::vector<RowSpec> rows;
    const std::vector<double> anchor{inside(generator), inside(generator), inside(generator)};
    for (int r = 0; r < 2; ++r) {
      RowSpec row{{{0, coefficient(generator)},
                   {1, coefficient(generator)},
                   {2, coefficient(generator)}},
                  {{0, 0, 1, coefficient(generator)},
                   {0, 1, 2, coefficient(generator)},
                   {0, 2, 2, coefficient(generator)}},
                  -kInfinity,
                  0.0};
      double activity = 0.0;
      for (const auto& [j, a] : row.linear) activity += a * anchor[static_cast<std::size_t>(j)];
      for (const QuadraticTerm& t : row.products) {
        activity += t.value * anchor[static_cast<std::size_t>(t.first)] *
                    anchor[static_cast<std::size_t>(t.second)];
      }
      row.upper = activity + 0.1;
      rows.push_back(row);
    }
    const QcqpModel model = make_model(
        3, -1.0, 2.0, {coefficient(generator), coefficient(generator), coefficient(generator)},
        rows);
    std::vector<std::pair<double, std::vector<double>>> feasible;
    double best = kInfinity;
    for (int a = 0; a <= kSteps; ++a) {
      for (int b = 0; b <= kSteps; ++b) {
        for (int c = 0; c <= kSteps; ++c) {
          const std::vector<double> x{-1.0 + 3.0 * a / kSteps, -1.0 + 3.0 * b / kSteps,
                                      -1.0 + 3.0 * c / kSteps};
          const std::vector<double> at = model.row_activity(x);
          if (at[0] > model.linear.row_upper[0] || at[1] > model.linear.row_upper[1]) continue;
          const double objective = model.objective(x);
          best = std::min(best, objective);
          feasible.emplace_back(objective, x);
        }
      }
    }
    if (feasible.empty()) continue;
    const double cutoff = best + 0.3;
    const global::Problem problem = global::build_problem(model);
    global::Box box{problem.col_lower, problem.col_upper};
    const global::ObbtOutcome outcome = global::obbt(problem, cutoff, quiet(), kInfinity, &box);
    ASSERT_FALSE(outcome.infeasible) << "trial " << trial << ": a feasible point is below "
                                     << "the cutoff, so the box is not empty";
    tightened += outcome.tightened > 0 ? 1 : 0;
    for (const auto& [objective, x] : feasible) {
      if (objective > cutoff) continue;
      ++kept;
      for (std::size_t j = 0; j < 3; ++j) {
        EXPECT_GE(x[j], box.lower[j] - 1e-9) << "trial " << trial << " column " << j;
        EXPECT_LE(x[j], box.upper[j] + 1e-9) << "trial " << trial << " column " << j;
      }
    }
  }
  EXPECT_GT(kept, 30);
  EXPECT_GT(tightened, 0) << "the cutoff never moved a bound, so the test checks nothing";
}

TEST(GlobalObbt, TheSearchWithItProvesTheSameOptima) {
  // x * y >= 4, x + y <= 5 over [0, 10]^2, maximize x - y: the far end of the hyperbola,
  // x = 4, y = 1, objective 3. The root relaxation is far off it.
  QcqpModel model = make_model(
      2, 0.0, 10.0, {1.0, -1.0},
      {{{}, {{0, 0, 1, 1.0}}, 4.0, kInfinity}, {{{0, 1.0}, {1, 1.0}}, {}, -kInfinity, 5.0}});
  model.linear.sense = ObjSense::kMaximize;
  Options with = quiet();
  with.set_bool("global_obbt", true);
  const Solution plain = solve_global(model, quiet());
  const Solution tightened = solve_global(model, with);
  ASSERT_EQ(plain.status, SolveStatus::kOptimal) << plain.message;
  ASSERT_EQ(tightened.status, SolveStatus::kOptimal) << tightened.message;
  EXPECT_NEAR(tightened.objective, 3.0, 1e-3);
  EXPECT_NEAR(tightened.objective, plain.objective, 1e-3);
  EXPECT_GE(tightened.dual_bound, tightened.objective - 1e-9);  // maximize: bound above
  EXPECT_LE(tightened.primal_infeasibility, 1e-7);
  EXPECT_NE(tightened.message.find("OBBT"), std::string::npos) << tightened.message;
}

}  // namespace
}  // namespace sankhya
