// SPDX-License-Identifier: Apache-2.0
// SANKHYA - cut selection (#415).
//
// Selection removes cuts and never invents one, so it cannot make a cut invalid; what it can
// do wrong is take the wrong few, or take two that say the same thing. The hand cases pin
// the score's three terms, the parallelism test and the cap; the search-level test runs the
// root and tree rounds with a small cap against enumeration, because a round that took the
// wrong cuts still has to reach the same optimum.

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "mip/cut_selection.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::mip {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

/// Rows given densely; integer columns by flag; unit box.
Model model_of(const std::vector<std::vector<double>>& rows,
               const std::vector<double>& row_lower, const std::vector<double>& row_upper,
               const std::vector<double>& cost, const std::vector<bool>& integral,
               bool maximize = false) {
  Model m;
  const auto n = cost.size();
  m.sense = maximize ? ObjSense::kMaximize : ObjSense::kMinimize;
  m.col_cost = cost;
  m.col_lower.assign(n, 0.0);
  m.col_upper.assign(n, 1.0);
  m.col_type.assign(n, VarType::kContinuous);
  for (std::size_t j = 0; j < n; ++j) {
    if (integral[j]) m.col_type[j] = VarType::kInteger;
  }
  m.matrix.reset(static_cast<Index>(rows.size()), static_cast<Index>(n));
  for (std::size_t i = 0; i < rows.size(); ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      if (rows[i][j] != 0.0) {
        m.matrix.add_entry(static_cast<Index>(i), static_cast<Index>(j), rows[i][j]);
      }
    }
  }
  m.matrix.finalize();
  m.row_lower = row_lower;
  m.row_upper = row_upper;
  m.hessian.reset(static_cast<Index>(n), static_cast<Index>(n));
  m.hessian.finalize();
  return m;
}

Cut cut_of(const std::vector<double>& coeff, double rhs) {
  Cut c;
  c.coeff = coeff;
  c.rhs = rhs;
  return c;
}

TEST(CutSelection, TheScoreHasItsThreeTermsWhereTheyAreDefined) {
  // Objective (1, 0, 0), column 0 integer. The cut x0 <= 0 at the point (0.5, 0, 0) is
  // violated by 0.5 on a unit normal (efficacy 0.5), parallel to the objective (1), and
  // supported on integer columns only (1): score 0.5 + 0.1 + 0.1.
  const Model m = model_of({{1, 1, 1}}, {-kInf}, {2.0}, {1, 0, 0}, {true, false, false});
  const CutScore s = score_cut(m, {0.5, 0.0, 0.0}, cut_of({1, 0, 0}, 0.0));
  EXPECT_NEAR(s.efficacy, 0.5, 1e-12);
  EXPECT_NEAR(s.objective_parallelism, 1.0, 1e-12);
  EXPECT_NEAR(s.integer_support, 1.0, 1e-12);
  EXPECT_NEAR(s.score, 0.7, 1e-12);
  // A cut on the continuous columns, orthogonal to the objective, with a violation of 1 on
  // a normal of length 2: efficacy a half and nothing else.
  const CutScore t = score_cut(m, {0.0, 1.0, 1.0}, cut_of({0, 2, 0}, 1.0));
  EXPECT_NEAR(t.efficacy, 0.5, 1e-12);
  EXPECT_NEAR(t.objective_parallelism, 0.0, 1e-12);
  EXPECT_NEAR(t.integer_support, 0.0, 1e-12);
}

TEST(CutSelection, OfTwoNearParallelCutsOnlyTheBetterIsTaken) {
  // At (1, 1, 0.5): A = x0 + x1 <= 1 and B = x0 + 1.05 x1 <= 1 are nearly parallel (cosine
  // above 0.99); B is the more efficacious and is taken, A waits. C = x2 <= 0 is orthogonal
  // to both and is taken as well.
  const Model m = model_of({{1, 1, 1}}, {-kInf}, {3.0}, {0, 0, 0}, {true, true, true});
  const std::vector<double> point{1.0, 1.0, 0.5};
  const CutSelection chosen = select_cuts(
      m, point, {cut_of({1, 1, 0}, 1.0), cut_of({1, 1.05, 0}, 1.0), cut_of({0, 0, 1}, 0.0)}, 10,
      0.9);
  ASSERT_EQ(chosen.selected.size(), 2u);
  EXPECT_EQ(chosen.selected[0].coeff, (std::vector<double>{1, 1.05, 0}))
      << "the more efficacious of the parallel pair comes first";
  EXPECT_EQ(chosen.selected[1].coeff, (std::vector<double>{0, 0, 1}));
  ASSERT_EQ(chosen.deferred.size(), 1u);
  EXPECT_EQ(chosen.deferred[0].coeff, (std::vector<double>{1, 1, 0}))
      << "the parallel one is kept for a later round, not dropped";
  EXPECT_GT(chosen.selected_scores[0].score, chosen.selected_scores[1].score);
}

TEST(CutSelection, TheCapKeepsTheBestAndDefersTheRestInOrder) {
  // Four orthogonal unit cuts with efficacies 0.4, 0.1, 0.3, 0.2 and a cap of two.
  const Model m =
      model_of({{1, 1, 1, 1}}, {-kInf}, {4.0}, {0, 0, 0, 0}, {false, false, false, false});
  const std::vector<double> point{0.4, 0.1, 0.3, 0.2};
  const CutSelection chosen =
      select_cuts(m, point,
                  {cut_of({1, 0, 0, 0}, 0.0), cut_of({0, 1, 0, 0}, 0.0),
                   cut_of({0, 0, 1, 0}, 0.0), cut_of({0, 0, 0, 1}, 0.0)},
                  2, 0.9);
  ASSERT_EQ(chosen.selected.size(), 2u);
  EXPECT_EQ(chosen.selected[0].coeff, (std::vector<double>{1, 0, 0, 0}));
  EXPECT_EQ(chosen.selected[1].coeff, (std::vector<double>{0, 0, 1, 0}));
  ASSERT_EQ(chosen.deferred.size(), 2u);
  EXPECT_EQ(chosen.deferred[0].coeff, (std::vector<double>{0, 0, 0, 1}));
  EXPECT_EQ(chosen.deferred[1].coeff, (std::vector<double>{0, 1, 0, 0}));
}

TEST(CutSelection, ATieIsBrokenByTheCandidatesOwnOrder) {
  // Two identical scores: the first candidate wins, on every run.
  const Model m = model_of({{1, 1}}, {-kInf}, {2.0}, {0, 0}, {false, false});
  const CutSelection chosen =
      select_cuts(m, {0.5, 0.5}, {cut_of({0, 1}, 0.0), cut_of({1, 0}, 0.0)}, 1, 0.9);
  ASSERT_EQ(chosen.selected.size(), 1u);
  EXPECT_EQ(chosen.selected[0].coeff, (std::vector<double>{0, 1}));
}

/// Independent of everything under test: integral, inside the box, every row satisfied.
bool feasible(const Model& m, const std::vector<double>& x) {
  for (std::size_t j = 0; j < x.size(); ++j) {
    if (m.col_type[j] == VarType::kInteger && std::fabs(x[j] - std::round(x[j])) > 1e-9) {
      return false;
    }
    if (x[j] < m.col_lower[j] - 1e-9 || x[j] > m.col_upper[j] + 1e-9) return false;
  }
  for (Index i = 0; i < m.num_rows(); ++i) {
    double activity = 0.0;
    for (Index j = 0; j < m.num_cols(); ++j) {
      activity += m.matrix.at(i, j) * x[static_cast<std::size_t>(j)];
    }
    const auto u = static_cast<std::size_t>(i);
    if (activity < m.row_lower[u] - 1e-9 || activity > m.row_upper[u] + 1e-9) return false;
  }
  return true;
}

TEST(CutSelection, TheSearchWithASmallCapAgreesWithBruteForce) {
  // Root and tree cut rounds with every family on and only three cuts allowed per round,
  // on random covering / packing models, against enumeration of every 0/1 point. The cap
  // and the parallelism filter decide what goes in; the optimum must not move.
  std::mt19937 rng(415);
  std::uniform_int_distribution<int> coefficient(0, 4);
  std::uniform_int_distribution<int> cost_value(-6, 6);
  int compared = 0;
  for (int trial = 0; trial < 120; ++trial) {
    const std::size_t n = 6 + static_cast<std::size_t>(trial % 5);
    std::vector<std::vector<double>> rows(4, std::vector<double>(n));
    std::vector<double> lower(4, -kInf);
    std::vector<double> upper(4, kInf);
    for (std::size_t i = 0; i < 4; ++i) {
      double sum = 0.0;
      for (std::size_t j = 0; j < n; ++j) {
        rows[i][j] = coefficient(rng);
        sum += rows[i][j];
      }
      if (i % 2 == 0) {
        upper[i] = std::floor(sum / 2.0);
      } else {
        lower[i] = std::floor(sum / 3.0);
      }
    }
    std::vector<double> cost(n);
    for (double& c : cost) c = cost_value(rng);
    const Model m =
        model_of(rows, lower, upper, cost, std::vector<bool>(n, true), trial % 2 == 1);

    bool any = false;
    double best = 0.0;
    for (std::uint32_t mask = 0; mask < (1u << n); ++mask) {
      std::vector<double> x(n);
      for (std::size_t j = 0; j < n; ++j) x[j] = (mask >> j) & 1u ? 1.0 : 0.0;
      if (!feasible(m, x)) continue;
      double value = 0.0;
      for (std::size_t j = 0; j < n; ++j) value += cost[j] * x[j];
      if (!any || (m.sense == ObjSense::kMaximize ? value > best : value < best)) best = value;
      any = true;
    }

    Options options;
    options.set_bool("log_to_console", false);
    options.set_bool("presolve", false);
    options.set_bool("enable_root_cuts", true);
    options.set_int("tree_cut_depth", 4);
    options.set_int("cut_max_per_round", 3);
    options.set_double("cut_max_parallelism", 0.8);
    const Solution solved = solve(m, options);
    ++compared;
    if (!any) {
      EXPECT_EQ(solved.status, SolveStatus::kInfeasible) << "trial " << trial;
      continue;
    }
    ASSERT_EQ(solved.status, SolveStatus::kOptimal)
        << "trial " << trial << ": " << solved.message;
    EXPECT_NEAR(solved.objective, best, 1e-6) << "trial " << trial;
    EXPECT_TRUE(feasible(m, solved.col_value)) << "trial " << trial;
  }
  EXPECT_EQ(compared, 120);
}

}  // namespace
}  // namespace sankhya::mip
