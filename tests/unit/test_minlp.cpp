// SPDX-License-Identifier: Apache-2.0
// SANKHYA - convex MINLP by NLP-based branch and bound (NLP stage 3).
//
// Every optimum here is found by ENUMERATION in the test itself - every integer point of the
// box, with the continuous part in closed form where there is one - so the reference shares
// no arithmetic with the solver: not its relaxations, not its KKT check, not its tree.

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "nlp/minlp_bnb.hpp"
#include "nlp/nlp_solve.hpp"
#include "nlp/nonlinear_model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::nlp {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

Options quiet() {
  Options o;
  o.set_bool("log_to_console", false);
  return o;
}

Model columns(Index n, double lo, double hi, bool integer) {
  Model base;
  base.resize_columns(n);
  std::fill(base.col_lower.begin(), base.col_lower.end(), lo);
  std::fill(base.col_upper.begin(), base.col_upper.end(), hi);
  if (integer) std::fill(base.col_type.begin(), base.col_type.end(), VarType::kInteger);
  base.matrix = SparseMatrix(0, n);
  base.matrix.finalize();
  return base;
}

void expect_optimal_at(const Solution& s, double want) {
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  // Within the MIP gap target (1e-4 relative, 1e-6 absolute) of the enumerated optimum.
  EXPECT_NEAR(s.objective, want, 1e-4 * std::max(1.0, std::fabs(want)));
  EXPECT_TRUE(std::isfinite(s.dual_bound));
}

TEST(Minlp, PureIntegerConvexAgainstEnumeration) {
  // min (x - 2.6)^2 + (y - 1.3)^2 + exp(0.1 x) s.t. x^2 + y^2 <= 10, x, y in {-5..5}.
  NonlinearModel m(columns(2, -5, 5, true));
  ExpressionGraph& g = m.graph;
  const ExprId x = g.variable(0), y = g.variable(1);
  m.objective = g.sum({g.power(g.subtract(x, g.constant(2.6)), 2.0),
                       g.power(g.subtract(y, g.constant(1.3)), 2.0),
                       g.exp(g.multiply(g.constant(0.1), x))});
  m.constraints.push_back({g.add(g.power(x, 2.0), g.power(y, 2.0)), -kInf, 10.0, ""});
  double best = kInf;
  for (int a = -5; a <= 5; ++a) {
    for (int b = -5; b <= 5; ++b) {
      if (a * a + b * b > 10) continue;
      best = std::min(best, std::pow(a - 2.6, 2) + std::pow(b - 1.3, 2) + std::exp(0.1 * a));
    }
  }
  const Solution s = solve_nlp(m, quiet());
  expect_optimal_at(s, best);
  EXPECT_EQ(s.col_value[0], std::round(s.col_value[0]));
  EXPECT_LE(s.col_value[0] * s.col_value[0] + s.col_value[1] * s.col_value[1], 10.0 + 1e-7);
  EXPECT_GT(s.nodes, 1) << "the relaxation optimum (2.6, 1.3) is fractional: it must branch";
}

TEST(Minlp, MixedBinaryWithLinearRowsAgainstEnumeration) {
  // min sum_i (x_i - a_i)^2 + c_i y_i  s.t. x_i - 3 y_i <= 0, sum y_i <= 2, 0 <= x_i <= 3,
  // y binary. For fixed y the x part is in closed form: x_i = clamp(a_i, 0, 3) when y_i = 1,
  // else 0 - so the optimum is the best subset of at most two switched-on items.
  const std::vector<double> a{2.5, 1.0, -0.5, 3.5};
  const std::vector<double> c{1.0, 0.4, 0.2, 2.0};
  const Index k = 4;
  Model base = columns(2 * k, 0.0, 3.0, false);
  for (Index i = 0; i < k; ++i) {
    base.col_type[static_cast<std::size_t>(k + i)] = VarType::kInteger;
    base.col_upper[static_cast<std::size_t>(k + i)] = 1.0;
    base.col_cost[static_cast<std::size_t>(k + i)] = c[static_cast<std::size_t>(i)];
  }
  base.resize_rows(k + 1);
  SparseMatrix rows(k + 1, 2 * k);
  for (Index i = 0; i < k; ++i) {
    rows.add_entry(i, i, 1.0);
    rows.add_entry(i, k + i, -3.0);
    rows.add_entry(k, k + i, 1.0);
    base.row_upper[static_cast<std::size_t>(i)] = 0.0;
  }
  base.row_upper[static_cast<std::size_t>(k)] = 2.0;
  rows.finalize();
  base.matrix = std::move(rows);
  NonlinearModel m(std::move(base));
  std::vector<ExprId> terms;
  for (Index i = 0; i < k; ++i) {
    terms.push_back(m.graph.power(
        m.graph.subtract(m.graph.variable(i), m.graph.constant(a[static_cast<std::size_t>(i)])),
        2.0));
  }
  m.objective = m.graph.sum(terms);
  double best = kInf;
  for (int mask = 0; mask < 16; ++mask) {
    if (__builtin_popcount(static_cast<unsigned>(mask)) > 2) continue;
    double f = 0.0;
    for (int i = 0; i < 4; ++i) {
      const auto u = static_cast<std::size_t>(i);
      const bool on = ((mask >> i) & 1) != 0;
      const double xi = on ? std::clamp(a[u], 0.0, 3.0) : 0.0;
      f += (xi - a[u]) * (xi - a[u]) + (on ? c[u] : 0.0);
    }
    best = std::min(best, f);
  }
  expect_optimal_at(solve_nlp(m, quiet()), best);
}

TEST(Minlp, MaximisingAConcaveObjectiveOverAConvexSet) {
  // maximize log(1 + x) + log(1 + y) s.t. x + 2 y <= 7, x^2 + y^2 <= 30, x, y in {0..10}.
  Model base = columns(2, 0, 10, true);
  base.sense = ObjSense::kMaximize;
  base.resize_rows(1);
  SparseMatrix row(1, 2);
  row.add_entry(0, 0, 1.0);
  row.add_entry(0, 1, 2.0);
  row.finalize();
  base.matrix = std::move(row);
  base.row_upper[0] = 7.0;
  NonlinearModel m(std::move(base));
  ExpressionGraph& g = m.graph;
  const ExprId x = g.variable(0), y = g.variable(1);
  m.objective = g.add(g.log(g.add(x, g.constant(1.0))), g.log(g.add(y, g.constant(1.0))));
  m.constraints.push_back({g.add(g.power(x, 2.0), g.power(y, 2.0)), -kInf, 30.0, ""});
  double best = -kInf;
  for (int a = 0; a <= 10; ++a) {
    for (int b = 0; b <= 10; ++b) {
      if (a + 2 * b <= 7 && a * a + b * b <= 30)
        best = std::max(best, std::log1p(a) + std::log1p(b));
    }
  }
  expect_optimal_at(solve_nlp(m, quiet()), best);
}

TEST(Minlp, AnIntegerInfeasibleConvexModelIsProvedInfeasible) {
  // x^2 + y^2 <= 0.8 admits only (0, 0) among integers, and x + y >= 1 excludes it.
  NonlinearModel m(columns(2, 0, 3, true));
  ExpressionGraph& g = m.graph;
  const ExprId x = g.variable(0), y = g.variable(1);
  m.objective = g.add(x, y);
  m.constraints.push_back({g.add(g.power(x, 2.0), g.power(y, 2.0)), -kInf, 0.8, ""});
  m.constraints.push_back({g.add(x, y), 1.0, kInf, ""});
  const Solution s = solve_nlp(m, quiet());
  EXPECT_EQ(s.status, SolveStatus::kInfeasible) << to_string(s.status) << ": " << s.message;
}

TEST(Minlp, ANonconvexModelIsRefusedAndAHeuristicRunIsNeverOptimal) {
  // sin is neither convex nor concave: no valid bound, so refused unless asserted.
  NonlinearModel m(columns(1, 0, 6, true));
  m.objective = m.graph.sin(m.graph.variable(0));
  Solution s = solve_nlp(m, quiet());
  EXPECT_EQ(s.status, SolveStatus::kNotSolved);
  EXPECT_NE(s.message.find("not proved convex"), std::string::npos) << s.message;
  Options asserted = quiet();
  asserted.set_bool("nlp_assume_convex", true);
  s = solve_nlp(m, asserted);
  EXPECT_NE(s.status, SolveStatus::kOptimal) << s.message;
  if (s.status == SolveStatus::kFeasible) {
    EXPECT_EQ(s.col_value[0], std::round(s.col_value[0]));
    EXPECT_NE(s.message.find("asserted"), std::string::npos) << s.message;
  }
}

}  // namespace
}  // namespace sankhya::nlp
