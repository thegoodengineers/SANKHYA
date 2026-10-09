// SPDX-License-Identifier: Apache-2.0
// SANKHYA - convex MINLP by outer approximation (#528), `minlp_method=oa`.
//
// Every optimum is found by ENUMERATION in the test itself - every integer point of the box,
// the continuous part in closed form where there is one - so the reference shares no
// arithmetic with the solver: not its master, not its cuts, not its NLP. Each model is also
// solved by the NLP-based branch and bound (the default method), and the two must agree, which
// is a second, independent route to the same number.

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "nlp/nlp_solve.hpp"
#include "nlp/nonlinear_model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::nlp {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

Options with_method(const char* method) {
  Options o;
  o.set_bool("log_to_console", false);
  o.set_string("minlp_method", method);
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

/// `optimal` within the MIP gap target of the enumerated optimum, with a bound that is a bound:
/// finite, no better than the optimum (to the NLP's accuracy), and the same method's name.
void expect_oa_optimal_at(const Solution& s, double want, bool maximize = false) {
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_EQ(s.algorithm, "minlp-oa");
  EXPECT_NEAR(s.objective, want, 1e-4 * std::max(1.0, std::fabs(want)));
  ASSERT_TRUE(std::isfinite(s.dual_bound));
  const double slack = 1e-5 * std::max(1.0, std::fabs(want));
  if (maximize) {
    EXPECT_GE(s.dual_bound, want - slack) << "the bound of a maximisation is above the optimum";
  } else {
    EXPECT_LE(s.dual_bound, want + slack) << "the bound of a minimisation is below the optimum";
  }
}

void expect_methods_agree(const NonlinearModel& m, double want, bool maximize = false) {
  const Solution oa = solve_nlp(m, with_method("oa"));
  expect_oa_optimal_at(oa, want, maximize);
  const Solution bnb = solve_nlp(m, with_method("bnb"));
  ASSERT_EQ(bnb.status, SolveStatus::kOptimal) << bnb.message;
  EXPECT_EQ(bnb.algorithm, "minlp-nlp-bnb");
  EXPECT_NEAR(oa.objective, bnb.objective, 2e-4 * std::max(1.0, std::fabs(want)));
}

TEST(MinlpOa, TheDefaultMethodIsTheTreeAndOaIsOptIn) {
  EXPECT_EQ(Options().get_string("minlp_method"), "bnb");
  NonlinearModel m(columns(1, 0, 4, true));
  m.objective =
      m.graph.power(m.graph.subtract(m.graph.variable(0), m.graph.constant(2.4)), 2.0);
  Options defaults;
  defaults.set_bool("log_to_console", false);
  EXPECT_EQ(solve_nlp(m, defaults).algorithm, "minlp-nlp-bnb");
  EXPECT_EQ(solve_nlp(m, with_method("oa")).algorithm, "minlp-oa");
}

TEST(MinlpOa, PureIntegerConvexAgainstEnumeration) {
  // min (x - 2.6)^2 + (y - 1.3)^2 + exp(0.1 x) s.t. x^2 + y^2 <= 10, x, y in {-5..5}. The
  // relaxation's optimum (2.6, 1.3) is fractional, and the circle excludes many integer
  // assignments of the first master, which the feasibility cuts must cut off.
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
  expect_methods_agree(m, best);
  const Solution s = solve_nlp(m, with_method("oa"));
  EXPECT_EQ(s.col_value[0], std::round(s.col_value[0]));
  EXPECT_EQ(s.col_value[1], std::round(s.col_value[1]));
  EXPECT_LE(s.col_value[0] * s.col_value[0] + s.col_value[1] * s.col_value[1], 10.0 + 1e-7);
  EXPECT_NE(s.message.find("cut(s)"), std::string::npos) << s.message;
}

TEST(MinlpOa, MixedBinaryWithLinearRowsAgainstEnumeration) {
  // min sum_i (x_i - a_i)^2 + c_i y_i  s.t. x_i - 3 y_i <= 0, sum y_i <= 2, 0 <= x_i <= 3,
  // y binary. For fixed y the x part is in closed form, so the optimum is the best subset of
  // at most two switched-on items.
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
    if (std::popcount(static_cast<unsigned>(mask)) > 2) continue;
    double f = 0.0;
    for (int i = 0; i < 4; ++i) {
      const auto u = static_cast<std::size_t>(i);
      const bool on = ((mask >> i) & 1) != 0;
      const double xi = on ? std::clamp(a[u], 0.0, 3.0) : 0.0;
      f += (xi - a[u]) * (xi - a[u]) + (on ? c[u] : 0.0);
    }
    best = std::min(best, f);
  }
  expect_methods_agree(m, best);
}

TEST(MinlpOa, MaximisingAConcaveObjectiveOverAConvexSet) {
  // maximize log(1 + x) + log(1 + y) s.t. x + 2 y <= 7, x^2 + y^2 <= 30, x, y in {0..10}. The
  // epigraph is of the minimisation form (-f), and the answer is reported in the model's sense.
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
  expect_methods_agree(m, best, /*maximize=*/true);
}

TEST(MinlpOa, ALinearObjectiveWithAnOffsetNeedsNoEpigraph) {
  // maximize 3x + 2y + 7 s.t. x^2 + y^2 <= 20, x, y in {0..6}: a linear objective (no epigraph
  // column) over a convex nonlinear set, maximised, with an offset the bound must carry.
  Model base = columns(2, 0, 6, true);
  base.sense = ObjSense::kMaximize;
  base.col_cost = {3.0, 2.0};
  base.objective_offset = 7.0;
  NonlinearModel m(std::move(base));
  ExpressionGraph& g = m.graph;
  m.constraints.push_back(
      {g.add(g.power(g.variable(0), 2.0), g.power(g.variable(1), 2.0)), -kInf, 20.0, ""});
  double best = -kInf;
  for (int a = 0; a <= 6; ++a) {
    for (int b = 0; b <= 6; ++b) {
      if (a * a + b * b <= 20) best = std::max(best, 3.0 * a + 2.0 * b + 7.0);
    }
  }
  expect_methods_agree(m, best, /*maximize=*/true);
}

TEST(MinlpOa, AQuadraticObjectiveTermGoesThroughTheEpigraphToo) {
  // min x^2 + y^2 - 3.3 x - 1.7 y  (Q = 2 I, in the model's c'x + 0.5 x'Qx convention)
  // s.t. exp(0.3 x) + y <= 6, x, y in {0..5}.
  Model base = columns(2, 0, 5, true);
  base.col_cost = {-3.3, -1.7};
  base.hessian.reset(2, 2);
  base.hessian.add_entry(0, 0, 2.0);
  base.hessian.add_entry(1, 1, 2.0);
  base.hessian.finalize();
  NonlinearModel m(std::move(base));
  ExpressionGraph& g = m.graph;
  m.constraints.push_back(
      {g.add(g.exp(g.multiply(g.constant(0.3), g.variable(0))), g.variable(1)), -kInf, 6.0,
       ""});
  double best = kInf;
  for (int a = 0; a <= 5; ++a) {
    for (int b = 0; b <= 5; ++b) {
      if (std::exp(0.3 * a) + b > 6.0) continue;
      best = std::min(best, a * a + b * b - 3.3 * a - 1.7 * b);
    }
  }
  expect_methods_agree(m, best);
}

TEST(MinlpOa, AnAffineEqualityRowKeptAsANonlinearConstraintIsLinearizedExactly) {
  // min (x - 1.4)^2 + (y - 0.2)^2 s.t. x + y = 3 (an expression row with both bounds equal: its
  // left side is affine, the only kind both sides may be), x, y in {0..5}.
  NonlinearModel m(columns(2, 0, 5, true));
  ExpressionGraph& g = m.graph;
  const ExprId x = g.variable(0), y = g.variable(1);
  m.objective = g.add(g.power(g.subtract(x, g.constant(1.4)), 2.0),
                      g.power(g.subtract(y, g.constant(0.2)), 2.0));
  m.constraints.push_back({g.add(x, y), 3.0, 3.0, ""});
  double best = kInf;
  for (int a = 0; a <= 5; ++a) {
    const int b = 3 - a;
    if (b < 0 || b > 5) continue;
    best = std::min(best, std::pow(a - 1.4, 2) + std::pow(b - 0.2, 2));
  }
  expect_methods_agree(m, best);
}

TEST(MinlpOa, AnIntegerInfeasibleConvexModelIsProvedInfeasible) {
  // x^2 + y^2 <= 0.8 admits only (0, 0) among integers, and x + y >= 1 excludes it. The
  // relaxation is feasible, so this is proved by the feasibility cuts emptying the master.
  NonlinearModel m(columns(2, 0, 3, true));
  ExpressionGraph& g = m.graph;
  const ExprId x = g.variable(0), y = g.variable(1);
  m.objective = g.add(x, y);
  m.constraints.push_back({g.add(g.power(x, 2.0), g.power(y, 2.0)), -kInf, 0.8, ""});
  m.constraints.push_back({g.add(x, y), 1.0, kInf, ""});
  const Solution s = solve_nlp(m, with_method("oa"));
  EXPECT_EQ(s.status, SolveStatus::kInfeasible) << to_string(s.status) << ": " << s.message;
  EXPECT_FALSE(claims_a_point(s.status));
}

TEST(MinlpOa, ANonconvexModelIsRefusedAndAHeuristicRunIsNeverOptimal) {
  // sin is neither convex nor concave: no valid bound, so refused unless asserted - the same
  // gate as the tree, since a linearization of a nonconvex function is not a cut.
  NonlinearModel m(columns(1, 0, 6, true));
  m.objective = m.graph.sin(m.graph.variable(0));
  Solution s = solve_nlp(m, with_method("oa"));
  EXPECT_EQ(s.status, SolveStatus::kNotSolved);
  EXPECT_NE(s.message.find("not proved convex"), std::string::npos) << s.message;
  Options asserted = with_method("oa");
  asserted.set_bool("nlp_assume_convex", true);
  s = solve_nlp(m, asserted);
  EXPECT_NE(s.status, SolveStatus::kOptimal) << s.message;
  if (s.status == SolveStatus::kFeasible) {
    EXPECT_EQ(s.col_value[0], std::round(s.col_value[0]));
    EXPECT_NE(s.message.find("asserted"), std::string::npos) << s.message;
    EXPECT_FALSE(std::isfinite(s.dual_bound)) << "a heuristic run claims no bound";
  }
}

TEST(MinlpOa, ARoundLimitStopsWithTheBestPointAndNoOptimalClaim) {
  NonlinearModel m(columns(2, -5, 5, true));
  ExpressionGraph& g = m.graph;
  const ExprId x = g.variable(0), y = g.variable(1);
  m.objective = g.sum({g.power(g.subtract(x, g.constant(2.6)), 2.0),
                       g.power(g.subtract(y, g.constant(1.3)), 2.0)});
  m.constraints.push_back({g.add(g.power(x, 2.0), g.power(y, 2.0)), -kInf, 10.0, ""});
  Options o = with_method("oa");
  o.set_int("minlp_oa_max_iterations", 1);
  const Solution s = solve_nlp(m, o);
  EXPECT_NE(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_EQ(s.stopped_by, LimitReason::kIterations);
}

}  // namespace
}  // namespace sankhya::nlp
