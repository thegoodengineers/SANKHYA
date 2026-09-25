// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the NLP interior point (NLP stage 2).
//
// What is asserted, and against what:
//   * HS071 by expression, against its published optimum 17.0140173 (Hock and Schittkowski,
//     "Test Examples for Nonlinear Programming Codes", LNEMS 187, Springer 1981, problem 71),
//     and labelled LOCALLY optimal because nothing proves it convex;
//   * a problem the composition rules prove convex is labelled `optimal`, with a bound;
//   * random convex QPs agree with the convex-QP engine through the frozen solve();
//   * infeasible models end `locally_infeasible`, never `optimal`;
//   * maximisation, a fixed column and a start outside the log's domain.
// The whole Hock-Schittkowski set read from data/nlp/hs is test_nlp_hock_schittkowski.cpp.

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "nlp/nlp_kkt_check.hpp"
#include "nlp/nlp_problem.hpp"
#include "nlp/nlp_solve.hpp"
#include "nlp/nonlinear_model.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::nlp {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

Options quiet() {
  Options o;
  o.set_bool("log_to_console", false);
  return o;
}

Model columns(Index n, double lo, double hi) {
  Model base;
  base.resize_columns(n);
  std::fill(base.col_lower.begin(), base.col_lower.end(), lo);
  std::fill(base.col_upper.begin(), base.col_upper.end(), hi);
  base.matrix = SparseMatrix(0, n);
  base.matrix.finalize();
  return base;
}

/// The answer's own KKT conditions, re-checked from its vectors at the project tolerances.
void expect_kkt(const NonlinearModel& model, const Solution& s) {
  NlpProblem p;
  ASSERT_EQ(NlpProblem::build(model, &p), "");
  const double sense = p.sense();
  std::vector<double> y(s.row_dual.size()), d(s.col_dual.size());
  for (std::size_t i = 0; i < y.size(); ++i) y[i] = sense * s.row_dual[i];
  for (std::size_t j = 0; j < d.size(); ++j) d[j] = sense * s.col_dual[j];
  const NlpKktReport r = check_nlp_kkt(p, s.col_value, y, d, tol::kPrimalFeasibility,
                                       tol::kDualFeasibility, tol::kComplementarity);
  EXPECT_TRUE(r.passed) << r.failed;
}

TEST(NlpIpm, Hs071ReachesThePublishedOptimumAndSaysItIsLocal) {
  NonlinearModel m(columns(4, 1.0, 5.0));
  m.base.col_cost[2] = 1.0;
  ExpressionGraph& g = m.graph;
  const ExprId x0 = g.variable(0), x1 = g.variable(1), x2 = g.variable(2), x3 = g.variable(3);
  m.objective = g.multiply(g.multiply(x0, x3), g.sum({x0, x1, x2}));
  m.constraints.push_back({g.multiply(g.multiply(x0, x1), g.multiply(x2, x3)), 25.0, kInf, ""});
  m.constraints.push_back(
      {g.sum({g.power(x0, 2), g.power(x1, 2), g.power(x2, 2), g.power(x3, 2)}), 40.0, 40.0,
       ""});
  m.start = {1, 5, 5, 1};
  const Solution s = solve_nlp(m, quiet());
  ASSERT_EQ(s.status, SolveStatus::kLocallyOptimal) << s.message;
  // The published optimum to the digits it is published with (9 significant).
  EXPECT_NEAR(s.objective, 17.0140173, 1e-6);
  EXPECT_NEAR(s.col_value[1], 4.7429994, 1e-5);
  EXPECT_NEAR(s.col_value[2], 3.8211503, 1e-5);
  EXPECT_NEAR(s.col_value[3], 1.3794082, 1e-5);
  EXPECT_TRUE(std::isinf(s.dual_bound)) << "a local optimum proves no bound";
  EXPECT_NE(s.message.find("LOCAL"), std::string::npos) << s.message;
  expect_kkt(m, s);
}

TEST(NlpIpm, AProvablyConvexProblemIsOptimalWithABound) {
  // min (x0 - 2)^2 + exp(x1) - x1 s.t. x0^2 + x1^2 <= 1: convex objective, convex set. The
  // optimum lies on the circle where the gradient is normal to it; checked by the KKT test
  // and by the objective being no larger than at any of 360 points of the circle.
  NonlinearModel m(columns(2, -kInf, kInf));
  ExpressionGraph& g = m.graph;
  const ExprId x0 = g.variable(0), x1 = g.variable(1);
  m.objective = g.sum({g.power(g.subtract(x0, g.constant(2.0)), 2.0), g.exp(x1), g.negate(x1)});
  m.constraints.push_back({g.add(g.power(x0, 2.0), g.power(x1, 2.0)), -kInf, 1.0, "disc"});
  const Solution s = solve_nlp(m, quiet());
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_EQ(s.dual_bound, s.objective);
  expect_kkt(m, s);
  for (int k = 0; k < 360; ++k) {
    const double t = k * std::numbers::pi / 180.0;
    const double f = std::pow(std::cos(t) - 2.0, 2.0) + std::exp(std::sin(t)) - std::sin(t);
    EXPECT_LE(s.objective, f + 1e-7);
  }
}

TEST(NlpIpm, ConvexQpsAgreeWithTheQpEngine) {
  std::mt19937 rng(20260927);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  for (int trial = 0; trial < 12; ++trial) {
    const Index n = 6, m = 3;
    Model qp = columns(n, -2.0, 3.0);
    for (Index j = 0; j < n; ++j) qp.col_cost[static_cast<std::size_t>(j)] = u(rng);
    // Q = L L^T + 0.1 I, lower triangle.
    std::vector<double> l(static_cast<std::size_t>(n * n), 0.0);
    for (Index i = 0; i < n; ++i) {
      for (Index j = 0; j <= i; ++j) l[static_cast<std::size_t>(i * n + j)] = u(rng);
    }
    SparseMatrix q(n, n);
    for (Index i = 0; i < n; ++i) {
      for (Index j = 0; j <= i; ++j) {
        double v = i == j ? 0.1 : 0.0;
        for (Index k = 0; k < n; ++k) {
          v += l[static_cast<std::size_t>(i * n + k)] * l[static_cast<std::size_t>(j * n + k)];
        }
        q.add_entry(i, j, v);
      }
    }
    q.finalize();
    qp.hessian = std::move(q);
    qp.resize_rows(m);
    SparseMatrix a(m, n);
    for (Index i = 0; i < m; ++i) {
      for (Index j = 0; j < n; ++j) a.add_entry(i, j, u(rng));
      qp.row_lower[static_cast<std::size_t>(i)] = -1.0;
      qp.row_upper[static_cast<std::size_t>(i)] = 1.0;
    }
    a.finalize();
    qp.matrix = std::move(a);
    // The reference is the proximal QP interior point (#490): the default first-order engine
    // stops at its own tolerance, and on these QPs its KKT gate declines to call that
    // optimal (measured: complementarity 6.7e-5), which is no reference for a 1e-6 check.
    Options reference_options = quiet();
    reference_options.set_string("qp_algorithm", "ipm");
    const Solution reference = solve(qp, reference_options);
    ASSERT_EQ(reference.status, SolveStatus::kOptimal) << reference.message;
    const NonlinearModel as_nlp(qp);
    const Solution s = solve_nlp(as_nlp, quiet());
    ASSERT_EQ(s.status, SolveStatus::kOptimal) << "trial " << trial << ": " << s.message;
    EXPECT_NEAR(s.objective, reference.objective,
                1e-6 * std::max(1.0, std::fabs(reference.objective)))
        << "trial " << trial;
    expect_kkt(as_nlp, s);
  }
}

TEST(NlpIpm, AnInfeasibleModelIsLocallyInfeasibleNeverOptimal) {
  {
    // x0^2 + x1^2 <= 1 and x0 + x1 >= 3: the disc never reaches the half-plane (the largest
    // x0 + x1 on the disc is sqrt(2)).
    NonlinearModel m(columns(2, -kInf, kInf));
    ExpressionGraph& g = m.graph;
    const ExprId x0 = g.variable(0), x1 = g.variable(1);
    m.objective = g.add(x0, g.multiply(g.constant(2.0), x1));
    m.constraints.push_back({g.add(g.power(x0, 2.0), g.power(x1, 2.0)), -kInf, 1.0, ""});
    m.constraints.push_back({g.add(x0, x1), 3.0, kInf, ""});
    const Solution s = solve_nlp(m, quiet());
    EXPECT_EQ(s.status, SolveStatus::kLocallyInfeasible)
        << to_string(s.status) << ": " << s.message;
    EXPECT_FALSE(claims_a_point(s));
    EXPECT_NE(s.message.find("not a proof"), std::string::npos) << s.message;
  }
  {
    // A nonconvex one: exp(x0) + x0^2 = 0.5 has no solution (the left side is at least
    // exp(x*) + x*^2 ~ 0.827 at its minimizer x* ~ -0.3517).
    NonlinearModel m(columns(1, -10.0, 10.0));
    ExpressionGraph& g = m.graph;
    const ExprId x0 = g.variable(0);
    m.constraints.push_back({g.add(g.exp(x0), g.power(x0, 2.0)), 0.5, 0.5, ""});
    const Solution s = solve_nlp(m, quiet());
    EXPECT_EQ(s.status, SolveStatus::kLocallyInfeasible)
        << to_string(s.status) << ": " << s.message;
  }
}

TEST(NlpIpm, MaximisationFixedColumnsAndAStartOutsideTheDomain) {
  // maximize log(x0) + log(x1) - x2 s.t. x0 + x1 <= 4, x2 fixed at 0.5: x0 = x1 = 2, and the
  // objective 2 log 2 - 0.5; the start (0, 0, 0) is where log is undefined, and the method
  // moves it inside the bounds (x >= 1e-3) before evaluating anything. The bound is positive
  // so the composition rules can prove log concave there: with x >= 0 its domain touches 0,
  // convexity is NOT proved, and the answer is (correctly) only locally_optimal.
  Model base = columns(3, 1e-3, kInf);
  base.col_lower[2] = base.col_upper[2] = 0.5;
  base.sense = ObjSense::kMaximize;
  base.resize_rows(1);
  SparseMatrix a(1, 3);
  a.add_entry(0, 0, 1.0);
  a.add_entry(0, 1, 1.0);
  a.finalize();
  base.matrix = std::move(a);
  base.row_upper[0] = 4.0;
  NonlinearModel m(std::move(base));
  ExpressionGraph& g = m.graph;
  m.objective = g.sum({g.log(g.variable(0)), g.log(g.variable(1)), g.negate(g.variable(2))});
  const Solution s = solve_nlp(m, quiet());
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;  // concave, maximised: convex
  EXPECT_NEAR(s.objective, 2.0 * std::log(2.0) - 0.5, 1e-7);
  EXPECT_NEAR(s.col_value[0], 2.0, 1e-6);
  EXPECT_EQ(s.col_value[2], 0.5) << "a fixed column does not move";
  // The row's multiplier. In minimisation space (f = -objective) stationarity in x0 reads
  // -1/x0 - y = 0, so y = -1/x0 = -0.5: negative, it prices the row's UPPER bound. The answer
  // reports multipliers in the model's own sense, -1 * y = 0.5: the rate at which the maximum
  // grows per unit of the right-hand side, d(2 log(r/2))/dr = 1/2 at r = 4.
  EXPECT_NEAR(s.row_dual[0], 0.5, 1e-6);
  expect_kkt(m, s);
}

TEST(NlpIpm, IntegerColumnsAreRefusedHere) {
  Model base = columns(1, 0.0, 3.0);
  base.col_type[0] = VarType::kInteger;
  NonlinearModel m(std::move(base));
  m.objective = m.graph.exp(m.graph.variable(0));
  const Solution s = solve_nlp(m, quiet());
  EXPECT_EQ(s.status, SolveStatus::kNotSolved);
  EXPECT_NE(s.message.find("MINLP"), std::string::npos) << s.message;
}

}  // namespace
}  // namespace sankhya::nlp
