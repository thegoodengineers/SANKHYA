// SPDX-License-Identifier: Apache-2.0
// SANKHYA - derivatives for the NLP engine (stage 1): sin and cos, the structural Hessian
// pattern, the compiled tapes, and the NLP form with the Hessian of the Lagrangian.
//
// Three independent references, so no one of them can hide a wrong rule:
//   * values DERIVED BY HAND in the comments (sin/cos, HS071's gradient, Jacobian and every
//     entry of its Lagrangian Hessian);
//   * central differences, compared relative to the size of the derivative;
//   * the graph's own evaluator (derivatives.cpp) against the compiled tape (tape.cpp) on
//     random expressions - they share only the per-node rules, so a bug in either sweep
//     shows up as a disagreement.

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "nlp/expression.hpp"
#include "nlp/nlp_problem.hpp"
#include "nlp/nonlinear_model.hpp"
#include "nlp/tape.hpp"

namespace sankhya::nlp {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

double dense_hessian(const std::vector<HessianEntry>& h, Index i, Index j) {
  for (const HessianEntry& e : h) {
    if ((e.row == i && e.col == j) || (e.row == j && e.col == i)) return e.value;
  }
  return 0.0;
}

TEST(NlpDerivatives, SinAndCosByHand) {
  // f = sin(x0) cos(x1).  df/dx0 = cos x0 cos x1, df/dx1 = -sin x0 sin x1,
  // d2f/dx0^2 = -sin x0 cos x1 = d2f/dx1^2, d2f/dx0 dx1 = -cos x0 sin x1.
  ExpressionGraph g(2);
  const ExprId f = g.multiply(g.sin(g.variable(0)), g.cos(g.variable(1)));
  const std::vector<double> x{0.7, -1.3};
  const double s0 = std::sin(x[0]), c0 = std::cos(x[0]), s1 = std::sin(x[1]),
               c1 = std::cos(x[1]);
  const Evaluation v = g.evaluate(f, x);
  ASSERT_TRUE(v.ok()) << v.message;
  EXPECT_NEAR(v.value, s0 * c1, 1e-15);
  SparseEntries grad;
  Evaluation err;
  ASSERT_TRUE(g.gradient(f, x, &grad, &err)) << err.message;
  ASSERT_EQ(grad.size(), 2u);
  EXPECT_NEAR(grad[0].second, c0 * c1, 1e-15);
  EXPECT_NEAR(grad[1].second, -s0 * s1, 1e-15);
  std::vector<HessianEntry> h;
  ASSERT_TRUE(g.hessian(f, x, &h, &err)) << err.message;
  EXPECT_NEAR(dense_hessian(h, 0, 0), -s0 * c1, 1e-15);
  EXPECT_NEAR(dense_hessian(h, 1, 1), -s0 * c1, 1e-15);
  EXPECT_NEAR(dense_hessian(h, 1, 0), -c0 * s1, 1e-15);
  // Constants fold: sin(0) is the constant 0, cos(0) the constant 1.
  EXPECT_EQ(g.node(g.sin(g.constant(0.0))).op, Op::kConstant);
  EXPECT_EQ(g.node(g.cos(g.constant(0.0))).value, 1.0);
  // Neither convex nor concave is claimed, and the degree is "not a polynomial".
  EXPECT_EQ(g.curvature(f, {-1, -1}, {1, 1}), Curvature::kUnknown);
  EXPECT_EQ(g.degree(g.sin(g.variable(0))), -1);
}

TEST(NlpDerivatives, HessianPatternOfKnownExpressions) {
  ExpressionGraph g(4);
  const ExprId x0 = g.variable(0), x1 = g.variable(1), x2 = g.variable(2), x3 = g.variable(3);
  using P = std::vector<std::pair<Index, Index>>;
  // x0*x1 + x2^2: the cross term and one square, nothing else.
  EXPECT_EQ(g.hessian_pattern(g.add(g.multiply(x0, x1), g.power(x2, 2.0))),
            (P{{1, 0}, {2, 2}}));
  // exp(x0 + x1): the whole 2x2 block, lower triangle.
  EXPECT_EQ(g.hessian_pattern(g.exp(g.add(x0, x1))), (P{{0, 0}, {1, 0}, {1, 1}}));
  // Linear: 3 x0 + x1 / 4 - x3 has no curvature at all.
  EXPECT_TRUE(g.hessian_pattern(g.sum({g.multiply(g.constant(3.0), x0),
                                       g.divide(x1, g.constant(4.0)), g.negate(x3)}))
                  .empty());
  // x0 / x1: d2/dx0 dx1 and d2/dx1^2, but no d2/dx0^2.
  EXPECT_EQ(g.hessian_pattern(g.divide(x0, x1)), (P{{1, 0}, {1, 1}}));
  // x0 * x0 is a square.
  EXPECT_EQ(g.hessian_pattern(g.multiply(x0, x0)), (P{{0, 0}}));
}

/// A random expression over `n` columns whose domain includes the box [0.5, 2]^n, built from
/// every operation, including sin and cos.
///
/// sin and cos take s / (1 + s^2), which lies in [-1/2, 1/2]. Measured while writing this: an
/// unbounded argument produced sin(exp(...)) at an argument near 1e65, where adjacent doubles
/// are ~1e49 apart and no central difference can resolve a period of 2 pi - the exact
/// derivative and the compiled tape agreed, and the DIFFERENCE was meaningless. Bounding the
/// argument keeps the difference oracle valid; it changes what is tested, not a tolerance.
ExprId random_expression(ExpressionGraph* g, std::mt19937* rng, Index n, int depth) {
  std::uniform_int_distribution<int> pick(0, depth <= 0 ? 1 : 10);
  std::uniform_int_distribution<Index> column(0, n - 1);
  std::uniform_real_distribution<double> value(0.5, 2.0);
  const auto sub = [&]() { return random_expression(g, rng, n, depth - 1); };
  const auto bounded = [&]() {
    const ExprId s = sub();
    return g->divide(s, g->add(g->power(s, 2.0), g->constant(1.0)));
  };
  switch (pick(*rng)) {
    case 0: return g->variable(column(*rng));
    case 1: return g->add(g->variable(column(*rng)), g->constant(value(*rng)));
    case 2: return g->add(sub(), sub());
    case 3: return g->multiply(sub(), sub());
    case 4: return g->divide(sub(), g->add(g->exp(sub()), g->constant(1.0)));
    case 5: return g->power(g->add(g->exp(sub()), g->constant(0.1)), value(*rng));
    case 6: return g->exp(g->multiply(g->constant(0.3), sub()));
    case 7: return g->log(g->add(g->exp(sub()), g->constant(0.5)));
    case 8: return g->sqrt(g->add(g->power(sub(), 2.0), g->constant(1.0)));
    case 9: return g->sin(bounded());
    default: return g->cos(g->subtract(bounded(), g->variable(column(*rng))));
  }
}

TEST(NlpDerivatives, CompiledTapeAgreesWithTheGraphAndWithDifferences) {
  std::mt19937 rng(20260926);
  std::uniform_real_distribution<double> point(0.5, 2.0);
  const Index n = 5;
  int checked = 0;
  for (int trial = 0; trial < 300; ++trial) {
    ExpressionGraph g(n);
    const ExprId f = random_expression(&g, &rng, n, 4);
    ASSERT_TRUE(g.invalid().empty()) << g.invalid();
    std::vector<double> x(static_cast<std::size_t>(n));
    for (double& v : x) v = point(rng);
    const Evaluation reference = g.evaluate(f, x);
    if (!reference.ok()) continue;  // overflowed; the generator allows it, rarely
    SparseEntries grad;
    std::vector<HessianEntry> hess;
    Evaluation err;
    ASSERT_TRUE(g.gradient(f, x, &grad, &err)) << err.message;
    ASSERT_TRUE(g.hessian(f, x, &hess, &err)) << err.message;

    const CompiledExpression tape(g, f);
    double value = 0.0;
    std::vector<double> tg;
    ASSERT_TRUE(tape.gradient(x, &value, &tg, &err)) << err.message;
    EXPECT_EQ(value, reference.value) << "the same rules in the same order: bit for bit";
    ASSERT_EQ(tg.size(), grad.size());
    for (std::size_t k = 0; k < grad.size(); ++k) {
      EXPECT_EQ(tape.variables()[k], grad[k].first);
      EXPECT_NEAR(tg[k], grad[k].second, 1e-12 * std::max(1.0, std::fabs(grad[k].second)));
    }
    std::vector<double> th;
    ASSERT_TRUE(tape.add_hessian(x, 1.0, &th, &err)) << err.message;
    // Every nonzero the graph finds is in the structural pattern, with the same value; and
    // every pattern entry the graph omits is zero on the tape.
    std::set<std::pair<Index, Index>> in_pattern(tape.hessian_pattern().begin(),
                                                 tape.hessian_pattern().end());
    for (const HessianEntry& e : hess) {
      ASSERT_TRUE(in_pattern.count({e.row, e.col}) == 1)
          << "(" << e.row << "," << e.col << ") missing from the pattern of " << g.to_string(f);
    }
    for (std::size_t k = 0; k < th.size(); ++k) {
      const auto [i, j] = tape.hessian_pattern()[k];
      const double want = dense_hessian(hess, i, j);
      EXPECT_NEAR(th[k], want, 1e-10 * std::max(1.0, std::fabs(want))) << g.to_string(f);
    }
    // The best of five steps, for the reason test_expression.cpp measured: one fixed step
    // tests the difference quotient's truncation, not the derivative.
    double gradient_error = 1e300, hessian_error = 1e300;
    bool any = false;
    for (const double step : {1e-3, 1e-4, 1e-5, 1e-6, 1e-7}) {
      const DerivativeCheck fd = check_derivatives(g, f, x, step);
      if (!fd.evaluated) continue;
      any = true;
      gradient_error = std::min(gradient_error, fd.gradient_error);
      hessian_error = std::min(hessian_error, fd.hessian_error);
    }
    if (any) {
      EXPECT_LT(gradient_error, 1e-6) << g.to_string(f);
      EXPECT_LT(hessian_error, 1e-5) << g.to_string(f);
      ++checked;
    }
  }
  EXPECT_GT(checked, 200) << "most random expressions were checked against differences";
}

/// Hock-Schittkowski problem 71 (HS071): minimize x0 x3 (x0 + x1 + x2) + x2 subject to
/// x0 x1 x2 x3 >= 25, x0^2 + x1^2 + x2^2 + x3^2 = 40, 1 <= x <= 5. Hock and Schittkowski,
/// "Test Examples for Nonlinear Programming Codes", Lecture Notes in Economics and
/// Mathematical Systems 187, Springer 1981, problem 71.
NonlinearModel hs071() {
  Model base;
  base.resize_columns(4);
  std::fill(base.col_lower.begin(), base.col_lower.end(), 1.0);
  std::fill(base.col_upper.begin(), base.col_upper.end(), 5.0);
  base.col_cost[2] = 1.0;  // the linear "+ x2" goes through c
  base.matrix = SparseMatrix(0, 4);
  base.matrix.finalize();
  NonlinearModel m(std::move(base));
  ExpressionGraph& g = m.graph;
  const ExprId x0 = g.variable(0), x1 = g.variable(1), x2 = g.variable(2), x3 = g.variable(3);
  m.objective = g.multiply(g.multiply(x0, x3), g.sum({x0, x1, x2}));
  m.constraints.push_back(
      {g.multiply(g.multiply(x0, x1), g.multiply(x2, x3)), 25.0, kInf, "prod"});
  m.constraints.push_back(
      {g.sum({g.power(x0, 2), g.power(x1, 2), g.power(x2, 2), g.power(x3, 2)}), 40.0, 40.0,
       "ball"});
  return m;
}

TEST(NlpDerivatives, Hs071LagrangianByHand) {
  const NonlinearModel model = hs071();
  NlpProblem p;
  ASSERT_EQ(NlpProblem::build(model, &p), "");
  ASSERT_EQ(p.num_variables(), 4);
  ASSERT_EQ(p.num_constraints(), 2);
  const std::vector<double> x{1.0, 4.743, 3.821, 1.379};
  const double a = x[0], b = x[1], c = x[2], d = x[3];
  Evaluation err;
  double f = 0.0;
  std::vector<double> grad;
  ASSERT_TRUE(p.objective_gradient(x, &f, &grad, &err)) << err.message;
  EXPECT_NEAR(f, a * d * (a + b + c) + c, 1e-13);
  // grad f = [d(2a+b+c), a d, a d + 1, a(a+b+c)].
  const std::vector<double> want_grad{d * (2 * a + b + c), a * d, a * d + 1.0, a * (a + b + c)};
  for (int j = 0; j < 4; ++j)
    EXPECT_NEAR(grad[static_cast<std::size_t>(j)], want_grad[static_cast<std::size_t>(j)],
                1e-13);

  std::vector<double> g_values;
  ASSERT_TRUE(p.constraints(x, &g_values, &err));
  EXPECT_NEAR(g_values[0], a * b * c * d, 1e-13);
  EXPECT_NEAR(g_values[1], a * a + b * b + c * c + d * d, 1e-13);

  std::vector<double> jac;
  ASSERT_TRUE(p.jacobian(x, &jac, &err));
  // Row 0: [bcd, acd, abd, abc]; row 1: [2a, 2b, 2c, 2d], columns ascending.
  const std::vector<double> want_jac{b * c * d, a * c * d, a * b * d, a * b * c,
                                     2 * a,     2 * b,     2 * c,     2 * d};
  ASSERT_EQ(jac.size(), want_jac.size());
  for (std::size_t k = 0; k < jac.size(); ++k) EXPECT_NEAR(jac[k], want_jac[k], 1e-13);

  // W = sigma d2f + l0 d2g0 + l1 d2g1, lower triangle, every entry by hand:
  //   d2f: (0,0)=2d (1,0)=d (2,0)=d (3,0)=2a+b+c (3,1)=a (3,2)=a
  //   d2g0: (1,0)=cd (2,0)=bd (3,0)=bc (2,1)=ad (3,1)=ac (3,2)=ab
  //   d2g1: 2 on the diagonal
  const double sigma = 0.5, l0 = -0.7, l1 = 1.3;
  std::vector<double> w;
  ASSERT_TRUE(p.hessian(x, sigma, {l0, l1}, &w, &err)) << err.message;
  double want[4][4] = {};
  want[0][0] = sigma * 2 * d + 2 * l1;
  want[1][0] = sigma * d + l0 * c * d;
  want[2][0] = sigma * d + l0 * b * d;
  want[3][0] = sigma * (2 * a + b + c) + l0 * b * c;
  want[1][1] = 2 * l1;
  want[2][1] = l0 * a * d;
  want[3][1] = sigma * a + l0 * a * c;
  want[2][2] = 2 * l1;
  want[3][2] = sigma * a + l0 * a * b;
  want[3][3] = 2 * l1;
  const auto& starts = p.hessian_starts();
  const auto& rows = p.hessian_rows();
  int entries = 0;
  for (Index j = 0; j < 4; ++j) {
    for (Index k = starts[static_cast<std::size_t>(j)];
         k < starts[static_cast<std::size_t>(j) + 1]; ++k) {
      const Index i = rows[static_cast<std::size_t>(k)];
      ASSERT_GE(i, j);
      EXPECT_NEAR(w[static_cast<std::size_t>(k)], want[i][j], 1e-12)
          << "(" << i << "," << j << ")";
      ++entries;
    }
  }
  EXPECT_EQ(entries, 10) << "the lower triangle of a dense 4x4";
}

TEST(NlpDerivatives, MaximisationQuadraticAndLinearRowsAgainstDifferences) {
  // maximize -(x0 - 1)^2 - 0.5 x'Qx + log(x1 + x2) + c'x subject to a linear row and two
  // nonlinear ones; every derivative the NLP form reports is checked against central
  // differences of the function it claims to differentiate.
  Model base;
  base.resize_columns(3);
  base.sense = ObjSense::kMaximize;
  base.col_cost = {0.3, -0.2, 0.1};
  base.objective_offset = 2.0;
  SparseMatrix q(3, 3);
  q.add_entry(0, 0, 2.0);
  q.add_entry(2, 1, 0.5);
  q.add_entry(2, 2, 1.0);
  q.finalize();
  base.hessian = std::move(q);
  base.resize_rows(1);
  SparseMatrix a(1, 3);
  a.add_entry(0, 0, 1.0);
  a.add_entry(0, 2, -2.0);
  a.finalize();
  base.matrix = std::move(a);
  base.row_upper[0] = 4.0;
  NonlinearModel m(std::move(base));
  ExpressionGraph& g = m.graph;
  const ExprId x0 = g.variable(0), x1 = g.variable(1), x2 = g.variable(2);
  m.objective =
      g.add(g.negate(g.power(g.subtract(x0, g.constant(1.0)), 2.0)), g.log(g.add(x1, x2)));
  m.constraints.push_back({g.multiply(g.exp(x0), g.sin(x1)), -kInf, 3.0, ""});
  m.constraints.push_back({g.divide(x2, g.add(x0, g.constant(2.0))), 0.1, 5.0, ""});
  NlpProblem p;
  ASSERT_EQ(NlpProblem::build(m, &p), "");
  EXPECT_EQ(p.sense(), -1.0);
  EXPECT_EQ(p.num_linear_rows(), 1);
  const std::vector<double> x{0.4, 1.1, 0.7};
  const std::vector<double> lambda{0.9, -0.4, 1.7};
  const double sigma = 1.3;
  const double h = 1e-6;
  Evaluation err;
  double f = 0.0;
  std::vector<double> grad;
  ASSERT_TRUE(p.objective_gradient(x, &f, &grad, &err));
  // The minimisation form of the model objective, by hand.
  const double model_objective =
      2.0 + 0.3 * x[0] - 0.2 * x[1] + 0.1 * x[2] +
      0.5 * (2.0 * x[0] * x[0] + 2 * 0.5 * x[1] * x[2] + x[2] * x[2]) -
      (x[0] - 1) * (x[0] - 1) + std::log(x[1] + x[2]);
  EXPECT_NEAR(f, -model_objective, 1e-14);

  // grad L = sigma grad f + J' lambda, by differences of f and g.
  const auto lagrangian_gradient = [&](const std::vector<double>& at) {
    double fv = 0.0;
    std::vector<double> gf, jv;
    EXPECT_TRUE(p.objective_gradient(at, &fv, &gf, &err));
    EXPECT_TRUE(p.jacobian(at, &jv, &err));
    std::vector<double> out(3, 0.0);
    for (std::size_t j = 0; j < 3; ++j) out[j] = sigma * gf[j];
    for (Index i = 0; i < 3; ++i) {
      for (Index k = p.jacobian_starts()[static_cast<std::size_t>(i)];
           k < p.jacobian_starts()[static_cast<std::size_t>(i) + 1]; ++k) {
        out[static_cast<std::size_t>(p.jacobian_columns()[static_cast<std::size_t>(k)])] +=
            lambda[static_cast<std::size_t>(i)] * jv[static_cast<std::size_t>(k)];
      }
    }
    return out;
  };
  std::vector<double> jac;
  ASSERT_TRUE(p.jacobian(x, &jac, &err));
  for (std::size_t j = 0; j < 3; ++j) {
    std::vector<double> plus = x, minus = x;
    plus[j] += h;
    minus[j] -= h;
    double fp = 0.0, fm = 0.0;
    ASSERT_TRUE(p.objective(plus, &fp, &err) && p.objective(minus, &fm, &err));
    EXPECT_NEAR(grad[j], (fp - fm) / (2 * h), 1e-7);
    std::vector<double> gp, gm;
    ASSERT_TRUE(p.constraints(plus, &gp, &err) && p.constraints(minus, &gm, &err));
    for (Index i = 0; i < 3; ++i) {
      double reported = 0.0;
      for (Index k = p.jacobian_starts()[static_cast<std::size_t>(i)];
           k < p.jacobian_starts()[static_cast<std::size_t>(i) + 1]; ++k) {
        if (p.jacobian_columns()[static_cast<std::size_t>(k)] == static_cast<Index>(j)) {
          reported = jac[static_cast<std::size_t>(k)];
        }
      }
      EXPECT_NEAR(reported,
                  (gp[static_cast<std::size_t>(i)] - gm[static_cast<std::size_t>(i)]) / (2 * h),
                  1e-7)
          << "row " << i << " column " << j;
    }
    // Column j of W by differences of grad L.
    const std::vector<double> lp = lagrangian_gradient(plus), lm = lagrangian_gradient(minus);
    std::vector<double> w;
    ASSERT_TRUE(p.hessian(x, sigma, lambda, &w, &err));
    for (std::size_t i = 0; i < 3; ++i) {
      double reported = 0.0;
      const auto r = static_cast<Index>(std::max(i, j)), c = static_cast<Index>(std::min(i, j));
      for (Index k = p.hessian_starts()[static_cast<std::size_t>(c)];
           k < p.hessian_starts()[static_cast<std::size_t>(c) + 1]; ++k) {
        if (p.hessian_rows()[static_cast<std::size_t>(k)] == r)
          reported = w[static_cast<std::size_t>(k)];
      }
      EXPECT_NEAR(reported, (lp[i] - lm[i]) / (2 * h), 1e-6) << "W(" << i << "," << j << ")";
    }
  }
}

TEST(NlpDerivatives, AnUndefinedPointIsAFailureNotANumber) {
  Model base;
  base.resize_columns(1);
  base.col_lower[0] = -kInf;
  base.matrix = SparseMatrix(0, 1);
  base.matrix.finalize();
  NonlinearModel m(std::move(base));
  m.objective = m.graph.log(m.graph.variable(0));
  NlpProblem p;
  ASSERT_EQ(NlpProblem::build(m, &p), "");
  double f = 0.0;
  Evaluation err;
  EXPECT_FALSE(p.objective({-1.0}, &f, &err));
  EXPECT_EQ(err.error, EvalError::kDomain) << err.message;
  std::vector<double> grad;
  EXPECT_FALSE(p.objective_gradient({0.0}, &f, &grad, &err));
  EXPECT_TRUE(p.objective({2.0}, &f, &err));
  EXPECT_NEAR(f, std::log(2.0), 1e-15);
}

}  // namespace
}  // namespace sankhya::nlp
