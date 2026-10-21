// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the conic interior point (#755): second-order cones and convex quadratic rows.
//
// Every optimum below is worked out by hand in its comment, and every answer is also held to
// the KKT conditions in the model's own terms - the row multipliers, the reduced costs, the
// cone duals - so a test passes on the conditions and not only on a matching number.

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "conic/conic_form.hpp"
#include "nlp/nlp_solve.hpp"
#include "nlp/nonlinear_model.hpp"
#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/qcqp.hpp"

#include "support/temp_file.hpp"

namespace sankhya {
namespace {

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  return options;
}

/// A model of n free continuous columns and no rows, to be filled in.
QcqpModel blank(Index n) {
  QcqpModel m;
  m.linear.resize_columns(n);
  for (Index j = 0; j < n; ++j) {
    m.linear.col_lower[static_cast<std::size_t>(j)] = -kInfinity;
    m.linear.col_names.push_back("x" + std::to_string(j));
  }
  m.linear.matrix.reset(0, n);
  m.linear.matrix.finalize();
  return m;
}

void add_row(QcqpModel* m, const std::vector<std::pair<Index, double>>& entries, double lower,
             double upper) {
  Model& lin = m->linear;
  const Index i = lin.num_rows();
  const Index n = lin.num_cols();
  lin.resize_rows(i + 1);
  lin.row_lower.back() = lower;
  lin.row_upper.back() = upper;
  lin.row_names.resize(static_cast<std::size_t>(i) + 1);
  lin.row_names.back() = "r" + std::to_string(i);
  // A matrix one row taller: the old entries copied, the new row's appended.
  SparseMatrix taller(i + 1, n);
  for (Index j = 0; j < n; ++j) {
    const ColumnView column = lin.matrix.column(j);
    for (Index p = 0; p < column.size; ++p)
      taller.add_entry(column.rows[p], j, column.values[p]);
  }
  for (const auto& [j, v] : entries) taller.add_entry(i, j, v);
  taller.finalize();
  lin.matrix = std::move(taller);
}

void fix(QcqpModel* m, Index j, double value) {
  m->linear.col_lower[static_cast<std::size_t>(j)] = value;
  m->linear.col_upper[static_cast<std::size_t>(j)] = value;
}

/// The threshold of every KKT condition below: the 1e-6 tools/verify_solution.py holds a
/// point's feasibility, its multipliers' signs and its complementarity products to, so a test
/// that passes here describes an answer the verifier accepts.
constexpr double kKktTolerance = 1e-6;

/// The KKT conditions of the model at the reported point: every row and bound feasible,
/// reduced costs with the right sign against the bounds (free columns: zero), row duals with
/// the right sign, and complementary slackness, all within kKktTolerance of the magnitudes
/// involved.
void expect_kkt(const QcqpModel& m, const Solution& s) {
  const Model& lin = m.linear;
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  const std::vector<double> activity = m.row_activity(s.col_value);
  const double sigma = lin.sense_multiplier();
  for (Index i = 0; i < lin.num_rows(); ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double scale = std::max(1.0, std::fabs(activity[u]));
    EXPECT_LE(lin.row_lower[u] - activity[u], kKktTolerance * scale) << "row " << i;
    EXPECT_LE(activity[u] - lin.row_upper[u], kKktTolerance * scale) << "row " << i;
    const double y = sigma * s.row_dual[u];
    if (!is_finite_bound(lin.row_lower[u])) {
      EXPECT_LE(y, kKktTolerance) << "row " << i;
    }
    if (!is_finite_bound(lin.row_upper[u])) {
      EXPECT_GE(y, -kKktTolerance) << "row " << i;
    }
    if (lin.row_lower[u] != lin.row_upper[u]) {
      const double slack =
          std::min(activity[u] - lin.row_lower[u], lin.row_upper[u] - activity[u]);
      if (std::isfinite(slack)) {
        EXPECT_LE(std::fabs(y) * slack, kKktTolerance * scale)
            << "row " << i << ": " << s.message;
      }
    }
  }
  std::vector<char> in_cone(static_cast<std::size_t>(lin.num_cols()), 0);
  for (const Cone& cone : m.cones) {
    for (const Index j : cone.columns) in_cone[static_cast<std::size_t>(j)] = 1;
  }
  for (Index j = 0; j < lin.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double x = s.col_value[u];
    EXPECT_LE(lin.col_lower[u] - x, kKktTolerance) << "column " << j;
    EXPECT_LE(x - lin.col_upper[u], kKktTolerance) << "column " << j;
    if (in_cone[u]) continue;
    const double d = sigma * s.col_dual[u];
    if (!is_finite_bound(lin.col_lower[u])) {
      EXPECT_LE(d, kKktTolerance) << "column " << j;
    }
    if (!is_finite_bound(lin.col_upper[u])) {
      EXPECT_GE(d, -kKktTolerance) << "column " << j;
    }
    if (lin.col_lower[u] != lin.col_upper[u]) {
      const double slack = std::min(x - lin.col_lower[u], lin.col_upper[u] - x);
      if (std::isfinite(slack)) {
        EXPECT_LE(std::fabs(d) * slack, kKktTolerance) << "column " << j << ": " << s.message;
      }
    }
  }
  // Each cone: x in K, the reduced costs in K* (the same cone), and x's = 0.
  for (const Cone& cone : m.cones) {
    const auto at = [&](const std::vector<double>& v, std::size_t i) {
      return v[static_cast<std::size_t>(cone.columns[i])];
    };
    const std::size_t head = cone.kind == Cone::Kind::kRotated ? 2 : 1;
    double xx = 0.0, dd = 0.0, xd = 0.0;
    for (std::size_t i = head; i < cone.columns.size(); ++i) {
      xx += at(s.col_value, i) * at(s.col_value, i);
      dd += sigma * at(s.col_dual, i) * sigma * at(s.col_dual, i);
    }
    for (std::size_t i = 0; i < cone.columns.size(); ++i) {
      xd += at(s.col_value, i) * sigma * at(s.col_dual, i);
    }
    if (cone.kind == Cone::Kind::kRotated) {
      EXPECT_GE(2.0 * at(s.col_value, 0) * at(s.col_value, 1) - xx, -kKktTolerance);
    } else {
      EXPECT_GE(at(s.col_value, 0) - std::sqrt(xx), -kKktTolerance);
    }
    // A member at a finite bound of its own has a reduced cost that is the cone's dual PLUS
    // that bound's multiplier, and the two cannot be told apart from the column's reduced
    // cost alone; tools/verify_solution.py skips such a cone's duals for the same reason.
    bool at_bound = false;
    for (std::size_t i = 0; i < cone.columns.size(); ++i) {
      const auto j = static_cast<std::size_t>(cone.columns[i]);
      at_bound = at_bound ||
                 (is_finite_bound(lin.col_lower[j]) &&
                  std::fabs(s.col_value[j] - lin.col_lower[j]) <= kKktTolerance) ||
                 (is_finite_bound(lin.col_upper[j]) &&
                  std::fabs(s.col_value[j] - lin.col_upper[j]) <= kKktTolerance);
    }
    if (at_bound) continue;
    if (cone.kind == Cone::Kind::kRotated) {
      EXPECT_GE(2.0 * sigma * at(s.col_dual, 0) * sigma * at(s.col_dual, 1) - dd,
                -kKktTolerance);
    } else {
      EXPECT_GE(sigma * at(s.col_dual, 0) - std::sqrt(dd), -kKktTolerance);
    }
    EXPECT_NEAR(xd, 0.0, kKktTolerance);
  }
}

// min -x1 - x2 subject to (t, x1, x2) in the quadratic cone, t = 1: the disc ||x|| <= 1, so
// x = (1, 1) / sqrt 2 and the objective is -sqrt 2.
TEST(Conic, TextbookSocp) {
  QcqpModel m = blank(3);
  m.linear.col_cost = {0.0, -1.0, -1.0};
  fix(&m, 0, 1.0);
  m.cones.push_back({Cone::Kind::kQuadratic, "disc", {0, 1, 2}});
  const Solution s = solve_conic(m, quiet());
  expect_kkt(m, s);
  EXPECT_NEAR(s.objective, -std::sqrt(2.0), 1e-7);
  EXPECT_NEAR(s.col_value[1], std::sqrt(0.5), 1e-6);
  EXPECT_NEAR(s.col_value[2], std::sqrt(0.5), 1e-6);
  EXPECT_EQ(s.algorithm, "conic-ipm");
}

// min p subject to 2 p q >= z^2, q = 1, z = 2: p = 2.
TEST(Conic, RotatedCone) {
  QcqpModel m = blank(3);
  m.linear.col_cost = {1.0, 0.0, 0.0};
  fix(&m, 1, 1.0);
  fix(&m, 2, 2.0);
  m.cones.push_back({Cone::Kind::kRotated, "r", {0, 1, 2}});
  const Solution s = solve_conic(m, quiet());
  expect_kkt(m, s);
  EXPECT_NEAR(s.objective, 2.0, 1e-7);
}

// min -x1 - 2 x2 subject to x1^2 + x2^2 <= 1 (a QCMATRIX row): x = (1, 2) / sqrt 5, objective
// -sqrt 5, and stationarity -1 - 2 y x1 = 0 gives the row's multiplier y = -sqrt 5 / 2.
TEST(Conic, ConvexQuadraticRow) {
  QcqpModel m = blank(2);
  m.linear.col_cost = {-1.0, -2.0};
  add_row(&m, {}, -kInfinity, 1.0);
  m.quadratic = {{0, 0, 0, 1.0}, {0, 1, 1, 1.0}};
  const Solution s = solve_conic(m, quiet());
  expect_kkt(m, s);
  EXPECT_NEAR(s.objective, -std::sqrt(5.0), 1e-7);
  EXPECT_NEAR(s.col_value[0], 1.0 / std::sqrt(5.0), 1e-6);
  EXPECT_NEAR(s.col_value[1], 2.0 / std::sqrt(5.0), 1e-6);
  EXPECT_NEAR(s.row_dual[0], -std::sqrt(5.0) / 2.0, 1e-6);
  EXPECT_NEAR(s.col_dual[0], 0.0, 1e-6);
  EXPECT_NEAR(s.col_dual[1], 0.0, 1e-6);
}

// The same disc written the concave way, -x1^2 - x2^2 >= -1, in a MAXIMIZATION: max x1 + 2 x2.
// Same point, objective +sqrt 5. In minimization space the lower side is priced by
// y = 1 / (2 x1) = sqrt 5 / 2 >= 0, and the model-sense multiplier of a maximization is its
// negative (Solution::row_dual's convention, as tools/verify_solution.py reads it).
TEST(Conic, ConcaveRowWithLowerSideMaximized) {
  QcqpModel m = blank(2);
  m.linear.sense = ObjSense::kMaximize;
  m.linear.col_cost = {1.0, 2.0};
  add_row(&m, {}, -1.0, kInfinity);
  m.quadratic = {{0, 0, 0, -1.0}, {0, 1, 1, -1.0}};
  const Solution s = solve_conic(m, quiet());
  expect_kkt(m, s);
  EXPECT_NEAR(s.objective, std::sqrt(5.0), 1e-7);
  EXPECT_NEAR(s.row_dual[0], -std::sqrt(5.0) / 2.0, 1e-6);
}

// min 0.5 (x1^2 + x2^2) subject to x1 + x2 >= 2 and x1^2 <= 1/2. Without the quadratic row
// the split is (1, 1); with it x1 = 1 / sqrt 2 and x2 = 2 - 1 / sqrt 2.
TEST(Conic, QuadraticObjectiveAndQuadraticRow) {
  QcqpModel m = blank(2);
  m.linear.hessian.reset(2, 2);
  m.linear.hessian.add_entry(0, 0, 1.0);
  m.linear.hessian.add_entry(1, 1, 1.0);
  m.linear.hessian.finalize();
  add_row(&m, {{0, 1.0}, {1, 1.0}}, 2.0, kInfinity);
  add_row(&m, {}, -kInfinity, 0.5);
  m.quadratic = {{1, 0, 0, 1.0}};
  const Solution s = solve_conic(m, quiet());
  expect_kkt(m, s);
  const double x1 = std::sqrt(0.5);
  const double x2 = 2.0 - x1;
  EXPECT_NEAR(s.col_value[0], x1, 1e-6);
  EXPECT_NEAR(s.col_value[1], x2, 1e-6);
  EXPECT_NEAR(s.objective, 0.5 * (x1 * x1 + x2 * x2), 1e-7);
}

// A planted indefinite row, x1 x2 <= 1, is refused with the LDL^T certificate in the message.
TEST(Conic, IndefiniteRowRefusedWithWitness) {
  QcqpModel m = blank(2);
  m.linear.col_cost = {-1.0, -1.0};
  m.linear.col_upper = {5.0, 5.0};
  add_row(&m, {}, -kInfinity, 1.0);
  m.quadratic = {{0, 0, 1, 1.0}};
  const std::vector<QuadraticRowClass> classes = classify_quadratic_rows(m);
  ASSERT_EQ(classes.size(), 1u);
  EXPECT_EQ(classes[0].verdict, QuadraticRowClass::Verdict::kNonConvex);
  ASSERT_EQ(classes[0].witness.size(), 2u);
  // z'Qz < 0 along the witness, checked against the row itself.
  const double z1 = classes[0].witness[0];
  const double z2 = classes[0].witness[1];
  EXPECT_LT(z1 * z2, 0.0);
  const Solution s = solve_conic(m, quiet());
  EXPECT_EQ(s.status, SolveStatus::kModelError);
  EXPECT_NE(s.message.find("not convex"), std::string::npos) << s.message;
  EXPECT_NE(s.message.find("witness"), std::string::npos) << s.message;
}

// A convex function with a finite lower side, 1 <= x1^2 + x2^2, is not a convex constraint.
TEST(Conic, RangedConvexRowRefused) {
  QcqpModel m = blank(2);
  add_row(&m, {}, 1.0, 4.0);
  m.quadratic = {{0, 0, 0, 1.0}, {0, 1, 1, 1.0}};
  const std::vector<QuadraticRowClass> classes = classify_quadratic_rows(m);
  ASSERT_EQ(classes.size(), 1u);
  EXPECT_EQ(classes[0].verdict, QuadraticRowClass::Verdict::kNonConvex);
  EXPECT_NE(classes[0].detail.find("LOWER"), std::string::npos) << classes[0].detail;
  EXPECT_EQ(solve_conic(m, quiet()).status, SolveStatus::kModelError);
}

// ||x|| <= 1 and x1 >= 2 cannot both hold.
TEST(Conic, InfeasibleSocp) {
  QcqpModel m = blank(3);
  fix(&m, 0, 1.0);
  m.linear.col_lower[1] = 2.0;
  m.cones.push_back({Cone::Kind::kQuadratic, "disc", {0, 1, 2}});
  const Solution s = solve_conic(m, quiet());
  EXPECT_EQ(s.status, SolveStatus::kInfeasible) << s.message;
  EXPECT_TRUE(s.col_value.empty());
}

// min -x1 subject to t >= |x1| with t free: x1 = t -> infinity.
TEST(Conic, UnboundedSocp) {
  QcqpModel m = blank(2);
  m.linear.col_cost = {0.0, -1.0};
  m.cones.push_back({Cone::Kind::kQuadratic, "ray", {0, 1}});
  const Solution s = solve_conic(m, quiet());
  EXPECT_EQ(s.status, SolveStatus::kUnbounded) << s.message;
}

// Q = F F' recovered from a rank-deficient lower triangle.
TEST(Conic, PivotedCholeskyRecoversTheMatrix) {
  // Q = [4 2 0; 2 1 0; 0 0 0] + e3 e3' * 9 -> rank 2.
  const std::vector<Index> rows = {0, 1, 1, 2};
  const std::vector<Index> cols = {0, 0, 1, 2};
  const std::vector<double> vals = {4.0, 2.0, 1.0, 9.0};
  const conic::DenseFactor f = conic::pivoted_cholesky(3, rows, cols, vals);
  EXPECT_EQ(f.rank, 2);
  const double q[3][3] = {{4, 2, 0}, {2, 1, 0}, {0, 0, 9}};
  for (Index i = 0; i < 3; ++i) {
    for (Index j = 0; j < 3; ++j) {
      double sum = 0.0;
      for (Index k = 0; k < f.rank; ++k) sum += f.at(i, k) * f.at(j, k);
      EXPECT_NEAR(sum, q[i][j], 1e-12);
    }
  }
}

// #1061 review: the rank stop judged every pivot against the LARGEST diagonal, so a small but
// real curvature was dropped. Each pivot is now judged at its own column's scale.
TEST(Conic, PivotedCholeskyKeepsSmallPivotsAtTheirOwnScale) {
  const conic::DenseFactor wide = conic::pivoted_cholesky(2, {0, 1}, {0, 1}, {1e6, 1e-7});
  ASSERT_EQ(wide.rank, 2);
  EXPECT_EQ(wide.negative_column, -1);
  EXPECT_NEAR(wide.at(0, 0) * wide.at(0, 0) + wide.at(0, 1) * wide.at(0, 1), 1e6, 1e-6);
  EXPECT_NEAR(wide.at(1, 0) * wide.at(1, 0) + wide.at(1, 1) * wide.at(1, 1), 1e-7, 1e-20);
  const conic::DenseFactor tiny = conic::pivoted_cholesky(1, {0}, {0}, {1e-14});
  ASSERT_EQ(tiny.rank, 1);
  EXPECT_NEAR(tiny.at(0, 0) * tiny.at(0, 0), 1e-14, 1e-27);
}

// The same scales in a model: min -x2 subject to 1e6 x1^2 + 1e-7 x2^2 <= 1. The optimum is
// x1 = 0, x2 = sqrt(1e7) = 3162.27766..., objective -sqrt(1e7). Dropping the 1e-7 pivot made
// the row a bound on x1 alone and the model unbounded.
TEST(Conic, BadlyScaledQuadraticRowKeepsItsSmallCurvature) {
  QcqpModel m = blank(2);
  m.linear.col_cost = {0.0, -1.0};
  add_row(&m, {}, -kInfinity, 1.0);
  m.quadratic = {{0, 0, 0, 1e6}, {0, 1, 1, 1e-7}};
  const Solution s = solve_conic(m, quiet());
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_NEAR(s.objective, -std::sqrt(1e7), 1e-7 * std::sqrt(1e7)) << s.message;
  EXPECT_NEAR(s.col_value[0], 0.0, 1e-6);
}

// #1061 review, item 6: the certificate tests read a ray off the iterate once kappa > tau.
// A feasible model with a large optimum - min t subject to t >= |x|, x = 1e9 - has a small
// tau throughout, and before the ray residual was weighed by the data's scale it reported
// "primal infeasible" at iteration 4 (a dual ray with residual 6e-9, no proof at |b| = 1e9).
// What this test holds: no certificate on a feasible model, and an optimal claim only with
// the right objective.
// ponytail: the embedding itself stops at its mu floor on this model (numerical_error, tau
// 2e-9 against kappa 6e-8) - an optimum 1e9 times the data's scale; equilibrating the conic
// form (Ruiz, as the LP path does) is the upgrade that would let it finish optimal.
TEST(Conic, LargeOptimumIsNotMistakenForARay) {
  QcqpModel m = blank(2);
  m.linear.col_cost = {1.0, 0.0};
  add_row(&m, {{1, 1.0}}, 1e9, 1e9);
  m.cones.push_back({Cone::Kind::kQuadratic, "abs", {0, 1}});
  const Solution s = solve_conic(m, quiet());
  EXPECT_NE(s.status, SolveStatus::kInfeasible) << s.message;
  EXPECT_NE(s.status, SolveStatus::kUnbounded) << s.message;
  if (s.status == SolveStatus::kOptimal) {
    EXPECT_NEAR(s.objective, 1e9, 1e9 * 1e-8) << s.message;
  }
}

// ---- a Haverly-style blending model with a convex quadratic blending row -------------------
//
// Crudes A (sulphur 3 %, cost 6) and B (1 %, cost 16) are blended into X (price 9, demand
// up to 100, sulphur <= 2.5 %) and Y (price 15, demand up to 200, sulphur <= 1.5 %): the
// linear part of Haverly's case 1 with the pool removed. The quadratic row is a budget on the
// SQUARED sulphur mass sent to the two products, (3 fAX + fBX)^2 + (3 fAY + fBY)^2 <= 150^2 -
// positive semidefinite of rank 2, a convex constraint and not a bilinear pooling balance.
// Columns: fAX, fBX, fAY, fBY. Maximize profit.
QcqpModel blending() {
  QcqpModel m = blank(4);
  m.linear.sense = ObjSense::kMaximize;
  m.linear.col_cost = {9.0 - 6.0, 9.0 - 16.0, 15.0 - 6.0, 15.0 - 16.0};
  // Bounded above by the demands, which the global method needs of a column in a product.
  m.linear.col_lower = {0.0, 0.0, 0.0, 0.0};
  m.linear.col_upper = {100.0, 100.0, 200.0, 200.0};
  add_row(&m, {{0, 1.0}, {1, 1.0}}, -kInfinity, 100.0);            // demand X
  add_row(&m, {{2, 1.0}, {3, 1.0}}, -kInfinity, 200.0);            // demand Y
  add_row(&m, {{0, 3.0 - 2.5}, {1, 1.0 - 2.5}}, -kInfinity, 0.0);  // X sulphur <= 2.5
  add_row(&m, {{2, 3.0 - 1.5}, {3, 1.0 - 1.5}}, -kInfinity, 0.0);  // Y sulphur <= 1.5
  add_row(&m, {}, -kInfinity, 150.0 * 150.0);                      // sulphur mass budget
  const Index r = 4;
  m.quadratic = {{r, 0, 0, 9.0}, {r, 0, 1, 6.0}, {r, 1, 1, 1.0},
                 {r, 2, 2, 9.0}, {r, 2, 3, 6.0}, {r, 3, 3, 1.0}};
  return m;
}

TEST(Conic, BlendingModelConicAndGlobalAgree) {
  const QcqpModel m = blending();
  const Solution conic = solve_conic(m, quiet());
  expect_kkt(m, conic);
  // Without the budget the blends would be X = 100 at 2.5 % and Y = 200 at 1.5 %, a sulphur
  // mass of 250 + 300: the budget 150^2 on the sum of squares binds, so the row is active.
  EXPECT_GT(std::fabs(conic.row_dual[4]), 1e-6);
  Options global = quiet();
  global.set_string("nonconvex", "global");
  global.set_double("time_limit", 120.0);
  const Solution bnb = solve_global(m, global);
  ASSERT_EQ(bnb.status, SolveStatus::kOptimal) << bnb.message;
  EXPECT_NEAR(bnb.objective, conic.objective, 1e-4 * std::max(1.0, std::fabs(conic.objective)));
}

// ---- random convex QCQPs against the NLP engine and the global method -----------------------

/// The quadratic row model as a NonlinearModel: the quadratic rows become expressions over
/// the same columns, with their bounds; the linear rows stay linear.
nlp::NonlinearModel as_nlp(const QcqpModel& m) {
  Model base = m.linear;
  std::vector<char> quadratic(static_cast<std::size_t>(base.num_rows()), 0);
  for (const QuadraticTerm& t : m.quadratic) quadratic[static_cast<std::size_t>(t.row)] = 1;
  nlp::NonlinearModel nl(base);
  for (Index i = 0; i < base.num_rows(); ++i) {
    const auto u = static_cast<std::size_t>(i);
    if (!quadratic[u]) continue;
    std::vector<nlp::ExprId> terms;
    for (Index j = 0; j < base.num_cols(); ++j) {
      const double a = base.matrix.at(i, j);
      if (a != 0.0)
        terms.push_back(nl.graph.multiply(nl.graph.constant(a), nl.graph.variable(j)));
    }
    for (const QuadraticTerm& t : m.quadratic) {
      if (t.row != i) continue;
      terms.push_back(nl.graph.multiply(
          nl.graph.constant(t.value),
          nl.graph.multiply(nl.graph.variable(t.first), nl.graph.variable(t.second))));
    }
    nl.constraints.push_back(
        {nl.graph.sum(terms), base.row_lower[u], base.row_upper[u], "q" + std::to_string(i)});
    nl.base.row_lower[u] = -kInfinity;  // the linear part now lives in the expression
    nl.base.row_upper[u] = kInfinity;
  }
  return nl;
}

QcqpModel random_qcqp(unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  const Index n = 4;
  QcqpModel m = blank(n);
  for (Index j = 0; j < n; ++j) {
    m.linear.col_lower[static_cast<std::size_t>(j)] = -2.0;
    m.linear.col_upper[static_cast<std::size_t>(j)] = 2.0;
    m.linear.col_cost[static_cast<std::size_t>(j)] = unit(gen);
  }
  add_row(&m, {{0, 1.0}, {1, 1.0}, {2, 1.0}, {3, 1.0}}, -kInfinity, 3.0);
  for (Index r = 0; r < 2; ++r) {
    // Q = G G' with G n x 2, the linear part random, the side chosen so that x = 0 is
    // strictly feasible.
    double g[4][2];
    for (auto& row : g) {
      for (double& e : row) e = unit(gen);
    }
    std::vector<std::pair<Index, double>> linear;
    for (Index j = 0; j < n; ++j) linear.emplace_back(j, unit(gen));
    add_row(&m, linear, -kInfinity, 1.0 + 0.5 * r);
    const Index row = m.linear.num_rows() - 1;
    for (Index a = 0; a < n; ++a) {
      for (Index b = a; b < n; ++b) {
        double q = 0.0;
        for (int k = 0; k < 2; ++k) q += g[a][k] * g[b][k];
        m.quadratic.push_back({row, a, b, a == b ? q : 2.0 * q});
      }
    }
  }
  return m;
}

TEST(Conic, RandomConvexQcqpsAgreeWithTheNlpEngineAndTheGlobalMethod) {
  for (unsigned seed = 1; seed <= 8; ++seed) {
    const QcqpModel m = random_qcqp(seed);
    for (const QuadraticRowClass& c : classify_quadratic_rows(m)) {
      ASSERT_EQ(c.verdict, QuadraticRowClass::Verdict::kConvex)
          << "seed " << seed << ": " << c.detail;
    }
    const Solution conic = solve_conic(m, quiet());
    expect_kkt(m, conic);
    const double scale = std::max(1.0, std::fabs(conic.objective));

    const Solution nlp = nlp::solve_nlp(as_nlp(m), quiet());
    ASSERT_TRUE(nlp.status == SolveStatus::kOptimal ||
                nlp.status == SolveStatus::kLocallyOptimal)
        << "seed " << seed << ": " << nlp.message;
    EXPECT_NEAR(nlp.objective, conic.objective, 1e-5 * scale) << "seed " << seed;

    Options global = quiet();
    global.set_string("nonconvex", "global");
    global.set_double("time_limit", 120.0);
    const Solution bnb = solve_global(m, global);
    ASSERT_EQ(bnb.status, SolveStatus::kOptimal) << "seed " << seed << ": " << bnb.message;
    EXPECT_NEAR(bnb.objective, conic.objective, 1e-4 * scale) << "seed " << seed;
  }
}

// ---- the MPS extension: CSECTION (MOSEK)
// ------------------------------------------------------

// The disc of TextbookSocp written as MOSEK's MPS cone section.
TEST(Conic, CsectionIsReadAndSolved) {
  const char* const text =
      "NAME disc\n"
      "ROWS\n"
      " N obj\n"
      " E tfix\n"
      "COLUMNS\n"
      " t obj 0 tfix 1\n"
      " x1 obj -1\n"
      " x2 obj -1\n"
      "RHS\n"
      " rhs tfix 1\n"
      "BOUNDS\n"
      " FR bnd t\n FR bnd x1\n FR bnd x2\n"
      "CSECTION disc 0.0 QUAD\n"
      " t\n x1\n x2\n"
      "ENDATA\n";
  const testing::TempFile file(text);
  QcqpModel m;
  const io::ReadResult result = io::read_qcqp_mps(file.path(), &m);
  ASSERT_TRUE(result.ok) << result.error;
  ASSERT_EQ(m.cones.size(), 1u);
  EXPECT_EQ(m.cones[0].kind, Cone::Kind::kQuadratic);
  EXPECT_EQ(m.cones[0].columns, (std::vector<Index>{0, 1, 2}));
  const Solution s = solve_conic(m, quiet());
  expect_kkt(m, s);
  EXPECT_NEAR(s.objective, -std::sqrt(2.0), 1e-7);
  // Without a sink the plain reader refuses the section by name, as it refuses QCMATRIX.
  Model plain;
  const io::ReadResult refused = io::read_mps(file.path(), &plain);
  EXPECT_FALSE(refused.ok);
  EXPECT_NE(refused.error.find("CSECTION"), std::string::npos) << refused.error;
}

}  // namespace
}  // namespace sankhya
