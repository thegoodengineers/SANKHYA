// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the two root causes behind #590's QP answers the verifier rejected.
//
// dpklo1: the solver was right and the independent verifier misread the file. Its RHS vector
// is named "1" and it has a row "1", so a reader that asks "does the first token name a row?"
// reads "1 77 3.577" as row 1 = 77. The C++ reader decides by the token count; this pins it on
// a reduced copy of the file's shape, end to end through solve() and the in-process KKT check
// (the verifier side is pinned in tools/test_verify_solution_names.py).
//
// stcqp1 and stcqp2: the point was right and the duals were not, because postsolve recovered
// the dual of a singleton row presolve had turned into a bound from c_j, the LP's price, and
// not from (c + Q x)_j. A two-row QP with one singleton row on a quadratic column shows it.

#include <cmath>
#include <string>

#include <gtest/gtest.h>

#include "core/kkt_check.hpp"
#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#include "support/temp_file.hpp"

namespace sankhya {
namespace {

using testing::TempFile;

// min x1^2/2 + x2^2/2 + x3  s.t.  x1 + x2 = 4,  x2 + x3 = 7,  x1 free,  x2 <= 10.
// Optimum x = (1.5, 2.5, 4.5), objective 8.75, row duals (1.5, 1). The vector names collide
// with row and column names exactly as dpklo1's do.
constexpr const char* kDpklo1Like =
    "NAME          DPKLO1-LIKE\n"
    "ROWS\n"
    "  E        1\n"
    "  E        2\n"
    "  N        3\n"
    "COLUMNS\n"
    "           1         1     1.0000000\n"
    "           2         1     1.0000000\n"
    "           2         2     1.0000000\n"
    "           3         2     1.0000000\n"
    "           3         3     1.0000000\n"
    "RHS\n"
    "           1         1     4.0000000\n"
    "           1         2     7.0000000\n"
    "BOUNDS\n"
    " FR        1         1\n"
    " UP        1         2    10.0000000\n"
    "QUADOBJ\n"
    "           1         1      1.000000\n"
    "           2         2      1.000000\n"
    "ENDATA\n";

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  return options;
}

TEST(Qp590, AVectorNamedLikeARowIsReadAsTheVectorName) {
  const TempFile file(kDpklo1Like, ".qps");
  Model model;
  const io::ReadResult read = io::read_mps(file.path(), &model);
  ASSERT_TRUE(read.ok) << read.error;
  ASSERT_EQ(model.num_rows(), 2);
  ASSERT_EQ(model.num_cols(), 3);
  EXPECT_DOUBLE_EQ(model.row_lower[0], 4.0);
  EXPECT_DOUBLE_EQ(model.row_upper[0], 4.0);
  EXPECT_DOUBLE_EQ(model.row_lower[1], 7.0);
  EXPECT_DOUBLE_EQ(model.row_upper[1], 7.0);
  EXPECT_FALSE(is_finite_bound(model.col_lower[0]));  // FR, bound vector "1"
  EXPECT_FALSE(is_finite_bound(model.col_upper[0]));  // not the UP line's column
  EXPECT_DOUBLE_EQ(model.col_upper[1], 10.0);         // UP, bound vector "1", column "2"

  const Solution s = solve(model, quiet());
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_NEAR(s.objective, 8.75, 1e-6);
  EXPECT_NEAR(s.col_value[0], 1.5, 1e-5);
  EXPECT_NEAR(s.col_value[1], 2.5, 1e-5);
  EXPECT_NEAR(s.col_value[2], 4.5, 1e-5);
  const KktVerdict verdict = check_qp_optimality(model, s);
  EXPECT_TRUE(verdict.passed) << verdict.check << ": " << verdict.detail;
}

// hues-mod: the reader dropped coefficients below kZeroDrop (1e-11), so the solver and its
// own KKT check both worked on a model the file does not state. A coefficient in an MPS or
// QPS file is data, however small, in the matrix and in the Hessian alike.
TEST(Qp590, ATinyCoefficientInTheFileIsKept) {
  const TempFile file(
      "NAME          TINY\n"
      "ROWS\n"
      " N  OBJ\n"
      " E  R1\n"
      "COLUMNS\n"
      "    X         R1        0.633333e-11\n"
      "    Y         R1        1.0\n"
      "RHS\n"
      "    RHS       R1        1.0\n"
      "QUADOBJ\n"
      "    X         X         0.500000e-12\n"
      "    Y         Y         1.0\n"
      "ENDATA\n",
      ".qps");
  Model model;
  const io::ReadResult read = io::read_mps(file.path(), &model);
  ASSERT_TRUE(read.ok) << read.error;
  EXPECT_EQ(model.matrix.num_nonzeros(), 2);
  EXPECT_DOUBLE_EQ(model.matrix.at(0, 0), 0.633333e-11);
  EXPECT_EQ(model.hessian.num_nonzeros(), 2);
  EXPECT_DOUBLE_EQ(model.hessian.at(0, 0), 0.5e-12);
}

/// min x1^2/2 + x2^2/2 - x1 - x2  s.t.  x1 + x2 = 1,  x1 = 0.2,  x >= 0.
/// Optimum x = (0.2, 0.8), gradient c + Q x = (-0.8, -0.2). Column 2 is interior, so the first
/// row's dual is -0.2; column 1 then needs -0.8 - (-0.2) - y1 = 0, so the singleton row's dual
/// is -0.6. Priced from c_1 = -1 instead, it comes out -0.8, off by (Q x)_1 = 0.2.
Model singleton_row_on_a_quadratic_column() {
  Model model;
  model.sense = ObjSense::kMinimize;
  model.col_cost = {-1.0, -1.0};
  model.col_lower = {0.0, 0.0};
  model.col_upper = {kInfinity, kInfinity};
  model.col_type = {VarType::kContinuous, VarType::kContinuous};
  model.row_lower = {1.0, 0.2};
  model.row_upper = {1.0, 0.2};
  model.matrix.reset(2, 2);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.add_entry(1, 0, 1.0);
  model.matrix.finalize();
  model.hessian.reset(2, 2);
  model.hessian.add_entry(0, 0, 1.0);
  model.hessian.add_entry(1, 1, 1.0);
  model.hessian.finalize();
  return model;
}

TEST(Qp590, PostsolvePricesASingletonRowFromTheGradient) {
  const Model model = singleton_row_on_a_quadratic_column();
  Options options = quiet();
  options.set_bool("presolve", true);
  const Solution s = solve(model, options);
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  // The reduction under test has to have happened, or this proves nothing about postsolve.
  ASSERT_GE(s.presolve_report.singleton_rows, 1) << "presolve removed no singleton row";
  EXPECT_NEAR(s.col_value[0], 0.2, 1e-6);
  EXPECT_NEAR(s.col_value[1], 0.8, 1e-6);
  EXPECT_NEAR(s.row_dual[0], -0.2, 1e-5);
  EXPECT_NEAR(s.row_dual[1], -0.6, 1e-5);
  const KktVerdict verdict = check_qp_optimality(model, s);
  EXPECT_TRUE(verdict.passed) << verdict.check << ": " << verdict.detail;
  EXPECT_EQ(s.message.find("#590"), std::string::npos) << s.message;
}

TEST(Qp590, TheSameQpWithoutPresolveHasTheSameDuals) {
  const Model model = singleton_row_on_a_quadratic_column();
  Options options = quiet();
  options.set_bool("presolve", false);
  const Solution s = solve(model, options);
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_NEAR(s.row_dual[0], -0.2, 1e-5);
  EXPECT_NEAR(s.row_dual[1], -0.6, 1e-5);
}

}  // namespace
}  // namespace sankhya
