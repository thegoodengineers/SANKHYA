// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the nonlinear C API (NLP stage 1), through include/sankhya/sankhya_nonlinear.h
// ONLY, as test_c_api.cpp goes through sankhya.h: the boundary is what is under test. Every
// expected value is derived by hand in a comment.

#include <cmath>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "sankhya/sankhya_nonlinear.h"

#include "support/temp_file.hpp"

namespace {

struct ModelHandle {
  sankhya_model* handle = sankhya_model_create();
  ~ModelHandle() { sankhya_model_free(handle); }
  operator sankhya_model*() const { return handle; }
};

struct SolutionHandle {
  sankhya_solution* handle = nullptr;
  ~SolutionHandle() { sankhya_solution_free(handle); }
};

/// HS071 through the C API: four columns in [1, 5], x2 as a linear cost, the rest as
/// expressions. Returns the objective expression's handle.
int build_hs071(sankhya_model* m) {
  for (int j = 0; j < 4; ++j) {
    EXPECT_EQ(sankhya_model_add_column(m, j == 2 ? 1.0 : 0.0, 1.0, 5.0, 0, nullptr, nullptr),
              SANKHYA_OK);
  }
  int x[4];
  for (int j = 0; j < 4; ++j) x[j] = sankhya_expr_variable(m, j);
  const int first_three[3] = {x[0], x[1], x[2]};
  const int objective = sankhya_expr_binary(
      m, SANKHYA_EXPR_MUL, sankhya_expr_binary(m, SANKHYA_EXPR_MUL, x[0], x[3]),
      sankhya_expr_sum(m, 3, first_three));
  const int product = sankhya_expr_binary(m, SANKHYA_EXPR_MUL,
                                          sankhya_expr_binary(m, SANKHYA_EXPR_MUL, x[0], x[1]),
                                          sankhya_expr_binary(m, SANKHYA_EXPR_MUL, x[2], x[3]));
  int squares[4];
  for (int j = 0; j < 4; ++j) squares[j] = sankhya_expr_power(m, x[j], 2.0);
  const int ball = sankhya_expr_sum(m, 4, squares);
  EXPECT_EQ(sankhya_model_set_nonlinear_objective(m, objective), SANKHYA_OK);
  int index = -1;
  EXPECT_EQ(
      sankhya_model_add_nonlinear_row(m, product, 25.0, sankhya_infinity(), "prod", &index),
      SANKHYA_OK);
  EXPECT_EQ(index, 0);
  EXPECT_EQ(sankhya_model_add_nonlinear_row(m, ball, 40.0, 40.0, nullptr, &index), SANKHYA_OK);
  EXPECT_EQ(index, 1);
  return objective;
}

TEST(CApiNonlinear, BuildsEvaluatesAndValidatesHs071) {
  ModelHandle m;
  const int objective = build_hs071(m);
  ASSERT_GE(objective, 0) << sankhya_last_error();
  EXPECT_EQ(sankhya_model_num_nonlinear_rows(m), 2);
  EXPECT_EQ(sankhya_model_validate(m), SANKHYA_OK) << sankhya_last_error();
  // At (1, 5, 5, 1): x0 x3 (x0 + x1 + x2) = 1 * 1 * 11 = 11 (the linear x2 is a cost, not in
  // the expression).
  const double start[4] = {1, 5, 5, 1};
  double value = 0.0;
  ASSERT_EQ(sankhya_expr_evaluate(m, objective, start, 4, &value), SANKHYA_OK);
  EXPECT_EQ(value, 11.0);
  EXPECT_EQ(sankhya_model_set_start(m, start, 4), SANKHYA_OK);
  EXPECT_NE(sankhya_model_set_start(m, start, 3), SANKHYA_OK) << "the wrong length is refused";
}

TEST(CApiNonlinear, ASolveReachesHs071sPublishedOptimumAndSaysItIsLocal) {
  ModelHandle m;
  build_hs071(m);
  const double start[4] = {1, 5, 5, 1};
  ASSERT_EQ(sankhya_model_set_start(m, start, 4), SANKHYA_OK);
  sankhya_options* options = sankhya_options_create();
  ASSERT_EQ(sankhya_options_set_bool(options, "log_to_console", 0), SANKHYA_OK);
  SolutionHandle s;
  ASSERT_EQ(sankhya_solve(m, options, &s.handle), SANKHYA_OK) << sankhya_last_error();
  sankhya_options_free(options);
  // HS071 is not proved convex, so a KKT point is reported as LOCALLY optimal, with the
  // published optimum 17.0140173 (Hock and Schittkowski 1981, problem 71).
  EXPECT_EQ(sankhya_solution_status(s.handle), SANKHYA_LOCALLY_OPTIMAL)
      << sankhya_solution_message(s.handle);
  EXPECT_EQ(sankhya_solution_claims_a_point(s.handle), 1);
  EXPECT_NEAR(sankhya_solution_objective(s.handle), 17.0140173, 1e-6);
  // Row vectors hold the linear rows (none) then the nonlinear rows: two duals.
  double duals[2] = {0, 0};
  EXPECT_EQ(sankhya_solution_row_duals(s.handle, duals, 2), SANKHYA_OK) << sankhya_last_error();
  EXPECT_GT(duals[0], 0.0) << "the product row sits at its lower bound 25";
}

TEST(CApiNonlinear, MisuseIsStickyAndNamed) {
  ModelHandle m;
  ASSERT_EQ(sankhya_model_add_column(m, 0.0, 0.0, 1.0, 0, nullptr, nullptr), SANKHYA_OK);
  EXPECT_EQ(sankhya_expr_variable(m, 7), -1);
  EXPECT_NE(std::string(sankhya_last_error()).find("column 7"), std::string::npos)
      << sankhya_last_error();
  // -1 propagates, and the model cannot be validated or solved past it.
  EXPECT_EQ(sankhya_expr_unary(m, SANKHYA_EXPR_EXP, -1), -1);
  EXPECT_EQ(sankhya_expr_unary(m, 99, sankhya_expr_variable(m, 0)), -1) << "an unknown op";
  EXPECT_EQ(sankhya_expr_constant(m, std::nan("")), -1);
  EXPECT_NE(sankhya_model_add_nonlinear_row(m, 12345, 0.0, 1.0, nullptr, nullptr), SANKHYA_OK);
  EXPECT_EQ(sankhya_model_validate(m), SANKHYA_ERROR_MODEL);
  EXPECT_NE(std::string(sankhya_last_error()).find("column 7"), std::string::npos)
      << "the FIRST problem: " << sankhya_last_error();
}

TEST(CApiNonlinear, ADomainErrorIsAnErrorNotANaN) {
  ModelHandle m;
  ASSERT_EQ(sankhya_model_add_column(m, 0.0, -1.0, 1.0, 0, nullptr, nullptr), SANKHYA_OK);
  const int lg = sankhya_expr_unary(m, SANKHYA_EXPR_LOG, sankhya_expr_variable(m, 0));
  ASSERT_GE(lg, 0);
  const double bad[1] = {-1.0};
  double value = 123.0;
  EXPECT_EQ(sankhya_expr_evaluate(m, lg, bad, 1, &value), SANKHYA_ERROR_ARGUMENT);
  EXPECT_EQ(value, 123.0) << "nothing written";
  EXPECT_NE(std::string(sankhya_last_error()).find("log"), std::string::npos);
  // A column added after the expression is usable by the next one; the old handle stays.
  ASSERT_EQ(sankhya_model_add_column(m, 0.0, 0.0, 1.0, 0, nullptr, nullptr), SANKHYA_OK);
  const int both = sankhya_expr_binary(m, SANKHYA_EXPR_ADD, lg, sankhya_expr_variable(m, 1));
  ASSERT_GE(both, 0) << sankhya_last_error();
  const double good[2] = {std::exp(1.0), 0.25};
  ASSERT_EQ(sankhya_expr_evaluate(m, both, good, 2, &value), SANKHYA_OK);
  EXPECT_NEAR(value, 1.25, 1e-15);  // log(e) + 0.25
}

TEST(CApiNonlinear, ReadsANlFile) {
  // HS071 in .nl form (see test_nl_reader.cpp for the derivation of every line).
  const std::string text =
      "g3 1 1 0\n 4 2 1 0 1\n 2 1\n 0 0\n 4 4 4\n 0 0 0 1\n 0 0 0 0 0\n 8 4\n 0 0\n 0 0 0 0 0\n"
      "C0\no2\no2\no2\nv0\nv1\nv2\nv3\n"
      "C1\no54\n4\no5\nv0\nn2\no5\nv1\nn2\no5\nv2\nn2\no5\nv3\nn2\n"
      "O0 0\no2\no2\nv0\nv3\no54\n3\nv0\nv1\nv2\n"
      "x4\n0 1\n1 5\n2 5\n3 1\nr\n2 25\n4 40\nb\n0 1 5\n0 1 5\n0 1 5\n0 1 5\n"
      "J0 4\n0 0\n1 0\n2 0\n3 0\nJ1 4\n0 0\n1 0\n2 0\n3 0\nG0 4\n0 0\n1 0\n2 1\n3 0\n";
  const sankhya::testing::TempFile file(text, ".nl");
  ModelHandle m;
  ASSERT_EQ(sankhya_model_read(m, file.path().c_str()), SANKHYA_OK) << sankhya_last_error();
  EXPECT_EQ(sankhya_model_num_cols(m), 4);
  EXPECT_EQ(sankhya_model_num_rows(m), 0);
  EXPECT_EQ(sankhya_model_num_nonlinear_rows(m), 2);
  EXPECT_EQ(sankhya_model_validate(m), SANKHYA_OK) << sankhya_last_error();
}

}  // namespace
