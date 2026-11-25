// SPDX-License-Identifier: Apache-2.0
// SANKHYA - CPLEX LP format reader tests.
//
// The LP reader's job is to produce exactly the same Model an MPS file would, so the
// central test here parses the same tiny problem from both formats and compares field by
// field. That is a stronger check than asserting on the LP reader's output alone: it
// catches an LP reader that is self-consistently wrong.

#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#include "support/temp_file.hpp"

namespace sankhya {
namespace {

using testing::TempFile;

[[nodiscard]] Model parse_or_fail(const std::string& text) {
  const TempFile file(text, ".lp");
  Model model;
  const io::ReadResult result = io::read_lp(file.path(), &model);
  EXPECT_TRUE(result.ok) << result.error;
  return model;
}

[[nodiscard]] std::string parse_expecting_failure(const std::string& text) {
  const TempFile file(text, ".lp");
  Model model;
  const io::ReadResult result = io::read_lp(file.path(), &model);
  EXPECT_FALSE(result.ok) << "expected this file to be rejected";
  return result.error;
}

[[nodiscard]] Index col_of(const Model& model, const std::string& name) {
  for (std::size_t j = 0; j < model.col_names.size(); ++j) {
    if (model.col_names[j] == name) return static_cast<Index>(j);
  }
  return -1;
}

[[nodiscard]] Index row_of(const Model& model, const std::string& name) {
  for (std::size_t i = 0; i < model.row_names.size(); ++i) {
    if (model.row_names[i] == name) return static_cast<Index>(i);
  }
  return -1;
}

TEST(LpReader, ReadsASmallCompleteModel) {
  const Model model = parse_or_fail(
      "\\ a comment\n"
      "Minimize\n"
      " obj: 3 x + 2 y + 4 z\n"
      "Subject To\n"
      " c1: x + y + z >= 10\n"
      " c2: x - y <= 4\n"
      " c3: 2 x + z = 6\n"
      "Bounds\n"
      " 0 <= x <= 40\n"
      " y >= 1\n"
      " z free\n"
      "End\n");

  EXPECT_EQ(model.num_rows(), 3);
  EXPECT_EQ(model.num_cols(), 3);
  EXPECT_EQ(model.sense, ObjSense::kMinimize);

  const Index x = col_of(model, "x");
  const Index y = col_of(model, "y");
  const Index z = col_of(model, "z");
  ASSERT_GE(x, 0);
  ASSERT_GE(y, 0);
  ASSERT_GE(z, 0);

  EXPECT_DOUBLE_EQ(model.col_cost[static_cast<std::size_t>(x)], 3.0);
  EXPECT_DOUBLE_EQ(model.col_cost[static_cast<std::size_t>(y)], 2.0);
  EXPECT_DOUBLE_EQ(model.col_cost[static_cast<std::size_t>(z)], 4.0);

  EXPECT_DOUBLE_EQ(model.col_lower[static_cast<std::size_t>(x)], 0.0);
  EXPECT_DOUBLE_EQ(model.col_upper[static_cast<std::size_t>(x)], 40.0);
  EXPECT_DOUBLE_EQ(model.col_lower[static_cast<std::size_t>(y)], 1.0);
  EXPECT_TRUE(is_infinite(model.col_lower[static_cast<std::size_t>(z)]));
  EXPECT_TRUE(is_infinite(model.col_upper[static_cast<std::size_t>(z)]));

  const Index c1 = row_of(model, "c1");
  const Index c2 = row_of(model, "c2");
  const Index c3 = row_of(model, "c3");
  EXPECT_DOUBLE_EQ(model.row_lower[static_cast<std::size_t>(c1)], 10.0);
  EXPECT_TRUE(is_infinite(model.row_upper[static_cast<std::size_t>(c1)]));
  EXPECT_TRUE(is_infinite(model.row_lower[static_cast<std::size_t>(c2)]));
  EXPECT_DOUBLE_EQ(model.row_upper[static_cast<std::size_t>(c2)], 4.0);
  EXPECT_DOUBLE_EQ(model.row_lower[static_cast<std::size_t>(c3)], 6.0);
  EXPECT_DOUBLE_EQ(model.row_upper[static_cast<std::size_t>(c3)], 6.0);

  EXPECT_DOUBLE_EQ(model.matrix.at(c3, x), 2.0);
  EXPECT_DOUBLE_EQ(model.matrix.at(c2, y), -1.0);
}

TEST(LpReader, AgreesWithTheMpsReaderOnTheSameProblem) {
  const Model from_lp = parse_or_fail(
      "Minimize\n"
      " obj: 1 x + 2 y\n"
      "Subject To\n"
      " r1: x + y >= 4\n"
      " r2: x - 2 y <= 3\n"
      "Bounds\n"
      " x <= 10\n"
      "End\n");

  const TempFile mps_file(
      "NAME          SAME\n"
      "ROWS\n"
      " N  obj\n"
      " G  r1\n"
      " L  r2\n"
      "COLUMNS\n"
      "    x         obj          1.0   r1           1.0\n"
      "    x         r2           1.0\n"
      "    y         obj          2.0   r1           1.0\n"
      "    y         r2          -2.0\n"
      "RHS\n"
      "    RHS       r1           4.0   r2           3.0\n"
      "BOUNDS\n"
      " UP BND       x           10.0\n"
      "ENDATA\n");
  Model from_mps;
  const io::ReadResult result = io::read_mps(mps_file.path(), &from_mps);
  ASSERT_TRUE(result.ok) << result.error;

  ASSERT_EQ(from_lp.num_rows(), from_mps.num_rows());
  ASSERT_EQ(from_lp.num_cols(), from_mps.num_cols());
  ASSERT_EQ(from_lp.num_nonzeros(), from_mps.num_nonzeros());
  EXPECT_EQ(from_lp.sense, from_mps.sense);

  for (Index j = 0; j < from_lp.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    EXPECT_EQ(from_lp.col_names[u], from_mps.col_names[u]);
    EXPECT_DOUBLE_EQ(from_lp.col_cost[u], from_mps.col_cost[u]);
    EXPECT_DOUBLE_EQ(from_lp.col_lower[u], from_mps.col_lower[u]);
    EXPECT_DOUBLE_EQ(from_lp.col_upper[u], from_mps.col_upper[u]);
  }
  for (Index i = 0; i < from_lp.num_rows(); ++i) {
    const auto u = static_cast<std::size_t>(i);
    EXPECT_EQ(from_lp.row_names[u], from_mps.row_names[u]);
    EXPECT_DOUBLE_EQ(from_lp.row_lower[u], from_mps.row_lower[u]);
    EXPECT_DOUBLE_EQ(from_lp.row_upper[u], from_mps.row_upper[u]);
    for (Index j = 0; j < from_lp.num_cols(); ++j) {
      EXPECT_DOUBLE_EQ(from_lp.matrix.at(i, j), from_mps.matrix.at(i, j))
          << "entry (" << i << ", " << j << ")";
    }
  }
}

TEST(LpReader, MaximizeIsRecorded) {
  const Model model = parse_or_fail(
      "Maximize\n"
      " 4 x + 3 y\n"
      "Subject To\n"
      " c: x + y <= 5\n"
      "End\n");
  EXPECT_EQ(model.sense, ObjSense::kMaximize);
  EXPECT_DOUBLE_EQ(model.col_cost[0], 4.0);
}

TEST(LpReader, RangeConstraintsAreTwoSided) {
  const Model model = parse_or_fail(
      "Minimize\n"
      " x\n"
      "Subject To\n"
      " r: -3 <= x + y <= 5\n"
      "End\n");
  EXPECT_DOUBLE_EQ(model.row_lower[0], -3.0);
  EXPECT_DOUBLE_EQ(model.row_upper[0], 5.0);
}

TEST(LpReader, SignsAreSeparateTokensOnBothSidesOfARelation) {
  // "-3" lexes as an operator followed by a number, so both the range-constraint lookahead
  // and the right-hand side have to walk past leading signs. Missing that turns a valid
  // range row into a parse error, and - worse - a lookahead that consumed the sign without
  // matching would silently drop the first term of "-3 x + y >= 2".
  const Model model = parse_or_fail(
      "Minimize\n"
      " x\n"
      "Subject To\n"
      " a: -3 x + y >= -2\n"
      " b: -4 <= x - y <= -1\n"
      "End\n");
  ASSERT_EQ(model.num_rows(), 2);
  EXPECT_DOUBLE_EQ(model.matrix.at(0, col_of(model, "x")), -3.0);
  EXPECT_DOUBLE_EQ(model.matrix.at(0, col_of(model, "y")), 1.0);
  EXPECT_DOUBLE_EQ(model.row_lower[0], -2.0);
  EXPECT_DOUBLE_EQ(model.row_lower[1], -4.0);
  EXPECT_DOUBLE_EQ(model.row_upper[1], -1.0);
}

TEST(LpReader, CoefficientsMayBeGluedToTheVariable) {
  const Model model = parse_or_fail(
      "Minimize\n"
      " 3x + 2.5y - z\n"
      "Subject To\n"
      " c: x + y + z >= 1\n"
      "End\n");
  EXPECT_DOUBLE_EQ(model.col_cost[static_cast<std::size_t>(col_of(model, "x"))], 3.0);
  EXPECT_DOUBLE_EQ(model.col_cost[static_cast<std::size_t>(col_of(model, "y"))], 2.5);
  EXPECT_DOUBLE_EQ(model.col_cost[static_cast<std::size_t>(col_of(model, "z"))], -1.0);
}

TEST(LpReader, AConstantInTheObjectiveBecomesTheOffset) {
  const Model model = parse_or_fail(
      "Minimize\n"
      " obj: 2 x + 17\n"
      "Subject To\n"
      " c: x >= 3\n"
      "End\n");
  EXPECT_DOUBLE_EQ(model.objective_offset, 17.0);
  const std::vector<double> x{3.0};
  EXPECT_DOUBLE_EQ(model.evaluate_objective(x.data()), 23.0);
}

TEST(LpReader, AConstantOnTheLeftOfAConstraintMovesToTheRight) {
  const Model model = parse_or_fail(
      "Minimize\n"
      " x\n"
      "Subject To\n"
      " c: x + 5 >= 12\n"
      "End\n");
  EXPECT_DOUBLE_EQ(model.row_lower[0], 7.0);
}

TEST(LpReader, GeneralAndBinarySectionsSetIntegrality) {
  const Model model = parse_or_fail(
      "Minimize\n"
      " x + y + z\n"
      "Subject To\n"
      " c: x + y + z >= 2\n"
      "General\n"
      " y\n"
      "Binary\n"
      " z\n"
      "End\n");
  EXPECT_EQ(model.col_type[static_cast<std::size_t>(col_of(model, "x"))], VarType::kContinuous);
  EXPECT_EQ(model.col_type[static_cast<std::size_t>(col_of(model, "y"))], VarType::kInteger);
  EXPECT_EQ(model.col_type[static_cast<std::size_t>(col_of(model, "z"))], VarType::kInteger);
  EXPECT_DOUBLE_EQ(model.col_upper[static_cast<std::size_t>(col_of(model, "z"))], 1.0);
  EXPECT_TRUE(is_infinite(model.col_upper[static_cast<std::size_t>(col_of(model, "y"))]));
}

TEST(LpReader, UnnamedConstraintsGetGeneratedNames) {
  const Model model = parse_or_fail(
      "Minimize\n"
      " x\n"
      "Subject To\n"
      " x + y >= 1\n"
      " x - y <= 2\n"
      "End\n");
  ASSERT_EQ(model.num_rows(), 2);
  EXPECT_EQ(model.row_names[0], "R1");
  EXPECT_EQ(model.row_names[1], "R2");
}

TEST(LpReader, RowNamesAreUniqueAsTheMpsReaderRequires) {
  // Two rows of one name wrote a .sol the warm start could only half match ("1 row(s) of
  // the file are gone"). A repeated explicit name is refused, as in MPS, and a generated
  // name skips the names the file has already taken.
  const std::string error = parse_expecting_failure(
      "Minimize\n"
      " obj: x + y\n"
      "Subject To\n"
      " c1: x >= 1\n"
      " c1: y >= 2\n"
      "End\n");
  EXPECT_NE(error.find("c1"), std::string::npos) << error;
  EXPECT_NE(error.find("twice"), std::string::npos) << error;

  const Model model = parse_or_fail(
      "Minimize\n"
      " obj: x + y\n"
      "Subject To\n"
      " R1: x >= 1\n"
      " y >= 2\n"
      " R3: x + y >= 3\n"
      " x - y <= 4\n"
      "End\n");
  ASSERT_EQ(model.num_rows(), 4);
  EXPECT_EQ(model.row_names, (std::vector<std::string>{"R1", "R2", "R3", "R4"}));
}

TEST(LpReader, AnExplicitNameLaterInTheFileWinsOverAGeneratedOne) {
  // The unnamed row comes first and would have been R1; the file names a later row R1, so
  // the generated name skips it instead of the file being refused for a clash it never wrote.
  const Model model = parse_or_fail(
      "Minimize\n"
      " obj: x + y\n"
      "Subject To\n"
      " x + y >= 1\n"
      " R1: x <= 5\n"
      " y <= 7\n"
      " R2: x - y <= 4\n"
      "End\n");
  EXPECT_EQ(model.row_names, (std::vector<std::string>{"R3", "R1", "R4", "R2"}));

  // A name the file itself repeats is still refused, whichever rows sit between.
  const std::string error = parse_expecting_failure(
      "Minimize\n"
      " obj: x + y\n"
      "Subject To\n"
      " R1: x >= 1\n"
      " x + y >= 2\n"
      " R1: y >= 2\n"
      "End\n");
  EXPECT_NE(error.find("R1"), std::string::npos) << error;
  EXPECT_NE(error.find("twice"), std::string::npos) << error;
}

TEST(LpReader, StatementsMayWrapAcrossLines) {
  const Model model = parse_or_fail(
      "Minimize\n"
      " obj: 3 x\n"
      "      + 2 y\n"
      "      - 4 z\n"
      "Subject To\n"
      " c: x + y\n"
      "    + z >= 10\n"
      "End\n");
  EXPECT_EQ(model.num_rows(), 1);
  EXPECT_DOUBLE_EQ(model.col_cost[static_cast<std::size_t>(col_of(model, "y"))], 2.0);
  EXPECT_DOUBLE_EQ(model.col_cost[static_cast<std::size_t>(col_of(model, "z"))], -4.0);
  EXPECT_DOUBLE_EQ(model.matrix.at(0, col_of(model, "z")), 1.0);
}

TEST(LpReader, InfinityKeywordsAreAcceptedInBounds) {
  const Model model = parse_or_fail(
      "Minimize\n"
      " x\n"
      "Subject To\n"
      " c: x >= 1\n"
      "Bounds\n"
      " -infinity <= x <= +inf\n"
      "End\n");
  EXPECT_TRUE(is_infinite(model.col_lower[0]));
  EXPECT_TRUE(is_infinite(model.col_upper[0]));
}

TEST(LpReader, AQuadraticObjectiveIsRejectedRatherThanIgnored) {
  // Dropping the quadratic term would solve the LP relaxation of a QP and report it as
  // optimal, which is the exact class of silent wrong answer this project cannot afford.
  const std::string error = parse_expecting_failure(
      "Minimize\n"
      " obj: x + [ 2 x ^ 2 ] / 2\n"
      "Subject To\n"
      " c: x >= 1\n"
      "End\n");
  EXPECT_NE(error.find("quadratic"), std::string::npos) << error;
}

TEST(LpReader, AConstraintWithNoRelationalOperatorIsRejected) {
  const std::string error = parse_expecting_failure(
      "Minimize\n"
      " x\n"
      "Subject To\n"
      " c: x + y\n"
      "End\n");
  EXPECT_NE(error.find("relational operator"), std::string::npos) << error;
}

TEST(LpReader, AFileWithNoObjectiveSectionIsRejected) {
  const std::string error = parse_expecting_failure(
      "Subject To\n"
      " c: x >= 1\n"
      "End\n");
  EXPECT_NE(error.find("Maximize"), std::string::npos) << error;
}

// =========================================================================================
// A zero coefficient written against its variable
// =========================================================================================
//
// What the format says, from its published description (IBM, "CPLEX LP file format", syntax
// rules, reproduced in the lp_solve reference guide's "CPLEX lp files" page): a coefficient is
// written against its variable with or without a space, "1x1 + 2x2 +3x3" being the format's
// own example; "a variable name cannot begin with a number or a period"; and the one notation
// reserved inside a number is the exponent, E or e. The format has no hexadecimal number, so
// "0x1" is the coefficient 0 on the variable x1. C's strtod reads "0x1" as the hexadecimal
// constant 1 (C99 7.20.1.3), which deletes the term and moves a 1 to the right-hand side.

TEST(LpReader, AZeroCoefficientGluedToItsVariableIsNotAHexadecimalNumber) {
  const Model model = parse_or_fail(
      "Minimize\n"
      " obj: x2\n"
      "Subject To\n"
      " c1: 0x1 + x2 >= 2\n"
      "End\n");
  ASSERT_EQ(model.num_cols(), 2);
  ASSERT_GE(col_of(model, "x1"), 0);
  ASSERT_GE(col_of(model, "x2"), 0);
  EXPECT_DOUBLE_EQ(model.row_lower[0], 2.0);
  EXPECT_TRUE(is_infinite(model.row_upper[0]));
  EXPECT_DOUBLE_EQ(model.matrix.at(0, col_of(model, "x1")), 0.0);
  EXPECT_DOUBLE_EQ(model.matrix.at(0, col_of(model, "x2")), 1.0);

  // And the answer: minimise x2 subject to x2 >= 2 is 2, not the 1 of x2 >= 2 - 1.
  Options options;
  options.set_bool("log_to_console", false);
  const Solution solution = solve(model, options);
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NEAR(solution.objective, 2.0, 1e-9);
}

TEST(LpReader, EverySpellingStrtodTakesForHexadecimalIsAZeroAndAName) {
  // 0x1 = 1, 0X1 = 1, 0xA = 10 and 0x1p3 = 8 to strtod. In a row the value would move to the
  // right-hand side, in the objective into the offset.
  struct Case {
    const char* term;
    const char* name;
  };
  for (const Case& c :
       {Case{"0x1", "x1"}, Case{"0X1", "X1"}, Case{"0xA", "xA"}, Case{"0x1p3", "x1p3"}}) {
    const Model model = parse_or_fail(std::string("Minimize\n obj: y + ") + c.term +
                                      "\nSubject To\n c1: 2 y + " + c.term +
                                      " <= 10\n c2: " + c.term + " + y >= 3\nEnd\n");
    ASSERT_EQ(model.num_cols(), 2) << c.term;
    const Index named = col_of(model, c.name);
    const Index y = col_of(model, "y");
    ASSERT_GE(named, 0) << c.term;
    ASSERT_GE(y, 0) << c.term;
    EXPECT_DOUBLE_EQ(model.objective_offset, 0.0) << c.term;
    EXPECT_DOUBLE_EQ(model.col_cost[static_cast<std::size_t>(named)], 0.0) << c.term;
    EXPECT_DOUBLE_EQ(model.col_cost[static_cast<std::size_t>(y)], 1.0) << c.term;
    EXPECT_DOUBLE_EQ(model.row_upper[0], 10.0) << c.term;
    EXPECT_DOUBLE_EQ(model.row_lower[1], 3.0) << c.term;
    EXPECT_DOUBLE_EQ(model.matrix.at(0, named), 0.0) << c.term;
    EXPECT_DOUBLE_EQ(model.matrix.at(0, y), 2.0) << c.term;
    EXPECT_DOUBLE_EQ(model.matrix.at(1, y), 1.0) << c.term;
  }
}

// =========================================================================================
// A binary variable the Bounds section already bounded
// =========================================================================================
//
// "Binary variables are automatically given bounds of 0 (zero) and 1 (one), unless alternative
// bounds are specified in the BOUNDS section" (IBM, "MIP features in LP file format"; the same
// sentence in the lp_solve reference guide's "CPLEX lp files" page), and the BINARY section
// follows the BOUNDS section. So a fixed binary is written "x = 0" under Bounds and x under
// Binaries, and the Binary section must not put x back to [0, 1]: that reads a relaxation of
// the model the file states.

TEST(LpReader, ABinaryKeepsTheBoundsTheBoundsSectionGaveIt) {
  struct Case {
    const char* bound;
    double lower;
    double upper;
  };
  // A side the Bounds section did not state keeps the binary default.
  for (const Case& c :
       {Case{"x = 0", 0.0, 0.0}, Case{"x = 1", 1.0, 1.0}, Case{"x <= 0", 0.0, 0.0},
        Case{"x >= 1", 1.0, 1.0}, Case{"0 <= x <= 1", 0.0, 1.0}}) {
    const Model model = parse_or_fail(std::string("Maximize\n"
                                                  " obj: 3 x + 2 y\n"
                                                  "Subject To\n"
                                                  " c1: x + y <= 1\n"
                                                  "Bounds\n ") +
                                      c.bound +
                                      "\n"
                                      "Binaries\n"
                                      " x y\n"
                                      "End\n");
    const auto x = static_cast<std::size_t>(col_of(model, "x"));
    const auto y = static_cast<std::size_t>(col_of(model, "y"));
    EXPECT_EQ(model.col_type[x], VarType::kInteger) << c.bound;
    EXPECT_DOUBLE_EQ(model.col_lower[x], c.lower) << c.bound;
    EXPECT_DOUBLE_EQ(model.col_upper[x], c.upper) << c.bound;
    EXPECT_EQ(model.col_type[y], VarType::kInteger) << c.bound;
    EXPECT_DOUBLE_EQ(model.col_lower[y], 0.0) << c.bound;
    EXPECT_DOUBLE_EQ(model.col_upper[y], 1.0) << c.bound;
  }
}

TEST(LpReader, ABinaryFixedAtZeroIsSolvedAtZero) {
  // maximise 3 x + 2 y, x + y <= 1, x fixed at 0: y = 1, objective 2. With x put back to
  // [0, 1] the answer is x = 1, objective 3 - a point outside the file's bound, and better
  // than the true optimum.
  const Model model = parse_or_fail(
      "Maximize\n"
      " obj: 3 x + 2 y\n"
      "Subject To\n"
      " c1: x + y <= 1\n"
      "Bounds\n"
      " x = 0\n"
      "Binaries\n"
      " x y\n"
      "End\n");
  Options options;
  options.set_bool("log_to_console", false);
  const Solution solution = solve(model, options);
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NEAR(solution.objective, 2.0, 1e-9);
  EXPECT_NEAR(solution.col_value[static_cast<std::size_t>(col_of(model, "x"))], 0.0, 1e-9);
  EXPECT_NEAR(solution.col_value[static_cast<std::size_t>(col_of(model, "y"))], 1.0, 1e-9);
}

}  // namespace
}  // namespace sankhya
