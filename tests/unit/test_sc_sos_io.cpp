// SPDX-License-Identifier: Apache-2.0
// SANKHYA - semi-continuous columns and special ordered sets in the readers and the writer
// (#754).
//
// Every spelling the readers accept has a test that reads it into the model, and the
// writer's output is read back for both formats and compared field by field: a round trip
// that loses a set, its order or a run range would hand the search a different problem.

#include <cstdio>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"

#include "support/temp_file.hpp"

namespace sankhya {
namespace {

using testing::TempFile;

[[nodiscard]] Model read_or_fail(const std::string& text, const char* extension) {
  const TempFile file(text, extension);
  Model model;
  const io::ReadResult result = io::read_model(file.path(), &model);
  EXPECT_TRUE(result.ok) << result.error;
  return model;
}

[[nodiscard]] std::string read_expecting_failure(const std::string& text,
                                                 const char* extension) {
  const TempFile file(text, extension);
  Model model;
  const io::ReadResult result = io::read_model(file.path(), &model);
  EXPECT_FALSE(result.ok) << "expected the file to be refused";
  return result.error;
}

[[nodiscard]] Index col_of(const Model& model, const std::string& name) {
  for (std::size_t j = 0; j < model.col_names.size(); ++j) {
    if (model.col_names[j] == name) return static_cast<Index>(j);
  }
  return -1;
}

/// The set's member names, in the stored (weight) order.
[[nodiscard]] std::vector<std::string> members(const Model& model, const SosSet& set) {
  std::vector<std::string> out;
  for (const Index j : set.columns) out.push_back(model.col_names[static_cast<std::size_t>(j)]);
  return out;
}

// The columns every MPS case below shares: four columns in one row.
const char* const kMpsHead =
    "NAME          SCSOS\n"
    "ROWS\n"
    " N  COST\n"
    " L  CAP\n"
    "COLUMNS\n"
    "    X1        COST         1.0   CAP          1.0\n"
    "    X2        COST         2.0   CAP          1.0\n"
    "    X3        COST         3.0   CAP          1.0\n"
    "    X4        COST         4.0   CAP          1.0\n"
    "RHS\n"
    "    RHS       CAP          10.0\n";

// ---- MPS: the SC bound ------------------------------------------------------------------

TEST(ScSosMps, ScBoundTakesTheLowerBoundFromLo) {
  const Model model = read_or_fail(std::string(kMpsHead) +
                                       "BOUNDS\n"
                                       " LO BND       X1           2.5\n"
                                       " SC BND       X1           8.0\n"
                                       " SC BND       X3           1e30\n"
                                       "ENDATA\n",
                                   ".mps");
  const Index x1 = col_of(model, "X1");
  const Index x3 = col_of(model, "X3");
  ASSERT_EQ(model.semicontinuous, (std::vector<Index>{x1, x3}));
  EXPECT_DOUBLE_EQ(model.col_lower[static_cast<std::size_t>(x1)], 2.5);
  EXPECT_DOUBLE_EQ(model.col_upper[static_cast<std::size_t>(x1)], 8.0);
  // 1e30 is infinity, as in every other bound; the default lower bound 0 stays.
  EXPECT_DOUBLE_EQ(model.col_lower[static_cast<std::size_t>(x3)], 0.0);
  EXPECT_EQ(model.col_upper[static_cast<std::size_t>(x3)], kInfinity);
  EXPECT_TRUE(model.has_integrality()) << "a semi-continuous model is a MIP to the dispatcher";
}

TEST(ScSosMps, ScOnAnIntegerColumnIsSemiInteger) {
  const Model model = read_or_fail(
      "NAME          SI\n"
      "ROWS\n"
      " N  COST\n"
      " L  CAP\n"
      "COLUMNS\n"
      "    M0        'MARKER'                 'INTORG'\n"
      "    N1        COST         1.0   CAP          1.0\n"
      "    M1        'MARKER'                 'INTEND'\n"
      "RHS\n"
      "    RHS       CAP          10.0\n"
      "BOUNDS\n"
      " LI BND       N1           3\n"
      " SC BND       N1           7\n"
      "ENDATA\n",
      ".mps");
  ASSERT_EQ(model.semicontinuous.size(), 1u);
  EXPECT_EQ(model.col_type[0], VarType::kInteger);
  EXPECT_DOUBLE_EQ(model.col_lower[0], 3.0);
  EXPECT_DOUBLE_EQ(model.col_upper[0], 7.0);
}

TEST(ScSosMps, ANegativeRunRangeIsRefused) {
  const std::string error = read_expecting_failure(std::string(kMpsHead) +
                                                       "BOUNDS\n"
                                                       " LO BND       X1           -1\n"
                                                       " SC BND       X1           4\n"
                                                       "ENDATA\n",
                                                   ".mps");
  EXPECT_NE(error.find("semi-continuous"), std::string::npos) << error;
}

// ---- MPS: the SOS section and its SETS spelling -------------------------------------------

TEST(ScSosMps, SosSectionWithHeadersPrioritiesAndBothMemberSpellings) {
  const Model model = read_or_fail(std::string(kMpsHead) +
                                       "SOS\n"
                                       " S1 SOS       pick         5\n"
                                       "    pick      X2           2\n"
                                       "    pick      X1           1\n"
                                       " S2 SOS       curve\n"
                                       "    curve     X3:10\n"
                                       "    X4        20\n"
                                       "    X1:5\n"
                                       "ENDATA\n",
                                   ".mps");
  ASSERT_EQ(model.sos.size(), 2u);
  const SosSet& pick = model.sos[0];
  EXPECT_EQ(pick.type, 1);
  EXPECT_EQ(pick.name, "pick");
  EXPECT_DOUBLE_EQ(pick.priority, 5.0);
  // Stored in weight order, whatever the file's order.
  EXPECT_EQ(members(model, pick), (std::vector<std::string>{"X1", "X2"}));
  EXPECT_EQ(pick.weights, (std::vector<double>{1.0, 2.0}));
  const SosSet& curve = model.sos[1];
  EXPECT_EQ(curve.type, 2);
  EXPECT_EQ(members(model, curve), (std::vector<std::string>{"X1", "X3", "X4"}));
  EXPECT_EQ(curve.weights, (std::vector<double>{5.0, 10.0, 20.0}));
  EXPECT_TRUE(model.has_integrality());
}

TEST(ScSosMps, SetsSectionIsTheSameLayout) {
  const Model model = read_or_fail(std::string(kMpsHead) +
                                       "SETS\n"
                                       " S2 SOS       s2\n"
                                       "    s2        X1           1\n"
                                       "    s2        X2           2\n"
                                       "    s2        X3           3\n"
                                       "ENDATA\n",
                                   ".mps");
  ASSERT_EQ(model.sos.size(), 1u);
  EXPECT_EQ(model.sos[0].type, 2);
  EXPECT_EQ(members(model, model.sos[0]), (std::vector<std::string>{"X1", "X2", "X3"}));
}

/// One fixed-format data line: the six fields at the IBM offsets 1, 4, 14, 24, 39 and 49.
[[nodiscard]] std::string fixed(const std::vector<std::string>& fields) {
  static const std::size_t kOffset[6] = {1, 4, 14, 24, 39, 49};
  std::string line;
  for (std::size_t k = 0; k < fields.size(); ++k) {
    if (fields[k].empty()) continue;
    if (line.size() < kOffset[k]) line.resize(kOffset[k], ' ');
    line += fields[k];
  }
  return line + "\n";
}

TEST(ScSosMps, FixedFormatSosSection) {
  // Field 1 in columns 2-3 holds S1, field 2 the word SOS, field 3 the name: the layout the
  // IBM fixed columns give it, with a blank inside a column name to force the fixed reader.
  const Model model = read_or_fail(
      "NAME          FIXEDSOS\n"
      "ROWS\n"
      " N  COST\n"
      " L  CAP\n"
      "COLUMNS\n" +
          fixed({"", "X 1", "COST", "1.0", "CAP", "1.0"}) +
          fixed({"", "X 2", "COST", "2.0", "CAP", "1.0"}) + "RHS\n" +
          fixed({"", "RHS", "CAP", "10.0"}) + "SOS\n" + fixed({"S1", "SOS", "one", "2"}) +
          fixed({"", "one", "X 1", "1"}) + fixed({"", "one", "X 2", "2"}) + "ENDATA\n",
      ".mps");
  ASSERT_EQ(model.sos.size(), 1u);
  EXPECT_EQ(model.sos[0].type, 1);
  EXPECT_EQ(members(model, model.sos[0]), (std::vector<std::string>{"X 1", "X 2"}));
  EXPECT_DOUBLE_EQ(model.sos[0].priority, 2.0);
}

TEST(ScSosMps, EqualWeightsAreRefused) {
  const std::string error = read_expecting_failure(std::string(kMpsHead) +
                                                       "SOS\n"
                                                       " S2 SOS       c\n"
                                                       "    c         X1           1\n"
                                                       "    c         X2           1\n"
                                                       "ENDATA\n",
                                                   ".mps");
  EXPECT_NE(error.find("strictly increasing"), std::string::npos) << error;
}

TEST(ScSosMps, AMemberThatIsNotAColumnIsRefused) {
  const std::string error = read_expecting_failure(std::string(kMpsHead) +
                                                       "SOS\n"
                                                       " S1 SOS       c\n"
                                                       "    c         NOPE         1\n"
                                                       "ENDATA\n",
                                                   ".mps");
  EXPECT_NE(error.find("NOPE"), std::string::npos) << error;
}

// ---- MPS: the S1 / S2 marker form ---------------------------------------------------------

TEST(ScSosMps, MarkerFormInColumns) {
  const Model model = read_or_fail(
      "NAME          MARKERS\n"
      "ROWS\n"
      " N  COST\n"
      " L  CAP\n"
      " L  R2\n"
      "COLUMNS\n"
      " S2 SET1      'MARKER'                 'SOSORG'\n"
      "    L0        COST         0.0   CAP          1.0\n"
      "    L1        COST         3.0   CAP          1.0\n"
      "    L1        R2           2.0\n"
      "    L2        COST         1.0   CAP          1.0\n"
      "    SET1END   'MARKER'                 'SOSEND'\n"
      "    Y         COST         1.0   CAP          1.0\n"
      " S1 SET2      'MARKER'                 'SOSORG'\n"
      "    A         COST         1.0   CAP          1.0\n"
      "    B         COST         1.0   CAP          1.0\n"
      "    SET2END   'MARKER'                 'SOSEND'\n"
      "RHS\n"
      "    RHS       CAP          1.0\n"
      "ENDATA\n",
      ".mps");
  ASSERT_EQ(model.sos.size(), 2u);
  EXPECT_EQ(model.sos[0].type, 2);
  EXPECT_EQ(model.sos[0].name, "SET1");
  // A column listed on two lines is one member; the weights are the order of appearance.
  EXPECT_EQ(members(model, model.sos[0]), (std::vector<std::string>{"L0", "L1", "L2"}));
  EXPECT_EQ(model.sos[0].weights, (std::vector<double>{1.0, 2.0, 3.0}));
  EXPECT_EQ(model.sos[1].type, 1);
  EXPECT_EQ(members(model, model.sos[1]), (std::vector<std::string>{"A", "B"}));
  // A column between the two sets belongs to neither, and integrality is untouched.
  EXPECT_EQ(model.num_integer_columns(), 0);
}

TEST(ScSosMps, AnUnclosedMarkerIsRefused) {
  const std::string error = read_expecting_failure(
      "NAME          BAD\n"
      "ROWS\n"
      " N  COST\n"
      "COLUMNS\n"
      " S1 SET1      'MARKER'                 'SOSORG'\n"
      "    A         COST         1.0\n"
      " S1 SET2      'MARKER'                 'SOSORG'\n"
      "ENDATA\n",
      ".mps");
  EXPECT_NE(error.find("SOSORG"), std::string::npos) << error;
}

// ---- LP format ------------------------------------------------------------------------------

TEST(ScSosLp, SemiContinuousSectionInEverySpelling) {
  for (const char* keyword : {"Semi-Continuous", "semi-continuous", "semis", "SEMI"}) {
    const Model model = read_or_fail(std::string("Minimize\n obj: x + 2 y\n"
                                                 "Subject To\n c1: x + y >= 1\n"
                                                 "Bounds\n 2 <= x <= 6\n y <= 4\n") +
                                         keyword + "\n x\nEnd\n",
                                     ".lp");
    const Index x = col_of(model, "x");
    ASSERT_EQ(model.semicontinuous, (std::vector<Index>{x})) << keyword;
    EXPECT_DOUBLE_EQ(model.col_lower[static_cast<std::size_t>(x)], 2.0);
    EXPECT_DOUBLE_EQ(model.col_upper[static_cast<std::size_t>(x)], 6.0);
  }
}

TEST(ScSosLp, SosSectionNamedUnnamedAndWrapped) {
  const Model model = read_or_fail(
      "Minimize\n obj: a + b + c + d\n"
      "Subject To\n c1: a + b + c + d >= 1\n"
      "SOS\n"
      " s1: S1:: b:2 a:1\n"
      " S2:: a:-1 c:0.5\n"
      "   d:7\n"
      "End\n",
      ".lp");
  ASSERT_EQ(model.sos.size(), 2u);
  EXPECT_EQ(model.sos[0].name, "s1");
  EXPECT_EQ(model.sos[0].type, 1);
  EXPECT_EQ(members(model, model.sos[0]), (std::vector<std::string>{"a", "b"}));
  EXPECT_EQ(model.sos[1].type, 2);
  EXPECT_EQ(members(model, model.sos[1]), (std::vector<std::string>{"a", "c", "d"}));
  EXPECT_EQ(model.sos[1].weights, (std::vector<double>{-1.0, 0.5, 7.0}));
}

TEST(ScSosLp, AMalformedSosStatementIsRefused) {
  const std::string error = read_expecting_failure(
      "Minimize\n obj: a\nSubject To\n c1: a >= 1\nSOS\n s1: S3:: a:1\nEnd\n", ".lp");
  EXPECT_NE(error.find("SOS"), std::string::npos) << error;
}

// ---- The writer, both formats, read back ----------------------------------------------------

[[nodiscard]] Model sample_model() {
  Model model;
  model.name = "ROUNDTRIP";
  model.resize_columns(6);
  model.col_names = {"x", "y", "z", "l0", "l1", "l2"};
  model.col_cost = {1.0, -2.0, 0.5, 3.0, 1.0, 2.0};
  model.col_lower = {1.5, 0.0, 2.0, 0.0, 0.0, 0.0};
  model.col_upper = {9.0, kInfinity, 6.0, 1.0, 1.0, 1.0};
  model.col_type[2] = VarType::kInteger;  // z is semi-integer
  model.resize_rows(2);
  model.row_names = {"r0", "r1"};
  model.row_lower = {-kInfinity, 1.0};
  model.row_upper = {20.0, 1.0};
  model.matrix.reset(2, 6);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.add_entry(0, 2, 1.0);
  for (Index j = 3; j < 6; ++j) model.matrix.add_entry(1, j, 1.0);
  model.matrix.finalize();
  model.hessian.reset(6, 6);
  model.hessian.finalize();
  model.semicontinuous = {0, 1, 2};  // y's run range is [0, inf): written as SC inf
  SosSet one;
  one.type = 1;
  one.name = "pick";
  one.priority = 3.0;
  one.columns = {0, 1};
  one.weights = {1.0, 2.0};
  SosSet two;
  two.type = 2;
  two.name = "curve";
  two.columns = {3, 4, 5};
  two.weights = {0.0, 10.0, 20.0};
  model.sos = {one, two};
  return model;
}

void expect_same(const Model& a, const Model& b, bool priority_kept) {
  ASSERT_EQ(a.num_cols(), b.num_cols());
  for (Index j = 0; j < a.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    const Index k = col_of(b, a.col_names[u]);
    ASSERT_GE(k, 0) << a.col_names[u];
    const auto v = static_cast<std::size_t>(k);
    EXPECT_EQ(a.col_lower[u], b.col_lower[v]) << a.col_names[u];
    EXPECT_EQ(a.col_upper[u], b.col_upper[v]) << a.col_names[u];
    EXPECT_EQ(a.col_type[u], b.col_type[v]) << a.col_names[u];
  }
  std::vector<std::string> sc_a;
  std::vector<std::string> sc_b;
  for (const Index j : a.semicontinuous)
    sc_a.push_back(a.col_names[static_cast<std::size_t>(j)]);
  for (const Index j : b.semicontinuous)
    sc_b.push_back(b.col_names[static_cast<std::size_t>(j)]);
  EXPECT_EQ(sc_a, sc_b);
  ASSERT_EQ(a.sos.size(), b.sos.size());
  for (std::size_t s = 0; s < a.sos.size(); ++s) {
    EXPECT_EQ(a.sos[s].type, b.sos[s].type);
    EXPECT_EQ(a.sos[s].name, b.sos[s].name);
    EXPECT_EQ(members(a, a.sos[s]), members(b, b.sos[s]));
    EXPECT_EQ(a.sos[s].weights, b.sos[s].weights);
    if (priority_kept) {
      EXPECT_EQ(a.sos[s].priority, b.sos[s].priority);
    }
  }
}

TEST(ScSosWriter, MpsRoundTrip) {
  const Model model = sample_model();
  ASSERT_EQ(model.validate(), "");
  const TempFile file("", ".mps");
  std::string error;
  ASSERT_TRUE(io::write_model(file.path(), model, &error)) << error;
  Model back;
  const io::ReadResult result = io::read_model(file.path(), &back);
  ASSERT_TRUE(result.ok) << result.error;
  expect_same(model, back, /*priority_kept=*/true);
  // And a second trip writes the same text: nothing drifts.
  const TempFile again("", ".mps");
  ASSERT_TRUE(io::write_model(again.path(), back, &error)) << error;
  Model third;
  ASSERT_TRUE(io::read_model(again.path(), &third).ok);
  EXPECT_EQ(back.fingerprint(), third.fingerprint());
}

TEST(ScSosWriter, LpRoundTrip) {
  const Model model = sample_model();
  const TempFile file("", ".lp");
  std::string error;
  ASSERT_TRUE(io::write_model(file.path(), model, &error)) << error;
  Model back;
  const io::ReadResult result = io::read_model(file.path(), &back);
  ASSERT_TRUE(result.ok) << result.error;
  // The LP format has no field for a set's priority.
  expect_same(model, back, /*priority_kept=*/false);
}

TEST(ScSosModel, FingerprintSeesTheSetsAndOnlyThem) {
  Model plain = sample_model();
  plain.semicontinuous.clear();
  plain.sos.clear();
  Model with = sample_model();
  EXPECT_NE(plain.fingerprint(), with.fingerprint());
  Model reordered = sample_model();
  reordered.sos[1].weights = {0.0, 10.0, 30.0};
  EXPECT_NE(with.fingerprint(), reordered.fingerprint());
}

TEST(ScSosModel, ValidateRefusesMalformedSets) {
  Model model = sample_model();
  model.sos[0].columns = {0, 0};
  EXPECT_NE(model.validate().find("twice"), std::string::npos);
  model = sample_model();
  model.sos[1].type = 3;
  EXPECT_NE(model.validate().find("type"), std::string::npos);
  model = sample_model();
  model.semicontinuous = {2, 0};
  EXPECT_NE(model.validate().find("ascending"), std::string::npos);
  model = sample_model();
  model.sos[1].columns.push_back(99);
  model.sos[1].weights.push_back(40.0);
  EXPECT_NE(model.validate().find("out of range"), std::string::npos);
}

}  // namespace
}  // namespace sankhya
