// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the .sol basis reader behind `sankhya solve --warm-start` (#218).
//
// The basis written by write_solution() must come back exactly, by name, and a re-solve
// seeded with it must reach the cold solve's optimum in fewer pivots - the pivot count is
// the proof that the file, not the slack basis, started the simplex.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"
#include "support/model_edit.hpp"
#include "support/temp_file.hpp"

namespace sankhya {
namespace {

using testing::TempFile;

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  return options;
}

Model netlib(const char* name) {
  Model model;
  const std::string path =
      (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
       "data/netlib" / (std::string(name) + ".mps"))
          .string();
  const io::ReadResult read = io::read_model(path, &model);
  EXPECT_TRUE(read.ok) << path << ": " << read.error;
  return model;
}

TEST(SolReader, TheWrittenBasisComesBackByNameAndSeedsTheResolve) {
  for (const char* name : {"afiro", "adlittle", "share2b"}) {
    Model model = netlib(name);
    const Solution cold = solve(model, quiet());
    ASSERT_EQ(cold.status, SolveStatus::kOptimal) << name << ": " << cold.message;

    const TempFile file("", ".sol");
    std::string error;
    ASSERT_TRUE(io::write_solution(file.path(), model, cold, &error)) << error;

    SolveControl control;
    ASSERT_TRUE(io::read_solution_basis(file.path(), model, &control.start_col_status,
                                        &control.start_row_status, &error))
        << error;
    EXPECT_EQ(control.start_col_status, cold.col_status) << name;
    EXPECT_EQ(control.start_row_status, cold.row_status) << name;

    // The same model from its own optimal basis: the dual simplex confirms optimality
    // without a phase 1, in a small fraction of the cold pivots.
    const Solution warm = solve(model, quiet(), &control);
    ASSERT_EQ(warm.status, SolveStatus::kOptimal) << name << ": " << warm.message;
    EXPECT_NEAR(warm.objective, cold.objective, 1e-9 * (1.0 + std::abs(cold.objective)));
    EXPECT_LT(warm.iterations, cold.iterations / 2 + 1)
        << name << ": warm " << warm.iterations << " cold " << cold.iterations;
    EXPECT_NE(warm.message.find("warm start"), std::string::npos) << warm.message;
  }
}

TEST(SolReader, QuotedNamesRoundTrip) {
  // writer.cpp quotes a name with whitespace or a quote in it and escapes the quote.
  Model model = netlib("afiro");
  model.col_names[0] = "crude A \"heavy\"";
  model.row_names[0] = "CDU cap";
  const Solution cold = solve(model, quiet());
  ASSERT_EQ(cold.status, SolveStatus::kOptimal) << cold.message;
  const TempFile file("", ".sol");
  std::string error;
  ASSERT_TRUE(io::write_solution(file.path(), model, cold, &error)) << error;
  std::vector<BasisStatus> cols;
  std::vector<BasisStatus> rows;
  ASSERT_TRUE(io::read_solution_basis(file.path(), model, &cols, &rows, &error)) << error;
  EXPECT_EQ(cols, cold.col_status);
  EXPECT_EQ(rows, cold.row_status);
}

TEST(SolReader, AFileOfAnotherModelOrWithoutABasisIsRefused) {
  Model afiro = netlib("afiro");
  const Solution cold = solve(afiro, quiet());
  ASSERT_EQ(cold.status, SolveStatus::kOptimal);
  const TempFile file("", ".sol");
  std::string error;
  ASSERT_TRUE(io::write_solution(file.path(), afiro, cold, &error)) << error;

  std::vector<BasisStatus> cols;
  std::vector<BasisStatus> rows;
  Model other = netlib("adlittle");
  EXPECT_FALSE(io::read_solution_basis(file.path(), other, &cols, &rows, &error));
  EXPECT_NE(error.find("not in the model"), std::string::npos) << error;

  // A verdict with no point writes no columns block, so there is no basis to start from.
  const TempFile bare("status infeasible\nrows 1\ncolumns 1\n", ".sol");
  EXPECT_FALSE(io::read_solution_basis(bare.path(), afiro, &cols, &rows, &error));
  EXPECT_NE(error.find("no basis"), std::string::npos) << error;

  // A partial listing shares too little with the model to be a basis of it.
  const TempFile partial("begin columns 1\n" + afiro.col_names[0] +
                             " 0 0 basic\nend columns\n" + "begin rows 1\n" +
                             afiro.row_names[0] + " 0 0 at_lower\nend rows\n",
                         ".sol");
  EXPECT_FALSE(io::read_solution_basis(partial.path(), afiro, &cols, &rows, &error));
  EXPECT_NE(error.find("missing"), std::string::npos) << error;

  // A file that carries a point but no basis (statuses `unknown`) has nothing to seed.
  const TempFile unknown("begin columns 1\n" + afiro.col_names[0] +
                             " 0 0 unknown\nend columns\n" + "begin rows 1\n" +
                             afiro.row_names[0] + " 0 0 at_lower\nend rows\n",
                         ".sol");
  EXPECT_FALSE(io::read_solution_basis(unknown.path(), afiro, &cols, &rows, &error));
  EXPECT_NE(error.find("no complete basis"), std::string::npos) << error;
}

TEST(SolReader, AFileOfTheModelBeforeRowsAndColumnsWereAddedOrRemovedSeedsTheResolve) {
  // #913, the CLI path: `sankhya solve today.mps --warm-start yesterday.sol` where today's
  // model gained and lost rows and columns since yesterday's file was written.
  for (const char* name : {"afiro", "adlittle", "sc50a", "share2b"}) {
    const Model before = netlib(name);
    const Solution first = solve(before, quiet());
    ASSERT_EQ(first.status, SolveStatus::kOptimal) << name << ": " << first.message;
    const TempFile file("", ".sol");
    std::string error;
    ASSERT_TRUE(io::write_solution(file.path(), before, first, &error)) << error;

    // Drop the first basic column and the last row, add a column and a row.
    Model after = before;
    Index basic = 0;
    while (first.col_status[static_cast<std::size_t>(basic)] != BasisStatus::kBasic) ++basic;
    testing::remove_column(&after, basic);
    testing::remove_row(&after, after.num_rows() - 1);
    testing::add_column(&after, "NEW_COL", 1.0, 0.0, 5.0, {{0, 0, 1.0}, {1, 0, 2.0}});
    testing::add_row(&after, "NEW_ROW", -kInfinity, 1e4, {{0, 0, 1.0}, {0, 2, 1.0}});
    ASSERT_EQ(after.validate(), "") << name;

    SolveControl control;
    BasisMapping mapping;
    ASSERT_TRUE(io::read_solution_basis(file.path(), after, &control.start_col_status,
                                        &control.start_row_status, &error, &mapping))
        << name << ": " << error;
    EXPECT_EQ(mapping.new_cols, 1) << name;
    EXPECT_EQ(mapping.new_rows, 1) << name;
    EXPECT_EQ(mapping.removed_cols, 1) << name;
    EXPECT_EQ(mapping.removed_rows, 1) << name;
    EXPECT_EQ(control.start_row_status.back(), BasisStatus::kBasic) << name;
    EXPECT_EQ(control.start_col_status.back(), BasisStatus::kAtLower) << name;

    const Solution cold = solve(after, quiet());
    const Solution warm = solve(after, quiet(), &control);
    ASSERT_EQ(warm.status, cold.status) << name << ": " << warm.message;
    if (cold.status != SolveStatus::kOptimal) continue;
    EXPECT_NEAR(warm.objective, cold.objective, 1e-7 * std::max(1.0, std::abs(cold.objective)))
        << name;
    EXPECT_NE(warm.message.find("warm start"), std::string::npos) << warm.message;
  }
}

TEST(SolReader, ReadSolutionPointRecoversValues) {
  Model model = netlib("afiro");
  const Solution cold = solve(model, quiet());
  ASSERT_EQ(cold.status, SolveStatus::kOptimal);
  const TempFile file("", ".sol");
  std::string error;
  ASSERT_TRUE(io::write_solution(file.path(), model, cold, &error)) << error;

  std::vector<double> values;
  ASSERT_TRUE(io::read_solution_point(file.path(), model, &values, &error)) << error;
  ASSERT_EQ(values.size(), model.num_cols());
  for (std::size_t j = 0; j < values.size(); ++j) {
    EXPECT_NEAR(values[j], cold.col_value[j], 1e-9);
  }
}

TEST(SolReader, ReadSolutionPointAllowsPartialStarts) {
  Model model = netlib("afiro");
  const TempFile partial("begin columns 1\n" + model.col_names[0] +
                             " 3.14 0 basic\nend columns\n" + "begin rows 1\n" +
                             model.row_names[0] + " 0 0 at_lower\nend rows\n",
                         ".sol");
  std::vector<double> values;
  std::string error;
  ASSERT_TRUE(io::read_solution_point(partial.path(), model, &values, &error)) << error;
  ASSERT_EQ(values.size(), model.num_cols());
  EXPECT_DOUBLE_EQ(values[0], 3.14);
  for (std::size_t j = 1; j < values.size(); ++j) {
    EXPECT_TRUE(std::isnan(values[j]));
  }
}

}  // namespace
}  // namespace sankhya
