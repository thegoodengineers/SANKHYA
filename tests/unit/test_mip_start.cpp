// SPDX-License-Identifier: Apache-2.0
// SANKHYA - mip_start (#753): a user-supplied starting point, checked against the model at
// the usual tolerances and offered as the search's first incumbent.
//
// Three things are shown here, each against the SAME model solved without mip_start as the
// baseline: a complete, feasible start is accepted and never leaves the search worse off
// (fewer or equal nodes, same optimum); an infeasible start is rejected and changes nothing
// about the answer; and a start missing a column - not yet repaired, per the option's own
// documentation - is rejected the same way.

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

/// A file in the temp directory unique to this process, as test_debug_solution.cpp uses.
std::string scratch_path(const std::string& name) {
  static const std::string nonce = std::to_string(std::random_device{}());
  return (std::filesystem::temp_directory_path() / ("sankhya_mip_start_" + nonce + "_" + name))
      .string();
}

void write_text(const std::string& path, const std::string& text) {
  std::ofstream out(path, std::ios::binary);
  out << text;
}

/// A 0-1 knapsack with enough items, and a fractional part on every value, that the root
/// relaxation does not close the gap and the search needs a real tree (the same shape
/// tests/unit/test_parallel_tree.cpp uses this for, #222).
Model knapsack(int items, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> weight(10, 60);
  Model model;
  model.sense = ObjSense::kMaximize;
  model.matrix.reset(1, items);
  double total = 0.0;
  for (int j = 0; j < items; ++j) {
    const int w = weight(rng);
    const int v = w + std::uniform_int_distribution<int>(-5, 12)(rng);
    model.col_cost.push_back(static_cast<double>(v) + 0.25 + 0.125 * static_cast<double>(j % 3));
    model.col_lower.push_back(0.0);
    model.col_upper.push_back(1.0);
    model.col_type.push_back(VarType::kInteger);
    model.col_names.push_back("x" + std::to_string(j));
    model.matrix.add_entry(0, j, static_cast<double>(w));
    total += w;
  }
  model.matrix.finalize();
  model.row_lower = {-kInfinity};
  model.row_upper = {std::floor(total / 2.0)};
  model.hessian.reset(items, items);
  model.hessian.finalize();
  return model;
}

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  // One thread: mip_start is wired into the sequential search only so far (#753).
  options.set_int("mip_threads", 1);
  return options;
}

std::string as_start_file(const Model& model, const std::vector<double>& x,
                          bool skip_last_column) {
  std::string text;
  const auto n = skip_last_column ? x.size() - 1 : x.size();
  for (std::size_t j = 0; j < n; ++j) {
    text += model.col_names[j] + " " + std::to_string(x[j]) + "\n";
  }
  return text;
}

TEST(MipStart, ACompleteFeasibleStartIsAcceptedAndNeverLeavesTheSearchWorseOff) {
  const Model model = knapsack(20, 753);

  const Solution baseline = solve(model, quiet());
  ASSERT_EQ(baseline.status, SolveStatus::kOptimal) << baseline.message;

  const std::string path = scratch_path("complete.sol");
  write_text(path, as_start_file(model, baseline.col_value, /*skip_last_column=*/false));

  Options options = quiet();
  options.set_string("mip_start", path);
  const Solution started = solve(model, options);

  ASSERT_EQ(started.status, SolveStatus::kOptimal) << started.message;
  EXPECT_NEAR(started.objective, baseline.objective, 1e-6);
  EXPECT_LE(started.nodes, baseline.nodes)
      << "the optimum itself as the start must not cost MORE nodes than starting cold";
}

TEST(MipStart, AnInfeasibleStartIsRejectedAndChangesNothingAboutTheAnswer) {
  const Model model = knapsack(20, 754);
  const Solution baseline = solve(model, quiet());
  ASSERT_EQ(baseline.status, SolveStatus::kOptimal) << baseline.message;

  // Every item "in", which overflows the knapsack row: infeasible on its face.
  std::vector<double> infeasible(static_cast<std::size_t>(model.num_cols()), 1.0);
  const std::string path = scratch_path("infeasible.sol");
  write_text(path, as_start_file(model, infeasible, /*skip_last_column=*/false));

  Options options = quiet();
  options.set_string("mip_start", path);
  const Solution started = solve(model, options);

  ASSERT_EQ(started.status, SolveStatus::kOptimal) << started.message;
  EXPECT_NEAR(started.objective, baseline.objective, 1e-6);
  EXPECT_EQ(started.nodes, baseline.nodes)
      << "a rejected start must leave the search byte-for-byte the same as no start at all";
}

TEST(MipStart, AStartMissingAColumnIsRejectedNotRepaired) {
  const Model model = knapsack(20, 755);
  const Solution baseline = solve(model, quiet());
  ASSERT_EQ(baseline.status, SolveStatus::kOptimal) << baseline.message;

  const std::string path = scratch_path("partial.sol");
  write_text(path, as_start_file(model, baseline.col_value, /*skip_last_column=*/true));

  Options options = quiet();
  options.set_string("mip_start", path);
  const Solution started = solve(model, options);

  ASSERT_EQ(started.status, SolveStatus::kOptimal) << started.message;
  EXPECT_NEAR(started.objective, baseline.objective, 1e-6);
  EXPECT_EQ(started.nodes, baseline.nodes)
      << "a partial start is documented as rejected, not repaired, until #753's item 3 lands";
}

TEST(MipStart, AMissingFileIsIgnoredRatherThanFailingTheSolve) {
  const Model model = knapsack(10, 756);
  Options options = quiet();
  options.set_string("mip_start", scratch_path("does_not_exist.sol"));
  const Solution started = solve(model, options);
  EXPECT_EQ(started.status, SolveStatus::kOptimal) << started.message;
}

}  // namespace
}  // namespace sankhya
