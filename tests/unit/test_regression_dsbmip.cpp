// SPDX-License-Identifier: Apache-2.0
// SANKHYA - #1187: dsbmip's seed-2 permutation under the root loop recipe ended
// numerical_error, the status guard refusing a reported point whose binary ZGB200B8 sat at
// -1.9e-8 and so put the big-M row 10080 y + x - z >= 0 at -2e-4. The incumbent is now the
// snapped point (offer_incumbent rounds every integer column before the checks), so the
// reported point carries exact integers and the guard has nothing to refuse.
//
// tests/data/miplib3/dsbmip-seed2.mps is MIPLIB 3's dsbmip.mps with its rows and columns in
// the order bench/runners/miplib_seeds.permute_mps gives seed 2 (the runner's third seed),
// the file the A/Bs on #1158 ran; sha256 3fcd6bde899dcc304354114277d5c708190b26a2...

#include <cmath>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

std::string fixture(const std::string& name) {
  return (std::filesystem::path(__FILE__).parent_path().parent_path() / "data" / "miplib3" /
          name)
      .string();
}

TEST(Regression, DsbmipSeed2UnderTheRootLoopRecipeReportsAFeasibleIntegralPoint) {
  Model model;
  const std::string path = fixture("dsbmip-seed2.mps");
  const io::ReadResult read = io::read_model(path, &model);
  ASSERT_TRUE(read.ok) << path << ": " << read.error;

  Options options;
  options.set_bool("log_to_console", false);
  options.set_int("mip_threads", 1);
  options.set_double("time_limit", 120.0);
  options.set_int("node_limit", 400);  // the refused point came at node 227
  options.set_bool("root_cut_loop", true);
  options.set_bool("mip_cut_pooling", true);
  options.set_bool("enable_flow_cover_cuts", true);
  options.set_string("root_cut_loop_when", "flow-covers");
  const Solution solved = solve(model, options);

  EXPECT_NE(solved.status, SolveStatus::kNumericalError) << solved.message;
  EXPECT_NE(solved.status, SolveStatus::kModelError) << solved.message;
  ASSERT_TRUE(solved.has_primal_values()) << "no point reported: " << solved.message;
  // The reported point is feasible for the file as read, and its integers are integers.
  EXPECT_LE(solved.primal_infeasibility_scaled, 1e-7) << solved.message;
  for (Index j = 0; j < model.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (model.col_type[u] != VarType::kInteger) continue;
    EXPECT_EQ(solved.col_value[u], std::round(solved.col_value[u]))
        << "column " << j << " (" << model.col_names[u] << ") is not integral";
  }
  // The published optimum is -305.19817501; the search either proves it or holds a point
  // no worse than the one the guard refused.
  EXPECT_LE(solved.objective, -305.19817501 + 1e-6 * 305.2) << solved.message;
}

}  // namespace
}  // namespace sankhya
