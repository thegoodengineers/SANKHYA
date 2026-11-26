// SPDX-License-Identifier: Apache-2.0
// SANKHYA - node bound propagation beside a large column bound (#1359, #1360).
//
// The node propagator (Savelsbergh 1994, section 1; Achterberg 2007, section 7.1) derives a
// column's bound from a row's activity range with the column's own term taken out:
//     a_j x_j >= row_lower - (max activity - a_j u_j).
// The subtraction is exact only on paper. When a_j u_j is large the sum has already rounded
// the other terms away, and what comes back is not the rest of the row:
//
//   1. beside 1e20 the whole rest of the row is below one unit in the last place, so the
//      "rest" is 0 and the column is given the row's bound outright (#1359);
//   2. beside 1e17 the rest is off by one, the implied bound of an integer column comes out
//      as 3.00001 instead of 3, and rounding it up to a whole number removes the optimum
//      (#1360).
// On main at 61c49acc every test below failed: 30 and `infeasible` where the optimum is 14,
// 8 where it is 7, 0 where it is 5.
//
// Every model below has integer data that a double holds exactly, so the optimum stated is
// the exact one. The models are read from LP text, which is where a bound of 1e20 or 1e30
// comes from: the LP reader keeps a number as written.

#include <string>

#include <gtest/gtest.h>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#include "support/temp_file.hpp"

namespace sankhya {
namespace {

using testing::TempFile;

Model parse(const std::string& text) {
  const TempFile file(text, ".lp");
  Model model;
  const io::ReadResult result = io::read_lp(file.path(), &model);
  EXPECT_TRUE(result.ok) << result.error;
  EXPECT_EQ(model.validate(), "");
  return model;
}

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  return options;
}

void expect_optimum(const Model& model, const Options& options, double optimum,
                    const std::string& what) {
  const Solution solved = solve(model, options);
  EXPECT_EQ(solved.status, SolveStatus::kOptimal) << what << ": " << solved.message;
  if (solved.status != SolveStatus::kOptimal) return;
  EXPECT_NEAR(solved.objective, optimum, 1e-6) << what;
  EXPECT_NEAR(solved.dual_bound, optimum, 1e-6) << what;
}

// s is 0 or in [2, 8] and z >= 10 - s. On s in [2, 8] the objective is 30 - 2 s, least at
// s = 8: 14 at (8, 2). At s = 0 it is 30.
const char* const kHugeBound = R"(Minimize
 obj: s + 3 z
Subject To
 row0: s + z >= 10
Bounds
 2 <= s <= 8
 0 <= z <= 1e20
Semi-Continuous
 s
End
)";

TEST(PropagationRounding, AHugeFiniteBoundDoesNotHideTheRestOfTheRow) {
  expect_optimum(parse(kHugeBound), quiet(), 14.0, "defaults");
}

TEST(PropagationRounding, AHugeFiniteBoundDoesNotMakeAFeasibleModelInfeasible) {
  // The same with z + w <= 8: still 14 at (8, 2, 0). A bound z >= 10 would empty the row.
  const Model model = parse(R"(Minimize
 obj: s + 3 z
Subject To
 row0: s + z >= 10
 row1: z + w <= 8
Bounds
 2 <= s <= 8
 0 <= z <= 1e20
 0 <= w <= 5
Semi-Continuous
 s
End
)");
  expect_optimum(model, quiet(), 14.0, "defaults");
}

TEST(PropagationRounding, AHugeFiniteBoundBesideAnIntegerColumn) {
  // s integer in [0, 8] instead of semi-continuous: 14 at (8, 2) again. Presolve and
  // Feasibility Jump are off so that the search stands on its own propagation.
  const Model model = parse(R"(Minimize
 obj: s + 3 z
Subject To
 row0: s + z >= 10
Bounds
 0 <= s <= 8
 0 <= z <= 1e20
General
 s
End
)");
  for (const bool incremental : {false, true}) {
    Options options = quiet();
    options.set_bool("presolve", false);
    options.set_string("mip_heur_fj", "off");
    options.set_bool("mip_incremental_propagation", incremental);
    expect_optimum(model, options, 14.0,
                   incremental ? "worklist propagation" : "sweep propagation");
  }
}

// y = 1 leaves 100000 w >= 300000, so w = 3 and the objective is 7, the row met with
// equality. y = 0 needs w >= 4: 8. w <= 2 is infeasible.
const char* const kCancellation = R"(Minimize
 obj: 2 w + y
Subject To
 r1: 100000 w + 10001 y >= 310001
Bounds
 0 <= w <= 1000000000000
 0 <= y <= 1
General
 w y
End
)";

TEST(PropagationRounding, AnImpliedIntegerBoundIsNotRoundedPastTheOptimum) {
  expect_optimum(parse(kCancellation), quiet(), 7.0, "defaults");
}

TEST(PropagationRounding, AnImpliedIntegerBoundIsNotRoundedPastTheOptimumOffTheDefaults) {
  const Model model = parse(kCancellation);
  for (const bool presolve : {true, false}) {
    for (const bool jump : {true, false}) {
      for (const int threads : {0, 1}) {  // 0: one worker per core
        for (const bool incremental : {false, true}) {
          Options options = quiet();
          options.set_bool("presolve", presolve);
          options.set_string("mip_heur_fj", jump ? "on" : "off");
          options.set_int("mip_threads", threads);
          options.set_bool("mip_incremental_propagation", incremental);
          expect_optimum(model, options, 7.0,
                         std::string("presolve ") + (presolve ? "on" : "off") +
                             ", mip_heur_fj " + (jump ? "on" : "off") + ", mip_threads " +
                             std::to_string(threads) + ", mip_incremental_propagation " +
                             (incremental ? "on" : "off"));
        }
      }
    }
  }
}

TEST(PropagationRounding, ABoxOfThirtyOrdersOfMagnitudeIsNoBound) {
  // x <= y <= 5, so the optimum is 5. 1e30 is "no bound" everywhere else in the solver;
  // read as a number, -1e30 - 5 is -1e30 and the rest of the row vanishes.
  const Model model = parse(R"(Maximize
 obj: x
Subject To
 r1: x - y <= 0
Bounds
 -1e30 <= x <= 1e30
 0 <= y <= 5
General
 y
End
)");
  expect_optimum(model, quiet(), 5.0, "defaults");
}

TEST(PropagationRounding, ABoxOfTwentyOrdersOfMagnitude) {
  // The same with a bound that is a number by every definition: still 5.
  const Model model = parse(R"(Maximize
 obj: x
Subject To
 r1: x - y <= 0
Bounds
 -1e20 <= x <= 1e20
 0 <= y <= 5
General
 y
End
)");
  expect_optimum(model, quiet(), 5.0, "defaults");
}

}  // namespace
}  // namespace sankhya
