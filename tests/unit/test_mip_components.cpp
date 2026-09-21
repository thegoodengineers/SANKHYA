// SPDX-License-Identifier: Apache-2.0
// SANKHYA - MILP engine composition (#297). describe_components() reads the same Options
// keys branch_and_bound.cpp itself reads; this pins that it reads the RIGHT ones and reports
// them faithfully, so a caller (the BranchAndBoundEngine wrapper, src/core/solve.cpp) can
// trust what it says about the engine's actual, unrewritten composition
// (NodeSelector/BranchingStrategy/CutManager/HeuristicManager/ConflictManager/
// RelaxationEngine).

#include <gtest/gtest.h>

#include "mip/components.hpp"
#include "sankhya/options.hpp"

namespace sankhya::mip {
namespace {

TEST(MilpComponents, DefaultsMatchWhatBranchAndBoundActuallyDefaultsTo) {
  const Options defaults;
  const MilpComponents parts = describe_components(defaults);
  EXPECT_EQ(parts.node_selection, "hybrid");
  EXPECT_EQ(parts.branching, "reliability");
  EXPECT_EQ(parts.relaxation_engine, "dual");
  EXPECT_FALSE(parts.cuts_enabled);
  EXPECT_FALSE(parts.heuristics_enabled);
  EXPECT_FALSE(parts.conflict_analysis_enabled);
}

TEST(MilpComponents, ReportsExactlyWhatWasSetNotAGuess) {
  Options options;
  options.set_string("mip_node_selection", "best-bound");
  options.set_string("mip_branching", "most-fractional");
  options.set_string("mip_node_engine", "primal");
  options.set_bool("enable_root_cuts", true);
  options.set_bool("mip_heuristics", true);
  options.set_bool("conflict_analysis", true);

  const MilpComponents parts = describe_components(options);
  EXPECT_EQ(parts.node_selection, "best-bound");
  EXPECT_EQ(parts.branching, "most-fractional");
  EXPECT_EQ(parts.relaxation_engine, "primal");
  EXPECT_TRUE(parts.cuts_enabled);
  EXPECT_TRUE(parts.heuristics_enabled);
  EXPECT_TRUE(parts.conflict_analysis_enabled);
}

}  // namespace
}  // namespace sankhya::mip
