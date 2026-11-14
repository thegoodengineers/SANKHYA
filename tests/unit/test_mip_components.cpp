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
  // On by default since the three-way at 0018254 (#415, #221): the root round proves
  // 10 of 30 where no cuts prove 9, at 1.015x the nodes. This line is the record of
  // that decision; a run that reverses it must change this expectation on purpose.
  EXPECT_TRUE(parts.cuts_enabled);
  // Parity on by default since its A/B (#841,
  // bench/results/miplib-841-parity-default-*-seeds3-fa9e8b4d.csv); Feasibility Jump
  // since the #506 A/B (#930).
  EXPECT_TRUE(parts.heuristics_enabled);
  // The rest only while the search has no incumbent (mip_heuristics=auto, the default
  // since the A/B at c77bc4f).
  EXPECT_EQ(parts.heuristics,
            "rounding, feasibility jump, parity, propagation dive, lock rounding, repair, "
            "fractional diving, coefficient diving, vector length diving, guided diving, "
            "feasibility pump, RENS; beyond the first four, only until the first incumbent");
  Options plain;
  plain.set_string("mip_heuristics", "false");
  EXPECT_EQ(describe_components(plain).heuristics,
            "rounding, feasibility jump, parity, fractional diving")
      << "the four that run once there is an incumbent, and nothing else";
  EXPECT_FALSE(parts.conflict_analysis_enabled);
}

TEST(MilpComponents, OneHeuristicOnByNameReadsAsHeuristicsOn) {
  // #414: a single mip_heur_* switch set to on, with the master switch untouched, is what
  // a one-heuristic A/B runs, and the composition report must say so.
  Options options;
  options.set_string("mip_heuristics", "false");
  options.set_string("mip_heur_rens", "on");
  const MilpComponents parts = describe_components(options);
  EXPECT_TRUE(parts.heuristics_enabled);
  EXPECT_EQ(parts.heuristics, "rounding, feasibility jump, parity, fractional diving, RENS");
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
  EXPECT_NE(parts.heuristics.find("RINS"), std::string::npos) << parts.heuristics;
  EXPECT_NE(parts.heuristics.find("guided diving"), std::string::npos) << parts.heuristics;
  EXPECT_TRUE(parts.conflict_analysis_enabled);
}

}  // namespace
}  // namespace sankhya::mip
