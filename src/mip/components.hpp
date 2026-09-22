// SPDX-License-Identifier: Apache-2.0
// SANKHYA - MILP engine composition (#297: "prefer composition over inheritance... MILPEngine
// with NodeSelector, BranchingStrategy, CutManager, HeuristicManager, ConflictManager,
// RelaxationEngine... individual components evolve independently").
//
// branch_and_bound.cpp ALREADY has exactly that composition, at the file level: node
// selection (branch_and_bound_node.cpp), the branching rule (inside branch_and_bound.cpp,
// option-selected), cuts (cuts.cpp, mir_cuts.cpp, combinatorial_cuts.cpp), heuristics
// (heuristics.cpp), conflict analysis (conflict.cpp), and the node LP relaxation engine
// (src/simplex/, dual or primal per mip_node_engine) each already live in their own file and
// evolve independently of the others - none of that is rewritten or wrapped in a new class
// hierarchy here, which would rewrite working, tested code for no behavioral gain.
//
// What was missing is a way to SEE that composition from outside branch_and_bound.cpp: this
// header reads the same Options keys the algorithm itself reads and reports which component
// each one selects, so a caller (the SolverEngine wrapper, a future CLI diagnostic, a test)
// can ask "what is this MILP solve actually composed of" without re-deriving the option
// names or guessing. It is a read of configuration, not a new abstraction layer the
// algorithm runs through - solve_branch_and_bound is untouched and still the only thing that
// decides what actually happens.
#pragma once

#include <string>

#include "sankhya/options.hpp"

namespace sankhya::mip {

/// The branch-and-bound components `options` selects, named the way #297's own example
/// names them. Every field is read from an Options key branch_and_bound.cpp (or a file it
/// calls into) already reads for the same purpose; this does not add a new option or change
/// what any of them do.
struct MilpComponents {
  std::string node_selection;     ///< NodeSelector: mip_node_selection (#293)
  std::string branching;          ///< BranchingStrategy: mip_branching (#69)
  std::string relaxation_engine;  ///< RelaxationEngine: mip_node_engine, the node LP (#65)
  bool cuts_enabled = false;      ///< CutManager: enable_root_cuts (#221, #358)
  /// HeuristicManager: anything beyond rounding and the fractional root dive is on, from
  /// mip_heuristics and the per-heuristic mip_heur_* switches resolved together (#290, #414)
  bool heuristics_enabled = false;
  std::string heuristics;  ///< the heuristics that run, by name, rounding first (#414)
  bool conflict_analysis_enabled = false;  ///< ConflictManager: conflict_analysis (#292)
};

/// Reads `options` and reports which MILP components it selects. Pure - no side effects, no
/// dependency on a model or a solve in progress - so it is safe to call before, during
/// (from another thread) or independently of an actual solve.
[[nodiscard]] MilpComponents describe_components(const Options& options);

}  // namespace sankhya::mip
