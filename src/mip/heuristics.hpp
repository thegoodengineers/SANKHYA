// SPDX-License-Identifier: Apache-2.0
// SANKHYA - MILP primal heuristics (#290, #414).
//
// A primal heuristic PROPOSES a point; it never decides one. Every candidate these functions
// return goes through BranchAndBound::offer_incumbent(), which re-checks integrality, the
// original bounds and every original row before it can become the incumbent. A heuristic with
// a bug therefore costs time, never an answer - which is the property that lets them be
// aggressive.
//
// They are pure functions of a Model and a point, with no access to the search, so each can
// be tested on its own model. The search owns the scheduling (when, how often, with what
// budget) and the statistics.
//
//   lock rounding    Achterberg, "Constraint Integer Programming", thesis, TU Berlin 2007,
//                    sec. 9.1 (simple rounding by locks): a column no row can be violated by
//                    moving down is rounded down, and symmetrically; the rest to nearest.
//   repair           a greedy one-unit shift of integer columns in the most violated row,
//                    choosing the shift that removes the most violation and, among equals,
//                    costs the objective least - a bounded local search in the spirit of
//                    Berthold's shift-and-propagate (2014), without the propagation.
//   diving           Achterberg 2007, sec. 9.2, and Berthold, "Primal heuristics for mixed
//                    integer programs", thesis, TU Berlin 2006, sec. 3.2: fix one integer
//                    column, re-solve the LP, repeat. The four rules differ only in which
//                    column and which way - fractional (the least fractional, to nearest),
//                    coefficient (the fewest row locks in the rounding direction), vector
//                    length (the least objective increase per row the column touches, the
//                    rule built for covering models) and guided (towards the incumbent).
//   RINS             Danna, Rothberg and Le Pape, "Exploring relaxation induced
//                    neighborhoods to improve MIP solutions", Math. Programming 102 (2005):
//                    fix the integer columns on which the incumbent and the node relaxation
//                    agree, and search what is left as a small sub-MIP.
//   RENS             Berthold, "RENS: the optimal rounding", Math. Programming Computation
//                    6 (2014): fix every integer column the relaxation already has integral,
//                    box every other to the two integers around its value, and search that
//                    sub-MIP - the best of every rounding of the relaxation at once.
//   feasibility pump Fischetti, Glover and Lodi, "The feasibility pump", Math. Programming
//                    104 (2005): alternate rounding the LP point and projecting the rounding
//                    back onto the LP polytope in the L1 distance, until they meet. The
//                    distance is linear only for a column rounded to one of its bounds; a
//                    general integer rounded strictly inside its bounds needs an auxiliary
//                    column (Bertacco, Fischetti and Lodi 2007) and is left out of the
//                    distance here, which the PR says.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::mip {

/// What one heuristic did over a search.
struct HeuristicStats {
  std::string name;
  Count calls = 0;
  Count found = 0;     ///< candidates proposed to offer_incumbent()
  Count improved = 0;  ///< of those, the ones it accepted: feasible, integral and better
  Count work = 0;      ///< LP solves, sub-MIP nodes or repair moves, whichever it spends
  double seconds = 0.0;
};

/// Row locks (Achterberg 2007, sec. 9.1): up[j] counts the rows that moving column j UP could
/// violate, down[j] the rows moving it down could.
struct Locks {
  std::vector<int> up;
  std::vector<int> down;
};
[[nodiscard]] Locks compute_locks(const Model& model);

/// x with every integer column rounded in the direction its locks allow, the others to the
/// nearest integer; clamped to the column bounds. Continuous columns are left as they are.
[[nodiscard]] std::vector<double> lock_round(const Model& model, const Locks& locks,
                                             const std::vector<Index>& integer_columns,
                                             const std::vector<double>& x);

/// Starting from a point whose integer columns are integral, shift integer columns one unit
/// at a time to remove row violations. Returns true when every row is satisfied to
/// `tolerance` within `max_moves` shifts; `moves` reports how many were made either way.
bool repair(const Model& model, const std::vector<Index>& integer_columns,
            std::vector<double>* x, int max_moves, double tolerance, Count* moves);

/// The rule a dive uses to pick the next column to fix and the value to fix it to. The
/// descent itself is shared; see the file comment for what each rule prefers.
enum class DiveRule { kFractional, kCoefficient, kVectorLength, kGuided };
inline constexpr std::size_t kDiveRules = 4;
[[nodiscard]] const char* to_string(DiveRule rule);

/// What a dive rule chose: the column to fix and the integer to fix it to, or column -1 when
/// x is integral on the integer columns (or, for guided diving, there is no incumbent).
struct DiveChoice {
  Index column = -1;
  double value = 0.0;
};
[[nodiscard]] DiveChoice choose_dive_column(const Model& model, const Locks& locks,
                                            const std::vector<Index>& integer_columns,
                                            const std::vector<double>& x, DiveRule rule,
                                            const std::vector<double>& incumbent,
                                            double integrality_tolerance);

/// The RINS sub-model: `model` with every integer column fixed where `relaxation` and
/// `incumbent` agree to `tolerance`. False, leaving `out` alone, when fewer than
/// `min_fixed_fraction` of the integer columns agree - a neighbourhood that large is not a
/// neighbourhood. `fixed` reports how many were fixed.
bool rins_submodel(const Model& model, const std::vector<Index>& integer_columns,
                   const std::vector<double>& relaxation, const std::vector<double>& incumbent,
                   double min_fixed_fraction, double tolerance, Model* out, Count* fixed);

/// The RENS sub-model: `model` with every integer column that is integral in `relaxation`
/// (to `tolerance`) fixed there, and every other integer column bounded to the two integers
/// around its value. False, leaving `out` alone, when fewer than `min_fixed_fraction` of the
/// integer columns are integral - that box is the model itself. `fixed` reports how many
/// were fixed.
bool rens_submodel(const Model& model, const std::vector<Index>& integer_columns,
                   const std::vector<double>& relaxation, double min_fixed_fraction,
                   double tolerance, Model* out, Count* fixed);

/// The feasibility pump from an LP point `start`. Returns the first point whose integer
/// columns are integral and which the pump's own LP says is feasible, or an empty vector.
/// `lp_solves` reports the LP projections spent. `lp_options` is what each projection is
/// solved with; the caller sets its limits.
[[nodiscard]] std::vector<double> feasibility_pump(const Model& model,
                                                   const std::vector<Index>& integer_columns,
                                                   const std::vector<double>& start,
                                                   const Options& lp_options, int max_rounds,
                                                   double integrality_tolerance,
                                                   Count* lp_solves);

/// Which heuristics a search runs, and with what budgets, resolved from the options once
/// (#414). Each heuristic has its own mip_heur_* switch: `auto` follows mip_heuristics, `on`
/// and `off` decide alone, so a measurement can run exactly one of them. Every budget is in
/// units the search counts (LP re-solves, sub-MIP nodes, pump rounds); the sub-MIP
/// heuristics also keep a budget in seconds unless the solve is deterministic.
struct HeuristicSchedule {
  bool lock_rounding = false;
  bool repair = false;
  bool pump = false;
  bool rins = false;
  bool rens = false;
  bool dive[kDiveRules] = {false, false, false, false};  ///< by DiveRule
  bool dive_backtrack = false;
  Count rins_frequency = 0;
  Count rins_nodes = 0;
  Count rens_nodes = 0;
  Count dive_frequency = 0;  ///< 0: the dives run at the root only
  int dive_lp_resolves = 0;
  int pump_rounds = 0;
  bool seconds_budgets = true;  ///< false under deterministic=true (#288's rule)

  [[nodiscard]] static HeuristicSchedule from(const Options& options);
  /// True when anything beyond rounding and the fractional root dive - the two every
  /// benchmark CSV was measured with - is on.
  [[nodiscard]] bool any_optional() const;
  /// The heuristics that run, by name, comma separated, rounding first.
  [[nodiscard]] std::string names() const;
};

}  // namespace sankhya::mip
