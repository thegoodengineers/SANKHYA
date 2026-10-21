// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the decomposition entry point solve() calls (#525).
//
// `decomposition=off` (the default) never reaches this file. With `auto` the model is looked at
// for block-angular structure (structure.hpp) and decomposed only when the structure is strong;
// with `benders` any structure with two blocks is tried. What is decomposed is an LP that is a
// minimisation with linking COLUMNS, by Benders (benders.hpp), or - only with `dantzig-wolfe`,
// which tries any split of the columns into two or more blocks - with linking ROWS, by
// Dantzig-Wolfe (dantzig_wolfe.hpp, #1048). Everything else - an integer or quadratic model, a
// maximisation, a structure that is too weak, linking rows under `auto`, or a decomposition
// that declines or whose answer fails solve()'s measurement - is solved by the ordinary
// engines, as if the option were off. The log says which, and why.
#pragma once

#include <optional>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"

namespace sankhya::decomp {

struct BlockStructure;

/// EARLY GIVE-UP (#1048). Benders over a structure that cannot beat the monolithic solve used
/// to find that out by spending half the time limit (150 s and more on the large refinery
/// model, decomposition-b54f4b53.csv, which the monolithic dual simplex solves in under 5 s).
/// This is a forecast made from the structure alone, before any LP is solved:
///
///   - the work of a simplex solve is taken as rows x nonzeros: the iteration count grows about
///     linearly with the rows (Dantzig's observation of 2m to 3m iterations, surveyed by
///     Shamir, "The efficiency of the simplex method: a survey", Management Science 33(3),
///     1987) and each iteration touches on the order of the nonzeros;
///   - a Benders round costs the largest block alone, as if every block had a core of its own
///     and the master were free - optimistic for Benders, and independent of the thread count,
///     so the choice of method is the same on every machine;
///   - the rounds are the linking columns plus one. This is an ESTIMATE, not a bound: each
///     round adds at most one optimality cut per block, and the master's point in the L
///     linking columns is pinned by L active rows (Benders 1962; the cutting-plane view of
///     Kelley, "The cutting-plane method for solving convex programs", J. SIAM 8(4), 1960),
///     but some of those can be column bounds. On the refinery ladder
///     (decomposition-b54f4b53.csv) Benders converged in 18 rounds for 16 linking columns and
///     188 for 48 on the medium model, and in 6 for 31 on the small one split eight ways.
///
/// Benders is run only when that forecast is below the monolithic work. A wrong forecast costs
/// time and never the answer: a model it gives up on is solved by the ordinary engines, exactly
/// as with decomposition off.
struct BendersForecast {
  double rounds = 0.0;
  double decomposed = 0.0;  ///< rounds x the largest block's rows x nonzeros
  double monolithic = 0.0;  ///< rows x nonzeros of the whole model
  [[nodiscard]] bool pays() const noexcept { return decomposed < monolithic; }
};
[[nodiscard]] BendersForecast forecast_benders(const Model& model,
                                               const BlockStructure& structure);

/// The decomposed answer, or nullopt when `model` is to be solved monolithically. A solution
/// with status `optimal` has NOT been measured yet - solve() does that - and one with a limit
/// status is final.
[[nodiscard]] std::optional<Solution> solve_by_decomposition(const Model& model,
                                                             const Options& options,
                                                             SolveControl* control,
                                                             Logger& logger);

}  // namespace sankhya::decomp
