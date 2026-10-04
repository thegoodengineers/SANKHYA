// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the decomposition entry point solve() calls (#525).
//
// `decomposition=off` (the default) never reaches this file. With `auto` the model is looked at
// for block-angular structure (structure.hpp) and decomposed only when the structure is strong;
// with `benders` any structure with two blocks is tried. What is decomposed is an LP that is a
// minimisation with linking COLUMNS, by Benders (benders.hpp). Everything else - an integer or
// quadratic model, a maximisation, a structure that is too weak, linking ROWS (the form
// Dantzig-Wolfe is for, which is detected and said so in the log but not solved), or a
// decomposition that declines or whose answer fails solve()'s measurement - is solved by the
// ordinary engines, as if the option were off. The log says which, and why.
#pragma once

#include <optional>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"

namespace sankhya::decomp {

/// The decomposed answer, or nullopt when `model` is to be solved monolithically. A solution
/// with status `optimal` has NOT been measured yet - solve() does that - and one with a limit
/// status is final.
[[nodiscard]] std::optional<Solution> solve_by_decomposition(const Model& model,
                                                             const Options& options,
                                                             SolveControl* control,
                                                             Logger& logger);

}  // namespace sankhya::decomp
