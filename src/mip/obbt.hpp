// SPDX-License-Identifier: Apache-2.0
// SANKHYA - optimality-based bound tightening (OBBT) at the root (#515).
//
// Reference:
//   Gleixner, Berthold, Muller and Weltge, "Three enhancements for
//   optimization-based bound tightening", J. Global Optimization 67, 2017.
//
// For each variable x_j that is not already fixed, solve
//   min x_j  s.t. Ax <= b,  c'x <= z_cutoff   -> new lower bound
//   max x_j  s.t. Ax <= b,  c'x <= z_cutoff   -> new upper bound
// Tighter bounds produce tighter LP relaxations throughout the tree.
#pragma once

#include <limits>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::mip {

/// Result of one OBBT pass: how many bounds were tightened and how many LP
/// solves were performed.
struct ObbtResult {
  int bounds_tightened = 0;
  int lp_solves = 0;
};

/// Run OBBT on `model` at the root, before the cut round.
///
/// When `incumbent` is finite a cutoff row keeps only strictly improving points, in the
/// model's sense (c'x <= incumbent - eps minimising, c'x >= incumbent + eps maximising); the
/// caller must then keep that incumbent, since the bounds may exclude it. Otherwise only the
/// feasible region is used.  Tightened bounds are
/// written back to `model` in place.  Returns a summary for logging.
ObbtResult obbt_root(Model& model, const Options& options, Logger& logger,
                     double incumbent = std::numeric_limits<double>::infinity());

}  // namespace sankhya::mip
