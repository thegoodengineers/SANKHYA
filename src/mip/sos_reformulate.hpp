// SPDX-License-Identifier: Apache-2.0
// SANKHYA - semi-continuous columns and special ordered sets written as binaries and big-M
// rows (#754, the `sos_reformulate` option, off by default).
//
// The native route (branch_and_bound_sos.cpp) branches on the conditions themselves. This is
// the other route, kept so the two can be A/B'd on the same model
// (bench/runners/sos_sc_ab.py): every condition whose columns have finite bounds becomes a
// plain MILP the ordinary search solves, with presolve and every other part of it on.
//
//   semi-continuous x in {0} or [l, u]:   z binary,  x - u z <= 0,  x - l z >= 0,  x >= 0
//   SOS1 x_1..x_n:                        z_k binary per member,  sum z <= 1,
//                                         x_k - u_k z_k <= 0,  x_k - l_k z_k >= 0 (l_k < 0)
//   SOS2 x_1..x_n:                        z_k binary per adjacent pair (k, k+1),  sum z <= 1,
//                                         x_k - u_k (z_{k-1} + z_k) <= 0, and the same with
//                                         l_k when l_k < 0
//
// The big-M of each row is the column's own bound, the tightest that is valid. A condition
// with an infinite bound on a column it constrains has no such M and stays in the model for
// the native branching. A semi-continuous column with l = 0, an SOS1 of one member and an
// SOS2 of two are satisfied by every point and are dropped. The added binaries carry no cost,
// so the reformulated objective is the original one.
//
// Reference: the standard formulations, e.g. Nemhauser and Wolsey, "Integer and
// Combinatorial Optimization" (Wiley, 1988), for the fixed-charge link a semi-continuous
// column is, and Vielma, Ahmed and Nemhauser, "Mixed-integer models for nonseparable
// piecewise-linear optimization: unifying framework and extensions", Operations Research
// 58(2), 2010, for the SOS2 "convex combination" model.
#pragma once

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"

namespace sankhya::mip {

struct ScSosReformulation {
  Model model;  ///< the reformulated model; the original's columns and rows come first
  Index original_cols = 0;
  Index original_rows = 0;
  Count semicontinuous_written = 0;  ///< columns given a binary and two rows
  Count sets_written = 0;            ///< sets given binaries and rows
  Count dropped = 0;                 ///< conditions every point satisfies, removed
  Count kept = 0;                    ///< conditions left native (an infinite bound)
  Count binaries = 0;
  Count rows = 0;
  [[nodiscard]] bool changed() const noexcept {
    return semicontinuous_written + sets_written + dropped > 0;
  }
};

/// Write `model`'s finite-bounded semi-continuous columns and sets as above.
[[nodiscard]] ScSosReformulation reformulate_sc_sos(const Model& model);

/// The answer to `r.model` as an answer to `original`: the added columns and rows cut off,
/// everything measured again against `original`, the message saying which route ran.
[[nodiscard]] Solution restrict_to_original(const Model& original, const ScSosReformulation& r,
                                            Solution solution);

}  // namespace sankhya::mip
