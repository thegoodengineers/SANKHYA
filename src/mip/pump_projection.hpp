// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the feasibility pump's projection LP, shared by the simplex pump (heuristics.cpp,
// #290) and the PDHG pump (pdhg_heuristics.cpp, #509). Written from the papers; no solver
// source was read.
//
//   Fischetti, Glover and Lodi, "The feasibility pump", Math. Programming 104 (2005): the
//   L1 distance to a rounding r is linear for a column whose integer domain is its two
//   integral bounds, x_j - l_j at r_j = l_j and u_j - x_j at r_j = u_j.
//   Bertacco, Fischetti and Lodi, "A feasibility pump heuristic for general mixed-integer
//   problems", Discrete Optimization 4 (2007): a GENERAL integer column takes one auxiliary
//   column d_k >= 0 and two rows d_k - x_j >= -r_j and d_k + x_j >= r_j, so d_k >= |x_j - r_j|,
//   and cost 1 on d_k measures its distance wherever r_j lies.
//
// The LP is built once; round by round only the two-valued columns' costs and the auxiliary
// rows' lower sides move, so its matrix stays the same.
#pragma once

#include <vector>

#include "sankhya/model.hpp"

namespace sankhya::mip {

struct PumpProjection {
  /// The model's rows and columns, every column continuous, then one auxiliary column per
  /// general integer column and its two rows after the model's.
  Model lp;
  /// Per model column: -2 not an integer column or one with a single integer in its domain
  /// (distance 0), -1 two-valued with integral bounds, k >= 0 the index of its auxiliary.
  std::vector<Index> kind;
};

[[nodiscard]] PumpProjection pump_projection(const Model& model,
                                             const std::vector<Index>& integer_columns,
                                             double integrality_tolerance);

/// Set the projection's objective to the L1 distance to `rounding` (one value per model
/// column, integral on the integer columns).
void set_pump_target(PumpProjection* projection, const Model& model,
                     const std::vector<Index>& integer_columns,
                     const std::vector<double>& rounding);

}  // namespace sankhya::mip
