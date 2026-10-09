// SPDX-License-Identifier: Apache-2.0
// SANKHYA - parity rows solved over GF(2), a primal heuristic (#841).
//
// A PARITY ROW is an equality row  sum_j a_j x_j = b  whose columns are all integer with
// integer coefficients, whose rhs is an integer, and whose odd-coefficient columns are all
// binary (or fixed). Read modulo 2 the even-coefficient columns vanish and an odd a_j x_j is
// x_j, so every integer point satisfies
//
//     sum_{j : a_j odd, j binary} x_j  =  b - sum_{fixed j, a_j odd} a_j x_j    (mod 2).
//
// The rows together are a linear system over GF(2), and Gaussian elimination over GF(2)
// (any linear algebra text; for this use, the parity reasoning of Sutner, "Linear cellular
// automata and the Garden-of-Eden", Math. Intelligencer 11, 1989, on lights-out models)
// either shows it inconsistent or gives every binary assignment the rows allow: one
// particular solution plus the span of a null space of dimension k. When k is small the
// system pins the binaries down - on a lights-out model it pins every one of them - and the
// rest of the model is a small sub-MIP the search solves around that fixing.
//
// The detection is generic: nothing here knows a grid, a button or a light. A model with no
// parity row gets no candidates and costs one pass over its rows.
//
// NOTHING HERE DECIDES AN ANSWER. A fixing is a guess about the binaries; the search solves
// the restricted sub-MIP and offers its point to offer_incumbent(), which checks it against
// the original model. An inconsistent system is reported, never claimed as infeasibility.
#pragma once

#include <cstdint>
#include <vector>

#include "sankhya/model.hpp"

namespace sankhya::mip {

struct ParityFixings {
  /// The binary columns the parity rows constrain, in increasing order.
  std::vector<Index> columns;
  /// Each candidate: one 0/1 value per entry of `columns`. Empty when there are no parity
  /// rows, the system is inconsistent, or it is too large to eliminate.
  std::vector<std::vector<std::uint8_t>> candidates;
  Index parity_rows = 0;
  Index null_space_dimension = 0;  ///< k: the system allows 2^k assignments
  bool inconsistent = false;       ///< no 0/1 assignment satisfies the rows mod 2
};

/// The candidates of the parity system of `model`, at most `max_candidates` of them. The free
/// columns of the elimination take the rounding of `guide` (one value per model column), and
/// the later candidates flip them in binary-counting order, lowest free column first.
[[nodiscard]] ParityFixings parity_fixings(const Model& model, const std::vector<double>& guide,
                                           int max_candidates);

}  // namespace sankhya::mip
