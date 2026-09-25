// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the size of the normal equations, known before they are built (#467).
//
// A Theta A^T has a nonzero at (r, i) whenever some column of A meets both rows, so its
// pattern - and its size - follows from the pattern of A alone. A column with c entries
// makes a c x c dense block: c (c + 1) / 2 lower-triangle entries on its own. On bdry2 one
// column has 126,002 entries and the product would hold about 7.9e9 nonzeros; on Linf_520c
// two columns with 30,479 and 4,022 entries take it to 4.7e8, and the assembly ran for
// 112 s and most of the machine's memory before the ordering could say the factor was too
// large. This counts first, cheaply when the answer is obvious and exactly when it is not.
//
// Reference: Davis, "Direct Methods for Sparse Linear Systems", SIAM (2006), sec. 2.8 on the
// pattern of a matrix product, and Andersen, Gondzio, Meszaros & Xu, "Implementation of
// interior point methods for large scale linear programming" (Kluwer 1996), sec. 5 on dense
// columns and the fill they cause in A Theta A^T.
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "sankhya/sparse.hpp"

namespace sankhya {

/// What predict_normal_nonzeros() established about the lower triangle (diagonal included)
/// of A Theta A^T + D over the columns it was told to keep.
struct NormalPrediction {
  /// The number of lower-triangle nonzeros: exact when `exact`, otherwise a bound - an
  /// upper bound when it is within the cap, a lower bound when `over_cap`.
  std::int64_t nonzeros = 0;
  bool exact = false;
  /// The count passed the cap; `nonzeros` is then a lower bound, and the counting stopped
  /// as soon as it knew.
  bool over_cap = false;
  /// The deadline fired before an answer was known.
  bool stopped = false;
};

/// Count the lower triangle of A diag(theta) A^T + D, D diagonal, with the columns where
/// `skip[j]` is nonzero left out (an empty `skip` keeps every column). Every diagonal entry
/// is counted, since D puts one there. `cap` < 0 means no cap.
///
/// Three stages, each as cheap as the answer allows: the largest single column gives a
/// lower bound (its dense block), the sum over columns an upper bound, and only when the
/// cap lies between the two is the pattern counted row by row with a marker array - the
/// work of the assembly without its storage, stopped the moment the count passes the cap.
[[nodiscard]] NormalPrediction predict_normal_nonzeros(
    const SparseMatrix& a, const std::vector<char>& skip, std::int64_t cap,
    const std::function<bool()>& should_stop = {});

}  // namespace sankhya
