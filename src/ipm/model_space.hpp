// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the interior point's iterate measured in the model's own units (#582).
//
// The interior point iterates on the Ruiz-scaled model Ahat = Dr A Dc (la/scaling.hpp) and
// its convergence test lives in that space. The status guard in solve() measures the answer
// in the model's units, with Solution::recompute_quality's per-term denominators. On a badly
// scaled model the two disagree by the scale factors of the worst rows and columns: on
// irish-electricity a scaled primal residual of 1.2e-8 was 1.0e-4 in the model. The
// function here makes the guard's measurement on the scaled iterate without leaving the
// scaled space: every model quantity is a scaled one times a known diagonal factor.
//
// The mapping (la/scaling.hpp): x = Dc xhat, a row activity is (Ahat xhat)_i / Dr_i,
// y = Dr yhat, a structural's reduced cost is dhat / Dc, and the scaled dual residual of a
// structural is Dc times the model's. A complementarity product |d| * distance is the same
// number in both spaces; only its denominator moves.
#pragma once

#include <vector>

#include "la/scaling.hpp"
#include "sankhya/model.hpp"

namespace sankhya::ipm {

/// The iterate as the interior point holds it, in scaled space and in the bounded form
/// [A | -I] [x; s] = 0 over total = n + m variables, structurals first. `scaling` null means
/// the model was not scaled (every factor 1). Every pointer must be set.
struct ScaledIterate {
  const SparseMatrix* matrix = nullptr;        ///< Ahat, m x n
  const std::vector<double>* cost = nullptr;   ///< min-sense scaled cost, n or more entries
  const std::vector<double>* lower = nullptr;  ///< scaled bounds over total (rows: row bounds)
  const std::vector<double>* upper = nullptr;
  const std::vector<double>* x = nullptr;   ///< total; only the n structurals are read
  const std::vector<double>* y = nullptr;   ///< m, min sense
  const std::vector<double>* zl = nullptr;  ///< total; only the n structurals are read
  const std::vector<double>* zu = nullptr;
  const Scaling* scaling = nullptr;
};

/// What the status guard would measure on the point the engine would report, in the
/// model's units and with the guard's denominators (Solution::recompute_quality).
struct ModelSpaceMeasure {
  /// primal_infeasibility_scaled: each row's violation over max(1, the largest |A_ij x_j|
  /// in the row), each column bound's over max(1, |x_j|).
  double primal = 0.0;
  /// dual_infeasibility_scaled: the consistency of d = zl - zu with c - A^T y over the
  /// column's terms, the sign of a price or reduced cost against a bound that does not
  /// exist, and each relative complementarity product.
  double dual = 0.0;
  /// The part of `dual` that is a residual: consistency and sign conditions, without the
  /// complementarity products. The products |d| * distance are the same number in scaled and
  /// model units - only their denominators move - and the loop drives them down itself; the
  /// residual is what the scaling and the loop's global norms can hide (#582).
  double dual_residual = 0.0;
  /// complementarity_violation: the largest absolute |multiplier| * distance to the nearest
  /// bound, which the guard compares with tol::kComplementarity on its own.
  double complementarity = 0.0;
  /// Per row, the relative primal violation behind `primal` (for the worst-rows table).
  std::vector<double> row_violation;
  /// Per variable over total, the relative dual violation behind `dual`.
  std::vector<double> dual_violation;
};

/// One pass over the matrix for the rows, one for the columns: the cost of a residual.
[[nodiscard]] ModelSpaceMeasure measure_in_model_space(const ScaledIterate& it);

}  // namespace sankhya::ipm
