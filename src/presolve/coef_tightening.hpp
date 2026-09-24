// SPDX-License-Identifier: Apache-2.0
// SANKHYA - coefficient tightening and big-M strengthening on integer rows (#511).
//
// References, written from the literature:
//   Savelsbergh, "Preprocessing and probing techniques for mixed integer programming
//     problems", ORSA J. Computing 6(4) (1994), sec. 3 - coefficient reduction on a row
//     with a binary column, and the bound propagation that feeds it
//   Achterberg, Bixby, Gu, Rothberg & Weninger, "Presolve reductions in mixed integer
//     programming", INFORMS J. Computing 32(2) (2020), sec. 3 - the same reduction for a
//     general integer column, and the rule that a propagated bound must stay in the model
//     the tightened row relies on
//
// WHAT IT DOES. A row a^T x <= b whose integer column x_j has a coefficient so large that
// the row can never bind while x_j sits one step away from the bound that maximises a_j x_j
// carries a coefficient that is loose: the LP relaxation lets x_j take fractional values the
// integer points never need. With M the row's maximum activity over the column box and
// g = M - b > 0, every integer column with |a_j| > g has its coefficient reduced to g (sign
// kept) and b reduced by (|a_j| - g) times the bound x_j maximises at. On x <= M y with x
// in [0, U] and U < M this turns the row into x <= U y: big-M strengthening is the special
// case. g is invariant under the change (M and b move by the same amount), so every such
// column of the row is tightened in one sweep against the same g.
//
// WHY THE INTEGER POINTS ARE UNCHANGED. At x_j on the maximising bound the new row is the
// old row rewritten (both sides moved by the same amount). A step t >= 1 away, the old row
// is slack everywhere on the box (|a_j| > g), and the new one needs M - |a_j| t <= b - d t
// with d = |a_j| - g, i.e. g <= g t, which holds for t >= 1. Any d in [0, |a_j| - g] keeps
// this, which is what makes the floating-point margin safe: g is rounded UP, never down.
// The argument holds on the column box, so every bound it used is written into the model;
// bound propagation below writes its bounds there for the same reason. No postsolve record
// is needed: the reduced model has the same integer points and the same objective, and the
// point it returns is re-measured against the original model like every other.
//
// ONLY FOR A MODEL WITH INTEGER COLUMNS. The reduction changes a row's coefficients and
// the LP duals of the reduced model then belong to a different matrix; a MILP reports no
// duals (solve() does not check them), an LP would, so presolve calls this only when
// integrality is present.
#pragma once

#include "sankhya/model.hpp"

namespace sankhya::presolve {

/// What coefficient tightening did to a model.
struct CoefficientTighteningStats {
  Count coefficients_tightened = 0;  ///< integer-column coefficients reduced
  Count rows_tightened = 0;          ///< rows with at least one coefficient reduced
  Count bounds_tightened = 0;        ///< column bounds tightened by propagation
  Count rounds = 0;                  ///< propagation plus tightening rounds run
  /// Propagation found a column whose bounds would cross: the model has no point. The
  /// reduction stops there and leaves the model for the engine to prove it; nothing
  /// crossing is written.
  bool found_crossing_bounds = false;
};

/// Tighten `model` in place: rows, their right-hand sides and column bounds change, the
/// dimensions, the objective and the integer points do not. `model->matrix` must be frozen,
/// and stays frozen.
CoefficientTighteningStats tighten_coefficients(Model* model);

}  // namespace sankhya::presolve
