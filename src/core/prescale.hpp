// SPDX-License-Identifier: Apache-2.0
// SANKHYA - an exact equilibration of the MODEL, by powers of two, for the prescaled retry of
// an LP that failed numerically (#792).
//
// WHY THE MODEL AND NOT ONLY THE ENGINE. The simplex equilibrates the matrix it is handed
// (src/la/scaling.cpp), but presolve runs before it, on the model as written, and so do the
// status guard's measurements. On a model whose rows and columns were scaled by factors up to
// 2^20 (the stress set of #762), presolve's reductions and the guard's tolerances are applied
// in units where one coefficient is 1e-12 and its neighbour 1e+6. Equilibrating the model
// first, by powers of two so that the change of variables is exact in binary floating point
// (only exponents change; Curtis and Reid round their scale factors to powers of the base for
// the same reason), hands presolve and the engine a model whose numbers are of one size, and
// the answer maps back without rounding:
//
//     A' = R A S,  c' = S c,  rows R [lo, hi],  columns [l, u] / S
//     x = S x',  y = R y',  d = d' / S,  row activity = activity' / R
//
// Measured on the 18 scaled Netlib models the default path left numerical_error or feasible
// at 0e1a0d25 (a Python prototype of this function in front of the same binary): 15 solved.
//
// References: D. Ruiz, "A scaling algorithm to equilibrate both rows and columns norms in
// matrices", RAL-TR-2001-034 (2001); A. R. Curtis and J. K. Reid, "On the automatic scaling
// of matrices for Gaussian elimination", J. Inst. Maths Applics 10 (1972) 118-124.
#pragma once

#include <vector>

#include "sankhya/model.hpp"

namespace sankhya {

/// A model equilibrated by powers of two, and the factors that map an answer back.
struct PrescaledModel {
  Model model;
  std::vector<double> row;     ///< R, one power of two per row
  std::vector<double> column;  ///< S, one power of two per column
};

/// `passes` rounds of Ruiz equilibration in the infinity norm, each factor then rounded to
/// the nearest power of two. For an LP (no Hessian, no integer columns); the caller checks.
[[nodiscard]] PrescaledModel prescale_by_powers_of_two(const Model& model, int passes);

/// Map an answer of `prescaled.model` back to the model it was built from: the point, the
/// activities, both dual vectors, a Farkas vector and a ray. The basis is unchanged (a
/// diagonal change of variables keeps a basis a basis) and so is the objective (c'x' = c x).
/// Ranging, exact verification and IIS are not mapped; the caller does not retry when any
/// of them was asked for.
void unscale_solution(const PrescaledModel& prescaled, Solution* solution);

}  // namespace sankhya
