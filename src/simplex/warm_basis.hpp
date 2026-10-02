// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a starting basis with the wrong number of basic entries, made a basis (#913).
//
// A basis carried across an edit that added or removed rows and columns
// (map_basis_by_name(), core/basis_map.cpp) has one status per entry of the new model but
// need not have exactly m of them basic: a removed basic column leaves it short, a removed
// row whose slack was nonbasic leaves it long. The simplex seeds only a basis of exactly m
// entries (Simplex::seed_basis) and would otherwise start from the slack basis, losing the
// warm start the edit was meant to keep.
//
// THE COMPLETION. A long basis parks its surplus structurals at a bound, keeping every basic
// logical (a logical is a unit column, so it never costs rank). A short one gets a
// placeholder per missing entry, an empty column, and the result is factorized with partial
// pivoting: the placeholders, and any column the edit made dependent, are exactly the
// columns no pivot reaches, and each is replaced by the logical of a row no pivot covered -
// the repair of Maros, "Computational Techniques of the Simplex Method", sec. 9.4, and Suhl
// & Suhl 1990, which Simplex::repair_basis applies to a basis that turns singular and
// crossover_guess applies to its guess. A logical is a unit vector on an uncovered row, so
// every round raises the rank and the loop ends. Nothing about the answer rests on the
// completion being a good basis: the simplex pivots from it to its own optimum, and the
// warm branch of solve() re-solves cold when a warm start reaches no verdict it can prove.
#pragma once

#include "primal_simplex.hpp"
#include "sankhya/model.hpp"
#include "sankhya/types.hpp"

namespace sankhya {

/// What complete_warm_basis() changed.
struct WarmBasisCompletion {
  Index given_basic = 0;     ///< basic entries the start named
  Index added_logicals = 0;  ///< row logicals made basic to fill a short basis or for rank
  Index parked = 0;          ///< structurals made nonbasic: surplus, or dependent
  bool complete = false;     ///< the result has m basic entries and factorizes

  [[nodiscard]] bool changed() const noexcept { return added_logicals + parked > 0; }
};

/// Bring `warm`, which has one status per column and row of `model`, to exactly
/// `model.num_rows()` basic entries that factorize, as described above. A start that
/// already has m basic entries is left exactly as it is (`complete` true, nothing changed):
/// whether it factorizes is the simplex's own first factorization to find out, as before.
[[nodiscard]] WarmBasisCompletion complete_warm_basis(const Model& model, WarmStart* warm);

}  // namespace sankhya
