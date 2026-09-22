// SPDX-License-Identifier: Apache-2.0
// SANKHYA - formulation symmetry (#413): detection of the permutations of a model's columns
// and rows that leave the model unchanged, and the ordering rows a search may add on their
// account.
//
// A MILP's formulation symmetry is the automorphism group of its coloured bipartite graph:
// one vertex per column coloured by (cost, bounds, type), one per row coloured by its
// bounds, an edge per matrix entry coloured by the coefficient (Margot 2010). A permutation
// in that group maps every feasible point to a feasible point of the same objective, so the
// search only needs one point per orbit. Detection here is McKay's individualisation and
// refinement (McKay and Piperno 2014) in its plainest form: colour refinement to a stable
// partition, then a search that individualises one vertex of the first non-singleton cell,
// refines, and compares the leaf it reaches with the leaf reached from each other vertex
// of that cell - a matching pair of discrete leaves is a candidate permutation. EVERY
// candidate is verified against the model entry by entry before it is reported, so a
// generator this file returns is an automorphism whatever the search did; the search's
// budget and its leftmost-leaf-only comparison decide only how many it finds.
#pragma once

#include <vector>

#include "sankhya/model.hpp"

namespace sankhya::mip {

/// A permutation of the model: `columns[j]` is where column j goes, `rows[i]` where row i
/// goes. Applied to a point x it gives x' with x'[columns[j]] = x[j].
struct Permutation {
  std::vector<Index> columns;
  std::vector<Index> rows;
};

/// What detection found.
struct SymmetryGroup {
  /// Verified automorphisms, each moving at least one column.
  std::vector<Permutation> generators;
  /// The orbit each column lies in under the generators: a representative column index,
  /// the column itself when nothing moves it.
  std::vector<Index> orbit_of;
  Count nontrivial_orbits = 0;  ///< orbits with more than one column
  Index largest_orbit = 1;      ///< columns in the largest orbit
  Count search_nodes = 0;       ///< individualisations the search performed
  Count refinements = 0;        ///< refinement rounds, over the whole detection
  /// True when the search stopped at its node budget: the generators are still exact, the
  /// group they generate may be a proper subgroup.
  bool budget_exhausted = false;
};

/// Detect the formulation symmetry of `model`, spending at most `search_limit`
/// individualisations. A model whose refined partition is already discrete costs one
/// refinement and no search.
[[nodiscard]] SymmetryGroup detect_symmetry(const Model& model, Count search_limit);

/// Is `p` an automorphism of `model`: a bijection on columns and on rows that preserves
/// every cost, bound, type and matrix entry exactly? The check every generator passed.
[[nodiscard]] bool is_automorphism(const Model& model, const Permutation& p);

/// The ordering rows a search may add for the generators (Liberti 2012): for a generator
/// whose first moved column is i, mapping it to k > i, the row x_i - x_k <= 0 holds for the
/// lexicographically smallest point of every orbit, so at least one optimum of every orbit
/// survives it. Each pair (i, k) once; returned as (i, k).
[[nodiscard]] std::vector<std::pair<Index, Index>> ordering_rows(const SymmetryGroup& group);

}  // namespace sankhya::mip
