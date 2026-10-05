// SPDX-License-Identifier: Apache-2.0
// SANKHYA - detecting block-angular structure in a constraint matrix (#525).
//
// WHAT IS LOOKED FOR. Multi-period and multi-site planning models are independent blocks joined
// by a few things they share. The joining part comes in two forms, and each has its method:
//
//   LINKING COLUMNS. Remove a few columns (first-stage decisions, stocks carried between
//   periods) and the rows fall apart into groups no column joins: every other column appears
//   in the rows of one group only. Benders decomposition fixes the linking columns and solves
//   the groups independently (benders.hpp).
//   LINKING ROWS. Remove a few rows (a shared crude, a shared capacity) and the columns fall
//   apart into groups no row joins. Dantzig-Wolfe decomposition prices a column for each group
//   against the linking rows. Detected and reported here; NOT solved by this module's caller.
//
// HOW. The matrix is a hypergraph: for linking columns the vertices are the ROWS and each
// COLUMN is a net over the rows it appears in; for linking rows the roles are swapped.
// Splitting the vertices into blocks so that few nets span blocks is balanced hypergraph
// partitioning, and a net that spans blocks is exactly a linking column (row). The partition is
// by recursive bisection; each bisection starts from a breadth-first ordering of the hypergraph
// from a pseudo-peripheral vertex (George and Liu, "Computer Solution of Large Sparse Positive
// Definite Systems", 1981, the starting vertex of a level structure) cut at half the weight,
// and is improved by Fiduccia-Mattheyses passes ("A linear-time heuristic for improving network
// partitions", 19th Design Automation Conference, 1982) under a balance bound. Recursive
// bisection of a hypergraph by net cut: Karypis, Aggarwal, Kumar and Shekhar, "Multilevel
// hypergraph partitioning: applications in VLSI domain", IEEE Trans. VLSI Systems 7(1), 1999,
// without the multilevel coarsening - the matrices here are far below the size that needs it,
// and above max_refine_vertices the refinement passes are skipped rather than made quadratic.
//
// THIS IS A HEURISTIC AND CLAIMS NOTHING ABOUT OPTIMALITY: a partition is valid whatever its
// quality (every column that is not linking is in the rows of exactly one block, which is
// checked structurally by the caller before it relies on it), and a poor one is simply rejected
// as not strong enough.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "sankhya/model.hpp"

namespace sankhya::decomp {

enum class Linking : std::uint8_t { kColumns, kRows };
[[nodiscard]] const char* to_string(Linking linking) noexcept;

/// `row_block` / `col_block` entry: the item belongs to no single block.
inline constexpr Index kLinking = -1;
/// `row_block` / `col_block` entry: the item is in no row (column), so nothing joins it to a
/// block.
inline constexpr Index kUnattached = -2;

struct BlockStructure {
  Linking kind = Linking::kColumns;
  Index blocks = 0;
  /// Per row: its block, kLinking for a row no single block owns (a linking row; or, for
  /// linking columns, a row whose columns are all linking), kUnattached for an empty row.
  std::vector<Index> row_block;
  /// Per column: its block, kLinking for a column in more than one block's rows, kUnattached
  /// for a column in no row.
  std::vector<Index> col_block;
  /// How many rows / columns each block owns.
  std::vector<Index> block_rows, block_columns;
  Index linking_rows = 0;
  Index linking_columns = 0;
  /// The linking part's share of the items it is counted over (columns that are in some row for
  /// linking columns; rows that have an entry for linking rows).
  double linking_fraction = 0.0;
  /// The largest block's share of the vertices that are in a block (rows for linking columns).
  double largest_share = 0.0;
};

struct DetectionOptions {
  /// Blocks to split into; 0 tries 4, then 2.
  Index blocks = 0;
  /// A structure whose linking part is a larger fraction than this is not strong.
  double max_linking_fraction = 0.15;
  /// ... or whose largest block holds a larger share of the blocked vertices than this.
  double max_largest_share = 0.8;
  /// Above this many vertices a bisection is not refined (the order's half split stands).
  Index max_refine_vertices = 5000;
};

struct Detection {
  bool accepted = false;
  BlockStructure structure;
  /// Why not, when `accepted` is false; what was found, when it is true.
  std::string description;
};

/// The best structure of `kind` in `model`'s constraint matrix, accepted only when it has at
/// least two blocks and a linking part and balance within `options`.
[[nodiscard]] Detection detect_block_structure(const Model& model, Linking kind,
                                               const DetectionOptions& options);

}  // namespace sankhya::decomp
