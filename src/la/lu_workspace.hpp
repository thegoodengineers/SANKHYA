// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the elimination's scratch state, shared by factorize() (lu.cpp), which loads it,
// and eliminate() (lu_eliminate.cpp), which consumes it. Internal to src/la; not installed.
#pragma once

#include <vector>

#include "lu.hpp"

namespace sankhya {

// =========================================================================================
// Workspace - the active submatrix during elimination
// =========================================================================================

struct SparseLu::Workspace {
  Index m = 0;

  /// Values live COLUMN-wise. The elimination updates whole columns at a time (see
  /// eliminate()), so this is the orientation that keeps the inner loop contiguous.
  std::vector<std::vector<Index>> col_rows;
  std::vector<std::vector<double>> col_values;

  /// Rows carry the PATTERN only, and it is allowed to go stale: an entry removed by
  /// cancellation is not hunted down here. Every consumer re-checks against the column
  /// storage, which is authoritative. Chasing exact row patterns costs more than it saves.
  std::vector<std::vector<Index>> row_cols;

  /// Exact active counts. These drive Markowitz, so unlike row_cols they are maintained
  /// precisely on every insertion and deletion.
  std::vector<Index> row_count;
  std::vector<Index> col_count;

  std::vector<char> row_active;
  std::vector<char> col_active;

  /// Dense scatter/gather buffers indexed by ROW, used one column at a time.
  std::vector<double> acc;
  std::vector<char> acc_present;

  /// Multipliers of the current step, and the pivot row's entries collected as the update
  /// sweeps its columns.
  std::vector<Index> mult_rows;
  std::vector<double> mult_values;
  std::vector<Index> u_cols;
  std::vector<double> u_vals;

  /// The pivot row's active columns, DEDUPLICATED. row_cols is allowed to go stale, and a
  /// column removed by cancellation and later re-created by fill appears in it twice. Both
  /// the count bookkeeping and the update below must visit each column exactly once, so the
  /// list is built through a marker array rather than iterated in place.
  std::vector<Index> pivot_row_columns;
  std::vector<char> column_seen;

  /// The active rows of the column being updated, recorded as it is scattered, so that the
  /// gather afterwards visits those rows and the multiplier rows and nothing else. It used
  /// to sweep all m rows for every column of every step - the other half of #210.
  std::vector<Index> old_rows;

  // ---- the singleton fast path (#463) ---------------------------------------------------
  // Suhl & Suhl (1990) keep the active submatrix both row-wise and column-wise so that a
  // column-singleton pivot, which has no multipliers and changes no value anywhere, can take
  // its U row from the row copy and never open the columns it crosses. Koberstein (2005,
  // sec. 5.3) describes the same pass. The general update below reads every such column in
  // full to find one coefficient and write the rest back unchanged, which on a basis with
  // one dense column is (column length) work per pivot row that crosses it: 36.8 s for one
  // factorization of bdry2 (#463).

  /// Values parallel to row_cols, valid for row i only while row_values_valid[i] is set.
  /// A row stays valid for as long as no elimination step has changed one of its values:
  /// the only thing that does is being a multiplier row of a step whose pivot row has other
  /// columns, and that step clears the flag. While valid, row_cols[i] is also EXACT - no
  /// stale entry (those come from cancellation, which only touches multiplier rows), no
  /// duplicate (those come from fill, likewise) - so row i's entry in an active column j is
  /// in column j's storage with exactly the value held here.
  std::vector<std::vector<double>> row_values;
  std::vector<char> row_values_valid;

  /// max |a_ij| over the ACTIVE rows of column j, for the threshold test, valid while
  /// col_max_valid[j] is set. Recomputed whenever the column is rewritten, kept across the
  /// retirement of a row whose entry was strictly smaller than it (the maximum of what
  /// remains is then unchanged, exactly), dropped otherwise. max is exact in floating point
  /// whatever the order, so the cached value is the value a rescan would produce.
  std::vector<double> col_max;
  std::vector<char> col_max_valid;

  /// Off when the input repeats a row within a column (never the case for a basis taken
  /// from CSC storage), or when the reference elimination is asked for: every decision is
  /// then taken from the column storage exactly as before #463.
  bool fast = false;

  void init(Index dimension) {
    m = dimension;
    const auto u = static_cast<std::size_t>(dimension);
    col_rows.assign(u, {});
    col_values.assign(u, {});
    row_cols.assign(u, {});
    row_count.assign(u, 0);
    col_count.assign(u, 0);
    row_active.assign(u, 1);
    col_active.assign(u, 1);
    acc.assign(u, 0.0);
    acc_present.assign(u, 0);
    column_seen.assign(u, 0);
    row_values.assign(u, {});
    row_values_valid.assign(u, 1);
    col_max.assign(u, 0.0);
    col_max_valid.assign(u, 0);
  }
};

}  // namespace sankhya
