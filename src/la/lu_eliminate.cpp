// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the Markowitz elimination behind SparseLu::factorize(). See lu.hpp for the
// references and lu.cpp for the derivation of the two solves the factors feed.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#include "lu.hpp"
#include "lu_workspace.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya {
namespace {

/// How many low-count columns and rows the pivot search inspects before settling. Suhl &
/// Suhl report that a small budget loses almost nothing against an exhaustive Markowitz
/// search while removing its quadratic cost; 4 is their recommendation and the value every
/// production code has converged on.
constexpr Index kCandidateBudget = 4;

/// Entries whose magnitude falls below this after an update are treated as exact
/// cancellation and removed. Keeping them would inflate the fill counts that drive the
/// refactorization trigger, and they carry no information.
constexpr double kDropTolerance = tol::kZeroDrop;

/// The singleton fast path (#463) leaves a retired row's entry in a column's storage and
/// compacts once the dead entries outnumber the live ones by this much. A storage bound, not
/// a numerical tolerance: it never changes which entries are live, only when the dead ones
/// are swept, and small enough that a short column is swept almost every time.
constexpr Index kCompactionSlack = 8;

/// Rows or columns of the active submatrix grouped by count, as intrusive doubly linked
/// lists: one node per index, one list per count, every index in at most one list. Moving
/// an index between counts is O(1) and leaves nothing behind, so the pivot search walks
/// live candidates only.
///
/// WHY LISTS AND NOT VECTORS (#210). The buckets used to be vectors that were appended to
/// on every count change and never purged; a pivoted column's entry stayed in the bucket it
/// was found in. The search starts from the lowest counts at every step, and those are
/// exactly the buckets that fill with retired singletons - so the walk past stale entries
/// grew with the number of steps taken, and one factorization of an 18,000-row basis spent
/// 120 ms choosing pivots against 25 ms of arithmetic.
class CountLists {
 public:
  explicit CountLists(Index m)
      : head_(static_cast<std::size_t>(m) + 1, -1),
        tail_(static_cast<std::size_t>(m) + 1, -1),
        next_(static_cast<std::size_t>(m), -1),
        prev_(static_cast<std::size_t>(m), -1),
        count_(static_cast<std::size_t>(m), -1) {}

  [[nodiscard]] Index first(Index count) const {
    return head_[static_cast<std::size_t>(count)];
  }
  [[nodiscard]] Index after(Index index) const {
    return next_[static_cast<std::size_t>(index)];
  }

  /// Take `index` out of whichever list it is in, if any.
  void remove(Index index) {
    const auto u = static_cast<std::size_t>(index);
    const Index count = count_[u];
    if (count < 0) return;
    const auto c = static_cast<std::size_t>(count);
    const Index p = prev_[u];
    const Index n = next_[u];
    if (p >= 0) {
      next_[static_cast<std::size_t>(p)] = n;
    } else {
      head_[c] = n;
    }
    if (n >= 0) {
      prev_[static_cast<std::size_t>(n)] = p;
    } else {
      tail_[c] = p;
    }
    count_[u] = -1;
    prev_[u] = -1;
    next_[u] = -1;
  }

  /// File `index` under `count`, at the back of that list. An index already filed under
  /// this count keeps its place.
  void place(Index index, Index count) {
    const auto u = static_cast<std::size_t>(index);
    if (count_[u] == count) return;
    remove(index);
    const auto c = static_cast<std::size_t>(count);
    count_[u] = count;
    prev_[u] = tail_[c];
    next_[u] = -1;
    if (tail_[c] >= 0) {
      next_[static_cast<std::size_t>(tail_[c])] = index;
    } else {
      head_[c] = index;
    }
    tail_[c] = index;
  }

 private:
  std::vector<Index> head_, tail_, next_, prev_, count_;
};

}  // namespace

bool SparseLu::eliminate(Workspace& w, double pivot_tolerance, double threshold,
                         const ShouldStop& should_stop) {
  const Index m = w.m;
  pivot_row_.reserve(static_cast<std::size_t>(m));
  pivot_col_.reserve(static_cast<std::size_t>(m));
  pivot_value_.reserve(static_cast<std::size_t>(m));

  smallest_pivot_ = std::numeric_limits<double>::max();
  largest_pivot_ = 0.0;

  // Active rows and columns filed by count, so the pivot search can start from the
  // sparsest without scanning everything. A retired row or column is taken out of its list
  // and a changed count moves it, so the lists hold live candidates only (see CountLists).
  CountLists col_lists(m);
  CountLists row_lists(m);
  for (Index j = 0; j < m; ++j) col_lists.place(j, w.col_count[static_cast<std::size_t>(j)]);
  for (Index i = 0; i < m; ++i) row_lists.place(i, w.row_count[static_cast<std::size_t>(i)]);

  // A count that drifts out of [0, m] is a bookkeeping bug, and the symptom is not a wrong
  // answer but an out-of-bounds write: a negative count casts to a huge size_t and indexes
  // past the end of the list heads. Assert on the way in rather than segfaulting later at
  // a place that says nothing about the cause.
  const auto rebucket_column = [&](Index j) {
    const Index count = w.col_count[static_cast<std::size_t>(j)];
    assert(count >= 0 && count <= m && "column count out of range");
    col_lists.place(j, count);
  };
  const auto rebucket_row = [&](Index i) {
    const Index count = w.row_count[static_cast<std::size_t>(i)];
    assert(count >= 0 && count <= m && "row count out of range");
    row_lists.place(i, count);
  };

  // ---- commit a step's factors ------------------------------------------------------------
  const auto commit_step = [&] {
    for (std::size_t t = 0; t < w.mult_rows.size(); ++t) {
      l_rows_.push_back(w.mult_rows[t]);
      l_values_.push_back(w.mult_values[t]);
    }
    l_start_.push_back(static_cast<Index>(l_rows_.size()));

    for (std::size_t t = 0; t < w.u_cols.size(); ++t) {
      u_steps_.push_back(w.u_cols[t]);  // still a COLUMN here; translated in factorize()
      u_values_.push_back(w.u_vals[t]);
    }
    u_start_.push_back(static_cast<Index>(u_steps_.size()));
  };

  // max |a_ij| over the active rows of column j, from the cache when it holds (#463).
  const auto active_column_max = [&](Index j) {
    const auto uj = static_cast<std::size_t>(j);
    if (w.fast && w.col_max_valid[uj] != 0) return w.col_max[uj];
    double column_max = 0.0;
    for (std::size_t t = 0; t < w.col_rows[uj].size(); ++t) {
      if (w.row_active[static_cast<std::size_t>(w.col_rows[uj][t])] == 0) continue;
      column_max = std::max(column_max, std::fabs(w.col_values[uj][t]));
    }
    if (w.fast) {
      w.col_max[uj] = column_max;
      w.col_max_valid[uj] = 1;
    }
    return column_max;
  };

  for (Index step = 0; step < m; ++step) {
    // ASKED BEFORE EVERY PIVOT, as the LDL^T asks before every elimination step (#197): on
    // Mittelmann's bdry2 one factorization of a 376,500-row basis ran for minutes, and the
    // simplex could only look at the clock once it returned - 648 s against a 300 s limit
    // (#208). A coarser cadence was measured wrong for the LDL^T (48 s past a 10 s limit
    // in one batch of 64 steps) and is not tried here. The check decides only whether an
    // unfinished factorization keeps running; it never touches the arithmetic.
    if (should_stop && should_stop()) {
      stopped_early_ = true;
      return false;
    }
    // ---- choose a pivot ------------------------------------------------------------------
    Index best_row = -1;
    Index best_col = -1;
    double best_value = 0.0;
    Index best_cost = std::numeric_limits<Index>::max();
    Index examined = 0;

    // A column of count 1 has Markowitz cost 0 whatever its row, and so does a row of
    // count 1. Sweeping the buckets in increasing count finds those first, which is the
    // singleton/triangular pre-pass falling out of the general rule rather than being
    // special-cased alongside it.
    for (Index count = 1; count <= m && examined < kCandidateBudget && best_cost > 0; ++count) {
      // -- candidate columns of this count
      for (Index j = col_lists.first(count); j >= 0 && examined < kCandidateBudget;
           j = col_lists.after(j)) {
        const auto uj = static_cast<std::size_t>(j);
        assert(w.col_active[uj] != 0 && w.col_count[uj] == count && "a stale list entry");
        ++examined;

        const double column_max = active_column_max(j);
        if (column_max < pivot_tolerance) continue;

        for (std::size_t t = 0; t < w.col_rows[uj].size(); ++t) {
          const Index i = w.col_rows[uj][t];
          const auto ui = static_cast<std::size_t>(i);
          if (w.row_active[ui] == 0) continue;
          const double value = w.col_values[uj][t];
          if (std::fabs(value) < threshold * column_max) continue;
          if (std::fabs(value) < pivot_tolerance) continue;
          const Index cost = (w.row_count[ui] - 1) * (count - 1);
          if (cost < best_cost) {
            best_cost = cost;
            best_row = i;
            best_col = j;
            best_value = value;
            if (cost == 0) break;
          }
        }
      }

      // -- candidate rows of this count, which the column sweep can miss entirely when a
      //    row singleton sits in a dense column
      for (Index i = row_lists.first(count); i >= 0 && examined < kCandidateBudget;
           i = row_lists.after(i)) {
        const auto ui = static_cast<std::size_t>(i);
        assert(w.row_active[ui] != 0 && w.row_count[ui] == count && "a stale list entry");
        ++examined;

        for (std::size_t p = 0; p < w.row_cols[ui].size(); ++p) {
          const Index j = w.row_cols[ui][p];
          const auto uj = static_cast<std::size_t>(j);
          if (w.col_active[uj] == 0) continue;
          const Index cost = (count - 1) * (w.col_count[uj] - 1);
          if (cost >= best_cost) continue;

          double column_max = 0.0;
          double value = 0.0;
          bool found = false;
          if (w.fast && w.row_values_valid[ui] != 0 && w.col_max_valid[uj] != 0) {
            // A valid row's pattern is exact and its values current (lu_workspace.hpp), and
            // the cached maximum is the one the scan below would compute: the same test on
            // the same numbers, without reading a column that may hold 100,000 entries.
            column_max = w.col_max[uj];
            value = w.row_values[ui][p];
            found = true;
          } else {
            for (std::size_t t = 0; t < w.col_rows[uj].size(); ++t) {
              const Index r = w.col_rows[uj][t];
              if (w.row_active[static_cast<std::size_t>(r)] == 0) continue;
              const double v = w.col_values[uj][t];
              column_max = std::max(column_max, std::fabs(v));
              if (r == i) {
                value = v;
                found = true;
              }
            }
            if (w.fast) {
              w.col_max[uj] = column_max;
              w.col_max_valid[uj] = 1;
            }
          }
          if (!found) continue;  // the row pattern was stale
          if (std::fabs(value) < pivot_tolerance) continue;
          if (std::fabs(value) < threshold * column_max) continue;

          best_cost = cost;
          best_row = i;
          best_col = j;
          best_value = value;
          if (cost == 0) break;
        }
      }
    }

    // BUDGETED SEARCH FAILURE IS NOT PROOF OF SINGULARITY (issue #143). kCandidateBudget
    // bounds the search to a handful of low-count buckets for speed, which is the right
    // trade on a healthy basis - but on THIS step it just means none of the columns and rows
    // LOOKED AT had an admissible entry, not that none exists anywhere in the active
    // submatrix. Bailing out here at the first such step is exactly the bug: the entire
    // REMAINING m - step columns get reported as "the singular set" when almost all of them
    // were never actually examined. Measured on grow15/pilot4/25fv47/perold/d6cube, that
    // inflated the reported defect from a handful of genuinely dependent columns to 33-94%
    // of the whole basis.
    //
    // So before concluding anything, fall back to an EXHAUSTIVE scan of every active column
    // still standing, unbounded by the budget. If that finds an admissible pivot, the
    // budgeted search merely got unlucky with bucket order and this step proceeds normally.
    // Only when the exhaustive scan ALSO finds nothing is the active submatrix genuinely
    // singular to working precision - every remaining entry is below pivot_tolerance or
    // below threshold * its column's max, which is the asserted condition below, not an
    // assumed one.
    if (best_row < 0) {
      for (Index j = 0; j < m; ++j) {
        const auto uj = static_cast<std::size_t>(j);
        if (w.col_active[uj] == 0) continue;

        double column_max = 0.0;
        for (std::size_t t = 0; t < w.col_rows[uj].size(); ++t) {
          if (w.row_active[static_cast<std::size_t>(w.col_rows[uj][t])] == 0) continue;
          column_max = std::max(column_max, std::fabs(w.col_values[uj][t]));
        }
        if (column_max < pivot_tolerance) continue;

        for (std::size_t t = 0; t < w.col_rows[uj].size(); ++t) {
          const Index i = w.col_rows[uj][t];
          const auto ui = static_cast<std::size_t>(i);
          if (w.row_active[ui] == 0) continue;
          const double value = w.col_values[uj][t];
          if (std::fabs(value) < pivot_tolerance) continue;
          if (std::fabs(value) < threshold * column_max) continue;
          const Index cost = (w.row_count[ui] - 1) * (w.col_count[uj] - 1);
          if (cost < best_cost) {
            best_cost = cost;
            best_row = i;
            best_col = j;
            best_value = value;
            if (cost == 0) break;
          }
        }
        if (best_cost == 0) break;
      }
    }

    if (best_row < 0) {
      // GENUINELY SINGULAR, not a search artefact: assert the condition this conclusion
      // rests on rather than assume it, exactly because "continuing must not turn singular
      // into silently wrong" (issue #143). Every active entry must fail admissibility for
      // SOME reason - too small outright, or too small relative to its own column's max - and
      // this recomputes both per column to check it directly, rather than trusting that the
      // exhaustive scan above could not itself have a bug that skipped a live entry.
#ifndef NDEBUG
      for (Index j = 0; j < m; ++j) {
        const auto uj = static_cast<std::size_t>(j);
        if (w.col_active[uj] == 0) continue;
        double column_max = 0.0;
        for (std::size_t t = 0; t < w.col_rows[uj].size(); ++t) {
          if (w.row_active[static_cast<std::size_t>(w.col_rows[uj][t])] == 0) continue;
          column_max = std::max(column_max, std::fabs(w.col_values[uj][t]));
        }
        for (std::size_t t = 0; t < w.col_rows[uj].size(); ++t) {
          if (w.row_active[static_cast<std::size_t>(w.col_rows[uj][t])] == 0) continue;
          const double magnitude = std::fabs(w.col_values[uj][t]);
          assert((magnitude < pivot_tolerance || magnitude < threshold * column_max) &&
                 "eliminate() declared the active submatrix singular but an admissible pivot "
                 "was still present - the exhaustive fallback scan has a bug");
        }
      }
#endif

      // The residual: every row and column still active is exactly the rank defect the
      // repair (elsewhere) needs to know about, reported by POSITION in the original
      // `columns` array factorize() was given, and by original row index - not by
      // elimination step, because these columns never reached one.
      dependent_positions_.clear();
      uncovered_rows_.clear();
      for (Index j = 0; j < m; ++j) {
        if (w.col_active[static_cast<std::size_t>(j)] != 0) dependent_positions_.push_back(j);
      }
      for (Index i = 0; i < m; ++i) {
        if (w.row_active[static_cast<std::size_t>(i)] != 0) uncovered_rows_.push_back(i);
      }
      return false;
    }

    const auto pivot_r = static_cast<std::size_t>(best_row);
    const auto pivot_c = static_cast<std::size_t>(best_col);
    const double pivot = best_value;
    smallest_pivot_ = std::min(smallest_pivot_, std::fabs(pivot));
    largest_pivot_ = std::max(largest_pivot_, std::fabs(pivot));

    pivot_row_.push_back(best_row);
    pivot_col_.push_back(best_col);
    pivot_value_.push_back(pivot);

    // ---- multipliers ---------------------------------------------------------------------
    w.mult_rows.clear();
    w.mult_values.clear();
    for (std::size_t t = 0; t < w.col_rows[pivot_c].size(); ++t) {
      const Index i = w.col_rows[pivot_c][t];
      const auto ui = static_cast<std::size_t>(i);
      if (w.row_active[ui] == 0 || i == best_row) continue;
      const double value = w.col_values[pivot_c][t];
      if (value == 0.0) continue;
      w.mult_rows.push_back(i);
      w.mult_values.push_back(value / pivot);
    }

    // The pivot row and column leave the active submatrix now, so that neither the update
    // below nor any later step has to keep excluding them by hand.
    w.row_active[pivot_r] = 0;
    w.col_active[pivot_c] = 0;
    row_lists.remove(best_row);
    col_lists.remove(best_col);

    // Every row that had an entry in the pivot column loses it.
    for (const Index i : w.mult_rows) --w.row_count[static_cast<std::size_t>(i)];

    w.u_cols.clear();
    w.u_vals.clear();

    // ---- a column singleton: take the U row from the row copy (#463) ---------------------
    // Suhl & Suhl (1990); Koberstein (2005), sec. 5.3. No multipliers means no other row
    // changes: each column the pivot row crosses loses one entry and nothing else. The
    // general update below would scatter every one of those columns to find that entry and
    // gather it back unchanged - the whole of a dense column for every pivot row that
    // crosses it. Here the entry comes from the row copy, which is exact for a row no step
    // has updated (lu_workspace.hpp), and the retired row's entry is left in the column's
    // storage as a dead entry that every reader already skips (they all test row_active).
    //
    // Why this is the SAME factorization and not merely an equivalent one: the columns are
    // visited in row_cols order, which for a valid row is the deduplicated order the general
    // path uses; each gets the same count decrement and the same rebucketing, in the same
    // order, and the same U entry. A dead entry changes no scan's result and the relative
    // order of a column's live entries is never disturbed, so every later tie is broken the
    // same way. The tests hold it to that bit for bit (test_sparse_lu_singletons.cpp).
    if (w.fast && w.mult_rows.empty() && w.row_values_valid[pivot_r] != 0) {
      const std::vector<Index>& row_columns = w.row_cols[pivot_r];
      const std::vector<double>& row_values = w.row_values[pivot_r];
      for (std::size_t p = 0; p < row_columns.size(); ++p) {
        const Index j = row_columns[p];
        const auto uj = static_cast<std::size_t>(j);
        if (w.col_active[uj] == 0) continue;
        const double value = row_values[p];
        --w.col_count[uj];
        w.u_cols.push_back(j);
        w.u_vals.push_back(value);
        // The maximum over what remains is unchanged when the entry leaving was strictly
        // below it; an entry at the maximum may have been the only one there.
        if (w.col_max_valid[uj] != 0 && !(std::fabs(value) < w.col_max[uj])) {
          w.col_max_valid[uj] = 0;
        }
        // Dead entries are dropped once they are half the storage, keeping the live ones in
        // order: amortised O(1) per retired entry, and no scan ever reads more than twice
        // the live entries plus kCompactionSlack.
        std::vector<Index>& rows = w.col_rows[uj];
        std::vector<double>& values = w.col_values[uj];
        if (static_cast<Index>(rows.size()) > 2 * w.col_count[uj] + kCompactionSlack) {
          std::size_t kept = 0;
          for (std::size_t t = 0; t < rows.size(); ++t) {
            if (w.row_active[static_cast<std::size_t>(rows[t])] == 0) continue;
            rows[kept] = rows[t];
            values[kept] = values[t];
            ++kept;
          }
          rows.resize(kept);
          values.resize(kept);
        }
        rebucket_column(j);
      }
      commit_step();
      continue;
    }

    // Deduplicate the pivot row's active columns before touching any counts. See the note
    // on pivot_row_columns: a stale pattern can name the same column twice, and visiting it
    // twice would double-decrement col_count and emit two U entries for one coefficient.
    w.pivot_row_columns.clear();
    for (const Index j : w.row_cols[pivot_r]) {
      const auto uj = static_cast<std::size_t>(j);
      if (w.col_active[uj] == 0 || w.column_seen[uj] != 0) continue;
      w.column_seen[uj] = 1;
      w.pivot_row_columns.push_back(j);
    }
    for (const Index j : w.pivot_row_columns) w.column_seen[static_cast<std::size_t>(j)] = 0;
    // NOTE: col_count is deliberately NOT decremented here. The pivot row's pattern is
    // allowed to go stale, so it can name a column whose entry was cancelled away at an
    // earlier step. Decrementing on the strength of the pattern alone undercounts that
    // column, and the error accumulates until the count goes NEGATIVE - at which point
    // col_bucket[static_cast<std::size_t>(-1)] writes off the end of the bucket array. The
    // decrement therefore happens in the update loop below, once the column's own storage
    // has confirmed the entry is really there.

    // ---- update the remaining columns ----------------------------------------------------
    // Every multiplier row's values change below, so its row copy stops being exact.
    if (w.fast && !w.pivot_row_columns.empty()) {
      for (const Index i : w.mult_rows) w.row_values_valid[static_cast<std::size_t>(i)] = 0;
    }

    for (const Index j : w.pivot_row_columns) {
      const auto uj = static_cast<std::size_t>(j);

      std::vector<Index>& rows = w.col_rows[uj];
      std::vector<double>& values = w.col_values[uj];

      // Scatter the column, keeping only rows that are still active. The pivot row's own
      // entry is what becomes the U coefficient.
      double pivot_row_value = 0.0;
      bool pivot_row_present = false;
      w.old_rows.clear();
      for (std::size_t t = 0; t < rows.size(); ++t) {
        const Index i = rows[t];
        const auto ui = static_cast<std::size_t>(i);
        if (i == best_row) {
          pivot_row_value = values[t];
          pivot_row_present = true;
          continue;
        }
        if (w.row_active[ui] == 0) continue;
        w.acc[ui] = values[t];
        w.acc_present[ui] = 1;
        w.old_rows.push_back(i);
      }

      if (!pivot_row_present || pivot_row_value == 0.0) {
        // A stale row-pattern entry: nothing to eliminate against in this column. Undo the
        // scatter and move on.
        for (const Index i : w.old_rows) {
          const auto ui = static_cast<std::size_t>(i);
          w.acc[ui] = 0.0;
          w.acc_present[ui] = 0;
        }
        continue;
      }

      // The entry is confirmed present, so now the column really does lose it.
      --w.col_count[uj];

      w.u_cols.push_back(j);
      w.u_vals.push_back(pivot_row_value);

      for (std::size_t t = 0; t < w.mult_rows.size(); ++t) {
        const auto ui = static_cast<std::size_t>(w.mult_rows[t]);
        const double contribution = w.mult_values[t] * pivot_row_value;
        if (w.acc_present[ui] == 0) {
          // Fill-in. The row gains a column it did not have; the row pattern has to learn
          // about it or a later pivot search will never consider this entry.
          w.acc[ui] = -contribution;
          w.acc_present[ui] = 1;
          w.row_cols[ui].push_back(j);
          ++w.row_count[ui];
          ++w.col_count[uj];
        } else {
          w.acc[ui] -= contribution;
        }
      }

      // Gather back, dropping exact cancellations. The column's new maximum is taken on the
      // way, since every live entry passes through here.
      rows.clear();
      values.clear();
      double column_max = 0.0;
      for (std::size_t t = 0; t < w.mult_rows.size(); ++t) {
        const auto ui = static_cast<std::size_t>(w.mult_rows[t]);
        if (w.acc_present[ui] == 0) continue;
        const double value = w.acc[ui];
        w.acc[ui] = 0.0;
        w.acc_present[ui] = 0;
        if (std::fabs(value) < kDropTolerance) {
          --w.row_count[ui];
          --w.col_count[uj];
          continue;
        }
        rows.push_back(w.mult_rows[t]);
        values.push_back(value);
        column_max = std::max(column_max, std::fabs(value));
      }
      // Whatever remains in the accumulator belongs to rows the pivot column did not touch:
      // rows of the column's old pattern, which the scatter recorded.
      for (const Index i : w.old_rows) {
        const auto ui = static_cast<std::size_t>(i);
        if (w.acc_present[ui] == 0) continue;
        rows.push_back(i);
        values.push_back(w.acc[ui]);
        column_max = std::max(column_max, std::fabs(w.acc[ui]));
        w.acc[ui] = 0.0;
        w.acc_present[ui] = 0;
      }
      if (w.fast) {
        w.col_max[uj] = column_max;
        w.col_max_valid[uj] = 1;
      }

      rebucket_column(j);
    }

    for (const Index i : w.mult_rows) rebucket_row(i);

    commit_step();
  }

  if (smallest_pivot_ == std::numeric_limits<double>::max()) smallest_pivot_ = 0.0;
  return true;
}

}  // namespace sankhya
