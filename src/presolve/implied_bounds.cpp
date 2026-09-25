// SPDX-License-Identifier: Apache-2.0
// SANKHYA - postsolve of propagated bounds (#485). See implied_bounds.hpp for the argument
// and the references.

#include "presolve/implied_bounds.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "sankhya/sparse.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::presolve {
namespace {

[[nodiscard]] bool finite(double v) {
  return std::fabs(v) < kInfinity;
}

/// On `bound` to the relative precision postsolve's at_bound() uses (presolve.cpp, #198).
[[nodiscard]] bool on(double value, double bound) {
  return finite(bound) &&
         std::fabs(value - bound) <= tol::kPrimalFeasibility * std::max(1.0, std::fabs(bound));
}

}  // namespace

void restore_implied_bound_basis(const Result& result, const Model& original,
                                 const std::vector<std::size_t>& column_removed_at,
                                 const std::vector<double>& implied_transfer,
                                 Solution* solution) {
  const auto n = static_cast<std::size_t>(original.num_cols());
  const auto m = static_cast<std::size_t>(original.num_rows());
  if (solution->col_value.size() != n) return;
  const std::size_t records = result.records.size();

  // The transfer chose the row's price to cancel the column's reduced cost; recomputing it
  // from c - A^T y leaves a rounding residue that the complementarity product multiplies by
  // the column's distance to its original bound. Exactly zero, as process_singleton_row
  // sets it, for a column inside its original box.
  if (solution->col_dual.size() == n) {
    for (std::size_t idx = 0; idx < records; ++idx) {
      const Record& record = result.records[idx];
      if (record.kind != Record::Kind::kImpliedBound || implied_transfer[idx] == 0.0) continue;
      const auto c = static_cast<std::size_t>(record.column);
      const double x = solution->col_value[c];
      if (!on(x, original.col_lower[c]) && !on(x, original.col_upper[c])) {
        solution->col_dual[c] = 0.0;
      }
    }
  }

  std::vector<BasisStatus>& col_status = solution->col_status;
  std::vector<BasisStatus>& row_status = solution->row_status;
  if (col_status.size() != n || row_status.size() != m) return;
  const auto unknown = [](BasisStatus s) { return s == BasisStatus::kUnknown; };
  if (std::any_of(col_status.begin(), col_status.end(), unknown) ||
      std::any_of(row_status.begin(), row_status.end(), unknown)) {
    return;
  }

  CsrView rows;
  bool have_rows = false;
  // A column is brought into the basis once, whichever of its records reaches it first: a
  // column propagated to a point from both sides has two records, and the second must not
  // push a second entry out for it.
  std::vector<bool> entered(n, false);
  for (std::size_t idx = records; idx-- > 0;) {
    const Record& record = result.records[idx];
    if (record.kind != Record::Kind::kImpliedBound) continue;
    const auto c = static_cast<std::size_t>(record.column);
    if (entered[c]) continue;
    const double x = solution->col_value[c];
    if (!on(x, record.value)) continue;
    if (on(x, original.col_lower[c]) || on(x, original.col_upper[c])) continue;
    // Which columns still need an entry to leave the basis for them. A column in the reduced
    // model the engine left nonbasic; a column fixed at the propagated bound (kFixedColumn)
    // that nothing has made basic yet - a singleton row that priced it made it basic and
    // put its own logical out in exchange; and a column dual fixing or dominance fixed there,
    // which the replay makes basic with no row restored to pay for it. A column eliminated
    // through a row is that row's basic entry already.
    const bool removed = column_removed_at[c] < records;
    const Record::Kind removed_by =
        removed ? result.records[column_removed_at[c]].kind : Record::Kind::kImpliedBound;
    const bool replayed_basic = removed_by == Record::Kind::kDualFixedColumn ||
                                removed_by == Record::Kind::kDominatedColumn;
    if (removed && !replayed_basic && removed_by != Record::Kind::kFixedColumn) continue;
    if (!replayed_basic && col_status[c] == BasisStatus::kBasic) continue;

    if (!have_rows) {
      rows.build(original.matrix);
      have_rows = true;
    }
    const Index row = record.index;
    const auto r = static_cast<std::size_t>(row);
    const ColumnView view = rows.row(row);  // `rows` holds COLUMN indices for a row view
    bool swapped = false;
    if (row_status[r] == BasisStatus::kBasic) {
      double activity = 0.0;
      for (Index k = 0; k < view.size; ++k) {
        activity +=
            view.values[k] * solution->col_value[static_cast<std::size_t>(view.rows[k])];
      }
      if (on(activity, original.row_lower[r])) {
        row_status[r] = BasisStatus::kAtLower;
        swapped = true;
      } else if (on(activity, original.row_upper[r])) {
        row_status[r] = BasisStatus::kAtUpper;
        swapped = true;
      }
    } else {
      for (Index k = 0; k < view.size && !swapped; ++k) {
        const auto uk = static_cast<std::size_t>(view.rows[k]);
        if (uk == c || col_status[uk] != BasisStatus::kBasic) continue;
        const double xk = solution->col_value[uk];
        if (on(xk, original.col_lower[uk])) {
          col_status[uk] = BasisStatus::kAtLower;
          swapped = true;
        } else if (on(xk, original.col_upper[uk])) {
          col_status[uk] = BasisStatus::kAtUpper;
          swapped = true;
        }
      }
    }
    if (swapped) {
      col_status[c] = BasisStatus::kBasic;
      entered[c] = true;
    }
  }
}

}  // namespace sankhya::presolve
