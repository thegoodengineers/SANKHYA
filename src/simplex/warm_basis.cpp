// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a starting basis with the wrong number of basic entries, made a basis (#913).
// The method and its references are in warm_basis.hpp.

#include "warm_basis.hpp"

#include <vector>

#include "la/lu.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya {
namespace {

/// A structural leaving the basis with no value to say where it was: the lower bound when
/// it has one, else the upper, else free at zero (seed_basis honours the same statuses).
BasisStatus parked_status(double lower, double upper) {
  if (is_finite_bound(lower) && lower == upper) return BasisStatus::kFixed;
  if (is_finite_bound(lower)) return BasisStatus::kAtLower;
  if (is_finite_bound(upper)) return BasisStatus::kAtUpper;
  return BasisStatus::kNonbasicFree;
}

}  // namespace

WarmBasisCompletion complete_warm_basis(const Model& model, WarmStart* warm) {
  WarmBasisCompletion result;
  const Index n = model.num_cols();
  const Index m = model.num_rows();
  if (static_cast<Index>(warm->col_status.size()) != n ||
      static_cast<Index>(warm->row_status.size()) != m) {
    return result;
  }

  std::vector<Index> structurals;
  std::vector<Index> logicals;
  for (Index j = 0; j < n; ++j) {
    if (warm->col_status[static_cast<std::size_t>(j)] == BasisStatus::kBasic)
      structurals.push_back(j);
  }
  for (Index i = 0; i < m; ++i) {
    if (warm->row_status[static_cast<std::size_t>(i)] == BasisStatus::kBasic)
      logicals.push_back(n + i);
  }
  result.given_basic = static_cast<Index>(structurals.size() + logicals.size());
  if (result.given_basic == m) {
    result.complete = true;
    return result;
  }

  const auto park = [&](Index j) {
    const auto u = static_cast<std::size_t>(j);
    warm->col_status[u] = parked_status(model.col_lower[u], model.col_upper[u]);
    ++result.parked;
  };

  // A long basis: every basic logical stays (at most m of them, one per row), and the
  // structurals beyond what is left of m are parked, the last ones by index first.
  const auto room = static_cast<std::size_t>(m) - logicals.size();
  while (structurals.size() > room) {
    park(structurals.back());
    structurals.pop_back();
  }

  // The entries, logicals first, then structurals, then one empty placeholder (-1) per
  // entry a short basis lacks.
  std::vector<Index> entries = logicals;
  entries.insert(entries.end(), structurals.begin(), structurals.end());
  entries.resize(static_cast<std::size_t>(m), -1);

  std::vector<Index> logical_rows(static_cast<std::size_t>(m));
  for (Index i = 0; i < m; ++i) logical_rows[static_cast<std::size_t>(i)] = i;
  const std::vector<double> logical_values(static_cast<std::size_t>(m), -1.0);
  std::vector<LuColumn> columns(static_cast<std::size_t>(m));
  SparseLu lu;
  // Each round replaces every column no pivot reached, so one or two rounds is usual; the
  // bound is a backstop, as in crossover_guess.
  for (int round = 0; round < 64; ++round) {
    for (Index slot = 0; slot < m; ++slot) {
      const Index k = entries[static_cast<std::size_t>(slot)];
      LuColumn& target = columns[static_cast<std::size_t>(slot)];
      if (k < 0) {
        target = LuColumn{};
      } else if (k < n) {
        const ColumnView column = model.matrix.column(k);
        target.rows = column.rows;
        target.values = column.values;
        target.size = column.size;
      } else {
        const auto row = static_cast<std::size_t>(k - n);
        target.rows = logical_rows.data() + row;
        target.values = logical_values.data() + row;
        target.size = 1;
      }
    }
    if (lu.factorize(columns, m, tol::kPivotTolerance, 1.0)) {
      result.complete = true;
      break;
    }
    const std::vector<Index>& dependent = lu.dependent_positions();
    const std::vector<Index>& uncovered = lu.uncovered_rows();
    if (dependent.empty() || dependent.size() != uncovered.size()) break;
    bool replaced = false;
    for (std::size_t t = 0; t < dependent.size(); ++t) {
      const Index slot = dependent[t];
      const Index row = uncovered[t];
      if (slot < 0 || slot >= m || row < 0 || row >= m) continue;
      // A logical already basic covers its own row, so its row cannot be uncovered; the
      // check is cheap and the invariant is not this function's to assume.
      if (warm->row_status[static_cast<std::size_t>(row)] == BasisStatus::kBasic) continue;
      const Index evicted = entries[static_cast<std::size_t>(slot)];
      if (evicted >= n) continue;  // a logical is a unit column: never a rank defect
      if (evicted >= 0) park(evicted);
      entries[static_cast<std::size_t>(slot)] = n + row;
      warm->row_status[static_cast<std::size_t>(row)] = BasisStatus::kBasic;
      ++result.added_logicals;
      replaced = true;
    }
    if (!replaced) break;
  }
  return result;
}

}  // namespace sankhya
