// SPDX-License-Identifier: Apache-2.0
// SANKHYA - an earlier basis placed on a model whose rows and columns changed (#913).
//
// A planner who adds a unit or a product, or drops a period, has yesterday's basis for a
// model of a different shape. Positions mean nothing across that edit; names do. Each
// column and row of today's model found by name in yesterday's basis keeps its status. A
// new row enters with its slack basic: the row has no say in the old vertex, so it starts
// slack and the simplex makes it bind if it must. A new column enters nonbasic at a bound:
// it was not in yesterday's plan, so it starts at the value that leaves the old point where
// it was (Chvatal, "Linear Programming", 1983, ch. 7, on restarting from an advanced
// basis). What the edit
// does to the COUNT of basic entries is solve()'s to repair (core/solve.cpp, the warm
// branch, and simplex/warm_basis.hpp), since it holds the matrix the repair factorizes.

#include <fmt/format.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sankhya/model.hpp"

namespace sankhya {
namespace {

/// The name the .sol writer gives entry `k` when the model carries none (writer.cpp).
std::string name_of(const std::vector<std::string>& names, Index k, char fallback_prefix) {
  const auto u = static_cast<std::size_t>(k);
  if (u < names.size() && !names[u].empty()) return names[u];
  return fmt::format("{}{}", fallback_prefix, k);
}

/// The status a column the earlier basis does not name starts with: nonbasic at the bound
/// it has, the lower one first, free at zero when it has neither.
BasisStatus new_column_status(double lower, double upper) {
  if (is_finite_bound(lower) && lower == upper) return BasisStatus::kFixed;
  if (is_finite_bound(lower)) return BasisStatus::kAtLower;
  if (is_finite_bound(upper)) return BasisStatus::kAtUpper;
  return BasisStatus::kNonbasicFree;
}

}  // namespace

bool map_basis_by_name(const std::vector<std::string>& from_col_names,
                       const std::vector<std::string>& from_row_names,
                       const std::vector<BasisStatus>& from_col_status,
                       const std::vector<BasisStatus>& from_row_status, const Model& model,
                       std::vector<BasisStatus>* col_status,
                       std::vector<BasisStatus>* row_status, BasisMapping* mapping) {
  if (from_col_names.size() != from_col_status.size() ||
      from_row_names.size() != from_row_status.size()) {
    return false;
  }
  // Earlier entries by name; the first of a repeated name wins, as the readers keep it.
  std::unordered_map<std::string, std::size_t> earlier_cols;
  std::unordered_map<std::string, std::size_t> earlier_rows;
  earlier_cols.reserve(from_col_names.size());
  earlier_rows.reserve(from_row_names.size());
  for (std::size_t k = 0; k < from_col_names.size(); ++k)
    earlier_cols.emplace(from_col_names[k], k);
  for (std::size_t k = 0; k < from_row_names.size(); ++k)
    earlier_rows.emplace(from_row_names[k], k);

  BasisMapping counts;
  std::vector<char> col_used(from_col_names.size(), 0);
  std::vector<char> row_used(from_row_names.size(), 0);
  std::vector<BasisStatus> cols(static_cast<std::size_t>(model.num_cols()));
  std::vector<BasisStatus> rows(static_cast<std::size_t>(model.num_rows()));
  for (Index j = 0; j < model.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    const auto it = earlier_cols.find(name_of(model.col_names, j, 'C'));
    if (it != earlier_cols.end() && col_used[it->second] == 0) {
      col_used[it->second] = 1;
      cols[u] = from_col_status[it->second];
      ++counts.matched_cols;
    } else {
      cols[u] = new_column_status(model.col_lower[u], model.col_upper[u]);
      ++counts.new_cols;
    }
  }
  for (Index i = 0; i < model.num_rows(); ++i) {
    const auto u = static_cast<std::size_t>(i);
    const auto it = earlier_rows.find(name_of(model.row_names, i, 'R'));
    if (it != earlier_rows.end() && row_used[it->second] == 0) {
      row_used[it->second] = 1;
      rows[u] = from_row_status[it->second];
      ++counts.matched_rows;
    } else {
      rows[u] = BasisStatus::kBasic;  // a new row starts slack
      ++counts.new_rows;
    }
  }
  for (std::size_t k = 0; k < col_used.size(); ++k) {
    if (col_used[k] != 0) continue;
    ++counts.removed_cols;
    if (from_col_status[k] == BasisStatus::kBasic) ++counts.removed_basic;
  }
  for (std::size_t k = 0; k < row_used.size(); ++k) {
    if (row_used[k] != 0) continue;
    ++counts.removed_rows;
    if (from_row_status[k] == BasisStatus::kBasic) ++counts.removed_basic;
  }
  *col_status = std::move(cols);
  *row_status = std::move(rows);
  if (mapping != nullptr) *mapping = counts;
  return true;
}

bool map_basis_by_name(const Model& from_model, const std::vector<BasisStatus>& from_col_status,
                       const std::vector<BasisStatus>& from_row_status, const Model& model,
                       std::vector<BasisStatus>* col_status,
                       std::vector<BasisStatus>* row_status, BasisMapping* mapping) {
  // Without names on both sides there is nothing to match by; positions would silently
  // pair a column with whatever now sits where it used to.
  const auto named = [](const Model& m) {
    return static_cast<Index>(m.col_names.size()) == m.num_cols() &&
           static_cast<Index>(m.row_names.size()) == m.num_rows();
  };
  if (!named(from_model) || !named(model)) return false;
  return map_basis_by_name(from_model.col_names, from_model.row_names, from_col_status,
                           from_row_status, model, col_status, row_status, mapping);
}

}  // namespace sankhya
