// SPDX-License-Identifier: Apache-2.0
// SANKHYA - structural edits of a Model, for the warm-start tests of #913.
//
// A planner who adds a unit or a product, or drops a period, changes the model's shape:
// rows and columns come and go, and the names of the ones that stay are what carries the
// old basis across. These helpers make those edits on a Model read from a file, rebuilding
// its matrix from triplets, and keep every per-column and per-row array in step.
#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "sankhya/model.hpp"
#include "sankhya/sparse.hpp"

namespace sankhya::testing {

/// One entry of the constraint matrix.
struct Entry {
  Index row = 0;
  Index col = 0;
  double value = 0.0;
};

inline std::vector<Entry> entries_of(const Model& model) {
  std::vector<Entry> entries;
  for (Index j = 0; j < model.num_cols(); ++j) {
    const ColumnView column = model.matrix.column(j);
    for (Index k = 0; k < column.size; ++k)
      entries.push_back({column.rows[k], j, column.values[k]});
  }
  return entries;
}

inline void rebuild(Model* model, const std::vector<Entry>& entries) {
  SparseMatrix matrix(model->num_rows(), model->num_cols());
  for (const Entry& e : entries) matrix.add_entry(e.row, e.col, e.value);
  matrix.finalize();
  model->matrix = std::move(matrix);
}

/// Remove column `j`; the model must carry column names.
inline void remove_column(Model* model, Index j) {
  std::vector<Entry> entries;
  for (const Entry& e : entries_of(*model)) {
    if (e.col != j) entries.push_back({e.row, e.col > j ? e.col - 1 : e.col, e.value});
  }
  const auto at = static_cast<std::ptrdiff_t>(j);
  model->col_cost.erase(model->col_cost.begin() + at);
  model->col_lower.erase(model->col_lower.begin() + at);
  model->col_upper.erase(model->col_upper.begin() + at);
  model->col_type.erase(model->col_type.begin() + at);
  model->col_names.erase(model->col_names.begin() + at);
  rebuild(model, entries);
}

/// Remove row `i`; the model must carry row names.
inline void remove_row(Model* model, Index i) {
  std::vector<Entry> entries;
  for (const Entry& e : entries_of(*model)) {
    if (e.row != i) entries.push_back({e.row > i ? e.row - 1 : e.row, e.col, e.value});
  }
  const auto at = static_cast<std::ptrdiff_t>(i);
  model->row_lower.erase(model->row_lower.begin() + at);
  model->row_upper.erase(model->row_upper.begin() + at);
  model->row_names.erase(model->row_names.begin() + at);
  rebuild(model, entries);
}

/// Append a continuous column; `column` holds its (row, value) pairs in the entries' row
/// and value fields.
inline void add_column(Model* model, const std::string& name, double cost, double lower,
                       double upper, const std::vector<Entry>& column) {
  std::vector<Entry> entries = entries_of(*model);
  const Index j = model->num_cols();
  model->col_cost.push_back(cost);
  model->col_lower.push_back(lower);
  model->col_upper.push_back(upper);
  model->col_type.push_back(VarType::kContinuous);
  model->col_names.push_back(name);
  for (const Entry& e : column) entries.push_back({e.row, j, e.value});
  rebuild(model, entries);
}

/// Append a row; `row` holds its (column, value) pairs in the entries' col and value fields.
inline void add_row(Model* model, const std::string& name, double lower, double upper,
                    const std::vector<Entry>& row) {
  std::vector<Entry> entries = entries_of(*model);
  const Index i = model->num_rows();
  model->row_lower.push_back(lower);
  model->row_upper.push_back(upper);
  model->row_names.push_back(name);
  for (const Entry& e : row) entries.push_back({i, e.col, e.value});
  rebuild(model, entries);
}

}  // namespace sankhya::testing
