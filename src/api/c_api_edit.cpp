// SPDX-License-Identifier: Apache-2.0
// SANKHYA - removing rows and columns through the C API (#913), and the thaw that lets a
// model read from a file take structural edits at all.
//
// A handle built call by call keeps its matrix as pending (row, col) triplets and
// materialises it at solve time; a handle filled by sankhya_model_read() holds the reader's
// frozen matrix instead. Before #913 the two did not mix: one sankhya_model_set_coefficient()
// on a read model replaced the whole file matrix with that single entry, and an added row
// or column left the frozen matrix one short. thaw() moves a frozen matrix (and Hessian)
// into the triplets the first time the handle is edited structurally, so every edit below
// works on one representation.
//
// The warm start across such an edit is sankhya_solve_from()'s: it places the earlier
// basis on the edited model by name (map_basis_by_name, core/basis_map.cpp), and solve()
// completes it (simplex/warm_basis.hpp, the repair of Maros, "Computational Techniques of
// the Simplex Method", sec. 9.4).

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "api/c_api_internal.hpp"
#include "sankhya/sankhya.h"

namespace sankhya::capi {
namespace {

void thaw_matrix(SparseMatrix* matrix, std::map<std::pair<int, int>, double>* entries) {
  if (!entries->empty()) return;
  // An empty one too: a reader's empty Hessian is num_cols wide, which an added column
  // would leave one short. A handle built call by call has the default, 0 x 0.
  if (!matrix->frozen()) matrix->finalize();
  for (Index j = 0; j < matrix->num_cols(); ++j) {
    const ColumnView column = matrix->column(j);
    for (Index k = 0; k < column.size; ++k) {
      (*entries)[{static_cast<int>(column.rows[k]), static_cast<int>(j)}] = column.values[k];
    }
  }
  matrix->reset(0, 0);
}

/// `indices` as a sorted set of distinct positions below `size`; false when one is not.
bool to_set(int count, const int* indices, int size, std::vector<int>* out) {
  out->assign(indices, indices + count);
  std::sort(out->begin(), out->end());
  out->erase(std::unique(out->begin(), out->end()), out->end());
  return out->empty() || (out->front() >= 0 && out->back() < size);
}

/// The new position of each old one, -1 for a removed one.
std::vector<int> renumbering(int size, const std::vector<int>& removed) {
  std::vector<int> map(static_cast<std::size_t>(size));
  int next = 0;
  auto r = removed.begin();
  for (int i = 0; i < size; ++i) {
    if (r != removed.end() && *r == i) {
      map[static_cast<std::size_t>(i)] = -1;
      ++r;
    } else {
      map[static_cast<std::size_t>(i)] = next++;
    }
  }
  return map;
}

template <typename T>
void erase_positions(std::vector<T>* values, const std::vector<int>& map) {
  std::size_t out = 0;
  for (std::size_t i = 0; i < values->size(); ++i) {
    if (map[i] >= 0) (*values)[out++] = std::move((*values)[i]);
  }
  values->resize(out);
}

/// Re-key `entries` through `first_map` on the first index and `second_map` on the second
/// (either may be null for unchanged), dropping an entry either map removes.
void rekey(std::map<std::pair<int, int>, double>* entries, const std::vector<int>* first_map,
           const std::vector<int>* second_map) {
  std::map<std::pair<int, int>, double> kept;
  for (const auto& [key, value] : *entries) {
    const int a = first_map ? (*first_map)[static_cast<std::size_t>(key.first)] : key.first;
    const int b = second_map ? (*second_map)[static_cast<std::size_t>(key.second)] : key.second;
    if (a >= 0 && b >= 0) kept.emplace_hint(kept.end(), std::make_pair(a, b), value);
  }
  *entries = std::move(kept);
}

sankhya_status refuse_unless_deletable(const sankhya_model* model) {
  if (model->nonlinear != nullptr && !model->nonlinear->empty()) {
    return fail(SANKHYA_ERROR_ARGUMENT,
                "rows and columns cannot be removed from a model with nonlinear parts");
  }
  if (model->model.has_semicontinuous_or_sos()) {
    return fail(SANKHYA_ERROR_ARGUMENT,
                "rows and columns cannot be removed from a model with semi-continuous "
                "columns or special ordered sets");
  }
  return SANKHYA_OK;
}

}  // namespace

void thaw(sankhya_model* handle) {
  thaw_matrix(&handle->model.matrix, &handle->entries);
  thaw_matrix(&handle->model.hessian, &handle->quadratic);
}

}  // namespace sankhya::capi

using sankhya::capi::fail;
using sankhya::capi::guarded;
using sankhya::capi::ok;

sankhya_status sankhya_model_delete_rows(sankhya_model* model, int count, const int* indices) {
  if (model == nullptr || count < 0 || (count > 0 && indices == nullptr)) {
    return fail(SANKHYA_ERROR_ARGUMENT, "model or indices is null, or count is negative");
  }
  if (const sankhya_status refused = sankhya::capi::refuse_unless_deletable(model);
      refused != SANKHYA_OK) {
    return refused;
  }
  return guarded([&]() -> sankhya_status {
    sankhya::Model& m = model->model;
    const int rows = static_cast<int>(m.row_lower.size());
    std::vector<int> removed;
    if (!sankhya::capi::to_set(count, indices, rows, &removed)) {
      return fail(SANKHYA_ERROR_ARGUMENT, "a row index is outside the model, which has " +
                                              std::to_string(rows) + " row(s)");
    }
    if (removed.empty()) return ok();
    sankhya::capi::thaw(model);
    const std::vector<int> map = sankhya::capi::renumbering(rows, removed);
    sankhya::capi::erase_positions(&m.row_lower, map);
    sankhya::capi::erase_positions(&m.row_upper, map);
    if (m.row_names.size() == map.size()) sankhya::capi::erase_positions(&m.row_names, map);
    sankhya::capi::rekey(&model->entries, &map, nullptr);
    return ok();
  });
}

sankhya_status sankhya_model_delete_cols(sankhya_model* model, int count, const int* indices) {
  if (model == nullptr || count < 0 || (count > 0 && indices == nullptr)) {
    return fail(SANKHYA_ERROR_ARGUMENT, "model or indices is null, or count is negative");
  }
  if (const sankhya_status refused = sankhya::capi::refuse_unless_deletable(model);
      refused != SANKHYA_OK) {
    return refused;
  }
  return guarded([&]() -> sankhya_status {
    sankhya::Model& m = model->model;
    const int cols = static_cast<int>(m.col_cost.size());
    std::vector<int> removed;
    if (!sankhya::capi::to_set(count, indices, cols, &removed)) {
      return fail(SANKHYA_ERROR_ARGUMENT, "a column index is outside the model, which has " +
                                              std::to_string(cols) + " column(s)");
    }
    if (removed.empty()) return ok();
    sankhya::capi::thaw(model);
    const std::vector<int> map = sankhya::capi::renumbering(cols, removed);
    sankhya::capi::erase_positions(&m.col_cost, map);
    sankhya::capi::erase_positions(&m.col_lower, map);
    sankhya::capi::erase_positions(&m.col_upper, map);
    sankhya::capi::erase_positions(&m.col_type, map);
    if (m.col_names.size() == map.size()) sankhya::capi::erase_positions(&m.col_names, map);
    sankhya::capi::rekey(&model->entries, nullptr, &map);
    sankhya::capi::rekey(&model->quadratic, &map, &map);
    return ok();
  });
}
