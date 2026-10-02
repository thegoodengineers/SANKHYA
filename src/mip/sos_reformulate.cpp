// SPDX-License-Identifier: Apache-2.0
// SANKHYA - semi-continuous columns and special ordered sets as binaries and big-M rows
// (#754). See sos_reformulate.hpp for the formulations and the references.

#include "sos_reformulate.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fmt/format.h>

namespace sankhya::mip {
namespace {

/// Builds the reformulated model: new binaries and rows are queued, then the matrix is
/// assembled once with the original's entries first.
class Writer {
 public:
  explicit Writer(const Model& model) : model_(model), out_(model) {
    for (const std::string& name : model.col_names) taken_cols_.insert(name);
    for (const std::string& name : model.row_names) taken_rows_.insert(name);
  }

  Model& model() { return out_; }

  Index add_binary(const std::string& name) {
    const auto j = out_.num_cols();
    out_.resize_columns(j + 1);
    const auto u = static_cast<std::size_t>(j);
    out_.col_lower[u] = 0.0;
    out_.col_upper[u] = 1.0;
    out_.col_type[u] = VarType::kInteger;
    if (!out_.col_names.empty()) out_.col_names[u] = unique(name, &taken_cols_);
    return j;
  }

  /// lower <= sum entries <= upper, a new row.
  void add_row(const std::string& name, std::vector<std::pair<Index, double>> entries,
               double lower, double upper) {
    const auto i = out_.num_rows();
    out_.resize_rows(i + 1);
    const auto u = static_cast<std::size_t>(i);
    out_.row_lower[u] = lower;
    out_.row_upper[u] = upper;
    if (!out_.row_names.empty()) out_.row_names[u] = unique(name, &taken_rows_);
    for (const auto& [j, value] : entries) {
      if (value != 0.0) rows_.push_back({i, j, value});
    }
  }

  void finish() {
    const Index n = out_.num_cols();
    const Index m = out_.num_rows();
    SparseMatrix matrix(m, n);
    for (Index j = 0; j < model_.num_cols(); ++j) {
      const ColumnView column = model_.matrix.column(j);
      for (Index k = 0; k < column.size; ++k)
        matrix.add_entry(column.rows[k], j, column.values[k]);
    }
    for (const Entry& e : rows_) matrix.add_entry(e.row, e.col, e.value);
    matrix.finalize(0.0);  // data, not residue: nothing here is dropped (#590)
    out_.matrix = std::move(matrix);
    SparseMatrix hessian(n, n);
    for (Index j = 0; j < model_.hessian.num_cols(); ++j) {
      const ColumnView column = model_.hessian.column(j);
      for (Index k = 0; k < column.size; ++k)
        hessian.add_entry(column.rows[k], j, column.values[k]);
    }
    hessian.finalize(0.0);
    out_.hessian = std::move(hessian);
  }

 private:
  struct Entry {
    Index row;
    Index col;
    double value;
  };

  static std::string unique(const std::string& base, std::unordered_set<std::string>* taken) {
    std::string name = base;
    for (int k = 1; taken->count(name) != 0; ++k) name = fmt::format("{}_{}", base, k);
    taken->insert(name);
    return name;
  }

  const Model& model_;
  Model out_;
  std::vector<Entry> rows_;
  std::unordered_set<std::string> taken_cols_;
  std::unordered_set<std::string> taken_rows_;
};

std::string column_label(const Model& model, Index j) {
  const auto u = static_cast<std::size_t>(j);
  return u < model.col_names.size() && !model.col_names[u].empty() ? model.col_names[u]
                                                                   : fmt::format("C{}", j);
}

/// Cut a per-column or per-row vector of the reformulated model down to the original's
/// length; a vector of any other length is not one of those and is left alone.
template <class T>
void cut_to(std::vector<T>* values, Index from, Index to) {
  if (values->size() == static_cast<std::size_t>(from))
    values->resize(static_cast<std::size_t>(to));
}

}  // namespace

ScSosReformulation reformulate_sc_sos(const Model& model) {
  ScSosReformulation r;
  r.original_cols = model.num_cols();
  r.original_rows = model.num_rows();
  Writer writer(model);
  Model& out = writer.model();
  out.semicontinuous.clear();
  out.sos.clear();

  for (const Index j : model.semicontinuous) {
    const auto u = static_cast<std::size_t>(j);
    const double l = out.col_lower[u];
    const double h = out.col_upper[u];
    if (l <= 0.0) {
      ++r.dropped;  // {0} or [0, u] is [0, u]
      continue;
    }
    if (!std::isfinite(h)) {
      out.semicontinuous.push_back(j);
      ++r.kept;
      continue;
    }
    const std::string name = column_label(model, j);
    const Index z = writer.add_binary("sc_on_" + name);
    writer.add_row("sc_up_" + name, {{j, 1.0}, {z, -h}}, -kInfinity, 0.0);
    writer.add_row("sc_lo_" + name, {{j, 1.0}, {z, -l}}, 0.0, kInfinity);
    out.col_lower[u] = 0.0;
    ++r.semicontinuous_written;
    ++r.binaries;
    r.rows += 2;
  }

  for (std::size_t s = 0; s < model.sos.size(); ++s) {
    const SosSet& set = model.sos[s];
    const std::size_t n = set.columns.size();
    if ((set.type == 1 && n <= 1) || (set.type == 2 && n <= 2)) {
      ++r.dropped;
      continue;
    }
    bool finite = true;
    for (const Index j : set.columns) {
      const auto u = static_cast<std::size_t>(j);
      // A member that may be nonzero on a side needs a finite bound on that side.
      if ((out.col_upper[u] > 0.0 && !std::isfinite(out.col_upper[u])) ||
          (out.col_lower[u] < 0.0 && !std::isfinite(out.col_lower[u]))) {
        finite = false;
      }
    }
    if (!finite) {
      out.sos.push_back(set);
      ++r.kept;
      continue;
    }
    const std::string base = set.name.empty() ? fmt::format("sos{}", s) : "sos_" + set.name;
    // SOS1: one binary per member. SOS2: one per adjacent pair, member k covered by the
    // pairs (k-1, k) and (k, k+1).
    const std::size_t count = set.type == 1 ? n : n - 1;
    std::vector<Index> z(count);
    std::vector<std::pair<Index, double>> pick;
    for (std::size_t k = 0; k < count; ++k) {
      z[k] = writer.add_binary(fmt::format("{}_z{}", base, k));
      pick.emplace_back(z[k], 1.0);
    }
    writer.add_row(base + "_pick", pick, -kInfinity, 1.0);
    r.rows += 1;
    for (std::size_t k = 0; k < n; ++k) {
      const Index j = set.columns[k];
      const auto u = static_cast<std::size_t>(j);
      std::vector<Index> cover;
      if (set.type == 1) {
        cover.push_back(z[k]);
      } else {
        if (k > 0) cover.push_back(z[k - 1]);
        if (k + 1 < n) cover.push_back(z[k]);
      }
      const double upper = out.col_upper[u];
      const double lower = out.col_lower[u];
      if (upper > 0.0) {  // x_k <= u_k * (sum of its binaries)
        std::vector<std::pair<Index, double>> entries{{j, 1.0}};
        for (const Index b : cover) entries.emplace_back(b, -upper);
        writer.add_row(fmt::format("{}_ub{}", base, k), entries, -kInfinity, 0.0);
        ++r.rows;
      }
      if (lower < 0.0) {  // x_k >= l_k * (sum of its binaries)
        std::vector<std::pair<Index, double>> entries{{j, 1.0}};
        for (const Index b : cover) entries.emplace_back(b, -lower);
        writer.add_row(fmt::format("{}_lb{}", base, k), entries, 0.0, kInfinity);
        ++r.rows;
      }
    }
    ++r.sets_written;
    r.binaries += static_cast<Count>(count);
  }

  writer.finish();
  r.model = std::move(out);
  return r;
}

Solution restrict_to_original(const Model& original, const ScSosReformulation& r,
                              Solution solution) {
  const Index n = r.model.num_cols();
  const Index m = r.model.num_rows();
  const Index n0 = r.original_cols;
  const Index m0 = r.original_rows;
  cut_to(&solution.col_value, n, n0);
  cut_to(&solution.col_dual, n, n0);
  cut_to(&solution.col_status, n, n0);
  cut_to(&solution.primal_ray, n, n0);
  cut_to(&solution.exact_col_value, n, n0);
  cut_to(&solution.col_ranging_lower, n, n0);
  cut_to(&solution.col_ranging_upper, n, n0);
  cut_to(&solution.row_activity, m, m0);
  cut_to(&solution.row_dual, m, m0);
  cut_to(&solution.row_status, m, m0);
  cut_to(&solution.farkas_dual, m, m0);
  cut_to(&solution.row_ranging_lower, m, m0);
  cut_to(&solution.row_ranging_upper, m, m0);
  // The pool's members are distinct over the reformulated model's integer columns, the added
  // binaries among them; over the original's they may repeat. A pool promises distinct
  // integer assignments of the model it reports on, so the best member of each is kept
  // (members come best first), and with no integer columns that is the first alone.
  std::vector<std::vector<double>> seen;
  std::vector<Solution::PoolEntry> pool;
  for (Solution::PoolEntry& entry : solution.pool) {
    cut_to(&entry.col_value, n, n0);
    if (entry.col_value.size() != static_cast<std::size_t>(n0)) continue;
    std::vector<double> key;
    for (Index j = 0; j < n0; ++j) {
      const auto u = static_cast<std::size_t>(j);
      if (original.col_type[u] == VarType::kInteger)
        key.push_back(std::round(entry.col_value[u]));
    }
    if (std::find(seen.begin(), seen.end(), key) != seen.end()) continue;
    seen.push_back(std::move(key));
    pool.push_back(std::move(entry));
  }
  solution.pool = std::move(pool);
  // A certificate or an IIS of the reformulated model is a statement about rows the
  // original does not have; none is carried over.
  solution.iis_rows.clear();
  solution.iis_col_lo.clear();
  solution.iis_col_hi.clear();
  solution.iis_witnesses.clear();
  if (solution.farkas_dual.size() != static_cast<std::size_t>(m0)) solution.farkas_dual.clear();
  if (solution.primal_ray.size() != static_cast<std::size_t>(n0)) solution.primal_ray.clear();
  if ((solution.status == SolveStatus::kOptimal || solution.status == SolveStatus::kFeasible) &&
      solution.col_value.size() == static_cast<std::size_t>(n0)) {
    // The added binaries cost nothing, so the objective is the same function of the same
    // values; recompute_quality() re-evaluates it against the original rather than trust that.
    solution.recompute_quality(original);
  }
  const std::string note = fmt::format(
      "solved through the binary reformulation (sos_reformulate): {} semi-continuous "
      "column(s) and {} set(s) written with {} binaries and {} rows, {} condition(s) "
      "dropped as always satisfied, {} left to native branching",
      r.semicontinuous_written, r.sets_written, r.binaries, r.rows, r.dropped, r.kept);
  solution.message = solution.message.empty() ? note : solution.message + "; " + note;
  return solution;
}

}  // namespace sankhya::mip
