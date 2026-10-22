// SPDX-License-Identifier: Apache-2.0
// SANKHYA - column substitution from equality rows: the reductions. The derivation, the
// citations and the dual recovery are in substitute.hpp; the replay is in
// substitute_postsolve.cpp.

#include "presolve/substitute.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <vector>

#include "sankhya/tolerances.hpp"

namespace sankhya::presolve {

namespace {

using Entry = std::pair<Index, double>;

/// Passes over the rows; each pass that substitutes nothing ends the stage first.
constexpr int kMaxPasses = 8;
double a527_env(const char* name, double fallback) {
  const char* v = std::getenv(name);
  return v ? std::atof(v) : fallback;
}

bool finite(double v) {
  return is_finite_bound(v);
}

struct Work {
  std::vector<std::vector<Entry>> rows;  ///< live entries, by row (authoritative)
  std::vector<std::vector<Index>> cols;  ///< rows a column may appear in (may be stale)
  std::vector<Index> count;              ///< live entries per column
  std::vector<double> col_lower, col_upper, cost, row_lower, row_upper;
  std::vector<char> row_dead, col_dead;
  double offset = 0.0;

  [[nodiscard]] std::ptrdiff_t find(Index row, Index col) const {
    const auto& r = rows[static_cast<std::size_t>(row)];
    for (std::size_t p = 0; p < r.size(); ++p) {
      if (r[p].first == col) return static_cast<std::ptrdiff_t>(p);
    }
    return -1;
  }

  /// Column `col`'s live entries outside `skip_row`, (row, value), rows ascending.
  [[nodiscard]] std::vector<Entry> column(Index col, Index skip_row) {
    auto& list = cols[static_cast<std::size_t>(col)];
    std::sort(list.begin(), list.end());
    list.erase(std::unique(list.begin(), list.end()), list.end());
    std::vector<Entry> out;
    std::size_t kept = 0;
    for (const Index i : list) {
      if (row_dead[static_cast<std::size_t>(i)] != 0) continue;
      const std::ptrdiff_t p = find(i, col);
      if (p < 0) continue;
      list[kept++] = i;
      if (i != skip_row)
        out.emplace_back(i,
                         rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(p)].second);
    }
    list.resize(kept);
    return out;
  }
};

/// Bounds of sum_{k != skip} a_k x_k over row `row`'s live entries: [lo, hi], each possibly
/// infinite.
std::pair<double, double> activity_without(const Work& w, Index row, Index skip) {
  double lo = 0.0, hi = 0.0;
  bool lo_inf = false, hi_inf = false;
  for (const auto& [k, a] : w.rows[static_cast<std::size_t>(row)]) {
    if (k == skip) continue;
    const double l = w.col_lower[static_cast<std::size_t>(k)];
    const double u = w.col_upper[static_cast<std::size_t>(k)];
    const double at_lo = a > 0.0 ? l : u;  // the bound that gives the least activity
    const double at_hi = a > 0.0 ? u : l;
    if (finite(at_lo))
      lo += a * at_lo;
    else
      lo_inf = true;
    if (finite(at_hi))
      hi += a * at_hi;
    else
      hi_inf = true;
  }
  return {lo_inf ? -kInfinity : lo, hi_inf ? kInfinity : hi};
}

/// The range a x_e = rhs - rest takes, divided by a: x_e's implied bounds from the row.
std::pair<double, double> implied_range(double a, double rhs, std::pair<double, double> rest) {
  const double lo = finite(rest.second) ? (rhs - rest.second) / a : -kInfinity;  // a > 0
  const double hi = finite(rest.first) ? (rhs - rest.first) / a : kInfinity;
  if (a > 0.0) return {lo, hi};
  // a < 0: dividing by a negative number swaps the ends.
  return {finite(rest.first) ? (rhs - rest.first) / a : -kInfinity,
          finite(rest.second) ? (rhs - rest.second) / a : kInfinity};
}

/// Eliminate column e through equality row r (substitute.hpp has the algebra).
void eliminate(Work* w, Index r, Index e, Substitution* rec) {
  const auto ur = static_cast<std::size_t>(r);
  const auto ue = static_cast<std::size_t>(e);
  const double a = rec->coefficient;
  const double rhs = rec->rhs;
  const double ce = w->cost[ue];
  for (const auto& [k, ark] : rec->row_entries) {
    w->cost[static_cast<std::size_t>(k)] -= ce / a * ark;
  }
  w->offset += ce * rhs / a;
  for (const auto& [i, aie] : rec->column_entries) {
    const auto ui = static_cast<std::size_t>(i);
    const double f = aie / a;
    auto& row = w->rows[ui];
    row.erase(row.begin() + w->find(i, e));
    for (const auto& [k, ark] : rec->row_entries) {
      const double added = -f * ark;
      const std::ptrdiff_t p = w->find(i, k);
      if (p >= 0) {
        double& v = row[static_cast<std::size_t>(p)].second;
        const double before = v;
        v += added;
        // Cancellation is rounding, not data (#792): the same relative test presolve.cpp
        // applies to its own fill-in.
        if (std::fabs(v) <=
            tol::kFillInCancellation * std::max(std::fabs(before), std::fabs(added))) {
          row.erase(row.begin() + p);
          --w->count[static_cast<std::size_t>(k)];
        }
      } else if (added != 0.0) {
        row.emplace_back(k, added);
        w->cols[static_cast<std::size_t>(k)].push_back(i);
        ++w->count[static_cast<std::size_t>(k)];
      }
    }
    if (finite(w->row_lower[ui])) w->row_lower[ui] -= f * rhs;
    if (finite(w->row_upper[ui])) w->row_upper[ui] -= f * rhs;
  }
  for (const auto& [k, ark] : w->rows[ur]) --w->count[static_cast<std::size_t>(k)];
  w->rows[ur].clear();
  w->row_dead[ur] = 1;
  w->col_dead[ue] = 1;
  w->count[ue] = 0;
}

/// Try to substitute a column out of equality row r. True when one was.
bool try_row(Work* w, Index r, SubstitutionResult* out) {
  const auto ur = static_cast<std::size_t>(r);
  const auto& row = w->rows[ur];
  const auto len = static_cast<Index>(row.size());
  if (len < 2) return false;
  if (len > 2 && len > tol::kSubstitutionMaxRowLength) return false;
  const double rhs = w->row_lower[ur];
  double row_max = 0.0;
  for (const auto& [k, a] : row) row_max = std::max(row_max, std::fabs(a));
  // Fewest entries first: the least fill-in.
  std::vector<Entry> order(row.begin(), row.end());
  std::stable_sort(order.begin(), order.end(), [&](const Entry& x, const Entry& y) {
    return w->count[static_cast<std::size_t>(x.first)] <
           w->count[static_cast<std::size_t>(y.first)];
  });
  for (const auto& [e, a] : order) {
    const auto ue = static_cast<std::size_t>(e);
    const Index count = w->count[ue];
    if (count > tol::kSubstitutionMaxColumnCount) continue;
    if ((len - 1) * (count - 1) > tol::kSubstitutionMaxFill) continue;
    if (std::fabs(a) < a527_env("A527_THR", tol::kSubstitutionPivotThreshold) * row_max) continue;
    std::vector<Entry> column = w->column(e, r);
    double col_max = std::fabs(a);
    for (const auto& entry : column) col_max = std::max(col_max, std::fabs(entry.second));
    if (std::fabs(a) < a527_env("A527_THR", tol::kSubstitutionPivotThreshold) * col_max) continue;

    const auto implied = implied_range(a, rhs, activity_without(*w, r, e));
    const bool implied_free =
        (!finite(w->col_lower[ue]) || implied.first >= w->col_lower[ue]) &&
        (!finite(w->col_upper[ue]) || implied.second <= w->col_upper[ue]);
    Substitution rec;
    if (!implied_free) {
      if (len != 2) continue;
      // DOUBLETON: carry x_e's box onto the partner, b x_k = rhs - a x_e.
      const Entry partner = row[0].first == e ? row[1] : row[0];
      const auto uk = static_cast<std::size_t>(partner.first);
      const double b = partner.second;
      if (std::fabs(b) < a527_env("A527_THR", tol::kSubstitutionPivotThreshold) * row_max) continue;
      const double le = w->col_lower[ue], he = w->col_upper[ue];
      // x_k = (rhs - a x_e) / b, monotone in x_e with slope -a / b.
      const double at_le =
          finite(le) ? (rhs - a * le) / b : (-a / b > 0 ? -kInfinity : kInfinity);
      const double at_he =
          finite(he) ? (rhs - a * he) / b : (-a / b > 0 ? kInfinity : -kInfinity);
      const double lo = std::max(w->col_lower[uk], std::min(at_le, at_he));
      const double hi = std::min(w->col_upper[uk], std::max(at_le, at_he));
      if (finite(lo) && finite(hi) && lo > hi) continue;  // infeasible: the engine says so
      rec.carried = true;
      rec.partner = partner.first;
      rec.partner_coefficient = b;
      rec.partner_lower = w->col_lower[uk];
      rec.partner_upper = w->col_upper[uk];
      rec.partner_new_lower = lo;
      rec.partner_new_upper = hi;
      rec.partner_cost = w->cost[uk];
      rec.partner_entries = w->column(partner.first, r);
      w->col_lower[uk] = lo;
      w->col_upper[uk] = hi;
      ++out->carried;
    } else {
      ++out->implied_free;
    }
    rec.row = r;
    rec.column = e;
    rec.coefficient = a;
    rec.rhs = rhs;
    rec.cost = w->cost[ue];
    rec.lower = w->col_lower[ue];
    rec.upper = w->col_upper[ue];
    for (const auto& entry : row) {
      if (entry.first != e) rec.row_entries.push_back(entry);
    }
    rec.column_entries = std::move(column);
    eliminate(w, r, e, &rec);
    out->records.push_back(std::move(rec));
    return true;
  }
  return false;
}

}  // namespace

SubstitutionResult substitute(const Model& model, const Options& options, Logger& logger) {
  SubstitutionResult out;
  const Index n = model.num_cols();
  const Index m = model.num_rows();
  const bool applies = !model.has_integrality() && !model.has_quadratic_objective() &&
                       !model.has_semicontinuous_or_sos();
  const double feasibility = options.get_double("primal_feasibility_tolerance");
  // Without an equality row there is nothing to substitute from: no work, no model copy.
  bool any_equality = false;
  for (Index i = 0; i < m && !any_equality; ++i) any_equality = model.is_equality_row(i);
  if (!applies || !any_equality) return out;

  Work w;
  w.rows.resize(static_cast<std::size_t>(m));
  w.cols.resize(static_cast<std::size_t>(n));
  w.count.assign(static_cast<std::size_t>(n), 0);
  w.col_lower = model.col_lower;
  w.col_upper = model.col_upper;
  w.cost = model.col_cost;
  w.row_lower = model.row_lower;
  w.row_upper = model.row_upper;
  w.row_dead.assign(static_cast<std::size_t>(m), 0);
  w.col_dead.assign(static_cast<std::size_t>(n), 0);
  w.offset = model.objective_offset;
  for (Index j = 0; j < n; ++j) {
    const ColumnView view = model.matrix.column(j);
    for (Index p = 0; p < view.size; ++p) {
      if (view.values[p] == 0.0) continue;
      w.rows[static_cast<std::size_t>(view.rows[p])].emplace_back(j, view.values[p]);
      w.cols[static_cast<std::size_t>(j)].push_back(view.rows[p]);
      ++w.count[static_cast<std::size_t>(j)];
    }
  }

  for (int pass = 0; applies && pass < kMaxPasses; ++pass) {
    bool changed = false;
    for (Index r = 0; r < m; ++r) {
      const auto ur = static_cast<std::size_t>(r);
      if (w.row_dead[ur] != 0) continue;
      const bool equality = finite(w.row_lower[ur]) && w.row_lower[ur] == w.row_upper[ur];
      if (w.rows[ur].empty()) {
        // Emptied by cancellation: removed when zero is inside its bounds, its dual zero.
        const bool holds = (!finite(w.row_lower[ur]) || w.row_lower[ur] <= feasibility) &&
                           (!finite(w.row_upper[ur]) || w.row_upper[ur] >= -feasibility);
        if (holds && !out.records.empty()) {
          Substitution rec;
          rec.row = r;
          w.row_dead[ur] = 1;
          out.records.push_back(std::move(rec));
        }
        continue;
      }
      if (equality && try_row(&w, r, &out)) changed = true;
    }
    if (!changed) break;
  }

  // Too little removed to be worth a different path (tol::kSubstitutionMinRowFraction): the
  // records are dropped before any model is built, so the declined stage costs one pass.
  if (!out.reduced_anything()) return out;
  const auto removed = std::count(w.row_dead.begin(), w.row_dead.end(), char{1});
  if (static_cast<double>(removed) <
      a527_env("A527_MINFRAC", tol::kSubstitutionMinRowFraction) * static_cast<double>(m)) {
    logger.info(
        "Substitution: {} of {} rows removable, under {:.0f}%; not applied (the model goes on "
        "as presolve left it)",
        removed, m, 100.0 * tol::kSubstitutionMinRowFraction);
    out.records.clear();
    return out;
  }

  // The reduced model, live rows and columns in their input order.
  Model& reduced = out.model;
  reduced.name = model.name;
  reduced.source_path = model.source_path;
  reduced.sense = model.sense;
  reduced.objective_offset = w.offset;
  std::vector<Index> col_new(static_cast<std::size_t>(n), -1);
  std::vector<Index> row_new(static_cast<std::size_t>(m), -1);
  for (Index j = 0; j < n; ++j) {
    if (w.col_dead[static_cast<std::size_t>(j)] != 0) continue;
    col_new[static_cast<std::size_t>(j)] = static_cast<Index>(out.col_to_input.size());
    out.col_to_input.push_back(j);
  }
  for (Index i = 0; i < m; ++i) {
    if (w.row_dead[static_cast<std::size_t>(i)] != 0) continue;
    row_new[static_cast<std::size_t>(i)] = static_cast<Index>(out.row_to_input.size());
    out.row_to_input.push_back(i);
  }
  const auto nn = static_cast<Index>(out.col_to_input.size());
  const auto mm = static_cast<Index>(out.row_to_input.size());
  reduced.resize_columns(nn);
  reduced.resize_rows(mm);
  const bool col_names = model.col_names.size() == static_cast<std::size_t>(n);
  const bool row_names = model.row_names.size() == static_cast<std::size_t>(m);
  if (col_names) reduced.col_names.resize(static_cast<std::size_t>(nn));
  if (row_names) reduced.row_names.resize(static_cast<std::size_t>(mm));
  for (Index j = 0; j < nn; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const auto from = static_cast<std::size_t>(out.col_to_input[u]);
    reduced.col_cost[u] = w.cost[from];
    reduced.col_lower[u] = w.col_lower[from];
    reduced.col_upper[u] = w.col_upper[from];
    reduced.col_type[u] = VarType::kContinuous;
    if (col_names) reduced.col_names[u] = model.col_names[from];
  }
  reduced.matrix.reset(mm, nn);
  for (Index i = 0; i < mm; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const auto from = static_cast<std::size_t>(out.row_to_input[u]);
    reduced.row_lower[u] = w.row_lower[from];
    reduced.row_upper[u] = w.row_upper[from];
    if (row_names) reduced.row_names[u] = model.row_names[from];
    for (const auto& [k, a] : w.rows[from]) {
      reduced.matrix.add_entry(i, col_new[static_cast<std::size_t>(k)], a);
    }
  }
  reduced.matrix.finalize(0.0);

  if (out.reduced_anything()) {
    logger.info(
        "Substitution: {} rows -> {}, {} columns -> {}, {} nonzeros -> {} ({} implied free, "
        "{} doubleton equations with a carried box)",
        m, mm, n, nn, model.num_nonzeros(), reduced.num_nonzeros(), out.implied_free,
        out.carried);
  }
  return out;
}

}  // namespace sankhya::presolve
