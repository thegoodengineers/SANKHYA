// SPDX-License-Identifier: Apache-2.0
// SANKHYA - building the conic standard form (#755); see conic_form.hpp for what it is.

#include "conic/conic_form.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <utility>

#include <fmt/format.h>

#include "qp/convexity.hpp"
#include "sankhya/tolerances.hpp"

// ---- building the form
// -------------------------------------------------------------------------

namespace sankhya::conic {
namespace {

constexpr double kRoot2Inverse = 0.70710678118654752440;

/// A reference to one or two logical columns with coefficients: what a model column, a
/// rotated cone's head or a slack expands to.
struct Ref {
  Index a = -1;
  Index b = -1;
  double ca = 1.0;
  double cb = 0.0;
};

enum class Kind : std::uint8_t { kFree, kNonneg, kCone };

class Builder {
 public:
  explicit Builder(const QcqpModel& model) : model_(model) {}

  Index column(Kind kind) {
    kind_.push_back(kind);
    cone_of_.push_back(-1);
    cost_.push_back(0.0);
    return static_cast<Index>(kind_.size()) - 1;
  }
  /// A plain quadratic cone of `size` members; member i is column start + i.
  Index cone(Index size) {
    const Index start = static_cast<Index>(kind_.size());
    const auto id = static_cast<Index>(cone_sizes_.size());
    for (Index i = 0; i < size; ++i) {
      column(Kind::kCone);
      cone_of_.back() = id;
    }
    cone_sizes_.push_back(size);
    cone_starts_.push_back(start);
    return start;
  }
  /// A rotated cone 2 p q >= ||z||^2 as the image of a plain one: p = (u_0 + u_1) / sqrt 2,
  /// q = (u_0 - u_1) / sqrt 2 (the map is orthogonal and symmetric, so the dual maps alike).
  void rotated(Index members, Ref* p, Ref* q, Index* z_start) {
    const Index start = cone(members);
    *p = Ref{start, start + 1, kRoot2Inverse, kRoot2Inverse};
    *q = Ref{start, start + 1, kRoot2Inverse, -kRoot2Inverse};
    *z_start = start + 2;
  }
  Index row(double rhs) {
    rhs_.push_back(rhs);
    return static_cast<Index>(rhs_.size()) - 1;
  }
  void entry(Index row, const Ref& ref, double value) {
    if (value == 0.0) return;
    tri_.push_back({row, ref.a, value * ref.ca});
    if (ref.b >= 0 && ref.cb != 0.0) tri_.push_back({row, ref.b, value * ref.cb});
  }
  void cost(const Ref& ref, double value) {
    cost_[static_cast<std::size_t>(ref.a)] += value * ref.ca;
    if (ref.b >= 0) cost_[static_cast<std::size_t>(ref.b)] += value * ref.cb;
  }
  /// coef * x_j for a MODEL column j: through its map, the shift going to the right-hand side.
  void entry_model(Index row, Index j, double coef) {
    const ColumnMap& map = col_map_[static_cast<std::size_t>(j)];
    rhs_[static_cast<std::size_t>(row)] -= coef * map.shift;
    if (map.a < 0) return;
    entry(row, Ref{map.a, map.b, map.a_scale, map.b_scale}, coef);
  }
  void cost_model(Index j, double coef) {
    const ColumnMap& map = col_map_[static_cast<std::size_t>(j)];
    offset_ += coef * map.shift;
    if (map.a < 0) return;
    cost(Ref{map.a, map.b, map.a_scale, map.b_scale}, coef);
  }

  void run(const std::vector<QuadraticRowClass>& classes, ConicForm* form);

 private:
  struct Triplet {
    Index row;
    Index col;
    double value;
  };
  void place_columns();
  void place_cones();
  void linear_rows(const std::vector<QuadraticRowClass>& classes);
  bool quadratic_row(const QuadraticRowClass& cls);
  bool quadratic_objective();
  void finish(ConicForm* form);

  const QcqpModel& model_;
  CsrView rows_view_;
  TermsByRow terms_;  ///< the model's quadratic terms by row, built once in run()
  std::vector<Kind> kind_;
  std::vector<Index> cone_of_;
  std::vector<double> cost_;
  std::vector<Index> cone_sizes_;
  std::vector<Index> cone_starts_;
  std::vector<double> rhs_;
  std::vector<Triplet> tri_;
  std::vector<ColumnMap> col_map_;
  std::vector<Index> row_of_;
  std::vector<double> row_sign_;
  std::vector<QuadraticRowMap> quadratic_rows_;
  double sigma_ = 1.0;
  double offset_ = 0.0;
  std::string problem_;
};

void Builder::place_cones() {
  // Explicit cones first: their members are the model's own columns.
  const Model& lin = model_.linear;
  col_map_.assign(static_cast<std::size_t>(lin.num_cols()), ColumnMap{});
  std::vector<char> placed(static_cast<std::size_t>(lin.num_cols()), 0);
  for (const Cone& spec : model_.cones) {
    const auto k = static_cast<Index>(spec.columns.size());
    std::vector<Ref> refs(spec.columns.size());
    if (spec.kind == Cone::Kind::kRotated) {
      Index z = 0;
      rotated(k, &refs[0], &refs[1], &z);
      for (Index i = 2; i < k; ++i) refs[static_cast<std::size_t>(i)] = Ref{z + i - 2};
    } else {
      const Index start = this->cone(k);
      for (Index i = 0; i < k; ++i) refs[static_cast<std::size_t>(i)] = Ref{start + i};
    }
    for (std::size_t i = 0; i < spec.columns.size(); ++i) {
      const auto j = static_cast<std::size_t>(spec.columns[i]);
      col_map_[j] = ColumnMap{refs[i].a, refs[i].b, refs[i].ca, refs[i].cb, 0.0};
      placed[j] = 1;
      // Bounds on a cone member stay as rows with a slack; the cone owns the column.
      const double l = lin.col_lower[j];
      const double u = lin.col_upper[j];
      if (is_finite_bound(l) && is_finite_bound(u) && l == u) {
        entry(row(l), refs[i], 1.0);
      } else {
        if (is_finite_bound(l)) {
          const Index r = row(l);
          entry(r, refs[i], 1.0);
          entry(r, Ref{column(Kind::kNonneg)}, -1.0);
        }
        if (is_finite_bound(u)) {
          const Index r = row(u);
          entry(r, refs[i], 1.0);
          entry(r, Ref{column(Kind::kNonneg)}, 1.0);
        }
      }
    }
  }
  for (Index j = 0; j < lin.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (placed[u]) continue;
    const double lo = lin.col_lower[u];
    const double hi = lin.col_upper[u];
    const bool has_lo = is_finite_bound(lo);
    const bool has_hi = is_finite_bound(hi);
    if (has_lo && has_hi && lo == hi) {
      col_map_[u] = ColumnMap{-1, -1, 0.0, 0.0, lo};
    } else if (has_lo && has_hi) {
      // x = lo + s, s >= 0, s + t = hi - lo.
      const Index s = column(Kind::kNonneg);
      const Index t = column(Kind::kNonneg);
      const Index r = row(hi - lo);
      entry(r, Ref{s}, 1.0);
      entry(r, Ref{t}, 1.0);
      col_map_[u] = ColumnMap{s, -1, 1.0, 0.0, lo};
    } else if (has_lo) {
      col_map_[u] = ColumnMap{column(Kind::kNonneg), -1, 1.0, 0.0, lo};
    } else if (has_hi) {
      col_map_[u] = ColumnMap{column(Kind::kNonneg), -1, -1.0, 0.0, hi};
    } else {
      col_map_[u] = ColumnMap{column(Kind::kFree), -1, 1.0, 0.0, 0.0};
    }
  }
}

void Builder::linear_rows(const std::vector<QuadraticRowClass>& classes) {
  const Model& lin = model_.linear;
  const Index m = lin.num_rows();
  row_of_.assign(static_cast<std::size_t>(m), -1);
  row_sign_.assign(static_cast<std::size_t>(m), 1.0);
  std::vector<char> quadratic(static_cast<std::size_t>(m), 0);
  for (const QuadraticRowClass& c : classes) quadratic[static_cast<std::size_t>(c.row)] = 1;
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    if (quadratic[u]) continue;
    const double lo = lin.row_lower[u];
    const double hi = lin.row_upper[u];
    const bool has_lo = is_finite_bound(lo);
    const bool has_hi = is_finite_bound(hi);
    if (!has_lo && !has_hi) continue;
    Index r = -1;
    if (has_lo && has_hi && lo == hi) {
      r = row(lo);
    } else if (has_lo) {
      r = row(lo);
      entry(r, Ref{column(Kind::kNonneg)}, -1.0);
      if (has_hi) {
        // The slack carries the range: s + t = hi - lo.
        const Index s = tri_.back().col;
        const Index t = column(Kind::kNonneg);
        const Index range = row(hi - lo);
        entry(range, Ref{s}, 1.0);
        entry(range, Ref{t}, 1.0);
      }
    } else {
      r = row(hi);
      entry(r, Ref{column(Kind::kNonneg)}, 1.0);
    }
    row_of_[u] = r;
    const ColumnView row_view = rows_view_.row(i);
    for (Index p = 0; p < row_view.size; ++p)
      entry_model(r, row_view.rows[p], row_view.values[p]);
  }
}

bool Builder::quadratic_row(const QuadraticRowClass& cls) {
  const Model& lin = model_.linear;
  const auto u = static_cast<std::size_t>(cls.row);
  const double sign = static_cast<double>(cls.sign);
  const bool has_lo = is_finite_bound(lin.row_lower[u]);
  const bool has_hi = is_finite_bound(lin.row_upper[u]);
  if (!has_lo && !has_hi) return true;  // a free row: nothing to enforce
  // The row as sign * (a'x + x'Qx) <= sign * side, with sign * Q positive semidefinite.
  const double side = sign * (cls.sign > 0 ? lin.row_upper[u] : lin.row_lower[u]);
  // Q over its support, compacted.
  const std::vector<const QuadraticTerm*>& terms = terms_[u];
  std::vector<Index> support;
  for (const QuadraticTerm* t : terms) {
    support.push_back(t->first);
    support.push_back(t->second);
  }
  std::sort(support.begin(), support.end());
  support.erase(std::unique(support.begin(), support.end()), support.end());
  const auto k = static_cast<Index>(support.size());
  if (k > tol::kConicDenseFactorLimit) {
    problem_ = fmt::format(
        "quadratic row {} has {} columns in its products; the dense factorization of its "
        "matrix is limited to {} (tolerances.hpp kConicDenseFactorLimit)",
        cls.row, k, tol::kConicDenseFactorLimit);
    return false;
  }
  std::vector<Index> qr, qc;
  std::vector<double> vals;
  for (const QuadraticTerm* t : terms) {
    const auto a = static_cast<Index>(
        std::lower_bound(support.begin(), support.end(), t->first) - support.begin());
    const auto b = static_cast<Index>(
        std::lower_bound(support.begin(), support.end(), t->second) - support.begin());
    qr.push_back(std::max(a, b));
    qc.push_back(std::min(a, b));
    vals.push_back(a == b ? sign * t->value : 0.5 * sign * t->value);
  }
  const DenseFactor factor = pivoted_cholesky(k, qr, qc, vals);
  if (factor.negative_column >= 0) {
    problem_ = fmt::format(
        "quadratic row {} has a pivot of {:.3e} at column {}, negative beyond rounding at that "
        "column's own scale; refused rather than factored without it",
        cls.row, factor.negative_pivot,
        support[static_cast<std::size_t>(factor.negative_column)]);
    return false;
  }

  Ref w;
  Ref v;
  Index z = 0;
  rotated(factor.rank + 2, &w, &v, &z);
  // w + sign * a'x = side, the row whose multiplier is the model row's.
  const Index primary = row(side);
  entry(primary, w, 1.0);
  const ColumnView row_view = rows_view_.row(cls.row);
  for (Index p = 0; p < row_view.size; ++p) {
    entry_model(primary, row_view.rows[p], sign * row_view.values[p]);
  }
  row_of_[u] = primary;
  row_sign_[u] = sign;
  entry(row(0.5), v, 1.0);
  // The z rows are the next `rank` rows, in order (row() appends; finish() keeps row order).
  quadratic_rows_.push_back(
      {cls.row, sign, primary, static_cast<Index>(rhs_.size()), support, factor});
  for (Index col = 0; col < factor.rank; ++col) {
    const Index r = row(0.0);
    entry(r, Ref{z + col}, 1.0);
    for (Index i = 0; i < k; ++i) {
      const double f = factor.at(i, col);
      if (f != 0.0) entry_model(r, support[static_cast<std::size_t>(i)], -f);
    }
  }
  return true;
}

bool Builder::quadratic_objective() {
  const Model& lin = model_.linear;
  if (!lin.has_quadratic_objective()) return true;
  const Index n = lin.num_cols();
  std::vector<Index> support;
  for (Index j = 0; j < n; ++j) {
    const ColumnView column = lin.hessian.column(j);
    if (column.size == 0) continue;
    support.push_back(j);
    for (Index p = 0; p < column.size; ++p) support.push_back(column.rows[p]);
  }
  std::sort(support.begin(), support.end());
  support.erase(std::unique(support.begin(), support.end()), support.end());
  const auto k = static_cast<Index>(support.size());
  if (k > tol::kConicDenseFactorLimit) {
    problem_ = fmt::format(
        "the quadratic objective has {} columns; the dense factorization of its Hessian is "
        "limited to {} (tolerances.hpp kConicDenseFactorLimit)",
        k, tol::kConicDenseFactorLimit);
    return false;
  }
  std::vector<Index> qr, qc;
  std::vector<double> vals;
  for (Index j = 0; j < n; ++j) {
    const ColumnView column = lin.hessian.column(j);
    if (column.size == 0) continue;
    const auto cj = static_cast<Index>(std::lower_bound(support.begin(), support.end(), j) -
                                       support.begin());
    for (Index p = 0; p < column.size; ++p) {
      const auto ri = static_cast<Index>(
          std::lower_bound(support.begin(), support.end(), column.rows[p]) - support.begin());
      qr.push_back(std::max(ri, cj));
      qc.push_back(std::min(ri, cj));
      vals.push_back(sigma_ * column.values[p]);
    }
  }
  const DenseFactor factor = pivoted_cholesky(k, qr, qc, vals);
  if (factor.negative_column >= 0) {
    problem_ = fmt::format(
        "the quadratic objective's Hessian has a pivot of {:.3e} at column {}, negative beyond "
        "rounding at that column's own scale; refused rather than factored without it",
        factor.negative_pivot, support[static_cast<std::size_t>(factor.negative_column)]);
    return false;
  }
  // 0.5 x'Hx = 0.5 ||z||^2 <= 0.5 t with 2 t v >= ||z||^2, v = 1/2: the objective takes 0.5 t.
  Ref t;
  Ref v;
  Index z = 0;
  rotated(factor.rank + 2, &t, &v, &z);
  cost(t, 0.5);
  entry(row(0.5), v, 1.0);
  for (Index col = 0; col < factor.rank; ++col) {
    const Index r = row(0.0);
    entry(r, Ref{z + col}, 1.0);
    for (Index i = 0; i < k; ++i) {
      const double f = factor.at(i, col);
      if (f != 0.0) entry_model(r, support[static_cast<std::size_t>(i)], -f);
    }
  }
  return true;
}

void Builder::finish(ConicForm* form) {
  // Positions: free, then non-negative, then the cones in order of creation.
  const auto total = kind_.size();
  std::vector<Index> position(total, -1);
  Index next = 0;
  for (std::size_t j = 0; j < total; ++j) {
    if (kind_[j] == Kind::kFree) position[j] = next++;
  }
  form->free_n = next;
  for (std::size_t j = 0; j < total; ++j) {
    if (kind_[j] == Kind::kNonneg) position[j] = next++;
  }
  form->nonneg_n = next - form->free_n;
  for (std::size_t c = 0; c < cone_sizes_.size(); ++c) {
    form->cones.push_back({next, cone_sizes_[c]});
    for (Index i = 0; i < cone_sizes_[c]; ++i) {
      position[static_cast<std::size_t>(cone_starts_[c] + i)] = next++;
    }
  }
  form->n = next;
  form->m = static_cast<Index>(rhs_.size());
  form->c.assign(static_cast<std::size_t>(form->n), 0.0);
  for (std::size_t j = 0; j < total; ++j) {
    form->c[static_cast<std::size_t>(position[j])] = cost_[j];
  }
  form->b = rhs_;
  form->a.reset(form->m, form->n);
  form->a.reserve(tri_.size());
  for (const Triplet& t : tri_) {
    form->a.add_entry(t.row, position[static_cast<std::size_t>(t.col)], t.value);
  }
  form->a.finalize(0.0);
  for (ColumnMap& map : col_map_) {
    if (map.a >= 0) map.a = position[static_cast<std::size_t>(map.a)];
    if (map.b >= 0) map.b = position[static_cast<std::size_t>(map.b)];
  }
  form->col_map = std::move(col_map_);
  form->row_of = std::move(row_of_);
  form->row_sign = std::move(row_sign_);
  form->quadratic_rows = std::move(quadratic_rows_);
  form->sigma = sigma_;
  form->offset = offset_;
}

void Builder::run(const std::vector<QuadraticRowClass>& classes, ConicForm* form) {
  const Model& lin = model_.linear;
  sigma_ = lin.sense_multiplier();
  offset_ = sigma_ * lin.objective_offset;
  rows_view_.build(lin.matrix);
  terms_ = index_terms_by_row(model_);
  place_cones();
  for (Index j = 0; j < lin.num_cols(); ++j) {
    cost_model(j, sigma_ * lin.col_cost[static_cast<std::size_t>(j)]);
  }
  linear_rows(classes);
  for (const QuadraticRowClass& cls : classes) {
    if (!quadratic_row(cls)) break;
  }
  if (problem_.empty()) quadratic_objective();
  if (!problem_.empty()) {
    form->problem = problem_;
    return;
  }
  finish(form);
}

}  // namespace

ConicForm build_conic_form(const QcqpModel& model,
                           const std::vector<QuadraticRowClass>& classes) {
  ConicForm form;
  for (const QuadraticRowClass& c : classes) {
    if (c.verdict != QuadraticRowClass::Verdict::kConvex) {
      form.problem = "a quadratic row is not convex: " + c.detail;
      return form;
    }
  }
  Builder builder(model);
  builder.run(classes, &form);
  return form;
}

}  // namespace sankhya::conic
