// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the quadratic rows of a conic model (#755): their classification by the
// LDL^T semidefiniteness test and the pivoted Cholesky that factors a convex one as
// Q = F F'. Split out of conic_form.cpp (#1086), which builds the form from both.

#include "conic/conic_form.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "qp/convexity.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::conic {

// ---- the pivoted Cholesky ----------------------------------------------------------------

DenseFactor pivoted_cholesky(Index k, const std::vector<Index>& rows,
                             const std::vector<Index>& cols,
                             const std::vector<double>& values) {
  const auto n = static_cast<std::size_t>(k);
  std::vector<double> q(n * n, 0.0);  // full symmetric, column-major
  for (std::size_t p = 0; p < values.size(); ++p) {
    const auto i = static_cast<std::size_t>(rows[p]);
    const auto j = static_cast<std::size_t>(cols[p]);
    q[j * n + i] += values[p];
    if (i != j) q[i * n + j] += values[p];
  }
  std::vector<Index> perm(n);
  for (std::size_t i = 0; i < n; ++i) perm[i] = static_cast<Index>(i);
  // EACH PIVOT IS JUDGED AT ITS OWN COLUMN'S SCALE (#1061 review). A floor relative to the
  // largest diagonal dropped real curvature: diag(1e6, 1e-7) came out rank 1 and diag(1e-14)
  // rank 0, turning 1e6 x1^2 + 1e-7 x2^2 <= 1 into a constraint on x1 alone and the
  // minimization of -x2 into an unbounded one. A column's elimination residue is rounding of
  // its own entries, so the floor is tol::kConicCholeskyPivotShare of the largest magnitude
  // in that column of the original Q, which keeps a small but real pivot and still stops at
  // the rounding a rank-deficient Q leaves.
  std::vector<double> floor(n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < n; ++i) floor[j] = std::max(floor[j], std::fabs(q[j * n + i]));
    floor[j] *= tol::kConicCholeskyPivotShare;
  }

  // Outer-product Cholesky with complete (diagonal) pivoting, in place: after step r the
  // leading r columns of `q` hold L's columns in the permuted order.
  Index rank = 0;
  for (std::size_t r = 0; r < n; ++r) {
    // The largest remaining pivot among those above their own column's floor.
    std::size_t pivot = n;
    for (std::size_t i = r; i < n; ++i) {
      if (q[i * n + i] <= floor[i]) continue;
      if (pivot == n || q[i * n + i] > q[pivot * n + pivot]) pivot = i;
    }
    if (pivot == n) break;
    if (pivot != r) {
      std::swap(perm[r], perm[pivot]);
      std::swap(floor[r], floor[pivot]);
      for (std::size_t j = 0; j < n; ++j) std::swap(q[j * n + r], q[j * n + pivot]);  // rows
      for (std::size_t i = 0; i < n; ++i) std::swap(q[r * n + i], q[pivot * n + i]);  // cols
    }
    const double d = std::sqrt(q[r * n + r]);
    q[r * n + r] = d;
    for (std::size_t i = r + 1; i < n; ++i) q[r * n + i] /= d;
    // The whole trailing block, both triangles, so the symmetric pivot swaps above stay
    // exact: only column r's lower part is ever read back as L.
    for (std::size_t j = r + 1; j < n; ++j) {
      const double l_jr = q[r * n + j];
      if (l_jr == 0.0) continue;
      for (std::size_t i = r + 1; i < n; ++i) q[j * n + i] -= q[r * n + i] * l_jr;
    }
    ++rank;
  }
  DenseFactor out;
  out.k = k;
  out.rank = rank;
  // What stopped it must be negligible: a remaining pivot below MINUS its column's floor is
  // curvature of the wrong sign at that column's own scale, and dropping it would solve a
  // different constraint. Refused, not truncated.
  for (std::size_t i = static_cast<std::size_t>(rank); i < n; ++i) {
    if (q[i * n + i] < -floor[i]) {
      out.negative_pivot = q[i * n + i];
      out.negative_column = perm[i];
      return out;
    }
  }
  out.f.assign(n * static_cast<std::size_t>(rank), 0.0);
  for (std::size_t col = 0; col < static_cast<std::size_t>(rank); ++col) {
    for (std::size_t i = col; i < n; ++i) {
      out.f[col * n + static_cast<std::size_t>(perm[i])] = q[col * n + i];
    }
  }
  return out;
}

// ---- classifying the rows -------------------------------------------------------------------

namespace {

/// The terms of row r as a Model whose Hessian (0.5 x'Hx convention) is 2 * scale * Q_r, so
/// check_convexity decides whether scale * Q_r is positive semidefinite.
Model probe_for(const QcqpModel& model, const std::vector<const QuadraticTerm*>& terms,
                double scale) {
  const Index n = model.linear.num_cols();
  Model probe;
  probe.resize_columns(n);
  probe.col_names = model.linear.col_names;
  probe.matrix.reset(0, n);
  probe.matrix.finalize();
  probe.hessian.reset(n, n);
  for (const QuadraticTerm* t : terms) {
    // x_a x_b (a < b) with coefficient v is Q_ab = Q_ba = v/2; the lower-triangle entry of
    // H = 2Q is v. A square v x_a^2 is Q_aa = v, H_aa = 2v.
    probe.hessian.add_entry(t->second, t->first,
                            t->first == t->second ? 2.0 * scale * t->value : scale * t->value);
  }
  probe.hessian.finalize(0.0);
  return probe;
}

}  // namespace

}  // namespace sankhya::conic

namespace sankhya {

std::vector<QuadraticRowClass> classify_quadratic_rows(const QcqpModel& model) {
  std::map<Index, std::vector<const QuadraticTerm*>> by_row;
  for (const QuadraticTerm& t : model.quadratic) by_row[t.row].push_back(&t);
  std::vector<QuadraticRowClass> out;
  for (const auto& [row, terms] : by_row) {
    QuadraticRowClass c;
    c.row = row;
    const auto u = static_cast<std::size_t>(row);
    const bool has_lower = is_finite_bound(model.linear.row_lower[u]);
    const bool has_upper = is_finite_bound(model.linear.row_upper[u]);
    const std::string name =
        u < model.linear.row_names.size() && !model.linear.row_names[u].empty()
            ? model.linear.row_names[u]
            : fmt::format("row {}", row);
    const qp::ConvexityResult psd = qp::check_convexity(conic::probe_for(model, terms, 1.0));
    if (psd.verdict == qp::Convexity::kConvex) {
      if (has_lower) {
        c.verdict = QuadraticRowClass::Verdict::kNonConvex;
        c.detail = fmt::format(
            "{}: its matrix is positive semidefinite, so the row is a convex function with a "
            "finite LOWER side{}, which is not a convex constraint",
            name, has_upper ? " (and an upper side)" : "");
      } else {
        c.verdict = QuadraticRowClass::Verdict::kConvex;
        c.sign = 1;
        c.detail = fmt::format("{}: positive semidefinite, upper side only", name);
      }
      out.push_back(std::move(c));
      continue;
    }
    const qp::ConvexityResult nsd = qp::check_convexity(conic::probe_for(model, terms, -1.0));
    if (nsd.verdict == qp::Convexity::kConvex) {
      if (has_upper) {
        c.verdict = QuadraticRowClass::Verdict::kNonConvex;
        c.detail = fmt::format(
            "{}: its matrix is negative semidefinite, so the row is a concave function with a "
            "finite UPPER side{}, which is not a convex constraint",
            name, has_lower ? " (and a lower side)" : "");
      } else {
        c.verdict = QuadraticRowClass::Verdict::kConvex;
        c.sign = -1;
        c.detail = fmt::format("{}: negative semidefinite, lower side only", name);
      }
      out.push_back(std::move(c));
      continue;
    }
    if (psd.verdict == qp::Convexity::kIndefinite) {
      c.verdict = QuadraticRowClass::Verdict::kNonConvex;
      c.witness = psd.witness;
      c.detail = fmt::format("{}: indefinite - {} (curvature {:.3e} along the witness)", name,
                             psd.detail, psd.witness_curvature);
    } else {
      c.verdict = QuadraticRowClass::Verdict::kUnverified;
      c.detail =
          fmt::format("{}: the semidefiniteness test did not decide ({})", name, psd.detail);
    }
    out.push_back(std::move(c));
  }
  return out;
}

}  // namespace sankhya

namespace sankhya::conic {

TermsByRow index_terms_by_row(const QcqpModel& model) {
  TermsByRow out(static_cast<std::size_t>(model.linear.num_rows()));
  for (const QuadraticTerm& t : model.quadratic) {
    out[static_cast<std::size_t>(t.row)].push_back(&t);
  }
  return out;
}

}  // namespace sankhya::conic
