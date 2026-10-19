// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the conic standard form the interior point iterates on (#755), built from a
// QcqpModel whose quadratic rows are convex, and the map back to that model's columns and rows.
//
//     minimize  c'x    subject to  A x = b,   x in K = R^f x R_+^l x Q^{k_1} x ... x Q^{k_p}
//
// with the free block first, the non-negative block next and every cone a contiguous block
// after them. Nothing but equalities and the cone: a bound becomes a non-negative column or a
// row with a slack, a ranged row two slacks, a rotated cone the orthogonal image of a plain
// one, a convex quadratic row a'x + x'Qx <= u the rotated cone 2 w v >= ||z||^2 over
// w = u - a'x, v = 1/2 and z = F'x where Q = F F' (Alizadeh & Goldfarb 2003, section 2.2;
// Lobo, Vandenberghe, Boyd & Lebret, "Applications of second-order cone programming",
// Linear Algebra Appl. 284 (1998), section 2.2), and the quadratic objective 0.5 x'Hx the
// same cone over an epigraph variable t with cost 0.5.
#pragma once

#include <string>
#include <vector>

#include "sankhya/model.hpp"
#include "sankhya/qcqp.hpp"
#include "sankhya/sparse.hpp"

namespace sankhya::conic {

struct ConeBlock {
  Index start = 0;
  Index size = 0;
};

/// How one column of the model is read off the form: x_j = shift + a_scale x[a] + b_scale x[b]
/// (b = -1 when unused; a = -1 for a fixed column, which is its shift alone).
struct ColumnMap {
  Index a = -1;
  Index b = -1;
  double a_scale = 1.0;
  double b_scale = 0.0;
  double shift = 0.0;
};

struct ConicForm {
  Index n = 0;
  Index m = 0;
  Index free_n = 0;
  Index nonneg_n = 0;
  std::vector<ConeBlock> cones;
  std::vector<double> c;
  std::vector<double> b;
  SparseMatrix a;
  /// The objective is sigma * (model objective) = c'x + offset, in minimization sense.
  double sigma = 1.0;
  double offset = 0.0;

  std::vector<ColumnMap> col_map;  ///< per model column
  /// Per model row: the form row whose multiplier is the row's, and the sign that takes the
  /// form row's multiplier to the model row's (a row read as -g(x) <= -l flips it). -1 when
  /// the row has no finite side and was dropped.
  std::vector<Index> row_of;
  std::vector<double> row_sign;
  /// Non-empty when the form could not be built; the form is then unusable.
  std::string problem;

  /// True when position j of the form is inside a cone block (not free, not non-negative).
  [[nodiscard]] bool in_cone(Index j) const noexcept { return j >= free_n + nonneg_n; }
};

/// Build the form. `classes` is classify_quadratic_rows(model) and every verdict in it must
/// be kConvex; `objective_convex` says sigma * H passed the convexity test (false with an
/// empty Hessian is fine).
[[nodiscard]] ConicForm build_conic_form(const QcqpModel& model,
                                         const std::vector<QuadraticRowClass>& classes);

/// Q = P' L L' P for a symmetric positive semidefinite Q of order k given by its lower
/// triangle (duplicates summed), by a complete-pivoting Cholesky that stops at the first
/// pivot below tol::kConicCholeskyPivotShare of the largest initial diagonal (Higham,
/// "Analysis of the Cholesky decomposition of a semi-definite matrix", 1990). Returns F = P'L
/// as `rank` dense columns of length k, column-major, so Q = F F'.
struct DenseFactor {
  Index k = 0;
  Index rank = 0;
  std::vector<double> f;  ///< k x rank, column-major
  [[nodiscard]] double at(Index i, Index col) const noexcept {
    return f[static_cast<std::size_t>(col) * static_cast<std::size_t>(k) +
             static_cast<std::size_t>(i)];
  }
};
[[nodiscard]] DenseFactor pivoted_cholesky(Index k, const std::vector<Index>& rows,
                                           const std::vector<Index>& cols,
                                           const std::vector<double>& values);

}  // namespace sankhya::conic
