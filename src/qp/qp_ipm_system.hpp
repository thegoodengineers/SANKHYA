// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the linear algebra side of the proximal interior point for convex QP (#490):
// the model in equality-and-bounds form, and the regularized quasi-definite KKT matrix.
// The iteration itself is src/qp/qp_ipm.cpp; see its header for the method and references.
#pragma once

#include <vector>

#include "sankhya/model.hpp"
#include "sankhya/sparse.hpp"

namespace sankhya::qp::ipm_detail {

/// The model as the interior point sees it, in MINIMIZATION sense:
///
///     min  g'v + 0.5 v'Hv   s.t.  M v = b,   lower <= v <= upper
///
/// v is the kept original columns (fixed ones substituted out) followed by one slack per
/// ranged or one-sided row (a'x - w = 0, w carrying the row's bounds). Rows with no finite
/// side are dropped and get a zero dual.
struct Standard {
  Index n = 0;                   ///< original columns
  Index free_n = 0;              ///< original columns kept (not fixed)
  Index slacks = 0;              ///< slack columns
  Index cols = 0;                ///< free_n + slacks
  Index rows = 0;                ///< rows kept
  std::vector<Index> column_of;  ///< original column -> internal column, or -1 when fixed
  std::vector<Index> row_of;     ///< original row -> internal row, or -1 when free
  std::vector<Index> slack_row;  ///< slack k (column free_n + k) -> its internal row
  std::vector<double> fixed_value;
  SparseMatrix h;  ///< lower triangle of H over the kept original columns
  SparseMatrix m;  ///< rows x cols
  std::vector<double> g, lower, upper, b;
};

[[nodiscard]] Standard standardize(const Model& model);

/// out = H v for the symmetric H whose lower triangle is `h`; out and v are at least
/// h.num_cols() long, and only that prefix of out is written (the rest is zeroed).
void hessian_times(const SparseMatrix& h, const std::vector<double>& v,
                   std::vector<double>* out);

[[nodiscard]] double inf_norm(const std::vector<double>& v);

/// The regularized KKT matrix
///
///     [ -(H + Theta^{-1} + rho I)   M'      ]
///     [  M                          delta I ]
///
/// as a lower triangle whose pattern is fixed at construction: column j < cols holds its
/// diagonal, the strictly-lower Hessian entries and M's column shifted down by cols; column
/// cols + i holds the delta diagonal alone.
class KktMatrix {
 public:
  explicit KktMatrix(const Standard& s);

  /// The values for the given Theta^{-1}, rho and delta, in the fixed pattern.
  [[nodiscard]] SparseMatrix build(const std::vector<double>& theta_inverse, double rho,
                                   double delta) const;

  /// out = K x for the symmetric K whose lower triangle is `k`.
  static void multiply(const SparseMatrix& k, const std::vector<double>& x,
                       std::vector<double>* out);

 private:
  const Standard& s_;
  std::vector<Index> starts_;
  std::vector<Index> rows_;
};

}  // namespace sankhya::qp::ipm_detail
