// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the model statistics a learned engine selection reads (#477). One pass over the
// column-compressed matrix and the bound vectors, O(m + n + nnz): cheap enough to compute
// before every solve, and computed HERE only, so the training script
// (bench/runners/engine_selection_data.py, via `sankhya info --features`) and any rules it
// emits read the same numbers the solver would.
//
// Algorithm selection from instance features: Rice, "The algorithm selection problem",
// Advances in Computers 15, 1976; Kotthoff, "Algorithm selection for combinatorial search
// problems: a survey", AI Magazine 35(3), 2014. The dense-column test follows the practice
// described by Andersen, Gondzio, Meszaros and Xu, "Implementation of interior point methods
// for large scale linear programming", in Terlaky (ed.), Interior Point Methods of
// Mathematical Programming, Kluwer 1996, sec. 5: a column is dense when it has more than
// tol::kIpmDenseColumnFactor * sqrt(rows) entries, because it fills the normal equations
// A D A^T. It is the test the interior point itself applies (find_dense_columns in
// src/ipm/dense_columns.cpp, at the default ipm_dense_column_factor), so the feature counts
// the columns the solver would treat as dense.
//
// The symbolic Cholesky fill (#477) is the size of the factor the interior point would build,
// from the elimination tree of the normal equations under the approximate minimum degree
// ordering (Davis, "Direct Methods for Sparse Linear Systems", SIAM 2006, ch. 4; Amestoy,
// Davis and Duff, SIAM J. Matrix Anal. Appl. 17, 1996), computed by the same SparseLdl
// analyze() the interior point calls, on A A^T without the dense columns it removes. It is
// NOT part of the one-pass features: it is computed only when asked for (the learned
// selection and `sankhya info --features`), and only under the fixed budgets below, so its
// cost is bounded by work, never by a clock, and the same model always gives the same value.
#pragma once

#include <cstdint>
#include <string>

#include "sankhya/model.hpp"

namespace sankhya {

struct EngineFeatures {
  Index rows = 0;
  Index columns = 0;
  Count nonzeros = 0;
  double density = 0.0;             ///< nnz / (m n), 0 for an empty model
  Index max_column_count = 0;       ///< longest column of A
  Index dense_columns = 0;          ///< more than kIpmDenseColumnFactor sqrt(m) entries
  double rows_per_column = 0.0;     ///< m / n
  double equality_row_share = 0.0;  ///< rows with finite lower == upper, over m
  double integer_share = 0.0;       ///< integer or binary columns over n
  double boxed_column_share = 0.0;  ///< finite lower and upper, lower < upper
  double free_column_share = 0.0;   ///< neither bound finite
  double fixed_column_share = 0.0;  ///< finite lower == upper
  /// Pattern-only upper bound on the nonzeros of the normal-equations matrix A A^T: each
  /// column j contributes an outer product of c_j^2 entries, overlaps counted twice. An
  /// upper bound, not the exact count and not the Cholesky fill; the fill is
  /// cholesky_nonzeros below, from a symbolic pass made only on request.
  double normal_equations_nnz_bound = 0.0;
  /// normal_equations_nnz_bound / nnz: how much larger than A itself the normal equations
  /// can be. Large on models with dense columns, near 1 on network-like models.
  double normal_equations_ratio = 0.0;
  /// Nonzeros of the Cholesky factor L of A A^T (diagonal included), the dense columns left
  /// out as the interior point leaves them out, from a symbolic pass; -1 when not computed.
  /// When the normal equations or the factor pass their budget the count stops there and
  /// `cholesky_capped` is set: the value is then a LOWER bound, the budget itself.
  double cholesky_nonzeros = -1.0;
  /// cholesky_nonzeros / nnz(A): the factor's size against the matrix's. Near 1 on a
  /// network or staircase, hundreds on an expander. -1 when not computed.
  double cholesky_fill_ratio = -1.0;
  bool cholesky_capped = false;
};

/// Budgets of the symbolic pass, in nonzeros. The normal equations are counted first
/// (predict_normal_nonzeros, which stops counting at the cap) and assembled only under
/// kFeatureNormalBudget; the ordering and the factor pattern are stopped by SparseLdl's own
/// budgets. 2e6 normal entries and 1e7 factor entries hold every Netlib model's factor
/// (dfl001, the largest fill there, is under both) at a few tens of MB; past them the
/// feature says "at least this", which is all a choice between engines needs.
inline constexpr std::int64_t kFeatureNormalBudget = 2000000;
inline constexpr std::int64_t kFeatureFactorBudget = 10000000;

/// Compute the features of `model`. The matrix must be finalized. With `symbolic` the
/// Cholesky fill is computed too (the pass described above); without it, the O(m + n + nnz)
/// features only, and the two cholesky_* fields stay -1.
[[nodiscard]] EngineFeatures compute_engine_features(const Model& model, bool symbolic = false);

/// One-line JSON object with every field above, keys equal to the field names. Doubles are
/// printed with 17 significant digits so a reader gets the exact value back.
[[nodiscard]] std::string format_engine_features_json(const EngineFeatures& features);

}  // namespace sankhya
