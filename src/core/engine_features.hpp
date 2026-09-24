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
#pragma once

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
  /// upper bound, not the exact count and not the Cholesky fill, both of which need a
  /// symbolic pass this function deliberately does not make.
  double normal_equations_nnz_bound = 0.0;
  /// normal_equations_nnz_bound / nnz: how much larger than A itself the normal equations
  /// can be. Large on models with dense columns, near 1 on network-like models.
  double normal_equations_ratio = 0.0;
};

/// Compute the features of `model`. The matrix must be finalized.
[[nodiscard]] EngineFeatures compute_engine_features(const Model& model);

/// One-line JSON object with every field above, keys equal to the field names. Doubles are
/// printed with 17 significant digits so a reader gets the exact value back.
[[nodiscard]] std::string format_engine_features_json(const EngineFeatures& features);

}  // namespace sankhya
