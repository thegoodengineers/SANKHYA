// SPDX-License-Identifier: Apache-2.0
// SANKHYA - quadratically constrained models and the global method that solves them (#514).
//
// WHAT THIS IS. A model whose ROWS may carry products of two columns:
//
//     minimize / maximize   objective_offset + c'x + 0.5 x'Hx
//     subject to            row_lower <= A x + sum_t value_t x[first_t] x[second_t] <=
//     row_upper
//                           col_lower <= x <= col_upper
//
// It is the frozen Model plus a list of quadratic terms, not a change to Model: `linear`
// holds everything Model already expresses (columns, bounds, the linear part of every row,
// the objective with its Hessian) and `quadratic` holds the products the rows add. A pooling
// problem's quality balance, flow times concentration, is the motivating case (#516).
//
// `linear` ALONE IS NOT THIS MODEL. Handing it to sankhya::solve() solves the problem with
// every product dropped, which is a different model with a different optimum. Nothing here
// does that; solve_global() is the entry point, and the CLI reaches it only under
// `--option nonconvex=global`. Without that option a file with quadratic rows is refused by
// the reader, exactly as before.
#pragma once

#include <string>
#include <vector>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {

class SolveControl;

/// One product in one row: `value * x[first] * x[second]`, first <= second. first == second
/// is a square.
struct QuadraticTerm {
  Index row = 0;
  Index first = 0;
  Index second = 0;
  double value = 0.0;
};

class QcqpModel {
 public:
  /// Columns, bounds, integrality, the linear part of every row, the objective.
  Model linear;
  /// The products the rows add to A x. At most one term per (row, first, second).
  std::vector<QuadraticTerm> quadratic;

  [[nodiscard]] bool has_quadratic_rows() const noexcept { return !quadratic.empty(); }

  /// Empty when well formed; otherwise the first problem (linear's own validation, a term
  /// naming a row or column that does not exist, first > second, a repeated or non-finite
  /// term).
  [[nodiscard]] std::string validate() const;

  /// Row activities A x + the quadratic terms, num_rows entries.
  [[nodiscard]] std::vector<double> row_activity(const std::vector<double>& x) const;

  /// The objective at x in the model's own sense, offset and Hessian included.
  [[nodiscard]] double objective(const std::vector<double>& x) const;
};

namespace io {

/// Read an MPS/QPS file whose rows may carry QCMATRIX sections (the CPLEX and Gurobi
/// extension: `QCMATRIX <row>` followed by `col col value` lines listing the FULL symmetric
/// matrix Q_r, with no factor 1/2, so the row is a'x + x'Q_r x). Entry (i, j) and entry
/// (j, i) both contribute to the one product x_i x_j. Everything else is read exactly as
/// read_mps() reads it. A file with no QCMATRIX section yields an empty `quadratic`.
ReadResult read_qcqp_mps(const std::string& path, QcqpModel* model,
                         MpsFormat format = MpsFormat::kAuto);

/// By extension, as read_model(): `.lp` is read by read_lp() (which has no quadratic rows),
/// anything else by read_qcqp_mps().
ReadResult read_qcqp_model(const std::string& path, QcqpModel* model);

}  // namespace io

/// Spatial branch and bound over McCormick relaxations (src/global/spatial_bnb.cpp): a
/// global optimum to the gap `mip_relative_gap` / `mip_absolute_gap`, proved by the
/// relaxations' bounds, or `feasible` with the bound it did prove. Continuous models only;
/// every column in a product needs finite bounds after bound tightening. A model with no
/// products is handed to sankhya::solve() unchanged.
[[nodiscard]] Solution solve_global(const QcqpModel& model, const Options& options,
                                    SolveControl* control = nullptr);

}  // namespace sankhya
