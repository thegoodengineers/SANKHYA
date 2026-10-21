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
// `--option nonconvex=global`. Without that option a quadratic row goes to the conic interior
// point (#755) when it is CONVEX - classify_quadratic_rows() decides, by the same LDL^T test
// the objective uses - and is refused with the test's certificate when it is not.
//
// CONES (#755). A model may also carry explicit second-order cones over its columns, the
// MOSEK MPS extension's CSECTION: `cones` lists them, each a quadratic cone x_0 >= ||x_1..||
// or a rotated one 2 x_0 x_1 >= ||x_2..||^2, x_0, x_1 >= 0. A column is in at most one cone.
// solve_conic() is the entry point for a model with cones or with convex quadratic rows.
#pragma once

#include <cstdint>
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

/// A second-order cone over some of the model's columns (#755), in the MOSEK CSECTION sense.
struct Cone {
  enum class Kind : std::uint8_t {
    kQuadratic,  ///< x[columns[0]] >= || x[columns[1..]] ||
    kRotated,    ///< 2 x[columns[0]] x[columns[1]] >= || x[columns[2..]] ||^2, both >= 0
  };
  Kind kind = Kind::kQuadratic;
  std::string name;
  std::vector<Index> columns;  ///< at least 2 for a quadratic cone, 3 for a rotated one
};

class QcqpModel {
 public:
  /// Columns, bounds, integrality, the linear part of every row, the objective.
  Model linear;
  /// The products the rows add to A x. At most one term per (row, first, second).
  std::vector<QuadraticTerm> quadratic;
  /// Explicit second-order cones over the columns (#755). Empty for every model read
  /// before cones existed, so nothing that ignores it sees a different model.
  std::vector<Cone> cones;

  [[nodiscard]] bool has_quadratic_rows() const noexcept { return !quadratic.empty(); }
  [[nodiscard]] bool has_cones() const noexcept { return !cones.empty(); }

  /// Empty when well formed; otherwise the first problem (linear's own validation, a term
  /// naming a row or column that does not exist, first > second, a repeated or non-finite
  /// term, a cone too short or naming a column twice or one another cone already has).
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

/// What a quadratic row is, decided at read time (#755) by the LDL^T semidefiniteness test of
/// src/qp/convexity.hpp on the row's own matrix Q_r.
struct QuadraticRowClass {
  enum class Verdict : std::uint8_t {
    kConvex,      ///< Q_r PSD with only an upper side, or NSD with only a lower side
    kNonConvex,   ///< Q_r indefinite (`witness` says along which direction), or a definite
                  ///< Q_r with a finite side on the wrong end (a lower bound on a convex
                  ///< function), or both sides finite
    kUnverified,  ///< the test could not decide (a deadline); treated as a refusal
  };
  Index row = 0;
  Verdict verdict = Verdict::kUnverified;
  /// kConvex: +1 when the row reads a'x + x'Qx <= u with Q PSD, -1 when it reads
  /// l <= a'x + x'Qx with Q NSD (and -Q is what the cone sees).
  int sign = 0;
  std::string detail;
  /// For an indefinite Q_r, a direction z with z'Q_r z < 0 over the model's columns.
  std::vector<double> witness;
};

/// One entry per quadratic row, in increasing row order.
[[nodiscard]] std::vector<QuadraticRowClass> classify_quadratic_rows(const QcqpModel& model);

/// The conic interior point (src/conic): every quadratic row must be convex and every column
/// continuous, else kNotSolved with the reason. The model's convex quadratic rows and its
/// quadratic objective become rotated cones, the cones stay cones, and the whole is solved by
/// a primal-dual interior point with Nesterov-Todd scaling on the homogeneous self-dual
/// embedding. Optimal with the multipliers of the original rows (a quadratic row's dual is the
/// multiplier of its gradient a_r + 2 Q_r x in the Lagrangian, LP sign convention), reduced
/// costs c + Hx - sum_i y_i grad_i, and activities with their products.
[[nodiscard]] Solution solve_conic(const QcqpModel& model, const Options& options,
                                   SolveControl* control = nullptr);

/// Spatial branch and bound over McCormick relaxations (src/global/spatial_bnb.cpp): a
/// global optimum to the gap `mip_relative_gap` / `mip_absolute_gap`, proved by the
/// relaxations' bounds, or `feasible` with the bound it did prove. Continuous models only;
/// every column in a product needs finite bounds after bound tightening. A model with no
/// products is handed to sankhya::solve() unchanged.
[[nodiscard]] Solution solve_global(const QcqpModel& model, const Options& options,
                                    SolveControl* control = nullptr);

}  // namespace sankhya
