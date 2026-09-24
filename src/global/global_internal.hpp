// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the pieces of the spatial branch and bound (#514), shared between its files.
//
// Not a public header. The public entry point is solve_global() in sankhya/qcqp.hpp.
//
//   qcqp_model.cpp     QcqpModel's own checks, and the Problem below built from it
//   relaxation.cpp     the McCormick relaxation of a box, and the bound its LP duals prove
//   fbbt.cpp           feasibility-based bound tightening over a box
//   local_search.cpp   feasibility against the ORIGINAL model, and the alternating LP
//   spatial_bnb.cpp    the tree
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/qcqp.hpp"

namespace sankhya::global {

/// One distinct product x_a * x_b, a <= b, wherever it appears (rows or objective). a == b
/// is a square.
struct Product {
  Index a = 0;
  Index b = 0;
  [[nodiscard]] bool square() const noexcept { return a == b; }
};

/// The model in MINIMIZATION form, row-wise, every product named once. The objective is
/// sigma * (offset + c'x + 0.5 x'Hx) with sigma = +1 to minimize and -1 to maximize; every
/// bound the search proves is on this, and is multiplied by sigma again on the way out.
struct Problem {
  const QcqpModel* source = nullptr;
  Index n = 0;
  Index m = 0;
  double sigma = 1.0;
  double offset = 0.0;
  std::vector<double> cost;
  std::vector<double> col_lower;
  std::vector<double> col_upper;
  std::vector<double> row_lower;
  std::vector<double> row_upper;
  /// Row i's linear part and its products, as (column, coefficient) and (product, coeff).
  std::vector<std::vector<std::pair<Index, double>>> row_linear;
  std::vector<std::vector<std::pair<Index, double>>> row_products;
  std::vector<Product> products;
  /// The objective's product coefficients (already times sigma): (product, coefficient).
  std::vector<std::pair<Index, double>> objective_products;
  /// Columns that appear in at least one product.
  std::vector<char> in_product;
};

[[nodiscard]] Problem build_problem(const QcqpModel& model);

/// Minimization-form objective at x (sigma times the model's own objective).
[[nodiscard]] double min_objective(const Problem& problem, const std::vector<double>& x);

/// A box of column bounds.
struct Box {
  std::vector<double> lower;
  std::vector<double> upper;
};

// ---- relaxation.cpp ---------------------------------------------------------------------

/// The McCormick relaxation of `problem` over `box` as an LP: columns x (n) then one w per
/// product; the original rows with every product replaced by its w; four envelope rows per
/// bilinear product and a secant plus three tangents per square. Every column in a product
/// must have finite bounds in `box`.
///
/// McCormick, "Computability of global solutions to factorable nonconvex programs: Part I -
/// Convex underestimating problems", Math. Programming 10 (1976) 147-175.
[[nodiscard]] Model build_relaxation(const Problem& problem, const Box& box);

/// A lower bound on the relaxation's minimum from ANY row multipliers y, by weak duality:
/// sum_i y_i * (row_lower_i if y_i > 0, row_upper_i if y_i < 0) + sum_j min over the box of
/// (c - A'y)_j z_j. Valid whatever y is, so it does not rest on the LP being solved to
/// optimality (Neumaier and Shcherbina, "Safe bounds in linear and mixed-integer linear
/// programming", Math. Programming 99 (2004) 283-296). -infinity when y would need an
/// infinite bound.
[[nodiscard]] double dual_bound_from_multipliers(const Model& lp, const std::vector<double>& y);

// ---- fbbt.cpp ---------------------------------------------------------------------------

struct FbbtOutcome {
  bool infeasible = false;
  Count tightened = 0;
};

/// Feasibility-based bound tightening: interval arithmetic over every row (and over the
/// objective against `cutoff` when it is finite), forward to the row's range and backward to
/// each column, repeated until nothing moves. Belotti, Lee, Liberti, Margot and Wachter,
/// "Branching and bounds tightening techniques for non-convex MINLP", Optimization Methods
/// and Software 24 (2009) 597-634, sec. 3.
FbbtOutcome fbbt(const Problem& problem, double cutoff, double feasibility_tolerance, Box* box);

// ---- local_search.cpp -------------------------------------------------------------------

/// Worst violation of the ORIGINAL model at x: every row with its products and every
/// column bound, each divided by the scale of the terms it was computed from (the
/// convention of tools/verify_solution.py). Also returns the absolute worst when asked.
[[nodiscard]] double original_violation(const Problem& problem, const std::vector<double>& x,
                                        double* absolute = nullptr);

struct LocalSearchResult {
  bool improved = false;
  std::vector<double> x;
  double objective = kInfinity;  ///< minimization form
  Count lp_iterations = 0;
  Count lp_solves = 0;
};

/// Alternating LP from `start`: fix one side of every product at its current value, solve
/// the LP that leaves, fix the other side at the new point, and repeat while the objective
/// improves. Every point is checked against the original model before it is accepted.
/// `incumbent` is the objective to beat.
[[nodiscard]] LocalSearchResult alternating_search(const Problem& problem,
                                                   const std::vector<double>& start,
                                                   double incumbent, const Options& lp_options,
                                                   double feasibility_tolerance);

/// The upper-bound heuristic run at a node: the alternating LP started from the node's
/// relaxation point (x then one w per product), once from each side with that side's
/// columns set to the values that make the relaxation's products exact, and once from the
/// relaxation's own x.
[[nodiscard]] LocalSearchResult relaxation_search(const Problem& problem,
                                                  const std::vector<double>& relaxation,
                                                  double incumbent, const Options& lp_options,
                                                  double feasibility_tolerance);

/// One LP of the search: the dual simplex called directly (no presolve, no engine
/// selection), started from `warm` when it is given. An `infeasible` verdict is kept only
/// with a Farkas certificate that verify_and_keep_certificate() accepted against `lp`; one
/// without a proof comes back as kNumericalError, so the caller does not prune on it.
[[nodiscard]] Solution solve_lp(const Model& lp, const Options& lp_options,
                                const std::vector<BasisStatus>* col_status = nullptr,
                                const std::vector<BasisStatus>* row_status = nullptr);

/// Options for an inner LP solve: defaults, quiet, dual simplex, the caller's feasibility
/// tolerances and whatever time is left. Nothing else from the caller's options leaks in (a
/// progress file or a presolved-model dump must not be written once per node).
[[nodiscard]] Options inner_lp_options(const Options& options, double seconds_left);

}  // namespace sankhya::global
