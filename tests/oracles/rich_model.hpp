// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a general-form model in exact rationals, for the oracle fuzz. TESTS ONLY.
//
// The oracle's own form is narrow on purpose (rational_simplex.hpp):
//
//     min c'x   subject to   A x >= b,   0 <= x <= u,   integer data
//
// and every sweep built directly on it hands the float solver that same narrow form. The
// two wrong answers of #1312 and #1319 sat on the default path for weeks behind it: neither
// needs anything exotic, but both need a model the narrow form does not write (a continuous
// column beside a binary under a coefficient of 1e7; a row-implied bound on an integer
// column far below one).
//
// RichModel is the general form a user writes: rows with either or both sides, columns with
// any bounds or none, integer columns with fractional bounds, fractional coefficients, a
// constant in the objective, either sense. It has two readings that must agree:
//
//   to_model()     the sankhya::Model the float solver is handed
//   solve_exact()  the same instance rewritten onto the oracle's form and solved exactly
//
// The rewrite is the textbook reduction to standard form (Chvatal, "Linear Programming",
// 1983, ch. 8): a column with a lower bound is shifted to zero, one with only an upper bound
// is reflected, a free one is split into a difference of two, a two-sided row becomes two
// rows, and each row is multiplied by the common denominator of its entries. The exact
// point is mapped back and checked against the RichModel itself (exact_violation), so a
// mistake in the rewrite shows up as a failure of the harness, not as agreement.
#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "sankhya/model.hpp"

#include "oracles/rational.hpp"
#include "oracles/rational_simplex.hpp"

namespace sankhya::oracle {

/// A bound, or none.
using Bound = std::optional<Rational>;

struct RichModel {
  Index num_rows = 0;
  Index num_cols = 0;
  /// Row-major, num_rows x num_cols.
  std::vector<std::vector<Rational>> a;
  std::vector<Bound> row_lower;
  std::vector<Bound> row_upper;
  std::vector<Bound> col_lower;
  std::vector<Bound> col_upper;
  std::vector<char> integral;
  std::vector<Rational> cost;
  Rational offset;
  bool maximize = false;

  [[nodiscard]] bool has_integers() const;
  /// The whole instance in exact fractions, one row per line: what a mismatch prints.
  [[nodiscard]] std::string to_text() const;
};

/// numerator / denominator. Spelled out because Rational's integer type is a class under
/// MSVC, where an int does not convert to a Rational implicitly.
[[nodiscard]] Rational rat(std::int64_t numerator, std::int64_t denominator = 1);

/// The model the float solver sees. Exact when every value is a dyadic fraction; otherwise
/// each value is the nearest double.
[[nodiscard]] Model to_model(const RichModel& rich);

struct RichExact {
  OracleStatus status = OracleStatus::kOptimal;
  /// In the model's own sense, constant included.
  Rational objective;
  /// One value per model column.
  std::vector<Rational> x;
};

/// Solve exactly. A mixed-integer model whose relaxation is unbounded is unbounded only if
/// it has a feasible point (Meyer 1974, for rational data), so that case is settled by a
/// second exact search with a zero objective. kOverflow and kIterationLimit mean the oracle
/// abstained.
[[nodiscard]] RichExact solve_exact(const RichModel& rich, std::int64_t node_limit = 20000);

/// Empty when `x` satisfies every row, bound and integrality requirement of `rich` exactly
/// and `objective` is its objective value; otherwise what fails.
[[nodiscard]] std::string exact_violation(const RichModel& rich, const std::vector<Rational>& x,
                                          const Rational& objective);

/// Greedy reduction (Zeller & Hildebrandt 2002, delta debugging, one element at a time):
/// rows, columns, entries, bounds, integrality and objective terms are removed while `fails`
/// still holds. Returns the smallest model reached.
[[nodiscard]] RichModel shrink(RichModel rich,
                               const std::function<bool(const RichModel&)>& fails);

/// Shape of a generated general-form model. Probabilities are per element.
struct RichConfig {
  Index min_rows = 2;
  Index max_rows = 5;
  Index min_cols = 2;
  Index max_cols = 6;
  double density = 0.6;
  /// Coefficients and costs are integers in [-magnitude, magnitude] before `dyadic` and
  /// `decimal` divide them.
  std::int64_t magnitude = 6;
  /// Share of coefficients divided by 2, 4 or 8: fractional, and exact in binary.
  double dyadic = 0.0;
  /// Share divided by 10 or 3: the model the float solver sees is then 1 ulp off.
  double decimal = 0.0;
  double integer_columns = 0.0;
  double binary_columns = 0.0;
  /// Row kinds; what is left over is a >= row.
  double equality_rows = 0.2;
  double ranged_rows = 0.15;
  double upper_rows = 0.3;
  /// Column kinds; what is left over is a box.
  double free_columns = 0.1;
  double lower_only_columns = 0.15;
  double upper_only_columns = 0.1;
  /// A box is [l, l + w] with l in [-shift, shift] and w in [0, span]; w = 0 is a fixed column.
  std::int64_t shift = 4;
  std::int64_t span = 6;
  /// Share of finite column bounds moved off the integers by 1/2, 1/4 or 1/3.
  double fractional_bounds = 0.0;
  /// Share of models whose rows are built around a point inside the bounds, so that they
  /// are feasible; the rest draw their right-hand sides blindly and are mostly infeasible.
  double planted = 0.8;
  /// Share of rows (columns) that are a multiple of an earlier row (column).
  double dependent_rows = 0.0;
  double dependent_columns = 0.0;
  /// Fixed-charge links x - M y <= l appended, y a new binary (see add_big_m_link).
  int big_m_links = 0;
  std::int64_t big_m_min = 1000;
  std::int64_t big_m_max = 10'000'000;
};

[[nodiscard]] RichModel random_rich(std::mt19937_64& rng, const RichConfig& config);

}  // namespace sankhya::oracle
