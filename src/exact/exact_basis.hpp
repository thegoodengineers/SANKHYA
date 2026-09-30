// SPDX-License-Identifier: Apache-2.0
// SANKHYA - an LP basis in exact rational arithmetic, shared by the certified sensitivity
// (exact_sensitivity.cpp) and the exact repair (exact_repair.cpp), #757.
//
// The model is taken as [A | -I]: variables 0..n-1 structural, n+i the logical of row i,
// whose value is the row activity and whose bounds are the row's. Costs are in minimise
// space. A Basis holds its exact LU (exact_lu.hpp), the point, the duals y (B^T y = c_B) and
// the reduced costs d = c - [A | -I]^T y, all exact.
#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "exact/exact_lu.hpp"
#include "exact/exact_verify.hpp"
#include "exact/rational.hpp"
#include "sankhya/model.hpp"

namespace sankhya::exact {

/// Product-form updates on one exact factor before it is rebuilt: each eta costs every later
/// solve its nonzeros, and a fresh factor in the float order is usually sparser than B
/// E_1..E_t.
inline constexpr Index kRefactorEvery = 32;

/// Where a variable is: basic, or nonbasic at its lower bound, its upper bound, or free at 0.
enum class Nb : unsigned char { kBasic, kLower, kUpper, kFree };

struct Problem {
  Index n = 0;
  Index m = 0;
  int sense = 1;
  std::vector<Rational> cost, lower, upper;
  std::vector<char> has_lower, has_upper;
  std::vector<RationalColumn> columns;
  /// Each column over its own common denominator: column k is scaled_columns[k] / scale[k]
  /// with integer entries, and scaled_rows holds the same integers row-wise as (variable,
  /// entry). Dot products run in integers and are reduced once (common_denominator).
  std::vector<BigInt> scale;
  std::vector<std::vector<std::pair<Index, BigInt>>> scaled_columns;
  std::vector<std::vector<std::pair<Index, BigInt>>> scaled_rows;

  explicit Problem(const Model& model);
  /// Equal finite bounds: the variable never moves, so its reduced cost may have any sign.
  [[nodiscard]] bool fixed(Index k) const {
    const auto u = static_cast<std::size_t>(k);
    return has_lower[u] && has_upper[u] && lower[u] == upper[u];
  }
};

struct Basis {
  std::vector<Index> basic;  ///< position -> variable
  std::vector<Nb> status;    ///< per variable
  ExactLu lu;
  std::vector<Rational> value, y, d;

  /// Factorise from scratch and compute the point, y and d. False when exactly singular.
  bool build(const Problem& problem, const Deadline& deadline);

  /// B^{-1} e_row, by position.
  [[nodiscard]] std::vector<Rational> column_of_inverse(const Problem& problem,
                                                        Index row) const;
  /// B^{-1} a_k for variable k, by position.
  [[nodiscard]] std::vector<Rational> tableau_column(const Problem& problem, Index k) const;
  /// Row p of B^{-1} [A | -I] over the nonbasic variables: (variable, nonzero alpha), by
  /// variable. With `row_of_inverse`, e_p^T B^{-1} (by row) is left there too.
  [[nodiscard]] std::vector<std::pair<std::size_t, Rational>> tableau_row(
      const Problem& problem, std::size_t p,
      std::vector<Rational>* row_of_inverse = nullptr) const;

  /// Variable `entering` replaces the basic variable at `position`, which leaves to the
  /// nonbasic status `to`, and the nonbasic variables' statuses are as `status` then says:
  /// the factor takes an eta (or is rebuilt every kRefactorEvery) and the point, y and d are
  /// recomputed. `alpha` is tableau_column(entering). False when singular.
  bool pivot(const Problem& problem, std::size_t position, Index entering, Nb to,
             const std::vector<Rational>& alpha, const Deadline& deadline);

  /// A DEGENERATE pivot: the basic variable at `position` sits exactly on the bound `to` it
  /// leaves to, so the primal step is zero and every value stays as it is. The duals and
  /// reduced costs move by the standard dual update (theta = d_q / alpha_pq):
  ///   d_k -= theta alpha_pk,  d_leaving = -theta,  d_q = 0,  y += theta e_p^T B^{-1},
  /// from the tableau row the ratio test already computed. False when singular.
  bool pivot_degenerate(const Problem& problem, std::size_t position, Index entering, Nb to,
                        const std::vector<std::pair<std::size_t, Rational>>& row,
                        const std::vector<Rational>& row_of_inverse, const Deadline& deadline);

  /// The point, y and d of the current factor and statuses: after a status change that
  /// moves no variable into or out of the basis (a nonbasic variable to its other bound).
  void refresh(const Problem& problem);
};

/// Why the basis is not exactly optimal, or empty: a basic variable outside its bounds, or a
/// nonbasic one that is not fixed with a reduced cost of the wrong sign.
[[nodiscard]] std::string not_optimal(const Problem& problem, const Basis& basis);

/// The reported basis of `solution` (statuses resolved as resolved_status does), built. On
/// failure `message` says why and `verdict` whether that is a decline (no basis to read) or
/// a failure (a basis that is not one).
struct LoadResult {
  ExactVerdict verdict = ExactVerdict::kDeclined;
  std::string message;
};
[[nodiscard]] LoadResult load_basis(const Model& model, const Solution& solution,
                                    const Problem& problem, const Deadline& deadline,
                                    Basis* basis);

}  // namespace sankhya::exact
