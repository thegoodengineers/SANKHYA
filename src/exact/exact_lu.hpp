// SPDX-License-Identifier: Apache-2.0
// SANKHYA - sparse LU of an LP basis in exact rational arithmetic (#757).
//
// The exact modules first formed the basis inverse densely by Gauss-Jordan elimination:
// O(m^3) rational operations, so they were capped at 300 rows and still ran past their time
// budget on bases of a few hundred rows. A basis is sparse, and so, in the right pivot order,
// are its factors. This is Gaussian elimination over the sparsity pattern (Markowitz, "The
// elimination form of the inverse and its application to linear programming", Management
// Science 3, 1957), with the product form of the inverse for basis changes (Dantzig and
// Orchard-Hays, "The product form for the inverse in the simplex method", MTAC 8, 1954).
//
// THE PIVOT ORDER IS THE FLOAT LU'S. Exact arithmetic needs no stability threshold - any
// nonzero pivot is exact - so the order only decides fill, and fill is what makes exact
// factors slow: every extra nonzero is a fraction whose numerator and denominator grow with
// each update it receives. The double-precision SparseLu (src/la/lu.hpp; Suhl and Suhl, ORSA
// J. Computing 2, 1990) already chooses a sparse order for this very matrix, so the exact
// factorisation takes the basis in double, factorises it with SparseLu, and replays that
// pivot sequence. The double copy is exact too: every coefficient came from a double, and
// Rational::to_double gives that double back.
//
// Where the float order proposes a pivot that is exactly zero (cancellation the float LU saw
// as a tiny nonzero), the step takes the active row of that column with the fewest entries.
// A column with no active nonzero at all is exact linear dependence: the basis is exactly
// singular, and factorize() says so.
//
// Solves are sparse and push-style: a step whose value is exactly zero costs one test.
#pragma once

#include <utility>
#include <vector>

#include "exact/rational.hpp"
#include "sankhya/types.hpp"

namespace sankhya::exact {

/// A sparse column or vector: (index, nonzero value) pairs, in any order.
using RationalColumn = std::vector<std::pair<Index, Rational>>;

/// The least common denominator D of the entries of `v` (1 when all are zero or integers),
/// and the integers N_i = v_i D. A dot product with v can then run in integers and be
/// reduced once at the end, instead of taking a gcd per term: the exact modules' main cost.
BigInt common_denominator(const std::vector<Rational>& v, std::vector<BigInt>* numerators);

class ExactLu {
 public:
  /// Factorise the m x m matrix whose column p is `*columns[p]` (row indices 0..m-1).
  /// False when it is exactly singular. Checks `deadline` once per elimination step.
  [[nodiscard]] bool factorize(const std::vector<const RationalColumn*>& columns, Index m,
                               const Deadline& deadline);

  /// B x = b, in place: `b` by row in, x by basis position out. Dense, m entries.
  void solve(std::vector<Rational>& b) const;
  /// B^T y = c, in place: `c` by basis position in, y by row out. Dense, m entries.
  void solve_transpose(std::vector<Rational>& c) const;

  /// The basis column at `position` is replaced by the column a whose alpha = B^{-1} a (by
  /// position) is given; alpha[position] must be nonzero. The product form: every later
  /// solve applies this eta. Refactorise when num_updates() grows.
  void update(Index position, const std::vector<Rational>& alpha);
  [[nodiscard]] Index num_updates() const noexcept {
    return static_cast<Index>(eta_position_.size());
  }

  /// Steps whose pivot the float order proposed and exact arithmetic kept, and the nonzeros
  /// of L and U: what the log reports and the tests hold the order to.
  [[nodiscard]] Index hinted_steps() const noexcept { return hinted_steps_; }
  [[nodiscard]] Index factor_nonzeros() const noexcept { return factor_nonzeros_; }

 private:
  Index m_ = 0;
  std::vector<Index> pivot_row_;       ///< step -> row
  std::vector<Index> pivot_position_;  ///< step -> basis position
  std::vector<Rational> pivot_value_;
  /// L: per step, the rows eliminated against the pivot row and their multipliers.
  std::vector<std::vector<std::pair<Index, Rational>>> lower_;
  /// U by column: per step k, the entries U[row of step j][position of step k], j < k, as
  /// (step j, value). Column storage makes the back substitution a push that skips zeros.
  std::vector<std::vector<std::pair<Index, Rational>>> upper_;
  /// Product-form etas, oldest first: the position replaced and the alpha column.
  std::vector<Index> eta_position_;
  std::vector<RationalColumn> eta_;
  Index hinted_steps_ = 0;
  Index factor_nonzeros_ = 0;
};

}  // namespace sankhya::exact
