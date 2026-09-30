// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the sparse exact LU (#757).
//
// Every solve is checked by multiplying back in exact arithmetic: B x must equal b and B^T y
// must equal c to the last bit, which no ordering, permutation or eta bug can pass by luck.

#include <cstddef>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "exact/bigint.hpp"
#include "exact/exact_lu.hpp"
#include "exact/rational.hpp"

namespace sankhya::exact {
namespace {

using Sz = std::size_t;

/// A random sparse m x m matrix with a nonzero diagonal, so it is (almost surely)
/// nonsingular, with decimal-looking doubles whose exact values have long mantissas.
std::vector<RationalColumn> random_matrix(Index m, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> value(-5.0, 5.0);
  std::uniform_int_distribution<Index> row(0, m - 1);
  std::vector<RationalColumn> columns(static_cast<Sz>(m));
  for (Index p = 0; p < m; ++p) {
    std::vector<char> used(static_cast<Sz>(m), 0);
    auto add = [&](Index i) {
      if (used[static_cast<Sz>(i)]) return;
      used[static_cast<Sz>(i)] = 1;
      const double v = static_cast<double>(static_cast<int>(value(rng) * 100)) / 10.0 + 0.3;
      columns[static_cast<Sz>(p)].emplace_back(i, Rational::from_double(v));
    };
    add(p);
    for (int k = 0; k < 2; ++k) add(row(rng));
  }
  return columns;
}

std::vector<const RationalColumn*> pointers(const std::vector<RationalColumn>& columns) {
  std::vector<const RationalColumn*> out;
  for (const auto& column : columns) out.push_back(&column);
  return out;
}

/// B x, by row.
std::vector<Rational> times(const std::vector<RationalColumn>& columns,
                            const std::vector<Rational>& x) {
  std::vector<Rational> out(columns.size(), Rational(0));
  for (Sz p = 0; p < columns.size(); ++p) {
    for (const auto& [i, a] : columns[p]) out[static_cast<Sz>(i)] += a * x[p];
  }
  return out;
}

/// B^T y, by position.
std::vector<Rational> times_transpose(const std::vector<RationalColumn>& columns,
                                      const std::vector<Rational>& y) {
  std::vector<Rational> out(columns.size(), Rational(0));
  for (Sz p = 0; p < columns.size(); ++p) {
    for (const auto& [i, a] : columns[p]) out[p] += a * y[static_cast<Sz>(i)];
  }
  return out;
}

std::vector<Rational> random_vector(Index m, std::mt19937_64& rng) {
  std::uniform_int_distribution<int> value(-9, 9);
  std::vector<Rational> v(static_cast<Sz>(m), Rational(0));
  for (auto& e : v) e = Rational(value(rng)) / Rational(7);
  return v;
}

TEST(ExactLu, SolvesAreExactOnRandomSparseMatrices) {
  std::mt19937_64 rng(757);
  const Deadline deadline(60.0);
  for (int trial = 0; trial < 8; ++trial) {
    const Index m = 5 + 7 * trial;
    const std::vector<RationalColumn> columns = random_matrix(m, rng);
    ExactLu lu;
    ASSERT_TRUE(lu.factorize(pointers(columns), m, deadline)) << "trial " << trial;
    // The float LU accepted this matrix, so every step's pivot came from its order.
    EXPECT_EQ(lu.hinted_steps(), m) << "trial " << trial;
    const std::vector<Rational> b = random_vector(m, rng);
    std::vector<Rational> x = b;
    lu.solve(x);
    EXPECT_EQ(times(columns, x), b) << "B x = b, trial " << trial;
    const std::vector<Rational> c = random_vector(m, rng);
    std::vector<Rational> y = c;
    lu.solve_transpose(y);
    EXPECT_EQ(times_transpose(columns, y), c) << "B^T y = c, trial " << trial;
  }
}

TEST(ExactLu, EtaUpdatesSolveTheChangedBasisExactly) {
  std::mt19937_64 rng(521);
  const Deadline deadline(60.0);
  const Index m = 30;
  std::vector<RationalColumn> columns = random_matrix(m, rng);
  ExactLu lu;
  ASSERT_TRUE(lu.factorize(pointers(columns), m, deadline));
  const std::vector<RationalColumn> extra = random_matrix(m, rng);
  for (Index step = 0; step < 6; ++step) {
    // Replace a position by a fresh column a, alpha = B^{-1} a: the first position from
    // 5*step on where alpha is nonzero, so the new basis is nonsingular.
    const RationalColumn& a = extra[static_cast<Sz>(step)];
    std::vector<Rational> alpha(static_cast<Sz>(m), Rational(0));
    for (const auto& [i, v] : a) alpha[static_cast<Sz>(i)] = v;
    lu.solve(alpha);
    auto position = static_cast<Sz>(5 * step);
    while (position < static_cast<Sz>(m) && alpha[position].is_zero()) ++position;
    ASSERT_LT(position, static_cast<Sz>(m));
    lu.update(static_cast<Index>(position), alpha);
    columns[position] = a;
    const std::vector<Rational> b = random_vector(m, rng);
    std::vector<Rational> x = b;
    lu.solve(x);
    EXPECT_EQ(times(columns, x), b) << "after " << step + 1 << " update(s)";
    const std::vector<Rational> c = random_vector(m, rng);
    std::vector<Rational> y = c;
    lu.solve_transpose(y);
    EXPECT_EQ(times_transpose(columns, y), c) << "after " << step + 1 << " update(s)";
  }
  EXPECT_EQ(lu.num_updates(), 6);
}

TEST(ExactLu, AnExactlySingularMatrixIsRefused) {
  // 0.2 is exactly twice 0.1 in binary, and 0.6 exactly twice 0.3, so the second column is
  // exactly twice the first: exactly singular, not merely close.
  const std::vector<RationalColumn> columns = {
      {{0, Rational::from_double(0.1)}, {1, Rational::from_double(0.3)}},
      {{0, Rational::from_double(0.2)}, {1, Rational::from_double(0.6)}},
  };
  ASSERT_EQ(Rational::from_double(0.6), Rational(2) * Rational::from_double(0.3));
  ExactLu lu;
  EXPECT_FALSE(lu.factorize(pointers(columns), 2, Deadline(10.0)));
}

TEST(ExactLu, ANearlySingularMatrixTheFloatOrderRefusesIsStillFactorised) {
  // The second column differs from a multiple of the first by one unit in the last place:
  // below the float LU's pivot tolerance, so it declares the matrix singular and offers no
  // order, but exactly it is nonsingular and the exact LU must still solve it.
  const double tiny_off = 0.6 + 1.1102230246251565e-16;
  ASSERT_NE(tiny_off, 0.6);
  const std::vector<RationalColumn> columns = {
      {{0, Rational::from_double(0.1)}, {1, Rational::from_double(0.3)}},
      {{0, Rational::from_double(0.2)}, {1, Rational::from_double(tiny_off)}},
  };
  ExactLu lu;
  ASSERT_TRUE(lu.factorize(pointers(columns), 2, Deadline(10.0)));
  EXPECT_EQ(lu.hinted_steps(), 0);
  const std::vector<Rational> b = {Rational(1), Rational(-3)};
  std::vector<Rational> x = b;
  lu.solve(x);
  EXPECT_EQ(times(columns, x), b);
}

}  // namespace
}  // namespace sankhya::exact
