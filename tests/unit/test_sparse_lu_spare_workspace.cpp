// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the LU's elimination workspace kept per thread between SparseLu objects (#1158).
//
// A destroyed SparseLu hands its workspace to the thread's spare slot, and the next
// factorize() on that thread starts from it instead of allocating (src/la/lu.hpp). The claim
// is that this changes no factorization: Workspace::init() sets everything the elimination
// reads. So each matrix is factorized twice:
//
//   1. on a NEW thread, whose spare slot is empty by construction: the factorization as it
//      was before the slot existed;
//   2. on this thread, straight after a SparseLu of a DIFFERENT matrix - larger, smaller,
//      denser - has been destroyed, so the workspace it starts from is a used one of another
//      shape.
//
// The two must agree exactly: the pivot sequence, and FTRAN and BTRAN results to the last
// bit. The matrices have values from a small set (so pivot ties are common, and a stale
// count or flag would break one differently) and a dominant diagonal (so they factorize).

#include <cstdint>
#include <cstring>
#include <iterator>
#include <random>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/tolerances.hpp"
#include "sankhya/types.hpp"

#include "la/lu.hpp"

namespace sankhya {
namespace {

struct Matrix {
  Index m = 0;
  std::vector<std::vector<Index>> rows;
  std::vector<std::vector<double>> values;

  [[nodiscard]] std::vector<LuColumn> columns() const {
    std::vector<LuColumn> out(static_cast<std::size_t>(m));
    for (std::size_t j = 0; j < out.size(); ++j) {
      out[j].size = static_cast<Index>(rows[j].size());
      out[j].rows = rows[j].data();
      out[j].values = values[j].data();
    }
    return out;
  }
};

Matrix random_matrix(Index m, double density, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> coin(0.0, 1.0);
  std::uniform_int_distribution<int> pick(0, 3);
  const double off_diagonal[] = {1.0, -1.0, 2.0, 0.5};
  Matrix a;
  a.m = m;
  a.rows.resize(static_cast<std::size_t>(m));
  a.values.resize(static_cast<std::size_t>(m));
  for (Index j = 0; j < m; ++j) {
    for (Index i = 0; i < m; ++i) {
      if (i == j) {
        a.rows[static_cast<std::size_t>(j)].push_back(i);
        a.values[static_cast<std::size_t>(j)].push_back(8.0 + static_cast<double>(pick(rng)));
      } else if (coin(rng) < density) {
        a.rows[static_cast<std::size_t>(j)].push_back(i);
        a.values[static_cast<std::size_t>(j)].push_back(off_diagonal[pick(rng)]);
      }
    }
  }
  return a;
}

/// Everything compared: the pivot order and one FTRAN and one BTRAN of a fixed vector.
struct Outcome {
  bool ok = false;
  std::vector<Index> pivot_rows;
  std::vector<Index> pivot_columns;
  std::vector<double> ftran;
  std::vector<double> btran;
};

Outcome factorize_and_solve(const Matrix& a) {
  Outcome out;
  SparseLu lu;
  out.ok = lu.factorize(a.columns(), a.m, tol::kPivotTolerance, tol::kMarkowitzThreshold);
  if (!out.ok) return out;
  out.pivot_rows = lu.pivot_rows();
  out.pivot_columns = lu.pivot_columns();
  out.ftran.resize(static_cast<std::size_t>(a.m));
  for (Index i = 0; i < a.m; ++i) {
    out.ftran[static_cast<std::size_t>(i)] = 1.0 + 0.25 * static_cast<double>(i % 7);
  }
  out.btran = out.ftran;
  lu.solve(out.ftran.data());
  lu.solve_transpose(out.btran.data());
  return out;  // lu is destroyed here: its workspace goes to this thread's spare slot
}

bool same_bits(const std::vector<double>& x, const std::vector<double>& y) {
  return x.size() == y.size() &&
         (x.empty() || std::memcmp(x.data(), y.data(), x.size() * sizeof(double)) == 0);
}

TEST(SparseLuSpareWorkspace, AUsedWorkspaceOfAnotherShapeGivesTheSameFactorization) {
  // Shapes chosen to grow, shrink and repeat, sparse and dense, so the spare is always one
  // left by a different matrix.
  const Index sizes[] = {12, 45, 7, 45, 120, 3, 64, 12, 1, 90};
  const double densities[] = {0.6, 0.15, 0.9, 0.5, 0.05, 1.0, 0.3, 0.2, 1.0, 0.1};
  std::uint32_t seed = 20261114;
  for (int round = 0; round < 3; ++round) {
    for (std::size_t k = 0; k < std::size(sizes); ++k) {
      const Matrix a = random_matrix(sizes[k], densities[k], seed++);
      Outcome fresh;
      std::thread([&] { fresh = factorize_and_solve(a); }).join();  // empty spare slot
      const Outcome reused = factorize_and_solve(a);                // the previous one's
      ASSERT_TRUE(fresh.ok) << "size " << sizes[k] << " seed " << seed - 1;
      ASSERT_EQ(reused.ok, fresh.ok);
      EXPECT_EQ(reused.pivot_rows, fresh.pivot_rows) << "size " << sizes[k];
      EXPECT_EQ(reused.pivot_columns, fresh.pivot_columns) << "size " << sizes[k];
      EXPECT_TRUE(same_bits(reused.ftran, fresh.ftran)) << "size " << sizes[k];
      EXPECT_TRUE(same_bits(reused.btran, fresh.btran)) << "size " << sizes[k];
    }
  }
}

TEST(SparseLuSpareWorkspace, AWorkspaceAboveTheRowBoundIsNotKept) {
  // The bound is a memory rule, not a correctness one, so all this can check from outside
  // is that a basis above it still factorizes the same way before and after another.
  const Index big = tol::kLuSpareWorkspaceRows + 8;
  const Matrix large = random_matrix(big, 2.0 / static_cast<double>(big), 7);
  const Matrix little = random_matrix(20, 0.4, 8);
  Outcome fresh_small;
  std::thread([&] { fresh_small = factorize_and_solve(little); }).join();
  const Outcome first_large = factorize_and_solve(large);
  const Outcome after_large = factorize_and_solve(little);
  const Outcome second_large = factorize_and_solve(large);
  ASSERT_TRUE(first_large.ok);
  ASSERT_TRUE(fresh_small.ok);
  EXPECT_EQ(after_large.pivot_rows, fresh_small.pivot_rows);
  EXPECT_TRUE(same_bits(after_large.ftran, fresh_small.ftran));
  EXPECT_EQ(second_large.pivot_rows, first_large.pivot_rows);
  EXPECT_TRUE(same_bits(second_large.ftran, first_large.ftran));
}

}  // namespace
}  // namespace sankhya
