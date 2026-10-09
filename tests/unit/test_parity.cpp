// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the parity heuristic (#841): GF(2) elimination over a model's parity rows.
//
// Every candidate is checked here against the parity rows independently of the elimination,
// on lights-out models whose null space is known: 3x3 has none (one candidate), 4x4 has
// dimension four (sixteen). A system with no 0/1 solution must say so and offer nothing.

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "mip/parity.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::mip {
namespace {

/// Lights out on an n x n grid: x_ij binary presses, y_ij integer in [0, 2], and per cell
/// x_ij + its neighbours - 2 y_ij = 1 (every light toggled an odd number of times).
Model lights_out(int n) {
  Model m;
  const int cells = n * n;
  m.col_cost.assign(static_cast<std::size_t>(2 * cells), 0.0);
  for (int c = 0; c < cells; ++c) m.col_cost[static_cast<std::size_t>(c)] = 1.0;
  m.col_lower.assign(static_cast<std::size_t>(2 * cells), 0.0);
  m.col_upper.assign(static_cast<std::size_t>(2 * cells), 1.0);
  for (int c = 0; c < cells; ++c) m.col_upper[static_cast<std::size_t>(cells + c)] = 2.0;
  m.col_type.assign(static_cast<std::size_t>(2 * cells), VarType::kInteger);
  m.matrix.reset(cells, 2 * cells);
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) {
      const int row = r * n + c;
      const int around[5][2] = {{0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}};
      for (const auto& d : around) {
        const int rr = r + d[0];
        const int cc = c + d[1];
        if (rr >= 0 && rr < n && cc >= 0 && cc < n) m.matrix.add_entry(row, rr * n + cc, 1.0);
      }
      m.matrix.add_entry(row, cells + row, -2.0);
    }
  }
  m.matrix.finalize();
  m.row_lower.assign(static_cast<std::size_t>(cells), 1.0);
  m.row_upper.assign(static_cast<std::size_t>(cells), 1.0);
  m.hessian.reset(2 * cells, 2 * cells);
  m.hessian.finalize();
  return m;
}

/// Each parity row's odd binaries, read from the candidate, sum to the rhs mod 2.
bool satisfies_parity(const Model& m, const ParityFixings& f,
                      const std::vector<std::uint8_t>& bits) {
  std::vector<int> value(static_cast<std::size_t>(m.num_cols()), 0);
  for (std::size_t k = 0; k < f.columns.size(); ++k)
    value[static_cast<std::size_t>(f.columns[k])] = bits[k];
  std::vector<int> sum(static_cast<std::size_t>(m.num_rows()), 0);
  for (Index j = 0; j < m.num_cols(); ++j) {
    const ColumnView col = m.matrix.column(j);
    for (Index k = 0; k < col.size; ++k) {
      if (static_cast<long long>(col.values[k]) % 2 != 0)
        sum[static_cast<std::size_t>(col.rows[k])] += value[static_cast<std::size_t>(j)];
    }
  }
  for (Index i = 0; i < m.num_rows(); ++i) {
    if ((sum[static_cast<std::size_t>(i)] -
         static_cast<int>(m.row_lower[static_cast<std::size_t>(i)])) %
            2 !=
        0)
      return false;
  }
  return true;
}

TEST(Parity, LightsOutThreeByThreeHasOneSolution) {
  const Model m = lights_out(3);
  const ParityFixings f = parity_fixings(m, {}, 16);
  EXPECT_EQ(f.parity_rows, 9);
  EXPECT_EQ(f.columns.size(), 9U);  // the presses; the y columns have even coefficients
  EXPECT_EQ(f.null_space_dimension, 0);
  EXPECT_FALSE(f.inconsistent);
  ASSERT_EQ(f.candidates.size(), 1U);
  EXPECT_TRUE(satisfies_parity(m, f, f.candidates[0]));
}

TEST(Parity, LightsOutFourByFourEnumeratesItsNullSpace) {
  const Model m = lights_out(4);
  const ParityFixings f = parity_fixings(m, {}, 64);
  EXPECT_EQ(f.null_space_dimension, 4);
  ASSERT_EQ(f.candidates.size(), 16U);
  for (std::size_t a = 0; a < f.candidates.size(); ++a) {
    EXPECT_TRUE(satisfies_parity(m, f, f.candidates[a]));
    for (std::size_t b = 0; b < a; ++b) EXPECT_NE(f.candidates[a], f.candidates[b]);
  }
  EXPECT_EQ(parity_fixings(m, {}, 3).candidates.size(), 3U);  // the cap holds
}

TEST(Parity, InconsistentSystemOffersNothing) {
  // x0 + x1 = 1 and x0 + x1 - 2 y = 0 disagree mod 2.
  Model m;
  m.col_cost = {0.0, 0.0, 0.0};
  m.col_lower = {0.0, 0.0, 0.0};
  m.col_upper = {1.0, 1.0, 5.0};
  m.col_type.assign(3, VarType::kInteger);
  m.matrix.reset(2, 3);
  m.matrix.add_entry(0, 0, 1.0);
  m.matrix.add_entry(0, 1, 1.0);
  m.matrix.add_entry(1, 0, 1.0);
  m.matrix.add_entry(1, 1, 1.0);
  m.matrix.add_entry(1, 2, -2.0);
  m.matrix.finalize();
  m.row_lower = {1.0, 0.0};
  m.row_upper = {1.0, 0.0};
  m.hessian.reset(3, 3);
  m.hessian.finalize();
  const ParityFixings f = parity_fixings(m, {}, 16);
  EXPECT_TRUE(f.inconsistent);
  EXPECT_TRUE(f.candidates.empty());
}

TEST(Parity, NonParityRowsAreIgnored) {
  // An inequality, a continuous column and a fractional coefficient: none is a parity row.
  Model m;
  m.col_cost = {0.0, 0.0};
  m.col_lower = {0.0, 0.0};
  m.col_upper = {1.0, 1.0};
  m.col_type = {VarType::kInteger, VarType::kContinuous};
  m.matrix.reset(3, 2);
  m.matrix.add_entry(0, 0, 1.0);  // x0 <= 1
  m.matrix.add_entry(1, 0, 1.0);  // x0 + y = 1, y continuous
  m.matrix.add_entry(1, 1, 1.0);
  m.matrix.add_entry(2, 0, 0.5);  // 0.5 x0 = 0
  m.matrix.finalize();
  m.row_lower = {-1e30, 1.0, 0.0};
  m.row_upper = {1.0, 1.0, 0.0};
  m.hessian.reset(2, 2);
  m.hessian.finalize();
  const ParityFixings f = parity_fixings(m, {}, 16);
  EXPECT_EQ(f.parity_rows, 0);
  EXPECT_TRUE(f.candidates.empty());
}

TEST(Parity, SearchFindsTheLightsOutOptimumWithItOn) {
  // 5x5 has a null space of dimension two: four fixings, the cheapest of which is the
  // optimum (15 presses). Each switch alone; the answer must not depend on it.
  const Model m = lights_out(5);
  for (const char* value : {"off", "on"}) {
    Options options;
    options.set_bool("log_to_console", false);
    options.set_string("mip_heur_parity", value);
    const Solution solved = solve(m, options);
    ASSERT_EQ(solved.status, SolveStatus::kOptimal) << value << ": " << solved.message;
    EXPECT_NEAR(solved.objective, 15.0, 1e-9) << value;
  }
}

}  // namespace
}  // namespace sankhya::mip
