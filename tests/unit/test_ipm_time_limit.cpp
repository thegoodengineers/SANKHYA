// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the interior point honours its time limit inside the linear algebra (#468).
//
// Measured before this file existed: Linf_520c (93,326 rows) with a 90 s limit returned
// after 153 s on one machine and 320 s on another, having assembled normal equations of 474
// million lower-triangle entries. The assembly asked the deadline every 256 rows, and one
// column that meets every row makes 256 rows 24 million multiply-adds; the assembled
// triplets were then sorted by SparseMatrix::finalize(), and the ordering built its graph
// from every entry, and neither asked at all. Both tests here use the same shape in
// miniature - a column in every row, as the L-infinity objective's `t` is - so the normal
// equations are dense.

#include <chrono>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include "la/ldl.hpp"
#include "sankhya/ipm.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/sparse.hpp"

namespace sankhya {
namespace {

/// min t  s.t.  x_i + t >= 1 + i/m,  0 <= x_i <= 1/2,  t free: m rows, one of them per x_i,
/// and the column of t in every one of them; the optimum is t = 0.5 + (m-1)/m. The value is
/// not what is tested, the shape is: A Theta A^T is dense, m(m+1)/2 lower-triangle entries.
Model l_infinity_shape(Index m) {
  Model model;
  const Index n = m + 1;
  const Index t = m;
  model.sense = ObjSense::kMinimize;
  model.col_cost.assign(static_cast<std::size_t>(n), 0.0);
  model.col_cost[static_cast<std::size_t>(t)] = 1.0;
  model.col_lower.assign(static_cast<std::size_t>(n), 0.0);
  model.col_upper.assign(static_cast<std::size_t>(n), 0.5);
  model.col_lower[static_cast<std::size_t>(t)] = -kInfinity;
  model.col_upper[static_cast<std::size_t>(t)] = kInfinity;
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kContinuous);
  model.row_lower.resize(static_cast<std::size_t>(m));
  model.row_upper.assign(static_cast<std::size_t>(m), kInfinity);
  model.matrix.reset(m, n);
  for (Index i = 0; i < m; ++i) {
    model.row_lower[static_cast<std::size_t>(i)] =
        1.0 + static_cast<double>(i) / static_cast<double>(m);
    model.matrix.add_entry(i, i, 1.0);
    model.matrix.add_entry(i, t, 1.0);
  }
  model.matrix.finalize();
  model.hessian.reset(n, n);
  model.hessian.finalize();
  return model;
}

TEST(InteriorPointTimeLimit, TheAssemblyAsksTheDeadlineInProportionToItsWork) {
  // Deterministic, no clock: the predicate never fires and only counts how often it was
  // asked. The dense column makes every row of the product cost the whole column, m*m
  // multiply-adds in all; a check every 256 rows asked m/256 = 8 times for 4.2 million of
  // them, one look per half a million. The deadline must now be asked at least once per
  // 100,000 units of work (it is asked every 65,536).
  const Index m = 2048;
  const Model model = l_infinity_shape(m);
  std::vector<double> theta(static_cast<std::size_t>(m) + 1, 1.0);
  std::size_t asked = 0;
  SparseMatrix normal;
  ASSERT_TRUE(normal_equations_lower(model.matrix, theta, {}, 1e-8, &normal, [&asked] {
    ++asked;
    return false;
  }));
  const auto work = static_cast<std::size_t>(m) * static_cast<std::size_t>(m);
  EXPECT_GE(asked, work / 100000) << "asked " << asked << " times for " << work << " work";

  // And asking changed nothing: the product is complete, dense, and equal entry for entry to
  // the one built with no deadline at all.
  SparseMatrix plain;
  ASSERT_TRUE(normal_equations_lower(model.matrix, theta, {}, 1e-8, &plain));
  EXPECT_EQ(normal.num_nonzeros(), static_cast<Index>(m) * (m + 1) / 2);
  ASSERT_EQ(plain.num_nonzeros(), normal.num_nonzeros());
  EXPECT_EQ(plain.row_indices(), normal.row_indices());
  EXPECT_EQ(plain.values(), normal.values());
  EXPECT_EQ(plain.column_starts(), normal.column_starts());
}

TEST(InteriorPointTimeLimit, ADenseColumnDoesNotCarryTheSolvePastItsTimeLimit) {
  // The clock this time, on the whole engine: a limit of half a second on a model whose
  // normal equations hold 4.5 million entries and whose factor is dense. The engine must
  // answer time_limit - not a numerical error, not optimal - and within the limit plus a
  // margin that covers returning, freeing what was built and a loaded CI machine.
  constexpr double kLimit = 0.5;
  constexpr double kMargin = 0.75;
  const Model model = l_infinity_shape(3000);
  Options options;
  options.set_bool("log_to_console", false);
  options.set_double("time_limit", kLimit);
  // The set-up share would decline at a fifth of the limit with not_solved (#357); this test
  // is about the limit itself, so the set-up is given all of it.
  options.set_double("ipm_setup_share", 1.0);
  Logger quiet(nullptr);
  const auto started = std::chrono::steady_clock::now();
  const Solution solved = ipm::solve_ipm(model, options, quiet);
  const double wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  EXPECT_EQ(solved.status, SolveStatus::kTimeLimit) << solved.message;
  EXPECT_LE(wall, kLimit + kMargin) << solved.message;
}

}  // namespace
}  // namespace sankhya
