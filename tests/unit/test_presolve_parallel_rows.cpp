// SPDX-License-Identifier: Apache-2.0
// SANKHYA - parallel rows in presolve (#412).
//
// Merging two rows that say the same thing is easy to get right for the point and easy to
// get wrong for the duals: the removed row's multiplier is not zero when the bound that
// binds was its own, and a postsolve that says otherwise returns a point with a certificate
// that does not price it. So the hand tests check which row carries the dual on both signs
// of the scale, that crossed bounds are left alone, and then random LPs with deliberately
// duplicated rows are solved with the reduction on against the exact rational oracle.

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "oracles/lp_generator.hpp"
#include "oracles/rational_simplex.hpp"
#include "presolve/presolve.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

Options with_parallel_rows(bool on) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", true);
  options.set_bool("presolve_parallel_rows", on);
  return options;
}

Model build(const std::vector<std::vector<double>>& rows, const std::vector<double>& row_lower,
            const std::vector<double>& row_upper, const std::vector<double>& cost,
            const std::vector<double>& col_lower, const std::vector<double>& col_upper) {
  Model model;
  const auto n = static_cast<Index>(cost.size());
  const auto m = static_cast<Index>(rows.size());
  model.col_cost = cost;
  model.col_lower = col_lower;
  model.col_upper = col_upper;
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kContinuous);
  model.row_lower = row_lower;
  model.row_upper = row_upper;
  model.matrix.reset(m, n);
  for (Index i = 0; i < m; ++i) {
    for (Index j = 0; j < n; ++j) {
      const double v = rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
      if (v != 0.0) model.matrix.add_entry(i, j, v);
    }
  }
  model.matrix.finalize();
  model.hessian.reset(n, n);
  model.hessian.finalize();
  EXPECT_EQ(model.validate(), "");
  return model;
}

presolve::Result reduce(const Model& model, bool on) {
  Logger logger(nullptr);
  return presolve::presolve(model, with_parallel_rows(on), logger);
}

TEST(PresolveParallelRows, AScaledCopyMergesAndTheDualGoesWhereTheBoundBinds) {
  //   x0 + x1 >= 1   (row 0)         2 x0 + 2 x1 <= 6   (row 1, twice row 0)
  // Merged: 1 <= x0 + x1 <= 3 on row 0, row 1 removed with scale 2 and the upper bound
  // marked as its own.
  const Model model =
      build({{1, 1}, {2, 2}}, {1.0, -kInf}, {kInf, 6.0}, {1, 1}, {0, 0}, {10, 10});
  const presolve::Result reduced = reduce(model, true);
  EXPECT_EQ(reduced.report.parallel_rows, 1);
  EXPECT_EQ(reduce(model, false).report.parallel_rows, 0) << "off means off";

  // Minimising x0 + x1 binds the lower bound, which is row 0's own: row 1 is slack.
  const Solution low = solve(model, with_parallel_rows(true));
  ASSERT_EQ(low.status, SolveStatus::kOptimal) << low.message;
  EXPECT_NEAR(low.objective, 1.0, 1e-9);
  EXPECT_NEAR(low.row_dual[1], 0.0, 1e-9);
  EXPECT_GT(low.row_dual[0], 0.5);
  EXPECT_LE(low.dual_infeasibility, 1e-7);

  // Minimising -x0 - x1 binds the upper bound, which came from row 1: the dual moves there,
  // scaled by a half, and row 0 is slack.
  const Model flipped =
      build({{1, 1}, {2, 2}}, {1.0, -kInf}, {kInf, 6.0}, {-1, -1}, {0, 0}, {10, 10});
  const Solution high = solve(flipped, with_parallel_rows(true));
  ASSERT_EQ(high.status, SolveStatus::kOptimal) << high.message;
  EXPECT_NEAR(high.objective, -3.0, 1e-9);
  EXPECT_NEAR(high.row_dual[0], 0.0, 1e-9);
  EXPECT_NEAR(high.row_dual[1], -0.5, 1e-9) << "a_1 = 2 a_0, so y_1 = y_0 / 2 = -1 / 2";
  EXPECT_EQ(high.row_status[1], BasisStatus::kAtUpper);
  EXPECT_EQ(high.row_status[0], BasisStatus::kBasic);
  EXPECT_LE(high.dual_infeasibility, 1e-7);
  const Solution off = solve(flipped, with_parallel_rows(false));
  ASSERT_EQ(off.status, SolveStatus::kOptimal) << off.message;
  EXPECT_NEAR(off.objective, high.objective, 1e-9);
}

TEST(PresolveParallelRows, ANegativeScaleSwapsTheBounds) {
  //   x0 + x1 >= 1   (row 0)        -x0 - x1 >= -3   (row 1, minus row 0)
  // Row 1's LOWER bound is, in row 0's units, an UPPER bound of 3. Minimising -x0 - x1
  // binds it, so the dual belongs to row 1 at its lower bound, with the sign the lower
  // bound's convention gives.
  const Model model =
      build({{1, 1}, {-1, -1}}, {1.0, -3.0}, {kInf, kInf}, {-1, -1}, {0, 0}, {10, 10});
  const presolve::Result reduced = reduce(model, true);
  EXPECT_EQ(reduced.report.parallel_rows, 1);
  ASSERT_EQ(reduced.records.size(), 1u);
  EXPECT_TRUE(reduced.records[0].upper_from_removed);
  EXPECT_FALSE(reduced.records[0].lower_from_removed);
  const Solution solved = solve(model, with_parallel_rows(true));
  ASSERT_EQ(solved.status, SolveStatus::kOptimal) << solved.message;
  EXPECT_NEAR(solved.objective, -3.0, 1e-9);
  EXPECT_NEAR(solved.row_dual[0], 0.0, 1e-9);
  EXPECT_NEAR(solved.row_dual[1], 1.0, 1e-9) << "y_1 = y_0 / (-1) with y_0 = -1";
  EXPECT_EQ(solved.row_status[1], BasisStatus::kAtLower);
  EXPECT_LE(solved.dual_infeasibility, 1e-7);
}

TEST(PresolveParallelRows, CrossedBoundsAreLeftForTheEngine) {
  //   x0 + x1 >= 4  and  x0 + x1 <= 3: parallel, and the merge would cross. Presolve
  // leaves both rows, and the engine reports the infeasibility with its own evidence.
  const Model model =
      build({{1, 1}, {1, 1}}, {4.0, -kInf}, {kInf, 3.0}, {1, 1}, {0, 0}, {10, 10});
  EXPECT_EQ(reduce(model, true).report.parallel_rows, 0);
  const Solution solved = solve(model, with_parallel_rows(true));
  EXPECT_EQ(solved.status, SolveStatus::kInfeasible) << solved.message;
}

/// `lp` with `extra` rows appended, each a scaled copy of an existing row with a bound of
/// its own: a tighter lower bound for a positive scale, an upper bound for a negative one.
oracle::GeneratedLp with_duplicates(const oracle::GeneratedLp& lp, std::mt19937_64& rng,
                                    int extra) {
  oracle::GeneratedLp out = lp;
  std::uniform_int_distribution<int> which(0, lp.num_rows - 1);
  std::uniform_int_distribution<int> scale_pick(0, 3);
  const std::int64_t scales[4] = {2, 3, -1, -2};
  for (int e = 0; e < extra; ++e) {
    const int source = which(rng);
    const std::int64_t scale = scales[scale_pick(rng)];
    std::vector<std::int64_t> row = out.a[static_cast<std::size_t>(source)];
    for (std::int64_t& v : row) v *= scale;
    const std::int64_t b = out.b[static_cast<std::size_t>(source)];
    // scale > 0: scale * (a x) >= scale * b + 1, one unit tighter than the source;
    // scale < 0: scale * (a x) >= scale * (b + 3), an upper bound of b + 3 on a x.
    out.a.push_back(row);
    out.b.push_back(scale > 0 ? scale * b + 1 : scale * (b + 3));
    ++out.num_rows;
  }
  return out;
}

TEST(PresolveParallelRows, RandomLpsWithDuplicatedRowsAgreeWithTheExactOracle) {
  std::mt19937_64 rng(4122);
  oracle::GeneratorConfig config;
  int compared = 0;
  Count merged = 0;
  for (int trial = 0; trial < 250; ++trial) {
    const oracle::GeneratedLp base = oracle::random_lp(rng, config);
    const oracle::GeneratedLp lp = with_duplicates(base, rng, 1 + trial % 3);
    const oracle::OracleResult exact = oracle::solve_exact(lp);
    if (exact.status == oracle::OracleStatus::kOverflow ||
        exact.status == oracle::OracleStatus::kIterationLimit) {
      continue;
    }
    const Solution got = solve(oracle::to_model(lp), with_parallel_rows(true));
    merged += got.presolve_report.parallel_rows;
    switch (exact.status) {
      case oracle::OracleStatus::kOptimal: {
        ASSERT_EQ(got.status, SolveStatus::kOptimal)
            << "trial " << trial << ": " << got.message << "\n"
            << lp.to_text();
        const double expected = exact.objective.to_double();
        EXPECT_NEAR(got.objective, expected, 1e-6 * std::max(1.0, std::fabs(expected)))
            << "trial " << trial << "\n"
            << lp.to_text();
        EXPECT_LE(got.dual_infeasibility, 1e-6) << "trial " << trial;
        break;
      }
      case oracle::OracleStatus::kInfeasible:
        EXPECT_EQ(got.status, SolveStatus::kInfeasible) << "trial " << trial;
        break;
      case oracle::OracleStatus::kUnbounded:
        EXPECT_EQ(got.status, SolveStatus::kUnbounded) << "trial " << trial;
        break;
      case oracle::OracleStatus::kOverflow:
      case oracle::OracleStatus::kIterationLimit: break;
    }
    ++compared;
  }
  EXPECT_GT(compared, 200);
  EXPECT_GT(merged, 100) << "the duplicated rows should be found";
}

}  // namespace
}  // namespace sankhya
