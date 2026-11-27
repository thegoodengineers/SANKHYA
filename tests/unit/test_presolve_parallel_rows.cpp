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

TEST(PresolveParallelRows, AMergeThatMakesAnEqualityPricesTheRowWhoseBoundBinds) {
  //   x0 + x1 + x2 <= 4   (row 0, kept)      x0 + x1 + x2 == 3   (row 1, removed)
  // The merge leaves row 0 as an equality at 3, whose status (fixed) does not say which
  // bound binds. Minimising -x0 - x1 - x2 prices it at -1: the upper bound, row 1's.
  // Leaving that price on row 0, slack by 1 in the original model, broke complementary
  // slackness; on fffff800 it was 1.3e+05 of dual infeasibility. Three columns, so the
  // equality is not a doubleton for that reduction to take first.
  const Model model = build({{1, 1, 1}, {1, 1, 1}}, {-kInf, 3.0}, {4.0, 3.0}, {-1, -1, -1},
                            {0, 0, 0}, {10, 10, 10});
  EXPECT_EQ(reduce(model, true).report.parallel_rows, 1);
  const Solution on = solve(model, with_parallel_rows(true));
  ASSERT_EQ(on.status, SolveStatus::kOptimal) << on.message;
  EXPECT_NEAR(on.objective, -3.0, 1e-9);
  EXPECT_NEAR(on.row_dual[0], 0.0, 1e-9);
  EXPECT_NEAR(on.row_dual[1], -1.0, 1e-9);
  EXPECT_EQ(on.row_status[0], BasisStatus::kBasic);
  EXPECT_LE(on.complementarity_violation, 1e-9);
  EXPECT_LE(on.dual_infeasibility, 1e-7);
}

TEST(PresolveParallelRows, AKeptRowRemovedLaterStillHandsItsPriceBack) {
  // #1354. 3 x0 - 4 x1 >= -3 (row 0, kept) and 6 x0 - 8 x1 <= -2 (row 1, twice row 0): the
  // merged row is -3 <= 3 x0 - 4 x1 <= -1, and x1, free and now in one row, is substituted
  // away with it. min -4 x0 + 2 x1 with x0 <= 3 ends at (3, 5/2), the row at -1: row 1's
  // bound, so the price is row 1's, -1/2 / 2 = -1/4, and row 0, a >= row, has none. The
  // replay of the merge ran before the kept row's dual existed and left -1/2 on row 0.
  // Row 2, x2 + x3 >= 1 at costs 1 and 2, is there so that the reduced model is not empty.
  //
  // Postsolve is called directly: through solve() an answer whose duals fail is solved
  // again without presolve (#783), which returns the right duals whatever postsolve did.
  const Model model =
      build({{3, -4, 0, 0}, {6, -8, 0, 0}, {0, 0, 1, 1}}, {-3.0, -kInf, 1.0},
            {kInf, -2.0, kInf}, {-4, 2, 1, 2}, {-kInf, -kInf, 0, 0}, {3, kInf, 10, 10});
  const presolve::Result reduced = reduce(model, true);
  ASSERT_EQ(reduced.report.parallel_rows, 1);
  ASSERT_GT(reduced.model.num_rows(), 0);
  Options inner;
  inner.set_bool("log_to_console", false);
  inner.set_bool("presolve", false);
  const Solution solved = solve(reduced.model, inner);
  ASSERT_EQ(solved.status, SolveStatus::kOptimal) << solved.message;
  const Solution restored = presolve::postsolve(reduced, model, solved);
  ASSERT_EQ(restored.col_value.size(), 4u);
  EXPECT_NEAR(restored.col_value[0], 3.0, 1e-9);
  EXPECT_NEAR(restored.col_value[1], 2.5, 1e-9);
  ASSERT_EQ(restored.row_dual.size(), 3u);
  EXPECT_NEAR(restored.row_dual[0], 0.0, 1e-9) << "a >= row that is slack by 2";
  EXPECT_NEAR(restored.row_dual[1], -0.25, 1e-9);
  EXPECT_NEAR(restored.row_dual[2], 1.0, 1e-9);
  // d = c - A'y: zero on the free x1, and x0 at its upper bound prices negative.
  ASSERT_EQ(restored.col_dual.size(), 4u);
  EXPECT_NEAR(restored.col_dual[1], 0.0, 1e-9);
  EXPECT_NEAR(restored.col_dual[0], -4.0 - 6.0 * -0.25, 1e-9);
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

TEST(PresolveParallelRows, BoundsThatMeetToRoundingMergeIntoAnEqualityNotACrossedRow) {
  // #1350. x0/10 - x1/2 + x2 <= 97/20 and -3 times it <= -291/20 are one equality. The
  // scale is -0.3 / 0.1 = -2.9999999999999996 and -14.55 divided by it is one ulp above
  // 4.85, so the merged row was written with its lower bound above its upper. x1 is an
  // integer column so that the row stays in the reduced model (a free continuous x1 would
  // be substituted away with it).
  Model first = build({{0.1, -0.5, 1.0}, {-0.3, 1.5, -3.0}}, {-kInf, -kInf}, {4.85, -14.55},
                      {0, 0, 0}, {-kInf, -kInf, -kInf}, {8, kInf, 6});
  first.col_type[1] = VarType::kInteger;
  const presolve::Result reduced = reduce(first, true);
  EXPECT_EQ(reduced.report.parallel_rows, 1);
  EXPECT_EQ(reduced.model.num_rows(), 1);
  EXPECT_EQ(reduced.model.validate(), "");
  for (Index i = 0; i < reduced.model.num_rows(); ++i) {
    const auto u = static_cast<std::size_t>(i);
    EXPECT_LE(reduced.model.row_lower[u], reduced.model.row_upper[u]) << "row " << i;
  }

  // The two models the general-form fuzz (#1334) reduced, as MILPs: the first came back
  // model_error from the search's validate(), the second infeasible from the integral-row
  // rounding, which read the crossed row [1, 1 - 9e-16] as holding no integer. Both have
  // points; the objective is zero.
  // x0 - 3 x2 + x3 >= -17, -x1 + x2/2 >= 0, 2/3 x0 + 5/8 x1 + x2 + 2/3 x3 <= 205/24, with
  // x1 = 3, x2 <= 6, x0 and x3 integer: x2 = 6 and x0 + x3 = 1.
  Model second = build({{1, 0, -3, 1}, {0, -1, 0.5, 0}, {2.0 / 3.0, 0.625, 1, 2.0 / 3.0}},
                       {-17, 0, -kInf}, {kInf, kInf, 205.0 / 24.0}, {0, 0, 0, 0},
                       {-kInf, 3, -kInf, -kInf}, {kInf, 3, 6, kInf});
  second.col_type[0] = VarType::kInteger;
  second.col_type[3] = VarType::kInteger;
  for (const Model* milp : {&first, &second}) {
    const Solution solved = solve(*milp, with_parallel_rows(true));
    ASSERT_EQ(solved.status, SolveStatus::kOptimal) << solved.message;
    EXPECT_NEAR(solved.objective, 0.0, 1e-9);
  }
}

TEST(PresolveParallelRows, BoundsThatCrossBeyondTheToleranceInEitherRowsUnitsAreLeft) {
  // x0 + x1 >= 4 against 1000 (x0 + x1) <= 3999.99996: the bounds cross by 4e-8 in the
  // first row's units, inside 1e-7, and by 4e-5 in the second's, outside it. Merging them
  // into an equality at 4 would leave the second row violated by 4e-5, so they are left.
  const Model model =
      build({{1, 1}, {1000, 1000}}, {4.0, -kInf}, {kInf, 3999.99996}, {1, 1}, {0, 0}, {10, 10});
  EXPECT_EQ(reduce(model, true).report.parallel_rows, 0);
}

TEST(PresolveParallelRows, RowsWithSmallCoefficientsAreComparedByTheirRatio) {
  // #1364. max r1 + 1.25 r2 + z:  s r1 + 1.5 s r2 <= 7,  s r1 + s r2 <= 5,  s = 2^-30,
  // r1, r2 >= 0, z binary. The ratios are 1 and 1.5, so the rows are not parallel; their
  // coefficients differ by 4.7e-10, which an absolute 1e-9 took for equal, and the merged
  // row s r1 + 1.5 s r2 <= 5 cut the optimum r = (2^30, 4 * 2^30), z = 1 down to
  // 5 * 2^30 + 1, feasible on the original model and so accepted as optimal.
  const double s = 0x1p-30;
  Model model =
      build({{s, 1.5 * s, 0}, {s, s, 0}}, {-kInf, -kInf}, {7, 5}, {1, 1.25, 1}, {0, 0, 0},
            {kInf, kInf, 1});
  model.sense = ObjSense::kMaximize;
  model.col_type[2] = VarType::kInteger;
  EXPECT_EQ(reduce(model, true).report.parallel_rows, 0);
  const Solution solved = solve(model, with_parallel_rows(true));
  ASSERT_EQ(solved.status, SolveStatus::kOptimal) << solved.message;
  const double exact = 6 * 0x1p30 + 1;
  EXPECT_NEAR(solved.objective, exact, 1e-9 * exact);

  // The same small entries in a true ratio, 2, still merge.
  const Model scaled = build({{s, 1.5 * s}, {2 * s, 3 * s}}, {-kInf, -kInf}, {7, 5}, {1, 1},
                             {0, 0}, {kInf, kInf});
  EXPECT_EQ(reduce(scaled, true).report.parallel_rows, 1);
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
