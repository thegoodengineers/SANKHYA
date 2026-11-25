// SPDX-License-Identifier: Apache-2.0
// SANKHYA - probing's row activities change the time, never a deduction (probing.cpp).
//
// The prober keeps every row's minimum and maximum activity and passes over a queued row
// whose activities show the full rescan could do nothing, charging the work the rescan
// would have been. That is only worth having if it is exact, so both propagations run here
// on the same random models - binaries, general integers, continuous columns with finite,
// infinite and fractional bounds, packing, partitioning, covering and knapsack rows - under
// generous and tight work limits, and every output must be identical to the bit: the
// bounds, the counts, the conflicts and the cliques. The pass-over must also have happened,
// or the comparison proved nothing.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "presolve/probing.hpp"
#include "sankhya/model.hpp"

namespace sankhya {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

Model random_model(std::mt19937_64& rng) {
  std::uniform_int_distribution<int> cols(8, 60);
  std::uniform_int_distribution<int> rows_count(4, 40);
  std::uniform_int_distribution<int> percent(0, 99);
  std::uniform_int_distribution<int> small(-6, 6);
  std::uniform_real_distribution<double> real(-4.0, 4.0);
  const int n = cols(rng);
  const int m = rows_count(rng);
  Model model;
  model.col_cost.assign(static_cast<std::size_t>(n), 0.0);
  model.col_lower.assign(static_cast<std::size_t>(n), 0.0);
  model.col_upper.assign(static_cast<std::size_t>(n), 1.0);
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kInteger);
  for (std::size_t j = 0; j < static_cast<std::size_t>(n); ++j) {
    const int kind = percent(rng);
    if (kind < 65) continue;  // binary
    if (kind < 78) {          // general integer, now and then with a fractional bound
      model.col_lower[j] = static_cast<double>(small(rng) / 3);
      model.col_upper[j] = model.col_lower[j] + static_cast<double>(1 + percent(rng) % 5);
      if (percent(rng) < 10) model.col_upper[j] += 0.5;
      continue;
    }
    model.col_type[j] = VarType::kContinuous;  // finite, half-open or free
    const double base = real(rng) - 2.0;
    model.col_lower[j] = percent(rng) < 20 ? -kInf : base;
    model.col_upper[j] = percent(rng) < 20 ? kInf : base + 0.25 + 3.0 * (real(rng) + 4.0);
  }
  model.matrix.reset(m, n);
  model.row_lower.assign(static_cast<std::size_t>(m), -kInf);
  model.row_upper.assign(static_cast<std::size_t>(m), kInf);
  std::uniform_int_distribution<int> pick(0, n - 1);
  for (Index i = 0; i < m; ++i) {
    const auto r = static_cast<std::size_t>(i);
    const int shape = percent(rng);
    const int length = 2 + percent(rng) % std::min(n - 1, 12);
    std::vector<char> used(static_cast<std::size_t>(n), 0);
    double reach = 0.0;
    for (int t = 0; t < length; ++t) {
      const Index j = pick(rng);
      if (used[static_cast<std::size_t>(j)] != 0) continue;
      used[static_cast<std::size_t>(j)] = 1;
      double a = 1.0;
      if (shape >= 40) a = shape < 70 ? static_cast<double>(small(rng)) : real(rng);
      if (a == 0.0) a = 2.0;
      model.matrix.add_entry(i, j, a);
      reach += std::fabs(a);
    }
    if (shape < 20) {  // partitioning
      model.row_lower[r] = 1.0;
      model.row_upper[r] = 1.0;
    } else if (shape < 30) {  // packing
      model.row_upper[r] = 1.0;
    } else if (shape < 40) {  // covering
      model.row_lower[r] = 1.0;
    } else {
      const double b = std::round(real(rng) * reach / 8.0);
      switch (percent(rng) % 3) {
        case 0: model.row_upper[r] = b; break;
        case 1: model.row_lower[r] = b; break;
        default:
          model.row_lower[r] = b;
          model.row_upper[r] = b + static_cast<double>(percent(rng) % 3);
          break;
      }
    }
  }
  model.matrix.finalize();
  model.hessian.reset(n, n);
  model.hessian.finalize();
  return model;
}

void expect_identical(const presolve::ProbingResult& kept, const presolve::ProbingResult& full,
                      int trial) {
  EXPECT_EQ(kept.probed, full.probed) << "trial " << trial;
  EXPECT_EQ(kept.fixings, full.fixings) << "trial " << trial;
  EXPECT_EQ(kept.tightenings, full.tightenings) << "trial " << trial;
  EXPECT_EQ(kept.implications, full.implications) << "trial " << trial;
  EXPECT_EQ(kept.infeasible, full.infeasible) << "trial " << trial;
  EXPECT_EQ(kept.work_limit_reached, full.work_limit_reached) << "trial " << trial;
  EXPECT_EQ(kept.conflicts, full.conflicts) << "trial " << trial;
  EXPECT_EQ(kept.conflicts_capped, full.conflicts_capped) << "trial " << trial;
  EXPECT_EQ(kept.cliques, full.cliques) << "trial " << trial;
  EXPECT_EQ(kept.largest_clique, full.largest_clique) << "trial " << trial;
  if (kept.infeasible) return;  // the bounds are not meaningful then
  ASSERT_EQ(kept.col_lower.size(), full.col_lower.size());
  for (std::size_t j = 0; j < kept.col_lower.size(); ++j) {
    // Bit for bit: the same deductions in the same order give the same doubles.
    EXPECT_EQ(kept.col_lower[j], full.col_lower[j]) << "trial " << trial << " column " << j;
    EXPECT_EQ(kept.col_upper[j], full.col_upper[j]) << "trial " << trial << " column " << j;
  }
}

TEST(PresolveProbingActivities, TheKeptActivitiesGiveTheFullRescansDeductions) {
  std::mt19937_64 rng(1345);
  Count passed_over = 0, fixings = 0, tightenings = 0, limited = 0, infeasible = 0;
  int compared = 0;
  for (int trial = 0; trial < 1500; ++trial) {
    const Model model = random_model(rng);
    ASSERT_EQ(model.validate(), "") << "trial " << trial;
    // Generous, then tight enough that the work limits stop probing part way.
    for (const std::int64_t work :
         {std::int64_t{10000000}, std::int64_t{400}, std::int64_t{60}}) {
      const presolve::ProbingResult kept =
          presolve::probe_binaries(model, model.col_lower, model.col_upper, work, true);
      const presolve::ProbingResult full =
          presolve::probe_binaries(model, model.col_lower, model.col_upper, work, false);
      expect_identical(kept, full, trial);
      if (::testing::Test::HasFailure()) return;
      EXPECT_EQ(full.rows_passed_over, 0);
      passed_over += kept.rows_passed_over;
      fixings += kept.fixings;
      tightenings += kept.tightenings;
      limited += kept.work_limit_reached ? 1 : 0;
      infeasible += kept.infeasible ? 1 : 0;
      ++compared;
    }
  }
  std::printf(
      "  %d comparisons: %lld rows passed over, %lld fixings, %lld tightenings, "
      "%lld stopped by the work limit, %lld infeasible\n",
      compared, static_cast<long long>(passed_over), static_cast<long long>(fixings),
      static_cast<long long>(tightenings), static_cast<long long>(limited),
      static_cast<long long>(infeasible));
  // The comparison proves something only if each path was exercised.
  EXPECT_GT(passed_over, 1000);
  EXPECT_GT(fixings, 100);
  EXPECT_GT(tightenings, 100);
  EXPECT_GT(limited, 100);
}

}  // namespace
}  // namespace sankhya
