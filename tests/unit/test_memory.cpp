// SPDX-License-Identifier: Apache-2.0
// The physical-memory probe and the interior point budgets derived from it (#576).
#include "util/memory.hpp"

#include <gtest/gtest.h>

namespace sankhya {
namespace {

constexpr std::size_t kGiB = std::size_t{1} << 30;

TEST(Memory, TheMachineReportsItsPhysicalMemory) {
  // Every machine this runs on - a laptop, a CI runner, a rented node - reports a size, and
  // one that is at least a gigabyte; 0 is the "platform did not say" value the budgets treat
  // as the laptop.
  EXPECT_GE(physical_memory_bytes(), kGiB);
}

TEST(Memory, TheOrderingBudgetIsASixteenthOfMemoryAndNeverBelowTheLaptopConstant) {
  // Unknown memory: the constant the 7.7 GB laptop shipped with (#246).
  EXPECT_EQ(auto_ordering_budget(0), std::size_t{100000000});
  // The laptop itself, 7.7 GB: a sixteenth is 481 MB, 1.2e8 entries of 4 bytes - a little
  // above the constant, as it should be.
  EXPECT_EQ(auto_ordering_budget(std::size_t{7700} << 20), (std::size_t{7700} << 20) / 16 / 4);
  // A 2 GB box gets the floor, not less.
  EXPECT_EQ(auto_ordering_budget(2 * kGiB), std::size_t{100000000});
  // The 96 GB node that abandoned three Mittelmann orderings at the constant: 1.5e9
  // entries, 6 GB - the budget that would have let them run.
  EXPECT_EQ(auto_ordering_budget(96 * kGiB), 96 * kGiB / 16 / 4);
  EXPECT_GT(auto_ordering_budget(96 * kGiB), std::size_t{1000000000});
}

TEST(Memory, TheFactorBudgetIsAnEighthOfMemoryAndNeverBelowTheLaptopConstant) {
  EXPECT_EQ(auto_factor_budget(0), std::size_t{100000000});
  EXPECT_EQ(auto_factor_budget(2 * kGiB), std::size_t{100000000});
  // 96 GB: an eighth is 12 GB, a billion nonzeros of 12 bytes.
  EXPECT_EQ(auto_factor_budget(96 * kGiB), 96 * kGiB / 8 / 12);
}

}  // namespace
}  // namespace sankhya
