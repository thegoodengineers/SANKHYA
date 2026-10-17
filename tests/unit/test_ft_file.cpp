// SPDX-License-Identifier: Apache-2.0
// The Forrest-Tomlin row/column file (src/la/lu_ft_file.hpp) against the nested vectors it
// replaced. The update's arithmetic does not read a list's order, but the claim that the
// replacement changes no bit rests on the lists holding the same entries in the same order
// after any sequence of appends, sets and erases - including lists that outgrow their span.
#include <gtest/gtest.h>

#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include "la/lu_ft_file.hpp"

namespace sankhya {
namespace {

using Reference = std::vector<std::vector<std::pair<Index, double>>>;

// The pre-#1006 ft_set/ft_erase on one side of the mirror, verbatim in effect.
bool reference_set(Reference& lists, Index s, Index other, double v) {
  for (auto& entry : lists[static_cast<std::size_t>(s)]) {
    if (entry.first == other) {
      entry.second = v;
      return false;
    }
  }
  lists[static_cast<std::size_t>(s)].emplace_back(other, v);
  return true;
}

bool reference_erase(Reference& lists, Index s, Index other) {
  auto& r = lists[static_cast<std::size_t>(s)];
  for (std::size_t i = 0; i < r.size(); ++i) {
    if (r[i].first != other) continue;
    r[i] = r.back();
    r.pop_back();
    return true;
  }
  return false;
}

void expect_same(const FtFile& file, const Reference& lists) {
  for (std::size_t s = 0; s < lists.size(); ++s) {
    const auto step = static_cast<Index>(s);
    ASSERT_EQ(file.length(step), static_cast<Index>(lists[s].size())) << "step " << s;
    Index p = file.begin(step);
    for (const auto& [other, v] : lists[s]) {
      EXPECT_EQ(file.index(p), other) << "step " << s;
      EXPECT_EQ(file.value(p), v) << "step " << s;
      ++p;
    }
  }
}

TEST(FtFile, MatchesTheNestedVectorsEntryForEntryAndInOrder) {
  constexpr Index kSteps = 40;
  std::mt19937 rng(20261017);
  std::uniform_int_distribution<Index> step(0, kSteps - 1);
  std::uniform_real_distribution<double> value(-1.0, 1.0);
  std::uniform_int_distribution<int> op(0, 9);

  // Start as ft_init() does: some entries per list, capacity = entries + a little slack.
  Reference lists(static_cast<std::size_t>(kSteps));
  std::vector<Index> room(static_cast<std::size_t>(kSteps));
  for (Index s = 0; s < kSteps; ++s) {
    const Index count = s % 5;
    room[static_cast<std::size_t>(s)] = count + 1;
    for (Index t = 0; t < count; ++t) {
      lists[static_cast<std::size_t>(s)].emplace_back((s + 7 * t + 1) % kSteps, value(rng));
    }
  }
  FtFile file;
  file.reset(room);
  for (Index s = 0; s < kSteps; ++s) {
    for (const auto& [other, v] : lists[static_cast<std::size_t>(s)]) file.append(s, other, v);
  }
  expect_same(file, lists);

  // Many more sets than erases, so lists outgrow their spans and move, repeatedly.
  for (int k = 0; k < 20000; ++k) {
    const Index s = step(rng);
    const Index other = step(rng);
    if (op(rng) < 7) {
      const double v = value(rng);
      EXPECT_EQ(file.set(s, other, v), reference_set(lists, s, other, v));
    } else {
      EXPECT_EQ(file.erase(s, other), reference_erase(lists, s, other));
    }
    if (k % 997 == 0) expect_same(file, lists);
  }
  expect_same(file, lists);
}

TEST(FtFile, ResetEmptiesEveryList) {
  FtFile file;
  file.reset({2, 0, 3});
  file.append(0, 5, 1.0);
  file.append(1, 6, 2.0);  // zero room: moves to the end first
  file.append(2, 7, 3.0);
  EXPECT_EQ(file.length(1), 1);
  EXPECT_EQ(file.index(file.begin(1)), 6);
  file.reset({1, 1, 1});
  for (Index s = 0; s < 3; ++s) EXPECT_EQ(file.length(s), 0);
}

}  // namespace
}  // namespace sankhya
