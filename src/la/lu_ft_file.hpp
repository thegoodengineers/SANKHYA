// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the row and column files of the Forrest-Tomlin update's U (#279, #1006).
// Internal to src/la; not installed.
//
// One contiguous array of (index, value) entries holds every step's list, each step owning a
// span [start, start + length) with room up to start + capacity. An entry is appended into
// the span's slack; a list that has outgrown its span is moved to the end of the array with
// room to grow, and the span it leaves is dead until the next reset. This is the row file of
// Suhl & Suhl, "A fast LU update for linear programming", Annals of Operations Research 43
// (1993), and of Koberstein's thesis (Paderborn 2005), ch. 5: one array to walk instead of a
// heap allocation per step and a pointer per visit. The factorization is reset at every
// refactorization, which bounds the dead space by the updates between two of them.
//
// THE ORDER WITHIN A LIST IS THE ONE THE NESTED VECTORS IT REPLACES KEPT: append at the back,
// erase by moving the last entry into the hole. Nothing that reads a list depends on that
// order for its arithmetic (every sum in the solves and the update accumulates per target in
// position order), but keeping it means the replacement changes no bit anywhere.
#pragma once

#include <cstddef>
#include <vector>

#include "sankhya/types.hpp"

namespace sankhya {

class FtFile {
 public:
  /// Empty lists for `lists` steps, the list of step s with room for capacity[s] entries.
  void reset(const std::vector<Index>& capacity) {
    const std::size_t n = capacity.size();
    start_.resize(n);
    length_.assign(n, 0);
    capacity_.assign(capacity.begin(), capacity.end());
    Index total = 0;
    for (std::size_t s = 0; s < n; ++s) {
      start_[s] = total;
      total += capacity_[s];
    }
    index_.resize(static_cast<std::size_t>(total));
    value_.resize(static_cast<std::size_t>(total));
  }

  void clear() {
    start_.clear();
    length_.clear();
    capacity_.clear();
    index_.clear();
    value_.clear();
  }

  [[nodiscard]] Index begin(Index s) const { return start_[static_cast<std::size_t>(s)]; }
  [[nodiscard]] Index end(Index s) const {
    return start_[static_cast<std::size_t>(s)] + length_[static_cast<std::size_t>(s)];
  }
  [[nodiscard]] Index length(Index s) const { return length_[static_cast<std::size_t>(s)]; }
  [[nodiscard]] Index index(Index p) const { return index_[static_cast<std::size_t>(p)]; }
  [[nodiscard]] double value(Index p) const { return value_[static_cast<std::size_t>(p)]; }

  /// Append (other, v) to step s's list, moving the list to the end of the file first when
  /// its span is full.
  void append(Index s, Index other, double v) {
    const auto us = static_cast<std::size_t>(s);
    if (length_[us] == capacity_[us]) grow(us);
    const auto at = static_cast<std::size_t>(start_[us] + length_[us]);
    index_[at] = other;
    value_[at] = v;
    ++length_[us];
  }

  /// Set the entry for `other` in step s's list to v, appending it when absent. True when it
  /// was appended.
  bool set(Index s, Index other, double v) {
    for (Index p = begin(s); p < end(s); ++p) {
      if (index_[static_cast<std::size_t>(p)] == other) {
        value_[static_cast<std::size_t>(p)] = v;
        return false;
      }
    }
    append(s, other, v);
    return true;
  }

  /// Remove the entry for `other` from step s's list, the list's last entry taking its
  /// place. True when there was one.
  bool erase(Index s, Index other) {
    const auto us = static_cast<std::size_t>(s);
    for (Index p = begin(s); p < end(s); ++p) {
      const auto up = static_cast<std::size_t>(p);
      if (index_[up] != other) continue;
      const auto last = static_cast<std::size_t>(end(s) - 1);
      index_[up] = index_[last];
      value_[up] = value_[last];
      --length_[us];
      return true;
    }
    return false;
  }

 private:
  // ponytail: a moved list's old span stays dead until reset(); the factorization is reset
  // every refactorization (at most kMaxEtaCount updates), so the waste is bounded by the
  // fill those updates add. Compact in place if a caller ever keeps one factorization longer.
  void grow(std::size_t us) {
    const Index length = length_[us];
    const Index capacity = 2 * length + kSlack;
    const auto from = static_cast<std::size_t>(start_[us]);
    const auto to = index_.size();
    index_.resize(to + static_cast<std::size_t>(capacity));
    value_.resize(to + static_cast<std::size_t>(capacity));
    for (std::size_t t = 0; t < static_cast<std::size_t>(length); ++t) {
      index_[to + t] = index_[from + t];
      value_[to + t] = value_[from + t];
    }
    start_[us] = static_cast<Index>(to);
    capacity_[us] = capacity;
  }

  /// Room a moved list gets beyond twice its length. A storage constant, not a tolerance:
  /// it decides only how often a growing list moves, never what it holds.
  static constexpr Index kSlack = 4;

  std::vector<Index> start_, length_, capacity_;
  std::vector<Index> index_;
  std::vector<double> value_;
};

}  // namespace sankhya
