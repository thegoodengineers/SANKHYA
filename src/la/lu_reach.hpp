// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the symbolic reach shared by the hyper-sparse solves (#464, lu_hyper.cpp) and
// their Forrest-Tomlin counterparts (#1042, lu_ft_hyper.cpp). Internal to src/la; not
// installed. Gilbert & Peierls (1988): the nonzero pattern of a triangular solve's result
// is the set of nodes reachable from the right-hand side's nonzeros.
#pragma once

#include <cstddef>
#include <vector>

#include "sankhya/types.hpp"

namespace sankhya::lu_reach {

/// Close `list` (whose entries are already marked) under the edges `for_each_target(k, visit)`
/// names, marking and appending each newly reached step. Returns false as soon as the list
/// passes `limit`; the marks of everything in `list` are the caller's to clear either way.
template <typename ForEachTarget>
bool close_reach(std::vector<char>& mark, std::vector<Index>& list, std::size_t limit,
                 ForEachTarget&& for_each_target) {
  for (std::size_t head = 0; head < list.size(); ++head) {
    for_each_target(list[head], [&](Index target) {
      const auto u = static_cast<std::size_t>(target);
      if (mark[u] != 0) return;
      mark[u] = 1;
      list.push_back(target);
    });
    if (list.size() > limit) return false;
  }
  return true;
}

inline void clear_marks(std::vector<char>& mark, const std::vector<Index>& list) {
  for (const Index k : list) mark[static_cast<std::size_t>(k)] = 0;
}

}  // namespace sankhya::lu_reach
