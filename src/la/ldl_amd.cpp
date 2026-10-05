// SPDX-License-Identifier: Apache-2.0
// SANKHYA - approximate minimum degree with supervariables and mass elimination (#471).
//
// Amestoy, Davis & Duff, "An approximate minimum degree ordering algorithm", SIAM J. Matrix
// Anal. Appl. 17(4), 1996. Implemented from the paper's description; none of it from another
// solver's or library's code. The unweighted version in ldl.cpp is the oracle this one is
// tested against, and it explains the quotient graph, the approximate degree and aggressive
// absorption, which are unchanged here. What this file adds is the weight.
//
// A SUPERVARIABLE is a set of variables with identical closed neighbourhoods in the quotient
// graph, kept as one node of weight nv. They are eliminated together, one after the other in
// the permutation, which costs no more fill than eliminating the first alone: once the first
// is gone the rest are adjacent to exactly the clique it formed. Every degree is a WEIGHTED
// degree, the number of variables and not of nodes, and the bound n - (eliminated) - nv_i
// replaces n - k - 1. After a pivot p, the members of L_p are compared in hash buckets of
// their (A_i, E_i) lists and merged when the lists are equal as sets. The hash only filters;
// the comparison is exact, so a collision costs a comparison and never a wrong merge.
//
// MASS ELIMINATION: a member i of L_p left with no variable neighbour and no live element
// except p has the neighbourhood L_p, the same as p's, so it leaves with p.
//
// Not built: dense-row postponement (the interior point removes its dense columns before the
// ordering, dense_columns.cpp). Ties go to the most recently inserted node of the minimum
// degree and every list is built in index order, so the permutation is deterministic.

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "la/ldl.hpp"
#include "la/ldl_internal.hpp"

namespace sankhya {

using ldl_detail::DeadlineByWork;

bool SparseLdl::minimum_degree_supervariable(const SparseMatrix& lower,
                                             const ShouldStop& should_stop) {
  const Index n = n_;
  const auto un = static_cast<std::size_t>(n);
  const auto at = [](Index i) { return static_cast<std::size_t>(i); };

  // The quotient graph, as in ldl.cpp: A_i, E_i and L_e; an element reuses its pivot's index.
  std::vector<std::vector<Index>> adjacent(un);
  std::vector<std::vector<Index>> elements(un);
  std::vector<std::vector<Index>> members(un);
  DeadlineByWork deadline(should_stop);
  for (Index c = 0; c < n; ++c) {
    const ColumnView column = lower.column(c);
    if (deadline.expired(static_cast<std::size_t>(column.size) + 1)) return false;
    for (Index p = 0; p < column.size; ++p) {
      const Index r = column.rows[p];
      if (r <= c) continue;  // the strict lower triangle defines the graph
      adjacent[at(r)].push_back(c);
      adjacent[at(c)].push_back(r);
    }
  }
  std::size_t live = 0;  // list entries held, against the caller's budget (#246)
  for (auto& list : adjacent) {
    if (deadline.expired(list.size() + 1)) return false;
    std::sort(list.begin(), list.end());
    list.erase(std::unique(list.begin(), list.end()), list.end());
    live += list.size();
  }

  // kGone: merged into a supervariable, or eliminated with a pivot. Neither is a variable
  // any more and neither is an element; stale mentions of it in other lists are skipped.
  enum class Status : std::uint8_t { kVariable, kElement, kAbsorbed, kGone };
  std::vector<Status> status(un, Status::kVariable);
  std::vector<Index> degree(un, 0);
  std::vector<Index> nv(un, 1);     // the weight of a supervariable; 0 once it is gone
  std::vector<Index> esize(un, 0);  // sum of nv over an element's live members
  // The variables a node stands for, as a linked list from the principal one: gnext chains
  // them and gtail is the end of a node's chain, so merging is a constant-time splice.
  std::vector<Index> gnext(un, -1);
  std::vector<Index> gtail(un);
  for (Index i = 0; i < n; ++i) {
    degree[at(i)] = static_cast<Index>(adjacent[at(i)].size());
    gtail[at(i)] = i;
  }

  // Degree buckets: doubly linked lists, one per degree, head insertion.
  std::vector<Index> head(un + 1, -1);
  std::vector<Index> next(un, -1);
  std::vector<Index> prev(un, -1);
  const auto insert = [&](Index i) {
    const auto d = at(degree[at(i)]);
    next[at(i)] = head[d];
    prev[at(i)] = -1;
    if (head[d] >= 0) prev[at(head[d])] = i;
    head[d] = i;
  };
  const auto remove = [&](Index i) {
    const auto d = at(degree[at(i)]);
    if (prev[at(i)] >= 0) {
      next[at(prev[at(i)])] = next[at(i)];
    } else {
      head[d] = next[at(i)];
    }
    if (next[at(i)] >= 0) prev[at(next[at(i)])] = prev[at(i)];
  };
  for (Index i = 0; i < n; ++i) insert(i);
  Index min_degree = 0;

  std::vector<Index> in_lp(un, -1);  // stamp: in_lp[i] == step means i is in L_p at this step
  std::vector<Index> w(un, 0);       // |L_e \ L_p|, weighted, for elements touching L_p
  std::vector<Index> w_stamp(un, -1);
  std::vector<Index> same(un, -1);  // stamp for the exact comparison of two neighbourhoods
  std::vector<Index> lp;
  std::vector<Index> external;  // per member of L_p: weighted sum of w over its live elements
  std::vector<Index> a_weight;  // per member of L_p: weighted size of A_i after pruning
  std::vector<std::pair<std::uint64_t, Index>> keyed;
  perm_.clear();
  perm_.reserve(un);
  Index pivots = 0;
  const auto emit = [&](Index principal) {
    for (Index v = principal; v >= 0; v = gnext[at(v)]) perm_.push_back(v);
  };

  Index done = 0;  // variables eliminated so far, supervariables counted by their weight
  Index step = 0;
  Index compare_stamp = 0;
  while (done < n) {
    // Asked every step, as in ldl.cpp: the deadline is what makes a time limit mean anything.
    if (should_stop && should_stop()) return false;
    if (live > ordering_budget_) {
      ordering_too_large_ = true;
      return false;
    }

    while (head[at(min_degree)] < 0) ++min_degree;
    const Index p = head[at(min_degree)];
    remove(p);
    status[at(p)] = Status::kElement;
    ++pivots;
    emit(p);
    done += nv[at(p)];
    ++step;

    // L_p: the live nodes of A_p and of every element p touches, which p absorbs.
    lp.clear();
    Index lp_weight = 0;
    in_lp[at(p)] = step;
    for (const Index i : adjacent[at(p)]) {
      if (status[at(i)] != Status::kVariable || in_lp[at(i)] == step) continue;
      in_lp[at(i)] = step;
      lp.push_back(i);
      lp_weight += nv[at(i)];
    }
    for (const Index e : elements[at(p)]) {
      if (status[at(e)] != Status::kElement) continue;
      for (const Index i : members[at(e)]) {
        if (status[at(i)] != Status::kVariable || in_lp[at(i)] == step) continue;
        in_lp[at(i)] = step;
        lp.push_back(i);
        lp_weight += nv[at(i)];
      }
      status[at(e)] = Status::kAbsorbed;
      live -= members[at(e)].size();
      std::vector<Index>().swap(members[at(e)]);
    }
    live -= adjacent[at(p)].size() + elements[at(p)].size();
    std::vector<Index>().swap(adjacent[at(p)]);
    std::vector<Index>().swap(elements[at(p)]);

    // w[e] = |L_e \ L_p|, weighted: start at the element's weight and take off each member of
    // L_p that lists it.
    for (const Index i : lp) {
      for (const Index e : elements[at(i)]) {
        if (status[at(e)] != Status::kElement) continue;
        if (w_stamp[at(e)] != step) {
          w_stamp[at(e)] = step;
          w[at(e)] = esize[at(e)];
        }
        w[at(e)] -= nv[at(i)];
      }
    }

    // First pass over L_p: prune each member's lists, absorb elements wholly inside L_p, and
    // send off the members that have nothing left but p (mass elimination).
    external.assign(lp.size(), 0);
    a_weight.assign(lp.size(), 0);
    std::size_t kept_members = 0;
    std::size_t visited = 0;
    for (std::size_t t = 0; t < lp.size(); ++t) {
      const Index i = lp[t];
      if (should_stop && (++visited & 4095) == 0 && should_stop()) return false;
      remove(i);
      std::size_t keep = 0;
      Index aw = 0;
      for (const Index v : adjacent[at(i)]) {
        if (status[at(v)] != Status::kVariable || in_lp[at(v)] == step) continue;
        adjacent[at(i)][keep++] = v;
        aw += nv[at(v)];
      }
      live -= adjacent[at(i)].size() - keep;
      adjacent[at(i)].resize(keep);
      Index ext = 0;
      keep = 0;
      for (const Index e : elements[at(i)]) {
        if (status[at(e)] != Status::kElement) continue;
        if (w[at(e)] <= 0) {
          status[at(e)] = Status::kAbsorbed;
          live -= members[at(e)].size();
          std::vector<Index>().swap(members[at(e)]);
          continue;
        }
        elements[at(i)][keep++] = e;
        ext += w[at(e)];
      }
      live -= elements[at(i)].size() - keep;
      elements[at(i)].resize(keep);

      if (adjacent[at(i)].empty() && elements[at(i)].empty()) {
        // Mass elimination: i's neighbourhood is exactly L_p, so it follows p at no cost.
        status[at(i)] = Status::kGone;
        emit(i);
        done += nv[at(i)];
        lp_weight -= nv[at(i)];
        std::vector<Index>().swap(adjacent[at(i)]);
        std::vector<Index>().swap(elements[at(i)]);
        continue;
      }
      elements[at(i)].push_back(p);
      ++live;
      lp[kept_members] = i;
      external[kept_members] = ext;
      a_weight[kept_members] = aw;
      ++kept_members;
    }
    lp.resize(kept_members);

    // Second pass: the approximate external degree, the smallest of three upper bounds as in
    // ldl.cpp, now over weights.
    for (std::size_t t = 0; t < lp.size(); ++t) {
      const Index i = lp[t];
      const Index others = lp_weight - nv[at(i)];
      const Index remaining = n - done - nv[at(i)];
      const Index by_old_degree = degree[at(i)] + others;
      const Index by_lists = a_weight[t] + others + external[t];
      degree[at(i)] = std::max<Index>(0, std::min({remaining, by_old_degree, by_lists}));
    }

    // Supervariable detection among the members that remain: equal hash, then equal lists.
    keyed.clear();
    for (const Index i : lp) {
      std::uint64_t h = 0;
      for (const Index v : adjacent[at(i)]) h += static_cast<std::uint64_t>(v);
      for (const Index e : elements[at(i)]) h += static_cast<std::uint64_t>(e);
      keyed.emplace_back(h, i);
    }
    std::sort(keyed.begin(), keyed.end());
    for (std::size_t a = 0; a < keyed.size();) {
      std::size_t z = a + 1;
      while (z < keyed.size() && keyed[z].first == keyed[a].first) ++z;
      for (std::size_t x = a; z - a > 1 && x < z; ++x) {
        const Index i = keyed[x].second;
        if (status[at(i)] != Status::kVariable) continue;
        ++compare_stamp;
        for (const Index v : adjacent[at(i)]) same[at(v)] = compare_stamp;
        for (const Index e : elements[at(i)]) same[at(e)] = compare_stamp;
        for (std::size_t y = x + 1; y < z; ++y) {
          const Index j = keyed[y].second;
          if (status[at(j)] != Status::kVariable) continue;
          if (adjacent[at(j)].size() != adjacent[at(i)].size() ||
              elements[at(j)].size() != elements[at(i)].size()) {
            continue;
          }
          const auto marked = [&](Index v) { return same[at(v)] == compare_stamp; };
          if (!std::all_of(adjacent[at(j)].begin(), adjacent[at(j)].end(), marked) ||
              !std::all_of(elements[at(j)].begin(), elements[at(j)].end(), marked)) {
            continue;
          }
          // j is indistinguishable from i: i takes its weight and its variables. j was part
          // of i's external degree and is now part of i itself.
          nv[at(i)] += nv[at(j)];
          degree[at(i)] = std::max<Index>(0, degree[at(i)] - nv[at(j)]);
          nv[at(j)] = 0;
          status[at(j)] = Status::kGone;
          gnext[at(gtail[at(i)])] = j;
          gtail[at(i)] = gtail[at(j)];
          live -= adjacent[at(j)].size() + elements[at(j)].size();
          std::vector<Index>().swap(adjacent[at(j)]);
          std::vector<Index>().swap(elements[at(j)]);
        }
      }
      a = z;
    }

    // The new element p holds the members that are still nodes; the merged ones are inside
    // their principal. Its weight is what the next steps start w from.
    std::size_t out = 0;
    for (const Index i : lp) {
      if (status[at(i)] != Status::kVariable) continue;
      lp[out++] = i;
      insert(i);
      if (degree[at(i)] < min_degree) min_degree = degree[at(i)];
    }
    lp.resize(out);
    members[at(p)] = lp;
    esize[at(p)] = lp_weight;
    live += lp.size();
  }
  // Every variable that was not a pivot was merged into one or left with one.
  absorbed_ = n - pivots;
  inverse_.assign(un, -1);
  for (Index k = 0; k < n; ++k) inverse_[at(perm_[at(k)])] = k;
  return true;
}

}  // namespace sankhya
