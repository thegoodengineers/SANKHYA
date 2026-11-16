// SPDX-License-Identifier: Apache-2.0
// SANKHYA - supernodal numeric factorization for the sparse LDL^T (#470).
//
// References:
//   Liu, Ng & Peyton, "On finding supernodes for sparse matrix computations", SIAM J. Matrix
//     Anal. Appl. 14 (1993) - fundamental supernodes from the elimination tree and the
//     column counts.
//   Ng & Peyton, "Block sparse Cholesky algorithms on advanced uniprocessor computers", SIAM
//     J. Sci. Comput. 14 (1993) - the left-looking supernodal factorization: each supernode
//     gathers the updates of the supernodes below it in the tree as dense products, then
//     factors its own dense block.
//   Rothberg & Gupta, "An efficient block-oriented approach to parallel sparse Cholesky
//     factorization", SIAM J. Sci. Comput. 15 (1994) - the supernodal elimination tree as the
//     task graph: a supernode is updated only by its descendants, so disjoint subtrees are
//     independent work.
//   Liu, "The role of elimination trees in sparse factorization", SIAM J. Matrix Anal. Appl.
//     11 (1990); Davis, "Direct Methods for Sparse Linear Systems", SIAM (2006), ch. 4.
//
// WHAT THIS IS. The symbolic analysis (ordering, elimination tree, pattern of L) is the scalar
// path's, unchanged. The numeric factorization works on supernodes - runs of columns with
// nested patterns, whose rows below the diagonal block are one shared list - so every update
// and every column operation is a dense loop over contiguous memory rather than a scatter
// through the pattern. The result is written back into the scalar layout (l_values_, d_), so
// solve() and every caller see exactly the factor they always saw, and the scalar path stays
// the oracle the tests compare with. The dense kernels are plain loops written here, no
// vendor BLAS (#470).
//
// THE THREADS. A supernode's updates come from its descendants in the supernodal elimination
// tree only (every row in a column's pattern is an ancestor of that column, Liu 1990), so a
// supernode is ready once its children are done, and the tree is walked from its leaves by a
// pool of threads over a ready queue. A thread that finishes a supernode takes the parent
// itself when that made it ready, so a chain costs no queue traffic. Every supernode is
// computed by exactly one thread, from the same loads, the same updates in the same order
// (sources ascending, fixed by the analysis) and the same dense factorization, so the factor
// is bit-for-bit the same under one thread and under eight.
//
// THE PIVOT RULE IS THE SCALAR ONE: a pivot not above `regularization` is replaced by it
// (Altman & Gondzio 1999), counted, and the factorization goes on. With `signs` (a
// quasi-definite matrix, the proximal interior point's augmented system) the same rule on the
// pivot's expected sign: a quasi-definite matrix has an LDL^T factorization with no
// interchanges under ANY symmetric permutation, D taking the signs of its two diagonal
// blocks (Vanderbei, "Symmetric quasidefinite matrices", SIAM J. Optim. 5 (1995)), so the
// supernodal order and the dense blocks need no pivoting there either.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

#include "la/ldl.hpp"

#ifdef SANKHYA_HAVE_OPENMP
#include <omp.h>
#endif

namespace sankhya {

/// One thread's scratch: the row-to-slot map of the supernode it is on, the accumulator of
/// one update column, its work since the deadline was last asked, and its pivot statistics.
struct SparseLdl::SupernodeWorkspace {
  std::vector<Index> position;
  std::vector<double> accumulator;
  std::size_t work = 0;
  Index regularized = 0;
  double smallest = std::numeric_limits<double>::infinity();
  double largest = 0.0;
};

void SparseLdl::build_supernodes() {
  const Index n = n_;
  const auto un = static_cast<std::size_t>(n);
  // Fundamental supernodes (Liu, Ng & Peyton 1993): column j + 1 continues j's supernode
  // when it is j's parent, j is its only child, and its pattern is j's without j + 1 - which,
  // given the parent, is the column counts differing by exactly one.
  std::vector<Index> children(un, 0);
  for (Index j = 0; j < n; ++j) {
    const Index p = parent_[static_cast<std::size_t>(j)];
    if (p >= 0) ++children[static_cast<std::size_t>(p)];
  }
  const auto count = [&](Index j) {
    return l_starts_[static_cast<std::size_t>(j) + 1] - l_starts_[static_cast<std::size_t>(j)];
  };
  super_start_.clear();
  super_of_.assign(un, 0);
  for (Index j = 0; j < n; ++j) {
    const bool continues = j > 0 && parent_[static_cast<std::size_t>(j) - 1] == j &&
                           children[static_cast<std::size_t>(j)] == 1 &&
                           count(j - 1) == count(j) + 1;
    if (!continues) super_start_.push_back(j);
    super_of_[static_cast<std::size_t>(j)] = static_cast<Index>(super_start_.size()) - 1;
  }
  super_start_.push_back(n);
  const auto supernodes = super_start_.size() - 1;

  // The supernodal elimination tree: the parent of a supernode's last column, by supernode.
  super_parent_.assign(supernodes, -1);
  for (std::size_t s = 0; s < supernodes; ++s) {
    const Index p = parent_[static_cast<std::size_t>(super_start_[s + 1]) - 1];
    if (p >= 0) super_parent_[s] = super_of_[static_cast<std::size_t>(p)];
  }

  // The lower triangle of the permuted matrix by column, for loading a supernode's columns:
  // a_ holds it by row (column k of the upper triangle), with the slot of each value.
  lower_starts_.assign(un + 1, 0);
  for (Index k = 0; k < n; ++k) {
    for (Index p = a_starts_[static_cast<std::size_t>(k)];
         p < a_starts_[static_cast<std::size_t>(k) + 1]; ++p) {
      ++lower_starts_[static_cast<std::size_t>(a_rows_[static_cast<std::size_t>(p)]) + 1];
    }
  }
  for (std::size_t i = 0; i < un; ++i) lower_starts_[i + 1] += lower_starts_[i];
  lower_rows_.assign(static_cast<std::size_t>(lower_starts_[un]), 0);
  lower_slots_.assign(static_cast<std::size_t>(lower_starts_[un]), 0);
  std::vector<Index> next(lower_starts_.begin(), lower_starts_.end() - 1);
  for (Index k = 0; k < n; ++k) {
    for (Index p = a_starts_[static_cast<std::size_t>(k)];
         p < a_starts_[static_cast<std::size_t>(k) + 1]; ++p) {
      const auto i = static_cast<std::size_t>(a_rows_[static_cast<std::size_t>(p)]);
      const auto slot = static_cast<std::size_t>(next[i]++);
      lower_rows_[slot] = k;
      lower_slots_[slot] = p;
    }
  }

  // One dense column-major block per supernode: the width w, times the rows it touches -
  // its own w columns and the shared rows below.
  block_start_.assign(supernodes + 1, 0);
  for (std::size_t s = 0; s < supernodes; ++s) {
    const Index f = super_start_[s];
    const auto w = static_cast<std::size_t>(super_start_[s + 1] - f);
    const auto rows = static_cast<std::size_t>(count(f)) + 1;  // f's own row, then its pattern
    block_start_[s + 1] = block_start_[s] + rows * w;
  }

  // The updates, by target (Ng & Peyton): each run of a supernode's shared rows that falls
  // in one target supernode is one dense update of that target. Sources are visited
  // ascending, so each target's list is in the order the sequential left-looking sweep
  // applied them, whichever thread applies them.
  update_starts_.assign(supernodes + 1, 0);
  for (int pass = 0; pass < 2; ++pass) {
    for (std::size_t s = 0; s < supernodes; ++s) {
      const Index f = super_start_[s];
      const Index w = super_start_[s + 1] - f;
      const Index* shared = l_rows_.data() + l_starts_[static_cast<std::size_t>(f)] + (w - 1);
      const Index below = count(f) - (w - 1);
      for (Index r = 0; r < below;) {
        const auto target =
            static_cast<std::size_t>(super_of_[static_cast<std::size_t>(shared[r])]);
        Index e = r;
        while (e < below && static_cast<std::size_t>(
                                super_of_[static_cast<std::size_t>(shared[e])]) == target)
          ++e;
        if (pass == 0) {
          ++update_starts_[target + 1];
        } else {
          updates_[static_cast<std::size_t>(update_starts_[target]++)] = {static_cast<Index>(s),
                                                                          r, e};
        }
        r = e;
      }
    }
    if (pass == 0) {
      for (std::size_t s = 0; s < supernodes; ++s) update_starts_[s + 1] += update_starts_[s];
      updates_.resize(static_cast<std::size_t>(update_starts_[supernodes]));
    }
  }
  // The second pass advanced every start to its end; shift back.
  for (std::size_t s = supernodes; s > 0; --s) update_starts_[s] = update_starts_[s - 1];
  update_starts_[0] = 0;
  supernodes_built_ = true;
}

void SparseLdl::factor_supernode(Index s, double regularization, const signed char* signs,
                                 SupernodeWorkspace& ws) {
  const auto us = static_cast<std::size_t>(s);
  const Index f = super_start_[us];
  const Index w = super_start_[us + 1] - f;
  const Index* shared = l_rows_.data() + l_starts_[static_cast<std::size_t>(f)] + (w - 1);
  const Index below = l_starts_[static_cast<std::size_t>(f) + 1] -
                      l_starts_[static_cast<std::size_t>(f)] - (w - 1);
  const Index ld = w + below;
  double* block = blocks_.data() + block_start_[us];
  std::fill(block, block + (block_start_[us + 1] - block_start_[us]), 0.0);
  std::vector<Index>& position = ws.position;
  for (Index c = 0; c < w; ++c) position[static_cast<std::size_t>(f + c)] = c;
  for (Index r = 0; r < below; ++r) position[static_cast<std::size_t>(shared[r])] = w + r;

  // The matrix's own entries of these columns.
  for (Index c = f; c < f + w; ++c) {
    double* column = block + static_cast<std::size_t>(c - f) * static_cast<std::size_t>(ld);
    for (Index p = lower_starts_[static_cast<std::size_t>(c)];
         p < lower_starts_[static_cast<std::size_t>(c) + 1]; ++p) {
      column[position[static_cast<std::size_t>(lower_rows_[static_cast<std::size_t>(p)])]] +=
          a_values_[static_cast<std::size_t>(lower_slots_[static_cast<std::size_t>(p)])];
    }
  }

  // Left-looking: every descendant whose shared rows reach these columns subtracts
  // L_d(I, :) D_d L_d(J, :)^T, J its rows inside this supernode and I those and every row
  // after them - a dense product over the source's contiguous columns. The descendants are
  // done: this supernode became ready when its last child finished.
  for (Index u = update_starts_[us]; u < update_starts_[us + 1]; ++u) {
    const SupernodeUpdate& up = updates_[static_cast<std::size_t>(u)];
    const auto ud = static_cast<std::size_t>(up.source);
    const Index fd = super_start_[ud];
    const Index wd = super_start_[ud + 1] - fd;
    const Index* rows_d = l_rows_.data() + l_starts_[static_cast<std::size_t>(fd)] + (wd - 1);
    const Index below_d = l_starts_[static_cast<std::size_t>(fd) + 1] -
                          l_starts_[static_cast<std::size_t>(fd)] - (wd - 1);
    const Index ld_d = wd + below_d;
    const double* source = blocks_.data() + block_start_[ud];
    for (Index jj = up.begin; jj < up.end; ++jj) {
      const Index target = rows_d[jj] - f;
      const auto length = static_cast<std::size_t>(below_d - jj);
      ws.accumulator.assign(length, 0.0);
      double* accumulator = ws.accumulator.data();
      for (Index k = 0; k < wd; ++k) {
        const double* column = source +
                               static_cast<std::size_t>(k) * static_cast<std::size_t>(ld_d) +
                               static_cast<std::size_t>(wd + jj);
        const double scale = column[0] * d_[static_cast<std::size_t>(fd + k)];
        if (scale == 0.0) continue;
        for (std::size_t ii = 0; ii < length; ++ii) accumulator[ii] += column[ii] * scale;
      }
      double* column = block + static_cast<std::size_t>(target) * static_cast<std::size_t>(ld);
      for (std::size_t ii = 0; ii < length; ++ii) {
        column[position[static_cast<std::size_t>(rows_d[static_cast<std::size_t>(jj) + ii])]] -=
            accumulator[ii];
      }
      ws.work += length * static_cast<std::size_t>(wd);
    }
  }

  // The dense block, right-looking LDL^T without interchanges, under the scalar pivot rule.
  for (Index k = 0; k < w; ++k) {
    double* column = block + static_cast<std::size_t>(k) * static_cast<std::size_t>(ld);
    double pivot = column[k];
    const double sign =
        signs == nullptr
            ? 1.0
            : static_cast<double>(
                  signs[static_cast<std::size_t>(perm_[static_cast<std::size_t>(f + k)])]);
    if (!(sign * pivot > regularization)) {
      pivot = sign * regularization;
      ++ws.regularized;
    }
    d_[static_cast<std::size_t>(f + k)] = pivot;
    ws.smallest = std::min(ws.smallest, sign * pivot);
    ws.largest = std::max(ws.largest, sign * pivot);
    column[k] = 1.0;
    // Columns k+1 .. w-1 of the block lose L(:, k) d_k L(j, k); y = d_k L(j, k) is the
    // entry before scaling.
    for (Index j = k + 1; j < w; ++j) {
      const double y = column[j];
      if (y == 0.0) continue;
      double* target = block + static_cast<std::size_t>(j) * static_cast<std::size_t>(ld);
      for (Index i = j; i < ld; ++i) target[i] -= column[i] * (y / pivot);
    }
    for (Index i = k + 1; i < ld; ++i) column[i] /= pivot;
    ws.work += static_cast<std::size_t>(ld) * static_cast<std::size_t>(w - k);
    // Back into the scalar layout: column f + k of L is its block column below the
    // diagonal, contiguous and in the order the symbolic pattern lists its rows.
    std::copy(column + k + 1, column + ld,
              l_values_.begin() + l_starts_[static_cast<std::size_t>(f + k)]);
  }
}

bool SparseLdl::factorize_supernodal(double regularization, const signed char* signs,
                                     const ShouldStop& should_stop) {
  if (!supernodes_built_) build_supernodes();
  const auto supernodes = super_start_.size() - 1;
  if (blocks_.size() != block_start_[supernodes]) blocks_.resize(block_start_[supernodes]);
  regularized_ = 0;
  smallest_pivot_ = n_ == 0 ? 0.0 : std::numeric_limits<double>::infinity();
  largest_pivot_ = 0.0;
  if (supernodes == 0) return true;

  int threads = threads_;
#ifdef SANKHYA_HAVE_OPENMP
  if (threads <= 0) threads = omp_get_max_threads();
#endif
  threads = std::max(1, std::min(threads, static_cast<int>(supernodes)));

  // The ready queue over the supernodal elimination tree (Rothberg & Gupta 1994): a
  // supernode enters when its last child is done; leaves are in from the start, ascending.
  // `remaining` is decremented by whichever thread finishes a child, so the counts of one
  // parent are touched by several threads at once; the last decrement hands the parent over,
  // and its acquire of the children's releases is what makes their blocks visible.
  std::vector<std::atomic<Index>> remaining(supernodes);
  for (std::size_t s = 0; s < supernodes; ++s) {
    if (super_parent_[s] >= 0) ++remaining[static_cast<std::size_t>(super_parent_[s])];
  }
  std::vector<Index> leaves;
  for (std::size_t s = 0; s < supernodes; ++s) {
    if (remaining[s] == 0) leaves.push_back(static_cast<Index>(s));
  }
  std::mutex mutex;
  std::condition_variable wake;
  std::size_t next = 0;  // leaves[next] is the first not yet taken, under the mutex
  std::atomic<std::size_t> finished{0};
  std::atomic<bool> abandoned{false};
  std::vector<SupernodeWorkspace> workspaces(static_cast<std::size_t>(threads));

  const auto worker = [&](SupernodeWorkspace& ws) {
    ws.position.assign(static_cast<std::size_t>(n_), 0);
    Index s = -1;
    for (;;) {
      if (abandoned.load()) return;
      if (s < 0) {
        std::unique_lock<std::mutex> lock(mutex);
        wake.wait(lock, [&] {
          return next < leaves.size() || finished.load() == supernodes || abandoned.load();
        });
        if (abandoned.load() || next >= leaves.size()) return;
        s = leaves[next++];
      }
      factor_supernode(s, regularization, signs, ws);
      if (ws.work >= (std::size_t{1} << 16)) {
        ws.work = 0;
        if (should_stop && should_stop()) {
          abandoned.store(true);
          std::lock_guard<std::mutex> lock(mutex);
          wake.notify_all();
          return;
        }
      }
      // The parent is next for this thread when this was its last child; otherwise, or at
      // a root, back to the queue for another subtree.
      const Index parent = super_parent_[static_cast<std::size_t>(s)];
      s = parent >= 0 && remaining[static_cast<std::size_t>(parent)].fetch_sub(1) == 1 ? parent
                                                                                       : -1;
      if (finished.fetch_add(1) + 1 == supernodes) {
        std::lock_guard<std::mutex> lock(mutex);
        wake.notify_all();
      }
    }
  };

  std::vector<std::thread> pool;
  pool.reserve(static_cast<std::size_t>(threads) - 1);
  for (int t = 1; t < threads; ++t) {
    pool.emplace_back([&, t] { worker(workspaces[static_cast<std::size_t>(t)]); });
  }
  worker(workspaces[0]);
  for (std::thread& thread : pool) thread.join();

  for (const SupernodeWorkspace& ws : workspaces) {
    regularized_ += ws.regularized;
    smallest_pivot_ = std::min(smallest_pivot_, ws.smallest);
    largest_pivot_ = std::max(largest_pivot_, ws.largest);
  }
  if (abandoned.load()) {
    stopped_early_ = true;
    return false;
  }
  return true;
}

}  // namespace sankhya
