// SPDX-License-Identifier: Apache-2.0
// SANKHYA - supernodal numeric factorization for the sparse LDL^T (#470), first slice.
//
// References:
//   Liu, Ng & Peyton, "On finding supernodes for sparse matrix computations", SIAM J. Matrix
//     Anal. Appl. 14 (1993) - fundamental supernodes from the elimination tree and the
//     column counts.
//   Ng & Peyton, "Block sparse Cholesky algorithms on advanced uniprocessor computers", SIAM
//     J. Sci. Comput. 14 (1993) - the left-looking supernodal factorization: each supernode
//     gathers the updates of the supernodes below it in the tree as dense products, then
//     factors its own dense block.
//   Liu, "The role of elimination trees in sparse factorization", SIAM J. Matrix Anal. Appl.
//     11 (1990); Davis, "Direct Methods for Sparse Linear Systems", SIAM (2006), ch. 4.
//
// WHAT THIS SLICE IS. The symbolic analysis (ordering, elimination tree, pattern of L) is the
// scalar path's, unchanged. The numeric factorization works on supernodes - runs of columns
// with nested patterns, whose rows below the diagonal block are one shared list - so every
// update and every column operation is a dense loop over contiguous memory rather than a
// scatter through the pattern. The result is written back into the scalar layout (l_values_,
// d_), so solve() and every caller see exactly the factor they always saw, and the scalar
// path stays the oracle the tests compare with. Sequential; the dense kernels are plain
// loops written here, no vendor BLAS (#470).
//
// THE PIVOT RULE IS THE SCALAR ONE: a pivot not above `regularization` is replaced by it
// (Altman & Gondzio 1999), counted, and the factorization goes on.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "la/ldl.hpp"

namespace sankhya {

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
  const auto supernodes = super_start_.size() - 1;
  block_start_.assign(supernodes + 1, 0);
  for (std::size_t s = 0; s < supernodes; ++s) {
    const Index f = super_start_[s];
    const auto w = static_cast<std::size_t>(super_start_[s + 1] - f);
    const auto rows = static_cast<std::size_t>(count(f)) + 1;  // f's own row, then its pattern
    block_start_[s + 1] = block_start_[s] + rows * w;
  }
  supernodes_built_ = true;
}

bool SparseLdl::factorize_supernodal(double regularization, const ShouldStop& should_stop) {
  if (!supernodes_built_) build_supernodes();
  const Index n = n_;
  const auto supernodes = super_start_.size() - 1;
  blocks_.assign(block_start_[supernodes], 0.0);
  std::vector<Index> position(static_cast<std::size_t>(n), 0);
  struct Update {
    std::size_t source;  ///< the supernode whose columns update
    Index begin;         ///< first of its shared rows that falls in the target
    Index end;           ///< one past the last
  };
  std::vector<std::vector<Update>> pending(supernodes);
  std::vector<double> accumulator;
  regularized_ = 0;
  smallest_pivot_ = std::numeric_limits<double>::infinity();
  largest_pivot_ = 0.0;
  std::size_t work = 0;

  for (std::size_t s = 0; s < supernodes; ++s) {
    const Index f = super_start_[s];
    const Index w = super_start_[s + 1] - f;
    const Index* shared = l_rows_.data() + l_starts_[static_cast<std::size_t>(f)] + (w - 1);
    const Index below = l_starts_[static_cast<std::size_t>(f) + 1] -
                        l_starts_[static_cast<std::size_t>(f)] - (w - 1);
    const Index ld = w + below;
    double* block = blocks_.data() + block_start_[s];
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

    // Left-looking: every earlier supernode whose shared rows reach these columns subtracts
    // L_d(I, :) D_d L_d(J, :)^T, J its rows inside this supernode and I those and every
    // row after them - a dense product over the source's contiguous columns.
    for (const Update& u : pending[s]) {
      const Index fd = super_start_[u.source];
      const Index wd = super_start_[u.source + 1] - fd;
      const Index* rows_d = l_rows_.data() + l_starts_[static_cast<std::size_t>(fd)] + (wd - 1);
      const Index below_d = l_starts_[static_cast<std::size_t>(fd) + 1] -
                            l_starts_[static_cast<std::size_t>(fd)] - (wd - 1);
      const Index ld_d = wd + below_d;
      const double* source = blocks_.data() + block_start_[u.source];
      for (Index jj = u.begin; jj < u.end; ++jj) {
        const Index target = rows_d[jj] - f;
        const auto length = static_cast<std::size_t>(below_d - jj);
        accumulator.assign(length, 0.0);
        for (Index k = 0; k < wd; ++k) {
          const double* column = source +
                                 static_cast<std::size_t>(k) * static_cast<std::size_t>(ld_d) +
                                 static_cast<std::size_t>(wd + jj);
          const double scale = column[0] * d_[static_cast<std::size_t>(fd + k)];
          if (scale == 0.0) continue;
          for (std::size_t ii = 0; ii < length; ++ii) accumulator[ii] += column[ii] * scale;
        }
        double* column =
            block + static_cast<std::size_t>(target) * static_cast<std::size_t>(ld);
        for (std::size_t ii = 0; ii < length; ++ii) {
          column[position[static_cast<std::size_t>(
              rows_d[static_cast<std::size_t>(jj) + ii])]] -= accumulator[ii];
        }
        work += length * static_cast<std::size_t>(wd);
      }
    }

    // The dense block, right-looking LDL^T without interchanges, under the scalar pivot rule.
    for (Index k = 0; k < w; ++k) {
      double* column = block + static_cast<std::size_t>(k) * static_cast<std::size_t>(ld);
      double pivot = column[k];
      if (!(pivot > regularization)) {
        pivot = regularization;
        ++regularized_;
      }
      d_[static_cast<std::size_t>(f + k)] = pivot;
      smallest_pivot_ = std::min(smallest_pivot_, pivot);
      largest_pivot_ = std::max(largest_pivot_, pivot);
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
      work += static_cast<std::size_t>(ld) * static_cast<std::size_t>(w - k);
    }

    // Hand this supernode's shared rows to the supernodes they fall in, run by run.
    for (Index r = 0; r < below;) {
      const auto target =
          static_cast<std::size_t>(super_of_[static_cast<std::size_t>(shared[r])]);
      Index e = r;
      while (e < below &&
             static_cast<std::size_t>(super_of_[static_cast<std::size_t>(shared[e])]) == target)
        ++e;
      pending[target].push_back({s, r, e});
      r = e;
    }
    std::vector<Update>().swap(pending[s]);

    if (work >= (std::size_t{1} << 16)) {
      work = 0;
      if (should_stop && should_stop()) {
        stopped_early_ = true;
        return false;
      }
    }
  }

  // Back into the scalar layout: column c of L is its block column below the diagonal,
  // contiguous and in the order the symbolic pattern lists its rows.
  for (std::size_t s = 0; s < supernodes; ++s) {
    const Index f = super_start_[s];
    const Index w = super_start_[s + 1] - f;
    const Index ld = static_cast<Index>((block_start_[s + 1] - block_start_[s]) /
                                        static_cast<std::size_t>(w));
    const double* block = blocks_.data() + block_start_[s];
    for (Index c = 0; c < w; ++c) {
      const double* column = block + static_cast<std::size_t>(c) * static_cast<std::size_t>(ld);
      std::copy(column + c + 1, column + ld,
                l_values_.begin() + l_starts_[static_cast<std::size_t>(f + c)]);
    }
  }
  if (n == 0) smallest_pivot_ = 0.0;
  return true;
}

}  // namespace sankhya
