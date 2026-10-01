// SPDX-License-Identifier: Apache-2.0
// SANKHYA - hyper-sparse FTRAN and BTRAN on the product-form factors (#464).
//
// References
//   Gilbert, J.R. and Peierls, T. (1988), "Sparse partial pivoting in time proportional to
//     arithmetic operations", SIAM J. Sci. Stat. Comput. 9(5), 862-874 - the nonzero
//     pattern of a triangular solve's result is the set of nodes reachable, in the graph of
//     the triangular factor, from the right-hand side's nonzeros.
//   Hall, J.A.J. and McKinnon, K.I.M. (2005), "Hyper-sparsity in the revised simplex method
//     and how to exploit it", Computational Optimization and Applications 32, 259-283 - that
//     the simplex's solves are mostly of this kind, and the switch to the full loops when
//     the result is predicted to be dense.
//
// WHAT THE FULL LOOPS COST. solve() already skips a step whose value is zero, but it still
// visits all m steps of L, gathers all m entries into step order, visits all m steps of U
// and scatters all m back; solve_transpose() likewise. On a basis of 100,000 rows whose
// entering column reaches 40 steps, that is four passes of 100,000 to do the work of 40.
//
// WHAT THIS DOES INSTEAD. One pass over the right-hand side finds its nonzeros; a search
// from them over the graph of the first factor (step k -> the steps its entries feed)
// collects every step the result of that factor can be nonzero at, and a second search from
// those over the graph of the second factor does the same for the whole solve. The numeric
// passes then run over those steps only.
//
// THE SAME ARITHMETIC, TO THE BIT. Gilbert and Peierls take the reached steps in a
// topological order of the search, which is a valid elimination order but not the one the
// full loops use, and floating-point sums in a different order round differently - so the
// simplex would take different pivots and "the same iteration counts" would be a hope. The
// reached steps are sorted instead (increasing where the full loop increases, decreasing
// where it decreases), which costs r log r for r reached steps and makes every entry receive
// the same updates in the same order as in the full loop. The steps not reached hold exact
// zeros there, and a zero pushes nothing in either form. The one difference is the sign of
// a zero: the full back-substitution divides an untouched 0 by its pivot and can leave -0.0
// where this leaves +0.0, which compares equal to it everywhere.
//
// THE SWITCH. A right-hand side with more than kHyperSparseDensity * m nonzeros goes to the
// full loops directly, and so does one whose reach passes that size - the search stops
// there, having written nothing but its own marks, so abandoning it costs at most the work
// the numeric pass over those steps would have done. Forrest-Tomlin factors (#279) keep
// the full loops: their U is held in a different structure.

#include "lu.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "sankhya/tolerances.hpp"

namespace sankhya {
namespace {

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

void clear_marks(std::vector<char>& mark, const std::vector<Index>& list) {
  for (const Index k : list) mark[static_cast<std::size_t>(k)] = 0;
}

}  // namespace

void SparseLu::build_step_inverses() {
  const auto m = static_cast<std::size_t>(m_);
  step_of_row_.assign(m, -1);
  step_of_position_.assign(m, -1);
  for (std::size_t k = 0; k < m; ++k) {
    step_of_row_[static_cast<std::size_t>(pivot_row_[k])] = static_cast<Index>(k);
    step_of_position_[static_cast<std::size_t>(pivot_col_[k])] = static_cast<Index>(k);
  }
  hs_mark_.assign(m, 0);
  hs_first_.clear();
  hs_second_.clear();
}

void SparseLu::record_density(std::array<std::int64_t, 5>& bins, Index nonzeros, Index m) {
  const double density =
      static_cast<double>(nonzeros) / static_cast<double>(std::max<Index>(m, 1));
  std::size_t bin = 0;
  while (bin < LuSolveStats::kBinEdges.size() && density >= LuSolveStats::kBinEdges[bin]) ++bin;
  ++bins[bin];
}

bool SparseLu::solve_hyper(double* b) const {
  const auto limit =
      static_cast<std::size_t>(tol::kHyperSparseDensity * static_cast<double>(m_));

  // SYMBOLIC. The seeds are the steps that retire the right-hand side's nonzero rows; L's
  // step k feeds the steps retiring the rows its multipliers touch.
  hs_first_.clear();
  for (Index i = 0; i < m_; ++i) {
    if (b[static_cast<std::size_t>(i)] == 0.0) continue;
    const Index step = step_of_row_[static_cast<std::size_t>(i)];
    hs_mark_[static_cast<std::size_t>(step)] = 1;
    hs_first_.push_back(step);
    if (hs_first_.size() > limit) {
      clear_marks(hs_mark_, hs_first_);
      return false;
    }
  }
  bool small = close_reach(hs_mark_, hs_first_, limit, [&](Index k, auto&& visit) {
    const auto uk = static_cast<std::size_t>(k);
    for (Index p = l_start_[uk]; p < l_start_[uk + 1]; ++p) {
      visit(step_of_row_[static_cast<std::size_t>(l_rows_[static_cast<std::size_t>(p)])]);
    }
  });
  // U's column-wise step j feeds the earlier steps with an entry above its diagonal. Its
  // reach from L's contains L's, whose marks are still set.
  hs_second_ = hs_first_;
  if (small) {
    small = close_reach(hs_mark_, hs_second_, limit, [&](Index j, auto&& visit) {
      const auto uj = static_cast<std::size_t>(j);
      for (Index p = uc_start_[uj]; p < uc_start_[uj + 1]; ++p) {
        visit(uc_steps_[static_cast<std::size_t>(p)]);
      }
    });
  }
  clear_marks(hs_mark_, hs_second_);
  if (!small) return false;

  // NUMERIC, L: forward_l() over the reached steps, in increasing step order.
  std::sort(hs_first_.begin(), hs_first_.end());
  for (const Index k : hs_first_) {
    const auto uk = static_cast<std::size_t>(k);
    const double pivot_component = b[static_cast<std::size_t>(pivot_row_[uk])];
    if (pivot_component == 0.0) continue;
    for (Index p = l_start_[uk]; p < l_start_[uk + 1]; ++p) {
      const auto up = static_cast<std::size_t>(p);
      b[static_cast<std::size_t>(l_rows_[up])] -= l_values_[up] * pivot_component;
    }
  }
  // Into step order for U. The rows L reached are the only nonzeros of b; they are cleared
  // so that b holds exactly the result once the reached positions are written back.
  for (const Index k : hs_second_) {
    const auto uk = static_cast<std::size_t>(k);
    work_[uk] = b[static_cast<std::size_t>(pivot_row_[uk])];
  }
  for (const Index k : hs_first_)
    b[static_cast<std::size_t>(pivot_row_[static_cast<std::size_t>(k)])] = 0.0;

  // NUMERIC, U: the column-wise back-substitution over the reached steps, decreasing.
  std::sort(hs_second_.begin(), hs_second_.end(), [](Index x, Index y) { return x > y; });
  Index nonzeros = 0;
  for (const Index k : hs_second_) {
    const auto uk = static_cast<std::size_t>(k);
    const double value = work_[uk] / pivot_value_[uk];
    work_[uk] = value;
    if (value == 0.0) continue;
    ++nonzeros;
    for (Index p = uc_start_[uk]; p < uc_start_[uk + 1]; ++p) {
      const auto up = static_cast<std::size_t>(p);
      work_[static_cast<std::size_t>(uc_steps_[up])] -= uc_values_[up] * value;
    }
  }
  for (const Index k : hs_second_) {
    const auto uk = static_cast<std::size_t>(k);
    b[static_cast<std::size_t>(pivot_col_[uk])] = work_[uk];
  }
  record_density(stats_.ftran_bins, nonzeros, m_);
  ++stats_.ftran_hyper;
  return true;
}

bool SparseLu::solve_transpose_hyper(double* b) const {
  const auto limit =
      static_cast<std::size_t>(tol::kHyperSparseDensity * static_cast<double>(m_));

  // SYMBOLIC. The seeds are the steps that pivoted on the right-hand side's nonzero basis
  // positions; U's row-wise step k feeds the later steps it has an entry at.
  hs_first_.clear();
  for (Index p = 0; p < m_; ++p) {
    if (b[static_cast<std::size_t>(p)] == 0.0) continue;
    const Index step = step_of_position_[static_cast<std::size_t>(p)];
    hs_mark_[static_cast<std::size_t>(step)] = 1;
    hs_first_.push_back(step);
    if (hs_first_.size() > limit) {
      clear_marks(hs_mark_, hs_first_);
      return false;
    }
  }
  bool small = close_reach(hs_mark_, hs_first_, limit, [&](Index k, auto&& visit) {
    const auto uk = static_cast<std::size_t>(k);
    for (Index p = u_start_[uk]; p < u_start_[uk + 1]; ++p) {
      const Index step = u_steps_[static_cast<std::size_t>(p)];
      if (step > k) visit(step);  // the pivot entry itself is not an edge
    }
  });
  // The row-wise L feeds, from step k, the earlier steps holding a multiplier on its row.
  hs_second_ = hs_first_;
  if (small) {
    small = close_reach(hs_mark_, hs_second_, limit, [&](Index k, auto&& visit) {
      const auto uk = static_cast<std::size_t>(k);
      for (Index p = lr_start_[uk]; p < lr_start_[uk + 1]; ++p) {
        visit(lr_steps_[static_cast<std::size_t>(p)]);
      }
    });
  }
  clear_marks(hs_mark_, hs_second_);
  if (!small) return false;

  // Into step order: every reached step starts at zero, then U's reach takes b's values
  // (zero at the steps that are not seeds). The seeds' positions are the only nonzeros of
  // b; they are cleared so that b holds exactly the result once the rows are written back.
  std::vector<double>& z = work_;
  for (const Index k : hs_second_) z[static_cast<std::size_t>(k)] = 0.0;
  for (const Index k : hs_first_) {
    const auto uk = static_cast<std::size_t>(k);
    z[uk] = b[static_cast<std::size_t>(pivot_col_[uk])];
  }
  for (const Index k : hs_first_)
    b[static_cast<std::size_t>(pivot_col_[static_cast<std::size_t>(k)])] = 0.0;

  // NUMERIC, U^T: forward_u_transposed() over the reached steps, increasing.
  std::sort(hs_first_.begin(), hs_first_.end());
  for (const Index k : hs_first_) {
    const auto uk = static_cast<std::size_t>(k);
    const double value = z[uk] / pivot_value_[uk];
    z[uk] = value;
    if (value == 0.0) continue;
    for (Index p = u_start_[uk]; p < u_start_[uk + 1]; ++p) {
      const auto up = static_cast<std::size_t>(p);
      const Index step = u_steps_[up];
      if (step <= k) continue;
      z[static_cast<std::size_t>(step)] -= u_values_[up] * value;
    }
  }
  // NUMERIC, L^T: apply_transposed_l() over the reached steps, decreasing.
  std::sort(hs_second_.begin(), hs_second_.end(), [](Index x, Index y) { return x > y; });
  Index nonzeros = 0;
  for (const Index k : hs_second_) {
    const auto uk = static_cast<std::size_t>(k);
    const double value = z[uk];
    if (value == 0.0) continue;
    for (Index p = lr_start_[uk]; p < lr_start_[uk + 1]; ++p) {
      const auto up = static_cast<std::size_t>(p);
      z[static_cast<std::size_t>(lr_steps_[up])] -= lr_values_[up] * value;
    }
  }
  for (const Index k : hs_second_) {
    const auto uk = static_cast<std::size_t>(k);
    const double value = z[uk];
    nonzeros += value != 0.0 ? 1 : 0;
    b[static_cast<std::size_t>(pivot_row_[uk])] = value;
  }
  record_density(stats_.btran_bins, nonzeros, m_);
  ++stats_.btran_hyper;
  return true;
}

}  // namespace sankhya
