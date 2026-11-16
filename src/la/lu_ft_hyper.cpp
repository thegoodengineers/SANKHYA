// SPDX-License-Identifier: Apache-2.0
// SANKHYA - hyper-sparse FTRAN and BTRAN over an active Forrest-Tomlin factorization, and
// the update's own partial BTRAN (#279, #1042).
//
// References
//   Forrest, J.J.H. and Tomlin, J.A. (1972), "Updated triangular factors of the basis to
//     maintain sparsity in the product form simplex method", Math. Programming 2, 263-278.
//   Gilbert, J.R. and Peierls, T. (1988), "Sparse partial pivoting in time proportional to
//     arithmetic operations", SIAM J. Sci. Stat. Comput. 9(5), 862-874.
//   Hall, J.A.J. and McKinnon, K.I.M. (2005), "Hyper-sparsity in the revised simplex method
//     and how to exploit it", Computational Optimization and Applications 32, 259-283 -
//     each vector of a hyper-sparse iteration is kept with its index list.
//
// The solve has a third stage between L and U, the row etas, and U is the folded one in
// ft_row_/ft_col_ with its own position order. ft_solve_hyper() and
// ft_solve_transpose_hyper() add the row etas to the reach - a row eta writes its pivot step
// once any step it reads is reached, taken in the order the full loop applies them - and
// sort the U stage by position, not by step. The marks stay set through the row etas'
// numeric pass, because those read entries the reach may not include, and an unreached
// entry must count as the full loop's zero. As in lu_hyper.cpp, the reached steps are taken
// in the full loops' order, so every result is the full loops' to the bit.
//
// With a `support` the seeds are read off it and the result's support is returned, so the
// dual simplex keeps its index lists under this update as under the product form (#1259).
// solve_entering()'s spike is kept off the reach (keep_spike()), so the entering column
// takes this path too.

#include <algorithm>
#include <cstddef>
#include <vector>

#include "sankhya/tolerances.hpp"

#include "lu.hpp"
#include "lu_reach.hpp"

namespace sankhya {
using lu_reach::clear_marks;
using lu_reach::close_reach;

bool SparseLu::seed_reach(const double* b, const std::vector<Index>* support,
                          const std::vector<Index>& step_of, std::size_t limit) const {
  // A repeated or zero entry seeds nothing; hs_first_ ends up the same set either way.
  const auto seed = [&](Index i) {
    if (b[static_cast<std::size_t>(i)] == 0.0) return true;
    const auto step = static_cast<std::size_t>(step_of[static_cast<std::size_t>(i)]);
    if (hs_mark_[step] != 0) return true;
    hs_mark_[step] = 1;
    hs_first_.push_back(static_cast<Index>(step));
    return hs_first_.size() <= limit;
  };
  if (support != nullptr) {
    for (const Index i : *support) {
      if (!seed(i)) return false;
    }
    return true;
  }
  for (Index i = 0; i < m_; ++i) {
    if (!seed(i)) return false;
  }
  return true;
}

bool SparseLu::ft_btran_reach(Index step) const {
  // ft_btran_unit()'s pattern: from `step`, a step reaches the steps its own row references
  // (ft_row_), every one at a later position. Sorted by position, the order the full loop
  // visits them in. hs_first_ is free here: an update runs between solves, never inside one.
  if (!hyper_sparse_) return false;
  const auto limit =
      static_cast<std::size_t>(tol::kHyperSparseDensity * static_cast<double>(m_));
  hs_first_.clear();
  hs_mark_[static_cast<std::size_t>(step)] = 1;
  hs_first_.push_back(step);
  const bool small = close_reach(hs_mark_, hs_first_, limit, [&](Index s, auto&& visit) {
    for (Index p = ft_row_.begin(s); p < ft_row_.end(s); ++p) visit(ft_row_.index(p));
  });
  clear_marks(hs_mark_, hs_first_);
  if (!small) return false;
  std::sort(hs_first_.begin(), hs_first_.end(), [&](Index x, Index y) {
    return ft_position_[static_cast<std::size_t>(x)] <
           ft_position_[static_cast<std::size_t>(y)];
  });
  return true;
}

bool SparseLu::ft_solve_hyper(double* b, std::vector<Index>* support) const {
  const auto limit =
      static_cast<std::size_t>(tol::kHyperSparseDensity * static_cast<double>(m_));
  const auto give_up = [&](const std::vector<Index>& list) {
    clear_marks(hs_mark_, list);
    return false;
  };

  // SYMBOLIC. L's reach from the right-hand side's rows, as solve_hyper() takes it: read
  // off `support` when the caller has it, else off all of b - the same seeds either way.
  hs_first_.clear();
  if (!seed_reach(b, support, step_of_row_, limit)) return give_up(hs_first_);
  if (!close_reach(hs_mark_, hs_first_, limit, [&](Index k, auto&& visit) {
        const auto uk = static_cast<std::size_t>(k);
        for (Index p = l_start_[uk]; p < l_start_[uk + 1]; ++p) {
          visit(step_of_row_[static_cast<std::size_t>(l_rows_[static_cast<std::size_t>(p)])]);
        }
      })) {
    return give_up(hs_first_);
  }
  // The row etas, oldest first: eta k writes its pivot step once a step it reads is reached.
  hs_second_ = hs_first_;
  const auto etas = static_cast<Index>(ft_reta_pivot_step_.size());
  for (Index k = 0; k < etas; ++k) {
    const auto uk = static_cast<std::size_t>(k);
    const Index pivot_step = ft_reta_pivot_step_[uk];
    if (hs_mark_[static_cast<std::size_t>(pivot_step)] != 0) continue;
    for (Index t = ft_reta_start_[uk]; t < ft_reta_start_[uk + 1]; ++t) {
      if (hs_mark_[static_cast<std::size_t>(ft_reta_steps_[static_cast<std::size_t>(t)])] ==
          0) {
        continue;
      }
      hs_mark_[static_cast<std::size_t>(pivot_step)] = 1;
      hs_second_.push_back(pivot_step);
      break;
    }
  }
  if (hs_second_.size() > limit) return give_up(hs_second_);
  // The folded U: a step pushes into the steps whose rows reference it (ft_col_).
  if (!close_reach(hs_mark_, hs_second_, limit, [&](Index s, auto&& visit) {
        for (Index p = ft_col_.begin(s); p < ft_col_.end(s); ++p) visit(ft_col_.index(p));
      })) {
    return give_up(hs_second_);
  }

  // NUMERIC, L: forward_l() over L's reach, increasing step.
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
  // Into step order over the whole reach (zero where L did not reach), b cleared behind.
  for (const Index s : hs_second_) {
    const auto us = static_cast<std::size_t>(s);
    work_[us] = b[static_cast<std::size_t>(pivot_row_[us])];
  }
  for (const Index k : hs_first_) {
    b[static_cast<std::size_t>(pivot_row_[static_cast<std::size_t>(k)])] = 0.0;
  }
  // NUMERIC, row etas: ft_apply_retas() reading reached steps only; an unreached one holds
  // the full loop's zero, whose term changes nothing.
  for (Index k = 0; k < etas; ++k) {
    const auto uk = static_cast<std::size_t>(k);
    const auto pivot_step = static_cast<std::size_t>(ft_reta_pivot_step_[uk]);
    if (hs_mark_[pivot_step] == 0) continue;
    double dot = 0.0;
    for (Index t = ft_reta_start_[uk]; t < ft_reta_start_[uk + 1]; ++t) {
      const auto ut = static_cast<std::size_t>(t);
      const auto step = static_cast<std::size_t>(ft_reta_steps_[ut]);
      if (hs_mark_[step] == 0) continue;
      dot += ft_reta_values_[ut] * work_[step];
    }
    work_[pivot_step] -= dot;
  }
  clear_marks(hs_mark_, hs_second_);
  // solve_entering(): work_ is the spike over the reach, the full loops' zero elsewhere.
  if (keep_spike_) keep_spike(&hs_second_);
  // NUMERIC, U: ft_back_substitute() over the reach, decreasing position.
  std::sort(hs_second_.begin(), hs_second_.end(), [&](Index x, Index y) {
    return ft_position_[static_cast<std::size_t>(x)] >
           ft_position_[static_cast<std::size_t>(y)];
  });
  Index nonzeros = 0;
  for (const Index s : hs_second_) {
    const auto us = static_cast<std::size_t>(s);
    const double value = work_[us] / ft_diag_[us];
    work_[us] = value;
    if (value == 0.0) continue;
    ++nonzeros;
    for (Index p = ft_col_.begin(s); p < ft_col_.end(s); ++p) {
      work_[static_cast<std::size_t>(ft_col_.index(p))] -= ft_col_.value(p) * value;
    }
  }
  for (const Index s : hs_second_) {
    const auto us = static_cast<std::size_t>(s);
    b[static_cast<std::size_t>(pivot_col_[us])] = work_[us];
  }
  // Every row L reached was cleared above, so these positions are all b can be nonzero at.
  if (support != nullptr) {
    support->clear();
    for (const Index s : hs_second_)
      support->push_back(pivot_col_[static_cast<std::size_t>(s)]);
  }
  record_density(stats_.ftran_bins, nonzeros, m_);
  ++stats_.ftran_hyper;
  return true;
}

bool SparseLu::ft_solve_transpose_hyper(double* b, std::vector<Index>* support) const {
  const auto limit =
      static_cast<std::size_t>(tol::kHyperSparseDensity * static_cast<double>(m_));
  const auto give_up = [&](const std::vector<Index>& list) {
    clear_marks(hs_mark_, list);
    return false;
  };

  // SYMBOLIC. The seeds are the steps of the right-hand side's nonzero basis positions
  // (off `support` when given); the folded U's step pushes into the steps its own row
  // references (ft_row_).
  hs_first_.clear();
  if (!seed_reach(b, support, step_of_position_, limit)) return give_up(hs_first_);
  if (!close_reach(hs_mark_, hs_first_, limit, [&](Index s, auto&& visit) {
        for (Index p = ft_row_.begin(s); p < ft_row_.end(s); ++p) visit(ft_row_.index(p));
      })) {
    return give_up(hs_first_);
  }
  // The row etas, newest first: eta k, once its pivot step is reached, writes every step it
  // holds an entry at.
  hs_second_ = hs_first_;
  const auto etas = static_cast<Index>(ft_reta_pivot_step_.size());
  for (Index k = etas - 1; k >= 0; --k) {
    const auto uk = static_cast<std::size_t>(k);
    if (hs_mark_[static_cast<std::size_t>(ft_reta_pivot_step_[uk])] == 0) continue;
    for (Index t = ft_reta_start_[uk]; t < ft_reta_start_[uk + 1]; ++t) {
      const auto step = static_cast<std::size_t>(ft_reta_steps_[static_cast<std::size_t>(t)]);
      if (hs_mark_[step] != 0) continue;
      hs_mark_[step] = 1;
      hs_second_.push_back(static_cast<Index>(step));
    }
    if (hs_second_.size() > limit) return give_up(hs_second_);
  }
  // L^T: the row-wise L feeds, from step k, the earlier steps holding a multiplier on its row.
  if (!close_reach(hs_mark_, hs_second_, limit, [&](Index k, auto&& visit) {
        const auto uk = static_cast<std::size_t>(k);
        for (Index p = lr_start_[uk]; p < lr_start_[uk + 1]; ++p) {
          visit(lr_steps_[static_cast<std::size_t>(p)]);
        }
      })) {
    return give_up(hs_second_);
  }

  // Into step order: zero over the reach, the seeds from b, b cleared behind.
  std::vector<double>& z = work_;
  for (const Index s : hs_second_) z[static_cast<std::size_t>(s)] = 0.0;
  for (const Index s : hs_first_) {
    const auto us = static_cast<std::size_t>(s);
    z[us] = b[static_cast<std::size_t>(pivot_col_[us])];
  }
  for (const Index s : hs_first_) {
    b[static_cast<std::size_t>(pivot_col_[static_cast<std::size_t>(s)])] = 0.0;
  }
  // NUMERIC, U^T: ft_forward_substitute() over U's reach, increasing position.
  std::sort(hs_first_.begin(), hs_first_.end(), [&](Index x, Index y) {
    return ft_position_[static_cast<std::size_t>(x)] <
           ft_position_[static_cast<std::size_t>(y)];
  });
  for (const Index s : hs_first_) {
    const auto us = static_cast<std::size_t>(s);
    const double value = z[us] / ft_diag_[us];
    z[us] = value;
    if (value == 0.0) continue;
    for (Index p = ft_row_.begin(s); p < ft_row_.end(s); ++p) {
      z[static_cast<std::size_t>(ft_row_.index(p))] -= ft_row_.value(p) * value;
    }
  }
  // NUMERIC, row etas: ft_apply_retas_transposed(), newest first; an unreached pivot step
  // holds the full loop's zero and pushes nothing.
  for (Index k = etas - 1; k >= 0; --k) {
    const auto uk = static_cast<std::size_t>(k);
    const auto pivot_step = static_cast<std::size_t>(ft_reta_pivot_step_[uk]);
    if (hs_mark_[pivot_step] == 0) continue;
    const double pivot_value = z[pivot_step];
    if (pivot_value == 0.0) continue;
    for (Index t = ft_reta_start_[uk]; t < ft_reta_start_[uk + 1]; ++t) {
      const auto ut = static_cast<std::size_t>(t);
      z[static_cast<std::size_t>(ft_reta_steps_[ut])] -= ft_reta_values_[ut] * pivot_value;
    }
  }
  clear_marks(hs_mark_, hs_second_);
  // NUMERIC, L^T: apply_transposed_l() over the reach, decreasing step.
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
  // The seeds' positions were cleared above, so these rows are all b can be nonzero at.
  if (support != nullptr) {
    support->clear();
    for (const Index k : hs_second_)
      support->push_back(pivot_row_[static_cast<std::size_t>(k)]);
  }
  record_density(stats_.btran_bins, nonzeros, m_);
  ++stats_.btran_hyper;
  return true;
}

}  // namespace sankhya
