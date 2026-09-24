// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the size of the normal equations, known before they are built (#467).
// See normal_pattern.hpp for the references and for why.

#include "la/normal_pattern.hpp"

#include <algorithm>
#include <cstddef>

namespace sankhya {

NormalPrediction predict_normal_nonzeros(const SparseMatrix& a, const std::vector<char>& skip,
                                         std::int64_t cap,
                                         const std::function<bool()>& should_stop) {
  const Index m = a.num_rows();
  const Index n = a.num_cols();
  const auto kept = [&](Index j) {
    return skip.empty() || skip[static_cast<std::size_t>(j)] == 0;
  };
  NormalPrediction out;
  const auto rows = static_cast<std::int64_t>(m);

  // Stage 1 and 2: one pass over the column counts. A column with c entries is a clique of
  // c rows, c (c - 1) / 2 strictly-lower entries that nothing can avoid (a lower bound on the
  // whole) and that at worst no other column shares (the sum is an upper bound).
  std::int64_t largest_clique = 0;
  std::int64_t sum_of_cliques = 0;
  for (Index j = 0; j < n; ++j) {
    if (!kept(j)) continue;
    const auto c = static_cast<std::int64_t>(a.column(j).size);
    const std::int64_t strict = c * (c - 1) / 2;
    largest_clique = std::max(largest_clique, strict);
    sum_of_cliques += strict;
  }
  const std::int64_t full_lower = rows * (rows + 1) / 2;
  const std::int64_t lower_bound = rows + largest_clique;
  const std::int64_t upper_bound = std::min(full_lower, rows + sum_of_cliques);
  if (cap >= 0 && lower_bound > cap) {
    out.nonzeros = lower_bound;
    out.over_cap = true;
    return out;
  }
  if (cap < 0 || upper_bound <= cap) {
    out.nonzeros = upper_bound;
    out.exact = lower_bound == upper_bound;
    return out;
  }

  // Stage 3: the exact count. Row i of the lower triangle holds every r >= i that shares a
  // kept column with row i - the pattern normal_equations_lower() emits, with the same
  // marker, and nothing stored. The clock is asked by work done, as the assembly asks it.
  const CsrView by_row(a);
  std::vector<Index> mark(static_cast<std::size_t>(m), -1);
  std::int64_t count = 0;
  std::size_t work = 0;
  constexpr std::size_t kWorkPerCheck = std::size_t{1} << 16;
  for (Index i = 0; i < m; ++i) {
    std::int64_t row_count = 1;  // the diagonal, which D always fills
    mark[static_cast<std::size_t>(i)] = i;
    const ColumnView row = by_row.row(i);
    for (Index p = 0; p < row.size; ++p) {
      const Index j = row.rows[p];  // a CSR view stores COLUMN indices in `rows`
      if (j < 0 || j >= n || !kept(j)) continue;
      const ColumnView column = a.column(j);
      for (Index q = 0; q < column.size; ++q) {
        const Index r = column.rows[q];
        if (r <= i || mark[static_cast<std::size_t>(r)] == i) continue;
        mark[static_cast<std::size_t>(r)] = i;
        ++row_count;
      }
      work += static_cast<std::size_t>(column.size);
      if (work >= kWorkPerCheck) {
        work = 0;
        if (should_stop && should_stop()) {
          out.nonzeros = count;
          out.stopped = true;
          return out;
        }
      }
    }
    count += row_count;
    if (count > cap) {
      out.nonzeros = count;
      out.over_cap = true;
      return out;
    }
  }
  out.nonzeros = count;
  out.exact = true;
  return out;
}

}  // namespace sankhya
