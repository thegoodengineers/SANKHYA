// SPDX-License-Identifier: Apache-2.0
// SANKHYA - parity rows solved over GF(2) (#841). See parity.hpp for the method and its
// citation.

#include "parity.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

#include "sankhya/tolerances.hpp"

namespace sankhya::mip {

namespace {

using Word = std::uint64_t;
constexpr std::size_t kBits = 64;

bool is_integer_value(double v) {
  return std::isfinite(v) && v == std::round(v);
}

/// Odd as an integer; the caller has checked `v` is one.
bool is_odd(double v) {
  return std::fmod(std::fabs(v), 2.0) == 1.0;
}

struct BitRow {
  std::vector<Word> bits;  ///< one bit per parity column, then the rhs bit
  bool get(std::size_t j) const { return ((bits[j / kBits] >> (j % kBits)) & 1U) != 0; }
  void flip(std::size_t j) { bits[j / kBits] ^= Word{1} << (j % kBits); }
  void add(const BitRow& other) {
    for (std::size_t w = 0; w < bits.size(); ++w) bits[w] ^= other.bits[w];
  }
};

}  // namespace

ParityFixings parity_fixings(const Model& model, const std::vector<double>& guide,
                             int max_candidates) {
  ParityFixings out;
  const Index m = model.num_rows();
  const Index n = model.num_cols();
  const auto integer = [&](Index j) {
    return model.col_type[static_cast<std::size_t>(j)] == VarType::kInteger;
  };
  const auto fixed = [&](Index j) {
    const auto u = static_cast<std::size_t>(j);
    return model.col_lower[u] == model.col_upper[u];
  };
  const auto binary = [&](Index j) {
    const auto u = static_cast<std::size_t>(j);
    return model.col_lower[u] == 0.0 && model.col_upper[u] == 1.0;
  };

  // Row-wise view: a parity row is decided by every entry it has.
  std::vector<std::vector<std::pair<Index, double>>> rows(static_cast<std::size_t>(m));
  for (Index j = 0; j < n; ++j) {
    const ColumnView col = model.matrix.column(j);
    for (Index k = 0; k < col.size; ++k)
      rows[static_cast<std::size_t>(col.rows[k])].emplace_back(j, col.values[k]);
  }

  // Which rows are parity rows, their rhs parity, and the binary columns they use.
  std::vector<Index> parity_rows;
  std::vector<std::uint8_t> rhs_odd;
  std::vector<Index> slot(static_cast<std::size_t>(n), -1);
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double b = model.row_lower[u];
    if (b != model.row_upper[u] || !is_integer_value(b) || rows[u].empty()) continue;
    bool ok = true;
    bool odd = is_odd(b);
    bool any_binary = false;
    for (const auto& [j, a] : rows[u]) {
      if (!integer(j) || !is_integer_value(a)) {
        ok = false;
        break;
      }
      if (!is_odd(a)) continue;  // vanishes mod 2, whatever the column's range
      if (fixed(j)) {
        const double v = model.col_lower[static_cast<std::size_t>(j)];
        if (!is_integer_value(v)) {
          ok = false;
          break;
        }
        odd = odd != is_odd(v);
      } else if (binary(j)) {
        any_binary = true;
      } else {
        ok = false;  // an odd general integer adds its own parity, which is not a 0/1 bit
        break;
      }
    }
    if (!ok || !any_binary) continue;
    parity_rows.push_back(i);
    rhs_odd.push_back(odd ? 1 : 0);
    for (const auto& [j, a] : rows[u]) {
      if (is_odd(a) && !fixed(j)) slot[static_cast<std::size_t>(j)] = 0;
    }
  }
  if (parity_rows.empty()) return out;
  for (Index j = 0; j < n; ++j) {
    if (slot[static_cast<std::size_t>(j)] == 0) {
      slot[static_cast<std::size_t>(j)] = static_cast<Index>(out.columns.size());
      out.columns.push_back(j);
    }
  }
  out.parity_rows = static_cast<Index>(parity_rows.size());
  const std::size_t cols = out.columns.size();
  // ponytail: dense elimination, O(rows * cols^2 / 64); a sparse GF(2) elimination when a
  // model with tens of thousands of parity columns turns up.
  if (cols > static_cast<std::size_t>(tol::kParityMaxColumns)) return out;

  const std::size_t words = (cols + 1 + kBits - 1) / kBits;
  std::vector<BitRow> system;
  system.reserve(parity_rows.size());
  for (std::size_t r = 0; r < parity_rows.size(); ++r) {
    BitRow row{std::vector<Word>(words, 0)};
    for (const auto& [j, a] : rows[static_cast<std::size_t>(parity_rows[r])]) {
      if (is_odd(a) && !fixed(j))
        row.flip(static_cast<std::size_t>(slot[static_cast<std::size_t>(j)]));
    }
    if (rhs_odd[r] != 0) row.flip(cols);
    system.push_back(std::move(row));
  }

  // Gauss-Jordan over GF(2): reduced row echelon form, one pivot per independent row.
  std::vector<std::size_t> pivot_of_row;
  std::vector<char> is_pivot(cols, 0);
  std::size_t rank = 0;
  for (std::size_t c = 0; c < cols && rank < system.size(); ++c) {
    std::size_t p = rank;
    while (p < system.size() && !system[p].get(c)) ++p;
    if (p == system.size()) continue;
    std::swap(system[rank], system[p]);
    for (std::size_t r = 0; r < system.size(); ++r) {
      if (r != rank && system[r].get(c)) system[r].add(system[rank]);
    }
    pivot_of_row.push_back(c);
    is_pivot[c] = 1;
    ++rank;
  }
  // A zero row with an odd rhs: 0 = 1, no 0/1 point satisfies the rows.
  for (std::size_t r = rank; r < system.size(); ++r) {
    if (system[r].get(cols)) {
      out.inconsistent = true;
      return out;
    }
  }

  std::vector<std::size_t> free_columns;
  for (std::size_t c = 0; c < cols; ++c) {
    if (is_pivot[c] == 0) free_columns.push_back(c);
  }
  out.null_space_dimension = static_cast<Index>(free_columns.size());

  // The free columns start at the guide's rounding; candidate t flips the free columns whose
  // bit is set in t. 2^k candidates at most.
  std::vector<std::uint8_t> base(cols, 0);
  for (const std::size_t c : free_columns) {
    const auto j = static_cast<std::size_t>(out.columns[c]);
    base[c] = j < guide.size() && guide[j] >= 0.5 ? 1 : 0;
  }
  const std::size_t span =
      free_columns.size() >= 31 ? std::size_t{1} << 30 : std::size_t{1} << free_columns.size();
  const std::size_t total =
      std::min<std::size_t>(span, static_cast<std::size_t>(std::max(0, max_candidates)));
  for (std::size_t t = 0; t < total; ++t) {
    std::vector<std::uint8_t> x = base;
    for (std::size_t f = 0; f < free_columns.size() && f < kBits; ++f) {
      if (((t >> f) & 1U) != 0) x[free_columns[f]] ^= 1U;
    }
    // Each pivot row reads  x_pivot + sum_{free c in row} x_c = rhs.
    for (std::size_t r = 0; r < rank; ++r) {
      std::uint8_t v = system[r].get(cols) ? 1 : 0;
      for (const std::size_t c : free_columns) {
        if (x[c] != 0 && system[r].get(c)) v ^= 1U;
      }
      x[pivot_of_row[r]] = v;
    }
    out.candidates.push_back(std::move(x));
  }
  return out;
}

}  // namespace sankhya::mip
