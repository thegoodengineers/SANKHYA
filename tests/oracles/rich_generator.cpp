// SPDX-License-Identifier: Apache-2.0
// SANKHYA - random general-form models for the oracle fuzz. TESTS ONLY.

#include <cstddef>

#include "oracles/rich_model.hpp"

namespace sankhya::oracle {
namespace {

std::size_t at(Index i) {
  return static_cast<std::size_t>(i);
}

bool happens(std::mt19937_64& rng, double probability) {
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  return unit(rng) < probability;
}

std::int64_t between(std::mt19937_64& rng, std::int64_t low, std::int64_t high) {
  std::uniform_int_distribution<std::int64_t> pick(low, high);
  return pick(rng);
}

/// A nonzero coefficient: an integer, or one divided by a power of two (still exact in
/// binary), or by 10 or 3 (not).
Rational coefficient(std::mt19937_64& rng, const RichConfig& config) {
  std::int64_t value = between(rng, -config.magnitude, config.magnitude);
  if (value == 0) value = 1;
  std::int64_t divisor = 1;
  if (happens(rng, config.dyadic)) {
    divisor = std::int64_t{1} << between(rng, 1, 3);
  } else if (happens(rng, config.decimal)) {
    divisor = happens(rng, 0.5) ? 10 : 3;
  }
  return rat(value, divisor);
}

/// 1/2, 1/4 or 1/3: how far a bound is moved off the integers.
Rational fraction(std::mt19937_64& rng) {
  const std::int64_t pick = between(rng, 0, 2);
  return rat(1, pick == 0 ? 2 : (pick == 1 ? 4 : 3));
}

/// x_c - M y <= l_c with y a new binary that costs something and x_c paid for: y = 0 holds
/// x_c at its lower bound, y = 1 lets it move by M. With M far above the column's range the
/// relaxation wants y = (x_c - l_c) / M, below the integrality tolerance, which is the
/// shape of #1312 and #1319. One link in five has M equal to the column's range instead.
void add_big_m_link(RichModel* m, std::mt19937_64& rng, const RichConfig& config) {
  std::vector<std::size_t> candidates;
  for (std::size_t j = 0; j < at(m->num_cols); ++j) {
    if (m->col_lower[j]) candidates.push_back(j);
  }
  if (candidates.empty()) return;
  const std::size_t c = candidates[static_cast<std::size_t>(
      between(rng, 0, static_cast<std::int64_t>(candidates.size()) - 1))];

  Rational big = rat(between(rng, config.big_m_min, config.big_m_max));
  if (happens(rng, 0.25)) big = big / rat(8);
  if (m->col_upper[c] && *m->col_upper[c] > *m->col_lower[c] && happens(rng, 0.2)) {
    big = *m->col_upper[c] - *m->col_lower[c];
  }
  const std::int64_t toward_better = m->maximize ? 1 : -1;
  m->cost[c] = rat(toward_better * between(rng, 1, 5));

  for (auto& row : m->a) row.push_back(Rational());
  std::vector<Rational> link(at(m->num_cols) + 1);
  link[c] = rat(1);
  link.back() = -big;
  m->a.push_back(std::move(link));
  m->row_lower.emplace_back();
  m->row_upper.emplace_back(*m->col_lower[c]);
  m->col_lower.emplace_back(rat(0));
  m->col_upper.emplace_back(rat(1));
  m->integral.push_back(1);
  m->cost.push_back(rat(-toward_better * between(rng, 1, 6)));
  ++m->num_cols;
  ++m->num_rows;
}

}  // namespace

RichModel random_rich(std::mt19937_64& rng, const RichConfig& config) {
  RichModel m;
  m.num_rows = static_cast<Index>(between(rng, config.min_rows, config.max_rows));
  m.num_cols = static_cast<Index>(between(rng, config.min_cols, config.max_cols));
  const std::size_t rows = at(m.num_rows);
  const std::size_t cols = at(m.num_cols);
  // Constructed, not assigned: see make_skeleton in lp_generator.cpp.
  m.a = std::vector<std::vector<Rational>>(rows, std::vector<Rational>(cols));
  m.row_lower = std::vector<Bound>(rows);
  m.row_upper = std::vector<Bound>(rows);
  m.col_lower = std::vector<Bound>(cols);
  m.col_upper = std::vector<Bound>(cols);
  m.integral = std::vector<char>(cols, 0);
  m.cost = std::vector<Rational>(cols);
  m.maximize = happens(rng, 0.5);
  if (happens(rng, 0.5)) m.offset = rat(between(rng, -40, 40), 2);

  // Columns, and a point inside their bounds for the rows to be built around.
  std::vector<Rational> point(cols);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  for (std::size_t j = 0; j < cols; ++j) {
    const bool binary = happens(rng, config.binary_columns);
    const bool integral = binary || happens(rng, config.integer_columns);
    m.integral[j] = integral ? 1 : 0;
    const double kind = unit(rng);
    if (binary) {
      m.col_lower[j] = rat(0);
      m.col_upper[j] = rat(1);
      point[j] = rat(between(rng, 0, 1));
    } else if (kind < config.free_columns) {
      point[j] = rat(between(rng, -4, 4));
    } else if (kind < config.free_columns + config.lower_only_columns) {
      const std::int64_t lower = between(rng, -config.shift, config.shift);
      m.col_lower[j] = rat(lower);
      point[j] = rat(lower + between(rng, 0, 4));
    } else if (kind <
               config.free_columns + config.lower_only_columns + config.upper_only_columns) {
      const std::int64_t upper = between(rng, -config.shift, config.shift);
      m.col_upper[j] = rat(upper);
      point[j] = rat(upper - between(rng, 0, 4));
    } else {
      const std::int64_t lower = between(rng, -config.shift, config.shift);
      const std::int64_t width = between(rng, 0, config.span);
      m.col_lower[j] = rat(lower);
      m.col_upper[j] = rat(lower + width);
      point[j] = rat(lower + between(rng, 0, width));
    }
    if (binary) continue;
    // Off the integers, and only ever outward: the point stays inside, an integer column
    // keeps the domain it had (what rounding inward leaves), a continuous one gains the
    // fraction.
    if (m.col_lower[j] && happens(rng, config.fractional_bounds)) {
      m.col_lower[j] = *m.col_lower[j] - fraction(rng);
    }
    if (m.col_upper[j] && happens(rng, config.fractional_bounds)) {
      m.col_upper[j] = *m.col_upper[j] + fraction(rng);
    }
    // A continuous column's point need not be an integer.
    if (!integral && happens(rng, 0.3)) {
      const Rational moved = point[j] + rat(1, 2);
      if (!m.col_upper[j] || moved <= *m.col_upper[j]) point[j] = moved;
    }
  }

  const bool planted = happens(rng, config.planted);
  const auto slack = [&] { return happens(rng, 0.5) ? rat(0) : rat(between(rng, 1, 3)); };
  for (std::size_t i = 0; i < rows; ++i) {
    bool any = false;
    for (std::size_t j = 0; j < cols; ++j) {
      if (!happens(rng, config.density)) continue;
      m.a[i][j] = coefficient(rng, config);
      any = true;
    }
    if (!any) {
      m.a[i][static_cast<std::size_t>(between(rng, 0, m.num_cols - 1))] =
          coefficient(rng, config);
    }
    Rational anchor = rat(between(rng, -config.magnitude, config.magnitude));
    if (planted) {
      anchor = Rational();
      for (std::size_t j = 0; j < cols; ++j) anchor += m.a[i][j] * point[j];
    }
    const double kind = unit(rng);
    if (kind < config.equality_rows) {
      m.row_lower[i] = anchor;
      m.row_upper[i] = anchor;
    } else if (kind < config.equality_rows + config.ranged_rows) {
      m.row_lower[i] = anchor - slack();
      m.row_upper[i] = anchor + slack() + rat(1);
    } else if (kind < config.equality_rows + config.ranged_rows + config.upper_rows) {
      m.row_upper[i] = anchor + slack();
    } else {
      m.row_lower[i] = anchor - slack();
    }
  }
  for (std::size_t j = 0; j < cols; ++j) {
    if (!happens(rng, 0.15)) m.cost[j] = coefficient(rng, config);
  }
  for (int k = 0; k < config.big_m_links; ++k) add_big_m_link(&m, rng, config);
  return m;
}

}  // namespace sankhya::oracle
