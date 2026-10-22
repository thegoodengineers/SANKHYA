// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a staircase LP of any size for tests: the shape of a multi-period plan, where
// only the stock joins one period to the next.
#pragma once

#include "sankhya/model.hpp"

namespace sankhya {

/// T periods, each with a stock s_t, a purchase b_t and a run r_t:
///   s_t - s_{t-1} - b_t + r_t = 0,  r_t <= 5,  b_t + r_t >= 1.
/// The stock is the only thing that joins one period to the next.
inline Model staircase(int periods) {
  const Index n = 3 * periods;
  const Index m = 3 * periods;
  Model model;
  model.resize_columns(n);
  model.resize_rows(m);
  model.matrix.reset(m, n);
  for (int t = 0; t < periods; ++t) {
    const Index s = 3 * t, b = 3 * t + 1, r = 3 * t + 2;
    const Index balance = 3 * t, cap = 3 * t + 1, need = 3 * t + 2;
    for (const Index j : {s, b, r}) {
      model.col_lower[static_cast<std::size_t>(j)] = 0.0;
      model.col_upper[static_cast<std::size_t>(j)] = 10.0;
    }
    model.col_cost[static_cast<std::size_t>(s)] = 0.1;
    model.col_cost[static_cast<std::size_t>(b)] = 1.0 + 0.05 * t;
    model.col_cost[static_cast<std::size_t>(r)] = -0.5;
    model.matrix.add_entry(balance, s, 1.0);
    if (t > 0) model.matrix.add_entry(balance, 3 * (t - 1), -1.0);
    model.matrix.add_entry(balance, b, -1.0);
    model.matrix.add_entry(balance, r, 1.0);
    model.row_lower[static_cast<std::size_t>(balance)] = 0.0;
    model.row_upper[static_cast<std::size_t>(balance)] = 0.0;
    model.matrix.add_entry(cap, r, 1.0);
    model.row_upper[static_cast<std::size_t>(cap)] = 5.0;
    model.matrix.add_entry(need, b, 1.0);
    model.matrix.add_entry(need, r, 1.0);
    model.row_lower[static_cast<std::size_t>(need)] = 1.0;
  }
  model.matrix.finalize();
  return model;
}

}  // namespace sankhya
