// SPDX-License-Identifier: Apache-2.0
// DIAGNOSTIC, NOT FOR MERGE: the reduced models of the first general-form fuzz run, each
// under the option sets that say which stage is at fault.

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

using OptionList = std::vector<std::pair<std::string, std::string>>;

Model build(const std::vector<std::vector<double>>& rows, const std::vector<double>& row_lower,
            const std::vector<double>& row_upper, const std::vector<double>& cost,
            const std::vector<double>& col_lower, const std::vector<double>& col_upper,
            const std::vector<bool>& integral) {
  Model model;
  const auto n = static_cast<Index>(cost.size());
  const auto m = static_cast<Index>(rows.size());
  model.resize_columns(n);
  model.resize_rows(m);
  model.col_cost = cost;
  model.col_lower = col_lower;
  model.col_upper = col_upper;
  for (Index j = 0; j < n; ++j) {
    if (integral[static_cast<std::size_t>(j)]) {
      model.col_type[static_cast<std::size_t>(j)] = VarType::kInteger;
    }
  }
  model.row_lower = row_lower;
  model.row_upper = row_upper;
  model.matrix.reset(m, n);
  for (Index i = 0; i < m; ++i) {
    for (Index j = 0; j < n; ++j) {
      const double v = rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
      if (v != 0.0) model.matrix.add_entry(i, j, v);
    }
  }
  model.matrix.finalize();
  return model;
}

void run(const char* name, const Model& model, const std::vector<OptionList>& sets) {
  for (const OptionList& set : sets) {
    Options options;
    std::string error;
    std::string label;
    for (const auto& [key, value] : set) {
      ASSERT_TRUE(options.set_from_string(key, value, &error)) << error;
      label += " " + key + "=" + value;
    }
    std::cout << "\n##### " << name << " under" << (label.empty() ? " defaults" : label)
              << " #####" << std::endl;
    const Solution s = solve(model, options);
    std::cout << "##### -> " << to_string(s.status) << " objective " << s.objective << " bound "
              << s.dual_bound << " algorithm " << s.algorithm << "\n##### message: "
              << s.message << "\n##### x:";
    for (const double x : s.col_value) std::cout << " " << x;
    std::cout << "\n##### y:";
    for (const double y : s.row_dual) std::cout << " " << y;
    std::cout << "\n##### d:";
    for (const double d : s.col_dual) std::cout << " " << d;
    std::cout << std::endl;
  }
}

const double inf = kInfinity;

TEST(DiagFindings, NearIntegerCoefficient) {
  // min x + y: (2 + 2^-21) x + 2 y >= 1000 + 400 * 2^-21, x and y integer in [0, 500].
  // x + y = 500 needs x >= 400, and (400, 100) meets the row exactly: the optimum is 500.
  const double delta = 1.0 / 2097152.0;
  const Model model = build({{2.0 + delta, 2.0}}, {1000.0 + 400.0 * delta}, {inf}, {1, 1},
                            {0, 0}, {500, 500}, {true, true});
  run("near-integer coefficient, optimum 500", model,
      {{}, {{"presolve", "false"}}, {{"mip_threads", "1"}}, {{"enable_root_cuts", "false"}}});
  // The same with a third column so the row is not a doubleton anywhere.
  const Model three = build({{2.0 + delta, 2.0, 2.0}}, {1000.0 + 400.0 * delta}, {inf},
                            {1, 1, 1}, {0, 0, 0}, {500, 500, 500}, {true, true, true});
  run("near-integer coefficient, three columns, optimum 500", three,
      {{}, {{"presolve", "false"}}});
}

TEST(DiagFindings, InfeasibleMilpWithAnUnboundedRelaxation) {
  // min -x: 2 y - 2 z = 1, y and z integer in [0, 5], x >= 0 in no row. No integer point;
  // the relaxation is feasible and unbounded.
  const Model model = build({{0, 2, -2}}, {1}, {1}, {-1, 0, 0}, {0, 0, 0}, {inf, 5, 5},
                            {false, true, true});
  run("infeasible MILP, unbounded relaxation (empty column)", model,
      {{}, {{"presolve", "false"}}, {{"mip_threads", "1"}}});
  // The same with x in a row of its own: x - w >= 0, w >= 0 free to grow.
  const Model rowed = build({{0, 2, -2, 0}, {1, 0, 0, -1}}, {1, 0}, {1, inf}, {-1, 0, 0, 0},
                            {0, 0, 0, 0}, {inf, 5, 5, inf}, {false, true, true, false});
  run("infeasible MILP, unbounded relaxation (x in a row)", rowed,
      {{}, {{"presolve", "false"}}, {{"mip_threads", "1"}}});
  // And an unbounded MILP that does have a point, which must stay unbounded: 2 y - 2 z = 2.
  const Model truly = build({{0, 2, -2}}, {2}, {2}, {-1, 0, 0}, {0, 0, 0}, {inf, 5, 5},
                            {false, true, true});
  run("unbounded MILP with a point", truly, {{}, {{"presolve", "false"}}});
}

}  // namespace
}  // namespace sankhya
