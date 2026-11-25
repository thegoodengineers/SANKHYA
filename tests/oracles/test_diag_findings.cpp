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

TEST(DiagFindings, ParallelRowsDuals) {
  // min -4 x0 + 2 x1: 3 x0 - 4 x1 >= -3, 6 x0 - 8 x1 <= -2; x0 <= 3, x1 free. Optimum -7.
  const Model model = build({{3, -4}, {6, -8}}, {-3, -inf}, {inf, -2}, {-4, 2}, {-inf, -inf},
                            {3, inf}, {false, false});
  run("parallel rows, optimum -7", model,
      {{{"presolve_parallel_rows", "true"}},
       {{"presolve_parallel_rows", "true"}, {"prescale_retry", "false"}},
       {{"presolve_parallel_rows", "true"}, {"engine_race", "true"}, {"deterministic", "true"}},
       {}});
}

TEST(DiagFindings, ImpliedFreeDuals) {
  // min 3 x1: -3/4 x0 - 3/8 x1 - 4 x2 = -57/4, 2 x0 <= 6, -x0 + 4 x2 = 9; x0 >= 3, x1 >= 0.
  const Model b = build({{-0.75, -0.375, -4}, {2, 0, 0}, {-1, 0, 4}}, {-14.25, -inf, 9},
                        {-14.25, 6, 9}, {0, 3, 0}, {3, 0, -inf}, {inf, inf, inf},
                        {false, false, false});
  run("implied free B, optimum 0", b,
      {{{"presolve_implied_free", "true"}},
       {{"presolve_implied_free", "true"}, {"engine_race", "true"}, {"deterministic", "true"}},
       {}});
  // min -2 x3: x2 + x3 = -17/2, 2 x0 >= 4, 5/2 x0 - 5 x1 + x2/2 = 19/2.
  const Model c = build({{0, 0, 1, 1}, {2, 0, 0, 0}, {2.5, -5, 0.5, 0}}, {-8.5, 4, 9.5},
                        {-8.5, inf, 9.5}, {0, 0, 0, -2}, {-inf, -1, -inf, -inf},
                        {2, -1, -1, -3}, {false, false, false, false});
  run("implied free C, optimum 15", c,
      {{{"presolve_implied_free", "true"}},
       {{"presolve_implied_free", "true"}, {"engine_race", "true"}, {"deterministic", "true"}},
       {}});
}

TEST(DiagFindings, DominatedColumnsOnAnUnboundedLp) {
  // min 5 x4: -8 x0 + 2 x3 - 9/2 x4 = 12967/4, -3 x1 + x4 <= -4705/2,
  // -6 x2 - 3/2 x3 + 9 x4 = -10635; x0 <= -498, x1 <= 877, x2 <= 2271, x3 and x4 free.
  const Model model =
      build({{-8, 0, 0, 2, -4.5}, {0, -3, 0, 0, 1}, {0, 0, -6, -1.5, 9}},
            {3241.75, -inf, -10635}, {3241.75, -2352.5, -10635}, {0, 0, 0, 0, 5},
            {-inf, -inf, -inf, -inf, -inf}, {-498, 877, 2271, inf, inf},
            {false, false, false, false, false});
  run("dominated columns, unbounded", model,
      {{{"presolve_dominated_columns", "true"}}, {{"presolve_dominated_columns", "true"}, {"log_level", "debug"}}, {}});
}

}  // namespace
}  // namespace sankhya
