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

TEST(DiagFindings, MarginalBoundAsLp) {
  // The model OBBT left of "OBBT one": min 0, 3 x0 - 2 x1 >= 2, 3 x1 + x2 >= 8,
  // x0 in [2 - 2e-7, 7/3], x1 = 2, x2 >= 2 - 2e-7. Feasible (x0 = 2, x2 = 2).
  const Model lp = build({{3, -2, 0}, {0, 3, 1}}, {2, 8}, {inf, inf}, {0, 0, 0},
                         {2.0 - 2e-7, 2, 2.0 - 2e-7}, {7.0 / 3.0, 2, inf},
                         {false, false, false});
  run("marginal bound, LP, feasible", lp,
      {{{"presolve", "false"}, {"log_level", "debug"}},
       {{"presolve", "false"}, {"algorithm", "simplex"}},
       {{"presolve", "false"}, {"algorithm", "dual-simplex"}},
       {{"presolve", "false"}, {"scaling", "false"}},
       {}});
}

TEST(DiagFindings, MarginalBoundAsMilp) {
  // The same with x1 an integer column: fixed at 2, then free in [2, 3] with min x1.
  const Model fixed = build({{3, -2, 0}, {0, 3, 1}}, {2, 8}, {inf, inf}, {0, 0, 0},
                            {2.0 - 2e-7, 2, 2.0 - 2e-7}, {7.0 / 3.0, 2, inf},
                            {false, true, false});
  run("marginal bound, MILP, x1 fixed, optimum 0", fixed,
      {{{"presolve", "false"}, {"mip_threads", "1"}, {"log_level", "debug"}},
       {{"presolve", "false"}},
       {{"mip_threads", "1"}},
       {}});
  const Model ranged = build({{3, -2, 0}, {0, 3, 1}}, {2, 8}, {inf, inf}, {0, 1, 0},
                             {2.0 - 2e-7, 2, 2.0 - 2e-7}, {7.0 / 3.0, 3, inf},
                             {false, true, false});
  run("marginal bound, MILP, x1 in [2, 3], optimum 2", ranged,
      {{{"presolve", "false"}, {"mip_threads", "1"}}, {{"presolve", "false"}}, {}});
}

TEST(DiagFindings, ObbtOneAgain) {
  const Model model = build({{0, 4, 0}, {3, -2, 0}, {-6, 0, 0}, {0, 3, 1}}, {6, 2, -14, 8},
                            {inf, inf, inf, inf}, {0, 0, 0}, {-inf, -inf, 0}, {inf, inf, inf},
                            {false, true, false});
  run("OBBT one, feasible, optimum 0", model,
      {{{"mip_obbt", "true"}, {"mip_threads", "1"}, {"log_level", "debug"}}});
}

TEST(DiagFindings, CoefficientTighteningAgain) {
  const Model model =
      build({{0, 6, -3, 0}, {-4, -1.5, 2, -0.75}, {0, -2, -0.25, -1}, {6, 0, 0, 1.5}},
            {12, 5.5, -7.5, -6}, {12, inf, inf, -6}, {0, 0, 0, -3}, {-inf, -inf, -inf, -inf},
            {inf, 5, inf, -1}, {false, true, false, true});
  run("coefficient tightening, optimum 6", model,
      {{{"presolve_coefficient_tightening", "true"}, {"mip_threads", "1"},
        {"log_level", "debug"}}});
}

}  // namespace
}  // namespace sankhya
