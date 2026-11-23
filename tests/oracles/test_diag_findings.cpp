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

TEST(DiagFindings, UnboundedLpA) {
  const Model model = build({{3, -6, 0}, {0, 6, 0.5}, {4, 3, 0}}, {21, -inf, -inf},
                            {inf, -11.25, 10}, {0, 0, 0.125}, {-inf, -inf, -inf},
                            {inf, inf, 1}, {false, false, false});
  run("unbounded LP A", model,
      {{},
       {{"presolve", "false"}},
       {{"presolve_dual_reductions", "false"}},
       {{"presolve_substitution", "false"}},
       {{"presolve_dual_reductions", "false"}, {"presolve_substitution", "false"}},
       {{"algorithm", "simplex"}},
       {{"algorithm", "simplex"}, {"presolve", "false"}},
       {{"presolve", "false"}, {"scaling", "false"}},
       {{"presolve", "false"}, {"dual_ratio_test", "textbook"}},
       {{"presolve", "false"}, {"dual_ratio_test", "harris"}}});
}

TEST(DiagFindings, UnboundedLpB) {
  const Model model = build({{0, -5, 2}, {-1, 0, 2.0 / 3.0}, {-0.5, 0, 5}}, {4, -inf, -inf},
                            {inf, -3, 0}, {0, 0, 4}, {-inf, -inf, -inf}, {inf, 0, inf},
                            {false, false, false});
  run("unbounded LP B", model,
      {{},
       {{"presolve", "false"}},
       {{"presolve_dual_reductions", "false"}},
       {{"presolve_substitution", "false"}},
       {{"presolve_dual_reductions", "false"}, {"presolve_substitution", "false"}},
       {{"algorithm", "simplex"}},
       {{"algorithm", "simplex"}, {"presolve", "false"}},
       {{"presolve", "false"}, {"dual_ratio_test", "textbook"}},
       {{"presolve", "false"}, {"dual_ratio_test", "harris"}}});
}

TEST(DiagFindings, UnboundedLpC) {
  const Model model = build({{-4, 2, 0}, {7, 7, 2}}, {-3546.5, -inf}, {inf, -6247}, {0, 0, 4},
                            {-inf, -inf, -inf}, {inf, inf, 701}, {false, false, false});
  run("unbounded LP C", model,
      {{},
       {{"presolve", "false"}},
       {{"presolve_dual_reductions", "false"}},
       {{"presolve_substitution", "false"}},
       {{"algorithm", "simplex"}}});
}

TEST(DiagFindings, InfeasibleLpUnderTheRace) {
  const Model model = build({{-3, -2}, {-5, 0}}, {-4, -5}, {-4, inf}, {0, 0}, {-inf, -inf},
                            {inf, -253}, {false, false});
  run("infeasible LP", model,
      {{},
       {{"engine_race", "true"}, {"deterministic", "true"}},
       {{"engine_race", "true"}},
       {{"engine_race", "true"}, {"deterministic", "true"}, {"presolve", "false"}},
       {{"engine_race", "true"},
        {"deterministic", "true"},
        {"presolve_dual_reductions", "false"}}});
}

TEST(DiagFindings, InfeasibleMilpReportedAsAModelError) {
  const Model model = build({{-6, 1.5, -6}}, {-6}, {-6}, {0, 0, 0}, {-inf, -1, -inf}, {1, -1, 1},
                            {true, false, true});
  run("infeasible MILP", model,
      {{},
       {{"enable_root_cuts", "false"}},
       {{"mir_cmir", "false"}},
       {{"enable_root_cuts", "false"}, {"mir_cmir", "false"}},
       {{"presolve", "false"}},
       {{"presolve", "false"}, {"enable_root_cuts", "false"}},
       {{"mip_threads", "1"}},
       {{"mip_threads", "1"}, {"enable_root_cuts", "false"}},
       {{"mip_heur_fj", "off"}},
       {{"mip_heur_parity", "off"}}});
}

}  // namespace
}  // namespace sankhya
