// SPDX-License-Identifier: Apache-2.0
// SANKHYA - sankhya_solve_scenarios (#752) through the C API: each scenario's answer equals a
// cold solve of the same model edited with the sankhya_model_set_* calls, an unreachable
// demand is proved infeasible and never called optimal, and malformed arguments are refused
// before anything is solved.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/sankhya.h"
#include "sankhya/tolerances.hpp"

namespace sankhya {
namespace {

std::string demo_path() {
  return (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
          "demo/crude_blend.mps")
      .string();
}

sankhya_options* quiet() {
  sankhya_options* options = sankhya_options_create();
  EXPECT_EQ(sankhya_options_set_bool(options, "log_to_console", 0), SANKHYA_OK);
  return options;
}

// demo/crude_blend.mps: columns AL BN MU, rows THRUPUT (90..120) DIESEL (= 40) SULPHUR (<= 0).
constexpr int kAL = 0;
constexpr int kDiesel = 1;
constexpr int kThruput = 0;

struct Batch {
  std::vector<int> begin{0}, targets, indices;
  std::vector<double> values;
  void add(std::vector<std::tuple<int, int, double>> overrides) {
    for (const auto& [t, i, v] : overrides) {
      targets.push_back(t);
      indices.push_back(i);
      values.push_back(v);
    }
    begin.push_back(static_cast<int>(targets.size()));
  }
  [[nodiscard]] int count() const { return static_cast<int>(begin.size()) - 1; }
};

double cold_objective(void (*edit)(sankhya_model*), sankhya_solve_status* status) {
  sankhya_model* model = sankhya_model_create();
  EXPECT_EQ(sankhya_model_read(model, demo_path().c_str()), SANKHYA_OK);
  edit(model);
  sankhya_options* options = quiet();
  sankhya_solution* solution = nullptr;
  EXPECT_EQ(sankhya_solve(model, options, &solution), SANKHYA_OK);
  *status = sankhya_solution_status(solution);
  const double objective = sankhya_solution_objective(solution);
  sankhya_solution_free(solution);
  sankhya_options_free(options);
  sankhya_model_free(model);
  return objective;
}

TEST(CApiScenarios, EachAnswerEqualsAColdSolveOfTheEditedModel) {
  sankhya_model* model = sankhya_model_create();
  ASSERT_EQ(sankhya_model_read(model, demo_path().c_str()), SANKHYA_OK);
  Batch batch;
  batch.add({{SANKHYA_SCENARIO_COST, kAL, 1.9}});      // a dearer Arab Light
  batch.add({{SANKHYA_SCENARIO_RHS, kDiesel, 42.0}});  // a larger diesel pool
  batch.add({{SANKHYA_SCENARIO_COST, kAL, 2.9},        // both, and a narrower CDU
             {SANKHYA_SCENARIO_ROW_UPPER, kThruput, 110.0}});
  batch.add({{SANKHYA_SCENARIO_RHS, kDiesel, 60.0}});  // out of reach (> 0.45 * 120)

  for (const char* start : {"base", "previous"}) {
    sankhya_options* options = quiet();
    ASSERT_EQ(sankhya_options_set_string(options, "scenario_start", start), SANKHYA_OK);
    std::vector<sankhya_solution*> solutions(4, nullptr);
    std::vector<int> verified(4, -1);
    std::vector<double> bounds(4, 0.0);
    ASSERT_EQ(
        sankhya_solve_scenarios(model, options, batch.count(), batch.begin.data(),
                                batch.targets.data(), batch.indices.data(), batch.values.data(),
                                solutions.data(), verified.data(), bounds.data()),
        SANKHYA_OK)
        << sankhya_last_error();

    using Edit = void (*)(sankhya_model*);
    const Edit edits[3] = {
        [](sankhya_model* m) { sankhya_model_set_objective_coefficient(m, kAL, 1.9); },
        [](sankhya_model* m) { sankhya_model_set_row_bounds(m, kDiesel, 42.0, 42.0); },
        [](sankhya_model* m) {
          sankhya_model_set_objective_coefficient(m, kAL, 2.9);
          sankhya_model_set_row_bounds(m, kThruput, 90.0, 110.0);
        }};
    for (std::size_t k = 0; k < 3; ++k) {
      sankhya_solve_status cold_status = SANKHYA_NOT_SOLVED;
      const double cold = cold_objective(edits[k], &cold_status);
      ASSERT_EQ(cold_status, SANKHYA_OPTIMAL);
      EXPECT_EQ(verified[k], 1) << start << " scenario " << k;
      ASSERT_EQ(sankhya_solution_status(solutions[k]), SANKHYA_OPTIMAL) << start << " " << k;
      const double ours = sankhya_solution_objective(solutions[k]);
      EXPECT_LE(std::fabs(ours - cold) / std::max(1.0, std::fabs(cold)),
                tol::kScenarioAgreement)
          << start << " scenario " << k << ": " << ours << " against " << cold;
      // A maximisation: the safe bound is an upper bound on the optimum, when proved.
      if (!std::isnan(bounds[k])) {
        EXPECT_GE(bounds[k], cold - 1e-9 * std::max(1.0, cold));
      }
    }
    EXPECT_NE(sankhya_solution_status(solutions[3]), SANKHYA_OPTIMAL);
    if (verified[3] == 1) {
      EXPECT_EQ(sankhya_solution_status(solutions[3]), SANKHYA_INFEASIBLE);
    }
    for (sankhya_solution* s : solutions) sankhya_solution_free(s);
    sankhya_options_free(options);
  }
  // The handle itself is untouched: a plain solve still gives the base optimum.
  sankhya_solve_status base_status = SANKHYA_NOT_SOLVED;
  const double base = cold_objective([](sankhya_model*) {}, &base_status);
  sankhya_solution* again = nullptr;
  ASSERT_EQ(sankhya_solve(model, nullptr, &again), SANKHYA_OK);
  EXPECT_DOUBLE_EQ(sankhya_solution_objective(again), base);
  sankhya_solution_free(again);
  sankhya_model_free(model);
}

TEST(CApiScenarios, MalformedArgumentsAreRefusedBeforeAnySolve) {
  sankhya_model* model = sankhya_model_create();
  ASSERT_EQ(sankhya_model_read(model, demo_path().c_str()), SANKHYA_OK);
  sankhya_solution* solution = nullptr;
  int verified = 0;
  const int begin[2] = {0, 1};
  int target = SANKHYA_SCENARIO_COST;
  int index = 7;  // the model has three columns
  double value = 1.0;
  EXPECT_EQ(sankhya_solve_scenarios(model, nullptr, 1, begin, &target, &index, &value,
                                    &solution, &verified, nullptr),
            SANKHYA_ERROR_ARGUMENT);
  EXPECT_EQ(solution, nullptr);
  index = 0;
  target = 9;  // not a target
  EXPECT_EQ(sankhya_solve_scenarios(model, nullptr, 1, begin, &target, &index, &value,
                                    &solution, &verified, nullptr),
            SANKHYA_ERROR_ARGUMENT);
  target = SANKHYA_SCENARIO_RHS;
  const int backwards[2] = {1, 0};
  EXPECT_EQ(sankhya_solve_scenarios(model, nullptr, 1, backwards, &target, &index, &value,
                                    &solution, &verified, nullptr),
            SANKHYA_ERROR_ARGUMENT);
  EXPECT_EQ(sankhya_solve_scenarios(nullptr, nullptr, 1, begin, &target, &index, &value,
                                    &solution, &verified, nullptr),
            SANKHYA_ERROR_ARGUMENT);
  // No scenario at all is an empty batch, not an error.
  const int empty[1] = {0};
  EXPECT_EQ(sankhya_solve_scenarios(model, nullptr, 0, empty, nullptr, nullptr, nullptr,
                                    nullptr, nullptr, nullptr),
            SANKHYA_OK);
  sankhya_model_free(model);
}

}  // namespace
}  // namespace sankhya
