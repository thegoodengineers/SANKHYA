// SPDX-License-Identifier: Apache-2.0
// SANKHYA - sankhya_solve_scenarios (#752): the C surface of src/core/scenarios.cpp, which
// `sankhya scenarios` also calls. Arguments are checked and converted here; the solving,
// the restart from a basis and the per-scenario check all live in the core, so the CLI, the
// C API and the Python bindings cannot disagree about what a scenario run does.
//
// Method: warm-started simplex re-optimisation, Chvatal, "Linear Programming" (1983),
// ch. 10; Maros, "Computational Techniques of the Simplex Method" (2003), sec. 9.4 on
// restarting from a given basis. References for the checks are in core/scenarios.hpp.

#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "api/c_api_internal.hpp"
#include "core/scenarios.hpp"
#include "sankhya/sankhya.h"

using sankhya::capi::fail;
using sankhya::capi::guarded;
using sankhya::capi::ok;

namespace {

namespace sc = sankhya::scenarios;

// The C enum is sc::Target in the same order, so a target converts by value.
constexpr int kFirstTarget = static_cast<int>(SANKHYA_SCENARIO_COST);
constexpr int kLastColumnTarget = static_cast<int>(SANKHYA_SCENARIO_COL_UPPER);
constexpr int kLastTarget = static_cast<int>(SANKHYA_SCENARIO_RHS);
static_assert(static_cast<int>(sc::Target::kCost) == kFirstTarget);
static_assert(static_cast<int>(sc::Target::kColLower) ==
              static_cast<int>(SANKHYA_SCENARIO_COL_LOWER));
static_assert(static_cast<int>(sc::Target::kColUpper) == kLastColumnTarget);
static_assert(static_cast<int>(sc::Target::kRowLower) ==
              static_cast<int>(SANKHYA_SCENARIO_ROW_LOWER));
static_assert(static_cast<int>(sc::Target::kRowUpper) ==
              static_cast<int>(SANKHYA_SCENARIO_ROW_UPPER));
static_assert(static_cast<int>(sc::Target::kRhs) == kLastTarget);

}  // namespace

extern "C" {

sankhya_status sankhya_solve_scenarios(sankhya_model* model, const sankhya_options* options,
                                       int count, const int* begin, const int* targets,
                                       const int* indices, const double* values,
                                       sankhya_solution** solutions, int* verified,
                                       double* safe_bounds) {
  if (model == nullptr || count < 0 || begin == nullptr ||
      (count > 0 && (solutions == nullptr || verified == nullptr))) {
    return fail(SANKHYA_ERROR_ARGUMENT,
                "model, begin, solutions or verified is null, or count is negative");
  }
  for (int k = 0; k < count; ++k) solutions[k] = nullptr;
  if (begin[0] != 0) return fail(SANKHYA_ERROR_ARGUMENT, "begin[0] must be 0");
  for (int k = 0; k < count; ++k) {
    if (begin[k + 1] < begin[k]) {
      return fail(SANKHYA_ERROR_ARGUMENT, "begin must be non-decreasing; begin[" +
                                              std::to_string(k + 1) + "] < begin[" +
                                              std::to_string(k) + "]");
    }
  }
  const int entries = begin[count];
  if (entries > 0 && (targets == nullptr || indices == nullptr || values == nullptr)) {
    return fail(SANKHYA_ERROR_ARGUMENT, "targets, indices or values is null");
  }
  if (model->nonlinear != nullptr && !model->nonlinear->empty()) {
    return fail(SANKHYA_ERROR_ARGUMENT, "scenarios are LP only: the model has nonlinear parts");
  }
  return guarded([&]() -> sankhya_status {
    sankhya::Model built;
    sankhya::capi::materialise(*model, &built);
    const auto cols = static_cast<int>(built.col_cost.size());
    const auto rows = static_cast<int>(built.row_lower.size());

    std::vector<sc::Scenario> scenarios(static_cast<std::size_t>(count));
    for (int k = 0; k < count; ++k) {
      sc::Scenario& s = scenarios[static_cast<std::size_t>(k)];
      s.name = "s" + std::to_string(k);
      for (int e = begin[k]; e < begin[k + 1]; ++e) {
        const int t = targets[e];
        if (t < kFirstTarget || t > kLastTarget) {
          return fail(SANKHYA_ERROR_ARGUMENT, "entry " + std::to_string(e) + ": target " +
                                                  std::to_string(t) +
                                                  " is not a sankhya_scenario_target");
        }
        const bool on_column = t <= kLastColumnTarget;
        const int size = on_column ? cols : rows;
        if (indices[e] < 0 || indices[e] >= size) {
          return fail(SANKHYA_ERROR_ARGUMENT,
                      "entry " + std::to_string(e) + ": " + (on_column ? "column " : "row ") +
                          std::to_string(indices[e]) + " is outside the model, which has " +
                          std::to_string(size));
        }
        if (std::isnan(values[e])) {
          return fail(SANKHYA_ERROR_ARGUMENT, "entry " + std::to_string(e) + " is NaN");
        }
        s.overrides.push_back({static_cast<sc::Target>(t), indices[e], values[e]});
      }
    }

    sankhya::Options effective;
    if (options != nullptr) effective = options->options;
    sc::BatchReport report = sc::solve_all(built, scenarios, effective);

    // Every handle is built before any is handed out, so an allocation failure part way
    // leaks nothing and leaves `solutions` all null.
    std::vector<std::unique_ptr<sankhya_solution>> handles(static_cast<std::size_t>(count));
    for (std::size_t k = 0; k < handles.size(); ++k) {
      handles[k] = std::make_unique<sankhya_solution>();
      handles[k]->solution = std::move(report.results[k].solution);
      handles[k]->col_names = built.col_names;
      handles[k]->row_names = built.row_names;
    }
    for (int k = 0; k < count; ++k) {
      const sc::ScenarioResult& r = report.results[static_cast<std::size_t>(k)];
      verified[k] = r.verified ? 1 : 0;
      if (safe_bounds != nullptr) safe_bounds[k] = r.safe_bound;
      solutions[k] = handles[static_cast<std::size_t>(k)].release();
    }
    return ok();
  });
}

}  // extern "C"
