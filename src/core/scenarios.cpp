// SPDX-License-Identifier: Apache-2.0
// SANKHYA - one LP under many cost, bound and right-hand-side sets (#752). The method and its
// references are in scenarios.hpp.

#include "core/scenarios.hpp"

#include <cmath>
#include <limits>
#include <utility>

#include "core/kkt_check.hpp"
#include "core/safe_bound.hpp"
#include "sankhya/certificate.hpp"
#include "sankhya/solve_control.hpp"
#include "sankhya/timer.hpp"

namespace sankhya::scenarios {

namespace {

/// Is `solution` what it claims for `model`? Fills `verdict` either way.
bool check(const Model& model, const Solution& solution, std::string* verdict) {
  switch (solution.status) {
    case SolveStatus::kOptimal: {
      const KktVerdict kkt = check_lp_optimality(model, solution);
      *verdict = kkt.passed ? "kkt" : kkt.check + ": " + kkt.detail;
      return kkt.passed;
    }
    case SolveStatus::kInfeasible: {
      std::string why;
      const bool proved = !solution.farkas_dual.empty() &&
                          farkas_proves_infeasible(model, solution.farkas_dual, &why);
      *verdict = proved ? "farkas" : (why.empty() ? "no infeasibility certificate" : why);
      return proved;
    }
    default:
      *verdict = std::string("no check for status ") + to_string(solution.status);
      return false;
  }
}

double safe_objective_bound(const Model& model, const Solution& solution) {
  if (solution.row_dual.size() != static_cast<std::size_t>(model.num_rows())) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const SafeBound bound = safe_dual_bound(model, solution.row_dual);
  if (!std::isfinite(bound.value)) return std::numeric_limits<double>::quiet_NaN();
  return model.sense_multiplier() * bound.value + model.objective_offset;
}

}  // namespace

Model apply(const Model& base, const Scenario& scenario) {
  Model model = base;
  for (const Override& o : scenario.overrides) {
    const auto u = static_cast<std::size_t>(o.index);
    switch (o.target) {
      case Target::kCost: model.col_cost[u] = o.value; break;
      case Target::kColLower: model.col_lower[u] = o.value; break;
      case Target::kColUpper: model.col_upper[u] = o.value; break;
      case Target::kRowLower: model.row_lower[u] = o.value; break;
      case Target::kRowUpper: model.row_upper[u] = o.value; break;
      case Target::kRhs:
        if (is_finite_bound(base.row_lower[u])) model.row_lower[u] = o.value;
        if (is_finite_bound(base.row_upper[u])) model.row_upper[u] = o.value;
        break;
    }
  }
  return model;
}

BatchReport solve_all(const Model& base, const std::vector<Scenario>& scenarios,
                      const Options& options) {
  const Timer clock;
  BatchReport report;
  report.results.resize(scenarios.size());
  if (base.has_integrality() || base.has_quadratic_objective()) {
    for (ScenarioResult& r : report.results) {
      r.verdict = "scenarios are LP only: the model has integer columns or a Hessian";
      r.safe_bound = std::numeric_limits<double>::quiet_NaN();
    }
    report.seconds = clock.elapsed_seconds();
    return report;
  }

  const bool automatic = options.get_string("algorithm") == "auto";
  report.base = solve(base, options);
  const bool have_basis = report.base.col_status.size() == base.col_cost.size() &&
                          report.base.row_status.size() == base.row_lower.size();

  for (std::size_t k = 0; k < scenarios.size(); ++k) {
    const Timer scenario_clock;
    const Model model = apply(base, scenarios[k]);
    ScenarioResult& r = report.results[k];

    Options warm = options;
    // The dual simplex restarts from the base basis whatever moved: a bound or right-hand
    // side move leaves it dual feasible, and a cost move is repaired by the dual engine's own
    // phase one. Measured on the 52-week refinery (#752): for five price sets the primal
    // simplex restart took 884-1096 pivots and more time than a cold solve, the dual
    // restart 127-296 pivots and a fraction of it.
    if (automatic) warm.set_string("algorithm", "dual-simplex");
    SolveControl control;
    if (have_basis) {
      control.start_col_status = report.base.col_status;
      control.start_row_status = report.base.row_status;
    }
    r.solution = solve(model, warm, &control);
    r.verified = check(model, r.solution, &r.verdict);

    if (!r.verified) {
      // Never reported on the warm run's word: a cold dual simplex from nothing, checked the
      // same way. Its answer replaces the warm one only when the check passes.
      Options cold = options;
      cold.set_string("algorithm", "dual-simplex");
      Solution again = solve(model, cold);
      std::string verdict;
      if (check(model, again, &verdict)) {
        r.solution = std::move(again);
        r.verified = true;
        r.verdict = verdict;
        r.resolved_cold = true;
      }
    }
    r.safe_bound = safe_objective_bound(model, r.solution);
    r.seconds = scenario_clock.elapsed_seconds();
  }
  report.seconds = clock.elapsed_seconds();
  return report;
}

}  // namespace sankhya::scenarios
