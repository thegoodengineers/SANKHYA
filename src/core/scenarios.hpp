// SPDX-License-Identifier: Apache-2.0
// SANKHYA - one LP under many cost, bound and right-hand-side sets in one run (#752).
//
// A planner asks the same model under fifty crude price sets or twenty demand forecasts.
// Fifty separate `sankhya solve` calls each pay for reading, presolve and a cold start. Here
// the model is read once, solved once for its base basis, and every scenario is re-solved
// from that basis by the dual simplex - the warm start of #218, the same restart the
// parametric sweep (#522) walks. A bound or right-hand-side move leaves the base basis dual
// feasible; a cost move breaks that and the dual engine's phase one repairs it, which on the
// refinery measured faster than the primal restart a cost move would suggest.
//
// NOTHING IS REPORTED ON THE BATCH'S WORD. Every scenario's answer is checked against the
// scenario's own model by the in-process KKT check (core/kkt_check, the verifier's checks
// reimplemented, #476) or, for an infeasible verdict, by its Farkas certificate. An answer
// that fails is re-solved cold by the dual simplex and checked again; one that still fails
// is returned with `verified` false and must not be called optimal by any caller. Each
// scenario with row duals also carries a Neumaier-Shcherbina safe bound (#519): a bound on
// its optimum that holds whatever the duals' accuracy.
//
// References: warm-started simplex re-optimisation, Chvatal, "Linear Programming" (1983),
// ch. 10 (sensitivity and re-optimisation); the KKT conditions, Nocedal & Wright,
// "Numerical Optimization" 2nd ed. (2006) thm. 13.1; Farkas' lemma, ibid. thm. 12.2 context;
// Neumaier & Shcherbina, Math. Programming 99 (2004) 283-296.
#pragma once

#include <string>
#include <vector>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::scenarios {

/// What one override changes. kRhs moves every finite side of a row to the value (both sides
/// of an equality), which is what "the demand is now 120" means for any row type.
enum class Target { kCost, kColLower, kColUpper, kRowLower, kRowUpper, kRhs };

struct Override {
  Target target = Target::kCost;
  Index index = 0;  ///< a column for kCost / kColLower / kColUpper, a row otherwise
  double value = 0.0;
};

struct Scenario {
  std::string name;
  std::vector<Override> overrides;
};

struct ScenarioResult {
  Solution solution;
  /// The answer was checked against this scenario's model: KKT for an optimum, the Farkas
  /// certificate for an infeasibility. False means nothing about it may be claimed.
  bool verified = false;
  /// "kkt", "farkas", or why the check failed.
  std::string verdict;
  /// The warm answer failed its check and the cold dual simplex produced this one.
  bool resolved_cold = false;
  /// Safe bound on the objective in the model's sense, offset included (a lower bound when
  /// minimising, an upper bound when maximising); NaN when no finite bound was proved.
  double safe_bound = 0.0;
  double seconds = 0.0;
};

struct BatchReport {
  Solution base;
  std::vector<ScenarioResult> results;
  double seconds = 0.0;  ///< the base solve plus every scenario
};

/// `base` with the scenario's overrides applied.
[[nodiscard]] Model apply(const Model& base, const Scenario& scenario);

/// Solve the base model, then every scenario from its basis, checking each answer. An
/// explicit `algorithm` other than "auto" is used as given for every scenario (the starting
/// basis is then only a hint the engine may ignore); "auto" restarts the dual simplex.
/// Never throws on a numerical failure. LP only: a model with integer columns or a Hessian
/// gets every result unverified with the reason in `verdict`.
[[nodiscard]] BatchReport solve_all(const Model& base, const std::vector<Scenario>& scenarios,
                                    const Options& options);

}  // namespace sankhya::scenarios
