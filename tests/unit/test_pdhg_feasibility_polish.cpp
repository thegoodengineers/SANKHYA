// SPDX-License-Identifier: Apache-2.0
// SANKHYA - PDHG feasibility polishing (#483, option pdhg_feasibility_polish).
//
// What the option promises, held on committed Netlib instances: PDHG's answer, stopped well
// short of convergence by an iteration limit, comes back primal feasible to 1e-8 relative
// (and to the project's absolute primal tolerance), and the gap it states is the gap of the
// point it reports. Both are RECOMPUTED here from the model and the reported vectors, by
// code that does not share src/pdhg/pdhg_evaluate.cpp, so the engine is not marking its own
// homework. And with the option off the engine is untouched: the same point, bit for bit.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <ostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"
#include "sankhya/types.hpp"

namespace sankhya {
namespace {

Model netlib(const std::string& name) {
  Model model;
  const std::string path =
      (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
       "data/netlib" / (name + ".mps"))
          .string();
  EXPECT_TRUE(io::read_model(path, &model).ok) << name;
  return model;
}

/// The engine on its own: no presolve, no interior-point finish (pdhg_polish, #229, is a
/// different thing and would hide what this option does).
Options pdhg_options(std::int64_t iteration_limit) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", false);
  options.set_string("algorithm", "pdhg");
  options.set_bool("pdhg_polish", false);
  options.set_int("iteration_limit", iteration_limit);
  return options;
}

struct Recomputed {
  double absolute_primal = 0.0;  ///< 2-norm of row and column bound violations
  double relative_primal = 0.0;  ///< the same over 1 + ||b||, PDLP's normalisation
  double objective = 0.0;        ///< c'x + offset
  double dual_objective = 0.0;   ///< the Lagrangian bound of row_dual and col_dual
};

/// Everything from the model and the reported vectors. Minimisation only, which every
/// instance used here is.
Recomputed recompute(const Model& model, const Solution& s) {
  Recomputed r;
  const auto m = static_cast<std::size_t>(model.num_rows());
  const auto n = static_cast<std::size_t>(model.num_cols());
  std::vector<double> activity(m, 0.0);
  model.matrix.multiply(s.col_value.data(), activity.data());

  double violation2 = 0.0;
  double bound2 = 0.0;
  for (std::size_t i = 0; i < m; ++i) {
    const double lo = model.row_lower[i];
    const double hi = model.row_upper[i];
    double v = 0.0;
    if (is_finite_bound(lo)) v = std::max(v, lo - activity[i]);
    if (is_finite_bound(hi)) v = std::max(v, activity[i] - hi);
    violation2 += v * v;
    const double b = is_finite_bound(lo) ? lo : (is_finite_bound(hi) ? hi : 0.0);
    bound2 += b * b;
  }
  for (std::size_t j = 0; j < n; ++j) {
    const double x = s.col_value[j];
    double v = 0.0;
    if (is_finite_bound(model.col_lower[j])) v = std::max(v, model.col_lower[j] - x);
    if (is_finite_bound(model.col_upper[j])) v = std::max(v, x - model.col_upper[j]);
    violation2 += v * v;
  }
  r.absolute_primal = std::sqrt(violation2);
  r.relative_primal = r.absolute_primal / (1.0 + std::sqrt(bound2));

  r.objective = model.objective_offset;
  for (std::size_t j = 0; j < n; ++j) r.objective += model.col_cost[j] * s.col_value[j];

  // The reduced costs are rebuilt from the row duals rather than read back, so the bound
  // below is the bound of row_dual alone: d = c - A'y.
  std::vector<double> reduced(model.col_cost.begin(), model.col_cost.end());
  model.matrix.transpose_multiply_add(s.row_dual.data(), reduced.data(), -1.0);
  r.dual_objective = model.objective_offset;
  for (std::size_t i = 0; i < m; ++i) {
    const double y = s.row_dual[i];
    if (y > 0.0 && is_finite_bound(model.row_lower[i]))
      r.dual_objective += y * model.row_lower[i];
    if (y < 0.0 && is_finite_bound(model.row_upper[i]))
      r.dual_objective += y * model.row_upper[i];
  }
  for (std::size_t j = 0; j < n; ++j) {
    const double d = reduced[j];
    if (d > 0.0 && is_finite_bound(model.col_lower[j]))
      r.dual_objective += d * model.col_lower[j];
    if (d < 0.0 && is_finite_bound(model.col_upper[j]))
      r.dual_objective += d * model.col_upper[j];
  }
  return r;
}

/// Each instance with an iteration limit well short of what PDHG needs to converge on it
/// (afiro, the quickest, converges at 920; the others take tens of thousands), so the main
/// run ends on the limit with its point visibly infeasible and the answer has to come from
/// the polish. Limits differ because the polish starts from wherever the limit left the run:
/// adlittle and blend from 300 are further out than one main-run budget can bring back.
struct Case {
  const char* name;
  std::int64_t iteration_limit;
};

void PrintTo(const Case& c, std::ostream* os) {
  *os << c.name << "@" << c.iteration_limit;
}

class PdhgFeasibilityPolish : public ::testing::TestWithParam<Case> {};

TEST_P(PdhgFeasibilityPolish, StoppedShortTheAnswerIsPrimalFeasibleWithItsMeasuredGap) {
  const Case c = GetParam();
  const Model model = netlib(c.name);
  ASSERT_EQ(model.sense, ObjSense::kMinimize);

  // Without the option the point the limit leaves is far from feasible to 1e-8; that is
  // what makes the assertions below about the polish rather than about the instance.
  const Solution plain = solve(model, pdhg_options(c.iteration_limit));
  ASSERT_EQ(plain.status, SolveStatus::kIterationLimit) << plain.message;
  EXPECT_GT(recompute(model, plain).relative_primal, 1e3 * tol::kPdhgTight) << c.name;

  Options options = pdhg_options(c.iteration_limit);
  options.set_bool("pdhg_feasibility_polish", true);
  const Solution s = solve(model, options);
  ASSERT_EQ(s.col_value.size(), static_cast<std::size_t>(model.num_cols())) << s.message;
  ASSERT_EQ(s.row_dual.size(), static_cast<std::size_t>(model.num_rows())) << s.message;
  EXPECT_NE(s.message.find("reported point polished"), std::string::npos) << s.message;
  // The limit still names why the run stopped, and the polishing work is counted.
  EXPECT_EQ(s.status, SolveStatus::kIterationLimit) << s.message;
  EXPECT_GT(s.iterations, c.iteration_limit) << s.message;

  const Recomputed r = recompute(model, s);
  EXPECT_LE(r.relative_primal, tol::kPdhgTight) << c.name << ": " << s.message;
  EXPECT_LE(r.absolute_primal, tol::kPrimalFeasibility) << c.name << ": " << s.message;
  EXPECT_NEAR(r.objective, s.objective, 1e-9 * (1.0 + std::fabs(s.objective))) << c.name;

  // The gap is STATED: a finite dual bound, the Lagrangian bound of the reported duals, and
  // absolute_gap and relative_gap are its distance from the reported objective.
  ASSERT_TRUE(std::isfinite(s.dual_bound)) << c.name << ": " << s.message;
  const double scale = 1.0 + std::fabs(s.objective);
  EXPECT_NEAR(s.dual_bound, r.dual_objective, 1e-7 * scale) << c.name;
  EXPECT_NEAR(s.absolute_gap, std::fabs(r.objective - r.dual_objective), 1e-7 * scale)
      << c.name;
  EXPECT_NEAR(s.relative_gap,
              std::fabs(r.objective - r.dual_objective) / std::max(1.0, std::fabs(r.objective)),
              1e-7)
      << c.name;
  // A Lagrangian bound of dual feasible multipliers cannot lie above a feasible point's
  // objective when minimising, beyond rounding.
  EXPECT_LE(r.dual_objective, r.objective + 1e-6 * scale) << c.name;
}

TEST_P(PdhgFeasibilityPolish, OffIsTheEngineAsItWas) {
  // Off by default, and off explicitly, is one code path: the same point, duals, count and
  // message, bit for bit, with no word of the polish and no bound stated for a point stopped
  // short, as before #483.
  const Case c = GetParam();
  const Model model = netlib(c.name);
  const Solution by_default = solve(model, pdhg_options(c.iteration_limit));
  Options off = pdhg_options(c.iteration_limit);
  off.set_bool("pdhg_feasibility_polish", false);
  const Solution explicit_off = solve(model, off);
  EXPECT_EQ(by_default.status, explicit_off.status);
  EXPECT_EQ(by_default.iterations, explicit_off.iterations);
  EXPECT_EQ(by_default.iterations, c.iteration_limit);
  EXPECT_EQ(by_default.col_value, explicit_off.col_value);
  EXPECT_EQ(by_default.row_dual, explicit_off.row_dual);
  EXPECT_EQ(by_default.message, explicit_off.message);
  EXPECT_EQ(by_default.message.find("feasibility polish"), std::string::npos);
  EXPECT_TRUE(std::isinf(by_default.dual_bound));
}

INSTANTIATE_TEST_SUITE_P(Netlib, PdhgFeasibilityPolish,
                         ::testing::Values(Case{"afiro", 300}, Case{"sc50a", 300},
                                           Case{"adlittle", 1000}, Case{"blend", 1000},
                                           Case{"share2b", 2000}),
                         [](const ::testing::TestParamInfo<Case>& param_info) {
                           return std::string(param_info.param.name);
                         });

}  // namespace
}  // namespace sankhya
