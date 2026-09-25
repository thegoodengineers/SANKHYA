// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the status guard's strong-duality test (#664), in both senses and with an
// objective constant, and the answers it must leave alone.
//
// THE CASE. One column, one row: minimize (1 + r) x, or maximize its negation, subject to
// x >= 0 and x >= 1e9, plus a constant. The optimum is x = 1e9, priced (in minimize space) by
// y = 1 + r with a zero reduced cost. Reporting y = 1 instead leaves c - A^T y - d = r,
// which with r = 5e-8 is inside the dual tolerance (1e-7) and the verifier's consistency
// threshold (1e-6), and a complementarity product of zero, so every per-item measure the
// guard already had accepts the claim - while primal and dual objectives differ by
// r * 1e9 = 50 objective units, fifty times the verifier's 1e-9 relative allowance. The
// preconditions are ASSERTed below, so these tests cannot pass because some other check
// fired: only the strong-duality test can move the status.
//
// The constant (7) is larger than the allowance (1), so an offset dropped from either side
// of the gap, or a sign convention got wrong for a maximize model, turns the genuine optimum
// into a downgrade and fails the test that expects it left alone.
//
// Reference: LP strong duality, e.g. Nocedal & Wright, "Numerical Optimization", 2nd ed.
// (2006), thm. 13.1; the accounted-gap form is tools/verify_solution.py's.

#include <string>

#include <gtest/gtest.h>

#include "core/status_guard.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya {
namespace {

constexpr double kResidue = 5e-8;
constexpr double kBound = 1e9;
constexpr double kOffset = 7.0;

/// The one-column model in the given sense, and a point at its optimum that claims optimal
/// with the row priced at `row_price` in MINIMIZE space (reported in the model's own sense).
struct Case {
  Model model;
  Solution solution;
};

Case one_column(ObjSense sense, double offset, double row_price) {
  Case c;
  Model& model = c.model;
  const double sigma = sense == ObjSense::kMaximize ? -1.0 : 1.0;
  model.sense = sense;
  model.objective_offset = offset;
  model.col_lower = {0.0};
  model.col_upper = {kInfinity};
  model.col_cost = {sigma * (1.0 + kResidue)};
  model.col_type = {VarType::kContinuous};
  model.row_lower = {kBound};
  model.row_upper = {kInfinity};
  model.matrix.reset(1, 1);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.finalize();

  Solution& s = c.solution;
  s.status = SolveStatus::kOptimal;
  s.algorithm = "fabricated";
  s.col_value = {kBound};
  s.row_activity = {kBound};
  s.row_dual = {sigma * row_price};
  s.col_dual = {0.0};
  s.recompute_quality(model);
  return c;
}

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  return options;
}

/// Every measure the guard applied before #664 accepts the point.
void assert_per_item_checks_pass(const Solution& s, const Options& options) {
  ASSERT_LE(s.primal_infeasibility_scaled, options.get_double("primal_feasibility_tolerance"));
  ASSERT_LE(s.dual_infeasibility_scaled, options.get_double("dual_feasibility_tolerance"));
  ASSERT_LE(s.complementarity_violation, tol::kComplementarity);
}

TEST(SolveStatusGuardDuality, AnUnaccountedGapIsFeasibleNotOptimalInEitherSense) {
  for (const ObjSense sense : {ObjSense::kMinimize, ObjSense::kMaximize}) {
    SCOPED_TRACE(sense == ObjSense::kMaximize ? "maximize" : "minimize");
    Case c = one_column(sense, kOffset, /*row_price=*/1.0);
    const Options options = quiet();
    assert_per_item_checks_pass(c.solution, options);

    Logger silent(nullptr);
    reconcile_status_with_measurement(c.model, &c.solution, options, silent,
                                      /*check_dual=*/true);
    EXPECT_EQ(c.solution.status, SolveStatus::kFeasible);
    EXPECT_NE(c.solution.message.find("strong duality"), std::string::npos)
        << c.solution.message;
    // The point is usable; only the proof is withdrawn.
    EXPECT_TRUE(claims_a_point(c.solution));
  }
}

TEST(SolveStatusGuardDuality, TheGenuineOptimumIsLeftAloneInEitherSense) {
  for (const ObjSense sense : {ObjSense::kMinimize, ObjSense::kMaximize}) {
    SCOPED_TRACE(sense == ObjSense::kMaximize ? "maximize" : "minimize");
    Case c = one_column(sense, kOffset, /*row_price=*/1.0 + kResidue);
    const Options options = quiet();
    assert_per_item_checks_pass(c.solution, options);
    const double objective = c.solution.objective;

    Logger silent(nullptr);
    reconcile_status_with_measurement(c.model, &c.solution, options, silent,
                                      /*check_dual=*/true);
    EXPECT_EQ(c.solution.status, SolveStatus::kOptimal) << c.solution.message;
    EXPECT_DOUBLE_EQ(c.solution.objective, objective);
  }
}

TEST(SolveStatusGuardDuality, NoDualCheckAndNoDualsLeaveTheClaimAlone) {
  Logger silent(nullptr);
  const Options options = quiet();

  // check_dual=false is what solve() passes for a MILP incumbent and a QP (whose own gate
  // follows): LP duality says nothing about either, so the gap must not be judged here.
  Case no_check = one_column(ObjSense::kMinimize, kOffset, /*row_price=*/1.0);
  reconcile_status_with_measurement(no_check.model, &no_check.solution, options, silent,
                                    /*check_dual=*/false);
  EXPECT_EQ(no_check.solution.status, SolveStatus::kOptimal) << no_check.solution.message;

  // An answer that carries no duals has no dual objective to compare, which is not a gap.
  Case dualless = one_column(ObjSense::kMinimize, kOffset, /*row_price=*/1.0);
  dualless.solution.row_dual.clear();
  dualless.solution.col_dual.clear();
  reconcile_status_with_measurement(dualless.model, &dualless.solution, options, silent,
                                    /*check_dual=*/true);
  EXPECT_EQ(dualless.solution.status, SolveStatus::kOptimal) << dualless.solution.message;
}

}  // namespace
}  // namespace sankhya
