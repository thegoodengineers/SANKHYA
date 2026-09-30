// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the status guard's complementarity test (#209) measures the slack SIGNED, the
// way tools/verify_solution.py and kkt_check.cpp do.
//
// THE CASE. One column, one row: minimize -1000 x subject to x <= 1, x >= 0. The optimum is
// x = 1, the row priced at y = -1000 with a zero reduced cost. The reported point moves the
// row 3e-9 off its bound, which is inside the primal tolerance, so the only question is what
// the complementarity product |y| * slack makes of it.
//
// Inside the bound the slack is 3e-9 and the product 3e-6, three times the verifier's 1e-6:
// the row is priced while not tight, and the claim is withdrawn. Past the bound the slack
// is -3e-9 and there is no product: the excess is primal infeasibility, already judged
// against its own tolerance. Measured as the distance |x - 1| it was charged a second time,
// and on Netlib adlittle a PDHG point 2.7e-9 past a bound of 0, priced at 765, was reported
// feasible although the verifier accepts it as optimal.

#include <string>

#include <gtest/gtest.h>

#include "core/status_guard.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya {
namespace {

constexpr double kPrice = 1000.0;
constexpr double kOffset = 3e-9;

struct Case {
  Model model;
  Solution solution;
};

/// The optimum's multipliers, at x = 1 + displacement.
Case priced_row(double displacement) {
  Case c;
  Model& model = c.model;
  model.col_lower = {0.0};
  model.col_upper = {kInfinity};
  model.col_cost = {-kPrice};
  model.col_type = {VarType::kContinuous};
  model.row_lower = {-kInfinity};
  model.row_upper = {1.0};
  model.matrix.reset(1, 1);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.finalize();

  Solution& s = c.solution;
  s.status = SolveStatus::kOptimal;
  s.algorithm = "fabricated";
  s.col_value = {1.0 + displacement};
  s.row_activity = {1.0 + displacement};
  s.row_dual = {-kPrice};
  s.col_dual = {0.0};
  s.recompute_quality(model);
  return c;
}

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  return options;
}

TEST(SolveStatusGuardComplementarity, ARowPastItsBoundWithinTolerancePricesNothing) {
  Case c = priced_row(+kOffset);
  const Options options = quiet();
  ASSERT_GT(c.solution.primal_infeasibility, 0.0) << "the row is supposed to be past its bound";
  ASSERT_LE(c.solution.primal_infeasibility_scaled,
            options.get_double("primal_feasibility_tolerance"));
  EXPECT_LE(c.solution.complementarity_violation, tol::kComplementarity);

  Logger silent(nullptr);
  reconcile_status_with_measurement(c.model, &c.solution, options, silent, /*check_dual=*/true);
  EXPECT_EQ(c.solution.status, SolveStatus::kOptimal) << c.solution.message;
}

TEST(SolveStatusGuardComplementarity, ARowInsideItsBoundIsStillNotAProof) {
  Case c = priced_row(-kOffset);
  const Options options = quiet();
  ASSERT_EQ(c.solution.primal_infeasibility, 0.0);
  EXPECT_NEAR(c.solution.complementarity_violation, kPrice * kOffset, 1e-12);

  Logger silent(nullptr);
  reconcile_status_with_measurement(c.model, &c.solution, options, silent, /*check_dual=*/true);
  EXPECT_EQ(c.solution.status, SolveStatus::kFeasible);
  EXPECT_NE(c.solution.message.find("|multiplier| * slack"), std::string::npos)
      << c.solution.message;
}

/// #806: minimize -x subject to x <= 1e14, the row priced at y = -1, reported at 1e14 - gap.
Case huge_bound(double gap) {
  Case c;
  Model& model = c.model;
  model.col_lower = {0.0};
  model.col_upper = {kInfinity};
  model.col_cost = {-1.0};
  model.col_type = {VarType::kContinuous};
  model.row_lower = {-kInfinity};
  model.row_upper = {1e14};
  model.matrix.reset(1, 1);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.finalize();
  Solution& s = c.solution;
  s.status = SolveStatus::kOptimal;
  s.algorithm = "fabricated";
  s.col_value = {1e14 - gap};
  s.row_activity = {1e14 - gap};
  s.row_dual = {-1.0};
  s.col_dual = {0.0};
  s.recompute_quality(model);
  return c;
}

TEST(SolveStatusGuardComplementarity, RoundingOfAHugeRightHandSideIsNotAViolation) {
  // Klee-Minty n = 20's shape: a product of 1.6e-02 against terms of 1e14 is 1.6e-16 of
  // them, under the kComplementarityRounding allowance of 1e-12 * 1 * 1e14 = 100.
  Case c = huge_bound(0.015625);
  ASSERT_GT(c.solution.complementarity_violation, tol::kComplementarity);
  Logger silent(nullptr);
  reconcile_status_with_measurement(c.model, &c.solution, quiet(), silent, /*check_dual=*/true);
  EXPECT_EQ(c.solution.status, SolveStatus::kOptimal) << c.solution.message;
}

TEST(SolveStatusGuardComplementarity,
     ASlackAboveRoundingOfAHugeRightHandSideIsStillAViolation) {
  // 1e3 inside a bound of 1e14 is 1e-11 of it: ten times the rounding allowance, and the
  // absolute product (1e3) is far above kComplementarity. Not a proof.
  Case c = huge_bound(1e3);
  Logger silent(nullptr);
  reconcile_status_with_measurement(c.model, &c.solution, quiet(), silent, /*check_dual=*/true);
  EXPECT_EQ(c.solution.status, SolveStatus::kFeasible);
  EXPECT_NE(c.solution.message.find("|multiplier| * slack"), std::string::npos)
      << c.solution.message;
}

}  // namespace
}  // namespace sankhya
