// SPDX-License-Identifier: Apache-2.0
// SANKHYA - postsolve prices a doubleton's fill-in row for a column that left before the
// row was folded (#766).
//
// Netlib pilot.ja at unit scale. The doubleton MPSF03 eliminated PRPP03 and filled its
// partner NPSF03 into PRPP03's other rows, among them KRPR03. The elimination also collapsed
// NPSF03's box to a point, so NPSF03 was fixed next, and only after that was KRPR03 folded
// in turn. NPSF03 was never a recipient of that fold, so KRPR03's dual still belongs in its
// reduced cost (Andersen & Andersen 1995, "Presolving in linear programming", sec. 4). The
// reduced-cost helper used to skip every folded fill-in row regardless, priced MPSF03 at 0
// where 0.0506 was needed, and pilot.ja came back `feasible` with a dual violation of
// 5.1e-02. #790 fixed it (received_fold_of in postsolve) with a test that solves the whole
// file; this one pins the record sequence itself on four columns, so the case stays covered
// if pilot.ja's presolve ever stops producing it.

#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

#include "presolve/presolve.hpp"

namespace sankhya {
namespace {

using Kind = presolve::Record::Kind;

Options with_presolve(bool on) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", on);
  return options;
}

// Columns P, N, S, U, all >= 0; minimise -N + S + 3U.
//   A:  -2P - N     = 0   doubleton: P = -N/2 is eliminated, P >= 0 caps N at 0, so N is fixed
//   K:   -P + S + U = 4   gets N by fill-in (+0.5), and once N is fixed it is a doubleton too
// Optimum S = 4, everything else 0, objective 4. S is interior, so y_K = 1. N sits at its
// lower bound with d_N = -1 + y_A, so y_A >= 1. Without K's dual in N's reduced cost the
// doubleton A is priced 0.5 and d_N comes back -0.5 (measured: `feasible`, dual violation
// 5.000e-01, with the #790 change reverted).
Model pilot_ja_in_miniature() {
  Model model;
  model.col_cost = {0.0, -1.0, 1.0, 3.0};
  model.col_lower = {0.0, 0.0, 0.0, 0.0};
  model.col_upper = {kInfinity, kInfinity, kInfinity, kInfinity};
  model.col_type.assign(4, VarType::kContinuous);
  model.row_lower = {0.0, 4.0};
  model.row_upper = {0.0, 4.0};
  model.matrix.reset(2, 4);
  model.matrix.add_entry(0, 0, -2.0);
  model.matrix.add_entry(0, 1, -1.0);
  model.matrix.add_entry(1, 0, -1.0);
  model.matrix.add_entry(1, 2, 1.0);
  model.matrix.add_entry(1, 3, 1.0);
  model.matrix.finalize();
  model.hessian.reset(4, 4);
  model.hessian.finalize();
  return model;
}

TEST(PresolveFillIn, ARowFoldedAfterItsFillInColumnWasFixedStillPricesThatColumn) {
  const Model model = pilot_ja_in_miniature();

  // The structure has to be the pilot.ja one or the test proves nothing: the doubleton on A,
  // then N fixed, then K folded with N already gone.
  Options options = with_presolve(true);
  Logger logger(nullptr);
  const presolve::Result reduced = presolve::presolve(model, options, logger);
  std::vector<Kind> kinds;
  for (const presolve::Record& record : reduced.records) kinds.push_back(record.kind);
  const std::vector<Kind> expected = {Kind::kDoubletonEquation, Kind::kFixedColumn,
                                      Kind::kDoubletonEquation};
  ASSERT_GE(kinds.size(), expected.size());
  EXPECT_TRUE(std::equal(expected.begin(), expected.end(), kinds.begin()));
  EXPECT_EQ(reduced.records[0].index, 0);  // row A
  EXPECT_EQ(reduced.records[1].index, 1);  // column N
  EXPECT_EQ(reduced.records[2].index, 1);  // row K

  const Solution on = solve(model, with_presolve(true));
  ASSERT_EQ(on.status, SolveStatus::kOptimal) << on.message;
  EXPECT_NEAR(on.objective, 4.0, 1e-9);
  EXPECT_NEAR(on.col_value[2], 4.0, 1e-9);
  EXPECT_NEAR(on.row_dual[1], 1.0, 1e-9);
  EXPECT_GE(on.row_dual[0], 1.0 - tol::kDualFeasibility);
  EXPECT_GE(on.col_dual[1], -tol::kDualFeasibility) << "N at its lower bound";
  EXPECT_LE(on.dual_infeasibility, tol::kDualFeasibility) << on.message;

  const Solution off = solve(model, with_presolve(false));
  ASSERT_EQ(off.status, SolveStatus::kOptimal) << off.message;
  EXPECT_NEAR(on.objective, off.objective, 1e-9);
}

}  // namespace
}  // namespace sankhya
