// SPDX-License-Identifier: Apache-2.0
// SANKHYA - how the interior point takes the n x n side of its normal equations (#469).
//
// Held here, one test per rule the reviews of #616 and #620 set:
//   - an unconverged column-side solve is not used as a direction: it sets the same flag an
//     unconverged dense-column solve sets, and gets the same raise / refactorize / stop;
//   - the count before assembly is of the system actually formed, so a model whose m side
//     is over the factor budget and whose n side is not (one dense column) is solved on the
//     n side - by `columns`, and by `auto` as a rescue - rather than declined;
//   - the n x n side is refused, with a log line, under ipm_proximal_regularization (no
//     normal equations) and under ipm_dense_columns (built on the row side's factor);
//   - the choice made by `auto` answers to the set-up's share of the time limit.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ipm/column_side.hpp"
#include "ipm/ipm_testing.hpp"
#include "la/ldl.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/sparse.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya {
namespace {

Options ipm_options(const char* side) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("algorithm", "ipm");
  options.set_string("ipm_normal_side", side);
  return options;
}

/// Runs a solve with the log at `level` captured from the console.
Solution solve_logged(const Model& model, Options options, const char* level,
                      std::string* log) {
  options.set_bool("log_to_console", true);
  options.set_string("log_level", level);
  ::testing::internal::CaptureStdout();
  const Solution solution = solve(model, options);
  *log = ::testing::internal::GetCapturedStdout();
  return solution;
}

/// 600 inequality rows over 20 columns, min c x s.t. A x >= 1, x >= 0, four columns a row:
/// the rows outnumber the columns, so `auto` compares the two sides.
Model tall_cover(std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<Index> pick(0, 19);
  std::uniform_real_distribution<double> value(0.5, 2.0);
  const Index m = 600;
  const Index n = 20;
  Model model;
  model.sense = ObjSense::kMinimize;
  model.matrix.reset(m, n);
  for (Index i = 0; i < m; ++i) {
    std::vector<Index> used;
    while (used.size() < 4) {
      const Index j = pick(rng);
      if (std::find(used.begin(), used.end(), j) == used.end()) used.push_back(j);
    }
    for (const Index j : used) model.matrix.add_entry(i, j, value(rng));
  }
  model.matrix.finalize();
  model.col_cost.resize(static_cast<std::size_t>(n));
  for (double& c : model.col_cost) c = value(rng);
  model.col_lower.assign(static_cast<std::size_t>(n), 0.0);
  model.col_upper.assign(static_cast<std::size_t>(n), kInfinity);
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kContinuous);
  model.row_lower.assign(static_cast<std::size_t>(m), 1.0);
  model.row_upper.assign(static_cast<std::size_t>(m), kInfinity);
  return model;
}

/// min sum x_i + 2 y s.t. x_i + y >= 1 for m rows, x, y >= 0; the optimum is 2 (y = 1).
/// The column y meets every row: A Theta A^T is dense (m (m + 1) / 2 lower entries), while
/// A^T A is an arrow, m + 1 diagonals and m off-diagonals, which factors without fill.
Model one_dense_column(Index m) {
  Model model;
  model.sense = ObjSense::kMinimize;
  model.matrix.reset(m, m + 1);
  for (Index i = 0; i < m; ++i) {
    model.matrix.add_entry(i, i, 1.0);
    model.matrix.add_entry(i, m, 1.0);
  }
  model.matrix.finalize();
  model.col_cost.assign(static_cast<std::size_t>(m + 1), 1.0);
  model.col_cost[static_cast<std::size_t>(m)] = 2.0;
  model.col_lower.assign(static_cast<std::size_t>(m + 1), 0.0);
  model.col_upper.assign(static_cast<std::size_t>(m + 1), kInfinity);
  model.col_type.assign(static_cast<std::size_t>(m + 1), VarType::kContinuous);
  model.row_lower.assign(static_cast<std::size_t>(m), 1.0);
  model.row_upper.assign(static_cast<std::size_t>(m), kInfinity);
  return model;
}

/// The one-dense-column model at 3000 rows under a factor budget of 1e6: the row side's
/// 4.5e6 lower entries are over it, the column side's 6001 are not.
Options over_budget(const char* side) {
  Options options = ipm_options(side);
  options.set_bool("presolve", false);
  options.set_int("ipm_max_factor_nonzeros", 1000000);
  return options;
}

TEST(IpmNormalSide, TheOptionDefaultsToTheRowSide) {
  const Options options;
  EXPECT_EQ(options.get_string("ipm_normal_side"), "rows");
}

// ---- 1. An unconverged column-side solve -------------------------------------------------

TEST(IpmNormalSide, AnUnconvergedSolveIsReportedAsSuch) {
  // The preconditioner built from the factors of N for one Theta, applied to M for Theta
  // eight decades away: conjugate gradients cannot reach the accepted backward error in
  // their budget, and the report must say so rather than hand back the iterate as solved.
  std::mt19937_64 rng(4690);
  const Index m = 600;
  const Index n = 150;
  SparseMatrix a(m, n);
  std::uniform_int_distribution<Index> pick(0, n - 1);
  std::uniform_real_distribution<double> value(0.5, 2.0);
  for (Index i = 0; i < m; ++i) {
    const Index first = pick(rng);
    Index second = pick(rng);
    while (second == first) second = pick(rng);
    a.add_entry(i, first, value(rng));
    a.add_entry(i, second, -value(rng));
  }
  a.finalize(0.0);
  std::uniform_real_distribution<double> decades(-4.0, 4.0);
  std::vector<double> theta(static_cast<std::size_t>(n));
  for (double& t : theta) t = std::pow(10.0, decades(rng));
  const std::vector<double> shift(static_cast<std::size_t>(m), 1e-6);
  const std::vector<bool> fixed(static_cast<std::size_t>(n), false);

  ipm::ColumnSide side;
  side.set_matrix(a, fixed);
  const std::vector<double> wrong_theta(static_cast<std::size_t>(n), 1.0);
  SparseMatrix lower;
  ASSERT_TRUE(side.assemble(wrong_theta, shift, 1e-10, &lower, {}));
  SparseLdl factors;
  ASSERT_TRUE(factors.analyze(lower));
  ASSERT_TRUE(factors.factorize(lower, 1e-10));
  SparseMatrix unused;
  ASSERT_TRUE(side.assemble(theta, shift, 1e-10, &unused, {}));  // M's Theta, N's stale
  std::vector<double> x(static_cast<std::size_t>(m));
  for (double& v : x) v = decades(rng);
  const ipm::ColumnSideReport report = side.solve(factors, x.data());
  EXPECT_FALSE(report.converged) << "backward error " << report.backward_error << " after "
                                 << report.iterations << " steps";
  EXPECT_GT(report.backward_error, tol::kIpmPcgAcceptedBackwardError);
}

TEST(IpmNormalSide, AnUnconvergedSolveRaisesTheRegularizationAndIsNotTaken) {
  const Model model = tall_cover(469);
  // One rejected solve: the step is recomputed after a raise, and the solve still ends at
  // the optimum.
  ipm::testing::reject_next_column_side_solves = 1;
  std::string log;
  const Solution once = solve_logged(model, ipm_options("columns"), "verbose", &log);
  ipm::testing::reject_next_column_side_solves = 0;
  EXPECT_NE(log.find("column-side conjugate gradients did not converge"), std::string::npos)
      << log;
  EXPECT_NE(log.find("inaccurate (conjugate gradients did not converge)"), std::string::npos)
      << log;
  EXPECT_NE(log.find("1 solve(s) not converged"), std::string::npos) << log;
  ASSERT_EQ(once.status, SolveStatus::kOptimal) << once.message;

  // Every solve rejected: no direction is ever taken, and the stop says why.
  ipm::testing::reject_next_column_side_solves = 1000000;
  const Solution never = solve(model, ipm_options("columns"));
  ipm::testing::reject_next_column_side_solves = 0;
  EXPECT_NE(never.status, SolveStatus::kOptimal) << never.message;
  EXPECT_NE(never.message.find("column-side conjugate gradients did not converge"),
            std::string::npos)
      << never.message;
}

// ---- 2. The count before assembly is of the system formed -------------------------------

TEST(IpmNormalSide, EachSideIsCountedAsTheSystemItForms) {
  const Model model = one_dense_column(3000);
  ipm::ColumnSide side;
  side.set_matrix(model.matrix, std::vector<bool>(3001, false));
  const NormalPrediction columns = side.predict_column_side(1000000, {});
  EXPECT_FALSE(columns.over_cap);
  EXPECT_LE(columns.nonzeros, 6001);
  EXPECT_FALSE(side.predict_column_side(6001, {}).over_cap);
  EXPECT_TRUE(side.predict_column_side(6000, {}).over_cap);
  EXPECT_TRUE(side.predict_row_side(1000000, {}).over_cap);
}

TEST(IpmNormalSide, TheColumnSideIsNotDeclinedOnTheRowSidesCount) {
  const Model model = one_dense_column(3000);
  const Solution rows = solve(model, over_budget("rows"));
  EXPECT_EQ(rows.status, SolveStatus::kNumericalError) << rows.message;
  EXPECT_NE(rows.message.find("declined before assembly"), std::string::npos) << rows.message;

  const Solution columns = solve(model, over_budget("columns"));
  EXPECT_EQ(columns.message.find("declined"), std::string::npos) << columns.message;
  ASSERT_EQ(columns.status, SolveStatus::kOptimal) << columns.message;
  EXPECT_NEAR(columns.objective, 2.0, 1e-7);

  // auto, with the columns outnumbering the rows, takes the n side only as a rescue.
  std::string log;
  const Solution rescued = solve_logged(model, over_budget("auto"), "verbose", &log);
  EXPECT_NE(log.find("over the factor budget 1000000 before assembly"), std::string::npos)
      << log;
  EXPECT_NE(log.find("normal equations on the column side"), std::string::npos) << log;
  ASSERT_EQ(rescued.status, SolveStatus::kOptimal) << rescued.message;
  EXPECT_NEAR(rescued.objective, 2.0, 1e-7);

  // Within the budget, auto on the same shape keeps the row side and never counts N.
  const Model small = one_dense_column(200);
  const Solution kept = solve_logged(small, over_budget("auto"), "verbose", &log);
  EXPECT_NE(log.find("fit the factor budget"), std::string::npos) << log;
  EXPECT_NE(log.find("normal equations on the row side"), std::string::npos) << log;
  ASSERT_EQ(kept.status, SolveStatus::kOptimal) << kept.message;
  EXPECT_NEAR(kept.objective, 2.0, 1e-7);
}

// ---- 3. Mutually exclusive with the proximal and the dense-column paths ------------------

TEST(IpmNormalSide, TheProximalPathRefusesTheColumnSide) {
  const Model model = tall_cover(469);
  Options options = ipm_options("columns");
  options.set_bool("ipm_proximal_regularization", true);
  std::string log;
  const Solution s = solve_logged(model, options, "info", &log);
  EXPECT_NE(log.find("ipm_normal_side = columns does not apply with "
                     "ipm_proximal_regularization"),
            std::string::npos)
      << log;
  EXPECT_EQ(log.find("normal equations on the column side"), std::string::npos) << log;
  EXPECT_EQ(log.find("n x n side (#469)"), std::string::npos) << log;
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
}

TEST(IpmNormalSide, TheDenseColumnPathRefusesTheColumnSide) {
  const Model model = one_dense_column(3000);
  for (const char* side : {"columns", "auto"}) {
    Options options = over_budget(side);
    options.set_bool("ipm_dense_columns", true);
    std::string log;
    const Solution s = solve_logged(model, options, "info", &log);
    EXPECT_NE(log.find(std::string("ipm_normal_side = ") + side +
                       " does not combine with ipm_dense_columns"),
              std::string::npos)
        << log;
    EXPECT_EQ(log.find("normal equations on the column side"), std::string::npos) << log;
    EXPECT_NE(log.find("split off the normal equations"), std::string::npos) << log;
    ASSERT_EQ(s.status, SolveStatus::kOptimal) << side << ": " << s.message;
    EXPECT_NEAR(s.objective, 2.0, 1e-7) << side;
  }
}

// ---- 4. The choice answers to the set-up's deadline --------------------------------------

TEST(IpmNormalSide, TheChoiceAnswersToTheSetupShare) {
  const Model model = tall_cover(469);
  // A share of 0 puts the set-up's deadline at the moment the build starts, so it has passed
  // before the choice begins: the choice must see it, not order both sides first.
  Options options = ipm_options("auto");
  options.set_double("time_limit", 1000.0);
  options.set_double("ipm_setup_share", 0.0);
  std::string log;
  const Solution declined = solve_logged(model, options, "verbose", &log);
  EXPECT_EQ(declined.status, SolveStatus::kNotSolved) << declined.message;
  EXPECT_NE(declined.message.find("choosing the side of the normal equations"),
            std::string::npos)
      << declined.message;
  EXPECT_EQ(declined.iterations, 0);
  EXPECT_EQ(log.find("the column side is taken"), std::string::npos) << log;
  EXPECT_EQ(log.find("the row side is kept"), std::string::npos) << log;

  // With the whole limit for the set-up, the same choice is made and the model solved.
  options.set_double("ipm_setup_share", 1.0);
  const Solution solved = solve_logged(model, options, "verbose", &log);
  EXPECT_NE(log.find("the column side is taken"), std::string::npos) << log;
  ASSERT_EQ(solved.status, SolveStatus::kOptimal) << solved.message;
}

}  // namespace
}  // namespace sankhya
