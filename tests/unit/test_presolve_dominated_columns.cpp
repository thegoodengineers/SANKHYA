// SPDX-License-Identifier: Apache-2.0
// SANKHYA - dominated columns in presolve (#412).
//
// A column fixed by a dominance argument that does not hold is a confident, feasible-looking
// answer to a different problem, so the hand examples check that the fixing fires exactly
// where the argument says it may - each of the two fixings, the integrality condition on the
// move - and stays out where it does not, and random instances are then solved with the
// reduction on against the exact rational oracle, LP and MILP, judged the way the fuzz
// harness judges the simplex: same status, same optimum.

#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "oracles/lp_generator.hpp"
#include "oracles/rational_simplex.hpp"
#include "presolve/presolve.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

Options with_dominated_columns(bool on) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", true);
  options.set_bool("presolve_dominated_columns", on);
  options.set_int("node_limit", 100000);
  return options;
}

Model build(const std::vector<std::vector<double>>& rows, const std::vector<double>& row_lower,
            const std::vector<double>& row_upper, const std::vector<double>& cost,
            const std::vector<double>& col_lower, const std::vector<double>& col_upper,
            bool maximize = false, const std::vector<bool>& integer = {}) {
  Model model;
  const auto n = static_cast<Index>(cost.size());
  const auto m = static_cast<Index>(rows.size());
  model.sense = maximize ? ObjSense::kMaximize : ObjSense::kMinimize;
  model.col_cost = cost;
  model.col_lower = col_lower;
  model.col_upper = col_upper;
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kContinuous);
  for (std::size_t j = 0; j < integer.size(); ++j) {
    if (integer[j]) model.col_type[j] = VarType::kInteger;
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
  model.hessian.reset(n, n);
  model.hessian.finalize();
  EXPECT_EQ(model.validate(), "");
  return model;
}

Count dominated(const Model& model, bool on) {
  Logger logger(nullptr);
  return presolve::presolve(model, with_dominated_columns(on), logger).report.dominated_columns;
}

TEST(PresolveDominatedColumns, TheDominatedColumnIsFixedAtItsLowerBoundWhenTheOtherHasRoom) {
  // min x0 + 2 x1  s.t.  x0 + x1 >= 3,  x0 >= 0 unbounded above, 0 <= x1 <= 5. x0 is as
  // cheap and as helpful as x1 in the one row and can absorb whatever x1 carried, so some
  // optimum has x1 at 0: it is fixed there and comes back nonbasic at its lower bound. The
  // answer is 3 either way, and postsolve checks the point against the original model.
  const Model model = build({{1, 1}}, {3.0}, {kInf}, {1, 2}, {0, 0}, {kInf, 5});
  EXPECT_EQ(dominated(model, true), 1);
  EXPECT_EQ(dominated(model, false), 0) << "off means off";
  const Solution on = solve(model, with_dominated_columns(true));
  const Solution off = solve(model, with_dominated_columns(false));
  ASSERT_EQ(on.status, SolveStatus::kOptimal) << on.message;
  ASSERT_EQ(off.status, SolveStatus::kOptimal) << off.message;
  EXPECT_NEAR(on.objective, 3.0, 1e-9);
  EXPECT_NEAR(off.objective, 3.0, 1e-9);
  ASSERT_EQ(on.col_value.size(), 2u);
  EXPECT_EQ(on.col_value[1], 0.0);
  EXPECT_EQ(on.col_status[1], BasisStatus::kAtLower);
  EXPECT_LE(on.primal_infeasibility, 1e-9);
  EXPECT_LE(on.dual_infeasibility, 1e-7) << "a column fixed at its lower bound needs a "
                                            "non-negative reduced cost, and this one has it";
}

TEST(PresolveDominatedColumns, TheDominatingColumnIsFixedAtItsUpperBoundWhenTheOtherHasRoom) {
  // max x0 + x1  s.t.  x0 + x1 <= 4,  0 <= x0 <= 2,  x1 <= 10 unbounded below. Each column
  // dominates the other (same cost, same entry), and only one fixing has room: x1 can go
  // down without limit, so x0 can always go up to 2 and some optimum has it there. The
  // other direction has no room - x0's upper bound and x1's lower bound are both finite
  // or both the wrong way - so exactly one column is fixed.
  const Model model = build({{1, 1}}, {-kInf}, {4.0}, {1, 1}, {0, -kInf}, {2, 10},
                            /*maximize=*/true);
  EXPECT_EQ(dominated(model, true), 1);
  const Solution on = solve(model, with_dominated_columns(true));
  ASSERT_EQ(on.status, SolveStatus::kOptimal) << on.message;
  EXPECT_NEAR(on.objective, 4.0, 1e-9);
  EXPECT_EQ(on.col_value[0], 2.0);
  EXPECT_EQ(on.col_status[0], BasisStatus::kAtUpper);
  EXPECT_LE(on.primal_infeasibility, 1e-9);
  EXPECT_LE(on.dual_infeasibility, 1e-7);
}

TEST(PresolveDominatedColumns, StaysOutWhereTheArgumentDoesNotHold) {
  // A row bounded on both sides with different entries: the move would change its activity.
  const Model ranged = build({{1, 2}}, {1.0}, {3.0}, {1, 1}, {0, 0}, {kInf, 5});
  EXPECT_EQ(dominated(ranged, true), 0);
  // The cheaper column is the less helpful one in a row bounded above: min x0 + 2 x1 with
  // 2 x0 + x1 <= 4. x0 is cheaper but its entry is larger, x1 is more helpful but dearer.
  const Model crossed = build({{2, 1}}, {-kInf}, {4.0}, {1, 2}, {0, 0}, {kInf, 5});
  EXPECT_EQ(dominated(crossed, true), 0);
  // Different row supports are never compared: x1 also sits in a second row, shared with
  // x2 so that the row is not a singleton the bound reduction would fold away first.
  const Model supports = build({{1, 1, 0}, {0, 1, 1}}, {3.0, -kInf}, {kInf, 2.0}, {1, 2, 0},
                               {0, 0, 0}, {kInf, 5, 5});
  EXPECT_EQ(dominated(supports, true), 0);
  // No room for either move: both upper bounds finite, both lower bounds finite.
  const Model boxed = build({{1, 1}}, {3.0}, {kInf}, {1, 2}, {0, 0}, {5, 5});
  EXPECT_EQ(dominated(boxed, true), 0);
}

TEST(PresolveDominatedColumns, TheMoveMustKeepIntegrality) {
  // min x0 + 2 x1  s.t.  x0 + x1 >= 3, x0 integer and unbounded above, x1 continuous in
  // [0, 5]. Fixing x1 at 0 would move a fractional amount onto the integer x0, so the
  // fixing is declined; with x1 integer as well the move is integral and it fires.
  const Model mixed = build({{1, 1}}, {3.0}, {kInf}, {1, 2}, {0, 0}, {kInf, 5},
                            /*maximize=*/false, {true, false});
  EXPECT_EQ(dominated(mixed, true), 0);
  const Model both = build({{1, 1}}, {3.0}, {kInf}, {1, 2}, {0, 0}, {kInf, 5},
                           /*maximize=*/false, {true, true});
  EXPECT_EQ(dominated(both, true), 1);
  const Solution on = solve(both, with_dominated_columns(true));
  ASSERT_EQ(on.status, SolveStatus::kOptimal) << on.message;
  EXPECT_NEAR(on.objective, 3.0, 1e-9);
  EXPECT_EQ(on.col_value[1], 0.0);
  // The other way round: a continuous dominating column may absorb from an integer one.
  const Model absorbs = build({{1, 1}}, {3.0}, {kInf}, {1, 2}, {0, 0}, {kInf, 5},
                              /*maximize=*/false, {false, true});
  EXPECT_EQ(dominated(absorbs, true), 1);
}

/// One random instance against the oracle, the fuzz harness's own comparison: the oracle's
/// status is the truth, its optimum to a relative 1e-6.
bool agree(const oracle::OracleResult& exact, const Solution& got, std::string* why) {
  switch (exact.status) {
    case oracle::OracleStatus::kOptimal: {
      if (got.status != SolveStatus::kOptimal) {
        *why = "the exact optimum exists but the solver said " +
               std::string(to_string(got.status)) + ": " + got.message;
        return false;
      }
      const double expected = exact.objective.to_double();
      const double difference = std::fabs(got.objective - expected);
      if (difference > 1e-6 * std::max(1.0, std::fabs(expected))) {
        *why = "objectives differ by " + std::to_string(difference) + " (" +
               std::to_string(got.objective) + " against " + std::to_string(expected) + ")";
        return false;
      }
      return true;
    }
    case oracle::OracleStatus::kInfeasible:
      if (got.status != SolveStatus::kInfeasible) {
        *why = "exactly infeasible, the solver said " + std::string(to_string(got.status));
        return false;
      }
      return true;
    case oracle::OracleStatus::kUnbounded:
      if (got.status != SolveStatus::kUnbounded) {
        *why = "exactly unbounded, the solver said " + std::string(to_string(got.status));
        return false;
      }
      return true;
    case oracle::OracleStatus::kOverflow:
    case oracle::OracleStatus::kIterationLimit: return true;
  }
  return true;
}

/// Append a column that shadows a random existing one: the same row support with each entry
/// nudged, a nudged cost, and a random upper bound. The generator alone rarely produces two
/// columns on the same rows with one dominating the other; this makes a candidate pair on
/// most instances, in either direction, and the oracle judges the augmented instance whole.
void add_a_shadow_column(oracle::GeneratedLp* lp, std::mt19937_64& rng) {
  if (lp->num_cols == 0 || lp->num_rows == 0) return;
  std::uniform_int_distribution<Index> pick(0, lp->num_cols - 1);
  std::uniform_int_distribution<int> nudge(-1, 1);
  std::uniform_int_distribution<int> coin(0, 1);
  const auto source = static_cast<std::size_t>(pick(rng));
  for (Index i = 0; i < lp->num_rows; ++i) {
    auto& row = lp->a[static_cast<std::size_t>(i)];
    const std::int64_t entry = row[source];
    std::int64_t shadow = entry == 0 ? 0 : entry + nudge(rng);
    if (entry != 0 && shadow == 0) shadow = entry;  // keep the support identical
    row.push_back(shadow);
  }
  lp->c.push_back(lp->c[source] + nudge(rng));
  lp->upper.push_back(coin(rng) == 0 ? oracle::kNoUpperBound : 3 + coin(rng));
  if (!lp->integral.empty()) lp->integral.push_back(static_cast<char>(coin(rng)));
  if (!lp->lower.empty()) lp->lower.push_back(0);
  ++lp->num_cols;
}

TEST(PresolveDominatedColumns, RandomLpsAgreeWithTheExactOracle) {
  std::mt19937_64 rng(4123);
  oracle::GeneratorConfig config;
  int compared = 0;
  int skipped = 0;
  Count fixed_total = 0;
  for (int trial = 0; trial < 400; ++trial) {
    oracle::GeneratedLp lp = oracle::random_lp(rng, config);
    add_a_shadow_column(&lp, rng);
    const oracle::OracleResult exact = oracle::solve_exact(lp);
    if (exact.status == oracle::OracleStatus::kOverflow ||
        exact.status == oracle::OracleStatus::kIterationLimit) {
      ++skipped;
      continue;
    }
    const Model model = oracle::to_model(lp);
    const Solution got = solve(model, with_dominated_columns(true));
    fixed_total += got.presolve_report.dominated_columns;
    std::string why;
    EXPECT_TRUE(agree(exact, got, &why)) << "trial " << trial << ": " << why << "\n"
                                         << lp.to_text();
    ++compared;
  }
  EXPECT_GT(compared, 300) << skipped << " skipped";
  EXPECT_GT(fixed_total, 0) << "the generator should give the reduction something to fix";
}

TEST(PresolveDominatedColumns, RandomMilpsAgreeWithTheExactOracle) {
  // The argument is about any feasible point and the move is checked for integrality, so
  // it holds for integer columns as it does for continuous ones; the exact branch and
  // bound is the judge here.
  std::mt19937_64 rng(41230);
  oracle::GeneratorConfig config;
  config.max_rows = 5;
  config.max_cols = 6;
  int compared = 0;
  Count fixed_total = 0;
  for (int trial = 0; trial < 200; ++trial) {
    oracle::GeneratedLp lp = oracle::random_lp(rng, config);
    lp.integral.assign(static_cast<std::size_t>(lp.num_cols), 0);
    for (Index j = 0; j < lp.num_cols; ++j) {
      lp.integral[static_cast<std::size_t>(j)] = (j + trial) % 2 == 0 ? 1 : 0;
    }
    add_a_shadow_column(&lp, rng);
    const oracle::OracleResult exact = oracle::solve_exact_milp(lp, 20000);
    if (exact.status != oracle::OracleStatus::kOptimal &&
        exact.status != oracle::OracleStatus::kInfeasible) {
      continue;  // unbounded or abstained: nothing this test can compare
    }
    // to_model() builds the LP; the integrality the exact branch and bound honoured is
    // put on the model here, so the two engines see one MILP.
    Model model = oracle::to_model(lp);
    for (Index j = 0; j < lp.num_cols; ++j) {
      if (lp.integral[static_cast<std::size_t>(j)] != 0) {
        model.col_type[static_cast<std::size_t>(j)] = VarType::kInteger;
      }
    }
    const Solution got = solve(model, with_dominated_columns(true));
    fixed_total += got.presolve_report.dominated_columns;
    std::string why;
    EXPECT_TRUE(agree(exact, got, &why)) << "trial " << trial << ": " << why << "\n"
                                         << lp.to_text();
    ++compared;
  }
  EXPECT_GT(compared, 100);
  EXPECT_GT(fixed_total, 0);
}

}  // namespace
}  // namespace sankhya
