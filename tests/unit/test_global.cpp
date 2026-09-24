// SPDX-License-Identifier: Apache-2.0
// SANKHYA - quadratic rows (QCMATRIX) and the spatial branch and bound (#514).
//
// Every optimum below is known by hand. The one that matters most is Haverly's pooling
// problem started where the recursion he studied gets stuck: the local method reports a
// cost of -100 there, and the global method has to find -400 and prove it.

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "global/global_internal.hpp"
#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/qcqp.hpp"

#include "support/temp_file.hpp"

namespace sankhya {
namespace {

using testing::TempFile;

// Haverly (1978) case 1, P-formulation: crude A (3 %, cost 6) and B (1 %, cost 16) into one
// pool, C (2 %, cost 10) direct; X sells at 9 (<= 2.5 %, <= 100), Y at 15 (<= 1.5 %, <= 200).
// Columns f_u_v are arc flows, p_4_1 the pool's quality. Written the way
// bench/runners/pooling_models.py writes it (#516).
const char* const kHaverly1 =
    "NAME haverly1_p\n"
    "ROWS\n"
    " N cost\n L src_1\n L src_2\n L src_3\n L dem_5\n L dem_6\n E bal_4\n L cap_4\n"
    " E pq_4_1\n L tq_5_1\n L tq_6_1\n"
    "COLUMNS\n"
    " f_1_4 cost 6\n f_1_4 src_1 1\n f_1_4 bal_4 1\n f_1_4 cap_4 1\n f_1_4 pq_4_1 3\n"
    " f_2_4 cost 16\n f_2_4 src_2 1\n f_2_4 bal_4 1\n f_2_4 cap_4 1\n f_2_4 pq_4_1 1\n"
    " f_3_5 cost 1\n f_3_5 src_3 1\n f_3_5 dem_5 1\n f_3_5 tq_5_1 -0.5\n"
    " f_3_6 cost -5\n f_3_6 src_3 1\n f_3_6 dem_6 1\n f_3_6 tq_6_1 0.5\n"
    " f_4_5 cost -9\n f_4_5 dem_5 1\n f_4_5 bal_4 -1\n f_4_5 tq_5_1 -2.5\n"
    " f_4_6 cost -15\n f_4_6 dem_6 1\n f_4_6 bal_4 -1\n f_4_6 tq_6_1 -1.5\n"
    " p_4_1 cost 0\n"
    "RHS\n"
    " rhs src_1 300\n rhs src_2 300\n rhs src_3 300\n rhs dem_5 100\n rhs dem_6 200\n"
    " rhs cap_4 300\n"
    "BOUNDS\n"
    " UP bnd f_1_4 300\n UP bnd f_2_4 300\n UP bnd f_3_5 100\n UP bnd f_3_6 200\n"
    " UP bnd f_4_5 100\n UP bnd f_4_6 200\n LO bnd p_4_1 1\n UP bnd p_4_1 3\n"
    "QCMATRIX pq_4_1\n"
    " f_4_5 p_4_1 -0.5\n p_4_1 f_4_5 -0.5\n f_4_6 p_4_1 -0.5\n p_4_1 f_4_6 -0.5\n"
    "QCMATRIX tq_5_1\n"
    " f_4_5 p_4_1 0.5\n p_4_1 f_4_5 0.5\n"
    "QCMATRIX tq_6_1\n"
    " f_4_6 p_4_1 0.5\n p_4_1 f_4_6 0.5\n"
    "ENDATA\n";

QcqpModel read_or_fail(const std::string& text) {
  const TempFile file(text);
  QcqpModel model;
  const io::ReadResult result = io::read_qcqp_mps(file.path(), &model);
  EXPECT_TRUE(result.ok) << result.error;
  return model;
}

Index column_named(const Model& model, const std::string& name) {
  for (std::size_t j = 0; j < model.col_names.size(); ++j) {
    if (model.col_names[j] == name) return static_cast<Index>(j);
  }
  return -1;
}

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  return options;
}

/// One row: lower <= linear + products <= upper.
struct RowSpec {
  std::vector<std::pair<Index, double>> linear;
  std::vector<QuadraticTerm> products;  ///< `row` is filled in by make_model
  double lower;
  double upper;
};

QcqpModel make_model(Index n, double lower, double upper, const std::vector<double>& cost,
                     const std::vector<RowSpec>& rows) {
  QcqpModel model;
  Model& lin = model.linear;
  const auto m = static_cast<Index>(rows.size());
  lin.resize_columns(n);
  lin.resize_rows(m);
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    lin.col_lower[u] = lower;
    lin.col_upper[u] = upper;
    lin.col_cost[u] = cost[u];
  }
  lin.matrix.reset(m, n);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    lin.row_lower[i] = rows[i].lower;
    lin.row_upper[i] = rows[i].upper;
    for (const auto& [column, value] : rows[i].linear) {
      lin.matrix.add_entry(static_cast<Index>(i), column, value);
    }
    for (QuadraticTerm term : rows[i].products) {
      term.row = static_cast<Index>(i);
      model.quadratic.push_back(term);
    }
  }
  lin.matrix.finalize(0.0);
  lin.hessian.reset(n, n);
  lin.hessian.finalize(0.0);
  return model;
}

struct HessianTriplet {
  Index row;
  Index col;
  double value;
};

void set_hessian(Model* model, const std::vector<HessianTriplet>& entries) {
  model->hessian.reset(model->num_cols(), model->num_cols());
  for (const HessianTriplet& entry : entries) {
    model->hessian.add_entry(entry.row, entry.col, entry.value);
  }
  model->hessian.finalize(0.0);
}

QuadraticTerm product(Index a, Index b, double value) {
  return {0, a, b, value};
}

// ---- the reader ---------------------------------------------------------------------------

TEST(QcqpReader, FoldsTheSymmetricPairIntoOneTermPerProduct) {
  const QcqpModel model = read_or_fail(kHaverly1);
  const Index p = column_named(model.linear, "p_4_1");
  const Index f45 = column_named(model.linear, "f_4_5");
  const Index f46 = column_named(model.linear, "f_4_6");
  ASSERT_EQ(model.quadratic.size(), 4u);  // pq: two products; tq_5_1 and tq_6_1: one each
  double pool_45 = 0.0;
  double spec_46 = 0.0;
  for (const QuadraticTerm& term : model.quadratic) {
    EXPECT_LE(term.first, term.second);
    const std::string& row = model.linear.row_names[static_cast<std::size_t>(term.row)];
    if (row == "pq_4_1" && term.first == std::min(p, f45) && term.second == std::max(p, f45)) {
      pool_45 = term.value;
    }
    if (row == "tq_6_1" && term.first == std::min(p, f46) && term.second == std::max(p, f46)) {
      spec_46 = term.value;
    }
  }
  // (p, f) and (f, p) of -0.5 each: the product p * f with coefficient -1, not -0.5 or -2.
  EXPECT_DOUBLE_EQ(pool_45, -1.0);
  EXPECT_DOUBLE_EQ(spec_46, 1.0);
  // Without a sink, the same file is still refused by name, and the refusal says how to
  // ask for the global method.
  const TempFile file(kHaverly1);
  Model plain;
  const io::ReadResult refused = io::read_mps(file.path(), &plain);
  EXPECT_FALSE(refused.ok);
  EXPECT_TRUE(refused.refused);
  EXPECT_NE(refused.error.find("nonconvex=global"), std::string::npos) << refused.error;
}

TEST(QcqpReader, RefusesARepeatedEntryTheObjectiveRowAndAnUnknownColumn) {
  const std::string head =
      "NAME t\nROWS\n N obj\n L r\nCOLUMNS\n x obj 1\n x r 1\n y r 1\nRHS\n rhs r 4\n";
  for (const std::string& tail : {std::string("QCMATRIX r\n x y 1\n x y 1\nENDATA\n"),
                                  std::string("QCMATRIX obj\n x y 1\n y x 1\nENDATA\n"),
                                  std::string("QCMATRIX r\n x z 1\nENDATA\n")}) {
    const TempFile file(head + tail);
    QcqpModel model;
    const io::ReadResult result = io::read_qcqp_mps(file.path(), &model);
    EXPECT_FALSE(result.ok) << tail;
  }
}

TEST(QcqpReader, ADiagonalEntryIsASquareAndCancellingEntriesAreNoTerm) {
  const QcqpModel model = read_or_fail(
      "NAME t\nROWS\n N obj\n L r\nCOLUMNS\n x obj 1\n x r 1\n y r 1\nRHS\n rhs r 4\n"
      "QCMATRIX r\n x x 2\n x y 1.5\n y x -1.5\nENDATA\n");
  ASSERT_EQ(model.quadratic.size(), 1u);
  EXPECT_EQ(model.quadratic[0].first, model.quadratic[0].second);
  EXPECT_DOUBLE_EQ(model.quadratic[0].value, 2.0);
  const std::vector<double> x{3.0, 5.0};
  EXPECT_DOUBLE_EQ(model.row_activity(x)[0], 3.0 + 5.0 + 2.0 * 9.0);
}

// ---- the pieces ---------------------------------------------------------------------------

TEST(GlobalPieces, TheMultiplierBoundMatchesTheRootLpAndHoldsForAnyMultipliers) {
  const QcqpModel model = read_or_fail(kHaverly1);
  const global::Problem problem = global::build_problem(model);
  const global::Box box{problem.col_lower, problem.col_upper};
  const Model lp = global::build_relaxation(problem, box);
  const Solution relaxed = global::solve_lp(lp, global::inner_lp_options(quiet(), kInfinity));
  ASSERT_EQ(relaxed.status, SolveStatus::kOptimal) << relaxed.message;
  const double bound = global::dual_bound_from_multipliers(lp, relaxed.row_dual);
  EXPECT_NEAR(bound, relaxed.objective, 1e-6 * std::max(1.0, std::fabs(relaxed.objective)));
  EXPECT_LE(bound, -400.0 + 1e-9);  // a relaxation bound is below the global optimum
  // Weak duality: ANY multipliers bound the LP from below.
  std::vector<double> perturbed = relaxed.row_dual;
  for (std::size_t i = 0; i < perturbed.size(); ++i) perturbed[i] += (i % 3 == 0) ? 0.7 : -0.4;
  EXPECT_LE(global::dual_bound_from_multipliers(lp, perturbed), relaxed.objective + 1e-9);
}

TEST(GlobalPieces, BoundTighteningDividesThroughAProductAndProvesEmptiness) {
  // x * y = 6 with x in [1, 4]: y in [1.5, 6] from y in [0, 10].
  QcqpModel model =
      make_model(2, 0.0, 10.0, {0.0, 0.0}, {{{}, {product(0, 1, 1.0)}, 6.0, 6.0}});
  model.linear.col_lower[0] = 1.0;
  model.linear.col_upper[0] = 4.0;
  const global::Problem problem = global::build_problem(model);
  global::Box box{problem.col_lower, problem.col_upper};
  const global::FbbtOutcome outcome = global::fbbt(problem, kInfinity, 1e-7, &box);
  EXPECT_FALSE(outcome.infeasible);
  EXPECT_NEAR(box.lower[1], 1.5, 1e-6);
  EXPECT_NEAR(box.upper[1], 6.0, 1e-6);
  EXPECT_LE(box.lower[1], 1.5);  // relaxed outward, never inward
  EXPECT_GE(box.upper[1], 6.0);
  // x * y >= 5 over [0, 2]^2 cannot hold: the product is at most 4.
  const QcqpModel empty =
      make_model(2, 0.0, 2.0, {0.0, 0.0}, {{{}, {product(0, 1, 1.0)}, 5.0, kInfinity}});
  const global::Problem empty_problem = global::build_problem(empty);
  global::Box empty_box{empty_problem.col_lower, empty_problem.col_upper};
  EXPECT_TRUE(global::fbbt(empty_problem, kInfinity, 1e-7, &empty_box).infeasible);
}

// ---- the search ---------------------------------------------------------------------------

TEST(SpatialBranchAndBound, HaverlyOneReachesAndProvesTheGlobalOptimum) {
  const QcqpModel model = read_or_fail(kHaverly1);
  const Solution solution = solve_global(model, quiet());
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NEAR(solution.objective, -400.0, 1e-6);
  EXPECT_LE(solution.dual_bound, solution.objective + 1e-9);
  EXPECT_GE(solution.dual_bound, -400.0 - 0.04 - 1e-9);  // mip_relative_gap 1e-4
  EXPECT_EQ(solution.algorithm, "spatial-branch-and-bound");
  // The point is checked here against the ORIGINAL rows, products included.
  const std::vector<double> activity = model.row_activity(solution.col_value);
  for (Index i = 0; i < model.linear.num_rows(); ++i) {
    const auto u = static_cast<std::size_t>(i);
    EXPECT_GE(activity[u], model.linear.row_lower[u] - 1e-7) << model.linear.row_names[u];
    EXPECT_LE(activity[u], model.linear.row_upper[u] + 1e-7) << model.linear.row_names[u];
    EXPECT_NEAR(solution.row_activity[u], activity[u], 1e-9);
  }
  EXPECT_LE(solution.primal_infeasibility, 1e-7);
}

TEST(SpatialBranchAndBound, BeatsTheLocalOptimumTheRecursionGetsStuckIn) {
  // From pool quality 3 % (all crude A) with 50 units through the pool to X and 50 of C
  // beside it, both LPs of the alternation return the same point: cost -100, Haverly's
  // local optimum. The global method must find -400.
  const QcqpModel model = read_or_fail(kHaverly1);
  const global::Problem problem = global::build_problem(model);
  std::vector<double> start(static_cast<std::size_t>(problem.n), 0.0);
  start[static_cast<std::size_t>(column_named(model.linear, "f_1_4"))] = 50.0;
  start[static_cast<std::size_t>(column_named(model.linear, "f_4_5"))] = 50.0;
  start[static_cast<std::size_t>(column_named(model.linear, "f_3_5"))] = 50.0;
  start[static_cast<std::size_t>(column_named(model.linear, "p_4_1"))] = 3.0;
  const global::LocalSearchResult local = global::alternating_search(
      problem, start, kInfinity, global::inner_lp_options(quiet(), kInfinity), 1e-7);
  ASSERT_TRUE(local.improved);
  EXPECT_NEAR(local.objective, -100.0, 1e-6);

  const Solution global_answer = solve_global(model, quiet());
  ASSERT_EQ(global_answer.status, SolveStatus::kOptimal) << global_answer.message;
  EXPECT_NEAR(global_answer.objective, -400.0, 1e-6);
  EXPECT_LT(global_answer.objective, local.objective - 299.0);
}

TEST(SpatialBranchAndBound, ANonConvexSquareObjectiveFindsTheFarEnd) {
  // min -x^2 over [-1, 2]: a local minimum at -1 (value -1), the global one at 2 (value -4).
  QcqpModel model = make_model(1, -1.0, 2.0, {0.0}, {{{{0, 1.0}}, {}, -kInfinity, 2.0}});
  set_hessian(&model.linear, {{0, 0, -2.0}});
  const Solution solution = solve_global(model, quiet());
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NEAR(solution.objective, -4.0, 1e-6);
  EXPECT_NEAR(solution.col_value[0], 2.0, 1e-6);
}

TEST(SpatialBranchAndBound, AMaximizedProductKeepsItsSense) {
  // max x*y s.t. x + y <= 2 on [0, 2]^2: 1 at x = y = 1. 0.5 x'Hx with H's lower entry
  // (1, 0) = 1 is the product x*y.
  QcqpModel model =
      make_model(2, 0.0, 2.0, {0.0, 0.0}, {{{{0, 1.0}, {1, 1.0}}, {}, -kInfinity, 2.0}});
  model.linear.sense = ObjSense::kMaximize;
  set_hessian(&model.linear, {{1, 0, 1.0}});
  const Solution solution = solve_global(model, quiet());
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NEAR(solution.objective, 1.0, 1e-6);
  EXPECT_GE(solution.dual_bound, solution.objective - 1e-9);  // an UPPER bound when maximizing
  EXPECT_LE(solution.dual_bound, 1.0 + 1e-4 + 1e-9);
}

TEST(SpatialBranchAndBound, AProductTheBoxCannotReachIsInfeasible) {
  const QcqpModel model =
      make_model(2, 0.0, 2.0, {1.0, 1.0}, {{{}, {product(0, 1, 1.0)}, 5.0, kInfinity}});
  const Solution solution = solve_global(model, quiet());
  EXPECT_EQ(solution.status, SolveStatus::kInfeasible) << solution.message;
  EXPECT_TRUE(solution.col_value.empty());
}

TEST(SpatialBranchAndBound, RefusesWhatItCannotRelax) {
  // A product of a free column has no McCormick envelope over an unbounded box.
  QcqpModel unbounded =
      make_model(2, 0.0, 2.0, {1.0, 0.0}, {{{}, {product(0, 1, 1.0)}, -kInfinity, 1.0}});
  unbounded.linear.col_lower[0] = -kInfinity;
  const Solution refused = solve_global(unbounded, quiet());
  EXPECT_EQ(refused.status, SolveStatus::kNotSolved);
  EXPECT_NE(refused.message.find("finite"), std::string::npos) << refused.message;
  // Integer columns are outside this first slice.
  QcqpModel integer =
      make_model(2, 0.0, 2.0, {1.0, 0.0}, {{{}, {product(0, 1, 1.0)}, -kInfinity, 1.0}});
  integer.linear.col_type[0] = VarType::kInteger;
  EXPECT_EQ(solve_global(integer, quiet()).status, SolveStatus::kNotSolved);
}

TEST(SpatialBranchAndBound, ANodeLimitOfZeroReportsNoPoint) {
  const QcqpModel model = read_or_fail(kHaverly1);
  Options options = quiet();
  options.set_int("node_limit", 0);
  const Solution solution = solve_global(model, options);
  EXPECT_EQ(solution.status, SolveStatus::kNodeLimit) << solution.message;
  EXPECT_TRUE(solution.col_value.empty());
  EXPECT_FALSE(claims_a_point(solution));
}

TEST(SpatialBranchAndBound, AModelWithoutProductsIsItsLinearPart) {
  // max x + y s.t. x + 2y <= 4 on [0, 3]^2: 3.5 at (3, 0.5).
  QcqpModel model =
      make_model(2, 0.0, 3.0, {1.0, 1.0}, {{{{0, 1.0}, {1, 2.0}}, {}, -kInfinity, 4.0}});
  model.linear.sense = ObjSense::kMaximize;
  const Solution solution = solve_global(model, quiet());
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NEAR(solution.objective, 3.5, 1e-7);
  EXPECT_NE(solution.algorithm, "spatial-branch-and-bound");
}

TEST(SpatialBranchAndBound, NoGridPointBeatsAClaimedGlobalOptimum) {
  // Random small non-convex models: three columns in [-1, 2], a non-convex quadratic
  // objective and two rows with products, each row feasible at a random point. A wrong
  // prune or an invalid bound shows up as a feasible grid point better than the claimed
  // optimum by more than the gap target; a wrong point as a violated row.
  std::mt19937 generator(514);
  std::uniform_real_distribution<double> coefficient(-2.0, 2.0);
  std::uniform_real_distribution<double> inside(-1.0, 2.0);
  constexpr int kSteps = 24;
  int claimed = 0;
  for (int trial = 0; trial < 25; ++trial) {
    std::vector<RowSpec> rows;
    const std::vector<double> anchor{inside(generator), inside(generator), inside(generator)};
    for (int r = 0; r < 2; ++r) {
      RowSpec row{{{0, coefficient(generator)},
                   {1, coefficient(generator)},
                   {2, coefficient(generator)}},
                  {product(0, 1, coefficient(generator)), product(1, 2, coefficient(generator)),
                   product(2, 2, coefficient(generator))},
                  -kInfinity,
                  0.0};
      double activity = 0.0;
      for (const auto& [j, a] : row.linear) activity += a * anchor[static_cast<std::size_t>(j)];
      for (const QuadraticTerm& t : row.products) {
        activity += t.value * anchor[static_cast<std::size_t>(t.first)] *
                    anchor[static_cast<std::size_t>(t.second)];
      }
      row.upper = activity + 0.25;  // the anchor satisfies it, with room
      rows.push_back(row);
    }
    QcqpModel model = make_model(
        3, -1.0, 2.0, {coefficient(generator), coefficient(generator), coefficient(generator)},
        rows);
    set_hessian(&model.linear, {{0, 0, coefficient(generator)},
                                {1, 0, coefficient(generator)},
                                {2, 1, coefficient(generator)},
                                {2, 2, coefficient(generator)}});
    const Solution solution = solve_global(model, quiet());
    ASSERT_TRUE(solution.status == SolveStatus::kOptimal) << trial << ": " << solution.message;
    ++claimed;
    const std::vector<double> activity = model.row_activity(solution.col_value);
    for (std::size_t i = 0; i < activity.size(); ++i) {
      EXPECT_LE(activity[i], model.linear.row_upper[i] + 1e-6) << trial;
    }
    const double allowed = 1e-4 * std::max(1.0, std::fabs(solution.objective)) + 1e-6;
    double best_grid = kInfinity;
    std::vector<double> x(3);
    for (int a = 0; a <= kSteps; ++a) {
      for (int b = 0; b <= kSteps; ++b) {
        for (int c = 0; c <= kSteps; ++c) {
          x = {-1.0 + 3.0 * a / kSteps, -1.0 + 3.0 * b / kSteps, -1.0 + 3.0 * c / kSteps};
          const std::vector<double> rows_at = model.row_activity(x);
          if (rows_at[0] > model.linear.row_upper[0] ||
              rows_at[1] > model.linear.row_upper[1]) {
            continue;
          }
          best_grid = std::min(best_grid, model.objective(x));
        }
      }
    }
    EXPECT_GE(best_grid, solution.objective - allowed)
        << "trial " << trial << ": a feasible grid point beats the claimed global optimum";
    EXPECT_LE(solution.dual_bound, solution.objective + 1e-9) << trial;
  }
  EXPECT_EQ(claimed, 25);
}

}  // namespace
}  // namespace sankhya
