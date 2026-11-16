// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the reformulation-linearization rows of the global relaxation (#514,
// `global_rlt`).
//
// Haverly's case 1 in the Q-formulation (Ben-Tal, Eiger and Gershovitz 1994), written the way
// bench/runners/pooling_models.py writes data/pooling/haverly1_q.mps: proportions q_i_4 of the
// pool, flows f, and path flows v_i_4_j = q_i_4 * f_4_j as quadratic rows. The RLT rows over it
// are the ones the PQ-formulation adds (Quesada and Grossmann 1995): sum_i q_i4 = 1 times each
// f_4j, and the pool capacity times each q_i4.

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
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

const char* const kHaverly1Q =
    "NAME haverly1_q\n"
    "ROWS\n"
    " N cost\n L src_1\n L src_2\n L src_3\n L dem_5\n L dem_6\n E sumq_4\n L cap_4\n"
    " L tq_5_1\n L tq_6_1\n E path_1_4_5\n E path_1_4_6\n E path_2_4_5\n E path_2_4_6\n"
    "COLUMNS\n"
    " q_1_4 sumq_4 1\n q_2_4 sumq_4 1\n"
    " f_3_5 cost 1\n f_3_5 src_3 1\n f_3_5 dem_5 1\n f_3_5 tq_5_1 -0.5\n"
    " f_3_6 cost -5\n f_3_6 src_3 1\n f_3_6 dem_6 1\n f_3_6 tq_6_1 0.5\n"
    " f_4_5 cost -9\n f_4_5 dem_5 1\n f_4_5 cap_4 1\n f_4_5 tq_5_1 -2.5\n"
    " f_4_6 cost -15\n f_4_6 dem_6 1\n f_4_6 cap_4 1\n f_4_6 tq_6_1 -1.5\n"
    " v_1_4_5 cost 6\n v_1_4_5 src_1 1\n v_1_4_5 tq_5_1 3\n v_1_4_5 path_1_4_5 1\n"
    " v_1_4_6 cost 6\n v_1_4_6 src_1 1\n v_1_4_6 tq_6_1 3\n v_1_4_6 path_1_4_6 1\n"
    " v_2_4_5 cost 16\n v_2_4_5 src_2 1\n v_2_4_5 tq_5_1 1\n v_2_4_5 path_2_4_5 1\n"
    " v_2_4_6 cost 16\n v_2_4_6 src_2 1\n v_2_4_6 tq_6_1 1\n v_2_4_6 path_2_4_6 1\n"
    "RHS\n"
    " rhs src_1 300\n rhs src_2 300\n rhs src_3 300\n rhs dem_5 100\n rhs dem_6 200\n"
    " rhs sumq_4 1\n rhs cap_4 300\n"
    "BOUNDS\n"
    " UP bnd q_1_4 1\n UP bnd q_2_4 1\n UP bnd f_3_5 100\n UP bnd f_3_6 200\n"
    " UP bnd f_4_5 100\n UP bnd f_4_6 200\n UP bnd v_1_4_5 100\n UP bnd v_1_4_6 200\n"
    " UP bnd v_2_4_5 100\n UP bnd v_2_4_6 200\n"
    "QCMATRIX path_1_4_5\n f_4_5 q_1_4 -0.5\n q_1_4 f_4_5 -0.5\n"
    "QCMATRIX path_1_4_6\n f_4_6 q_1_4 -0.5\n q_1_4 f_4_6 -0.5\n"
    "QCMATRIX path_2_4_5\n f_4_5 q_2_4 -0.5\n q_2_4 f_4_5 -0.5\n"
    "QCMATRIX path_2_4_6\n f_4_6 q_2_4 -0.5\n q_2_4 f_4_6 -0.5\n"
    "ENDATA\n";

QcqpModel read_or_fail(const std::string& text) {
  const TempFile file(text);
  QcqpModel model;
  const io::ReadResult result = io::read_qcqp_mps(file.path(), &model);
  EXPECT_TRUE(result.ok) << result.error;
  return model;
}

Index named(const std::vector<std::string>& names, const std::string& name) {
  for (std::size_t k = 0; k < names.size(); ++k) {
    if (names[k] == name) return static_cast<Index>(k);
  }
  ADD_FAILURE() << "no " << name;
  return -1;
}

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("nonconvex", "global");
  return options;
}

TEST(GlobalRlt, FindsThePqRowsOfThePoolingQFormulation) {
  const QcqpModel model = read_or_fail(kHaverly1Q);
  const global::Problem problem = global::build_problem(model);
  const std::vector<global::RltRow> rows = global::rlt_rows(problem);
  const auto& row_names = model.linear.row_names;
  const auto& col_names = model.linear.col_names;
  const Index sumq = named(row_names, "sumq_4");
  const Index cap = named(row_names, "cap_4");
  int equalities = 0;
  int capacities = 0;
  for (const global::RltRow& row : rows) {
    if (row.row == sumq) {
      // sum_i q_i4 = 1 times f_4j: one equality per outflow.
      EXPECT_EQ(row.side, 'e');
      EXPECT_EQ(col_names[static_cast<std::size_t>(row.column)].rfind("f_4_", 0), 0u);
      ++equalities;
    } else {
      // f_45 + f_46 <= 300 times both bound factors of each q_i4.
      EXPECT_EQ(row.row, cap);
      EXPECT_EQ(row.side, 'u');
      EXPECT_EQ(col_names[static_cast<std::size_t>(row.column)].rfind("q_", 0), 0u);
      ++capacities;
    }
  }
  // No other row qualifies: dem_5 has f_3_5, the src rows have path flows, neither of which is
  // in a product, and the quadratic rows are never multiplied.
  EXPECT_EQ(equalities, 2);
  EXPECT_EQ(capacities, 4);
}

TEST(GlobalRlt, EveryRowHoldsAtEveryPointOfTheModelInAnyBox) {
  const QcqpModel model = read_or_fail(kHaverly1Q);
  global::Problem problem = global::build_problem(model);
  problem.rlt = global::rlt_rows(problem);
  const auto& names = model.linear.col_names;
  const auto q1 = static_cast<std::size_t>(named(names, "q_1_4"));
  const auto q2 = static_cast<std::size_t>(named(names, "q_2_4"));
  const auto f5 = static_cast<std::size_t>(named(names, "f_4_5"));
  const auto f6 = static_cast<std::size_t>(named(names, "f_4_6"));
  std::mt19937 rng(514);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  const Index first = problem.m + 4 * static_cast<Index>(problem.products.size());
  int checked = 0;
  for (int trial = 0; trial < 200; ++trial) {
    // A random sub-box of the model's box, and points in it with sum_i q_i4 = 1, which is all
    // the RLT rows assume (the capacity row holds since f_45 + f_46 <= 300 in any box).
    global::Box box{problem.col_lower, problem.col_upper};
    for (const std::size_t j : {q1, f5, f6}) {
      const double a = box.lower[j] + unit(rng) * (box.upper[j] - box.lower[j]);
      const double b = box.lower[j] + unit(rng) * (box.upper[j] - box.lower[j]);
      box.lower[j] = std::min(a, b);
      box.upper[j] = std::max(a, b);
    }
    box.lower[q2] = 1.0 - box.upper[q1];
    box.upper[q2] = 1.0 - box.lower[q1];
    const Model lp = global::build_relaxation(problem, box);
    ASSERT_EQ(lp.num_rows(), first + static_cast<Index>(problem.rlt.size()));
    for (int sample = 0; sample < 20; ++sample) {
      std::vector<double> x(static_cast<std::size_t>(lp.num_cols()), 0.0);
      for (const std::size_t j : {q1, f5, f6}) {
        x[j] = box.lower[j] + unit(rng) * (box.upper[j] - box.lower[j]);
      }
      x[q2] = 1.0 - x[q1];
      for (std::size_t p = 0; p < problem.products.size(); ++p) {
        const global::Product& product = problem.products[p];
        x[static_cast<std::size_t>(problem.n) + p] =
            x[static_cast<std::size_t>(product.a)] * x[static_cast<std::size_t>(product.b)];
      }
      std::vector<double> activity(static_cast<std::size_t>(lp.num_rows()), 0.0);
      lp.matrix.multiply(x.data(), activity.data());
      for (Index r = first; r < lp.num_rows(); ++r) {
        const auto u = static_cast<std::size_t>(r);
        const double slack = 1e-9 * (1.0 + std::fabs(activity[u]));
        EXPECT_GE(activity[u], lp.row_lower[u] - slack) << "trial " << trial << " row " << r;
        EXPECT_LE(activity[u], lp.row_upper[u] + slack) << "trial " << trial << " row " << r;
        ++checked;
      }
    }
  }
  EXPECT_EQ(checked, 200 * 20 * 6);
}

TEST(GlobalRlt, TightensTheRootAndProvesTheSameOptimum) {
  const QcqpModel model = read_or_fail(kHaverly1Q);
  Options without = quiet();
  without.set_bool("global_rlt", false);
  const Solution plain = solve_global(model, without);
  Options with_rlt = quiet();
  with_rlt.set_bool("global_rlt", true);
  const Solution rlt = solve_global(model, with_rlt);
  ASSERT_EQ(plain.status, SolveStatus::kOptimal) << plain.message;
  ASSERT_EQ(rlt.status, SolveStatus::kOptimal) << rlt.message;
  // Haverly (1978) case 1: the global optimum is a cost of -400.
  EXPECT_NEAR(rlt.objective, -400.0, 0.04);
  EXPECT_LE(rlt.dual_bound, rlt.objective + 1e-9);
  EXPECT_LE(rlt.primal_infeasibility, 1e-7);
  // More rows can only raise the root bound (both are proved by their multipliers), and it
  // stays a bound.
  EXPECT_GE(rlt.root_bound, plain.root_bound - 1e-6);
  EXPECT_LE(rlt.root_bound, -400.0 + 1e-6);
}

}  // namespace
}  // namespace sankhya
