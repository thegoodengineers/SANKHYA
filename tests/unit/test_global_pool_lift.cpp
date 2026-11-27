// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a P-formulated pool lifted to the Q-formulation's proportions (#514,
// `global_pool_lift`).
//
// Haverly's case 1 in the P-formulation, written the way bench/runners/pooling_models.py writes
// data/pooling/haverly1_p.mps: one pool fed by sources 1 (quality 3) and 2 (quality 1), a
// balance row, a capacity row over the inflows, and the quality row 3 f_14 + f_24 - p (f_45 +
// f_46) = 0. Its Q-formulation (tests/unit/test_global_rlt.cpp) must be left alone.

#include <cmath>
#include <fstream>
#include <random>
#include <sstream>
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

const char* const kHaverly1P =
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
    "QCMATRIX pq_4_1\n f_4_5 p_4_1 -0.5\n p_4_1 f_4_5 -0.5\n f_4_6 p_4_1 -0.5\n"
    " p_4_1 f_4_6 -0.5\n"
    "QCMATRIX tq_5_1\n f_4_5 p_4_1 0.5\n p_4_1 f_4_5 0.5\n"
    "QCMATRIX tq_6_1\n f_4_6 p_4_1 0.5\n p_4_1 f_4_6 0.5\n"
    "ENDATA\n";

// The same pool in the Q-formulation: proportions and path flows, no quality column.
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

// The same pool in the PQ-formulation: the Q-formulation plus sum_i v_i_4_j = f_4_j and the
// capacity times each q_i_4. Each path row v - q f = 0 with sum_i v_ij = f_j has the shape of a
// pool with ONE outflow, which the lift must not take.
const char* const kHaverly1Pq =
    "NAME haverly1_pq\n"
    "ROWS\n"
    " N cost\n L src_1\n L src_2\n L src_3\n L dem_5\n L dem_6\n E sumq_4\n L cap_4\n"
    " L tq_5_1\n L tq_6_1\n E path_1_4_5\n E path_1_4_6\n E path_2_4_5\n E path_2_4_6\n"
    " E rlt_4_5\n E rlt_4_6\n L rltc_1_4\n L rltc_2_4\n"
    "COLUMNS\n"
    " q_1_4 sumq_4 1\n q_1_4 rltc_1_4 -300\n q_2_4 sumq_4 1\n q_2_4 rltc_2_4 -300\n"
    " f_3_5 cost 1\n f_3_5 src_3 1\n f_3_5 dem_5 1\n f_3_5 tq_5_1 -0.5\n"
    " f_3_6 cost -5\n f_3_6 src_3 1\n f_3_6 dem_6 1\n f_3_6 tq_6_1 0.5\n"
    " f_4_5 cost -9\n f_4_5 dem_5 1\n f_4_5 cap_4 1\n f_4_5 tq_5_1 -2.5\n f_4_5 rlt_4_5 -1\n"
    " f_4_6 cost -15\n f_4_6 dem_6 1\n f_4_6 cap_4 1\n f_4_6 tq_6_1 -1.5\n f_4_6 rlt_4_6 -1\n"
    " v_1_4_5 cost 6\n v_1_4_5 src_1 1\n v_1_4_5 tq_5_1 3\n v_1_4_5 path_1_4_5 1\n"
    " v_1_4_5 rlt_4_5 1\n v_1_4_5 rltc_1_4 1\n"
    " v_1_4_6 cost 6\n v_1_4_6 src_1 1\n v_1_4_6 tq_6_1 3\n v_1_4_6 path_1_4_6 1\n"
    " v_1_4_6 rlt_4_6 1\n v_1_4_6 rltc_1_4 1\n"
    " v_2_4_5 cost 16\n v_2_4_5 src_2 1\n v_2_4_5 tq_5_1 1\n v_2_4_5 path_2_4_5 1\n"
    " v_2_4_5 rlt_4_5 1\n v_2_4_5 rltc_2_4 1\n"
    " v_2_4_6 cost 16\n v_2_4_6 src_2 1\n v_2_4_6 tq_6_1 1\n v_2_4_6 path_2_4_6 1\n"
    " v_2_4_6 rlt_4_6 1\n v_2_4_6 rltc_2_4 1\n"
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

std::size_t named(const std::vector<std::string>& names, const std::string& name) {
  for (std::size_t k = 0; k < names.size(); ++k) {
    if (names[k] == name) return k;
  }
  ADD_FAILURE() << "no " << name;
  return 0;
}

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("nonconvex", "global");
  return options;
}

TEST(GlobalPoolLift, RecognisesThePoolOfThePFormulation) {
  const QcqpModel model = read_or_fail(kHaverly1P);
  QcqpModel lifted;
  const global::PoolLift lift = global::lift_pools(model, &lifted);
  EXPECT_EQ(lift.pools, 1);
  EXPECT_EQ(lift.columns, 2);  // q_1, q_2
  // sum q = 1; two inflow rows; the quality times each of two outflows; the capacity.
  EXPECT_EQ(lift.rows, 6);
  EXPECT_EQ(lifted.linear.num_cols(), model.linear.num_cols() + 2);
  EXPECT_EQ(lifted.linear.num_rows(), model.linear.num_rows() + 6);
  EXPECT_TRUE(lifted.validate().empty()) << lifted.validate();
}

TEST(GlobalPoolLift, LeavesTheQFormulationAlone) {
  const QcqpModel model = read_or_fail(kHaverly1Q);
  QcqpModel lifted;
  EXPECT_EQ(global::lift_pools(model, &lifted).pools, 0);
}

TEST(GlobalPoolLift, LeavesThePqFormulationAlone) {
  const QcqpModel model = read_or_fail(kHaverly1Pq);
  QcqpModel lifted;
  EXPECT_EQ(global::lift_pools(model, &lifted).pools, 0);
}

TEST(GlobalPoolLift, EveryAddedRowHoldsAtEveryPointOfThePool) {
  // Random points of the pool's own rows (balance and quality), extended with q_i = f_i / Y,
  // and Y = 0 with any q on the simplex: the added rows hold at all of them, so the lift
  // removes no point of the model.
  const QcqpModel model = read_or_fail(kHaverly1P);
  QcqpModel lifted;
  ASSERT_EQ(global::lift_pools(model, &lifted).pools, 1);
  const auto& names = lifted.linear.col_names;
  const std::size_t f14 = named(names, "f_1_4");
  const std::size_t f24 = named(names, "f_2_4");
  const std::size_t f45 = named(names, "f_4_5");
  const std::size_t f46 = named(names, "f_4_6");
  const std::size_t p = named(names, "p_4_1");
  const auto n0 = static_cast<std::size_t>(model.linear.num_cols());
  const Index m0 = model.linear.num_rows();
  std::mt19937 rng(514);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  int checked = 0;
  for (int trial = 0; trial < 200; ++trial) {
    std::vector<double> x(static_cast<std::size_t>(lifted.linear.num_cols()), 0.0);
    const bool empty = trial % 10 == 0;
    x[f14] = empty ? 0.0 : 100.0 * unit(rng);
    x[f24] = empty ? 0.0 : 100.0 * unit(rng);
    const double y = x[f14] + x[f24];
    const double share = unit(rng);
    x[f45] = share * y;
    x[f46] = y - x[f45];
    x[p] = y > 0.0 ? (3.0 * x[f14] + x[f24]) / y : 1.0 + 2.0 * unit(rng);
    x[n0] = y > 0.0 ? x[f14] / y : unit(rng);
    x[n0 + 1] = 1.0 - x[n0];
    const std::vector<double> activity = lifted.row_activity(x);
    for (Index r = m0; r < lifted.linear.num_rows(); ++r) {
      const auto u = static_cast<std::size_t>(r);
      const double slack = 1e-9 * (1.0 + std::fabs(activity[u]) + y);
      EXPECT_GE(activity[u], lifted.linear.row_lower[u] - slack)
          << "trial " << trial << " row " << r;
      EXPECT_LE(activity[u], lifted.linear.row_upper[u] + slack)
          << "trial " << trial << " row " << r;
      ++checked;
    }
  }
  EXPECT_EQ(checked, 200 * 6);
}

TEST(GlobalPoolLift, ProvesTheSameOptimumOnTheModelsOwnColumns) {
  const QcqpModel model = read_or_fail(kHaverly1P);
  const Solution plain = solve_global(model, quiet());
  Options with_lift = quiet();
  with_lift.set_bool("global_pool_lift", true);
  const Solution lifted = solve_global(model, with_lift);
  ASSERT_EQ(plain.status, SolveStatus::kOptimal) << plain.message;
  ASSERT_EQ(lifted.status, SolveStatus::kOptimal) << lifted.message;
  // Haverly (1978) case 1: the global optimum is a cost of -400.
  EXPECT_NEAR(lifted.objective, -400.0, 0.04);
  EXPECT_LE(lifted.dual_bound, lifted.objective + 1e-9);
  ASSERT_EQ(lifted.col_value.size(), static_cast<std::size_t>(model.linear.num_cols()));
  ASSERT_EQ(lifted.row_activity.size(), static_cast<std::size_t>(model.linear.num_rows()));
  EXPECT_LE(lifted.primal_infeasibility, 1e-7);
  EXPECT_NE(lifted.message.find("1 pool(s) lifted"), std::string::npos) << lifted.message;
  // More rows can only raise the root bound, and it stays a bound.
  EXPECT_GE(lifted.root_bound, plain.root_bound - 1e-6);
  EXPECT_LE(lifted.root_bound, -400.0 + 1e-6);
}

TEST(GlobalPoolLift, NamesThePoolInTheCertificate) {
  // The certificate counts the lifted model and names the rows the pool was read from (bal_4,
  // pq_4_1 and cap_4: rows 5, 7 and 6 of the file, the objective not counted), so its checker
  // can rebuild the lift; tools/test_verify_global_certificate.py checks one end to end.
  const QcqpModel model = read_or_fail(kHaverly1P);
  Options options = quiet();
  options.set_bool("global_pool_lift", true);
  const TempFile certificate("", ".json");
  options.set_string("write_certificate", certificate.path());
  const Solution solution = solve_global(model, options);
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NE(solution.message.find("1 pool(s) lifted"), std::string::npos) << solution.message;
  std::ifstream in(certificate.path());
  std::stringstream buffer;
  buffer << in.rdbuf();
  const std::string text = buffer.str();
  EXPECT_NE(text.find("\"lift\": [{\"balance\": 5, \"qualities\": [7], \"capacities\": [6]}]"),
            std::string::npos)
      << text.substr(0, 400);
  EXPECT_NE(text.find("\"columns\": 9,"), std::string::npos);
  EXPECT_NE(text.find("\"rows\": 16,"), std::string::npos);
}

}  // namespace
}  // namespace sankhya
