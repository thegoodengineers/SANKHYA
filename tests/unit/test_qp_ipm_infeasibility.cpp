// SPDX-License-Identifier: Apache-2.0
// SANKHYA - infeasibility and unboundedness from the proximal QP interior point's iterates
// (#893, option qp_ipm_detect_infeasibility). See the #893 comment in src/qp/qp_ipm.cpp for
// where the candidates come from and its citation.
//
// What #893 asks of it, each held by its own tests:
//   1. A primal infeasible convex QP ends `infeasible` with a Farkas vector, and a dual
//      infeasible one ends `unbounded` with a ray and a primal feasible point - each
//      certificate re-checked here by the same functions every engine's proof is held to
//      (farkas_proves_infeasible, ray_proves_unbounded), and the point measured.
//   2. A feasible QP is never reported infeasible or unbounded, including ones whose feasible
//      region is unbounded or whose objective is flat along a direction.
//   3. With the option off nothing changes: the infeasible models end as they did before,
//      with no certificate, and the feasible ones take the same iterates to the same answer.

#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/certificate.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/qp.hpp"

namespace sankhya {
namespace {

Options quiet(bool detect) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("qp_ipm_detect_infeasibility", detect);
  return options;
}

Model empty_model(Index rows, Index cols) {
  Model model;
  model.sense = ObjSense::kMinimize;
  model.col_cost.assign(static_cast<std::size_t>(cols), 0.0);
  model.col_lower.assign(static_cast<std::size_t>(cols), -kInfinity);
  model.col_upper.assign(static_cast<std::size_t>(cols), kInfinity);
  model.col_type.assign(static_cast<std::size_t>(cols), VarType::kContinuous);
  model.row_lower.assign(static_cast<std::size_t>(rows), -kInfinity);
  model.row_upper.assign(static_cast<std::size_t>(rows), kInfinity);
  model.matrix.reset(rows, cols);
  model.hessian.reset(cols, cols);
  return model;
}

// ---- primal infeasible ------------------------------------------------------------------

/// min (x1^2 + x2^2) / 2  s.t.  x1 + x2 >= 3,  0 <= x <= 1. The row asks for 3 and the box
/// gives at most 2; y = 1 on the row is the proof (3 > 2).
Model box_infeasible() {
  Model model = empty_model(1, 2);
  model.col_lower = {0.0, 0.0};
  model.col_upper = {1.0, 1.0};
  model.row_lower = {3.0};
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.finalize();
  model.hessian.add_entry(0, 0, 1.0);
  model.hessian.add_entry(1, 1, 1.0);
  model.hessian.finalize();
  return model;
}

/// x1, x2 free, x3 >= 0, Q = I:  x1 + x2 >= 2,  x1 - x2 >= 2,  x1 + x3 <= 1. The first two
/// add to x1 >= 2 and the third says x1 <= 1 - x3 <= 1. The proof needs all three rows,
/// y = (1, 1, -2), and A'y = (0, 0, -2) has to vanish on the two FREE columns.
Model three_row_infeasible() {
  Model model = empty_model(3, 3);
  model.col_lower = {-kInfinity, -kInfinity, 0.0};
  model.row_lower = {2.0, 2.0, -kInfinity};
  model.row_upper = {kInfinity, kInfinity, 1.0};
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.add_entry(1, 0, 1.0);
  model.matrix.add_entry(1, 1, -1.0);
  model.matrix.add_entry(2, 0, 1.0);
  model.matrix.add_entry(2, 2, 1.0);
  model.matrix.finalize();
  for (Index j = 0; j < 3; ++j) model.hessian.add_entry(j, j, 1.0);
  model.hessian.finalize();
  return model;
}

/// Two equality rows that cannot both hold, x free, Q = [2 1; 1 2]:  x1 + x2 = 1,
/// 2 x1 + 2 x2 = 5. y = (2, -1) gives 0 = -3 (or its negation), on free columns only.
Model equality_infeasible() {
  Model model = empty_model(2, 2);
  model.row_lower = {1.0, 5.0};
  model.row_upper = {1.0, 5.0};
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.add_entry(1, 0, 2.0);
  model.matrix.add_entry(1, 1, 2.0);
  model.matrix.finalize();
  model.hessian.add_entry(0, 0, 2.0);
  model.hessian.add_entry(1, 0, 1.0);
  model.hessian.add_entry(1, 1, 2.0);
  model.hessian.finalize();
  return model;
}

// ---- dual infeasible (unbounded) --------------------------------------------------------

/// min x1^2 - 2 x1 - x2  s.t.  x1 - x2 <= 1,  x >= 0. Feasible (x = 0), and along d = (0, 1)
/// the row falls (it has no lower side), no bound blocks, d'Qd = 0 and c'd = -1.
Model flat_direction_unbounded() {
  Model model = empty_model(1, 2);
  model.col_cost = {-2.0, -1.0};
  model.col_lower = {0.0, 0.0};
  model.row_upper = {1.0};
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, -1.0);
  model.matrix.finalize();
  model.hessian.add_entry(0, 0, 2.0);
  model.hessian.finalize();
  return model;
}

/// min x1^2 - x2 - x3  s.t.  x2 - x3 = 0,  x1 + x2 + x3 >= 1,  x2, x3 >= 0, x1 free. The ray
/// d = (0, 1, 1) keeps the equality EXACTLY (A d has to vanish there) and improves by 2.
Model equality_ray_unbounded() {
  Model model = empty_model(2, 3);
  model.col_cost = {0.0, -1.0, -1.0};
  model.col_lower = {-kInfinity, 0.0, 0.0};
  model.row_lower = {0.0, 1.0};
  model.row_upper = {0.0, kInfinity};
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.add_entry(0, 2, -1.0);
  model.matrix.add_entry(1, 0, 1.0);
  model.matrix.add_entry(1, 1, 1.0);
  model.matrix.add_entry(1, 2, 1.0);
  model.matrix.finalize();
  model.hessian.add_entry(0, 0, 2.0);
  model.hessian.finalize();
  return model;
}

/// The first unbounded model as a maximization: max -x1^2 + 2 x1 + x2.
Model maximize_unbounded() {
  Model model = flat_direction_unbounded();
  model.sense = ObjSense::kMaximize;
  model.col_cost = {2.0, 1.0};
  model.hessian.reset(2, 2);
  model.hessian.add_entry(0, 0, -2.0);
  model.hessian.finalize();
  return model;
}

// ---- feasible, and must stay so ---------------------------------------------------------

/// min x1^2 + x2^2 - 2 x1 - 4 x2  s.t.  x1 + x2 <= 2,  x >= 0: optimum (0.5, 1.5).
Model inequality_qp() {
  Model model = empty_model(1, 2);
  model.col_cost = {-2.0, -4.0};
  model.col_lower = {0.0, 0.0};
  model.row_upper = {2.0};
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.finalize();
  model.hessian.add_entry(0, 0, 2.0);
  model.hessian.add_entry(1, 1, 2.0);
  model.hessian.finalize();
  return model;
}

/// min x1^2 + x2  s.t.  x1 - x2 <= 1,  x >= 0: the region is unbounded along (0, 1) and
/// (1, 1), but the objective rises along both; optimum 0 at the origin.
Model unbounded_region_bounded_objective() {
  Model model = flat_direction_unbounded();
  model.col_cost = {0.0, 1.0};
  return model;
}

/// min x1^2 - 2 x1  s.t.  x1 + x2 >= 1,  x >= 0: flat along x2, so a direction exists along
/// which nothing improves; optimum -1 on the segment x1 = 1, any x2 >= 0.
Model flat_but_bounded() {
  Model model = empty_model(1, 2);
  model.col_cost = {-2.0, 0.0};
  model.col_lower = {0.0, 0.0};
  model.row_lower = {1.0};
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.finalize();
  model.hessian.add_entry(0, 0, 2.0);
  model.hessian.finalize();
  return model;
}

/// min (x1^2 + x2^2 + x3^2) / 2  s.t.  x1 + x2 + x3 = 3,  1 <= x1 - x2 <= 2,  x1 free,
/// x2 >= 0, x3 fixed at 0.5 (test_qp_ipm.cpp's model): equality, ranged, free and fixed.
Model equality_ranged_free_fixed_qp() {
  Model model = empty_model(2, 3);
  model.col_lower = {-kInfinity, 0.0, 0.5};
  model.col_upper = {kInfinity, kInfinity, 0.5};
  model.row_lower = {3.0, 1.0};
  model.row_upper = {3.0, 2.0};
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.add_entry(0, 2, 1.0);
  model.matrix.add_entry(1, 0, 1.0);
  model.matrix.add_entry(1, 1, -1.0);
  model.matrix.finalize();
  for (Index j = 0; j < 3; ++j) model.hessian.add_entry(j, j, 1.0);
  model.hessian.finalize();
  return model;
}

/// A feasible QP whose rows nearly contradict: x1 + x2 >= 2 - 1e-6 with 0 <= x <= 1, where
/// the only feasible points are within 1e-6 of (1, 1).
Model barely_feasible() {
  Model model = box_infeasible();
  model.row_lower = {2.0 - 1e-6};
  return model;
}

/// `from_engine`: the engine itself returns no point with the verdict (#191); through solve()
/// postsolve allocates the vectors as it does for every verdict and the writer leaves them
/// out (#200), so only the engine's own answer is held to the empty point.
void expect_infeasible_with_proof(const Model& model, const Solution& s,
                                  bool from_engine = true) {
  ASSERT_EQ(s.status, SolveStatus::kInfeasible) << s.message;
  ASSERT_EQ(s.farkas_dual.size(), static_cast<std::size_t>(model.num_rows()));
  std::string why;
  EXPECT_TRUE(farkas_proves_infeasible(model, s.farkas_dual, &why)) << why;
  if (from_engine) {
    EXPECT_TRUE(s.col_value.empty()) << "an infeasible verdict carries no point";
    EXPECT_EQ(s.primal_infeasibility, 0.0) << "nor numbers measured on one (#505)";
  }
  EXPECT_TRUE(s.primal_ray.empty());
}

void expect_unbounded_with_proof(const Model& model, const Solution& s) {
  ASSERT_EQ(s.status, SolveStatus::kUnbounded) << s.message;
  ASSERT_EQ(s.primal_ray.size(), static_cast<std::size_t>(model.num_cols()));
  std::string why;
  EXPECT_TRUE(ray_proves_unbounded(model, s.primal_ray, &why)) << why;
  // The other half of the claim: the reported point is feasible.
  ASSERT_EQ(s.col_value.size(), static_cast<std::size_t>(model.num_cols()));
  Solution measured;
  measured.col_value = s.col_value;
  measured.recompute_quality(model);
  EXPECT_LE(measured.primal_infeasibility_scaled, 1e-7);
  EXPECT_TRUE(s.farkas_dual.empty());
}

TEST(QpIpmInfeasibility, ABoxThatCannotReachTheRowIsProvedInfeasible) {
  const Model model = box_infeasible();
  Logger logger(nullptr);
  const Solution s = qp::solve_convex_qp_ipm(model, quiet(true), logger);
  expect_infeasible_with_proof(model, s);
  EXPECT_NE(s.message.find("#893"), std::string::npos) << s.message;
}

TEST(QpIpmInfeasibility, AProofOverThreeRowsAndTwoFreeColumnsIsFound) {
  const Model model = three_row_infeasible();
  Logger logger(nullptr);
  expect_infeasible_with_proof(model, qp::solve_convex_qp_ipm(model, quiet(true), logger));
}

TEST(QpIpmInfeasibility, ContradictoryEqualityRowsAreProvedInfeasible) {
  const Model model = equality_infeasible();
  Logger logger(nullptr);
  expect_infeasible_with_proof(model, qp::solve_convex_qp_ipm(model, quiet(true), logger));
}

TEST(QpIpmInfeasibility, AFlatImprovingDirectionIsProvedUnbounded) {
  const Model model = flat_direction_unbounded();
  Logger logger(nullptr);
  const Solution s = qp::solve_convex_qp_ipm(model, quiet(true), logger);
  expect_unbounded_with_proof(model, s);
  EXPECT_NE(s.message.find("#893"), std::string::npos) << s.message;
}

TEST(QpIpmInfeasibility, ARayThatMustKeepAnEqualityIsProvedUnbounded) {
  const Model model = equality_ray_unbounded();
  Logger logger(nullptr);
  expect_unbounded_with_proof(model, qp::solve_convex_qp_ipm(model, quiet(true), logger));
}

TEST(QpIpmInfeasibility, AMaximizationIsProvedUnboundedInItsOwnSense) {
  const Model model = maximize_unbounded();
  Logger logger(nullptr);
  expect_unbounded_with_proof(model, qp::solve_convex_qp_ipm(model, quiet(true), logger));
}

TEST(QpIpmInfeasibility, FeasibleQpsAreNeverReportedInfeasibleOrUnbounded) {
  const std::vector<Model> models = {inequality_qp(), unbounded_region_bounded_objective(),
                                     flat_but_bounded(), equality_ranged_free_fixed_qp(),
                                     barely_feasible()};
  for (std::size_t k = 0; k < models.size(); ++k) {
    Logger logger(nullptr);
    const Solution s = qp::solve_convex_qp_ipm(models[k], quiet(true), logger);
    EXPECT_NE(s.status, SolveStatus::kInfeasible) << "model " << k << ": " << s.message;
    EXPECT_NE(s.status, SolveStatus::kUnbounded) << "model " << k << ": " << s.message;
    EXPECT_TRUE(s.farkas_dual.empty()) << "model " << k;
    EXPECT_TRUE(s.primal_ray.empty()) << "model " << k;
    EXPECT_EQ(s.status, SolveStatus::kOptimal) << "model " << k << ": " << s.message;
  }
}

TEST(QpIpmInfeasibility, OffIsTheDefaultAndChangesNothing) {
  EXPECT_FALSE(Options().get_bool("qp_ipm_detect_infeasibility"));
  // Feasible models: the detection only reads the iterates, so on and off take the same
  // steps to the same point.
  const std::vector<Model> feasible = {inequality_qp(), unbounded_region_bounded_objective(),
                                       flat_but_bounded(), equality_ranged_free_fixed_qp(),
                                       barely_feasible()};
  for (std::size_t k = 0; k < feasible.size(); ++k) {
    Logger logger(nullptr);
    const Solution off = qp::solve_convex_qp_ipm(feasible[k], quiet(false), logger);
    const Solution on = qp::solve_convex_qp_ipm(feasible[k], quiet(true), logger);
    EXPECT_EQ(off.status, on.status) << "model " << k;
    EXPECT_EQ(off.iterations, on.iterations) << "model " << k;
    EXPECT_EQ(off.col_value, on.col_value) << "model " << k;
    EXPECT_EQ(off.row_dual, on.row_dual) << "model " << k;
    EXPECT_EQ(off.message, on.message) << "model " << k;
  }
  // Infeasible and unbounded models: off, no verdict and no certificate, as before #893.
  const std::vector<Model> no_optimum = {box_infeasible(),         three_row_infeasible(),
                                         equality_infeasible(),    flat_direction_unbounded(),
                                         equality_ray_unbounded(), maximize_unbounded()};
  for (std::size_t k = 0; k < no_optimum.size(); ++k) {
    Logger logger(nullptr);
    const Solution off = qp::solve_convex_qp_ipm(no_optimum[k], quiet(false), logger);
    EXPECT_NE(off.status, SolveStatus::kInfeasible) << "model " << k << ": " << off.message;
    EXPECT_NE(off.status, SolveStatus::kUnbounded) << "model " << k << ": " << off.message;
    EXPECT_NE(off.status, SolveStatus::kOptimal) << "model " << k << ": " << off.message;
    EXPECT_TRUE(off.farkas_dual.empty()) << "model " << k;
    EXPECT_TRUE(off.primal_ray.empty()) << "model " << k;
    EXPECT_EQ(off.message.find("#893"), std::string::npos) << off.message;
  }
}

TEST(QpIpmInfeasibility, SolveCarriesTheCertificateThroughToTheCaller) {
  // solve() with presolve off runs the engine on the model as given; with presolve on the
  // verdict may come from presolve or from the engine on the reduced model, and either way
  // the certificate that reaches the caller has to prove the claim about THIS model.
  for (const bool presolve : {false, true}) {
    Options options = quiet(true);
    options.set_bool("presolve", presolve);
    {
      const Model model = three_row_infeasible();
      const Solution s = solve(model, options);
      expect_infeasible_with_proof(model, s, /*from_engine=*/false);
      if (!presolve) {
        EXPECT_EQ(s.algorithm, "qp-ipm");
      }
    }
    {
      const Model model = equality_ray_unbounded();
      const Solution s = solve(model, options);
      expect_unbounded_with_proof(model, s);
      if (!presolve) {
        EXPECT_EQ(s.algorithm, "qp-ipm");
      }
    }
  }
}

}  // namespace
}  // namespace sankhya
