// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the dual simplex does not cycle on feasibility claims fresh factors refute.
//
// THE CASE. tests/data/dual_refuted_feasibility.mps is a node LP of neos-3072252-nete
// (MIPLIB 2017) after the root cut loop (#495): the model's 432 rows, 433 cut rows and the
// symmetry row, with a node's bounds. tests/data/dual_refuted_feasibility.bas is the parent's
// optimal basis the tree warm-started it from (BasisStatus values, columns then rows). From
// that start the dual simplex reached a basis whose updated basic values said primal
// feasible and whose recomputed values said 7.56e-6 infeasible, pivoted, and came back: a
// refactorization every second pivot at a fixed objective, 27,573 iterations until the
// time limit, where the unscaled retry then finished in 22. The loop now hands such a basis
// to the primal loop after kRefutedFeasibilityLimit refuted claims (dual_simplex.cpp).

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <gtest/gtest.h>

#include "sankhya/io.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"
#include "simplex/primal_simplex.hpp"

namespace sankhya {
namespace {

std::string fixture(const std::string& name) {
  return (std::filesystem::path(__FILE__).parent_path().parent_path() / "data" / name).string();
}

TEST(DualRefutedFeasibility, TheNeteNodeLpFinishesWithoutCycling) {
  Model model;
  const std::string path = fixture("dual_refuted_feasibility.mps");
  const io::ReadResult read = io::read_model(path, &model);
  ASSERT_TRUE(read.ok) << path << ": " << read.error;
  ASSERT_EQ(model.num_rows(), 866);
  ASSERT_EQ(model.num_cols(), 576);

  WarmStart warm;
  std::ifstream basis(fixture("dual_refuted_feasibility.bas"));
  ASSERT_TRUE(basis.good());
  for (auto* statuses : {&warm.col_status, &warm.row_status}) {
    const Index count = statuses == &warm.col_status ? model.num_cols() : model.num_rows();
    for (Index k = 0; k < count; ++k) {
      int value = 0;
      ASSERT_TRUE(basis >> value);
      statuses->push_back(static_cast<BasisStatus>(value));
    }
  }

  // The options branch and bound gives its node LPs (with_node_lp_defaults): the textbook
  // ratio test. The time limit is a net: the cycle ran into it, the fix never comes near.
  Options options;
  options.set_string("dual_ratio_test", "textbook");
  options.set_double("time_limit", 60.0);
  Logger quiet(nullptr);
  Timer clock;
  const Solution solved = solve_dual_simplex(model, options, quiet, nullptr, &warm);
  std::cout << "dual refuted feasibility: " << solved.iterations << " iterations, "
            << clock.elapsed_seconds() << " s; " << solved.message << "\n";
  ASSERT_EQ(solved.status, SolveStatus::kOptimal) << solved.message;
  // Not rescued by the unscaled retry after the scaled attempt ran out of time: before the
  // fix the message read "route: the scaled attempt returned time_limit ...".
  EXPECT_EQ(solved.message.find("time_limit"), std::string::npos) << solved.message;
  EXPECT_LT(solved.iterations, 2000);
  // Optimal as measured on the model: a basis that is primal and dual feasible.
  EXPECT_LE(solved.primal_infeasibility, tol::kPrimalFeasibility);
  EXPECT_LE(solved.dual_infeasibility, tol::kDualFeasibility);
}

}  // namespace
}  // namespace sankhya
