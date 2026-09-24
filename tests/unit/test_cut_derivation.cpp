// SPDX-License-Identifier: Apache-2.0
// SANKHYA - certified derivations of cut rows for MILP certificates (#518).
//
// tools/test_verify_certificate_cuts.py checks the written derivations in exact arithmetic
// without linking this code. These tests hold the C++ side: each family's derivation is
// accepted for the cut its generator built, a cut changed by one coefficient is refused, and
// a search with root cuts on writes a certificate whose cut rows are derived.

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#include "mip/combinatorial_cuts.hpp"
#include "mip/cut_derivation.hpp"
#include "mip/cuts.hpp"
#include "mip/mir_cuts.hpp"
#include "support/temp_file.hpp"

namespace sankhya::mip {
namespace {

constexpr double kInf = kInfinity;

Model make_model(const std::vector<std::vector<double>>& a, std::vector<double> row_lower,
                 std::vector<double> row_upper, std::vector<double> col_lower,
                 std::vector<double> col_upper, std::vector<bool> integer) {
  Model m;
  const auto n = static_cast<Index>(col_lower.size());
  m.resize_columns(n);
  m.col_lower = std::move(col_lower);
  m.col_upper = std::move(col_upper);
  for (std::size_t j = 0; j < integer.size(); ++j) {
    m.col_type[j] = integer[j] ? VarType::kInteger : VarType::kContinuous;
  }
  m.resize_rows(static_cast<Index>(a.size()));
  m.row_lower = std::move(row_lower);
  m.row_upper = std::move(row_upper);
  m.matrix.reset(static_cast<Index>(a.size()), n);
  for (std::size_t i = 0; i < a.size(); ++i) {
    for (std::size_t j = 0; j < a[i].size(); ++j) {
      if (a[i][j] != 0.0)
        m.matrix.add_entry(static_cast<Index>(i), static_cast<Index>(j), a[i][j]);
    }
  }
  m.matrix.finalize();
  return m;
}

Cut cover_cut(std::vector<double> coeff, double rhs, Index row) {
  Cut cut;
  cut.family = CutFamily::kKnapsackCover;
  cut.coeff = std::move(coeff);
  cut.rhs = rhs;
  auto d = std::make_shared<CutDerivation>();
  d->kind = CutDerivation::Kind::kSingleRowRounding;
  d->row = row;
  cut.derivation = std::move(d);
  return cut;
}

Solution lp_solution(Model model) {
  model.col_type.assign(static_cast<std::size_t>(model.num_cols()), VarType::kContinuous);
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", false);
  options.set_string("algorithm", "dual-simplex");
  return solve(model, options);
}

TEST(CutDerivation, ACoverIsARoundingOfItsRowAndAStrongerCutIsNot) {
  // 5 x0 + 4 x1 + 3 x2 <= 7 over binaries: {0, 1} is a cover, x0 + x1 <= 1 its inequality,
  // 1/5 of the row rounded down. 2 x0 + x1 <= 1 cuts off (1, 0, 0) and must be refused.
  const Model m =
      make_model({{5, 4, 3}}, {-kInf}, {7}, {0, 0, 0}, {1, 1, 1}, {true, true, true});
  std::vector<Cut> cuts{cover_cut({1, 1, 0}, 1, 0), cover_cut({2, 1, 0}, 1, 0)};
  const CutCertification outcome = certify_cuts(m, m.col_lower, m.col_upper, &cuts);
  EXPECT_EQ(outcome.derived, 1);
  EXPECT_EQ(outcome.dropped, 1);
  ASSERT_EQ(cuts.size(), 1u);
  EXPECT_EQ(cuts[0].coeff, (std::vector<double>{1, 1, 0}));
  ASSERT_NE(cuts[0].proof, nullptr);
  EXPECT_FALSE(cuts[0].proof->split);
  ASSERT_EQ(cuts[0].proof->side[0].rows.size(), 1u);
  EXPECT_LT(cuts[0].proof->side[0].rows[0].second, 0.0);  // the row's upper side
}

TEST(CutDerivation, ACutWithNoDerivationIsDropped) {
  const Model m =
      make_model({{5, 4, 3}}, {-kInf}, {7}, {0, 0, 0}, {1, 1, 1}, {true, true, true});
  Cut cut = cover_cut({1, 1, 0}, 1, 0);
  cut.derivation = nullptr;
  std::vector<Cut> cuts{cut};
  const CutCertification outcome = certify_cuts(m, m.col_lower, m.col_upper, &cuts);
  EXPECT_EQ(outcome.dropped, 1);
  EXPECT_TRUE(cuts.empty());
  EXPECT_NE(outcome.first_failure.find("no derivation"), std::string::npos);
}

TEST(CutDerivation, EveryMirCutIsASplitTheProofReaches) {
  // x - y <= 1.5 with x integer in [0, 3] and y >= 0: the MIR cut x - 2 y <= 1 at (1.5, 0).
  const Model m = make_model({{1, -1}}, {-kInf}, {1.5}, {0, 0}, {3, kInf}, {true, false});
  Solution at;
  at.col_value = {1.5, 0.0};
  MirOptions options;
  options.derive = true;
  std::vector<Cut> cuts = generate_mir_cuts(m, at, m.col_lower, m.col_upper, nullptr, options);
  ASSERT_FALSE(cuts.empty());
  const std::size_t found = cuts.size();
  const CutCertification outcome = certify_cuts(m, m.col_lower, m.col_upper, &cuts);
  EXPECT_EQ(outcome.derived, static_cast<Count>(found)) << outcome.first_failure;
  for (const Cut& cut : cuts) {
    ASSERT_NE(cut.proof, nullptr);
    EXPECT_TRUE(cut.proof->split);
  }
  // The same cut made stronger on y (x - y <= 1) is no longer a split of this row.
  Cut stronger = cuts[0];
  stronger.proof = nullptr;
  stronger.coeff[1] = -1.0;
  stronger.rhs = 1.0;
  std::vector<Cut> bad{stronger};
  EXPECT_EQ(certify_cuts(m, m.col_lower, m.col_upper, &bad).dropped, 1);
}

TEST(CutDerivation, GomoryCutsFromTheRootTableauAreSplitsTheProofReaches) {
  // min -x0 - 2 x1  s.t.  2 x0 + 2 x1 <= 3, -2 x0 + 2 x1 <= 1, integers in [0, 10], and a
  // continuous x2 in 3 x0 + x1 - x2 <= 4: the LP vertex (1/2, 1) is fractional.
  const Model m = make_model({{2, 2, 0}, {-2, 2, 0}, {3, 1, -1}}, {-kInf, -kInf, -kInf},
                             {3, 1, 4}, {0, 0, 0}, {10, 10, 5}, {true, true, false});
  Model with_cost = m;
  with_cost.col_cost = {-1, -2, 0};
  const Solution root = lp_solution(with_cost);
  ASSERT_EQ(root.status, SolveStatus::kOptimal);
  ASSERT_NEAR(root.col_value[0], 0.5, 1e-9);
  std::vector<Cut> cuts = generate_gmi_cuts(with_cost, root, /*safe=*/false, /*derive=*/true);
  ASSERT_FALSE(cuts.empty());
  const std::size_t found = cuts.size();
  const CutCertification outcome =
      certify_cuts(with_cost, with_cost.col_lower, with_cost.col_upper, &cuts);
  EXPECT_EQ(outcome.derived, static_cast<Count>(found)) << outcome.first_failure;
  // Without `derive` the cuts are the same rows, and carry nothing.
  const std::vector<Cut> plain = generate_gmi_cuts(with_cost, root);
  ASSERT_EQ(plain.size(), found);
  for (const Cut& cut : plain) EXPECT_EQ(cut.derivation, nullptr);
}

TEST(CutDerivation, ZeroHalfCutsAreRoundingsOfHalfTheirRows) {
  // The odd triangle: x0 + x1 <= 1, x1 + x2 <= 1, x0 + x2 <= 1 gives sum x <= 1.
  const Model m = make_model({{1, 1, 0}, {0, 1, 1}, {1, 0, 1}}, {-kInf, -kInf, -kInf},
                             {1, 1, 1}, {0, 0, 0}, {1, 1, 1}, {true, true, true});
  Solution at;
  at.col_value = {0.5, 0.5, 0.5};
  std::vector<Cut> cuts =
      generate_zero_half_cuts(m, at, m.col_lower, m.col_upper, nullptr, /*derive=*/true);
  ASSERT_FALSE(cuts.empty());
  const std::size_t found = cuts.size();
  const CutCertification outcome = certify_cuts(m, m.col_lower, m.col_upper, &cuts);
  EXPECT_EQ(outcome.derived, static_cast<Count>(found)) << outcome.first_failure;
}

TEST(CutDerivation, ASearchWithRootCutsWritesTheirDerivations) {
  Model m = make_model({{2, 3, 1}, {4, 1, 2}, {3, 4, 2}}, {-kInf, -kInf, -kInf}, {5, 11, 8},
                       {0, 0, 0}, {10, 10, 10}, {true, true, true});
  m.col_cost = {-5, -4, -3};
  const testing::TempFile file("", ".vipr");
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("enable_root_cuts", true);
  options.set_string("write_certificate", file.path());
  const Solution solved = solve(m, options);
  ASSERT_EQ(solved.status, SolveStatus::kOptimal);
  EXPECT_DOUBLE_EQ(solved.objective, -13.0);
  std::ifstream in(file.path());
  std::stringstream text;
  text << in.rdbuf();
  EXPECT_NE(text.str().find("RTP range -13 -13"), std::string::npos)
      << text.str().substr(0, 400);
  // The model is chosen so that the root round applies a cut; without one the check below
  // would pass vacuously.
  ASSERT_GT(solved.cuts_applied, 0);
  EXPECT_NE(text.str().find("\ncut0 L "), std::string::npos) << text.str();
}

}  // namespace
}  // namespace sankhya::mip
