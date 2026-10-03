// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the learned engine selection (#477): the generated C++ tree picks what the
// trainer's tree picked on every instance it was trained and tested on, the option is off by
// default, and when on it decides only inside the range it was trained on, names the path it
// took, and leaves an explicit engine and a warm start alone.

#include <cstring>
#include <string>

#include <gtest/gtest.h>

#include "core/engine_features.hpp"
#include "core/engine_selection.hpp"
#include "core/engine_selection_tree.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

struct TreeCase {
  const char* instance;
  const char* engine;
  double x[32];
};

// kTreeCaseFeatures and kTreeCases, written by bench/runners/engine_selection_cart.py.
#include "engine_selection_tree_cases.inc"

bool set_feature(EngineFeatures* f, const std::string& name, double v) {
  if (name == "rows") {
    f->rows = static_cast<Index>(v);
  } else if (name == "columns") {
    f->columns = static_cast<Index>(v);
  } else if (name == "nonzeros") {
    f->nonzeros = static_cast<Count>(v);
  } else if (name == "density") {
    f->density = v;
  } else if (name == "max_column_count") {
    f->max_column_count = static_cast<Index>(v);
  } else if (name == "dense_columns") {
    f->dense_columns = static_cast<Index>(v);
  } else if (name == "rows_per_column") {
    f->rows_per_column = v;
  } else if (name == "equality_row_share") {
    f->equality_row_share = v;
  } else if (name == "integer_share") {
    f->integer_share = v;
  } else if (name == "boxed_column_share") {
    f->boxed_column_share = v;
  } else if (name == "free_column_share") {
    f->free_column_share = v;
  } else if (name == "fixed_column_share") {
    f->fixed_column_share = v;
  } else if (name == "normal_equations_ratio") {
    f->normal_equations_ratio = v;
  } else if (name == "cholesky_nonzeros") {
    f->cholesky_nonzeros = v;
  } else if (name == "cholesky_fill_ratio") {
    f->cholesky_fill_ratio = v;
  } else {
    return false;
  }
  return true;
}

TEST(EngineSelectionTree, TheGeneratedRulesAreTheTrainedTree) {
  constexpr std::size_t kFeatures = sizeof(kTreeCaseFeatures) / sizeof(kTreeCaseFeatures[0]);
  static_assert(kFeatures <= 32, "TreeCase holds at most 32 features");
  ASSERT_GT(sizeof(kTreeCases) / sizeof(kTreeCases[0]), 0u);
  for (const TreeCase& c : kTreeCases) {
    EngineFeatures f;
    for (std::size_t k = 0; k < kFeatures; ++k) {
      ASSERT_TRUE(set_feature(&f, kTreeCaseFeatures[k], c.x[k])) << kTreeCaseFeatures[k];
    }
    const LearnedTreeChoice choice = learned_engine_tree(f);
    EXPECT_STREQ(choice.algorithm.c_str(), c.engine) << c.instance << ": " << choice.path;
  }
}

/// A feasible, bounded LP: `rows` rows, `cols` columns, two entries per column.
Model small_lp(Index rows, Index cols) {
  Model m;
  m.resize_columns(cols);
  m.resize_rows(rows);
  m.matrix.reset(rows, cols);
  for (Index j = 0; j < cols; ++j) {
    m.matrix.add_entry(j % rows, j, 1.0);
    m.matrix.add_entry((j + 1) % rows, j, 1.0);
  }
  m.matrix.finalize();
  for (Index j = 0; j < cols; ++j) {
    const auto u = static_cast<std::size_t>(j);
    m.col_cost[u] = 1.0 + static_cast<double>(j % 3);
    m.col_lower[u] = 0.0;
    m.col_upper[u] = 4.0;
  }
  for (Index i = 0; i < rows; ++i) {
    const auto u = static_cast<std::size_t>(i);
    m.row_lower[u] = 1.0;
    m.row_upper[u] = kInfinity;
  }
  return m;
}

Options learned_options() {
  Options o;
  o.set_bool("log_to_console", false);
  o.set_string("algorithm_selection", "learned");
  return o;
}

TEST(EngineSelectionTree, OffByDefault) {
  Options o;
  EXPECT_EQ(o.get_string("algorithm_selection"), "rules");
  const EngineSelection s = select_engine(small_lp(20, 40), o, false);
  EXPECT_EQ(s.rule.rfind("learned", 0), std::string::npos) << s.rule;
  EXPECT_EQ(s.reason.find("learned tree"), std::string::npos) << s.reason;
}

TEST(EngineSelectionTree, InsideItsRangeTheTreeDecidesAndNamesItsPath) {
  const Model m = small_lp(20, 40);
  ASSERT_LE(m.num_rows(), learned_tree_domain().max_rows);
  const EngineSelection s = select_engine(m, learned_options(), false);
  const LearnedTreeChoice expected =
      learned_engine_tree(compute_engine_features(m, /*symbolic=*/true));
  EXPECT_EQ(s.algorithm, expected.algorithm);
  EXPECT_EQ(s.rule, "learned:" + expected.algorithm);
  EXPECT_EQ(s.reason.rfind("learned tree: ", 0), 0u) << s.reason;
  EXPECT_NE(s.reason.find("-> " + expected.algorithm), std::string::npos) << s.reason;
  if (!expected.path.empty()) {
    EXPECT_NE(s.reason.find(expected.path), std::string::npos) << s.reason;
  }
}

TEST(EngineSelectionTree, AnExplicitEngineAndAWarmStartAreLeftAlone) {
  Options o = learned_options();
  o.set_string("algorithm", "pdhg");
  EXPECT_EQ(select_engine(small_lp(20, 40), o, false).rule, "requested");
  const EngineSelection warm = select_engine(small_lp(20, 40), learned_options(), true);
  EXPECT_EQ(warm.rule, "warm-start");
  EXPECT_EQ(warm.algorithm, "dual-simplex");
}

TEST(EngineSelectionTree, OutsideItsRangeTheRuleTableDecidesAndSaysWhy) {
  const LearnedTreeDomain domain = learned_tree_domain();
  const Index rows = domain.max_rows + 1;
  const Model big = small_lp(rows, rows);
  Options rules;
  rules.set_bool("log_to_console", false);
  const EngineSelection table = select_engine(big, rules, false);
  const EngineSelection s = select_engine(big, learned_options(), false);
  EXPECT_EQ(s.rule, table.rule);
  EXPECT_EQ(s.algorithm, table.algorithm);
  EXPECT_EQ(s.reason.rfind("learned tree not consulted (", 0), 0u) << s.reason;
  EXPECT_NE(s.reason.find(table.reason), std::string::npos) << s.reason;
}

TEST(EngineSelectionTree, WithAGpuTheRuleTableDecides) {
  const EngineSelection s =
      select_engine(small_lp(20, 40), learned_options(), false, true, "test device");
  EXPECT_EQ(s.rule.rfind("learned:", 0), std::string::npos) << s.rule;
  EXPECT_NE(s.reason.find("trained on CPU timings"), std::string::npos) << s.reason;
}

TEST(EngineSelectionTree, TheAnswerCarriesTheLearnedRule) {
  const Solution s = solve(small_lp(20, 40), learned_options());
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_EQ(s.engine_rule.rfind("learned:", 0), 0u) << s.engine_rule;
  EXPECT_EQ(s.engine_reason.rfind("learned tree: ", 0), 0u) << s.engine_reason;
}

}  // namespace
}  // namespace sankhya
