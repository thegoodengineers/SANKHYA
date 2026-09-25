// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the debug-solution check (#500).
//
// Three things are shown here:
//
//   1. the point is read from both file formats, by column name;
//   2. a planted invalid cut is caught INSIDE a real search, with the family, the round, the
//      node and the row in the message (the negative control: a check that has never fired
//      is not evidence that it can);
//   3. a fuzz over small random MILPs, each solved exactly by enumeration (every integer
//      assignment, the continuous part of each by the rational simplex), keeping only those
//      whose optimum is unique, then solved by the solver with every cut family, presolve
//      reduction, symmetry, reduced-cost fixing and objective branching on and the exact
//      optimum as the debug solution. Any cut or reduction that removes it aborts the test.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <random>
#include <regex>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/io.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#include "mip/debug_solution.hpp"
#include "presolve/presolve.hpp"

#include "oracles/lp_generator.hpp"
#include "oracles/rational_simplex.hpp"

namespace sankhya {
namespace {

using oracle::Rational;

/// A file in the temp directory unique to this process: ctest runs these tests in parallel,
/// and another checkout may be running them at the same moment.
std::string scratch_path(const std::string& name) {
  static const std::string nonce = std::to_string(std::random_device{}());
  return (std::filesystem::temp_directory_path() / ("sankhya_" + nonce + "_" + name)).string();
}

/// max 3x0 + 2x1 + 4x2 s.t. 2x0 + x1 + 3x2 <= 4, binaries. The feasible points of more than
/// one item are (1,1,0), value 5, and (0,1,1), value 6; so the unique optimum is (0,1,1).
Model small_knapsack() {
  Model model;
  model.name = "debug-knapsack";
  model.sense = ObjSense::kMaximize;
  model.col_cost = {3.0, 2.0, 4.0};
  model.col_lower = {0.0, 0.0, 0.0};
  model.col_upper = {1.0, 1.0, 1.0};
  model.col_type = {VarType::kInteger, VarType::kInteger, VarType::kInteger};
  model.col_names = {"x0", "x1", "x2"};
  model.row_lower = {-kInfinity};
  model.row_upper = {4.0};
  model.matrix.reset(1, 3);
  model.matrix.add_entry(0, 0, 2.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.add_entry(0, 2, 3.0);
  model.matrix.finalize();
  return model;
}

void write_text(const std::string& path, const std::string& text) {
  std::ofstream out(path, std::ios::binary);
  out << text;
}

TEST(DebugSolution, ReadsASankhyaSolAndAMiplibSolByName) {
  const Model model = small_knapsack();
  Solution known;
  known.allocate_for(model);
  known.status = SolveStatus::kOptimal;
  known.col_value = {0.0, 1.0, 1.0};
  const std::string sankhya_path = scratch_path("sankhya_debug_read.sol");
  std::string error;
  ASSERT_TRUE(io::write_solution(sankhya_path, model, known, &error)) << error;
  const auto from_sankhya = mip::DebugSolution::load(sankhya_path, model, &error);
  ASSERT_TRUE(from_sankhya.has_value()) << error;
  EXPECT_EQ(from_sankhya->x(), (std::vector<double>{0.0, 1.0, 1.0}));

  // MIPLIB style: an objective line, names in any order, a zero left out.
  const std::string miplib_path = scratch_path("sankhya_debug_read_miplib.sol");
  write_text(miplib_path, "=obj= 6\nx2 1\n# a comment\nx1 1\n");
  Index missing = -1;
  const auto from_miplib = mip::DebugSolution::load(miplib_path, model, &error, &missing);
  ASSERT_TRUE(from_miplib.has_value()) << error;
  EXPECT_EQ(from_miplib->x(), (std::vector<double>{0.0, 1.0, 1.0}));
  EXPECT_EQ(missing, 1);

  // A file that names none of the columns is refused, not read as the zero vector.
  write_text(miplib_path, "y0 1\ny1 1\n");
  EXPECT_FALSE(mip::DebugSolution::load(miplib_path, model, &error).has_value());
  std::filesystem::remove(sankhya_path);
  std::filesystem::remove(miplib_path);
}

TEST(DebugSolution, NamesTheViolatedCutRowAndBound) {
  const Model model = small_knapsack();
  const mip::DebugSolution point(std::vector<double>{0.0, 1.0, 1.0});
  mip::Cut valid;
  valid.coeff = {1.0, 1.0, 1.0};
  valid.rhs = 2.0;  // at most two items fit: valid, and tight at the point
  EXPECT_TRUE(point.cut_violation(valid).empty());
  mip::Cut invalid = valid;
  invalid.rhs = 1.0;
  EXPECT_NE(point.cut_violation(invalid).find("by 1.000e+00"), std::string::npos)
      << point.cut_violation(invalid);
  EXPECT_TRUE(point.model_violation(model).empty());
  Model tighter = model;
  tighter.row_upper[0] = 3.0;
  EXPECT_NE(point.model_violation(tighter).find("above its upper bound"), std::string::npos);
  Model fixed = model;
  fixed.col_upper[2] = 0.0;
  EXPECT_NE(point.model_violation(fixed).find("column x2"), std::string::npos);
  EXPECT_EQ(point.first_column_outside(fixed.col_lower, fixed.col_upper), 2);
}

Options every_family_on() {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", true);
  for (const char* name :
       {"enable_root_cuts", "enable_mir_cuts", "enable_clique_cuts", "enable_zero_half_cuts",
        "enable_flow_cover_cuts", "presolve_dual_fixing", "presolve_parallel_rows",
        "presolve_dominated_columns", "presolve_implied_free", "mip_symmetry",
        "mip_reduced_cost_fixing", "mip_objective_branching", "conflict_analysis",
        "root_cut_loop", "gmi_safety"}) {
    options.set_bool(name, true);
  }
  options.set_int("tree_cut_depth", 4);
  options.set_int("mip_restarts", 1);
  options.set_int("mip_threads", 1);
  // The filter's floor (#496, #608): on models this small 0.2 n is one or two nonzeros and
  // refuses nearly every cut, so without it the appended-row path would barely be reached.
  options.set_int("cut_support_floor", 100);
  return options;
}

TEST(DebugSolution, APlantedInvalidCutIsCaughtInsideTheSearch) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const Model model = small_knapsack();
  const std::string path = scratch_path("sankhya_debug_planted.sol");
  write_text(path, "x0 0\nx1 1\nx2 1\n");
  Options options = every_family_on();
  options.set_bool("presolve", false);  // the root LP must reach the cut round
  options.set_string("debug_solution", path);

  // Without a planted cut the search runs to the known optimum and nothing fires.
  mip::testing::set_planted_cut(std::nullopt);
  const Solution clean = solve(model, options);
  ASSERT_EQ(clean.status, SolveStatus::kOptimal);
  EXPECT_NEAR(clean.objective, 6.0, 1e-9);

  // x1 + x2 <= 1 is violated by the unique optimum: an invalid "Gomory" cut.
  mip::Cut planted;
  planted.coeff = {0.0, 1.0, 1.0};
  planted.rhs = 1.0;
  planted.family = mip::CutFamily::kGomory;
  mip::testing::set_planted_cut(planted);
  // The patterns use only `.` and `*`: gtest's own regex on Windows has \d and no brackets,
  // POSIX regex on Linux has brackets and no \d, and CI runs both.
  EXPECT_DEATH((void)solve(model, options),
               "debug solution cut off: gomory cut, root round 1, node 0, row .*: activity 2 "
               "> rhs 1");
  mip::testing::set_planted_cut(std::nullopt);
  std::filesystem::remove(path);
}

TEST(DebugSolution, APointTheModelExcludesIsRefusedBeforePresolve) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  // A checking run against a point that is not feasible would check nothing: refused.
  const Model model = small_knapsack();
  const std::string path = scratch_path("sankhya_debug_outside.sol");
  write_text(path, "x0 1\nx1 1\nx2 1\n");  // weight 6 > 4
  Options options = every_family_on();
  options.set_string("debug_solution", path);
  EXPECT_DEATH((void)solve(model, options),
               "debug solution cut off: the debug solution is not feasible for the model as "
               "given: row");
  std::filesystem::remove(path);
}

TEST(DebugSolution, APlantedInvalidPresolveReductionIsCaught) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const Model model = small_knapsack();
  const std::string path = scratch_path("sankhya_debug_presolve.sol");
  write_text(path, "x0 0\nx1 1\nx2 1\n");
  Options options = every_family_on();
  options.set_string("debug_solution", path);
  Logger quiet(nullptr);
  presolve::Result reduced = presolve::presolve(model, options, quiet);
  // Presolve as it is keeps the optimum.
  mip::check_presolve_against_debug_solution(model, reduced, options, quiet);
  // A reduction that fixes x2 at 0 removes it, and is named.
  presolve::Record wrong;
  wrong.kind = presolve::Record::Kind::kFixedColumn;
  wrong.index = 2;
  wrong.value = 0.0;
  reduced.records.push_back(wrong);
  EXPECT_DEATH(mip::check_presolve_against_debug_solution(model, reduced, options, quiet),
               "debug solution cut off: presolve reduction .* .fixed column. fixed column "
               "x2 at 0; the debug solution has 1");
  std::filesystem::remove(path);
}

// =========================================================================================
// The fuzz: exact enumeration, unique optima, every family on
// =========================================================================================

struct Enumerated {
  bool usable = false;  ///< optimal, unique, and every sub-LP settled
  Rational objective;
  std::vector<Rational> x;
};

/// Every integer assignment in the box, the continuous columns of each by the exact simplex
/// (a pure-integer point is evaluated directly, in integers). Usable only when exactly one
/// assignment attains the optimum.
Enumerated enumerate(const oracle::GeneratedLp& lp) {
  Enumerated result;
  std::vector<Index> integers;
  for (Index j = 0; j < lp.num_cols; ++j) {
    if (lp.integral[static_cast<std::size_t>(j)] != 0) integers.push_back(j);
  }
  const bool pure = static_cast<Index>(integers.size()) == lp.num_cols;
  std::vector<std::int64_t> value(integers.size(), 0);
  int ties = 0;
  bool have = false;
  while (true) {
    oracle::OracleResult sub;
    if (pure) {
      sub.status = oracle::OracleStatus::kOptimal;
      for (Index i = 0; i < lp.num_rows && sub.status == oracle::OracleStatus::kOptimal; ++i) {
        std::int64_t activity = 0;
        for (std::size_t k = 0; k < value.size(); ++k) {
          activity += lp.a[static_cast<std::size_t>(i)][k] * value[k];
        }
        if (activity < lp.b[static_cast<std::size_t>(i)]) {
          sub.status = oracle::OracleStatus::kInfeasible;
        }
      }
      std::int64_t objective = 0;
      for (std::size_t k = 0; k < value.size(); ++k) objective += lp.c[k] * value[k];
      sub.objective = Rational(objective);
      sub.x.assign(value.begin(), value.end());
    } else {
      oracle::GeneratedLp fixed = lp;
      fixed.lower.assign(static_cast<std::size_t>(lp.num_cols), 0);
      for (std::size_t k = 0; k < integers.size(); ++k) {
        const auto u = static_cast<std::size_t>(integers[k]);
        fixed.lower[u] = value[k];
        fixed.upper[u] = value[k];
      }
      sub = oracle::solve_exact(fixed);
      if (sub.status == oracle::OracleStatus::kUnbounded ||
          sub.status == oracle::OracleStatus::kOverflow ||
          sub.status == oracle::OracleStatus::kIterationLimit) {
        return result;  // not settled: unusable
      }
    }
    if (sub.status == oracle::OracleStatus::kOptimal) {
      if (!have || sub.objective < result.objective) {
        have = true;
        ties = 0;
        result.objective = sub.objective;
        result.x = sub.x;
      } else if (sub.objective == result.objective) {
        ++ties;
      }
    }
    std::size_t k = 0;
    for (; k < integers.size(); ++k) {
      const auto u = static_cast<std::size_t>(integers[k]);
      if (value[k] < lp.upper[u]) {
        ++value[k];
        break;
      }
      value[k] = 0;
    }
    if (k == integers.size()) break;
  }
  result.usable = have && ties == 0;
  return result;
}

std::string exact_text(double v) {
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "%.17g", v);
  return buffer;
}

/// What the fuzz saw, over all instances.
struct FuzzTally {
  int checked = 0;
  std::int64_t cuts = 0;
  std::int64_t nodes = 0;
  std::int64_t aged_out = 0;     ///< cut rows freed by age (#497)
  std::int64_t reactivated = 0;  ///< freed cut rows re-imposed by the cut pool (#497)
  /// Root candidates per family, read off the filter's report: every one of them was checked
  /// against the optimum BEFORE the filter, so this is how much the check saw.
  std::map<std::string, std::int64_t> candidates;
};

/// Enumerate `lp`; when its optimum is unique, solve it with every family on and that optimum
/// as the debug solution. A cut or reduction that removes it aborts the process.
void check_instance(const oracle::GeneratedLp& lp, int attempt, FuzzTally* tally,
                    Options options = every_family_on()) {
  const std::string path = scratch_path("debug_fuzz_" + std::to_string(attempt) + ".sol");
  const Enumerated exact = enumerate(lp);
  if (!exact.usable) return;
  // The two exact oracles must agree before either is trusted as the reference.
  const oracle::OracleResult tree = oracle::solve_exact_milp(lp, 20000);
  ASSERT_EQ(tree.status, oracle::OracleStatus::kOptimal) << lp.to_text();
  ASSERT_TRUE(tree.objective == exact.objective) << lp.to_text();

  Model model = oracle::to_model(lp);
  for (Index j = 0; j < model.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (lp.integral[u] != 0) model.col_type[u] = VarType::kInteger;
  }
  // Unnamed columns: the pipeline must name them before presolve renumbers anything.
  std::string text = "=obj= " + exact_text(exact.objective.to_double()) + "\n";
  for (Index j = 0; j < model.num_cols(); ++j) {
    text += "C" + std::to_string(j) + " " +
            exact_text(exact.x[static_cast<std::size_t>(j)].to_double()) + "\n";
  }
  write_text(path, text);
  options.set_string("debug_solution", path);
  // A failure aborts the process, so the instance is printed first when asked for: a
  // failing instance nobody can reproduce is not evidence.
  if (const char* dump = std::getenv("SANKHYA_DEBUG_FUZZ_DUMP"); dump != nullptr) {
    std::fprintf(stderr, "INSTANCE %d\n%s\n", attempt, lp.to_text().c_str());
    (void)io::write_mps(std::string(dump) + ".mps", model, nullptr);
    write_text(std::string(dump) + ".sol", text);
  }
  const Solution s = solve(model, options);
  std::filesystem::remove(path);
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << lp.to_text();
  const double expected = exact.objective.to_double();
  EXPECT_LE(std::fabs(s.objective - expected), 1e-6 * std::max(1.0, std::fabs(expected)))
      << lp.to_text();
  ++tally->checked;
  tally->cuts += s.cuts_applied;
  tally->nodes += s.nodes;
  tally->aged_out += s.cut_rows_aged_out;
  tally->reactivated += s.cuts_reactivated;
  static const std::regex family_count("([a-z_]+) ([0-9]+):");
  for (auto it = std::sregex_iterator(s.cut_filter_report.begin(), s.cut_filter_report.end(),
                                      family_count);
       it != std::sregex_iterator(); ++it) {
    tally->candidates[(*it)[1].str()] += std::stoll((*it)[2].str());
  }
}

std::string describe(const FuzzTally& tally, std::int64_t* total) {
  std::string seen;
  *total = 0;
  for (const auto& [family, count] : tally.candidates) {
    seen += " " + family + " " + std::to_string(count);
    *total += count;
  }
  return seen;
}

TEST(DebugSolution, FuzzRandomMilpsAgainstTheUniqueExactOptimum) {
  std::mt19937_64 rng(20260924);
  oracle::GeneratorConfig config;
  config.min_rows = 2;
  config.max_rows = 5;
  config.min_cols = 3;
  config.max_cols = 6;
  config.magnitude = 5;
  std::uniform_int_distribution<std::int64_t> cost(-40, 40);
  std::uniform_int_distribution<int> coin(0, 1);
  FuzzTally tally;
  int mixed = 0;
  for (int attempt = 0; attempt < 2500 && tally.checked < 150; ++attempt) {
    oracle::GeneratedLp lp = oracle::random_lp(rng, config);
    // Half the instances pure integer, half mixed (every other column continuous). Integer
    // columns are boxed to [0, 3] so the enumeration is at most 4^6 assignments.
    const bool pure = coin(rng) == 0;
    lp.integral.assign(static_cast<std::size_t>(lp.num_cols), 1);
    for (Index j = 0; j < lp.num_cols; ++j) {
      const auto u = static_cast<std::size_t>(j);
      if (!pure && j % 2 == 1) {
        lp.integral[u] = 0;
        if (lp.upper[u] == oracle::kNoUpperBound) lp.upper[u] = 8;
      } else {
        lp.upper[u] = std::min<std::int64_t>(lp.upper[u], 3);
      }
      // Wide, nonzero costs: ties between assignments become rare, and an empty column
      // always has a cost that decides its value.
      do {
        lp.c[u] = cost(rng);
      } while (lp.c[u] == 0);
    }
    const int before = tally.checked;
    check_instance(lp, attempt, &tally);
    if (HasFatalFailure()) return;
    if (!pure && tally.checked > before) ++mixed;
  }
  std::int64_t total = 0;
  const std::string seen = describe(tally, &total);
  EXPECT_GE(tally.checked, 60) << "too few instances with a unique exact optimum";
  EXPECT_GE(mixed, 15) << "too few mixed instances";
  EXPECT_GE(tally.candidates.size(), 2u) << "fewer than two cut families produced a candidate";
  std::printf(
      "[  INFO    ] debug solution, random: %d instances (%d mixed), root candidates "
      "checked:%s; %lld cut rows appended, %lld nodes; the unique optimum never cut off\n",
      tally.checked, mixed, seen.c_str(), static_cast<long long>(tally.cuts),
      static_cast<long long>(tally.nodes));
}

/// A structured pure-binary instance, 8 to 12 columns: knapsack rows (sum a x <= b, a > 0)
/// are what covers and cliques are built from, and covering rows (sum x >= 1) are what the
/// zero-half cuts combine.
oracle::GeneratedLp binary_rows_instance(std::mt19937_64& rng) {
  std::uniform_int_distribution<Index> width(8, 12);
  std::uniform_int_distribution<Index> height(2, 4);
  std::uniform_int_distribution<std::int64_t> weight(1, 9);
  std::uniform_int_distribution<std::int64_t> profit(1, 40);
  std::uniform_int_distribution<int> percent(0, 99);
  oracle::GeneratedLp lp;
  lp.num_cols = width(rng);
  const Index knapsacks = height(rng);
  const Index covers = height(rng) - 1;
  lp.num_rows = knapsacks + covers;
  const auto n = static_cast<std::size_t>(lp.num_cols);
  lp.integral.assign(n, 1);
  lp.upper.assign(n, 1);
  lp.c.resize(n);
  for (std::size_t j = 0; j < n; ++j) lp.c[j] = -profit(rng);  // maximise profit
  for (Index i = 0; i < knapsacks; ++i) {
    std::vector<std::int64_t> row(n, 0);
    std::int64_t sum = 0;
    for (std::size_t j = 0; j < n; ++j) {
      if (percent(rng) < 70) row[j] = weight(rng);
      sum += row[j];
    }
    // sum a x <= b, as -a x >= -b, with b about half the row's weight.
    for (std::int64_t& a : row) a = -a;
    lp.a.push_back(row);
    lp.b.push_back(-std::max<std::int64_t>(1, sum / 2));
  }
  for (Index i = 0; i < covers; ++i) {
    std::vector<std::int64_t> row(n, 0);
    for (std::size_t j = 0; j < n; ++j) row[j] = percent(rng) < 30 ? 1 : 0;
    lp.a.push_back(row);
    lp.b.push_back(1);
  }
  return lp;
}

TEST(DebugSolution, FuzzBinaryKnapsackAndCoveringRowsAgainstTheUniqueExactOptimum) {
  // Structured pure-binary instances, 8 to 12 columns: knapsack rows (sum a x <= b, a > 0)
  // are what covers and cliques are built from, and covering rows (sum x >= 1) are what the
  // zero-half cuts combine. At this width the density cap lets some cuts through, so the
  // appended-row path is checked on real separators and not only on the planted cut.
  std::mt19937_64 rng(20260925);
  FuzzTally tally;
  for (int attempt = 0; attempt < 1200 && tally.checked < 150; ++attempt) {
    const oracle::GeneratedLp lp = binary_rows_instance(rng);
    check_instance(lp, attempt, &tally);
    if (HasFatalFailure()) return;
  }
  std::int64_t total = 0;
  const std::string seen = describe(tally, &total);
  EXPECT_GE(tally.checked, 40) << "too few instances with a unique exact optimum";
  EXPECT_GE(tally.candidates.size(), 2u) << "fewer than two cut families produced a candidate";
  EXPECT_GT(total, 0) << "no root candidate: the check saw no cut";
  std::printf(
      "[  INFO    ] debug solution, binary: %d instances, root candidates checked:%s; %lld "
      "cut rows appended, %lld nodes; the unique optimum never cut off\n",
      tally.checked, seen.c_str(), static_cast<long long>(tally.cuts),
      static_cast<long long>(tally.nodes));
}

// The cut pool (#497) inside the same check: with an age limit of 1 a cut row is freed at
// almost every node where it is slack, and re-imposed wherever a later node's point violates
// it. Every node LP whose box holds the unique optimum is checked against it after the rows
// come back, so a row re-imposed wrongly - the wrong right-hand side, the wrong row, a node
// re-solve that goes astray - aborts naming the node. The tally must show the pool worked.
TEST(DebugSolution, FuzzTheCutPoolAgainstTheUniqueExactOptimum) {
  std::mt19937_64 rng(20260925);
  Options options = every_family_on();
  options.set_bool("mip_cut_pooling", true);
  options.set_int("mip_cut_age_limit", 1);
  // A tree to age rows in: at the settings above the root closes almost every one of these
  // instances, and a pool with no second node never re-imposes anything. Two cuts a round,
  // no loop and no heuristics leave the root open; the tree rounds still add rows.
  options.set_bool("root_cut_loop", false);
  options.set_int("cut_max_per_round", 2);
  options.set_int("tree_cut_rows_per_round", 2);
  options.set_bool("mip_heuristics", false);
  FuzzTally tally;
  for (int attempt = 0; attempt < 1200 && tally.checked < 150; ++attempt) {
    const oracle::GeneratedLp lp = binary_rows_instance(rng);
    check_instance(lp, attempt, &tally, options);
    if (HasFatalFailure()) return;
  }
  EXPECT_GE(tally.checked, 40) << "too few instances with a unique exact optimum";
  EXPECT_GT(tally.aged_out, 0) << "no cut row was ever freed";
  EXPECT_GT(tally.reactivated, 0) << "no freed cut row was ever re-imposed";
  std::printf(
      "[  INFO    ] debug solution, cut pool: %d instances, %lld cut rows appended, %lld "
      "freed by age, %lld re-imposed, %lld nodes; the unique optimum never cut off\n",
      tally.checked, static_cast<long long>(tally.cuts), static_cast<long long>(tally.aged_out),
      static_cast<long long>(tally.reactivated), static_cast<long long>(tally.nodes));
}

}  // namespace
}  // namespace sankhya
