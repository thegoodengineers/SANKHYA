// SPDX-License-Identifier: Apache-2.0
// SANKHYA - strong-branch fixing, incremental propagation to a fixpoint, and the heap open
// list (#502).
//
// Each is behind its own option and off by default. The tests below check, for each, that it
// changes no answer AND that it actually did something on a model built so that it must:
// an equality that holds because the option never ran would prove nothing. What it did is
// read from the profiler's counters (profile_out), the same channel #501's cache uses.
//
// References:
//   Achterberg, Koch and Martin, "Branching rules revisited", Operations Research Letters 33
//     (2005), 42-54
//   Achterberg, "Constraint Integer Programming" (thesis, TU Berlin, 2007), ch. 5-7

#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "support/temp_file.hpp"

namespace sankhya {
namespace {

Options quiet() {
  Options o;
  o.set_bool("log_to_console", false);
  return o;
}

/// The search's profiler counters, from a solve with profile_out set.
struct Profiled {
  Solution solution;
  nlohmann::json counters;
};

Profiled solve_profiled(const Model& model, Options options) {
  testing::TempFile file("", ".json");
  options.set_string("profile", "basic");
  options.set_string("profile_out", file.path());
  Profiled out;
  out.solution = solve(model, options);
  std::ifstream in(file.path());
  std::stringstream text;
  text << in.rdbuf();
  EXPECT_FALSE(text.str().empty()) << "profile_out was not written";
  if (!text.str().empty()) out.counters = nlohmann::json::parse(text.str())["counters"];
  return out;
}

std::int64_t counter(const Profiled& p, const char* name) {
  if (!p.counters.contains(name)) {
    ADD_FAILURE() << "no counter \"" << name << "\" in " << p.counters.dump(2);
    return -1;
  }
  return p.counters[name].get<std::int64_t>();
}

/// Presolve and root cuts off, most of all for the strong-branching models: a Chvatal-Gomory
/// cut or a presolve reduction would find what the probe is meant to find, and the probe would
/// have nothing left to prove.
Options bare() {
  Options o = quiet();
  o.set_bool("presolve", false);
  o.set_bool("enable_root_cuts", false);
  // Objective integrality (#221) would round a bound of 1.5 up to 2 and close these trees at
  // the root without any branching, fix or no fix.
  o.set_bool("mip_objective_integrality", false);
  o.set_string("mip_node_selection", "best-bound");
  return o;
}

void finish(Model& m) {
  m.matrix.finalize();
  m.hessian.reset(m.num_cols(), m.num_cols());
  m.hessian.finalize();
  EXPECT_EQ(m.validate(), "");
}

/// The a-block: integer a in [0, 5], continuous b in [-10, 10], and
///   a + b >= 2.5,   a - b >= 0.5,
/// so a >= 1.5 by adding the rows - which no single row shows, so propagation cannot see
/// it. Minimising a puts the relaxation at a = 1.5, and the down probe a <= 1 is
/// LP-infeasible: the side strong-branch fixing closes.
///
/// With `knapsack` the c-block is added: three integer c in [0, 1] and a continuous s >= 0
/// with 2 c1 + 2 c2 + 2 c3 - s <= 3, maximising the c at a cost on s. Its relaxation is
/// fractional wherever it is branched, and s absorbs any probe, so no c probe is ever
/// infeasible: every fix the search makes is a fix of a.
Model closed_side(bool knapsack) {
  Model m;
  const Index cols = knapsack ? 6 : 2;
  const Index rows = knapsack ? 3 : 2;
  m.col_cost.assign(static_cast<std::size_t>(cols), 0.0);
  m.col_lower.assign(static_cast<std::size_t>(cols), 0.0);
  m.col_upper.assign(static_cast<std::size_t>(cols), 1.0);
  m.col_type.assign(static_cast<std::size_t>(cols), VarType::kInteger);
  m.col_cost[0] = 1.0;  // a
  m.col_upper[0] = 5.0;
  m.col_type[1] = VarType::kContinuous;  // b
  m.col_lower[1] = -10.0;
  m.col_upper[1] = 10.0;
  m.matrix.reset(rows, cols);
  m.matrix.add_entry(0, 0, 1.0);
  m.matrix.add_entry(0, 1, 1.0);
  m.matrix.add_entry(1, 0, 1.0);
  m.matrix.add_entry(1, 1, -1.0);
  m.row_lower = {2.5, 0.5};
  m.row_upper = {kInfinity, kInfinity};
  if (knapsack) {
    for (Index j = 2; j <= 4; ++j) {
      m.col_cost[static_cast<std::size_t>(j)] = -10.0;
      m.matrix.add_entry(2, j, 2.0);
    }
    m.col_cost[5] = 7.0;  // s: each unit beyond the cap costs 14 and buys 10
    m.col_type[5] = VarType::kContinuous;
    m.col_upper[5] = 100.0;
    m.matrix.add_entry(2, 5, -1.0);
    m.row_lower.push_back(-kInfinity);
    m.row_upper.push_back(3.0);
  }
  finish(m);
  return m;
}

// =========================================================================================

TEST(BranchingFixpoint, EveryNewOptionIsOffByDefault) {
  // The project's rule for a new search behaviour: off until an A/B on main.
  const Options defaults;
  EXPECT_FALSE(defaults.get_bool("mip_strong_branch_fix"));
  EXPECT_FALSE(defaults.get_bool("mip_incremental_propagation"));
  EXPECT_FALSE(defaults.get_bool("mip_heap_open_list"));
}

TEST(BranchingFixpoint, AnInfeasibleProbeFixesTheColumnAndTheReSolveClosesTheNode) {
  const Model model = closed_side(false);
  Options off = bare();
  off.set_bool("mip_strong_branch_fix", false);
  Options on = bare();
  on.set_bool("mip_strong_branch_fix", true);

  const Solution plain = solve(model, off);
  const Profiled fixed = solve_profiled(model, on);

  ASSERT_EQ(plain.status, SolveStatus::kOptimal);
  ASSERT_EQ(fixed.solution.status, SolveStatus::kOptimal);
  EXPECT_NEAR(plain.objective, 2.0, 1e-9);
  EXPECT_NEAR(fixed.solution.objective, 2.0, 1e-9);
  // Without the fix the root branches on a and spends a child on the closed side a <= 1;
  // with it, a >= 2 is fixed, the re-solve is integral and the root is the whole tree.
  EXPECT_EQ(counter(fixed, "strong-branch columns fixed"), 1);
  EXPECT_EQ(fixed.solution.nodes, 1);
  EXPECT_GT(plain.nodes, fixed.solution.nodes);
}

TEST(BranchingFixpoint, TheSubtreeInheritsTheFixes) {
  // The root fixes a >= 2 and then has to branch on the c-block. Were the fix held only in
  // the node's working bounds, leave() would undo it, every child's LP would put a back at
  // 1.5, and the child would probe and fix it all over again - once per node. Hung under the
  // children as a link, it is made exactly once.
  const Model model = closed_side(true);
  Options off = bare();
  Options on = bare();
  on.set_bool("mip_strong_branch_fix", true);

  const Solution plain = solve(model, off);
  const Profiled fixed = solve_profiled(model, on);

  ASSERT_EQ(plain.status, SolveStatus::kOptimal);
  ASSERT_EQ(fixed.solution.status, SolveStatus::kOptimal);
  EXPECT_NEAR(fixed.solution.objective, plain.objective, 1e-9);
  EXPECT_NEAR(fixed.solution.col_value[0], 2.0, 1e-9);
  ASSERT_GT(fixed.solution.nodes, 1) << "the test needs the c-block to branch";
  EXPECT_EQ(counter(fixed, "strong-branch columns fixed"), 1);
}

TEST(BranchingFixpoint, BothSidesInfeasibleFathomsTheNode) {
  // An integer column with no integer inside its LP range: a in [0, 5], b in [-10, 10],
  //   2.2 <= a + b <= 2.8,   0.2 <= a - b <= 0.8,
  // so 1.2 <= a <= 1.8 by adding the rows, which no single row shows. Both probes are
  // infeasible and the fix rule proves the root empty without creating a child.
  Model m;
  m.col_cost = {1.0, 0.0};
  m.col_lower = {0.0, -10.0};
  m.col_upper = {5.0, 10.0};
  m.col_type = {VarType::kInteger, VarType::kContinuous};
  m.matrix.reset(2, 2);
  m.matrix.add_entry(0, 0, 1.0);
  m.matrix.add_entry(0, 1, 1.0);
  m.matrix.add_entry(1, 0, 1.0);
  m.matrix.add_entry(1, 1, -1.0);
  m.row_lower = {2.2, 0.2};
  m.row_upper = {2.8, 0.8};
  finish(m);

  Options off = bare();
  Options on = bare();
  on.set_bool("mip_strong_branch_fix", true);
  const Solution plain = solve(m, off);
  const Profiled fixed = solve_profiled(m, on);
  EXPECT_EQ(plain.status, SolveStatus::kInfeasible);
  EXPECT_EQ(fixed.solution.status, SolveStatus::kInfeasible);
  EXPECT_EQ(counter(fixed, "strong-branch fix prunes"), 1);
  EXPECT_EQ(fixed.solution.nodes, 1);
  EXPECT_GT(plain.nodes, 1);
}

/// A chain x_0 .. x_{n-1}, integer in [0, 5], rows 3 x_i - 2 x_{i+1} <= 1, x_{n-1} fixed at
/// 0, maximising the sum. Each row, given x_{i+1} <= 0, implies x_i <= 1/3, which rounds to
/// 0; the fixpoint therefore fixes the whole chain at the root. The rows are in the order
/// that makes a sweep advance one link at a time, so three sweeps fix three links and leave
/// the relaxation fractional (x_i <= (1 + 2 x_{i+1}) / 3 admits positive values), and the
/// tree has to branch. The optimum is 0 either way.
Model reversed_chain(Index n) {
  Model m;
  m.sense = ObjSense::kMaximize;
  m.col_cost.assign(static_cast<std::size_t>(n), 1.0);
  m.col_lower.assign(static_cast<std::size_t>(n), 0.0);
  m.col_upper.assign(static_cast<std::size_t>(n), 5.0);
  m.col_upper[static_cast<std::size_t>(n - 1)] = 0.0;
  m.col_type.assign(static_cast<std::size_t>(n), VarType::kInteger);
  m.matrix.reset(n - 1, n);
  for (Index i = 0; i + 1 < n; ++i) {
    m.matrix.add_entry(i, i, 3.0);
    m.matrix.add_entry(i, i + 1, -2.0);
  }
  m.row_lower.assign(static_cast<std::size_t>(n - 1), -kInfinity);
  m.row_upper.assign(static_cast<std::size_t>(n - 1), 1.0);
  finish(m);
  return m;
}

TEST(BranchingFixpoint, IncrementalPropagationReachesTheFixpointThreeSweepsMiss) {
  const Model model = reversed_chain(12);
  Options sweeps = bare();
  sweeps.set_string("mip_branching", "most-fractional");
  Options fixpoint = sweeps;
  fixpoint.set_bool("mip_incremental_propagation", true);

  const Solution swept = solve(model, sweeps);
  const Profiled reached = solve_profiled(model, fixpoint);

  ASSERT_EQ(swept.status, SolveStatus::kOptimal);
  ASSERT_EQ(reached.solution.status, SolveStatus::kOptimal);
  EXPECT_NEAR(swept.objective, 0.0, 1e-9);
  EXPECT_NEAR(reached.solution.objective, 0.0, 1e-9);
  EXPECT_GT(counter(reached, "propagation rows processed"), 0);
  EXPECT_EQ(reached.solution.nodes, 1) << "the fixpoint fixes the chain at the root";
  EXPECT_GT(swept.nodes, 1) << "three sweeps must leave the chain to the tree";
}

/// A strongly correlated multi-row knapsack (test_node_selection's family): hundreds of
/// nodes, many ties, several fractional columns per node.
Model correlated_knapsack(int columns, int rows) {
  Model m;
  const auto n = static_cast<Index>(columns);
  const auto r = static_cast<Index>(rows);
  m.sense = ObjSense::kMaximize;
  m.col_cost.resize(static_cast<std::size_t>(n));
  m.col_lower.assign(static_cast<std::size_t>(n), 0.0);
  m.col_upper.assign(static_cast<std::size_t>(n), 1.0);
  m.col_type.assign(static_cast<std::size_t>(n), VarType::kInteger);
  m.matrix.reset(r, n);
  std::vector<double> total(static_cast<std::size_t>(r), 0.0);
  for (Index j = 0; j < n; ++j) {
    for (Index i = 0; i < r; ++i) {
      const double weight = 20.0 + static_cast<double>((j * 37 + i * 29) % 51);
      if (i == 0) m.col_cost[static_cast<std::size_t>(j)] = weight + 10.0;
      m.matrix.add_entry(i, j, weight);
      total[static_cast<std::size_t>(i)] += weight;
    }
  }
  m.row_lower.assign(static_cast<std::size_t>(r), -kInfinity);
  m.row_upper.resize(static_cast<std::size_t>(r));
  for (Index i = 0; i < r; ++i) {
    m.row_upper[static_cast<std::size_t>(i)] =
        std::floor(total[static_cast<std::size_t>(i)] / 2.0);
  }
  finish(m);
  return m;
}

void expect_same_tree_with_heap(const char* policy) {
  const Model model = correlated_knapsack(18, 3);
  Options scan = quiet();
  scan.set_string("mip_node_selection", policy);
  scan.set_bool("mip_heap_open_list", false);
  Options heap = scan;
  heap.set_bool("mip_heap_open_list", true);

  const Solution linear = solve(model, scan);
  const Profiled heaped = solve_profiled(model, heap);

  ASSERT_EQ(linear.status, SolveStatus::kOptimal);
  ASSERT_EQ(heaped.solution.status, SolveStatus::kOptimal);
  ASSERT_GT(linear.nodes, 10) << "the test needs a real tree";
  // The heap's order is the scan's key with the scan's tie-break, so the search is the same
  // one, node for node and pivot for pivot.
  EXPECT_EQ(heaped.solution.nodes, linear.nodes);
  EXPECT_EQ(heaped.solution.iterations, linear.iterations);
  EXPECT_EQ(heaped.solution.objective, linear.objective);
  EXPECT_EQ(heaped.solution.col_value, linear.col_value);
  // Every node explored came off the heap; so did the ones pruned on their stored bound the
  // moment they were taken, which the node count does not include.
  EXPECT_GE(counter(heaped, "open-list heap selections"), heaped.solution.nodes)
      << "the nodes the search explored must have come off the heap";
}

TEST(BranchingFixpoint, TheHeapOpenListIsTheBestBoundScanNodeForNode) {
  expect_same_tree_with_heap("best-bound");
}

TEST(BranchingFixpoint, TheHeapOpenListIsTheBestEstimateScanNodeForNode) {
  expect_same_tree_with_heap("best-estimate");
}

TEST(BranchingFixpoint, EveryCombinationReachesTheSameOptimum) {
  const Model model = correlated_knapsack(15, 2);
  const Solution reference = solve(model, quiet());
  ASSERT_EQ(reference.status, SolveStatus::kOptimal);
  for (const bool fix : {true, false}) {
    for (const bool incremental : {true, false}) {
      for (const bool heap : {true, false}) {
        Options o = quiet();
        o.set_string("mip_node_selection", "best-bound");
        o.set_bool("mip_strong_branch_fix", fix);
        o.set_bool("mip_incremental_propagation", incremental);
        o.set_bool("mip_heap_open_list", heap);
        const Solution s = solve(model, o);
        EXPECT_EQ(s.status, SolveStatus::kOptimal)
            << "fix=" << fix << " incremental=" << incremental << " heap=" << heap;
        EXPECT_NEAR(s.objective, reference.objective, 1e-6)
            << "fix=" << fix << " incremental=" << incremental << " heap=" << heap;
      }
    }
  }
}

}  // namespace
}  // namespace sankhya
