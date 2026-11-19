// SPDX-License-Identifier: Apache-2.0
// SANKHYA - c-MIR aggregation by variable-bound distance (`mir_variable_bound_aggregation`).
//
// Marchand and Wolsey (2001, sec. 2) walk the aggregation by each continuous column's
// distance to its closest simple OR variable bound. On a fixed-charge network the LP puts
// most flows exactly on x = u y, inside [0, u] but on their variable bound: those must be
// left for c-MIR to substitute, and only the flows strictly inside every bound eliminated.
// The tests: the choice itself, by hand; and every cut the separator returns on random
// fixed-charge transshipment networks, with the option on, at the LP point and at random
// points below the switch bounds, is checked against every switch assignment by maximising
// the cut over the flows the assignment allows.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "mip/mir_cmir.hpp"
#include "mip/mir_cuts.hpp"
#include "mip/row_aggregation.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

using mip::Cut;
constexpr double kInf = std::numeric_limits<double>::infinity();

struct Arc {
  int from;
  int to;
  double capacity;
};

/// Fixed-charge transshipment: columns x_a (flow, [0, u_a]) then y_a (binary switch); rows the
/// variable upper bounds x_a - u_a y_a <= 0, then one balance row per node, inflow - outflow
/// = demand (the source's demand is minus the total). Costs: c on flows, f on switches.
Model network(int nodes, const std::vector<Arc>& arcs, const std::vector<double>& demand,
              const std::vector<double>& flow_cost, const std::vector<double>& fixed_cost) {
  const auto k = static_cast<Index>(arcs.size());
  Model m;
  m.resize_columns(2 * k);
  m.resize_rows(k + nodes);
  m.matrix.reset(k + nodes, 2 * k);
  for (Index a = 0; a < k; ++a) {
    const auto u = static_cast<std::size_t>(a);
    m.col_upper[u] = arcs[u].capacity;
    m.col_cost[u] = flow_cost[u];
    m.col_type[static_cast<std::size_t>(k + a)] = VarType::kInteger;
    m.col_upper[static_cast<std::size_t>(k + a)] = 1.0;
    m.col_cost[static_cast<std::size_t>(k + a)] = fixed_cost[u];
    m.matrix.add_entry(a, a, 1.0);
    m.matrix.add_entry(a, k + a, -arcs[u].capacity);
    m.row_lower[u] = -kInf;
    m.row_upper[u] = 0.0;
    m.matrix.add_entry(k + arcs[u].to, a, 1.0);
    m.matrix.add_entry(k + arcs[u].from, a, -1.0);
  }
  for (int v = 0; v < nodes; ++v) {
    const auto r = static_cast<std::size_t>(k + v);
    m.row_lower[r] = demand[static_cast<std::size_t>(v)];
    m.row_upper[r] = demand[static_cast<std::size_t>(v)];
  }
  m.matrix.finalize();
  m.hessian.reset(2 * k, 2 * k);
  m.hessian.finalize();
  EXPECT_EQ(m.validate(), "");
  return m;
}

Options quiet_lp() {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", false);
  options.set_string("algorithm", "dual-simplex");
  return options;
}

/// The largest cut left side minus its right side over every switch assignment's flows.
double worst_violation(const Model& model, Index arcs, const Cut& cut) {
  double worst = -kInf;
  for (std::uint32_t mask = 0; mask < (1u << arcs); ++mask) {
    Model fixed = model;
    for (Index j = 0; j < 2 * arcs; ++j) {
      const auto u = static_cast<std::size_t>(j);
      fixed.col_type[u] = VarType::kContinuous;
      fixed.col_cost[u] = -cut.coeff[u];
    }
    for (Index a = 0; a < arcs; ++a) {
      const double on = ((mask >> a) & 1u) != 0 ? 1.0 : 0.0;
      fixed.col_lower[static_cast<std::size_t>(arcs + a)] = on;
      fixed.col_upper[static_cast<std::size_t>(arcs + a)] = on;
    }
    const Solution s = solve(fixed, quiet_lp());
    if (s.status == SolveStatus::kOptimal) worst = std::max(worst, -s.objective - cut.rhs);
  }
  return worst;
}

TEST(CmirAggregation, AFlowOnItsSwitchBoundIsNotEliminated) {
  // One node fed by two arcs, demand 3: x_0 + x_1 = 3, x_0 <= 4 y_0, x_1 <= 4 y_1. At the point
  // x = (2, 1), y = (0.5, 0.75): x_0 sits on 4 y_0 = 2, x_1 is 2 below 4 y_1 = 3. By simple
  // bounds x_0 is the deeper (2 from either end of [0, 4], against 1 for x_1); by variable
  // bounds it is at a bound, and x_1 is the one to eliminate.
  const std::vector<Arc> arcs{{0, 1, 4.0}, {0, 1, 4.0}};
  const Model m = network(2, arcs, {-3.0, 3.0}, {1.0, 1.0}, {1.0, 1.0});
  Solution point;
  point.col_value = {2.0, 1.0, 0.5, 0.75};
  mip::BaseRow base;
  base.columns = {0, 1};
  base.values = {1.0, 1.0};
  base.rhs = 3.0;
  const mip::VariableBounds bounds = mip::find_variable_bounds(m);
  EXPECT_EQ(mip::column_to_eliminate(m, point, m.col_lower, m.col_upper, base), 0);
  EXPECT_EQ(
      mip::column_to_eliminate(m, point, m.col_lower, m.col_upper, base, nullptr, &bounds), 1);
  // Both on their switch bounds: nothing is inside, nothing is eliminated.
  point.col_value = {2.0, 1.0, 0.5, 0.25};
  EXPECT_EQ(
      mip::column_to_eliminate(m, point, m.col_lower, m.col_upper, base, nullptr, &bounds), -1);
}

TEST(CmirAggregation, NoCutRemovesAFeasibleFlowOfRandomNetworks) {
  std::mt19937_64 rng(2001);
  std::uniform_int_distribution<int> capacity(3, 9);
  std::uniform_int_distribution<int> cost(1, 5);
  std::uniform_int_distribution<int> fixed(5, 25);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  int cuts_checked = 0;
  int aggregated = 0;
  int networks = 0;
  for (int trial = 0; trial < 40; ++trial) {
    // A source 0 and 3 or 4 other nodes; a path 0 -> 1 -> ... so every node is reachable,
    // plus random arcs, at most 8 in all.
    const int nodes = 4 + trial % 2;
    std::vector<Arc> arcs;
    for (int v = 1; v < nodes; ++v) arcs.push_back({v - 1, v, double(capacity(rng))});
    while (arcs.size() < 8) {
      const int from = static_cast<int>(unit(rng) * nodes);
      const int to = static_cast<int>(unit(rng) * nodes);
      if (from == to || to == 0) continue;
      arcs.push_back({from, to, double(capacity(rng))});
    }
    std::vector<double> demand(static_cast<std::size_t>(nodes), 0.0);
    double total = 0.0;
    for (int v = 1; v < nodes; ++v) {
      demand[static_cast<std::size_t>(v)] = (trial + v) % 3 == 0 ? 0.0 : 1.0 + v % 2;
      total += demand[static_cast<std::size_t>(v)];
    }
    demand[0] = -total;
    std::vector<double> c;
    std::vector<double> f;
    for (std::size_t a = 0; a < arcs.size(); ++a) {
      c.push_back(cost(rng));
      f.push_back(fixed(rng));
    }
    const Model m = network(nodes, arcs, demand, c, f);
    Model relaxed = m;
    relaxed.col_type.assign(relaxed.col_type.size(), VarType::kContinuous);
    const Solution lp = solve(relaxed, quiet_lp());
    if (lp.status != SolveStatus::kOptimal) continue;
    ++networks;
    // The LP point puts every flow on its switch bound, x = u y, so nothing is inside every
    // bound and the walk eliminates nothing there; random points below the switch bounds,
    // where the separator must still return only valid cuts, exercise the walk itself.
    std::vector<Solution> points{lp};
    const auto k = arcs.size();
    for (int probe = 0; probe < 2; ++probe) {
      Solution p;
      p.col_value.assign(2 * k, 0.0);
      for (std::size_t a = 0; a < k; ++a) {
        p.col_value[k + a] = unit(rng);
        p.col_value[a] = unit(rng) * arcs[a].capacity * p.col_value[k + a];
      }
      points.push_back(p);
    }
    mip::MirOptions options;
    options.cmir = true;
    options.variable_bound_aggregation = true;
    for (const Solution& point : points) {
      mip::MirStats stats;
      const std::vector<Cut> cuts =
          mip::generate_mir_cuts(m, point, m.col_lower, m.col_upper, &stats, options);
      aggregated += stats.aggregated_cuts;
      for (const Cut& cut : cuts) {
        const double worst = worst_violation(m, static_cast<Index>(k), cut);
        ASSERT_LE(worst, 1e-7 * std::max(1.0, std::fabs(cut.rhs)))
            << "trial " << trial << ": a switch assignment's flow violates the cut by "
            << worst;
        ++cuts_checked;
      }
    }
  }
  EXPECT_GT(networks, 20);
  EXPECT_GT(cuts_checked, 20) << "the sweep should exercise the separator";
  EXPECT_GT(aggregated, 0) << "no cut came from an aggregate, so the walk was not tested";
  std::printf(
      "[  INFO    ] c-MIR with variable-bound aggregation: %d cuts (%d aggregated) on %d "
      "networks, none removes a feasible flow\n",
      cuts_checked, aggregated, networks);
}

}  // namespace
}  // namespace sankhya
