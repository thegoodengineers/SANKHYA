// SPDX-License-Identifier: Apache-2.0
// SANKHYA - upper bounds for the spatial branch and bound: feasibility against the ORIGINAL
// model, and the alternating LP that produces candidate points (#514).
//
// THE ALTERNATING LP. Fix one factor of every product at a value and the model is an LP in
// the rest. For a pooling model in the P-formulation the factors split into qualities and
// flows: fix the qualities and the flows are an LP, fix the flows and the qualities are.
// Alternating the two is the recursion (successive linear programming) Haverly (1978)
// studied, and the classic way to reach a local optimum of a bilinear program; starting it
// from the McCormick relaxation's point is what makes it a useful upper-bound heuristic
// inside a branch and bound (Tawarmalani and Sahinidis 2002, ch. 7; Audet, Brimberg, Hansen,
// Le Digabel and Mladenovic, "Pooling problem: alternate formulations and solution
// methods", Management Science 50, 2004, for the alternating scheme on pooling).
//
// Which factor to fix: when the graph whose edges are the products is bipartite, its two
// colour classes are the two sides and every product has exactly one factor in each. When
// it is not (a square, or an odd cycle of products), a vertex cover is fixed instead and
// the search is a single LP. Either way every point it returns is re-evaluated against the
// original rows, products and all, and is kept only if it is feasible there; the LP's own
// status is not taken as evidence.

#include <algorithm>
#include <cmath>
#include <deque>
#include <map>
#include <utility>
#include <vector>

#include "global/global_internal.hpp"
#include "sankhya/certificate.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/tolerances.hpp"
#include "simplex/primal_simplex.hpp"

namespace sankhya::global {
namespace {

/// The sets of columns to fix, in the order they are tried. Two sets when the product graph
/// is bipartite, one (a vertex cover) when it is not.
std::vector<std::vector<char>> fixing_sides(const Problem& problem) {
  const auto n = static_cast<std::size_t>(problem.n);
  std::vector<std::vector<Index>> adjacent(n);
  bool bipartite = true;
  for (const Product& product : problem.products) {
    if (product.square()) {
      bipartite = false;
      continue;
    }
    adjacent[static_cast<std::size_t>(product.a)].push_back(product.b);
    adjacent[static_cast<std::size_t>(product.b)].push_back(product.a);
  }
  std::vector<int> colour(n, -1);
  for (std::size_t s = 0; s < n && bipartite; ++s) {
    if (problem.in_product[s] == 0 || colour[s] >= 0) continue;
    colour[s] = 0;
    std::deque<std::size_t> queue{s};
    while (!queue.empty() && bipartite) {
      const std::size_t v = queue.front();
      queue.pop_front();
      for (const Index w : adjacent[v]) {
        const auto uw = static_cast<std::size_t>(w);
        if (colour[uw] < 0) {
          colour[uw] = 1 - colour[v];
          queue.push_back(uw);
        } else if (colour[uw] == colour[v]) {
          bipartite = false;
        }
      }
    }
  }
  if (bipartite) {
    std::vector<std::vector<char>> sides(2, std::vector<char>(n, 0));
    for (std::size_t j = 0; j < n; ++j) {
      if (colour[j] >= 0) sides[static_cast<std::size_t>(colour[j])][j] = 1;
    }
    return sides;
  }
  // A greedy vertex cover: every squared column, then for each product left uncovered the
  // factor with more products.
  std::vector<char> cover(n, 0);
  std::vector<Index> degree(n, 0);
  for (const Product& product : problem.products) {
    if (product.square()) {
      cover[static_cast<std::size_t>(product.a)] = 1;
    } else {
      ++degree[static_cast<std::size_t>(product.a)];
      ++degree[static_cast<std::size_t>(product.b)];
    }
  }
  for (const Product& product : problem.products) {
    const auto a = static_cast<std::size_t>(product.a);
    const auto b = static_cast<std::size_t>(product.b);
    if (cover[a] != 0 || cover[b] != 0) continue;
    cover[degree[a] >= degree[b] ? a : b] = 1;
  }
  return {cover};
}

/// The LP left when the columns in `fixed` are held at `point`, or an empty Model when some
/// product has neither factor fixed.
Model linearized(const Problem& problem, const std::vector<char>& fixed,
                 const std::vector<double>& point) {
  const auto n = static_cast<std::size_t>(problem.n);
  Model lp;
  lp.name = "alternating-lp";
  lp.resize_columns(problem.n);
  lp.resize_rows(problem.m);
  lp.objective_offset = problem.offset;
  std::vector<double> value(n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    lp.col_cost[j] = problem.cost[j];
    lp.col_lower[j] = problem.col_lower[j];
    lp.col_upper[j] = problem.col_upper[j];
    if (fixed[j] != 0) {
      value[j] = std::clamp(point[j], problem.col_lower[j], problem.col_upper[j]);
      lp.col_lower[j] = value[j];
      lp.col_upper[j] = value[j];
    }
  }
  // (column -> coefficient) for one row, and the constant the fixed products contribute.
  bool ok = true;
  const auto fold = [&](const std::vector<std::pair<Index, double>>& products,
                        std::map<Index, double>* coefficients) {
    double constant = 0.0;
    for (const auto& [index, c] : products) {
      const Product& product = problem.products[static_cast<std::size_t>(index)];
      const auto a = static_cast<std::size_t>(product.a);
      const auto b = static_cast<std::size_t>(product.b);
      if (fixed[a] != 0 && fixed[b] != 0) {
        constant += c * value[a] * value[b];
      } else if (fixed[a] != 0) {
        (*coefficients)[product.b] += c * value[a];
      } else if (fixed[b] != 0) {
        (*coefficients)[product.a] += c * value[b];
      } else {
        ok = false;
      }
    }
    return constant;
  };

  std::map<Index, double> objective;
  lp.objective_offset += fold(problem.objective_products, &objective);
  for (const auto& [column, c] : objective) lp.col_cost[static_cast<std::size_t>(column)] += c;

  std::vector<std::map<Index, double>> rows(static_cast<std::size_t>(problem.m));
  for (Index i = 0; i < problem.m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    for (const auto& [column, c] : problem.row_linear[u]) rows[u][column] += c;
    const double constant = fold(problem.row_products[u], &rows[u]);
    lp.row_lower[u] = problem.row_lower[u] - constant;
    lp.row_upper[u] = problem.row_upper[u] - constant;
  }
  if (!ok) return {};
  lp.matrix.reset(problem.m, problem.n);
  for (Index i = 0; i < problem.m; ++i) {
    for (const auto& [column, c] : rows[static_cast<std::size_t>(i)]) {
      if (c != 0.0) lp.matrix.add_entry(i, column, c);
    }
  }
  lp.matrix.finalize(0.0);
  return lp;
}

}  // namespace

double original_violation(const Problem& problem, const std::vector<double>& x,
                          double* absolute) {
  double worst = 0.0;
  double worst_absolute = 0.0;
  for (Index j = 0; j < problem.n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double v = std::max({problem.col_lower[u] - x[u], x[u] - problem.col_upper[u], 0.0});
    worst_absolute = std::max(worst_absolute, v);
    worst = std::max(worst, v / std::max(1.0, std::fabs(x[u])));
  }
  for (Index i = 0; i < problem.m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    double activity = 0.0;
    double scale = 1.0;
    for (const auto& [column, c] : problem.row_linear[u]) {
      const double term = c * x[static_cast<std::size_t>(column)];
      activity += term;
      scale = std::max(scale, std::fabs(term));
    }
    for (const auto& [index, c] : problem.row_products[u]) {
      const Product& product = problem.products[static_cast<std::size_t>(index)];
      const double term =
          c * x[static_cast<std::size_t>(product.a)] * x[static_cast<std::size_t>(product.b)];
      activity += term;
      scale = std::max(scale, std::fabs(term));
    }
    const double v =
        std::max({problem.row_lower[u] - activity, activity - problem.row_upper[u], 0.0});
    worst_absolute = std::max(worst_absolute, v);
    worst = std::max(worst, v / scale);
  }
  if (absolute != nullptr) *absolute = worst_absolute;
  return worst;
}

Solution solve_lp(const Model& lp, const Options& lp_options,
                  const std::vector<BasisStatus>* col_status,
                  const std::vector<BasisStatus>* row_status) {
  Logger quiet(nullptr);
  WarmStart warm;
  if (col_status != nullptr && row_status != nullptr &&
      static_cast<Index>(col_status->size()) == lp.num_cols() &&
      static_cast<Index>(row_status->size()) == lp.num_rows()) {
    warm.col_status = *col_status;
    warm.row_status = *row_status;
  }
  Solution solution =
      solve_dual_simplex(lp, lp_options, quiet, nullptr, warm.empty() ? nullptr : &warm);
  if (solution.status == SolveStatus::kInfeasible) {
    const std::vector<double> raw = solution.farkas_dual;
    verify_and_keep_certificate(&solution, lp, quiet);
    // A multiplier of rounding size on the side of a row that has no bound spoils an
    // otherwise sound proof. Any multipliers may be tried, so those entries are set to zero
    // and the proof checked again, both signs, by the same checker.
    for (const double sign : {1.0, -1.0}) {
      if (!solution.farkas_dual.empty() || raw.empty()) break;
      std::vector<double> y = raw;
      for (std::size_t i = 0; i < y.size(); ++i) {
        y[i] *= sign;
        if ((y[i] > 0.0 && !std::isfinite(lp.row_lower[i])) ||
            (y[i] < 0.0 && !std::isfinite(lp.row_upper[i]))) {
          y[i] = 0.0;
        }
      }
      if (farkas_proves_infeasible(lp, y)) solution.farkas_dual = std::move(y);
    }
    if (solution.farkas_dual.empty()) {
      solution.status = SolveStatus::kNumericalError;
      solution.message = "the relaxation was reported infeasible without a Farkas proof";
    }
  }
  return solution;
}

Options inner_lp_options(const Options& options, double seconds_left) {
  Options inner;
  inner.set_bool("log_to_console", false);
  inner.set_string("algorithm", "dual-simplex");
  inner.set_double("primal_feasibility_tolerance",
                   options.get_double("primal_feasibility_tolerance"));
  inner.set_double("dual_feasibility_tolerance",
                   options.get_double("dual_feasibility_tolerance"));
  if (std::isfinite(seconds_left)) inner.set_double("time_limit", std::max(0.0, seconds_left));
  return inner;
}

namespace {

/// One alternation from `start`, fixing sides[first] first.
void alternate(const Problem& problem, const std::vector<std::vector<char>>& sides,
               std::size_t first, std::vector<double> point, const Options& lp_options,
               double feasibility_tolerance, LocalSearchResult* result) {
  double previous = kInfinity;
  int failures = 0;
  for (int round = 0; round < tol::kGlobalLocalSearchRounds; ++round) {
    const std::vector<char>& fixed =
        sides[(first + static_cast<std::size_t>(round)) % sides.size()];
    const Model lp = linearized(problem, fixed, point);
    if (lp.num_cols() == 0) return;
    const Solution solved = solve_lp(lp, lp_options);
    ++result->lp_solves;
    result->lp_iterations += solved.iterations;
    if (solved.status != SolveStatus::kOptimal ||
        static_cast<Index>(solved.col_value.size()) != problem.n) {
      // This side's values leave no feasible LP; the other side may still move.
      if (++failures >= 2 || sides.size() == 1) return;
      continue;
    }
    failures = 0;
    point = solved.col_value;
    if (original_violation(problem, point) > feasibility_tolerance) continue;
    const double objective = min_objective(problem, point);
    if (objective < result->objective) {
      result->improved = true;
      result->objective = objective;
      result->x = point;
    }
    // Stop when a full cycle through the sides has not moved the objective.
    if (round + 1 >= static_cast<int>(sides.size()) &&
        !(objective <
          previous - tol::kGlobalLocalSearchProgress * std::max(1.0, std::fabs(objective)))) {
      return;
    }
    previous = std::min(previous, objective);
    if (sides.size() == 1) return;
  }
}

}  // namespace

LocalSearchResult alternating_search(const Problem& problem, const std::vector<double>& start,
                                     double incumbent, const Options& lp_options,
                                     double feasibility_tolerance) {
  LocalSearchResult result;
  result.objective = incumbent;
  const std::vector<std::vector<char>> sides = fixing_sides(problem);
  alternate(problem, sides, 0, start, lp_options, feasibility_tolerance, &result);
  return result;
}

LocalSearchResult relaxation_search(const Problem& problem,
                                    const std::vector<double>& relaxation, double incumbent,
                                    const Options& lp_options, double feasibility_tolerance) {
  LocalSearchResult result;
  result.objective = incumbent;
  const auto n = static_cast<std::size_t>(problem.n);
  const std::vector<double> raw(relaxation.begin(), relaxation.begin() + problem.n);
  // The value each factor would need for the relaxation's products to be exact: the
  // least-squares a in w_p = a * x_b over every product p = a * b (a pool's quality is then
  // its inflow-weighted source quality, the consistent one).
  std::vector<double> numerator(n, 0.0);
  std::vector<double> denominator(n, 0.0);
  for (std::size_t p = 0; p < problem.products.size(); ++p) {
    const Product& product = problem.products[p];
    if (product.square()) continue;
    const double w = relaxation[n + p];
    const auto a = static_cast<std::size_t>(product.a);
    const auto b = static_cast<std::size_t>(product.b);
    numerator[a] += w * raw[b];
    denominator[a] += raw[b] * raw[b];
    numerator[b] += w * raw[a];
    denominator[b] += raw[a] * raw[a];
  }
  const std::vector<std::vector<char>> sides = fixing_sides(problem);
  for (std::size_t first = 0; first < sides.size(); ++first) {
    std::vector<double> implied = raw;
    for (std::size_t j = 0; j < n; ++j) {
      if (sides[first][j] == 0 || denominator[j] <= tol::kGlobalImpliedValueFloor) continue;
      implied[j] =
          std::clamp(numerator[j] / denominator[j], problem.col_lower[j], problem.col_upper[j]);
    }
    alternate(problem, sides, first, implied, lp_options, feasibility_tolerance, &result);
  }
  alternate(problem, sides, 0, raw, lp_options, feasibility_tolerance, &result);
  return result;
}

}  // namespace sankhya::global
