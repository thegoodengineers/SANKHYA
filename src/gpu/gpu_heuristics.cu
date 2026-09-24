// SPDX-License-Identifier: Apache-2.0
// SANKHYA - GPU MIP heuristics (#509): feasibility pump and fix-and-propagate.
//
// Feasibility pump (Fischetti, Glover & Lodi, Math. Programming 104, 2005; GPU variant:
// Mexi et al., arXiv:2307.03466; Corduk et al., arXiv:2510.20499):
//   Outer loop:
//     1. Build a modified model where the objective is min sum_j |x_j - x^INT_j| over the
//        integer columns, linearised per column into two non-negative auxiliary variables
//        d_j^+ and d_j^- with x_j - x^INT_j = d_j^+ - d_j^- and cost 1 on each.  The
//        constraint matrix is unchanged; only the objective and column bounds differ from the
//        original.  GPU PDHG solves the L1 projection LP using pdhg_gpu.cu.
//     2. x^INT = round(x^LP) to the nearest integer for each integer column.
//     3. If x^INT is row-feasible in the original model -> offer as incumbent.
//     4. If ||x^LP - x^INT||_1 has not improved in 10 consecutive rounds -> perturb
//        x^INT by flipping the most-fractional variable.
//     5. Stop after gpu_pump_max_iter rounds (default 50) or when interrupted.
//
// Fix-and-propagate (Corduk et al., arXiv:2510.20499):
//   1. Start from the LP relaxation point (caller supplies it via options or it is solved).
//   2. For each integer column j sorted by ascending fractionality
//      (most-integer first): fix lb_j = ub_j = round(x^LP_j).
//   3. After each fix run GPU domain propagation (domain_prop.cu) to tighten the rest.
//   4. If propagation detects infeasibility, backtrack and fix to the other rounding.
//      Allow at most gpu_fix_backtrack (default 5) backtracks; give up if exceeded.
//   5. Solve the residual LP on CPU.  If the result is integer-feasible, return it.

#include "gpu_heuristics.hpp"
#include "device.hpp"
#include "domain_prop.hpp"
#include "pdhg_gpu.hpp"

#include "sankhya/logging.hpp"
#include "sankhya/tolerances.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace sankhya::gpu {
namespace {

/// Row-feasibility check for x on the original model (no LP solve).
bool is_row_feasible(const Model& model, const std::vector<double>& x, double tol) {
  const Index m = model.num_rows();
  for (Index i = 0; i < m; ++i) {
    double act = 0.0;
    for (Index k = model.matrix.row_start()[static_cast<std::size_t>(i)];
         k < model.matrix.row_start()[static_cast<std::size_t>(i) + 1]; ++k) {
      act += model.matrix.value()[static_cast<std::size_t>(k)] *
             x[static_cast<std::size_t>(model.matrix.column()[static_cast<std::size_t>(k)])];
    }
    if (act < model.row_lower[static_cast<std::size_t>(i)] - tol ||
        act > model.row_upper[static_cast<std::size_t>(i)] + tol) {
      return false;
    }
  }
  return true;
}

/// Build the L1-distance model for one pump iteration: minimise sum |x_j - target_j|
/// over integer columns j, subject to the original constraints. Linearised as two
/// auxiliary variables per integer column.
Model build_pump_model(const Model& original, const std::vector<Index>& int_cols,
                       const std::vector<double>& target) {
  Model pump = original;
  // Zero out the original objective; the pump objective is the L1 distance.
  std::fill(pump.col_cost.begin(), pump.col_cost.end(), 0.0);
  pump.sense = ObjSense::kMinimize;
  const Index n_orig = original.num_cols();
  const Index n_int = static_cast<Index>(int_cols.size());
  // Add two auxiliary variables d^+_j and d^-_j per integer column.
  // d^+_j, d^-_j >= 0, cost = 1 each.
  // Constraint: x_j - d^+_j + d^-_j = target_j  (one equality row per integer column).
  for (Index k = 0; k < n_int; ++k) {
    const Index j = int_cols[static_cast<std::size_t>(k)];
    // d^+_j
    pump.col_cost.push_back(1.0);
    pump.col_lower.push_back(0.0);
    pump.col_upper.push_back(kInfinity);
    pump.col_type.push_back(VarType::kContinuous);
    // d^-_j
    pump.col_cost.push_back(1.0);
    pump.col_lower.push_back(0.0);
    pump.col_upper.push_back(kInfinity);
    pump.col_type.push_back(VarType::kContinuous);
  }
  // Add equality rows x_j - d^+_j + d^-_j = target_j.
  const Index d_start = n_orig;
  pump.matrix.reset(pump.num_rows() + n_int, pump.num_cols());
  // Rebuild matrix with original entries first, then auxiliary rows.
  // Copy original matrix.
  for (Index i = 0; i < original.num_rows(); ++i) {
    for (Index p = original.matrix.row_start()[static_cast<std::size_t>(i)];
         p < original.matrix.row_start()[static_cast<std::size_t>(i) + 1]; ++p) {
      pump.matrix.add_entry(i, original.matrix.column()[static_cast<std::size_t>(p)],
                            original.matrix.value()[static_cast<std::size_t>(p)]);
    }
  }
  // Add auxiliary rows.
  for (Index k = 0; k < n_int; ++k) {
    const Index j = int_cols[static_cast<std::size_t>(k)];
    const Index row = original.num_rows() + k;
    pump.matrix.add_entry(row, j, 1.0);                     // x_j
    pump.matrix.add_entry(row, d_start + 2 * k, -1.0);     // -d^+_j
    pump.matrix.add_entry(row, d_start + 2 * k + 1, 1.0);  // +d^-_j
    pump.row_lower.push_back(target[static_cast<std::size_t>(k)]);
    pump.row_upper.push_back(target[static_cast<std::size_t>(k)]);
  }
  pump.matrix.finalize();
  pump.hessian.reset(pump.num_cols(), pump.num_cols());
  pump.hessian.finalize();
  return pump;
}

}  // namespace

std::optional<HeuristicSolution>
feasibility_pump(const Model& model, const Options& options) {
  if (!options.get_bool("gpu_pump") || !device_available(nullptr)) return std::nullopt;

  const int max_iter = options.get_int("gpu_pump_max_iter");

  // Collect integer columns.
  std::vector<Index> int_cols;
  for (Index j = 0; j < model.num_cols(); ++j) {
    if (model.col_type[static_cast<std::size_t>(j)] == VarType::kInteger) {
      int_cols.push_back(j);
    }
  }
  if (int_cols.empty()) return std::nullopt;

  // Solve the LP relaxation once to get an initial point.
  Options lp_opts;
  lp_opts.set_bool("log_to_console", false);
  lp_opts.set_bool("presolve", false);
  Logger silent = Logger::null();
  // Treat the model as a pure LP (set all columns continuous) for the relaxation.
  Model relaxed = model;
  for (auto& t : relaxed.col_type) t = VarType::kContinuous;
  const Solution init = solve_pdhg_gpu(relaxed, lp_opts, silent);
  if (init.status != SolveStatus::kOptimal && init.status != SolveStatus::kFeasible) {
    return std::nullopt;
  }

  std::vector<double> x_lp = init.col_value;
  // Build x^INT by rounding.
  std::vector<double> target(int_cols.size());
  for (std::size_t k = 0; k < int_cols.size(); ++k) {
    target[k] = std::round(x_lp[static_cast<std::size_t>(int_cols[k])]);
  }

  double best_dist = std::numeric_limits<double>::infinity();
  int stall = 0;

  for (int iter = 0; iter < max_iter; ++iter) {
    // Check feasibility of x^INT.
    std::vector<double> candidate = x_lp;
    for (std::size_t k = 0; k < int_cols.size(); ++k) {
      candidate[static_cast<std::size_t>(int_cols[k])] = target[k];
    }
    // Clamp to bounds.
    for (std::size_t j = 0; j < static_cast<std::size_t>(model.num_cols()); ++j) {
      candidate[j] = std::max(model.col_lower[j], std::min(model.col_upper[j], candidate[j]));
    }
    if (is_row_feasible(model, candidate, tol::kPrimalFeasibility)) {
      const double obj = std::inner_product(model.col_cost.begin(), model.col_cost.end(),
                                            candidate.begin(), 0.0) *
                         (model.sense == ObjSense::kMaximize ? -1.0 : 1.0);
      return HeuristicSolution{candidate, obj};
    }

    // Solve L1-projection LP on GPU.
    const Model pump_model = build_pump_model(model, int_cols, target);
    const Solution proj = solve_pdhg_gpu(pump_model, lp_opts, silent);
    if (proj.status != SolveStatus::kOptimal && proj.status != SolveStatus::kFeasible) break;

    // Extract the original columns from the projection solution.
    for (Index j = 0; j < model.num_cols(); ++j) {
      x_lp[static_cast<std::size_t>(j)] = proj.col_value[static_cast<std::size_t>(j)];
    }

    // Compute L1 distance.
    double dist = 0.0;
    for (std::size_t k = 0; k < int_cols.size(); ++k) {
      dist += std::fabs(x_lp[static_cast<std::size_t>(int_cols[k])] - target[k]);
    }

    // Stall detection: if distance is not improving, perturb by flipping the most-fractional.
    if (dist >= best_dist - 1e-4) {
      ++stall;
      if (stall >= 10) {
        // Find the most-fractional integer column and flip its rounding.
        std::size_t worst = 0;
        double worst_frac = -1.0;
        for (std::size_t k = 0; k < int_cols.size(); ++k) {
          const double v = x_lp[static_cast<std::size_t>(int_cols[k])];
          const double frac = std::fabs(v - std::round(v));
          if (frac > worst_frac) {
            worst_frac = frac;
            worst = k;
          }
        }
        target[worst] = (target[worst] == std::floor(x_lp[static_cast<std::size_t>(int_cols[worst])]))
                            ? std::ceil(x_lp[static_cast<std::size_t>(int_cols[worst])])
                            : std::floor(x_lp[static_cast<std::size_t>(int_cols[worst])]);
        stall = 0;
      }
    } else {
      best_dist = dist;
      stall = 0;
      // Re-round to new x^INT.
      for (std::size_t k = 0; k < int_cols.size(); ++k) {
        target[k] = std::round(x_lp[static_cast<std::size_t>(int_cols[k])]);
      }
    }
  }
  return std::nullopt;
}

std::optional<HeuristicSolution>
fix_and_propagate(const Model& model, const Options& options) {
  if (!options.get_bool("gpu_fix_and_prop") || !device_available(nullptr)) return std::nullopt;

  const int max_backtracks = options.get_int("gpu_fix_backtrack");

  // Collect integer columns sorted by fractionality (most-integer first).
  Options lp_opts;
  lp_opts.set_bool("log_to_console", false);
  lp_opts.set_bool("presolve", false);
  Logger silent = Logger::null();
  Model relaxed = model;
  for (auto& t : relaxed.col_type) t = VarType::kContinuous;
  const Solution init = solve_pdhg_gpu(relaxed, lp_opts, silent);
  if (init.status != SolveStatus::kOptimal && init.status != SolveStatus::kFeasible) {
    return std::nullopt;
  }

  const Index n = model.num_cols();
  std::vector<Index> int_order;
  for (Index j = 0; j < n; ++j) {
    if (model.col_type[static_cast<std::size_t>(j)] == VarType::kInteger) {
      int_order.push_back(j);
    }
  }
  // Sort by ascending fractionality (most-integer, i.e. smallest fractionality, first).
  std::sort(int_order.begin(), int_order.end(), [&](Index a, Index b) {
    const double fa = std::fabs(init.col_value[static_cast<std::size_t>(a)] -
                                std::round(init.col_value[static_cast<std::size_t>(a)]));
    const double fb = std::fabs(init.col_value[static_cast<std::size_t>(b)] -
                                std::round(init.col_value[static_cast<std::size_t>(b)]));
    return fa < fb;
  });

  std::vector<double> col_lb = model.col_lower;
  std::vector<double> col_ub = model.col_upper;
  int backtracks = 0;

  for (std::size_t k = 0; k < int_order.size(); ++k) {
    const Index j = int_order[k];
    const double v = init.col_value[static_cast<std::size_t>(j)];
    const double rounded = std::round(v);
    const double lb_save = col_lb[static_cast<std::size_t>(j)];
    const double ub_save = col_ub[static_cast<std::size_t>(j)];
    // Fix to rounded value.
    col_lb[static_cast<std::size_t>(j)] = rounded;
    col_ub[static_cast<std::size_t>(j)] = rounded;
    // GPU propagation.
    const PropResult prop = propagate_bounds(model, col_lb, col_ub, 50, tol::kIntegrality);
    if (prop.infeasible) {
      if (backtracks >= max_backtracks) return std::nullopt;
      ++backtracks;
      // Try the other rounding.
      const double alt = (rounded == std::floor(v)) ? std::ceil(v) : std::floor(v);
      col_lb[static_cast<std::size_t>(j)] = lb_save;
      col_ub[static_cast<std::size_t>(j)] = ub_save;
      col_lb[static_cast<std::size_t>(j)] = alt;
      col_ub[static_cast<std::size_t>(j)] = alt;
      const PropResult prop2 = propagate_bounds(model, col_lb, col_ub, 50, tol::kIntegrality);
      if (prop2.infeasible) return std::nullopt;
      if (prop2.ran) { col_lb = prop2.col_lb; col_ub = prop2.col_ub; }
    } else if (prop.ran) {
      col_lb = prop.col_lb;
      col_ub = prop.col_ub;
    }
  }

  // Solve residual LP (all integer columns fixed) on CPU via the GPU PDHG fallback.
  Model fixed = model;
  fixed.col_lower = col_lb;
  fixed.col_upper = col_ub;
  for (auto& t : fixed.col_type) t = VarType::kContinuous;
  const Solution residual = solve_pdhg_gpu(fixed, lp_opts, silent);
  if (residual.status != SolveStatus::kOptimal && residual.status != SolveStatus::kFeasible) {
    return std::nullopt;
  }
  // Check that the result is integer-feasible.
  for (Index j = 0; j < n; ++j) {
    if (model.col_type[static_cast<std::size_t>(j)] != VarType::kInteger) continue;
    if (std::fabs(residual.col_value[static_cast<std::size_t>(j)] -
                  std::round(residual.col_value[static_cast<std::size_t>(j)])) >
        tol::kIntegrality) {
      return std::nullopt;
    }
  }
  const double obj =
      std::inner_product(model.col_cost.begin(), model.col_cost.end(),
                         residual.col_value.begin(), 0.0) *
      (model.sense == ObjSense::kMaximize ? -1.0 : 1.0);
  return HeuristicSolution{residual.col_value, obj};
}

}  // namespace sankhya::gpu
