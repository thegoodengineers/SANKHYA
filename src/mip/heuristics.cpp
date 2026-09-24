// SPDX-License-Identifier: Apache-2.0
// SANKHYA - MILP primal heuristics (#290, #507). References on the declarations.

#include "mip/heuristics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

#include "sankhya/logging.hpp"
#include "sankhya/sparse.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::mip {
namespace {

double clamp_to(double value, double lower, double upper) {
  if (is_finite_bound(lower)) value = std::max(value, std::ceil(lower - 1e-9));
  if (is_finite_bound(upper)) value = std::min(value, std::floor(upper + 1e-9));
  return value;
}

/// How far `activity` is outside [lower, upper].
double violation(double activity, double lower, double upper) {
  double v = 0.0;
  if (is_finite_bound(lower)) v = std::max(v, lower - activity);
  if (is_finite_bound(upper)) v = std::max(v, activity - upper);
  return v;
}

}  // namespace

Locks compute_locks(const Model& model) {
  Locks locks;
  const auto n = static_cast<std::size_t>(model.num_cols());
  locks.up.assign(n, 0);
  locks.down.assign(n, 0);
  for (Index j = 0; j < model.num_cols(); ++j) {
    const ColumnView col = model.matrix.column(j);
    const auto u = static_cast<std::size_t>(j);
    for (Index k = 0; k < col.size; ++k) {
      const auto i = static_cast<std::size_t>(col.rows[k]);
      const double a = col.values[k];
      if (a == 0.0) continue;
      // Moving the column up moves a positive-coefficient row's activity up, towards its
      // upper bound; a negative coefficient's down, towards its lower bound.
      const bool lower_matters = is_finite_bound(model.row_lower[i]);
      const bool upper_matters = is_finite_bound(model.row_upper[i]);
      if ((a > 0.0 && upper_matters) || (a < 0.0 && lower_matters)) ++locks.up[u];
      if ((a > 0.0 && lower_matters) || (a < 0.0 && upper_matters)) ++locks.down[u];
    }
  }
  return locks;
}

std::vector<double> lock_round(const Model& model, const Locks& locks,
                               const std::vector<Index>& integer_columns,
                               const std::vector<double>& x) {
  std::vector<double> rounded = x;
  for (const Index j : integer_columns) {
    const auto u = static_cast<std::size_t>(j);
    const double v = x[u];
    double r = std::round(v);
    if (std::fabs(v - r) > tol::kIntegrality) {
      // A direction no row locks can be taken without breaking any row that the LP point
      // satisfied; with both directions locked, nearest is as good a guess as any.
      if (locks.down[u] == 0) {
        r = std::floor(v);
      } else if (locks.up[u] == 0) {
        r = std::ceil(v);
      }
    }
    rounded[u] = clamp_to(r, model.col_lower[u], model.col_upper[u]);
  }
  return rounded;
}

bool repair(const Model& model, const std::vector<Index>& integer_columns,
            std::vector<double>* x_io, int max_moves, double tolerance, Count* moves) {
  std::vector<double>& x = *x_io;
  *moves = 0;
  const auto m = static_cast<std::size_t>(model.num_rows());
  std::vector<double> activity(m, 0.0);
  model.matrix.multiply_add(x.data(), activity.data());
  std::vector<double> v(m, 0.0);
  double total = 0.0;
  for (std::size_t i = 0; i < m; ++i) {
    v[i] = violation(activity[i], model.row_lower[i], model.row_upper[i]);
    total += v[i];
  }
  std::vector<char> is_integer(static_cast<std::size_t>(model.num_cols()), 0);
  for (const Index j : integer_columns) is_integer[static_cast<std::size_t>(j)] = 1;
  const CsrView by_row(model.matrix);
  const double sense = model.sense_multiplier();

  while (*moves < max_moves) {
    std::size_t worst = m;
    double worst_v = tolerance;
    for (std::size_t i = 0; i < m; ++i) {
      if (v[i] > worst_v) {
        worst_v = v[i];
        worst = i;
      }
    }
    if (worst == m) return true;  // every row within tolerance

    // The best one-unit shift of an integer column in the worst row: the most violation
    // removed over ALL rows the column touches, then the smallest objective cost, then the
    // lowest index - so the result does not depend on anything but the model and the point.
    Index best_column = -1;
    double best_step = 0.0;
    double best_total = total;
    double best_cost = std::numeric_limits<double>::infinity();
    const ColumnView row = by_row.row(static_cast<Index>(worst));
    for (Index k = 0; k < row.size; ++k) {
      const Index j = row.rows[k];
      const auto u = static_cast<std::size_t>(j);
      if (is_integer[u] == 0) continue;
      for (const double step : {1.0, -1.0}) {
        const double moved = x[u] + step;
        if (is_finite_bound(model.col_lower[u]) && moved < model.col_lower[u] - 1e-9) continue;
        if (is_finite_bound(model.col_upper[u]) && moved > model.col_upper[u] + 1e-9) continue;
        double after = total;
        const ColumnView col = model.matrix.column(j);
        for (Index t = 0; t < col.size; ++t) {
          const auto i = static_cast<std::size_t>(col.rows[t]);
          after -= v[i];
          after += violation(activity[i] + step * col.values[t], model.row_lower[i],
                             model.row_upper[i]);
        }
        const double cost = sense * model.col_cost[u] * step;
        const bool better = after < best_total - 1e-12 ||
                            (after <= best_total + 1e-12 && best_column >= 0 &&
                             (cost < best_cost || (cost == best_cost && j < best_column)));
        if (better && after < total - 1e-12) {
          best_column = j;
          best_step = step;
          best_total = after;
          best_cost = cost;
        }
      }
    }
    if (best_column < 0) return false;  // stuck: no shift in the worst row helps

    const auto u = static_cast<std::size_t>(best_column);
    x[u] += best_step;
    const ColumnView col = model.matrix.column(best_column);
    for (Index t = 0; t < col.size; ++t) {
      const auto i = static_cast<std::size_t>(col.rows[t]);
      activity[i] += best_step * col.values[t];
      total -= v[i];
      v[i] = violation(activity[i], model.row_lower[i], model.row_upper[i]);
      total += v[i];
    }
    ++*moves;
  }
  // Out of moves: the same test the loop makes, on the rows as they now stand.
  return std::all_of(v.begin(), v.end(), [&](double value) { return value <= tolerance; });
}

bool rins_submodel(const Model& model, const std::vector<Index>& integer_columns,
                   const std::vector<double>& relaxation, const std::vector<double>& incumbent,
                   double min_fixed_fraction, double tolerance, Model* out, Count* fixed) {
  *fixed = 0;
  std::vector<Index> agree;
  for (const Index j : integer_columns) {
    const auto u = static_cast<std::size_t>(j);
    if (std::fabs(relaxation[u] - incumbent[u]) <= tolerance) agree.push_back(j);
  }
  if (integer_columns.empty() ||
      static_cast<double>(agree.size()) <
          min_fixed_fraction * static_cast<double>(integer_columns.size())) {
    return false;
  }
  *out = model;
  for (const Index j : agree) {
    const auto u = static_cast<std::size_t>(j);
    const double value = std::round(incumbent[u]);
    out->col_lower[u] = value;
    out->col_upper[u] = value;
  }
  *fixed = static_cast<Count>(agree.size());
  return true;
}

std::vector<double> feasibility_pump(const Model& model,
                                     const std::vector<Index>& integer_columns,
                                     const std::vector<double>& start,
                                     const Options& lp_options, int max_rounds,
                                     double integrality_tolerance, Count* lp_solves) {
  *lp_solves = 0;
  if (integer_columns.empty()) return {};
  // The projection LP: the model's rows and bounds, every column continuous, and an
  // objective replaced round by round with the L1 distance to the current rounding.
  Model lp = model;
  std::fill(lp.col_type.begin(), lp.col_type.end(), VarType::kContinuous);
  lp.sense = ObjSense::kMinimize;
  lp.objective_offset = 0.0;
  lp.hessian.reset(model.num_cols(), model.num_cols());
  lp.hessian.finalize();

  std::vector<double> x = start;
  std::vector<double> previous_rounding;
  for (int round = 0; round < max_rounds; ++round) {
    // Round, and test whether the LP point is already integral on the integer columns.
    std::vector<double> rounding(x.size(), 0.0);
    bool integral = true;
    for (const Index j : integer_columns) {
      const auto u = static_cast<std::size_t>(j);
      rounding[u] = clamp_to(std::round(x[u]), model.col_lower[u], model.col_upper[u]);
      integral = integral && std::fabs(x[u] - rounding[u]) <= integrality_tolerance;
    }
    if (integral) {
      // The LP point is feasible (the LP said so) and integral where it must be: done. The
      // integer columns are snapped to their integers so the caller's check sees them exact.
      for (const Index j : integer_columns) {
        x[static_cast<std::size_t>(j)] = rounding[static_cast<std::size_t>(j)];
      }
      return x;
    }
    // A CYCLE - the same rounding as last round - is broken by flipping the columns whose
    // LP value is furthest from their rounding, the pump's standard perturbation. Ordered by
    // that distance and then by index, so a rerun makes the same flips.
    if (rounding == previous_rounding) {
      std::vector<std::pair<double, Index>> far;
      for (const Index j : integer_columns) {
        const auto u = static_cast<std::size_t>(j);
        far.emplace_back(-std::fabs(x[u] - rounding[u]), j);
      }
      std::sort(far.begin(), far.end());
      const std::size_t flips = std::max<std::size_t>(1, integer_columns.size() / 10);
      for (std::size_t k = 0; k < std::min(flips, far.size()); ++k) {
        const auto u = static_cast<std::size_t>(far[k].second);
        const double away = x[u] > rounding[u] ? rounding[u] + 1.0 : rounding[u] - 1.0;
        rounding[u] = clamp_to(away, model.col_lower[u], model.col_upper[u]);
      }
    }
    previous_rounding = rounding;

    // The L1 distance to the rounding is linear in x for a column rounded to one of its
    // bounds: x - l at the lower, u - x at the upper. A general integer rounded strictly
    // between its bounds would need an auxiliary column and is left out of the distance.
    std::fill(lp.col_cost.begin(), lp.col_cost.end(), 0.0);
    for (const Index j : integer_columns) {
      const auto u = static_cast<std::size_t>(j);
      if (is_finite_bound(model.col_lower[u]) && rounding[u] <= model.col_lower[u] + 1e-9) {
        lp.col_cost[u] = 1.0;
      } else if (is_finite_bound(model.col_upper[u]) &&
                 rounding[u] >= model.col_upper[u] - 1e-9) {
        lp.col_cost[u] = -1.0;
      }
    }
    const Solution projected = solve(lp, lp_options);
    ++*lp_solves;
    if (projected.status != SolveStatus::kOptimal) return {};
    x = projected.col_value;
  }
  return {};
}

const char* to_string(DiveRule rule) {
  switch (rule) {
    case DiveRule::kFractional: return "fractional diving";
    case DiveRule::kCoefficient: return "coefficient diving";
    case DiveRule::kVectorLength: return "vector length diving";
    case DiveRule::kGuided: return "guided diving";
  }
  return "diving";
}

DiveChoice choose_dive_column(const Model& model, const Locks& locks,
                              const std::vector<Index>& integer_columns,
                              const std::vector<double>& x, DiveRule rule,
                              const std::vector<double>& incumbent,
                              double integrality_tolerance) {
  DiveChoice best;
  if (rule == DiveRule::kGuided && incumbent.size() != x.size()) return best;
  double best_score = std::numeric_limits<double>::infinity();
  const double sense = model.sense_multiplier();
  for (const Index j : integer_columns) {
    const auto u = static_cast<std::size_t>(j);
    const double v = x[u];
    const double nearest = std::round(v);
    const double fraction = std::fabs(v - nearest);
    if (fraction <= integrality_tolerance) continue;
    const double down = std::floor(v);
    const double up = std::ceil(v);
    bool round_up = nearest > v;
    double score = fraction;
    switch (rule) {
      case DiveRule::kFractional:
        // The least fractional column, to its nearest integer: the rule the root dive has
        // always used (#25) - lock in what the relaxation already nearly agrees on.
        break;
      case DiveRule::kCoefficient: {
        // Achterberg 2007, sec. 9.2.2: the direction fewer rows object to, and among the
        // columns the one with the fewest objections, ties broken by fractionality.
        const int up_locks = locks.up[u];
        const int down_locks = locks.down[u];
        round_up = up_locks < down_locks || (up_locks == down_locks && round_up);
        score = static_cast<double>(round_up ? up_locks : down_locks) + fraction;
        break;
      }
      case DiveRule::kVectorLength: {
        // Achterberg 2007, sec. 9.2.4: round against the objective (up when the cost is
        // non-negative in minimise space), and prefer the column whose objective increase
        // is spread over the most rows - on a covering model one such fix satisfies many.
        const double cost = sense * model.col_cost[u];
        round_up = cost >= 0.0;
        const double increase = round_up ? (up - v) * cost : (v - down) * (-cost);
        score = increase / static_cast<double>(model.matrix.column(j).size + 1);
        break;
      }
      case DiveRule::kGuided: {
        // Achterberg 2007, sec. 9.2.3: towards the incumbent, the closest column first.
        const double target = incumbent[u];
        round_up = target >= up;
        score = std::fabs(v - target);
        break;
      }
    }
    // Strictly better only, so the lowest index wins a tie and a rerun makes the same dive.
    if (score < best_score) {
      best_score = score;
      best.column = j;
      best.value = clamp_to(round_up ? up : down, model.col_lower[u], model.col_upper[u]);
    }
  }
  return best;
}

bool rens_submodel(const Model& model, const std::vector<Index>& integer_columns,
                   const std::vector<double>& relaxation, double min_fixed_fraction,
                   double tolerance, Model* out, Count* fixed) {
  *fixed = 0;
  Count integral = 0;
  for (const Index j : integer_columns) {
    const auto u = static_cast<std::size_t>(j);
    if (std::fabs(relaxation[u] - std::round(relaxation[u])) <= tolerance) ++integral;
  }
  if (integer_columns.empty() ||
      static_cast<double>(integral) <
          min_fixed_fraction * static_cast<double>(integer_columns.size())) {
    return false;
  }
  *out = model;
  for (const Index j : integer_columns) {
    const auto u = static_cast<std::size_t>(j);
    const double v = relaxation[u];
    const double nearest = std::round(v);
    if (std::fabs(v - nearest) <= tolerance) {
      const double value = clamp_to(nearest, model.col_lower[u], model.col_upper[u]);
      out->col_lower[u] = value;
      out->col_upper[u] = value;
    } else {
      // The two integers around the value, inside the column's own bounds.
      const double low = std::floor(v);
      const double high = std::ceil(v);
      out->col_lower[u] =
          is_finite_bound(model.col_lower[u]) ? std::max(model.col_lower[u], low) : low;
      out->col_upper[u] =
          is_finite_bound(model.col_upper[u]) ? std::min(model.col_upper[u], high) : high;
    }
  }
  *fixed = integral;
  return true;
}

// =============================================================================
// Local-MIP improvement (#507)
// =============================================================================

// Lin, Zou and Cai, "Local-MIP: efficient local search for mixed integer
// programming", CP 2024, LIPIcs 307.

bool local_mip_improve(const Model& model, const Options& options, Solution& incumbent,
                       Logger& logger) {
  (void)options;  // option guard already checked by caller
  if (incumbent.col_value.empty()) return false;

  // Collect integer columns.
  std::vector<Index> int_cols;
  int_cols.reserve(static_cast<std::size_t>(model.num_integer_columns()));
  for (Index j = 0; j < model.num_cols(); ++j) {
    if (model.col_type[static_cast<std::size_t>(j)] == VarType::kInteger) int_cols.push_back(j);
  }
  if (int_cols.empty()) return false;

  // Working point.
  std::vector<double> x = incumbent.col_value;
  double best_obj = incumbent.objective;  // in model sense (with offset)
  const double sense = model.sense_multiplier();

  // Row activities.
  const auto m = static_cast<std::size_t>(model.num_rows());
  std::vector<double> activity(m, 0.0);
  model.matrix.multiply_add(x.data(), activity.data());

  // Constraint weights (FJ scheme): one per original row.
  std::vector<double> weights(m, 1.0);

  // Breakthrough: virtual row  sense * (c . x) <= sense * z* - 1  (cut off anything no
  // better than the current incumbent in minimisation space).
  // cutoff_min = sense * best_obj - offset already folded in evaluate_objective, so:
  //   sense * sum c_j x_j <= sense * (best_obj - offset) - 1
  // We track sense * sum c_j x_j separately.
  auto min_obj_activity = [&]() {
    double v = 0.0;
    for (std::size_t j = 0; j < static_cast<std::size_t>(model.num_cols()); ++j) {
      v += model.col_cost[j] * x[j];
    }
    return sense * v;
  };

  double cutoff_min = sense * (best_obj - model.objective_offset) - 1.0;
  double obj_act = min_obj_activity();
  double bt_weight = 1.0;

  // Tabu list: (column, direction) pairs.
  struct TabuEntry {
    Index column;
    double direction;  // +1.0 or -1.0
    int tenure;
  };
  std::vector<TabuEntry> tabu;
  tabu.reserve(32);

  std::mt19937 rng(12345);
  std::uniform_int_distribution<int> rand10(0, 9);

  auto is_tabu = [&](Index col, double dir) -> bool {
    for (const auto& e : tabu) {
      if (e.column == col && e.direction == dir) return true;
    }
    return false;
  };
  auto add_tabu = [&](Index col, double dir) { tabu.push_back({col, dir, 3 + rand10(rng)}); };
  auto tick_tabu = [&]() {
    for (auto& e : tabu) --e.tenure;
    tabu.erase(std::remove_if(tabu.begin(), tabu.end(),
                              [](const TabuEntry& e) { return e.tenure <= 0; }),
               tabu.end());
  };

  constexpr int kMaxIter = 500;
  constexpr double kWeightGrowth = 1.1;  // multiplicative increase on violation
  constexpr double kWeightDecay = 0.95;  // smoothing: floor at 1.0

  bool improved = false;

  for (int iter = 0; iter < kMaxIter; ++iter) {
    tick_tabu();

    // Feasibility check for the current point.
    bool feasible = true;
    for (std::size_t i = 0; i < m; ++i) {
      if (violation(activity[i], model.row_lower[i], model.row_upper[i]) >
          tol::kPrimalFeasibility) {
        feasible = false;
        break;
      }
    }

    if (feasible) {
      // ---- Lift move ----------------------------------------------------------
      // Shift an integer column by ±1 in the direction that strictly improves the
      // objective, provided the move stays feasible.
      bool found = false;
      for (const Index j : int_cols) {
        const auto u = static_cast<std::size_t>(j);
        const double c_min = sense * model.col_cost[u];  // in minimise space
        if (c_min == 0.0) continue;
        const double step = (c_min < 0.0) ? 1.0 : -1.0;
        if (is_tabu(j, step)) continue;
        const double moved = x[u] + step;
        if (is_finite_bound(model.col_lower[u]) && moved < model.col_lower[u] - 1e-9) continue;
        if (is_finite_bound(model.col_upper[u]) && moved > model.col_upper[u] + 1e-9) continue;

        const ColumnView col = model.matrix.column(j);
        bool stays_feasible = true;
        for (Index k = 0; k < col.size; ++k) {
          const auto i = static_cast<std::size_t>(col.rows[k]);
          if (violation(activity[i] + step * col.values[k], model.row_lower[i],
                        model.row_upper[i]) > tol::kPrimalFeasibility) {
            stays_feasible = false;
            break;
          }
        }
        if (!stays_feasible) continue;

        // Accept lift move.
        x[u] += step;
        for (Index k = 0; k < col.size; ++k) {
          const auto i = static_cast<std::size_t>(col.rows[k]);
          activity[i] += step * col.values[k];
        }
        obj_act += c_min * step;
        add_tabu(j, -step);
        found = true;

        const double new_obj = model.evaluate_objective(x.data());
        if (new_obj < best_obj - tol::kPrimalFeasibility) {
          best_obj = new_obj;
          incumbent.col_value = x;
          incumbent.objective = new_obj;
          improved = true;
          cutoff_min = sense * (best_obj - model.objective_offset) - 1.0;
          logger.verbose("local_mip: lift move improved objective to {:.10g}", new_obj);
        }
        break;
      }
      if (!found) {
        // No lift move available; decay weights and stop.
        break;
      }
    } else {
      // ---- Tight / breakthrough move ------------------------------------------
      // Pick the integer variable and direction (±1) that most reduces the weighted
      // sum of constraint violations plus the breakthrough penalty.

      Index best_col = -1;
      double best_step = 0.0;
      double best_delta = 0.0;  // reduction in total weighted violation (positive = better)

      for (const Index j : int_cols) {
        const auto u = static_cast<std::size_t>(j);
        for (const double step : {1.0, -1.0}) {
          if (is_tabu(j, step)) continue;
          const double moved = x[u] + step;
          if (is_finite_bound(model.col_lower[u]) && moved < model.col_lower[u] - 1e-9)
            continue;
          if (is_finite_bound(model.col_upper[u]) && moved > model.col_upper[u] + 1e-9)
            continue;

          // Change in weighted row violations.
          double delta = 0.0;
          const ColumnView col = model.matrix.column(j);
          for (Index k = 0; k < col.size; ++k) {
            const auto i = static_cast<std::size_t>(col.rows[k]);
            delta -=
                weights[i] * violation(activity[i], model.row_lower[i], model.row_upper[i]);
            delta += weights[i] * violation(activity[i] + step * col.values[k],
                                            model.row_lower[i], model.row_upper[i]);
          }
          // Breakthrough contribution.
          const double c_min = sense * model.col_cost[u];
          const double new_obj_act = obj_act + c_min * step;
          delta -= bt_weight * std::max(0.0, obj_act - cutoff_min);
          delta += bt_weight * std::max(0.0, new_obj_act - cutoff_min);

          const double reduction = -delta;
          if (reduction > best_delta ||
              (reduction == best_delta && best_col >= 0 && j < best_col)) {
            best_delta = reduction;
            best_col = j;
            best_step = step;
          }
        }
      }

      if (best_col < 0 || best_delta <= 0.0) {
        // Stuck: bump weights on the most-violated constraint and continue.
        double wv_max = 0.0;
        std::size_t wv_row = m;
        for (std::size_t i = 0; i < m; ++i) {
          const double wv =
              weights[i] * violation(activity[i], model.row_lower[i], model.row_upper[i]);
          if (wv > wv_max) {
            wv_max = wv;
            wv_row = i;
          }
        }
        const double bt_v = bt_weight * std::max(0.0, obj_act - cutoff_min);
        if (wv_row < m && wv_max >= bt_v) {
          weights[wv_row] *= kWeightGrowth;
        } else {
          bt_weight *= kWeightGrowth;
        }
        continue;
      }

      // Apply the best move.
      const auto u = static_cast<std::size_t>(best_col);
      x[u] += best_step;
      const ColumnView col = model.matrix.column(best_col);
      for (Index k = 0; k < col.size; ++k) {
        const auto i = static_cast<std::size_t>(col.rows[k]);
        activity[i] += best_step * col.values[k];
        if (violation(activity[i], model.row_lower[i], model.row_upper[i]) >
            tol::kPrimalFeasibility) {
          weights[i] *= kWeightGrowth;
        }
      }
      obj_act += sense * model.col_cost[u] * best_step;
      add_tabu(best_col, -best_step);

      // Additive weight smoothing: decay all weights (floor at 1).
      for (double& w : weights) w = std::max(1.0, w * kWeightDecay);
      bt_weight = std::max(1.0, bt_weight * kWeightDecay);

      // If the point is now feasible and better, record it.
      bool now_feasible = true;
      for (std::size_t i = 0; i < m; ++i) {
        if (violation(activity[i], model.row_lower[i], model.row_upper[i]) >
            tol::kPrimalFeasibility) {
          now_feasible = false;
          break;
        }
      }
      if (now_feasible) {
        bool integral = true;
        for (const Index j2 : int_cols) {
          const double v = x[static_cast<std::size_t>(j2)];
          if (std::fabs(v - std::round(v)) > tol::kIntegrality) {
            integral = false;
            break;
          }
        }
        if (integral) {
          const double new_obj = model.evaluate_objective(x.data());
          if (new_obj < best_obj - tol::kPrimalFeasibility) {
            best_obj = new_obj;
            incumbent.col_value = x;
            incumbent.objective = new_obj;
            improved = true;
            cutoff_min = sense * (best_obj - model.objective_offset) - 1.0;
            logger.verbose("local_mip: breakthrough improved objective to {:.10g}", new_obj);
          }
        }
      }
    }
  }

  return improved;
}

namespace {
/// auto follows the master switch; on and off decide alone.
bool resolve_switch(const Options& options, const char* name, bool master) {
  const std::string& value = options.get_string(name);
  return value == "on" || (value == "auto" && master);
}
}  // namespace

HeuristicSchedule HeuristicSchedule::from(const Options& options) {
  HeuristicSchedule s;
  const bool master = options.get_bool("mip_heuristics");
  s.lock_rounding = resolve_switch(options, "mip_heur_lock_rounding", master);
  s.repair = resolve_switch(options, "mip_heur_repair", master);
  s.pump = resolve_switch(options, "mip_heur_pump", master);
  s.rins = resolve_switch(options, "mip_heur_rins", master);
  s.rens = resolve_switch(options, "mip_heur_rens", master);
  s.dive[static_cast<std::size_t>(DiveRule::kFractional)] =
      resolve_switch(options, "mip_heur_dive_fractional", master);
  s.dive[static_cast<std::size_t>(DiveRule::kCoefficient)] =
      resolve_switch(options, "mip_heur_dive_coefficient", master);
  s.dive[static_cast<std::size_t>(DiveRule::kVectorLength)] =
      resolve_switch(options, "mip_heur_dive_vector_length", master);
  s.dive[static_cast<std::size_t>(DiveRule::kGuided)] =
      resolve_switch(options, "mip_heur_dive_guided", master);
  s.dive_backtrack = options.get_bool("mip_dive_backtrack");
  s.rins_frequency = options.get_int("mip_rins_frequency");
  s.rins_nodes = options.get_int("mip_rins_nodes");
  s.rens_nodes = options.get_int("mip_rens_nodes");
  s.dive_frequency = options.get_int("mip_dive_frequency");
  s.dive_lp_resolves = static_cast<int>(options.get_int("mip_dive_lp_resolves"));
  s.pump_rounds = static_cast<int>(options.get_int("mip_pump_rounds"));
  s.seconds_budgets = !options.get_bool("deterministic");
  return s;
}

bool HeuristicSchedule::any_optional() const {
  return lock_rounding || repair || pump || rins || rens ||
         dive[static_cast<std::size_t>(DiveRule::kCoefficient)] ||
         dive[static_cast<std::size_t>(DiveRule::kVectorLength)] ||
         dive[static_cast<std::size_t>(DiveRule::kGuided)];
}

std::string HeuristicSchedule::names() const {
  std::string out = "rounding";
  const auto add = [&out](bool on, const char* name) {
    if (!on) return;
    out += ", ";
    out += name;
  };
  add(lock_rounding, "lock rounding");
  add(repair, "repair");
  for (std::size_t r = 0; r < kDiveRules; ++r)
    add(dive[r], to_string(static_cast<DiveRule>(r)));
  add(pump, "feasibility pump");
  add(rins, "RINS");
  add(rens, "RENS");
  return out;
}

}  // namespace sankhya::mip
