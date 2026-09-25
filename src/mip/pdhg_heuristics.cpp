// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the PDHG feasibility pump and fix-and-propagate (#509). See pdhg_heuristics.hpp
// for the algorithms, their citations, and the rule that nothing here decides an answer.

#include "pdhg_heuristics.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <optional>
#include <random>
#include <utility>

#include "domain_propagation.hpp"
#include "feasibility_jump.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/pdhg.hpp"
#include "sankhya/timer.hpp"

#ifdef SANKHYA_ENABLE_CUDA
#include "../gpu/device.hpp"
#include "../gpu/domain_prop.hpp"
#include "../gpu/pdhg_gpu.hpp"
#endif

namespace sankhya::mip {
namespace {

/// The integer domain of a column: [ceil(lower), floor(upper)] to the integrality tolerance,
/// an infinite side left infinite.
double integer_floor_of(double upper, double integrality) {
  return is_finite_bound(upper) ? std::floor(upper + integrality) : kInfinity;
}
double integer_ceil_of(double lower, double integrality) {
  return is_finite_bound(lower) ? std::ceil(lower - integrality) : -kInfinity;
}

/// The integer closest to v inside [lower, upper].
double nearest_inside(double v, double lower, double upper, double integrality) {
  double r = std::round(v);
  r = std::max(r, integer_ceil_of(lower, integrality));
  r = std::min(r, integer_floor_of(upper, integrality));
  return r;
}

double clamp_inside(double v, double lower, double upper) {
  if (is_finite_bound(lower)) v = std::max(v, lower);
  if (is_finite_bound(upper)) v = std::min(v, upper);
  return v;
}

/// The seconds one run may spend (PdhgHeuristicSettings::seconds), measured from its start.
class Budget {
 public:
  explicit Budget(double seconds) : seconds_(seconds) {}
  [[nodiscard]] bool limited() const { return seconds_ >= 0.0; }
  [[nodiscard]] double left() const {
    return limited() ? std::max(0.0, seconds_ - clock_.elapsed_seconds())
                     : std::numeric_limits<double>::infinity();
  }

 private:
  double seconds_;
  Timer clock_;
};

bool device_ready(const PdhgHeuristicSettings& settings) {
#ifdef SANKHYA_ENABLE_CUDA
  return settings.use_device && gpu::device_available(nullptr);
#else
  (void)settings;
  return false;
#endif
}

/// A PDHG solve for a heuristic: quiet, to kPdhgLoose, stopped at the request, capped in
/// iterations and (when the caller has one) seconds. The point is a guide only.
Solution solve_pdhg_for(const Model& lp, const PdhgHeuristicSettings& settings, bool device,
                        const Budget& budget) {
  Options o;
  o.set_bool("log_to_console", false);
  o.set_double("pdhg_tolerance", tol::kPdhgLoose);
  o.set_bool("pdhg_stop_at_request", true);
  o.set_int("iteration_limit", settings.pdhg_iterations);
  if (budget.limited()) o.set_double("time_limit", budget.left());
  Logger quiet(nullptr);
#ifdef SANKHYA_ENABLE_CUDA
  if (device) return gpu::solve_pdhg_gpu(lp, o, quiet);
#else
  (void)device;
#endif
  return pdhg::solve_pdhg(lp, o, quiet);
}

/// A PDHG result carries a usable point when it stopped on its tolerance or a limit; a
/// certificate or a model error carries none.
bool has_point(const Solution& s, Index cols) {
  if (static_cast<Index>(s.col_value.size()) < cols) return false;
  switch (s.status) {
    case SolveStatus::kOptimal:
    case SolveStatus::kFeasible:
    case SolveStatus::kIterationLimit:
    case SolveStatus::kTimeLimit: break;
    default: return false;
  }
  for (Index j = 0; j < cols; ++j) {
    if (!std::isfinite(s.col_value[static_cast<std::size_t>(j)])) return false;
  }
  return true;
}

/// The LP relaxation's point by PDHG, for a caller that has none.
std::vector<double> relaxation_point(const Model& model, const PdhgHeuristicSettings& settings,
                                     bool device, const Budget& budget,
                                     PdhgHeuristicResult* result) {
  Model lp = model;
  std::fill(lp.col_type.begin(), lp.col_type.end(), VarType::kContinuous);
  const Solution relaxed = solve_pdhg_for(lp, settings, device, budget);
  ++result->pdhg_solves;
  if (!has_point(relaxed, model.num_cols())) return {};
  return {relaxed.col_value.begin(), relaxed.col_value.begin() + model.num_cols()};
}

/// A necessary condition for the continuous columns to complete `x` (whose integer columns
/// are fixed): every row's activity range, the integer part fixed and the continuous part
/// free in its box, meets the row's sides. On a pure integer model it is the row check itself.
bool rows_can_hold(const Model& model, const std::vector<char>& is_integer,
                   const std::vector<double>& x) {
  const auto m = static_cast<std::size_t>(model.num_rows());
  std::vector<double> fixed(m, 0.0);
  std::vector<double> low(m, 0.0);
  std::vector<double> high(m, 0.0);
  std::vector<char> low_infinite(m, 0);
  std::vector<char> high_infinite(m, 0);
  for (Index j = 0; j < model.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    const ColumnView column = model.matrix.column(j);
    for (Index k = 0; k < column.size; ++k) {
      const auto i = static_cast<std::size_t>(column.rows[k]);
      const double a = column.values[k];
      if (is_integer[u] != 0) {
        fixed[i] += a * x[u];
        continue;
      }
      const double at_low = a > 0.0 ? model.col_lower[u] : model.col_upper[u];
      const double at_high = a > 0.0 ? model.col_upper[u] : model.col_lower[u];
      if (is_finite_bound(at_low)) {
        low[i] += a * at_low;
      } else {
        low_infinite[i] = 1;
      }
      if (is_finite_bound(at_high)) {
        high[i] += a * at_high;
      } else {
        high_infinite[i] = 1;
      }
    }
  }
  for (std::size_t i = 0; i < m; ++i) {
    if (low_infinite[i] == 0 && is_finite_bound(model.row_upper[i]) &&
        fixed[i] + low[i] > model.row_upper[i] + tol::kPrimalFeasibility) {
      return false;
    }
    if (high_infinite[i] == 0 && is_finite_bound(model.row_lower[i]) &&
        fixed[i] + high[i] < model.row_lower[i] - tol::kPrimalFeasibility) {
      return false;
    }
  }
  return true;
}

/// The continuous columns of `x` completed by an LP with every integer column fixed at its
/// value in `x`, solved with the caller's completion options; the point returned has passed
/// point_is_feasible(), or is empty.
std::vector<double> complete(const Model& model, const std::vector<Index>& integer_columns,
                             const std::vector<double>& x,
                             const PdhgHeuristicSettings& settings, const Budget& budget,
                             PdhgHeuristicResult* result) {
  Model lp = model;
  for (const Index j : integer_columns) {
    const auto u = static_cast<std::size_t>(j);
    lp.col_lower[u] = x[u];
    lp.col_upper[u] = x[u];
  }
  std::fill(lp.col_type.begin(), lp.col_type.end(), VarType::kContinuous);
  Options options = settings.completion_options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", false);
  if (budget.limited()) options.set_double("time_limit", budget.left());
  const Solution completed = solve(lp, options);
  ++result->completions;
  if (completed.status != SolveStatus::kOptimal ||
      completed.col_value.size() != static_cast<std::size_t>(model.num_cols())) {
    return {};
  }
  std::vector<double> point = completed.col_value;
  for (const Index j : integer_columns) {
    point[static_cast<std::size_t>(j)] = x[static_cast<std::size_t>(j)];
  }
  if (!point_is_feasible(model, integer_columns, point, settings.integrality_tolerance))
    return {};
  return point;
}

/// FNV-1a over the integer columns' values: the pump's cycle memory. A collision can only
/// cause one perturbation too many.
std::uint64_t rounding_hash(const std::vector<Index>& integer_columns,
                            const std::vector<double>& x) {
  std::uint64_t h = 1469598103934665603ULL;
  for (const Index j : integer_columns) {
    const auto v = static_cast<std::int64_t>(x[static_cast<std::size_t>(j)]);
    h ^= static_cast<std::uint64_t>(v);
    h *= 1099511628211ULL;
  }
  return h;
}

bool should_stop(const PdhgHeuristicSettings& settings, const Budget& budget) {
  if (budget.limited() && budget.left() <= 0.0) return true;
  return settings.should_stop && settings.should_stop();
}

}  // namespace

bool point_is_feasible(const Model& model, const std::vector<Index>& integer_columns,
                       const std::vector<double>& x, double integrality) {
  const Index n = model.num_cols();
  if (static_cast<Index>(x.size()) != n) return false;
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (!std::isfinite(x[u])) return false;
    if (is_finite_bound(model.col_lower[u]) &&
        x[u] < model.col_lower[u] - tol::kPrimalFeasibility) {
      return false;
    }
    if (is_finite_bound(model.col_upper[u]) &&
        x[u] > model.col_upper[u] + tol::kPrimalFeasibility) {
      return false;
    }
  }
  for (const Index j : integer_columns) {
    const double v = x[static_cast<std::size_t>(j)];
    if (std::fabs(v - std::round(v)) > integrality) return false;
  }
  const auto m = static_cast<std::size_t>(model.num_rows());
  std::vector<double> activity(m, 0.0);
  if (m > 0) model.matrix.multiply(x.data(), activity.data());
  for (std::size_t i = 0; i < m; ++i) {
    if (!std::isfinite(activity[i])) return false;
    if (is_finite_bound(model.row_lower[i]) &&
        activity[i] < model.row_lower[i] - tol::kPrimalFeasibility) {
      return false;
    }
    if (is_finite_bound(model.row_upper[i]) &&
        activity[i] > model.row_upper[i] + tol::kPrimalFeasibility) {
      return false;
    }
  }
  return true;
}

// ---- The feasibility pump -----------------------------------------------------------------
//
// THE PROJECTION LP is built once and only its data changes between rounds, so the matrix is
// the same every round. Its columns are the model's, all continuous, plus one auxiliary
// column d_k >= 0 per GENERAL integer column (Bertacco, Fischetti & Lodi 2007) with the two
// rows d_k - x_j >= -r_j and d_k + x_j >= r_j, so that d_k >= |x_j - r_j| and cost 1 on d_k
// measures the distance. A column whose integer domain is exactly its two integral bounds
// needs no auxiliary: its distance is x_j - l_j at r_j = l_j and u_j - x_j at r_j = u_j, a
// cost of +1 or -1 on x_j (FGL 2005). A column with one integer in its domain has distance 0.
// Round by round only those costs and the auxiliary rows' lower sides move.
PdhgHeuristicResult pdhg_feasibility_pump(const Model& model,
                                          const std::vector<Index>& integer_columns,
                                          const std::vector<double>& start,
                                          const PdhgHeuristicSettings& settings) {
  PdhgHeuristicResult result;
  const Budget budget(settings.seconds);
  const Index n = model.num_cols();
  const Index m = model.num_rows();
  const double tol_int = settings.integrality_tolerance;
  if (integer_columns.empty()) {
    result.stopped = "no integer columns";
    return result;
  }
  const bool device = device_ready(settings);
  result.on_device = device;

  std::vector<char> is_integer(static_cast<std::size_t>(n), 0);
  for (const Index j : integer_columns) is_integer[static_cast<std::size_t>(j)] = 1;
  const bool mixed = static_cast<Index>(integer_columns.size()) < n;

  // Classify each integer column: -2 fixed, -1 two-valued with integral bounds, k >= 0 the
  // index of its auxiliary column.
  std::vector<Index> kind(static_cast<std::size_t>(n), -2);
  std::vector<Index> generals;  // the column of each auxiliary, by auxiliary index
  for (const Index j : integer_columns) {
    const auto u = static_cast<std::size_t>(j);
    const double lo = integer_ceil_of(model.col_lower[u], tol_int);
    const double hi = integer_floor_of(model.col_upper[u], tol_int);
    if (std::isfinite(lo) && std::isfinite(hi) && hi <= lo) continue;  // one value, or none
    if (std::isfinite(lo) && std::isfinite(hi) && hi - lo == 1.0 && model.col_lower[u] == lo &&
        model.col_upper[u] == hi) {
      kind[u] = -1;
      continue;
    }
    kind[u] = static_cast<Index>(generals.size());
    generals.push_back(j);
  }
  const auto g = static_cast<Index>(generals.size());

  Model lp;
  lp.name = model.name;
  lp.sense = ObjSense::kMinimize;
  lp.col_cost.assign(static_cast<std::size_t>(n + g), 0.0);
  lp.col_lower = model.col_lower;
  lp.col_upper = model.col_upper;
  for (Index k = 0; k < g; ++k) {
    lp.col_cost[static_cast<std::size_t>(n + k)] = 1.0;
    lp.col_lower.push_back(0.0);
    lp.col_upper.push_back(kInfinity);
  }
  lp.col_type.assign(static_cast<std::size_t>(n + g), VarType::kContinuous);
  lp.row_lower = model.row_lower;
  lp.row_upper = model.row_upper;
  lp.row_lower.resize(static_cast<std::size_t>(m + 2 * g), 0.0);
  lp.row_upper.resize(static_cast<std::size_t>(m + 2 * g), kInfinity);
  {
    std::vector<Index> starts;
    std::vector<Index> rows;
    std::vector<double> values;
    starts.reserve(static_cast<std::size_t>(n + g) + 1);
    starts.push_back(0);
    for (Index j = 0; j < n; ++j) {
      const ColumnView column = model.matrix.column(j);
      for (Index k = 0; k < column.size; ++k) {
        rows.push_back(column.rows[k]);
        values.push_back(column.values[k]);
      }
      const Index aux = kind[static_cast<std::size_t>(j)];
      if (aux >= 0) {
        rows.push_back(m + 2 * aux);
        values.push_back(-1.0);
        rows.push_back(m + 2 * aux + 1);
        values.push_back(1.0);
      }
      starts.push_back(static_cast<Index>(rows.size()));
    }
    for (Index k = 0; k < g; ++k) {
      rows.push_back(m + 2 * k);
      values.push_back(1.0);
      rows.push_back(m + 2 * k + 1);
      values.push_back(1.0);
      starts.push_back(static_cast<Index>(rows.size()));
    }
    lp.matrix.assign_columns(m + 2 * g, n + g, std::move(starts), std::move(rows),
                             std::move(values));
  }
  lp.hessian.reset(n + g, n + g);
  lp.hessian.finalize();

  std::vector<double> x = start;
  if (static_cast<Index>(x.size()) != n)
    x = relaxation_point(model, settings, device, budget, &result);
  if (static_cast<Index>(x.size()) != n) {
    result.stopped = "no relaxation point";
    return result;
  }

  std::mt19937_64 rng(settings.seed);
  std::vector<Index> movable;  // the integer columns with more than one value
  for (const Index j : integer_columns) {
    if (kind[static_cast<std::size_t>(j)] != -2) movable.push_back(j);
  }
  // Move column j of `rounding` to another integer of its domain: the other value of a
  // two-valued column; one unit towards the LP point for a general one (away from it when the
  // point sits on the rounding), staying inside the domain.
  const auto flip = [&](std::vector<double>* rounding, Index j) {
    const auto u = static_cast<std::size_t>(j);
    const double lo = integer_ceil_of(model.col_lower[u], tol_int);
    const double hi = integer_floor_of(model.col_upper[u], tol_int);
    const double r = (*rounding)[u];
    const double step = x[u] < r ? -1.0 : 1.0;
    for (const double candidate : {r + step, r - step}) {
      if (candidate >= lo && candidate <= hi) {
        (*rounding)[u] = candidate;
        return;
      }
    }
  };

  std::vector<double> previous;  // the rounding projected on last round
  std::deque<std::uint64_t> recent;
  std::vector<std::uint64_t> completed;  // roundings whose completion LP failed
  for (int round = 0;; ++round) {
    if (should_stop(settings, budget)) {
      result.stopped = "stopped by the interrupt or the time limit";
      return result;
    }
    result.rounds = round + 1;
    // 1. The rounding; the continuous columns as the LP point has them, inside their box.
    std::vector<double> rounding(static_cast<std::size_t>(n));
    for (Index j = 0; j < n; ++j) {
      const auto u = static_cast<std::size_t>(j);
      rounding[u] = is_integer[u] != 0
                        ? nearest_inside(x[u], model.col_lower[u], model.col_upper[u], tol_int)
                        : clamp_inside(x[u], model.col_lower[u], model.col_upper[u]);
    }
    // 2. Cycles (FGL 2005, section 3). The same rounding as the one just projected: flip the
    // TT columns furthest from their rounding, TT drawn from [T/2, 3T/2]. A rounding seen
    // within the last kPumpCycleWindow rounds: the random restart.
    bool same = !previous.empty();
    for (std::size_t k = 0; same && k < integer_columns.size(); ++k) {
      const auto u = static_cast<std::size_t>(integer_columns[k]);
      same = rounding[u] == previous[u];
    }
    if (same && !movable.empty()) {
      std::vector<std::pair<double, Index>> far;
      far.reserve(movable.size());
      for (const Index j : movable) {
        const auto u = static_cast<std::size_t>(j);
        far.emplace_back(-std::fabs(x[u] - rounding[u]), j);
      }
      std::sort(far.begin(), far.end());
      std::uniform_int_distribution<int> draw(tol::kPumpFlips / 2, 3 * tol::kPumpFlips / 2);
      const auto flips = std::min(far.size(), static_cast<std::size_t>(draw(rng)));
      for (std::size_t k = 0; k < flips; ++k) flip(&rounding, far[k].second);
      ++result.perturbations;
    } else if (std::find(recent.begin(), recent.end(),
                         rounding_hash(integer_columns, rounding)) != recent.end()) {
      std::uniform_real_distribution<double> rho(tol::kPumpRestartLow, tol::kPumpRestartHigh);
      for (const Index j : movable) {
        const auto u = static_cast<std::size_t>(j);
        if (std::fabs(x[u] - rounding[u]) + std::max(rho(rng), 0.0) > 0.5) flip(&rounding, j);
      }
      ++result.perturbations;
    }
    const std::uint64_t hash = rounding_hash(integer_columns, rounding);

    // 3. Is the rounding a solution? Directly, and on a mixed model by completing the
    // continuous columns, when the rows can still hold and the budget allows.
    if (point_is_feasible(model, integer_columns, rounding, tol_int)) {
      result.x = std::move(rounding);
      result.stopped = "found";
      return result;
    }
    if (mixed && result.completions < tol::kPumpCompletionLimit &&
        std::find(completed.begin(), completed.end(), hash) == completed.end() &&
        rows_can_hold(model, is_integer, rounding)) {
      std::vector<double> point =
          complete(model, integer_columns, rounding, settings, budget, &result);
      if (!point.empty()) {
        result.x = std::move(point);
        result.stopped = "found by completing the continuous columns";
        return result;
      }
      completed.push_back(hash);
    }
    if (round >= settings.pump_rounds) {
      result.stopped = "round limit";
      return result;
    }
    previous = rounding;
    recent.push_back(hash);
    while (recent.size() > static_cast<std::size_t>(tol::kPumpCycleWindow)) recent.pop_front();

    // 4. Project the rounding back onto the LP polytope in the L1 distance, by PDHG.
    for (const Index j : integer_columns) {
      const auto u = static_cast<std::size_t>(j);
      const Index aux = kind[u];
      if (aux == -1) {
        lp.col_cost[u] = rounding[u] <= model.col_lower[u] ? 1.0 : -1.0;
      } else if (aux >= 0) {
        lp.row_lower[static_cast<std::size_t>(m + 2 * aux)] = -rounding[u];
        lp.row_lower[static_cast<std::size_t>(m + 2 * aux + 1)] = rounding[u];
      }
    }
    const Solution projected = solve_pdhg_for(lp, settings, device, budget);
    ++result.pdhg_solves;
    if (!has_point(projected, n)) {
      result.stopped = std::string("a projection returned no point: ") +
                       std::string(to_string(projected.status));
      return result;
    }
    x.assign(projected.col_value.begin(), projected.col_value.begin() + n);
  }
}

// ---- Fix-and-propagate --------------------------------------------------------------------
namespace {

/// The propagator fix-and-propagate calls: the device's when it may and a card answers, the
/// CPU reference otherwise - the same bounds either way (#510's bit-for-bit test).
class Propagator {
 public:
  Propagator(const Model& model, bool device, double integrality, PdhgHeuristicResult* result)
      : model_(model), device_(device), integrality_(integrality), result_(result) {}

  /// Tighten [lower, upper] in place; false when the box is proved empty (the vectors are
  /// then unspecified, and the caller restores them).
  bool run(std::vector<double>* lower, std::vector<double>* upper) {
    ++result_->propagations;
#ifdef SANKHYA_ENABLE_CUDA
    if (device_) {
      gpu::PropResult got = gpu::propagate_bounds(model_, *lower, *upper,
                                                  tol::kDomainPropagationRounds, integrality_);
      if (got.ran) {
        result_->on_device = true;
        if (got.infeasible) return false;
        *lower = std::move(got.col_lb);
        *upper = std::move(got.col_ub);
        return true;
      }
    }
#endif
    if (!rows_built_) {
      rows_ = row_major(model_);
      rows_built_ = true;
    }
    return !propagate_jacobi(model_, rows_, lower, upper, tol::kDomainPropagationRounds,
                             integrality_)
                .infeasible;
  }

 private:
  const Model& model_;
  [[maybe_unused]] bool device_;  // read only by the CUDA build
  double integrality_;
  PdhgHeuristicResult* result_;
  RowMajor rows_;
  bool rows_built_ = false;
};

struct BoundChange {
  Index column;
  double lower;
  double upper;
};

/// One committed batch of fixes: the columns and values, the bounds it and its propagation
/// changed (to undo it), and whether its single fix is already the alternative value.
struct Level {
  std::vector<std::pair<Index, double>> fixes;
  std::vector<BoundChange> trail;
  bool alternative = false;
};

}  // namespace

// THE SEARCH. Columns are fixed in batches: a batch that propagates to a non-empty box is
// committed and the next batch is twice as large; one that empties the box is undone and
// retried at half the size, so a device call is paid per batch rather than per column. A
// single column that empties the box at its nearest integer is tried at the other integer
// next to the relaxation's value; when that empties it too, the search backs up: the last
// committed level is undone and, if it was one column at its first value, that column takes
// its other value; a level of several columns is re-queued to be fixed one at a time. Each
// back-up spends one of `settings.backtracks`; the bound on them, and the halving before
// them, make the search finite. Every value is chosen inside the column's CURRENT domain,
// which propagation may have tightened past the relaxation's rounding.
PdhgHeuristicResult fix_and_propagate(const Model& model,
                                      const std::vector<Index>& integer_columns,
                                      const std::vector<double>& start,
                                      const PdhgHeuristicSettings& settings) {
  PdhgHeuristicResult result;
  const Budget budget(settings.seconds);
  const Index n = model.num_cols();
  const auto un = static_cast<std::size_t>(n);
  const double tol_int = settings.integrality_tolerance;
  if (integer_columns.empty()) {
    result.stopped = "no integer columns";
    return result;
  }
  const bool device = device_ready(settings);
  result.on_device = device;
  std::vector<double> x0 = start;
  if (static_cast<Index>(x0.size()) != n)
    x0 = relaxation_point(model, settings, device, budget, &result);
  if (static_cast<Index>(x0.size()) != n) {
    result.stopped = "no relaxation point";
    return result;
  }

  Propagator propagator(model, device, tol_int, &result);
  std::vector<double> lo = model.col_lower;
  std::vector<double> hi = model.col_upper;
  if (!propagator.run(&lo, &hi)) {
    result.stopped = "the model's box propagates empty";
    return result;
  }

  std::vector<std::pair<double, Index>> keyed;
  keyed.reserve(integer_columns.size());
  for (const Index j : integer_columns) {
    const double v = x0[static_cast<std::size_t>(j)];
    keyed.emplace_back(std::fabs(v - std::round(v)), j);
  }
  std::sort(keyed.begin(), keyed.end());  // least fractional first, ties by index

  const auto integer_lo = [&](Index j) {
    return integer_ceil_of(lo[static_cast<std::size_t>(j)], tol_int);
  };
  const auto integer_hi = [&](Index j) {
    return integer_floor_of(hi[static_cast<std::size_t>(j)], tol_int);
  };
  const auto is_fixed = [&](Index j) { return integer_lo(j) >= integer_hi(j); };
  const auto value_for = [&](Index j) {
    const auto u = static_cast<std::size_t>(j);
    return nearest_inside(x0[u], lo[u], hi[u], tol_int);
  };
  // The other integer next to the relaxation's value, inside the current domain.
  const auto alternative_for = [&](Index j, double v) -> std::optional<double> {
    const double step = x0[static_cast<std::size_t>(j)] > v ? 1.0 : -1.0;
    for (const double candidate : {v + step, v - step}) {
      if (candidate >= integer_lo(j) && candidate <= integer_hi(j)) return candidate;
    }
    return std::nullopt;
  };

  std::vector<Level> stack;
  const auto apply = [&](std::vector<std::pair<Index, double>> fixes, bool alternative) {
    const std::vector<double> lo_before = lo;
    const std::vector<double> hi_before = hi;
    for (const auto& [j, v] : fixes) {
      lo[static_cast<std::size_t>(j)] = v;
      hi[static_cast<std::size_t>(j)] = v;
    }
    ++result.rounds;
    if (!propagator.run(&lo, &hi)) {
      lo = lo_before;
      hi = hi_before;
      return false;
    }
    Level level;
    level.fixes = std::move(fixes);
    level.alternative = alternative;
    for (std::size_t u = 0; u < un; ++u) {
      if (lo[u] != lo_before[u] || hi[u] != hi_before[u]) {
        level.trail.push_back({static_cast<Index>(u), lo_before[u], hi_before[u]});
      }
    }
    stack.push_back(std::move(level));
    return true;
  };
  const auto undo = [&]() {
    Level level = std::move(stack.back());
    stack.pop_back();
    for (const BoundChange& c : level.trail) {
      lo[static_cast<std::size_t>(c.column)] = c.lower;
      hi[static_cast<std::size_t>(c.column)] = c.upper;
    }
    return level;
  };

  std::vector<Index> pending;  // LIFO: columns to fix before the order continues
  std::size_t next = 0;
  std::size_t batch = 1;
  bool all_fixed = false;
  while (true) {
    if (should_stop(settings, budget)) {
      result.stopped = "stopped by the interrupt or the time limit";
      return result;
    }
    std::vector<std::pair<Index, double>> fixes;
    while (fixes.size() < batch) {
      Index j = -1;
      if (!pending.empty()) {
        j = pending.back();
        pending.pop_back();
      } else if (next < keyed.size()) {
        j = keyed[next++].second;
      } else {
        break;
      }
      if (!is_fixed(j)) fixes.emplace_back(j, value_for(j));
    }
    if (fixes.empty()) {
      all_fixed = true;
      break;
    }
    const std::size_t tried = fixes.size();
    if (apply(fixes, false)) {
      batch = 2 * tried;
      continue;
    }
    if (tried > 1) {
      for (auto it = fixes.rbegin(); it != fixes.rend(); ++it) pending.push_back(it->first);
      batch = tried / 2;
      continue;
    }
    const Index j = fixes.front().first;
    if (const auto other = alternative_for(j, fixes.front().second);
        other.has_value() && apply({{j, *other}}, true)) {
      batch = 1;
      continue;
    }
    // Both integers next to the relaxation empty the box: back up.
    pending.push_back(j);
    bool resumed = false;
    while (!stack.empty() && result.backtracks < settings.backtracks) {
      ++result.backtracks;
      const Level level = undo();
      if (level.fixes.size() == 1 && !level.alternative) {
        const Index k = level.fixes.front().first;
        if (const auto other = alternative_for(k, level.fixes.front().second);
            other.has_value() && apply({{k, *other}}, true)) {
          resumed = true;
          break;
        }
        pending.push_back(k);  // neither value of k holds here either: back up further
        continue;
      }
      for (auto it = level.fixes.rbegin(); it != level.fixes.rend(); ++it) {
        pending.push_back(it->first);
      }
      resumed = true;
      break;
    }
    if (!resumed) {
      result.stopped = stack.empty() ? "no committed fix left to undo" : "back-up budget spent";
      break;
    }
    batch = 1;
  }

  // The point the fixes describe: each integer column at its fixed value (or, when the search
  // gave up, nearest the relaxation inside its current domain); continuous columns as the
  // relaxation has them, inside their box.
  std::vector<double> point(un);
  for (std::size_t u = 0; u < un; ++u) {
    point[u] = clamp_inside(x0[u], model.col_lower[u], model.col_upper[u]);
  }
  for (const Index j : integer_columns) {
    point[static_cast<std::size_t>(j)] = is_fixed(j) ? integer_lo(j) : value_for(j);
  }
  if (all_fixed) {
    if (point_is_feasible(model, integer_columns, point, tol_int)) {
      result.x = std::move(point);
      result.stopped = "found";
      return result;
    }
    if (static_cast<Index>(integer_columns.size()) < n) {
      std::vector<double> completed =
          complete(model, integer_columns, point, settings, budget, &result);
      if (!completed.empty()) {
        result.x = std::move(completed);
        result.stopped = "found by completing the continuous columns";
        return result;
      }
    }
    result.stopped = "every integer column fixed, but the point is not feasible";
  }
  // The repair (Corduk et al.): Feasibility Jump from the point, its own points re-checked.
  if (settings.repair_work <= 0 || should_stop(settings, budget)) return result;
  FeasibilityJumpSettings jump;
  jump.work_limit = settings.repair_work;
  jump.seed = settings.seed;
  jump.integrality_tolerance = tol_int;
  jump.should_stop = [&settings, &budget]() { return should_stop(settings, budget); };
  const FeasibilityJumpResult repaired = feasibility_jump(model, point, jump);
  for (auto it = repaired.points.rbegin(); it != repaired.points.rend(); ++it) {
    if (point_is_feasible(model, integer_columns, *it, tol_int)) {
      result.x = *it;
      result.repaired = true;
      result.stopped += "; repaired by feasibility jump";
      return result;
    }
  }
  result.stopped += "; the repair found nothing";
  return result;
}

}  // namespace sankhya::mip
