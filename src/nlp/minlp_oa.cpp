// SPDX-License-Identifier: Apache-2.0
// SANKHYA - convex MINLP by outer approximation (#528). See minlp_oa.hpp for the method
// (Duran and Grossmann 1986), the three kinds of cut and what `optimal` means.

#include "nlp/minlp_oa.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "nlp/minlp_bnb.hpp"
#include "nlp/nlp_problem.hpp"
#include "nlp/nlp_solve.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::nlp {
namespace {

/// One linear row of the master: lower <= sum values[k] x[columns[k]] <= upper. Column index
/// num_cols is the objective's epigraph column.
struct Cut {
  std::vector<Index> columns;
  std::vector<double> values;
  double lower = -kInfinity;
  double upper = kInfinity;
};

struct Linearization {
  std::vector<Cut> cuts;
  bool failed = false;  ///< the point could not be evaluated (a domain error, an overflow)
};

bool is_point(SolveStatus s) {
  return s == SolveStatus::kOptimal || s == SolveStatus::kLocallyOptimal;
}

bool is_infeasible(SolveStatus s) {
  return s == SolveStatus::kInfeasible || s == SolveStatus::kLocallyInfeasible;
}

/// The linearizations of the nonlinear rows - and, with `epigraph`, of the objective - at `x`.
/// With `all`, every finite side of every nonlinear row and the objective; otherwise only what
/// `x` violates by more than kMinlpOaViolation (the objective against `t_hat`, the epigraph
/// column's value at x). The linear rows are in the master already and are not cut.
Linearization linearize(const NlpProblem& problem, bool epigraph, const std::vector<double>& x,
                        bool all, double t_hat) {
  Linearization out;
  const Index n = problem.num_variables();
  Evaluation error;
  std::vector<double> g, jacobian;
  if (!problem.constraints(x, &g, &error) || !problem.jacobian(x, &jacobian, &error)) {
    out.failed = true;
    return out;
  }
  const auto exceeds = [](double violation, double bound) {
    return violation > tol::kMinlpOaViolation * std::max(1.0, std::fabs(bound));
  };
  const std::vector<Index>& starts = problem.jacobian_starts();
  const std::vector<Index>& columns = problem.jacobian_columns();
  for (Index row = problem.num_linear_rows(); row < problem.num_constraints(); ++row) {
    const auto r = static_cast<std::size_t>(row);
    const double value = g[r];
    if (!std::isfinite(value)) {
      out.failed = true;
      return out;
    }
    const double lo = problem.g_lower()[r];
    const double up = problem.g_upper()[r];
    const bool cut_up = std::isfinite(up) && (all || exceeds(value - up, up));
    const bool cut_lo = std::isfinite(lo) && (all || exceeds(lo - value, lo));
    if (!cut_up && !cut_lo) continue;
    Cut cut;
    double at_x = 0.0;
    for (Index q = starts[r]; q < starts[r + 1]; ++q) {
      const double d = jacobian[static_cast<std::size_t>(q)];
      if (!std::isfinite(d)) {
        out.failed = true;
        return out;
      }
      if (d == 0.0) continue;
      const Index j = columns[static_cast<std::size_t>(q)];
      cut.columns.push_back(j);
      cut.values.push_back(d);
      at_x += d * x[static_cast<std::size_t>(j)];
    }
    // g(x) ~ value + d'(x - x_k): a convex g stays above it, a concave one below it.
    if (cut_up) cut.upper = up - value + at_x;
    if (cut_lo) cut.lower = lo - value + at_x;
    out.cuts.push_back(std::move(cut));
  }
  if (epigraph) {
    double f = 0.0;
    std::vector<double> gradient;
    if (!problem.objective_gradient(x, &f, &gradient, &error) || !std::isfinite(f)) {
      out.failed = true;
      return out;
    }
    if (all || exceeds(f - t_hat, f)) {
      Cut cut;
      double at_x = 0.0;
      for (Index j = 0; j < n; ++j) {
        const double d = gradient[static_cast<std::size_t>(j)];
        if (!std::isfinite(d)) {
          out.failed = true;
          return out;
        }
        if (d == 0.0) continue;
        cut.columns.push_back(j);
        cut.values.push_back(d);
        at_x += d * x[static_cast<std::size_t>(j)];
      }
      cut.columns.push_back(n);  // - t
      cut.values.push_back(-1.0);
      cut.upper = at_x - f;  // f(x_k) + grad'(x - x_k) <= t
      out.cuts.push_back(std::move(cut));
    }
  }
  return out;
}

/// The master MILP: the model's columns (and an epigraph column when the objective is not
/// linear), its linear rows, and every cut so far, minimising the objective in the
/// minimisation form (sense = +1 minimise, -1 maximise).
Model build_master(const Model& base, bool epigraph, double epigraph_lower, double sense,
                   const std::vector<Cut>& cuts) {
  const Index n = base.num_cols();
  const Index rows = base.num_rows();
  const Index columns = n + (epigraph ? 1 : 0);
  const Index total = rows + static_cast<Index>(cuts.size());
  Model master = base;
  master.col_names.clear();
  master.row_names.clear();
  master.hessian = SparseMatrix();
  master.sense = ObjSense::kMinimize;
  master.resize_columns(columns);
  master.resize_rows(total);
  std::fill(master.col_cost.begin(), master.col_cost.end(), 0.0);
  master.objective_offset = 0.0;
  if (epigraph) {
    const auto t = static_cast<std::size_t>(n);
    master.col_cost[t] = 1.0;
    master.col_lower[t] = epigraph_lower;
    master.col_upper[t] = kInfinity;
    master.col_type[t] = VarType::kContinuous;
  } else {
    for (Index j = 0; j < n; ++j) {
      const auto u = static_cast<std::size_t>(j);
      master.col_cost[u] = sense * base.col_cost[u];
    }
    master.objective_offset = sense * base.objective_offset;
  }
  SparseMatrix matrix(total, columns);
  for (Index j = 0; j < n; ++j) {
    const ColumnView column = base.matrix.column(j);
    for (Index k = 0; k < column.size; ++k)
      matrix.add_entry(column.rows[k], j, column.values[k]);
  }
  for (std::size_t k = 0; k < cuts.size(); ++k) {
    const Index row = rows + static_cast<Index>(k);
    for (std::size_t q = 0; q < cuts[k].columns.size(); ++q)
      matrix.add_entry(row, cuts[k].columns[q], cuts[k].values[q]);
    master.row_lower[static_cast<std::size_t>(row)] = cuts[k].lower;
    master.row_upper[static_cast<std::size_t>(row)] = cuts[k].upper;
  }
  matrix.finalize(0.0);  // a cut's small coefficient on a wide column is not rounding
  master.matrix = std::move(matrix);
  return master;
}

}  // namespace

Solution solve_minlp_oa(const NonlinearModel& model, const Options& options,
                        SolveControl* control, bool assumed) {
  const Timer timer;
  Logger logger(options.get_bool("log_to_console") ? stdout : nullptr);
  LogLevel level = LogLevel::kInfo;
  if (parse_log_level(options.get_string("log_level"), &level)) logger.set_level(level);
  Logger silent(nullptr);
  Solution out;
  out.algorithm = "minlp-oa";

  NlpProblem problem;
  const std::string built = NlpProblem::build(model, &problem);
  if (!built.empty()) {
    out.status = SolveStatus::kModelError;
    out.message = built;
    return out;
  }
  const double sense = model.base.sense == ObjSense::kMaximize ? -1.0 : 1.0;
  const Index n = model.base.num_cols();
  const bool epigraph = model.objective != kNoExpr || model.base.has_quadratic_objective();
  const double int_tol = options.get_double("integrality_tolerance");
  const double rel_target = options.get_double("mip_relative_gap");
  const double abs_target = options.get_double("mip_absolute_gap");
  const std::int64_t node_limit = options.get_int("node_limit");
  const std::int64_t max_rounds = options.get_int("minlp_oa_max_iterations");
  const double time_limit = options.get_double("time_limit");
  const auto remaining = [&] { return std::max(0.0, time_limit - timer.elapsed_seconds()); };

  Options quiet = options;
  quiet.set_bool("log_to_console", false);
  Options master_options = quiet;
  master_options.set_int("node_limit", -1);  // the MINLP's node limit is counted below

  std::vector<Index> integers;
  for (Index j = 0; j < n; ++j) {
    if (model.base.col_type[static_cast<std::size_t>(j)] == VarType::kInteger)
      integers.push_back(j);
  }
  NonlinearModel work = model;
  for (const Index j : integers) {
    const auto u = static_cast<std::size_t>(j);
    if (std::isfinite(work.base.col_lower[u]))
      work.base.col_lower[u] = std::ceil(work.base.col_lower[u] - int_tol);
    if (std::isfinite(work.base.col_upper[u]))
      work.base.col_upper[u] = std::floor(work.base.col_upper[u] + int_tol);
  }
  const std::vector<double> box_lower = work.base.col_lower;
  const std::vector<double> box_upper = work.base.col_upper;

  Count nlp_iterations = 0;
  Count nlp_solves = 0;
  const auto nlp = [&](const NonlinearModel& m, const std::vector<double>& start) {
    quiet.set_double("time_limit", remaining());
    ++nlp_solves;
    Solution s = solve_nlp_relaxation(m, quiet, start, control, silent);
    nlp_iterations += s.iterations;
    return s;
  };
  // The least total violation of any point in `m`'s box, and where; `certified` when it is a
  // global optimum of the convex problem and positive, which proves the box empty.
  struct Violation {
    bool certified = false;
    std::vector<double> point;
  };
  const auto minimum_violation = [&](const NonlinearModel& m) {
    const Solution v = nlp(violation_model(m), {});
    Violation r;
    if (v.status == SolveStatus::kOptimal &&
        v.col_value.size() >= static_cast<std::size_t>(n)) {
      r.certified = v.objective > tol::kMinlpEmptyViolation;
      r.point.assign(v.col_value.begin(), v.col_value.begin() + n);
    }
    return r;
  };

  // ---- the continuous relaxation: where the master starts, and its bound -----------------
  const Solution root = nlp(work, model.start);
  const auto early = [&](SolveStatus status, const std::string& why) {
    Solution r;
    r.algorithm = "minlp-oa";
    r.status = status;
    r.message = why;
    r.nodes = 0;
    r.iterations = nlp_iterations;
    r.solve_seconds = timer.elapsed_seconds();
    return r;
  };
  if (root.status == SolveStatus::kTimeLimit || root.status == SolveStatus::kInterrupted) {
    Solution r = early(root.status, "stopped before the continuous relaxation was solved");
    r.stopped_by =
        root.status == SolveStatus::kTimeLimit ? LimitReason::kTime : LimitReason::kInterrupt;
    return r;
  }
  if (root.status == SolveStatus::kUnbounded) {
    return early(
        SolveStatus::kUnbounded,
        "the continuous relaxation is unbounded, so the MINLP is unbounded or infeasible");
  }
  if (is_infeasible(root.status)) {
    if (!assumed && minimum_violation(work).certified) {
      return early(SolveStatus::kInfeasible,
                   "the continuous relaxation is certified empty (its least total violation is "
                   "positive), so the MINLP is infeasible");
    }
    return early(SolveStatus::kNotSolved,
                 "the continuous relaxation is infeasible and that could not be certified");
  }
  if (!is_point(root.status) || (!assumed && root.status != SolveStatus::kOptimal)) {
    return early(SolveStatus::kNumericalError,
                 fmt::format("the continuous relaxation ended {} ({}), so there is no valid "
                             "bound to start the master from",
                             to_string(root.status), root.message));
  }
  const double root_value = sense * root.objective;
  const double epigraph_lower =
      root_value - tol::kMinlpOaEpigraphSlack * std::max(1.0, std::fabs(root_value));

  // ---- the alternation ------------------------------------------------------------------
  std::vector<Cut> cuts;
  std::set<std::vector<double>> cut_points;   // points cut at in full
  std::set<std::vector<double>> assignments;  // integer assignments whose NLP was solved
  const auto add_cuts = [&](Linearization l) {
    for (Cut& cut : l.cuts) cuts.push_back(std::move(cut));
    return l.cuts.size();
  };
  const auto cut_in_full = [&](const std::vector<double>& x) {
    if (!cut_points.insert(x).second) return std::size_t{0};
    const Linearization l = linearize(problem, epigraph, x, true, 0.0);
    if (l.failed) return std::size_t{0};
    return add_cuts(l);
  };
  cut_in_full(root.col_value);

  double lower = root_value - tol::kMinlpOaEpigraphSlack * std::max(1.0, std::fabs(root_value));
  double incumbent = kInfinity;  // minimisation form
  Solution best;
  bool closed = false;
  bool inconsistent = false;
  bool master_infeasible = false;
  std::string lost_because;
  const auto lose = [&](const std::string& why) {
    if (lost_because.empty()) lost_because = why;
  };
  LimitReason stopped = LimitReason::kNone;
  Count rounds = 0;
  Count masters = 0;
  Count master_nodes = 0;

  const auto gap_closed = [&] {
    if (!std::isfinite(incumbent)) return false;
    const double gap = incumbent - lower;
    return gap <= abs_target || gap / std::max(1.0, std::fabs(incumbent)) <= rel_target;
  };

  while (true) {
    if (gap_closed()) {
      closed = true;
      break;
    }
    if (control != nullptr && control->interruption_requested()) {
      stopped = LimitReason::kInterrupt;
      break;
    }
    if (timer.elapsed_seconds() > time_limit) {
      stopped = LimitReason::kTime;
      break;
    }
    if (node_limit >= 0 && master_nodes >= node_limit) {
      stopped = LimitReason::kNodes;
      break;
    }
    if (rounds >= max_rounds) {
      stopped = LimitReason::kIterations;
      break;
    }
    ++rounds;

    // The master over every cut so far.
    master_options.set_double("time_limit", remaining());
    const Model master = build_master(model.base, epigraph, epigraph_lower, sense, cuts);
    const Solution ms = solve(master, master_options, control);
    ++masters;
    master_nodes += ms.nodes;
    // Only a master solved to `optimal` has a bound that means something: an unfinished one
    // reports whatever its tree had proved, or a default, and a default is not a bound.
    if (ms.status == SolveStatus::kOptimal && std::isfinite(ms.dual_bound)) {
      lower = std::max(lower, ms.dual_bound);
    }
    if (ms.status == SolveStatus::kInfeasible) {
      // Every cut is valid, so an infeasible master is an infeasible MINLP - and an incumbent
      // satisfies every cut, so one beside an infeasible master is a cut that was not valid.
      if (std::isfinite(incumbent)) {
        inconsistent = true;
        lose("the master was infeasible although a feasible point is known");
      } else if (assumed) {
        lose("the master is infeasible, but convexity was only asserted, not proved");
      } else {
        master_infeasible = true;
      }
      break;
    }
    if (ms.status != SolveStatus::kOptimal) {
      if (ms.stopped_by != LimitReason::kNone) {
        stopped = ms.stopped_by;
      } else {
        lose(fmt::format("the master ended {} ({})", to_string(ms.status), ms.message));
      }
      break;
    }
    if (ms.col_value.size() < static_cast<std::size_t>(n + (epigraph ? 1 : 0))) {
      lose("the master reported no point");
      break;
    }
    if (std::isfinite(incumbent) &&
        lower > incumbent +
                    std::max(abs_target, rel_target * std::max(1.0, std::fabs(incumbent)))) {
      // A bound above a verified feasible value, past the gap target: a cut cut off a feasible
      // point. Neither is believed.
      inconsistent = true;
      lose(fmt::format("the master's bound {:.10g} exceeds a verified feasible value {:.10g}",
                       lower, incumbent));
      break;
    }
    logger.verbose("OA round {}: master bound {:.10g}, incumbent {:.10g}, {} cut(s)", rounds,
                   lower, incumbent, cuts.size());
    if (gap_closed()) {
      closed = true;
      break;
    }

    const std::vector<double> x_hat(ms.col_value.begin(), ms.col_value.begin() + n);
    const double t_hat = epigraph ? ms.col_value[static_cast<std::size_t>(n)] : 0.0;
    std::vector<double> assignment;
    NonlinearModel fixed = work;
    for (const Index j : integers) {
      const auto u = static_cast<std::size_t>(j);
      const double v = std::min(std::max(std::round(x_hat[u]), box_lower[u]), box_upper[u]);
      assignment.push_back(v);
      fixed.base.col_lower[u] = fixed.base.col_upper[u] = v;
    }

    std::size_t added = 0;
    if (assignments.insert(assignment).second) {
      const Solution r = nlp(fixed, x_hat);
      if (r.status == SolveStatus::kTimeLimit || r.status == SolveStatus::kInterrupted) {
        stopped =
            r.status == SolveStatus::kTimeLimit ? LimitReason::kTime : LimitReason::kInterrupt;
        break;
      }
      if (is_point(r.status) && r.col_value.size() == static_cast<std::size_t>(n)) {
        const double value = sense * r.objective;
        if (value < incumbent) {
          incumbent = value;
          best = r;
          logger.verbose("OA round {}: incumbent {:.10g}", rounds, r.objective);
        }
        added += cut_in_full(r.col_value);
      } else if (is_infeasible(r.status)) {
        // Cut this assignment off with the linearizations at the least-violating point.
        const Violation v = minimum_violation(fixed);
        if (v.certified) {
          added += cut_in_full(v.point);
        } else {
          lose("an integer assignment's NLP was infeasible and that could not be certified");
        }
      } else {
        lose(fmt::format("an integer assignment's NLP ended {} ({})", to_string(r.status),
                         r.message));
      }
    }
    // The master's own point, for whatever it violates: the round makes progress even when the
    // NLP above gave no usable point.
    const Linearization at_master = linearize(problem, epigraph, x_hat, false, t_hat);
    if (at_master.failed) {
      lose("the master's point could not be evaluated");
    } else {
      added += add_cuts(at_master);
    }
    if (added == 0) {
      lose("a round added no cut: the master's point is feasible for every nonlinear row");
      break;
    }
  }

  // ---- the answer ----------------------------------------------------------------------
  const bool have = std::isfinite(incumbent);
  const bool proved = !assumed && !inconsistent;
  out = have ? best : Solution{};
  out.algorithm = "minlp-oa";
  out.nodes = master_nodes;
  out.iterations = nlp_iterations;
  out.stopped_by = stopped;
  const std::string effort =
      fmt::format("{} round(s), {} master solve(s), {} NLP solve(s), {} cut(s)", rounds,
                  masters, nlp_solves, cuts.size());
  if (!have) {
    out.objective = sense * kInfinity;
    out.dual_bound = proved ? sense * lower : sense * -kInfinity;
  } else {
    out.objective = sense * incumbent;
    out.dual_bound = proved ? sense * std::min(lower, incumbent) : sense * -kInfinity;
    out.absolute_gap = proved ? std::max(0.0, incumbent - lower) : kInfinity;
    out.relative_gap =
        proved ? out.absolute_gap / std::max(1.0, std::fabs(incumbent)) : kInfinity;
  }
  if (stopped != LimitReason::kNone) {
    out.status = have ? SolveStatus::kFeasible : status_for(stopped);
    out.message = fmt::format("stopped by the {} after {}{}", to_string(stopped), effort,
                              have ? "; the best integer point found is reported" : "");
  } else if (master_infeasible && proved) {
    out = Solution{};
    out.algorithm = "minlp-oa";
    out.nodes = master_nodes;
    out.iterations = nlp_iterations;
    out.status = SolveStatus::kInfeasible;
    out.message = fmt::format(
        "the master over the linearizations of a convex model is infeasible, so the MINLP is "
        "infeasible ({})",
        effort);
  } else if (closed && have && proved) {
    out.status = SolveStatus::kOptimal;
    out.message = fmt::format(
        "a convex MINLP: the incumbent is within the gap target of the master's bound after "
        "outer approximation ({})",
        effort);
  } else if (have) {
    out.status = SolveStatus::kFeasible;
    out.message =
        assumed
            ? "convexity asserted (nlp_assume_convex), not proved: the best integer point "
              "found, with no optimality claim and no bound"
            : "the best integer point found; not proved optimal because " +
                  (lost_because.empty() ? std::string("the gap did not close") : lost_because);
  } else {
    out.status = SolveStatus::kNumericalError;
    out.message = "no integer point was found: " +
                  (lost_because.empty() ? std::string("the gap did not close") : lost_because);
  }
  out.solve_seconds = timer.elapsed_seconds();
  logger.info("MINLP outer approximation: {} - {}", to_string(out.status), out.message);
  return out;
}

}  // namespace sankhya::nlp
