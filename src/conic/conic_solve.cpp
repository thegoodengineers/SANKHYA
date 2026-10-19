// SPDX-License-Identifier: Apache-2.0
// SANKHYA - solve_conic() (#755): classify the quadratic rows, build the conic form, run the
// interior point (conic_ipm.cpp) and read the answer back in the model's own terms.
//
// WHAT IS REPORTED, in the model's sense. A quadratic row's multiplier is the multiplier of
// its gradient a_r + 2 Q_r x in the Lagrangian, which is what the cone's dual reduces to: on
// the rotated cone 2 w v >= ||z||^2 the dual at the boundary is a multiple beta of (v, w, -z),
// and with v = 1/2 and z = F'x the stationarity c - A'y - y_w (a_r + 2 F F'x) = 0 follows with
// y_w = -beta/2 (conic_form.hpp). Reduced costs are c + Hx - sum_i y_i grad_i, which for a
// column of an explicit cone is the cone's dual variable s, so tools/verify_solution.py can
// check x in K, s in K* and x's = 0 from the .sol file alone.

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "conic/conic_form.hpp"
#include "conic/conic_ipm.hpp"
#include "qp/convexity.hpp"
#include "qp/qp_ipm_system.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/qcqp.hpp"
#include "sankhya/solve_control.hpp"
#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya {
namespace {

Solution refused(const std::string& why, const Timer& timer) {
  Solution s;
  s.status = SolveStatus::kNotSolved;
  s.message = why;
  s.algorithm = "conic-ipm";
  s.solve_seconds = timer.elapsed_seconds();
  return s;
}

/// The refusal a non-convex row earns: its verdict, and the witness direction's largest
/// entries so the certificate is in the message the user reads.
std::string describe_refusal(const QcqpModel& model, const QuadraticRowClass& c) {
  std::string text = fmt::format("quadratic row {} is not convex ({})", c.row, c.detail);
  if (!c.witness.empty()) {
    std::vector<std::pair<double, Index>> entries;
    for (std::size_t j = 0; j < c.witness.size(); ++j) {
      if (c.witness[j] != 0.0)
        entries.emplace_back(std::fabs(c.witness[j]), static_cast<Index>(j));
    }
    std::sort(entries.rbegin(), entries.rend());
    text += "; witness direction z with z'Q_r z < 0:";
    for (std::size_t k = 0; k < std::min<std::size_t>(entries.size(), 4); ++k) {
      const Index j = entries[k].second;
      const auto uj = static_cast<std::size_t>(j);
      text +=
          fmt::format(" {} = {:.4g}",
                      uj < model.linear.col_names.size() && !model.linear.col_names[uj].empty()
                          ? model.linear.col_names[uj]
                          : fmt::format("x{}", j),
                      c.witness[uj]);
    }
  }
  return text + "; --option nonconvex=global solves it by spatial branch and bound (#514)";
}

/// How far x is from the cones of `model`, as an absolute violation.
double cone_violation(const QcqpModel& model, const std::vector<double>& x) {
  double worst = 0.0;
  for (const Cone& cone : model.cones) {
    const auto at = [&](std::size_t i) { return x[static_cast<std::size_t>(cone.columns[i])]; };
    double tail = 0.0;
    const std::size_t head = cone.kind == Cone::Kind::kRotated ? 2 : 1;
    for (std::size_t i = head; i < cone.columns.size(); ++i) tail += at(i) * at(i);
    if (cone.kind == Cone::Kind::kRotated) {
      worst = std::max({worst, -at(0), -at(1), tail - 2.0 * at(0) * at(1)});
    } else {
      worst = std::max(worst, std::sqrt(tail) - at(0));
    }
  }
  return worst;
}

}  // namespace

Solution solve_conic(const QcqpModel& model, const Options& options, SolveControl* control) {
  const Timer timer;
  Logger logger(options.get_bool("log_to_console") ? stdout : nullptr);
  const std::string problem = model.validate();
  if (!problem.empty()) return refused("model failed validation: " + problem, timer);
  const Model& lin = model.linear;
  if (lin.has_integrality()) {
    return refused(
        "integer columns with quadratic rows or cones: the conic interior point solves the "
        "continuous problem only (MIQCP nodes are not built yet, #755)",
        timer);
  }
  const std::vector<QuadraticRowClass> classes = classify_quadratic_rows(model);
  for (const QuadraticRowClass& c : classes) {
    if (c.verdict != QuadraticRowClass::Verdict::kConvex) {
      return refused(describe_refusal(model, c), timer);
    }
  }
  if (lin.has_quadratic_objective()) {
    const qp::ConvexityResult objective = qp::check_convexity(lin);
    if (objective.verdict != qp::Convexity::kConvex) {
      return refused("the quadratic objective is not convex: " + objective.detail, timer);
    }
  }
  const conic::ConicForm form = conic::build_conic_form(model, classes);
  if (!form.problem.empty()) return refused(form.problem, timer);
  logger.info(
      "conic interior point: {} rows, {} columns ({} free, {} non-negative, {} cone(s)) in "
      "the standard form, from {} quadratic row(s) and {} explicit cone(s)",
      form.m, form.n, form.free_n, form.nonneg_n, form.cones.size(), classes.size(),
      model.cones.size());

  const conic::HsdResult run = conic::solve_hsd(form, options, logger, control, timer);
  Solution solution;
  solution.status = run.status;
  solution.stopped_by = run.stopped_by;
  solution.message = run.message;
  solution.iterations = run.iterations;
  solution.algorithm = "conic-ipm";
  solution.solve_seconds = timer.elapsed_seconds();
  const double sigma = lin.sense_multiplier();
  solution.dual_bound = sigma * -kInfinity;
  if (run.status == SolveStatus::kInfeasible) {
    solution.dual_bound = sigma * kInfinity;
    solution.message +=
        " (the ray is on the conic standard form; no Farkas vector over the "
        "model's rows is written)";
    return solution;
  }
  if (run.status != SolveStatus::kOptimal) {
    if (run.status == SolveStatus::kUnbounded) {
      solution.message +=
          " (the ray is on the conic standard form; no primal ray over the "
          "model's columns is written)";
    }
    return solution;
  }

  // ---- the point and the multipliers, in the model's terms ------------------------------
  const Index n = lin.num_cols();
  const Index m = lin.num_rows();
  solution.col_value.assign(static_cast<std::size_t>(n), 0.0);
  for (Index j = 0; j < n; ++j) {
    const conic::ColumnMap& map = form.col_map[static_cast<std::size_t>(j)];
    double value = map.shift;
    if (map.a >= 0) value += map.a_scale * run.x[static_cast<std::size_t>(map.a)];
    if (map.b >= 0) value += map.b_scale * run.x[static_cast<std::size_t>(map.b)];
    solution.col_value[static_cast<std::size_t>(j)] = value;
  }
  const std::vector<double>& x = solution.col_value;
  solution.row_activity = model.row_activity(x);
  solution.objective = model.objective(x);
  solution.dual_bound = solution.objective;
  solution.row_dual.assign(static_cast<std::size_t>(m), 0.0);
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const Index r = form.row_of[u];
    if (r >= 0)
      solution.row_dual[u] = sigma * form.row_sign[u] * run.y[static_cast<std::size_t>(r)];
  }
  // c + Hx - A'y - sum over quadratic rows of y_r grad(x'Q_r x), the model's own sense.
  std::vector<double> d(static_cast<std::size_t>(n), 0.0);
  if (lin.has_quadratic_objective()) qp::ipm_detail::hessian_times(lin.hessian, x, &d);
  for (Index j = 0; j < n; ++j)
    d[static_cast<std::size_t>(j)] += lin.col_cost[static_cast<std::size_t>(j)];
  lin.matrix.transpose_multiply_add(solution.row_dual.data(), d.data(), -1.0);
  for (const QuadraticTerm& t : model.quadratic) {
    const double y = solution.row_dual[static_cast<std::size_t>(t.row)];
    const auto a = static_cast<std::size_t>(t.first);
    const auto b = static_cast<std::size_t>(t.second);
    if (a == b) {
      d[a] -= y * 2.0 * t.value * x[a];
    } else {
      d[a] -= y * t.value * x[b];
      d[b] -= y * t.value * x[a];
    }
  }
  solution.col_dual = std::move(d);

  // ---- quality, measured on the model with its products and cones -------------------------
  double primal = 0.0;
  double scale = 1.0;
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double act = solution.row_activity[u];
    primal = std::max({primal, lin.row_lower[u] - act, act - lin.row_upper[u]});
    scale = std::max(scale, std::fabs(act));
  }
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    primal = std::max({primal, lin.col_lower[u] - x[u], x[u] - lin.col_upper[u]});
    scale = std::max(scale, std::fabs(x[u]));
  }
  primal = std::max(primal, cone_violation(model, x));
  solution.primal_infeasibility = primal;
  solution.primal_infeasibility_scaled = primal / scale;
  double dual = 0.0;
  std::vector<char> in_cone(static_cast<std::size_t>(n), 0);
  for (const Cone& cone : model.cones) {
    for (const Index j : cone.columns) in_cone[static_cast<std::size_t>(j)] = 1;
  }
  double dual_scale = 1.0;
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double dj = sigma * solution.col_dual[u];
    dual_scale = std::max(dual_scale, std::fabs(dj));
    if (in_cone[u]) continue;
    if (dj > 0.0 && !is_finite_bound(lin.col_lower[u])) dual = std::max(dual, dj);
    if (dj < 0.0 && !is_finite_bound(lin.col_upper[u])) dual = std::max(dual, -dj);
  }
  {
    // The reduced costs of a cone's columns must lie in the (self-)dual cone.
    QcqpModel probe;
    probe.cones = model.cones;
    std::vector<double> ds(static_cast<std::size_t>(n));
    for (std::size_t u = 0; u < ds.size(); ++u) ds[u] = sigma * solution.col_dual[u];
    dual = std::max(dual, cone_violation(probe, ds));
  }
  solution.dual_infeasibility = dual;
  solution.dual_infeasibility_scaled = dual / dual_scale;
  solution.absolute_gap = 0.0;
  solution.relative_gap = 0.0;
  solution.message = fmt::format(
      "optimal: relative residuals {:.2e} primal, {:.2e} dual, {:.2e} gap on the conic form",
      run.primal_residual, run.dual_residual, run.gap);
  logger.info("conic interior point: {} iterations, {}", run.iterations, solution.message);
  return solution;
}

}  // namespace sankhya
