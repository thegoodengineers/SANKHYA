// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the CUDA PDHG convergence evaluation on the device against the host reference
// (#478 item 3).
//
// pdhg::evaluate on the host is the reference; src/gpu/pdhg_device_eval.cu forms the same
// residuals on the card from the SCALED products, A x = Dr^-1 (Ahat xhat) and
// A^T y = Dc^-1 (Ahat^T yhat). In exact arithmetic the two are the same number, so the test
// holds them to a rounding bound, not to a fitted tolerance. For a dot product of k terms,
// |fl(a.x) - a.x| <= gamma_k |a|.|x| with gamma_k = k u / (1 - k u), u = 2^-53 (Higham,
// "Accuracy and Stability of Numerical Algorithms", 2nd ed., SIAM 2002, eq. 3.5). The host
// takes a k-term product on A and x_u; the device takes one on Ahat and xhat and divides by
// r_i, where every entry of Ahat carries the roundings of the scaling (src/la/scaling.cpp:
// 11 passes, each rounding the entry twice and r_i and c_j once: 44) - so the two row
// activities differ by at most
//     delta_i = gamma_{2 k_i + kScalingRoundings} * sum_j |a_ij x_j|
// and the reduced costs likewise with |a_ij y_i| and |c_j|. Every residual is 1-Lipschitz in
// the activities or the reduced costs, which carries delta into a bound for each field; the
// sums of identical terms in a different order add gamma_(terms) of their magnitude.
//
// Points: the starting point, a random point (random restart reference), a basic optimal
// point from the simplex (the residuals there are cancellations, the hard case), and the
// average path (a running sum divided on the device). The end-to-end half solves with the
// device evaluation and with the host one and asks for the same answer at the stopping
// tolerance; bitwise repeatability with the device evaluation on is
// test_pdhg_cuda_determinism.cpp, which now also asserts that this path ran.
//
// Skipped, not passed, without CUDA in the build or a device in the machine.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/types.hpp"

#ifdef SANKHYA_ENABLE_CUDA
#include "gpu/pdhg_device_eval.hpp"
#include "la/scaling.hpp"
#include "pdhg/pdhg_evaluate.hpp"
#endif

namespace sankhya {
namespace {

std::string repository_path(const std::string& relative) {
  return (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / relative)
      .string();
}

#ifdef SANKHYA_ENABLE_CUDA

constexpr double kUnitRoundoff = std::numeric_limits<double>::epsilon() / 2.0;
// 44 roundings in an entry of Ahat and its two scale factors (see the header), one in the
// division by r_i, one in x_u = c_j xhat_j, and two of margin.
constexpr int kScalingRoundings = 48;

double gamma_k(double k) {
  return k * kUnitRoundoff / (1.0 - k * kUnitRoundoff);
}

// The same Problem both CUDA engines build.
pdhg::Problem make_problem(const Model& model) {
  pdhg::Problem prob;
  prob.model = &model;
  const auto n = static_cast<std::size_t>(model.num_cols());
  prob.cost.resize(n);
  for (std::size_t j = 0; j < n; ++j)
    prob.cost[j] = model.sense_multiplier() * model.col_cost[j];
  prob.cost_norm = pdhg::euclidean_norm(prob.cost);
  double bsq = 0.0;
  for (std::size_t i = 0; i < static_cast<std::size_t>(model.num_rows()); ++i) {
    const double b = is_finite_bound(model.row_lower[i])
                         ? model.row_lower[i]
                         : (is_finite_bound(model.row_upper[i]) ? model.row_upper[i] : 0.0);
    bsq += b * b;
  }
  prob.bound_norm = std::sqrt(bsq);
  return prob;
}

struct Bounds {
  double absolute_primal, absolute_dual, complementarity, primal_objective, dual_objective;
  double gap, restart;
};

// The rounding bound of every compared field for the point (x_u, y_u), from the header's
// argument. `host` is the reference evaluation; `reduced` its reduced costs.
Bounds rounding_bounds(const pdhg::Problem& prob, const std::vector<double>& x,
                       const std::vector<double>& y, const std::vector<double>& reduced,
                       const pdhg::Residuals& host, double restart) {
  const Model& model = *prob.model;
  const auto n = static_cast<std::size_t>(model.num_cols());
  const auto m = static_cast<std::size_t>(model.num_rows());
  std::vector<double> row_magnitude(m, 0.0), column_magnitude(n, 0.0);
  std::vector<int> row_terms(m, 0);
  const auto& starts = model.matrix.column_starts();
  const auto& rows = model.matrix.row_indices();
  const auto& values = model.matrix.values();
  for (std::size_t j = 0; j < n; ++j) {
    column_magnitude[j] = std::fabs(prob.cost[j]);
    for (auto e = static_cast<std::size_t>(starts[j]);
         e < static_cast<std::size_t>(starts[j + 1]); ++e) {
      const auto i = static_cast<std::size_t>(rows[e]);
      row_magnitude[i] += std::fabs(values[e] * x[j]);
      column_magnitude[j] += std::fabs(values[e] * y[i]);
      ++row_terms[i];
    }
  }
  double activity_sq = 0.0, reduced_sq = 0.0, complementarity = 0.0;
  for (std::size_t i = 0; i < m; ++i) {
    const double delta = gamma_k(2.0 * row_terms[i] + kScalingRoundings) * row_magnitude[i];
    activity_sq += delta * delta;
    complementarity = std::max(complementarity, std::fabs(y[i]) * delta);
  }
  double bound_move = 0.0, bound_terms = 0.0, objective_terms = 0.0, support_terms = 0.0;
  for (std::size_t j = 0; j < n; ++j) {
    const auto k = static_cast<double>(starts[j + 1] - starts[j]);
    const double delta = gamma_k(2.0 * k + kScalingRoundings + 1.0) * column_magnitude[j];
    reduced_sq += delta * delta;
    const double lo = model.col_lower[j], hi = model.col_upper[j];
    double widest = 0.0, slack = 0.0;
    if (is_finite_bound(lo)) widest = std::max(widest, std::fabs(lo));
    if (is_finite_bound(hi)) widest = std::max(widest, std::fabs(hi));
    if (is_finite_bound(lo)) slack = std::fabs(x[j] - lo);
    if (is_finite_bound(hi))
      slack =
          is_finite_bound(lo) ? std::min(slack, std::fabs(hi - x[j])) : std::fabs(hi - x[j]);
    bound_move += delta * widest;
    complementarity = std::max(complementarity, delta * slack);
    bound_terms += std::fabs(reduced[j]) * widest;
    objective_terms += std::fabs(prob.cost[j] * x[j]);
  }
  for (std::size_t i = 0; i < m; ++i) {
    double widest = 0.0;
    if (is_finite_bound(model.row_lower[i])) widest = std::fabs(model.row_lower[i]);
    if (is_finite_bound(model.row_upper[i]))
      widest = std::max(widest, std::fabs(model.row_upper[i]));
    support_terms += std::fabs(y[i]) * widest;
  }
  const double dn = static_cast<double>(n), dm = static_cast<double>(m);
  Bounds b{};
  const double da = std::sqrt(activity_sq), dr = std::sqrt(reduced_sq);
  b.absolute_primal = da + 2.0 * gamma_k(dm + 2.0) * (host.absolute_primal + da);
  b.absolute_dual = dr + 2.0 * gamma_k(dn + dm + 2.0) * (host.absolute_dual + dr);
  b.complementarity = complementarity + 4.0 * kUnitRoundoff * host.complementarity;
  b.primal_objective = gamma_k(dn + 1.0) * objective_terms;
  b.dual_objective = bound_move + gamma_k(dn + 2.0) * (bound_terms + bound_move) +
                     gamma_k(dm + 2.0) * support_terms;
  const double scale = 1.0 + std::fabs(host.primal_objective) + std::fabs(host.dual_objective);
  b.gap = 2.0 * (b.primal_objective + b.dual_objective) / scale + gamma_k(4.0) * host.gap;
  b.restart = gamma_k(std::max(dn, dm) + 2.0) * restart;
  return b;
}

bool device_present(const pdhg::Problem& prob, const Scaling& scaling) {
  const auto n = static_cast<std::size_t>(prob.model->num_cols());
  const auto m = static_cast<std::size_t>(prob.model->num_rows());
  gpu::eval::TestEvaluation probe;
  return gpu::eval::evaluate_for_testing(
      prob, scaling, std::vector<double>(n, 0.0), std::vector<double>(m, 0.0),
      std::vector<double>(n, 0.0), std::vector<double>(m, 0.0), 0.0, &probe);
}

struct Point {
  const char* label;
  std::vector<double> x_s, y_s, x_restart, y_restart;
  double count;  // > 0: x_s, y_s are running sums over `count` iterates
};

// One point, device against host, every field against its bound. Returns the number of
// fields compared (each is an EXPECT).
int compare_point(const std::string& name, const pdhg::Problem& prob, const Scaling& scaling,
                  const Point& p) {
  const Model& model = *prob.model;
  const auto n = static_cast<std::size_t>(model.num_cols());
  const auto m = static_cast<std::size_t>(model.num_rows());
  // The host path's own arithmetic: average by division, then x = Dc xhat, y = Dr yhat.
  std::vector<double> x(n), y(m), activity, reduced;
  for (std::size_t j = 0; j < n; ++j)
    x[j] = (p.count > 0.0 ? p.x_s[j] / p.count : p.x_s[j]) * scaling.column[j];
  for (std::size_t i = 0; i < m; ++i)
    y[i] = (p.count > 0.0 ? p.y_s[i] / p.count : p.y_s[i]) * scaling.row[i];
  const pdhg::Residuals host = pdhg::evaluate(prob, x, y, activity, reduced);
  double restart_x = 0.0, restart_y = 0.0;
  for (std::size_t j = 0; j < n; ++j) restart_x += std::pow(p.x_s[j] - p.x_restart[j], 2);
  for (std::size_t i = 0; i < m; ++i) restart_y += std::pow(p.y_s[i] - p.y_restart[i], 2);
  restart_x = std::sqrt(restart_x);
  restart_y = std::sqrt(restart_y);

  gpu::eval::TestEvaluation dev;
  EXPECT_TRUE(gpu::eval::evaluate_for_testing(prob, scaling, p.x_s, p.y_s, p.x_restart,
                                              p.y_restart, p.count, &dev))
      << name << " " << p.label;
  const pdhg::Residuals& d = dev.residuals;
  const Bounds b = rounding_bounds(prob, x, y, reduced, host, std::max(restart_x, restart_y));
  const std::string what = name + " " + p.label;
  EXPECT_LE(std::fabs(d.absolute_primal - host.absolute_primal), b.absolute_primal) << what;
  EXPECT_LE(std::fabs(d.absolute_dual - host.absolute_dual), b.absolute_dual) << what;
  if (std::isinf(host.complementarity) || std::isinf(d.complementarity)) {
    // |multiplier| * inf on a free column or row: infinite for ANY nonzero multiplier, so a
    // reduced cost that is exactly zero on one side and a rounding error on the other
    // legitimately gives 0 against inf. Only possible where such a free variable exists.
    bool free_variable = false;
    for (std::size_t j = 0; j < n; ++j)
      free_variable = free_variable || (!is_finite_bound(model.col_lower[j]) &&
                                        !is_finite_bound(model.col_upper[j]));
    for (std::size_t i = 0; i < m; ++i)
      free_variable = free_variable || (!is_finite_bound(model.row_lower[i]) &&
                                        !is_finite_bound(model.row_upper[i]));
    EXPECT_TRUE(free_variable) << what << ": complementarity " << d.complementarity
                               << " on the device, " << host.complementarity << " on the host";
  } else {
    EXPECT_LE(std::fabs(d.complementarity - host.complementarity), b.complementarity) << what;
  }
  EXPECT_LE(std::fabs(d.primal_objective - host.primal_objective), b.primal_objective) << what;
  EXPECT_LE(std::fabs(d.dual_objective - host.dual_objective), b.dual_objective) << what;
  EXPECT_LE(std::fabs(d.gap - host.gap), b.gap) << what;
  // The relative residuals are the absolute ones over a common positive constant.
  EXPECT_LE(std::fabs(d.primal - host.primal),
            b.absolute_primal / (1.0 + prob.bound_norm) + 2.0 * kUnitRoundoff * host.primal)
      << what;
  EXPECT_LE(std::fabs(d.dual - host.dual),
            b.absolute_dual / (1.0 + prob.cost_norm) + 2.0 * kUnitRoundoff * host.dual)
      << what;
  if (p.count <= 0.0) {
    EXPECT_LE(std::fabs(dev.restart_dx - restart_x), b.restart) << what;
    EXPECT_LE(std::fabs(dev.restart_dy - restart_y), b.restart) << what;
  }
  std::printf("%-9s %-8s primal %.3e/%.3e dual %.3e/%.3e gap %.3e/%.3e (device/host)\n",
              name.c_str(), p.label, d.absolute_primal, host.absolute_primal, d.absolute_dual,
              host.absolute_dual, d.gap, host.gap);
  return 10;
}

TEST(PdhgDeviceEvaluation, AgreesWithTheHostEvaluationToRoundingOnNetlib) {
  const std::vector<const char*> names = {"afiro",    "sc50a",  "adlittle", "blend",  "share2b",
                                          "stocfor1", "israel", "25fv47",   "degen2", "fit2p"};
  int compared = 0;
  for (const char* name : names) {
    Model model;
    ASSERT_TRUE(
        io::read_model(repository_path(std::string("data/netlib/") + name + ".mps"), &model).ok)
        << name;
    const pdhg::Problem prob = make_problem(model);
    const Scaling scaling = build_scaling(model, prob.cost, 10);  // the engines' 10 Ruiz rounds
    if (!device_present(prob, scaling)) {
      GTEST_SKIP() << "no CUDA device: skipped, not passed.";
    }
    const auto n = static_cast<std::size_t>(model.num_cols());
    const auto m = static_cast<std::size_t>(model.num_rows());

    // The engines' starting point: zero projected onto the scaled column bounds, y = 0.
    Point start{"start", std::vector<double>(n), std::vector<double>(m, 0.0), {}, {}, 0.0};
    for (std::size_t j = 0; j < n; ++j)
      start.x_s[j] = std::clamp(0.0, scaling.col_lower[j], scaling.col_upper[j]);
    start.x_restart = start.x_s;
    start.y_restart = start.y_s;

    std::mt19937_64 rng(478);
    std::normal_distribution<double> normal(0.0, 1.0);
    Point random{"random",
                 start.x_s,
                 std::vector<double>(m),
                 std::vector<double>(n),
                 std::vector<double>(m),
                 0.0};
    for (std::size_t j = 0; j < n; ++j) {
      const double lo = std::max(scaling.col_lower[j], -10.0);
      const double hi = std::min(scaling.col_upper[j], 10.0);
      random.x_s[j] = lo + (hi - lo) * std::uniform_real_distribution<double>(0.0, 1.0)(rng);
      random.x_restart[j] = normal(rng);
    }
    for (std::size_t i = 0; i < m; ++i) {
      random.y_s[i] = normal(rng);
      random.y_restart[i] = normal(rng);
    }

    // A basic optimum from the simplex, in the engines' sign convention (row_dual = -sense y)
    // and scaled space.
    Options simplex;
    simplex.set_bool("log_to_console", false);
    const Solution solved = solve(model, simplex);
    ASSERT_EQ(solved.status, SolveStatus::kOptimal) << name;
    Point optimum{
        "optimum", std::vector<double>(n), std::vector<double>(m), start.x_s, start.y_s, 0.0};
    for (std::size_t j = 0; j < n; ++j)
      optimum.x_s[j] = solved.col_value[j] / scaling.column[j];
    for (std::size_t i = 0; i < m; ++i)
      optimum.y_s[i] = -model.sense_multiplier() * solved.row_dual[i] / scaling.row[i];

    // The average path: three times the random point as a running sum over three iterates.
    Point average{"average", random.x_s, random.y_s, random.x_restart, random.y_restart, 3.0};
    for (double& v : average.x_s) v *= 3.0;
    for (double& v : average.y_s) v *= 3.0;

    for (const Point* p : {&start, &random, &optimum, &average})
      compared += compare_point(name, prob, scaling, *p);
  }
  std::cout << "device against host evaluation: " << compared << " field comparisons on "
            << names.size() << " Netlib instances\n";
}

#endif  // SANKHYA_ENABLE_CUDA

// ---- End to end: the same answer with the evaluation on either side -----------------

Options gpu_pdhg(bool device_evaluation, bool device_loop) {
  Options o;
  o.set_bool("log_to_console", true);  // read below to prove which evaluation ran
  o.set_string("algorithm", "pdhg");
  o.set_bool("gpu", true);
  o.set_bool("deterministic", true);
  o.set_bool("gpu_on_device_loop", device_loop);
  o.set_bool("gpu_device_evaluation", device_evaluation);
  o.set_bool("pdhg_polish", false);
  o.set_double("pdhg_tolerance", 1e-8);
  o.set_int("iteration_limit", 1000000);
  return o;
}

struct Logged {
  Solution solution;
  std::string log;
};

Logged solve_logged(const Model& model, const Options& options) {
  ::testing::internal::CaptureStdout();
  Logged run;
  run.solution = solve(model, options);
  std::fflush(stdout);
  run.log = ::testing::internal::GetCapturedStdout();
  return run;
}

TEST(PdhgDeviceEvaluation, DeviceAndHostEvaluationReachTheSameAnswer) {
  // Held to the stopping tolerance both runs are asked for, not to the bit: the two
  // evaluations differ by rounding (above), so a restart can be decided one evaluation
  // apart and the trajectories part from there - the argument of
  // test_pdhg_cuda_determinism.cpp's per-iteration-against-device-loop test.
  Model probe;
  ASSERT_TRUE(io::read_model(repository_path("data/netlib/afiro.mps"), &probe).ok);
  Options probe_options = gpu_pdhg(true, false);
  probe_options.set_bool("log_to_console", false);
  probe_options.set_int("iteration_limit", 1);
  if (solve(probe, probe_options).algorithm.find("cuda") == std::string::npos) {
    GTEST_SKIP() << "CUDA backend or device not present: skipped, not passed.";
  }
  const std::vector<const char*> names = {"afiro", "sc50a",    "sc50b", "adlittle",
                                          "blend", "stocfor1", "israel"};
  constexpr double kAgreementTol = 1e-8;  // the stopping tolerance
  int agreed = 0, runs = 0;
  for (const char* name : names) {
    Model model;
    ASSERT_TRUE(
        io::read_model(repository_path(std::string("data/netlib/") + name + ".mps"), &model).ok)
        << name;
    for (const bool device_loop : {false, true}) {
      const Logged host = solve_logged(model, gpu_pdhg(false, device_loop));
      const Logged dev = solve_logged(model, gpu_pdhg(true, device_loop));
      const std::string what = std::string(name) + (device_loop ? " (device loop)" : "");
      EXPECT_NE(host.log.find("Evaluation on the host"), std::string::npos) << what;
      EXPECT_NE(dev.log.find("Evaluation on the device"), std::string::npos) << what;
      const Solution& a = host.solution;
      const Solution& b = dev.solution;
      const bool both =
          (a.status == SolveStatus::kOptimal || a.status == SolveStatus::kFeasible) &&
          (b.status == SolveStatus::kOptimal || b.status == SolveStatus::kFeasible);
      EXPECT_TRUE(both) << what << ": host " << a.message << "; device " << b.message;
      const double scale = std::max({1.0, std::fabs(a.objective), std::fabs(b.objective)});
      const double diff = std::fabs(a.objective - b.objective);
      EXPECT_LE(diff, kAgreementTol * scale)
          << what << ": host " << a.objective << ", device " << b.objective;
      ++runs;
      if (both && diff <= kAgreementTol * scale) ++agreed;
      std::cout << what << ": host evaluation " << a.iterations << " iterations, device "
                << b.iterations << ", relative objective difference " << diff / scale << "\n";
    }
  }
  std::cout << "host against device evaluation: " << agreed << "/" << runs << " agree at "
            << kAgreementTol << "\n";
  EXPECT_EQ(agreed, runs);
}

}  // namespace
}  // namespace sankhya
