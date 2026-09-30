// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the first-order QP operator, host against device (#493).
//
// The host operator is the arithmetic qp_condat_vu.cpp ran inline before the operator
// existed; the first test pins it to that arithmetic, expression for expression, so a
// refactor cannot move it. The CUDA tests hold the device operator to the host one: one
// step to rounding, a whole solve to the tolerance, with and without the Halpern blend, and
// the engine's fallback when the device is asked for and absent.

#include <cmath>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "qp/qp_operator.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/qp.hpp"

#ifdef SANKHYA_ENABLE_CUDA
#include "gpu/qp_device.hpp"
#endif

namespace sankhya {
namespace {

/// A convex QP with every kind of bound: n columns, m rows, Q = L L^T + diag on a sparse L.
Model random_qp(Index n, Index m, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  Model model;
  model.name = "qp-device";
  const auto un = static_cast<std::size_t>(n);
  model.col_cost.resize(un);
  model.col_lower.resize(un);
  model.col_upper.resize(un);
  model.col_type.assign(un, VarType::kContinuous);
  for (std::size_t j = 0; j < un; ++j) {
    model.col_cost[j] = unit(rng);
    switch (j % 4) {
      case 0:
        model.col_lower[j] = 0.0;
        model.col_upper[j] = 10.0;
        break;
      case 1:
        model.col_lower[j] = -kInfinity;
        model.col_upper[j] = 5.0;
        break;
      case 2:
        model.col_lower[j] = -3.0;
        model.col_upper[j] = kInfinity;
        break;
      default:
        model.col_lower[j] = -kInfinity;
        model.col_upper[j] = kInfinity;
        break;
    }
  }
  model.matrix.reset(m, n);
  const auto um = static_cast<std::size_t>(m);
  model.row_lower.resize(um);
  model.row_upper.resize(um);
  for (Index i = 0; i < m; ++i) {
    for (int k = 0; k < 4; ++k) {
      const auto j = static_cast<Index>(rng() % static_cast<unsigned>(n));
      model.matrix.add_entry(i, j, unit(rng));
    }
    const auto u = static_cast<std::size_t>(i);
    switch (i % 3) {
      case 0:
        model.row_lower[u] = -1.0;
        model.row_upper[u] = 1.0;
        break;
      case 1:
        model.row_lower[u] = -kInfinity;
        model.row_upper[u] = 2.0;
        break;
      default:
        model.row_lower[u] = 0.5;
        model.row_upper[u] = 0.5;
        break;
    }
  }
  model.matrix.finalize();
  // Q = L L^T + I on a sparse lower L: convex, and its lower triangle has off-diagonals.
  std::vector<std::vector<std::pair<Index, double>>> l_rows(un);
  for (Index i = 0; i < n; ++i) {
    l_rows[static_cast<std::size_t>(i)].push_back({i, 1.0 + std::fabs(unit(rng))});
    if (i > 0) {
      const auto j = static_cast<Index>(rng() % static_cast<unsigned>(i));
      l_rows[static_cast<std::size_t>(i)].push_back({j, unit(rng)});
    }
  }
  model.hessian.reset(n, n);
  for (Index i = 0; i < n; ++i) {
    for (Index j = 0; j <= i; ++j) {
      double q = i == j ? 1.0 : 0.0;
      for (const auto& [ki, vi] : l_rows[static_cast<std::size_t>(i)]) {
        for (const auto& [kj, vj] : l_rows[static_cast<std::size_t>(j)]) {
          if (ki == kj) q += vi * vj;
        }
      }
      if (q != 0.0) model.hessian.add_entry(i, j, q);
    }
  }
  model.hessian.finalize();
  EXPECT_EQ(model.validate(), "");
  return model;
}

double project(double value, double lower, double upper) {
  if (is_finite_bound(lower) && value < lower) return lower;
  if (is_finite_bound(upper) && value > upper) return upper;
  return value;
}

/// One Condat-Vu step from (x, y), written out as the engine had it inline.
void reference_step(const Model& model, double tau, double sigma, const std::vector<double>& x,
                    const std::vector<double>& y, std::vector<double>* xn,
                    std::vector<double>* yn) {
  const auto n = static_cast<std::size_t>(model.num_cols());
  const auto m = static_cast<std::size_t>(model.num_rows());
  const double sense = model.sense_multiplier();
  std::vector<double> qx(n, 0.0), at_y(n, 0.0), ext(n), ax(m, 0.0);
  qp::hessian_multiply(model, x, &qx);
  if (m > 0) model.matrix.transpose_multiply_add(y.data(), at_y.data());
  xn->resize(n);
  for (std::size_t u = 0; u < n; ++u) {
    const double gradient = sense * model.col_cost[u] + sense * qx[u] + at_y[u];
    (*xn)[u] = project(x[u] - tau * gradient, model.col_lower[u], model.col_upper[u]);
    ext[u] = 2.0 * (*xn)[u] - x[u];
  }
  yn->resize(m);
  if (m > 0) {
    model.matrix.multiply(ext.data(), ax.data());
    for (std::size_t u = 0; u < m; ++u) {
      const double v = y[u] + sigma * ax[u];
      (*yn)[u] = v - sigma * project(v / sigma, model.row_lower[u], model.row_upper[u]);
    }
  }
}

void expect_close(const std::vector<double>& a, const std::vector<double>& b, double rel,
                  const char* what) {
  ASSERT_EQ(a.size(), b.size()) << what;
  for (std::size_t u = 0; u < a.size(); ++u) {
    EXPECT_NEAR(a[u], b[u], rel * std::max(1.0, std::fabs(b[u]))) << what << " at " << u;
  }
}

TEST(QpOperator, HostOperatorIsTheInlineArithmetic) {
  const Model model = random_qp(40, 25, 7u);
  std::unique_ptr<qp::QpOperator> op = qp::make_host_operator(model);
  ASSERT_EQ(std::string(op->where()), "host");
  std::vector<double> x, y, xn, yn, ox, oy;
  ASSERT_TRUE(op->download(false, &x, &y));
  const double tau = 0.05, sigma = 0.3;
  for (int k = 0; k < 3; ++k) {
    reference_step(model, tau, sigma, x, y, &xn, &yn);
    ASSERT_TRUE(op->step(tau, sigma));
    ASSERT_TRUE(op->download(true, &ox, &oy));
    expect_close(ox, xn, 0.0, "x' of the host operator");
    expect_close(oy, yn, 0.0, "y' of the host operator");
    double dx2 = 0.0, dy2 = 0.0;
    for (std::size_t u = 0; u < x.size(); ++u) dx2 += (xn[u] - x[u]) * (xn[u] - x[u]);
    for (std::size_t u = 0; u < y.size(); ++u) dy2 += (yn[u] - y[u]) * (yn[u] - y[u]);
    EXPECT_DOUBLE_EQ(op->fixed_point_residual(tau, sigma), std::sqrt(dx2 / tau + dy2 / sigma));
    ASSERT_TRUE(op->advance(0.0, -1.0));
    x = xn;
    y = yn;
  }
}

TEST(QpOperator, EngineWithoutADeviceKeepsTheHostAndSaysSo) {
  // qp_gpu=true on a CPU build, or on a CUDA build with no card, is a warning and the host
  // answer; the status and the objective are the host's.
  const Model model = random_qp(30, 20, 3u);
  Options host;
  host.set_bool("log_to_console", false);
  host.set_double("qp_tolerance", 1e-6);
  host.set_string("qp_algorithm", "condat-vu");  // the operator belongs to this engine
  host.set_int("iteration_limit", 500000);
  Options asked = host;
  asked.set_bool("qp_gpu", true);
  const Solution a = solve(model, host);
  const Solution b = solve(model, asked);
  ASSERT_EQ(a.status, SolveStatus::kOptimal) << a.message;
  EXPECT_EQ(b.status, a.status) << b.message;
  EXPECT_NEAR(b.objective, a.objective, 1e-6 * std::max(1.0, std::fabs(a.objective)));
}

#ifdef SANKHYA_ENABLE_CUDA

// GTEST_SKIP returns from the function it is written in, so it cannot live in a helper that
// returns a value: the helper hands back the reason and each test skips on it.
std::unique_ptr<qp::QpOperator> device_operator(const Model& model, std::string* reason) {
  return gpu::make_qp_device_operator(model, reason);
}

TEST(QpDevice, OneStepMatchesTheHostToRounding) {
  const Model model = random_qp(300, 200, 11u);
  std::string reason;
  std::unique_ptr<qp::QpOperator> device = device_operator(model, &reason);
  if (device == nullptr) GTEST_SKIP() << "no device operator: " << reason;
  std::unique_ptr<qp::QpOperator> host = qp::make_host_operator(model);
  const double tau = 0.02, sigma = 0.5;
  std::vector<double> hx, hy, dx, dy;
  for (int k = 0; k < 5; ++k) {
    ASSERT_TRUE(host->step(tau, sigma));
    ASSERT_TRUE(device->step(tau, sigma));
    ASSERT_TRUE(host->download(true, &hx, &hy));
    ASSERT_TRUE(device->download(true, &dx, &dy));
    expect_close(dx, hx, 1e-12, "x' on the device");
    expect_close(dy, hy, 1e-12, "y' on the device");
    const double hr = host->fixed_point_residual(tau, sigma);
    const double dr = device->fixed_point_residual(tau, sigma);
    EXPECT_NEAR(dr, hr, 1e-12 * std::max(1.0, hr)) << "fixed-point residual";
    // Alternate the plain advance and the Halpern blend so both kernels are covered.
    ASSERT_TRUE(host->set_anchor());
    ASSERT_TRUE(device->set_anchor());
    const double rho = k % 2 == 0 ? -1.0 : 0.3;
    ASSERT_TRUE(host->advance(static_cast<double>(k), rho));
    ASSERT_TRUE(device->advance(static_cast<double>(k), rho));
  }
  ASSERT_TRUE(host->set_restart());
  ASSERT_TRUE(device->set_restart());
  ASSERT_TRUE(host->step(tau, sigma));
  ASSERT_TRUE(device->step(tau, sigma));
  ASSERT_TRUE(host->take_tz());
  ASSERT_TRUE(device->take_tz());
  double hdx = 0.0, hdy = 0.0, ddx = 0.0, ddy = 0.0;
  ASSERT_TRUE(host->restart_distance(&hdx, &hdy));
  ASSERT_TRUE(device->restart_distance(&ddx, &ddy));
  EXPECT_NEAR(ddx, hdx, 1e-12 * std::max(1.0, hdx));
  EXPECT_NEAR(ddy, hdy, 1e-12 * std::max(1.0, hdy));
}

TEST(QpDevice, WholeSolveMatchesTheHostAtTheTolerance) {
  const Model model = random_qp(400, 250, 5u);
  {
    std::string reason;
    if (device_operator(model, &reason) == nullptr) {
      GTEST_SKIP() << "no device operator: " << reason;
    }
  }
  for (const bool halpern : {false, true}) {
    Options host;
    host.set_bool("log_to_console", false);
    host.set_double("qp_tolerance", 1e-8);
    host.set_string("qp_algorithm", "condat-vu");  // not the default interior point
    host.set_int("iteration_limit", 200000);
    host.set_bool("qp_halpern", halpern);
    Options device = host;
    device.set_bool("qp_gpu", true);
    const Solution a = solve(model, host);
    const Solution b = solve(model, device);
    // Whatever the host reaches, the device reaches the same (#493: the same arithmetic).
    ASSERT_NE(a.status, SolveStatus::kNotSolved) << a.message;
    EXPECT_EQ(b.algorithm, "qp-condat-vu-cuda") << "halpern " << halpern;
    EXPECT_EQ(b.status, a.status) << b.message;
    EXPECT_NEAR(b.objective, a.objective, 1e-6 * std::max(1.0, std::fabs(a.objective)))
        << "halpern " << halpern;
    // The two trajectories differ only by the products' summation order, so the iteration
    // counts should be the same or within a restart period of each other.
    EXPECT_LE(std::abs(static_cast<long long>(b.iterations - a.iterations)), 50)
        << "halpern " << halpern;
  }
}

#endif  // SANKHYA_ENABLE_CUDA

}  // namespace
}  // namespace sankhya
