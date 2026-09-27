// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the host implementation of the first-order QP operator (#493): the arithmetic
// qp_condat_vu.cpp ran inline before the device operator existed, expression for expression.
// See qp_operator.hpp.

#include "qp_operator.hpp"

#include <algorithm>
#include <cmath>

#include "qp_first_order_accel.hpp"
#include "sankhya/types.hpp"

namespace sankhya::qp {
namespace {

[[nodiscard]] double project(double value, double lower, double upper) {
  if (is_finite_bound(lower) && value < lower) return lower;
  if (is_finite_bound(upper) && value > upper) return upper;
  return value;
}

[[nodiscard]] double squared_distance(const std::vector<double>& a,
                                      const std::vector<double>& b) {
  double total = 0.0;
  for (std::size_t u = 0; u < a.size(); ++u) {
    const double d = a[u] - b[u];
    total += d * d;
  }
  return total;
}

class HostOperator final : public QpOperator {
 public:
  explicit HostOperator(const Model& model)
      : model_(model),
        n_(model.num_cols()),
        m_(model.num_rows()),
        sense_(model.sense_multiplier()),
        x_(static_cast<std::size_t>(n_), 0.0),
        y_(static_cast<std::size_t>(m_), 0.0),
        x_next_(x_),
        y_next_(y_),
        extrapolated_(x_),
        qx_(x_),
        at_y_(x_),
        ax_(y_) {
    for (Index j = 0; j < n_; ++j) {
      const auto u = static_cast<std::size_t>(j);
      x_[u] = project(0.0, model.col_lower[u], model.col_upper[u]);
    }
  }

  bool step(double tau, double sigma) override {
    // Primal: x' = proj_box( x - tau (c + Qx + A'y) ). The Qx term is the whole difference
    // from the LP engine; everything else is Chambolle-Pock unchanged.
    hessian_multiply(model_, x_, &qx_);
    std::fill(at_y_.begin(), at_y_.end(), 0.0);
    if (m_ > 0) model_.matrix.transpose_multiply_add(y_.data(), at_y_.data());
    for (Index j = 0; j < n_; ++j) {
      const auto u = static_cast<std::size_t>(j);
      const double gradient = sense_ * model_.col_cost[u] + sense_ * qx_[u] + at_y_[u];
      x_next_[u] = project(x_[u] - tau * gradient, model_.col_lower[u], model_.col_upper[u]);
      extrapolated_[u] = 2.0 * x_next_[u] - x_[u];
    }
    // Dual: y' = prox_{sigma sigma_C}(y + sigma A xbar) = v - sigma proj_C(v / sigma).
    if (m_ > 0) {
      model_.matrix.multiply(extrapolated_.data(), ax_.data());
      for (Index i = 0; i < m_; ++i) {
        const auto u = static_cast<std::size_t>(i);
        const double v = y_[u] + sigma * ax_[u];
        y_next_[u] = v - sigma * project(v / sigma, model_.row_lower[u], model_.row_upper[u]);
      }
    }
    return true;
  }

  double fixed_point_residual(double tau, double sigma) override {
    const double dx2 = squared_distance(x_next_, x_);
    const double dy2 = squared_distance(y_next_, y_);
    return std::sqrt(dx2 / tau + (m_ > 0 ? dy2 / sigma : 0.0));
  }

  bool advance(double k, double rho) override {
    if (rho < 0.0) {
      x_.swap(x_next_);
      y_.swap(y_next_);
      return true;
    }
    halpern_blend(&x_, x_next_, x_anchor_, k, rho);
    halpern_blend(&y_, y_next_, y_anchor_, k, rho);
    return true;
  }

  bool take_tz() override {
    // A copy, not a swap: T z stays where download(true) reads the evaluated point.
    x_ = x_next_;
    y_ = y_next_;
    return true;
  }

  bool set_anchor() override {
    x_anchor_ = x_;
    y_anchor_ = y_;
    return true;
  }

  bool set_restart() override {
    x_restart_ = x_;
    y_restart_ = y_;
    return true;
  }

  bool restart_distance(double* dx, double* dy) override {
    *dx = std::sqrt(squared_distance(x_, x_restart_));
    *dy = std::sqrt(squared_distance(y_, y_restart_));
    return true;
  }

  bool download(bool tz, std::vector<double>* x, std::vector<double>* y) override {
    *x = tz ? x_next_ : x_;
    *y = tz ? y_next_ : y_;
    return true;
  }

  const char* where() const override { return "host"; }

 private:
  const Model& model_;
  const Index n_, m_;
  const double sense_;
  std::vector<double> x_, y_, x_next_, y_next_, extrapolated_, qx_, at_y_, ax_;
  std::vector<double> x_anchor_, y_anchor_, x_restart_, y_restart_;
};

}  // namespace

void hessian_multiply(const Model& model, const std::vector<double>& x,
                      std::vector<double>* out) {
  std::fill(out->begin(), out->end(), 0.0);
  for (Index j = 0; j < model.hessian.num_cols(); ++j) {
    const ColumnView column = model.hessian.column(j);
    const auto uj = static_cast<std::size_t>(j);
    for (Index k = 0; k < column.size; ++k) {
      const auto ui = static_cast<std::size_t>(column.rows[k]);
      const double value = column.values[k];
      (*out)[ui] += value * x[uj];
      if (column.rows[k] != j) (*out)[uj] += value * x[ui];
    }
  }
}

std::unique_ptr<QpOperator> make_host_operator(const Model& model) {
  return std::make_unique<HostOperator>(model);
}

}  // namespace sankhya::qp
