// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a Farkas certificate from the elastic LP (#559). See elastic_certificate.hpp.

#include "core/elastic_certificate.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "sankhya/sparse.hpp"
#include "sankhya/tolerances.hpp"
#include "sankhya/types.hpp"

namespace sankhya::detail {
namespace {

// The elastic model: every column and row of `model`, costs zero, no Hessian, every column
// continuous (a certificate of the LP is what is asked for), and one slack column per finite
// row side with cost 1 - `+1` in the row for the lower side, `-1` for the upper.
Model build_elastic(const Model& model) {
  const Index m = model.num_rows();
  const Index n = model.num_cols();
  Model elastic;
  elastic.name = model.name + "_ELASTIC";
  elastic.sense = ObjSense::kMinimize;
  elastic.row_lower = model.row_lower;
  elastic.row_upper = model.row_upper;
  elastic.row_names = model.row_names;
  elastic.col_lower = model.col_lower;
  elastic.col_upper = model.col_upper;
  elastic.col_names = model.col_names;
  elastic.col_cost.assign(static_cast<std::size_t>(n), 0.0);
  elastic.col_type.assign(static_cast<std::size_t>(n), VarType::kContinuous);

  std::vector<Index> starts = model.matrix.column_starts();
  std::vector<Index> rows = model.matrix.row_indices();
  std::vector<double> values = model.matrix.values();
  const auto add_slack = [&](Index row, double sign, const char* side) {
    rows.push_back(row);
    values.push_back(sign);
    starts.push_back(static_cast<Index>(rows.size()));
    elastic.col_cost.push_back(1.0);
    elastic.col_lower.push_back(0.0);
    elastic.col_upper.push_back(kInfinity);
    elastic.col_type.push_back(VarType::kContinuous);
    // Names only when the model carries them: a names vector shorter than the columns is a
    // model error, an empty one is not.
    if (!model.col_names.empty()) {
      const auto u = static_cast<std::size_t>(row);
      elastic.col_names.push_back(std::string(side) + "[" +
                                  (u < model.row_names.size() ? model.row_names[u] : "") + "]");
    }
  };
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    if (std::isfinite(model.row_lower[u])) add_slack(i, 1.0, "ELASTIC_LO");
    if (std::isfinite(model.row_upper[u])) add_slack(i, -1.0, "ELASTIC_HI");
  }
  const auto columns = static_cast<Index>(elastic.col_cost.size());
  elastic.matrix.assign_columns(m, columns, std::move(starts), std::move(rows),
                                std::move(values));
  return elastic;
}

}  // namespace

std::vector<double> farkas_from_elastic(const Model& model, const Options& options,
                                        double seconds) {
  const Index m = model.num_rows();
  if (m == 0 || !(seconds > 0.0)) return {};
  // The elastic LP cannot repair a column whose own bounds cross; that contradiction needs no
  // row multipliers and is not what this is for.
  for (Index j = 0; j < model.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (model.col_lower[u] > model.col_upper[u]) return {};
  }

  const Model elastic = build_elastic(model);
  // A plain LP solve of a model that is feasible by construction: nothing below may recurse
  // into another certificate search, write a file, or print.
  Options sub = options;
  sub.set_bool("log_to_console", false);
  sub.set_bool("compute_iis", false);
  sub.set_bool("certificate_elastic", false);
  sub.set_bool("engine_race", false);
  sub.set_bool("exact", false);
  sub.set_string("progress_out", "");
  sub.set_string("write_certificate", "");
  sub.set_string("algorithm", "auto");
  sub.set_double("time_limit", seconds);
  const Solution solved = solve(elastic, sub, nullptr);
  if (solved.status != SolveStatus::kOptimal) return {};
  // A positive optimal total violation is what the certificate proves; at rounding size
  // there is nothing to certify.
  if (!(solved.objective > tol::kPrimalFeasibility)) return {};
  if (static_cast<Index>(solved.row_dual.size()) < m) return {};

  std::vector<double> y(solved.row_dual.begin(), solved.row_dual.begin() + m);
  // A dual of rounding size on the side a row does not have makes the checker reject the
  // whole vector (measured on ceria3d: 50 such multipliers, the largest 1.7e-16). Each is set
  // to zero. One that was not of rounding size cannot slip through this way: the caller
  // checks the result with farkas_proves_infeasible and drops it if the contradiction no
  // longer holds.
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    if ((y[u] > 0.0 && !std::isfinite(model.row_lower[u])) ||
        (y[u] < 0.0 && !std::isfinite(model.row_upper[u]))) {
      y[u] = 0.0;
    }
  }
  if (std::none_of(y.begin(), y.end(), [](double v) { return v != 0.0; })) return {};
  return y;
}

}  // namespace sankhya::detail
