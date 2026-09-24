// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the VIPR certificate writer's model and claim sections (#518): VAR, INT, OBJ, CON,
// RTP and SOL. Split from certificate_writer.cpp, which writes the derivations, to keep both
// under the file-length rule. See certificate_writer.hpp for the format and references.

#include <cmath>
#include <ostream>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "certificate_writer.hpp"
#include "core/safe_bound.hpp"

namespace sankhya::mip::detail {
namespace {

/// A VIPR name: the model's own when it is a plain token, else a generated one.
std::string token_name(const std::vector<std::string>& names, Index j, char prefix) {
  const auto u = static_cast<std::size_t>(j);
  if (u < names.size() && !names[u].empty()) {
    std::string name = names[u];
    bool plain = true;
    for (const char ch : name) {
      if (ch <= ' ' || ch == '%' || ch == '{' || ch == '}') plain = false;
    }
    if (plain) return name;
  }
  return fmt::format("{}{}", prefix, j);
}

}  // namespace

/// VAR, INT, OBJ and CON: the model, every number exact. CON holds the rows first, one E or
/// a G and/or an L each, then the finite column bounds - the order Writer numbers them in
/// and tools/verify_certificate_model.py reproduces from an MPS file.
void write_model(std::ostream& out, const Model& model, Index num_con) {
  const auto n = static_cast<std::size_t>(model.num_cols());
  const auto m = static_cast<std::size_t>(model.num_rows());
  out << "% SANKHYA MILP certificate (#518), VIPR layout (Cheung, Gleixner and Steffy, "
         "IPCO 2017)\n% check with tools/verify_certificate.py; every bound excludes the "
         "objective constant\n";
  out << "VER 1.0\nVAR " << n << '\n';
  for (std::size_t j = 0; j < n; ++j) {
    out << token_name(model.col_names, static_cast<Index>(j), 'x') << '\n';
  }
  std::string integers;
  Index int_count = 0;
  for (std::size_t j = 0; j < n; ++j) {
    if (model.col_type[j] != VarType::kInteger) continue;
    integers += fmt::format(" {}", j);
    ++int_count;
  }
  out << "INT " << int_count << '\n' << integers << '\n';
  std::string objective;
  Index obj_count = 0;
  for (std::size_t j = 0; j < n; ++j) {
    if (model.col_cost[j] == 0.0) continue;
    objective += fmt::format(" {} {}", j, exact_decimal(model.col_cost[j]));
    ++obj_count;
  }
  out << "OBJ " << (model.sense == ObjSense::kMaximize ? "max" : "min") << '\n'
      << obj_count << objective << '\n';
  Index bound_count = 0;
  for (std::size_t j = 0; j < n; ++j) {
    bound_count += is_finite_bound(model.col_lower[j]) ? 1 : 0;
    bound_count += is_finite_bound(model.col_upper[j]) ? 1 : 0;
  }
  out << "CON " << num_con << ' ' << bound_count << '\n';
  const CsrView rows(model.matrix);
  for (std::size_t i = 0; i < m; ++i) {
    const ColumnView row = rows.row(static_cast<Index>(i));
    std::string entries = fmt::format("{}", row.size);
    for (Index p = 0; p < row.size; ++p) {
      entries += fmt::format(" {} {}", row.rows[p], exact_decimal(row.values[p]));
    }
    const std::string name = token_name(model.row_names, static_cast<Index>(i), 'c');
    const double lo = model.row_lower[i];
    const double hi = model.row_upper[i];
    if (is_finite_bound(lo) && lo == hi) {
      out << name << " E " << exact_decimal(lo) << ' ' << entries << '\n';
      continue;
    }
    if (is_finite_bound(lo))
      out << name << " G " << exact_decimal(lo) << ' ' << entries << '\n';
    if (is_finite_bound(hi))
      out << name << " L " << exact_decimal(hi) << ' ' << entries << '\n';
  }
  for (std::size_t j = 0; j < n; ++j) {
    if (is_finite_bound(model.col_lower[j])) {
      out << "lb" << j << " G " << exact_decimal(model.col_lower[j]) << " 1 " << j << " 1\n";
    }
    if (is_finite_bound(model.col_upper[j])) {
      out << "ub" << j << " L " << exact_decimal(model.col_upper[j]) << " 1 " << j << " 1\n";
    }
  }
}

/// RTP and SOL for a point. Its integer columns are written as the integers they round to,
/// and its objective is enclosed with outward rounding, so the primal side of the claim is
/// never tighter than what the written point attains.
void write_claim(std::ostream& out, const Model& model, const std::vector<double>& incumbent,
                 double proved) {
  if (incumbent.empty()) {
    out << (model.sense == ObjSense::kMaximize ? "RTP range -inf " : "RTP range ")
        << exact_decimal(proved) << (model.sense == ObjSense::kMaximize ? "" : " inf")
        << "\nSOL 0\n";
    return;
  }
  const auto n = static_cast<std::size_t>(model.num_cols());
  std::vector<double> x(incumbent);
  for (std::size_t j = 0; j < n; ++j) {
    if (model.col_type[j] == VarType::kInteger) x[j] = std::nearbyint(x[j]);
  }
  const auto [value_lo, value_hi] = dot_enclosure(model.col_cost, x);
  if (model.sense == ObjSense::kMaximize) {
    out << "RTP range " << exact_decimal(value_lo) << ' ' << exact_decimal(proved) << '\n';
  } else {
    out << "RTP range " << exact_decimal(proved) << ' ' << exact_decimal(value_hi) << '\n';
  }
  std::string values;
  Index nonzeros = 0;
  for (std::size_t j = 0; j < n; ++j) {
    if (x[j] == 0.0) continue;
    values += fmt::format(" {} {}", j, exact_decimal(x[j]));
    ++nonzeros;
  }
  out << "SOL 1\nincumbent " << nonzeros << values << '\n';
}

}  // namespace sankhya::mip::detail
