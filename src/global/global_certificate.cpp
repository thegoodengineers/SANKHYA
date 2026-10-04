// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the global bound certificate of the spatial branch and bound (#514).
//
// WHAT IS WRITTEN. The tree the search built - every node, its parent, and for an internal node
// the column and point it was split at - and, for each node whose relaxation proved something,
// the row multipliers of that relaxation: dual multipliers (a lower bound by weak duality) or a
// Farkas vector (the relaxation is empty). Nothing else is needed to check the bound, and
// nothing in it is taken on trust:
//
//   * the boxes are NOT written: the checker splits the model's own column bounds at the
//     written points, so the leaves cover the root box by construction;
//   * a bound is NOT written: the checker rebuilds each relaxation exactly (rational
//     arithmetic) from the original model and the box, and evaluates the Lagrangian at the
//     multipliers - valid for ANY multipliers, so a wrong multiplier can only weaken a bound
//     (Neumaier and Shcherbina, Math. Programming 99, 2004);
//   * the model is NOT written: the checker reads the original file with its own reader.
//
// THE CHECKER, tools/verify_global_certificate.py, never links this code. It shares with it
// only the layout of the relaxation's rows, which this file's header (in the certificate)
// states: the model's rows, then four rows per product in the order relaxation.cpp builds them.
//
// A certificate is written only for a search that used no bound tightening (FBBT removes boxes
// by an argument the checker does not re-derive), which is what `write_certificate` selects.

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "global/global_internal.hpp"

namespace sankhya::global {
namespace {

/// A double as JSON: shortest text that reads back to the same double, or a string for a
/// non-finite one (JSON has none).
std::string number(double value) {
  if (std::isnan(value)) return "\"nan\"";
  if (std::isinf(value)) return value > 0.0 ? "\"inf\"" : "\"-inf\"";
  return fmt::format("{:.17g}", value);
}

}  // namespace

std::string certificate_text(const CertificateInput& in) {
  const Problem& p = in.problem;
  std::string out = "{\n";
  out += "  \"format\": \"sankhya-global-certificate-1\",\n";
  out += fmt::format("  \"columns\": {},\n  \"rows\": {},\n", p.n, p.m);
  out += fmt::format("  \"sigma\": {},\n", number(p.sigma));
  out += fmt::format("  \"status\": \"{}\",\n", in.status);
  out += fmt::format("  \"absolute_gap\": {},\n  \"relative_gap\": {},\n",
                     number(in.absolute_gap), number(in.relative_gap));
  out += fmt::format("  \"feasibility_tolerance\": {},\n", number(in.tolerance));
  out += fmt::format("  \"lower_bound_min_form\": {},\n", number(in.lower_min_form));
  out +=
      "  \"layout\": \"relaxation rows: the model rows in order, then four per product; a "
      "bilinear product a<b gives w>=lb*xa+la*xb-la*lb, w>=ub*xa+ua*xb-ua*ub, "
      "w<=ub*xa+la*xb-la*ub, w<=lb*xa+ua*xb-ua*lb; a square gives the secant w<=(l+u)x-l*u, "
      "then tangents w>=2tx-t*t at t=l, t=u, t=(l+u)/2; columns are x then one w per "
      "product\",\n";
  out += "  \"products\": [";
  for (std::size_t k = 0; k < p.products.size(); ++k) {
    out += fmt::format("{}[{}, {}]", k == 0 ? "" : ", ", p.products[k].a, p.products[k].b);
  }
  out += "],\n";
  if (in.incumbent.empty()) {
    out += "  \"incumbent\": null,\n";
  } else {
    out += "  \"incumbent\": [";
    for (std::size_t j = 0; j < in.incumbent.size(); ++j) {
      out += fmt::format("{}{}", j == 0 ? "" : ", ", number(in.incumbent[j]));
    }
    out += "],\n";
  }
  out += "  \"nodes\": [";
  for (std::size_t k = 0; k < in.nodes.size(); ++k) {
    const CertificateNode& node = in.nodes[k];
    out += k == 0 ? "\n" : ",\n";
    out +=
        fmt::format("    {{\"id\": {}, \"parent\": {}, \"side\": \"{}\"", node.id, node.parent,
                    node.side == 1 ? "low" : (node.side == 2 ? "high" : "root"));
    if (node.column >= 0) {
      out += fmt::format(", \"branch\": [{}, {}]", node.column, number(node.point));
    }
    if (node.proof != CertificateNode::Proof::kNone) {
      out += fmt::format(", \"proof\": \"{}\", \"y\": [",
                         node.proof == CertificateNode::Proof::kDual ? "dual" : "farkas");
      for (std::size_t q = 0; q < node.multipliers.size(); ++q) {
        out += fmt::format("{}[{}, {}]", q == 0 ? "" : ", ", node.multipliers[q].first,
                           number(node.multipliers[q].second));
      }
      out += "]";
    }
    out += "}";
  }
  out += "\n  ]\n}\n";
  return out;
}

}  // namespace sankhya::global
