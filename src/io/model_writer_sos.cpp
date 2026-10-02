// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the model writers' semi-continuous and SOS output (#754). See
// model_writer_sos.hpp; the readers are mps_sos.cpp and lp_sos.cpp, and a round trip through
// write_model() and read_model() must give the same model back (tests/unit/test_sc_sos_io.cpp).

#include "model_writer_sos.hpp"

#include <fmt/format.h>

namespace sankhya::io {
namespace {

/// Full precision, and the spellings of infinity both readers accept.
std::string number(double v) {
  if (v == kInfinity) return "inf";
  if (v == -kInfinity) return "-inf";
  return fmt::format("{:.17g}", v == 0.0 ? 0.0 : v);
}

std::string set_name(const SosSet& set, std::size_t s) {
  return set.name.empty() ? fmt::format("SOS{}", s + 1) : set.name;
}

}  // namespace

void write_mps_semicontinuous_bounds(std::FILE* out, const std::string& name, double lower,
                                     double upper, bool integer) {
  if (lower != 0.0)
    fmt::print(out, " {} BND  {}  {}\n", integer ? "LI" : "LO", name, number(lower));
  fmt::print(out, " SC BND  {}  {}\n", name, number(upper));
}

void write_mps_sos(std::FILE* out, const Model& model,
                   const std::vector<std::string>& columns) {
  if (model.sos.empty()) return;
  fmt::print(out, "SOS\n");
  for (std::size_t s = 0; s < model.sos.size(); ++s) {
    const SosSet& set = model.sos[s];
    const std::string name = set_name(set, s);
    fmt::print(out, " S{} SOS  {}  {}\n", static_cast<int>(set.type), name,
               number(set.priority));
    for (std::size_t k = 0; k < set.columns.size(); ++k) {
      fmt::print(out, "    {}  {}  {}\n", name,
                 columns[static_cast<std::size_t>(set.columns[k])], number(set.weights[k]));
    }
  }
}

void write_lp_sc_sos(std::FILE* out, const Model& model,
                     const std::vector<std::string>& columns) {
  if (!model.semicontinuous.empty()) {
    fmt::print(out, "Semi-Continuous\n");
    for (const Index j : model.semicontinuous) {
      fmt::print(out, "  {}\n", columns[static_cast<std::size_t>(j)]);
    }
  }
  if (model.sos.empty()) return;
  // The LP format has no field for a set's priority; it is the one thing a round trip
  // through this format does not keep.
  fmt::print(out, "SOS\n");
  for (std::size_t s = 0; s < model.sos.size(); ++s) {
    const SosSet& set = model.sos[s];
    std::string line = fmt::format("  {}: S{}::", set_name(set, s), static_cast<int>(set.type));
    for (std::size_t k = 0; k < set.columns.size(); ++k) {
      line += fmt::format(" {}:{}", columns[static_cast<std::size_t>(set.columns[k])],
                          number(set.weights[k]));
    }
    fmt::print(out, "{}\n", line);
  }
}

}  // namespace sankhya::io
