// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the debug-solution check (#500). See debug_solution.hpp for what is checked and
// why; this file reads the point and tests it.
#include "debug_solution.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <utility>

#include <fmt/format.h>

#include "presolve/presolve.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::mip {

namespace {

/// The name a column is written under in a solution file: its own, or `C<index>`, exactly as
/// src/io/writer.cpp prints an unnamed one.
std::string column_name(const Model& model, Index j) {
  const auto u = static_cast<std::size_t>(j);
  if (u < model.col_names.size() && !model.col_names[u].empty()) return model.col_names[u];
  return fmt::format("C{}", j);
}

std::string row_name(const Model& model, Index i) {
  const auto u = static_cast<std::size_t>(i);
  if (u < model.row_names.size() && !model.row_names[u].empty()) return model.row_names[u];
  return fmt::format("R{}", i);
}

/// One whitespace-delimited field of `line` from `pos`, honouring the writer's quoting (a
/// name with blanks is written in double quotes with \" and \\ escaped). False at the end.
bool next_field(const std::string& line, std::size_t* pos, std::string* field) {
  std::size_t p = *pos;
  while (p < line.size() && std::isspace(static_cast<unsigned char>(line[p])) != 0) ++p;
  if (p >= line.size()) return false;
  field->clear();
  if (line[p] == '"') {
    ++p;
    while (p < line.size() && line[p] != '"') {
      if (line[p] == '\\' && p + 1 < line.size()) ++p;
      field->push_back(line[p++]);
    }
    if (p < line.size()) ++p;  // the closing quote
  } else {
    while (p < line.size() && std::isspace(static_cast<unsigned char>(line[p])) == 0) {
      field->push_back(line[p++]);
    }
  }
  *pos = p;
  return true;
}

bool parse_value(const std::string& text, double* value) {
  if (text.empty()) return false;
  char* end = nullptr;
  *value = std::strtod(text.c_str(), &end);
  return end != nullptr && *end == '\0' && std::isfinite(*value);
}

/// name -> value from a SANKHYA .sol (the `begin columns` block only) or from plain
/// `name value` lines (a MIPLIB .sol; `#` comments and the `=obj=` line skipped).
bool read_point(const std::string& path, std::unordered_map<std::string, double>* values,
                std::string* error) {
  std::ifstream in(path);
  if (!in) {
    *error = fmt::format("{}: cannot open the debug solution", path);
    return false;
  }
  std::vector<std::string> lines;
  bool sankhya_format = false;
  for (std::string line; std::getline(in, line);) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.rfind("begin columns", 0) == 0) sankhya_format = true;
    lines.push_back(std::move(line));
  }
  bool inside = !sankhya_format;
  for (const std::string& line : lines) {
    if (sankhya_format) {
      if (line.rfind("begin columns", 0) == 0) {
        inside = true;
        continue;
      }
      if (line.rfind("end columns", 0) == 0) break;
    }
    if (!inside || line.empty() || line[0] == '#' || line[0] == '=') continue;
    std::size_t pos = 0;
    std::string name;
    std::string text;
    if (!next_field(line, &pos, &name) || !next_field(line, &pos, &text)) continue;
    double value = 0.0;
    if (!parse_value(text, &value)) continue;
    (*values)[name] = value;
  }
  return true;
}

/// The scale a violation is measured against: the right-hand side and the largest term of
/// the activity, so a row of large coefficients is not held to an absolute 1e-6.
double scale_of(double rhs, double largest_term) {
  return std::max({1.0, std::fabs(rhs), largest_term});
}

std::mutex& planted_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::optional<Cut>& planted() {
  static std::optional<Cut> cut;
  return cut;
}

const char* record_kind_name(presolve::Record::Kind kind, bool* optimality_based) {
  using Kind = presolve::Record::Kind;
  *optimality_based = false;
  switch (kind) {
    case Kind::kFixedColumn: return "fixed column";
    case Kind::kEmptyColumn:
      *optimality_based = true;
      return "empty column parked at the bound its cost prefers";
    case Kind::kDualFixedColumn: *optimality_based = true; return "dual fixing";
    case Kind::kDominatedColumn: *optimality_based = true; return "dominated column";
    default: return nullptr;  // a kind that fixes no column by itself
  }
}

}  // namespace

std::optional<DebugSolution> DebugSolution::load(const std::string& path, const Model& model,
                                                 std::string* error, Index* missing) {
  std::unordered_map<std::string, double> values;
  if (!read_point(path, &values, error)) return std::nullopt;
  std::vector<double> x(static_cast<std::size_t>(model.num_cols()), 0.0);
  Index absent = 0;
  for (Index j = 0; j < model.num_cols(); ++j) {
    const auto found = values.find(column_name(model, j));
    if (found == values.end()) {
      ++absent;
      continue;
    }
    x[static_cast<std::size_t>(j)] = found->second;
  }
  if (model.num_cols() > 0 && absent == model.num_cols()) {
    *error = fmt::format("{}: names none of the model's {} columns", path, model.num_cols());
    return std::nullopt;
  }
  if (missing != nullptr) *missing = absent;
  return DebugSolution(std::move(x));
}

std::string DebugSolution::cut_violation(const Cut& cut) const {
  double activity = 0.0;
  double largest = 0.0;
  const std::size_t n = std::min(cut.coeff.size(), x_.size());
  for (std::size_t j = 0; j < n; ++j) {
    const double term = cut.coeff[j] * x_[j];
    activity += term;
    largest = std::max(largest, std::fabs(term));
  }
  const double excess = activity - cut.rhs;
  if (excess <= tol::kDebugSolutionTolerance * scale_of(cut.rhs, largest)) return {};
  return fmt::format("activity {:.17g} > rhs {:.17g} (by {:.3e})", activity, cut.rhs, excess);
}

std::string DebugSolution::model_violation(const Model& model) const {
  if (static_cast<Index>(x_.size()) != model.num_cols()) {
    return fmt::format("the point has {} values for {} columns", x_.size(), model.num_cols());
  }
  for (Index j = 0; j < model.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double v = x_[u];
    const double lo = model.col_lower[u];
    const double hi = model.col_upper[u];
    if (is_finite_bound(lo) && v < lo - tol::kDebugSolutionTolerance * scale_of(lo, 0.0)) {
      return fmt::format("column {} = {:.17g} is below its lower bound {:.17g}",
                         column_name(model, j), v, lo);
    }
    if (is_finite_bound(hi) && v > hi + tol::kDebugSolutionTolerance * scale_of(hi, 0.0)) {
      return fmt::format("column {} = {:.17g} is above its upper bound {:.17g}",
                         column_name(model, j), v, hi);
    }
    if (model.col_type[u] == VarType::kInteger &&
        std::fabs(v - std::round(v)) > tol::kIntegrality) {
      return fmt::format("integer column {} = {:.17g} is fractional", column_name(model, j), v);
    }
  }
  std::vector<double> activity(static_cast<std::size_t>(model.num_rows()), 0.0);
  std::vector<double> largest(static_cast<std::size_t>(model.num_rows()), 0.0);
  for (Index j = 0; j < model.num_cols(); ++j) {
    const ColumnView column = model.matrix.column(j);
    for (Index k = 0; k < column.size; ++k) {
      const auto r = static_cast<std::size_t>(column.rows[k]);
      const double term = column.values[k] * x_[static_cast<std::size_t>(j)];
      activity[r] += term;
      largest[r] = std::max(largest[r], std::fabs(term));
    }
  }
  for (Index i = 0; i < model.num_rows(); ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double lo = model.row_lower[u];
    const double hi = model.row_upper[u];
    if (is_finite_bound(lo) &&
        activity[u] < lo - tol::kDebugSolutionTolerance * scale_of(lo, largest[u])) {
      return fmt::format("row {} (index {}) activity {:.17g} is below its lower bound {:.17g}",
                         row_name(model, i), i, activity[u], lo);
    }
    if (is_finite_bound(hi) &&
        activity[u] > hi + tol::kDebugSolutionTolerance * scale_of(hi, largest[u])) {
      return fmt::format("row {} (index {}) activity {:.17g} is above its upper bound {:.17g}",
                         row_name(model, i), i, activity[u], hi);
    }
  }
  return {};
}

Index DebugSolution::first_column_outside(const std::vector<double>& lower,
                                          const std::vector<double>& upper) const {
  const std::size_t n = std::min({x_.size(), lower.size(), upper.size()});
  for (std::size_t j = 0; j < n; ++j) {
    const double v = x_[j];
    if (is_finite_bound(lower[j]) &&
        v < lower[j] - tol::kDebugSolutionTolerance * scale_of(lower[j], 0.0)) {
      return static_cast<Index>(j);
    }
    if (is_finite_bound(upper[j]) &&
        v > upper[j] + tol::kDebugSolutionTolerance * scale_of(upper[j], 0.0)) {
      return static_cast<Index>(j);
    }
  }
  return -1;
}

void debug_solution_abort(Logger& logger, const std::string& what) {
  const std::string message = "debug solution cut off: " + what;
  logger.error("{}", message);
  std::fprintf(stderr, "%s\n", message.c_str());
  std::fflush(stderr);
  std::abort();
}

std::optional<DebugSolution> load_debug_solution(const Options& options, const Model& model,
                                                 Logger& logger) {
  const std::string& path = options.get_string("debug_solution");
  if (path.empty()) return std::nullopt;
  std::string error;
  Index missing = 0;
  std::optional<DebugSolution> point = DebugSolution::load(path, model, &error, &missing);
  if (!point.has_value()) {
    const std::string message = "debug_solution: " + error;
    logger.error("{}", message);
    std::fprintf(stderr, "%s\n", message.c_str());
    std::fflush(stderr);
    std::abort();
  }
  if (missing > 0) {
    logger.verbose("debug_solution: {} of {} column(s) not named in the file, taken as 0",
                   missing, model.num_cols());
  }
  return point;
}

void check_presolve_against_debug_solution(const Model& original,
                                           const presolve::Result& reduced,
                                           const Options& options, Logger& logger) {
  const std::optional<DebugSolution> point = load_debug_solution(options, original, logger);
  if (!point.has_value()) return;
  if (const std::string bad = point->model_violation(original); !bad.empty()) {
    debug_solution_abort(logger,
                         "the debug solution is not feasible for the model as given: " + bad);
  }
  if (reduced.proved_infeasible) {
    debug_solution_abort(
        logger, fmt::format("presolve proved the model infeasible ({}) and the debug solution "
                            "is feasible",
                            reduced.message));
  }
  // Record by record, in the order presolve applied them: a column fixed at a value the
  // point does not take is the reduction that removed it.
  for (std::size_t k = 0; k < reduced.records.size(); ++k) {
    const presolve::Record& record = reduced.records[k];
    bool optimality_based = false;
    const char* kind = record_kind_name(record.kind, &optimality_based);
    if (kind == nullptr || record.index < 0 || record.index >= original.num_cols()) continue;
    const double v = point->x()[static_cast<std::size_t>(record.index)];
    if (std::fabs(v - record.value) <=
        tol::kDebugSolutionTolerance * scale_of(record.value, std::fabs(v))) {
      continue;
    }
    debug_solution_abort(
        logger,
        fmt::format("presolve reduction {} ({}) fixed column {} at {:.17g}; the debug solution "
                    "has {:.17g}{}",
                    k, kind, column_name(original, record.index), record.value, v,
                    optimality_based
                        ? " (an optimality-based reduction: legitimate unless the debug "
                          "solution is the unique optimum)"
                        : ""));
  }
  std::vector<double> x(reduced.col_to_original.size());
  for (std::size_t j = 0; j < x.size(); ++j) {
    x[j] = point->x()[static_cast<std::size_t>(reduced.col_to_original[j])];
  }
  if (const std::string bad = DebugSolution(std::move(x)).model_violation(reduced.model);
      !bad.empty()) {
    debug_solution_abort(logger, "the presolved model excludes the debug solution: " + bad);
  }
}

Model with_column_names(const Model& model) {
  Model named = model;
  named.col_names.resize(static_cast<std::size_t>(model.num_cols()));
  for (Index j = 0; j < model.num_cols(); ++j) {
    auto& name = named.col_names[static_cast<std::size_t>(j)];
    if (name.empty()) name = fmt::format("C{}", j);
  }
  return named;
}

namespace testing {
void set_planted_cut(std::optional<Cut> cut) {
  const std::lock_guard<std::mutex> lock(planted_mutex());
  planted() = std::move(cut);
}
}  // namespace testing

void append_planted_cut(std::vector<Cut>* accepted) {
  const std::lock_guard<std::mutex> lock(planted_mutex());
  if (planted().has_value()) accepted->push_back(*planted());
}

}  // namespace sankhya::mip
