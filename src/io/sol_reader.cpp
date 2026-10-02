// SPDX-License-Identifier: Apache-2.0
// SANKHYA - read the basis a .sol file carries, for `sankhya solve --warm-start` (#218).
//
// The .sol file writer.cpp produces lists every column and row with its basis status in the
// last field. Yesterday's plan is that file; today's re-solve starts the simplex from that
// basis instead of the slack basis (Chvatal, "Linear Programming", 1983, ch. 7, on restarting
// from an advanced basis). Only the statuses are read: values and duals are not, because the
// engine recomputes both from the basis and the model it is given. Names are matched, not
// positions, so a file written for a differently ordered model of the same rows and columns
// still seeds the right basis, and a file for a different model is refused by name.

#include <fmt/format.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "io/line_reader.hpp"
#include "io/token.hpp"
#include "sankhya/io.hpp"

namespace sankhya::io {
namespace {

/// The record's leading name, unquoted, and its remaining text. writer.cpp quotes a name
/// containing whitespace, a quote or a backslash, escaping the last two with a backslash.
bool leading_name(std::string_view line, std::string* name, std::string_view* rest) {
  line = trim(line);
  if (line.empty()) return false;
  if (line.front() != '"') {
    std::size_t end = 0;
    while (end < line.size() && !is_space(line[end])) ++end;
    *name = std::string(line.substr(0, end));
    *rest = line.substr(end);
    return true;
  }
  name->clear();
  for (std::size_t i = 1; i < line.size(); ++i) {
    if (line[i] == '\\' && i + 1 < line.size()) {
      name->push_back(line[++i]);
    } else if (line[i] == '"') {
      *rest = line.substr(i + 1);
      return true;
    } else {
      name->push_back(line[i]);
    }
  }
  return false;  // an opened quote never closed
}

bool parse_status(std::string_view text, BasisStatus* out) {
  static const std::unordered_map<std::string_view, BasisStatus> kNames = {
      {"basic", BasisStatus::kBasic},      {"at_lower", BasisStatus::kAtLower},
      {"at_upper", BasisStatus::kAtUpper}, {"free", BasisStatus::kNonbasicFree},
      {"fixed", BasisStatus::kFixed},      {"unknown", BasisStatus::kUnknown}};
  const auto it = kNames.find(text);
  if (it == kNames.end()) return false;
  *out = it->second;
  return true;
}

std::unordered_map<std::string, Index> index_by_name(const std::vector<std::string>& names,
                                                     Index count, char fallback_prefix) {
  std::unordered_map<std::string, Index> index;
  for (Index k = 0; k < count; ++k) {
    const auto u = static_cast<std::size_t>(k);
    const std::string name = u < names.size() && !names[u].empty()
                                 ? names[u]
                                 : fmt::format("{}{}", fallback_prefix, k);
    index.emplace(name, k);
  }
  return index;
}

}  // namespace

bool read_solution_basis(const std::string& path, const Model& model,
                         std::vector<BasisStatus>* col_status,
                         std::vector<BasisStatus>* row_status, std::string* error) {
  LineReader reader;
  if (!reader.open(path, error)) return false;
  const auto cols = index_by_name(model.col_names, model.num_cols(), 'C');
  const auto rows = index_by_name(model.row_names, model.num_rows(), 'R');
  col_status->assign(static_cast<std::size_t>(model.num_cols()), BasisStatus::kUnknown);
  row_status->assign(static_cast<std::size_t>(model.num_rows()), BasisStatus::kUnknown);

  enum class Section { kNone, kColumns, kRows, kOther } section = Section::kNone;
  bool saw_columns = false;
  bool saw_rows = false;
  std::string line;
  std::vector<std::string_view> tokens;
  while (reader.next(&line)) {
    const std::string_view text = trim(line);
    if (text.empty() || text.front() == '#') continue;
    tokenize(text, &tokens);
    if (tokens[0] == "begin") {
      const std::string_view what = tokens.size() > 1 ? tokens[1] : std::string_view();
      section = what == "columns" ? Section::kColumns
                : what == "rows"  ? Section::kRows
                                  : Section::kOther;
      saw_columns = saw_columns || section == Section::kColumns;
      saw_rows = saw_rows || section == Section::kRows;
      continue;
    }
    if (tokens[0] == "end") {
      section = Section::kNone;
      continue;
    }
    if (section != Section::kColumns && section != Section::kRows) continue;

    std::string name;
    std::string_view rest;
    if (!leading_name(text, &name, &rest)) {
      *error = reader.error_at("unterminated quoted name");
      return false;
    }
    tokenize(rest, &tokens);
    BasisStatus status = BasisStatus::kUnknown;
    if (tokens.empty() || !parse_status(tokens.back(), &status)) {
      *error = reader.error_at(fmt::format("no basis status on the record for '{}'", name));
      return false;
    }
    const auto& index = section == Section::kColumns ? cols : rows;
    const auto it = index.find(name);
    if (it == index.end()) {
      *error =
          reader.error_at(fmt::format("{} '{}' is not in the model, so this file is not "
                                      "a solution of it",
                                      section == Section::kColumns ? "column" : "row", name));
      return false;
    }
    (section == Section::kColumns ? *col_status
                                  : *row_status)[static_cast<std::size_t>(it->second)] = status;
  }
  if (!saw_columns || !saw_rows) {
    *error = fmt::format(
        "{}: no columns and rows sections: the file carries no point, so "
        "it carries no basis to start from",
        path);
    return false;
  }
  // A status the file never gave, or gave as `unknown`, is a hole the simplex cannot seed.
  std::size_t unknown = 0;
  for (const auto* v : {col_status, row_status}) {
    for (const BasisStatus s : *v) unknown += s == BasisStatus::kUnknown ? 1 : 0;
  }
  if (unknown > 0) {
    *error = fmt::format(
        "{}: {} column or row status(es) are missing or `unknown`: the file "
        "carries no complete basis (the interior point without crossover and "
        "PDHG produce none)",
        path, unknown);
    return false;
  }
  return true;
}

bool read_solution_point(const std::string& path, const Model& model,
                         std::vector<double>* col_values, std::string* error) {
  LineReader reader;
  if (!reader.open(path, error)) return false;
  const auto cols = index_by_name(model.col_names, model.num_cols(), 'C');
  col_values->assign(static_cast<std::size_t>(model.num_cols()),
                     std::numeric_limits<double>::quiet_NaN());

  enum class Section { kNone, kColumns, kOther } section = Section::kNone;
  bool saw_columns = false;
  std::string line;
  std::vector<std::string_view> tokens;
  while (reader.next(&line)) {
    const std::string_view text = trim(line);
    if (text.empty() || text.front() == '#') continue;
    tokenize(text, &tokens);
    if (tokens[0] == "begin") {
      const std::string_view what = tokens.size() > 1 ? tokens[1] : std::string_view();
      section = what == "columns" ? Section::kColumns : Section::kOther;
      saw_columns = saw_columns || section == Section::kColumns;
      continue;
    }
    if (tokens[0] == "end") {
      section = Section::kNone;
      continue;
    }
    if (section != Section::kColumns) continue;

    std::string name;
    std::string_view rest;
    if (!leading_name(text, &name, &rest)) {
      *error = reader.error_at("unterminated quoted name");
      return false;
    }
    tokenize(rest, &tokens);
    if (tokens.empty()) {
      *error = reader.error_at(fmt::format("no value on the record for '{}'", name));
      return false;
    }

    double value = 0.0;
    try {
      std::string token_str(tokens[0]);
      std::size_t pos;
      value = std::stod(token_str, &pos);
      if (pos != token_str.size()) {
        throw std::invalid_argument("not a full number");
      }
    } catch (...) {
      *error = reader.error_at(fmt::format("invalid value '{}' for column '{}'", tokens[0], name));
      return false;
    }

    const auto it = cols.find(name);
    if (it == cols.end()) {
      *error = reader.error_at(fmt::format(
          "column '{}' is not in the model, so this file is not a solution of it", name));
      return false;
    }
    (*col_values)[static_cast<std::size_t>(it->second)] = value;
  }

  if (!saw_columns) {
    *error = fmt::format("{}: no columns section: the file carries no point", path);
    return false;
  }

  std::size_t unknown = 0;
  for (const double v : *col_values) {
    if (std::isnan(v)) ++unknown;
  }
  if (unknown > 0) {
    *error = fmt::format(
        "{}: {} column(s) are missing values: the file carries no complete assignment",
        path, unknown);
    return false;
  }
  return true;
}

}  // namespace sankhya::io
