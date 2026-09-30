// SPDX-License-Identifier: Apache-2.0
// SANKHYA - `sankhya scenarios MODEL SCENARIOS.csv` (#752): the file format and the table.
// The solving and the checking are src/core/scenarios.cpp.
//
// SCENARIOS.csv: a header row, then one row per scenario. The first column is the scenario's
// name. Every other header names what its column overrides:
//
//     cost:COL        the objective coefficient of column COL
//     col_lower:COL   col_upper:COL      a column bound
//     row_lower:ROW   row_upper:ROW      a row bound
//     rhs:ROW         every finite side of row ROW (both sides of an equality)
//
// An empty cell keeps the model's own value. Lines starting with '#' are comments. Plain
// comma-separated values: no quoting, so names and values may not contain commas.

#include "scenarios_cli.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "core/scenarios.hpp"
#include "sankhya/io.hpp"
#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"
#include "sankhya/version.hpp"

namespace sankhya::cli {

namespace {

namespace sc = sankhya::scenarios;

std::vector<std::string> split(const std::string& line) {
  std::vector<std::string> cells;
  std::stringstream stream(line);
  std::string cell;
  while (std::getline(stream, cell, ',')) {
    const auto first = cell.find_first_not_of(" \t\r");
    const auto last = cell.find_last_not_of(" \t\r");
    cells.push_back(first == std::string::npos ? "" : cell.substr(first, last - first + 1));
  }
  if (!line.empty() && line.back() == ',') cells.emplace_back();
  return cells;
}

std::map<std::string, Index> index_of(const std::vector<std::string>& names) {
  std::map<std::string, Index> map;
  for (std::size_t i = 0; i < names.size(); ++i) map.emplace(names[i], static_cast<Index>(i));
  return map;
}

struct Column {
  sc::Target target;
  Index index;
};

/// Reads the scenario file against `model`'s names. Prints the first problem and returns
/// false.
bool read_scenarios(const std::string& path, const Model& model,
                    std::vector<sc::Scenario>* scenarios) {
  std::ifstream in(path);
  if (!in) {
    fmt::print(stderr, "error: cannot open {}\n", path);
    return false;
  }
  const auto cols = index_of(model.col_names);
  const auto rows = index_of(model.row_names);
  const std::map<std::string, sc::Target> kinds = {
      {"cost", sc::Target::kCost},          {"col_lower", sc::Target::kColLower},
      {"col_upper", sc::Target::kColUpper}, {"row_lower", sc::Target::kRowLower},
      {"row_upper", sc::Target::kRowUpper}, {"rhs", sc::Target::kRhs}};

  std::vector<Column> header;
  std::string line;
  int line_number = 0;
  bool have_header = false;
  while (std::getline(in, line)) {
    ++line_number;
    if (line.find_first_not_of(" \t\r") == std::string::npos || line[0] == '#') continue;
    const std::vector<std::string> cells = split(line);
    if (!have_header) {
      have_header = true;
      for (std::size_t c = 1; c < cells.size(); ++c) {
        const auto colon = cells[c].find(':');
        const auto kind =
            colon == std::string::npos ? kinds.end() : kinds.find(cells[c].substr(0, colon));
        if (kind == kinds.end()) {
          fmt::print(stderr,
                     "error: {}:{}: header '{}' is not cost:, col_lower:, col_upper:, "
                     "row_lower:, row_upper: or rhs: followed by a name\n",
                     path, line_number, cells[c]);
          return false;
        }
        const std::string name = cells[c].substr(colon + 1);
        const bool on_column = kind->second == sc::Target::kCost ||
                               kind->second == sc::Target::kColLower ||
                               kind->second == sc::Target::kColUpper;
        const auto& names = on_column ? cols : rows;
        const auto found = names.find(name);
        if (found == names.end()) {
          fmt::print(stderr, "error: {}:{}: the model has no {} named '{}'\n", path,
                     line_number, on_column ? "column" : "row", name);
          return false;
        }
        header.push_back({kind->second, found->second});
      }
      continue;
    }
    if (cells.size() > header.size() + 1) {
      fmt::print(stderr, "error: {}:{}: {} cells, the header has {}\n", path, line_number,
                 cells.size(), header.size() + 1);
      return false;
    }
    sc::Scenario scenario;
    scenario.name = cells.empty() || cells[0].empty()
                        ? fmt::format("s{}", scenarios->size() + 1)
                        : cells[0];
    for (std::size_t c = 1; c < cells.size(); ++c) {
      if (cells[c].empty()) continue;
      char* end = nullptr;
      const double value = std::strtod(cells[c].c_str(), &end);
      if (end == cells[c].c_str() || *end != '\0' || std::isnan(value)) {
        fmt::print(stderr, "error: {}:{}: '{}' is not a number\n", path, line_number, cells[c]);
        return false;
      }
      scenario.overrides.push_back({header[c - 1].target, header[c - 1].index, value});
    }
    scenarios->push_back(std::move(scenario));
  }
  if (scenarios->empty()) {
    fmt::print(stderr, "error: {} holds no scenario\n", path);
    return false;
  }
  return true;
}

std::string name_or_index(const std::vector<std::string>& names, std::size_t i, char prefix) {
  return i < names.size() && !names[i].empty() ? names[i] : fmt::format("{}{}", prefix, i);
}

/// The rows the answer prices, largest shadow price first: "NAME(price);...", at most three.
std::string binding(const Model& model, const Solution& s) {
  std::vector<std::pair<double, std::size_t>> priced;
  for (std::size_t i = 0; i < s.row_dual.size(); ++i) {
    if (std::fabs(s.row_dual[i]) > tol::kDualFeasibility) priced.emplace_back(s.row_dual[i], i);
  }
  std::sort(priced.begin(), priced.end(), [](const auto& a, const auto& b) {
    return std::fabs(a.first) > std::fabs(b.first);
  });
  std::string text;
  for (std::size_t k = 0; k < priced.size() && k < 3; ++k) {
    text += fmt::format("{}{}({:.6g})", k ? ";" : "",
                        name_or_index(model.row_names, priced[k].second, 'R'), priced[k].first);
  }
  return text.empty() ? "-" : text;
}

/// How many decisions moved from the base plan, and the three that moved most.
std::string changed(const Model& model, const Solution& base, const Solution& s) {
  if (base.col_value.size() != s.col_value.size()) return "-";
  std::vector<std::pair<double, std::size_t>> moved;
  for (std::size_t j = 0; j < s.col_value.size(); ++j) {
    const double delta = s.col_value[j] - base.col_value[j];
    if (std::fabs(delta) > tol::kPrimalFeasibility * (1.0 + std::fabs(base.col_value[j]))) {
      moved.emplace_back(delta, j);
    }
  }
  std::sort(moved.begin(), moved.end(), [](const auto& a, const auto& b) {
    return std::fabs(a.first) > std::fabs(b.first);
  });
  std::string text = fmt::format("{}", moved.size());
  for (std::size_t k = 0; k < moved.size() && k < 3; ++k) {
    text += fmt::format("{}{}({:+.6g})", k ? ";" : ":",
                        name_or_index(model.col_names, moved[k].second, 'C'), moved[k].first);
  }
  return text;
}

std::string status_label(const sc::ScenarioResult& r) {
  if (r.verified) return to_string(r.solution.status);
  return fmt::format("unverified-{}", to_string(r.solution.status));
}

std::string number(double v) {
  return std::isfinite(v) ? fmt::format("{:.12g}", v) : std::string("-");
}

}  // namespace

int run_scenarios(const Model& model, const std::string& model_path,
                  const std::string& csv_path, const Options& options,
                  const std::string& out_path, bool compare) {
  std::vector<sc::Scenario> scenarios;
  if (!read_scenarios(csv_path, model, &scenarios)) return 3;

  const sc::BatchReport report = sc::solve_all(model, scenarios, options);

  // The separate single solves, when asked: each re-reads the file and starts from nothing.
  std::vector<Solution> single(compare ? scenarios.size() : 0);
  std::vector<double> single_wall(single.size(), 0.0);
  double single_seconds = 0.0;
  for (std::size_t k = 0; k < single.size(); ++k) {
    const Timer clock;
    Model fresh;
    const io::ReadResult read = io::read_model(model_path, &fresh);
    if (!read.ok) {
      fmt::print(stderr, "error: {}\n", read.error);
      return 3;
    }
    single[k] = solve(sc::apply(fresh, scenarios[k]), options);
    single_wall[k] = clock.elapsed_seconds();
    single_seconds += single_wall[k];
  }

  std::string header =
      "scenario,status,objective,safe_bound,verified,verdict,resolved_cold,iterations,seconds,"
      "binding,changed";
  if (compare) header += ",single_status,single_objective,single_seconds,rel_diff,agrees";
  std::vector<std::string> lines;
  std::size_t unverified = 0;
  std::size_t disagreements = 0;
  for (std::size_t k = 0; k < scenarios.size(); ++k) {
    const sc::ScenarioResult& r = report.results[k];
    const bool point = r.verified && r.solution.status == SolveStatus::kOptimal;
    if (!r.verified) ++unverified;
    std::string verdict = r.verdict;
    std::replace(verdict.begin(), verdict.end(), ',', ';');
    std::string line =
        fmt::format("{},{},{},{},{},{},{},{},{:.6f},{},{}", scenarios[k].name, status_label(r),
                    point ? number(r.solution.objective) : "-", number(r.safe_bound),
                    r.verified ? "yes" : "no", verdict, r.resolved_cold ? "yes" : "no",
                    r.solution.iterations, r.seconds, point ? binding(model, r.solution) : "-",
                    point ? changed(model, report.base, r.solution) : "-");
    if (compare) {
      const Solution& s = single[k];
      bool agrees = s.status == r.solution.status;
      double rel = 0.0;
      if (agrees && point) {
        rel = std::fabs(s.objective - r.solution.objective) /
              std::max(1.0, std::fabs(s.objective));
        agrees = rel <= tol::kScenarioAgreement;
      }
      if (!agrees) ++disagreements;
      line += fmt::format(",{},{},{:.6f},{:.3e},{}", to_string(s.status),
                          claims_a_point(s) ? number(s.objective) : "-", single_wall[k], rel,
                          agrees ? "yes" : "no");
    }
    lines.push_back(std::move(line));
  }

  if (!out_path.empty()) {
    std::FILE* file = std::fopen(out_path.c_str(), "w");
    if (file == nullptr) {
      fmt::print(stderr, "error: cannot write {}\n", out_path);
      return 3;
    }
    fmt::print(file, "{}\n", header);
    for (const std::string& line : lines) fmt::print(file, "{}\n", line);
    std::fclose(file);
  }
  fmt::print("{}\n", header);
  for (const std::string& line : lines) fmt::print("{}\n", line);

  fmt::print(
      "\n{} scenario(s) from one read, base solve {} in {} iteration(s): {:.4f}s total\n",
      scenarios.size(), to_string(report.base.status), report.base.iterations, report.seconds);
  if (compare) {
    fmt::print("{} separate single solve(s), each reading the file again: {:.4f}s total\n",
               single.size(), single_seconds);
    fmt::print("agreement with the single solves to {:.0e} relative: {} of {}\n",
               tol::kScenarioAgreement, single.size() - disagreements, single.size());
  }
  fmt::print("verified: {} of {} ({})\n", scenarios.size() - unverified, scenarios.size(),
             git_commit());
  return unverified == 0 && disagreements == 0 ? 0 : 1;
}

}  // namespace sankhya::cli
