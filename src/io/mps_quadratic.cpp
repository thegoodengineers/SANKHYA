// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the quadratic sections of an MPS/QPS file: QUADOBJ (the objective's Hessian)
// and QCMATRIX (a quadratic ROW, #514).
//
// do_quadratic() moved here unchanged from mps_reader.cpp when QCMATRIX was added, to keep
// that file from growing further past the project's ~600 line guideline.
//
// QCMATRIX is the CPLEX and Gurobi extension for a quadratic constraint (CPLEX file-format
// reference, "QCMATRIX section"; Gurobi reference manual, "MPS format"): a header
// `QCMATRIX <row>` followed by `col col value` lines that list the FULL symmetric matrix Q_r
// of that row - both triangles, and WITHOUT the factor 1/2 the objective's QUADOBJ carries -
// so the row reads a_r'x + x'Q_r x. An entry (i, j) therefore contributes value * x_i * x_j,
// and a product x_i x_j with coefficient c appears as the two entries (i, j) and (j, i) of
// c/2 each. The reader sums what the file lists into one term per product, which is exact
// whether or not the file wrote the matrix symmetrically; what it refuses is the same entry
// listed twice, since MPS has no accumulate semantics.
//
// A parser with no sink for quadratic rows (every caller of read_mps) still refuses the
// section by name, as it did before #514: reading the rows and dropping their products
// would solve a different model and report its optimum as this one.

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "sankhya/io.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/qcqp.hpp"

#include "mps_parser.hpp"
#include "token.hpp"

namespace sankhya::io {

bool MpsParser::do_quadratic(std::string* error) {
  // QPS QUADOBJ: "colname1 colname2 value", giving one entry of the Hessian of the OBJECTIVE.
  //
  // THE CONVENTION, and it is the trap in this section. QPS states the objective as
  //
  //     c'x + 0.5 x' Q x
  //
  // and lists only the LOWER TRIANGLE of the symmetric Q. A stored off-diagonal entry
  // therefore stands for TWO entries of Q, and the 0.5 is part of the objective rather than
  // part of the data. `sankhya::Model` was defined in exactly this convention - see the note
  // in model.hpp - so entries map across with no transformation at all. A reader that
  // "helpfully" halved the off-diagonals, or mirrored them into both triangles, would produce
  // a model that solves cleanly to the optimum of a different problem.
  //
  // Files differ on which order the two column names appear in, so the pair is normalised to
  // (max, min) rather than trusted.
  if (tok_.size() < 3) {
    *error = reader_.error_at(fmt::format(
        "QUADOBJ entry has {} field(s), expected 3 (column, column, value)", tok_.size()));
    return false;
  }

  const Index first = find_column(tok_[0]);
  const Index second = find_column(tok_[1]);
  if (first < 0 || second < 0) {
    *error = reader_.error_at(
        fmt::format("QUADOBJ names column '{}' which never appeared in COLUMNS",
                    first < 0 ? std::string(tok_[0]) : std::string(tok_[1])));
    return false;
  }

  double value = 0.0;
  if (!parse_double(tok_[2], &value)) {
    *error = reader_.error_at(fmt::format("'{}' is not a number", tok_[2]));
    return false;
  }
  if (!std::isfinite(value)) {
    *error = reader_.error_at(fmt::format(
        "Hessian entry for ('{}', '{}') is {}; it must be finite", tok_[0], tok_[1], value));
    return false;
  }

  const Index row = std::max(first, second);
  const Index col = std::min(first, second);
  const auto key = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(row)) << 32) |
                   static_cast<std::uint64_t>(static_cast<std::uint32_t>(col));
  if (!seen_quad_entries_.insert(key).second) {
    // A file listing both (i, j) and (j, i) hits this, which is the point: the two name the
    // same entry of a symmetric matrix, and summing them would double the coefficient.
    *error = reader_.error_at(fmt::format(
        "duplicate Hessian entry for columns '{}' and '{}'; QPS lists the lower triangle of a "
        "symmetric matrix, so each pair may appear only once and has no accumulate semantics",
        col_names_[static_cast<std::size_t>(row)], col_names_[static_cast<std::size_t>(col)]));
    return false;
  }

  if (value != 0.0) {
    quad_row_.push_back(row);
    quad_col_.push_back(col);
    quad_value_.push_back(value);
  }
  return true;
}

bool MpsParser::begin_qcmatrix(std::string* error) {
  if (tok_.size() < 2) {
    *error = reader_.error_at("QCMATRIX header names no row");
    return false;
  }
  const Index row = find_row(tok_[1]);
  if (row == kObjectiveRow) {
    *error = reader_.error_at(fmt::format(
        "QCMATRIX names the objective row '{}'; the objective's quadratic part belongs in "
        "QUADOBJ, whose convention (lower triangle, factor 1/2) is different",
        tok_[1]));
    return false;
  }
  if (row < 0) {
    *error = reader_.error_at(
        fmt::format("QCMATRIX names row '{}', which is not a constraint row of ROWS", tok_[1]));
    return false;
  }
  if (!qc_rows_seen_.insert(row).second) {
    *error = reader_.error_at(
        fmt::format("a second QCMATRIX section for row '{}'; each row has one", tok_[1]));
    return false;
  }
  qc_row_ = row;
  return true;
}

bool MpsParser::do_qcmatrix(std::string* error) {
  if (tok_.size() < 3) {
    *error = reader_.error_at(fmt::format(
        "QCMATRIX entry has {} field(s), expected 3 (column, column, value)", tok_.size()));
    return false;
  }
  const Index first = find_column(tok_[0]);
  const Index second = find_column(tok_[1]);
  if (first < 0 || second < 0) {
    *error = reader_.error_at(
        fmt::format("QCMATRIX names column '{}' which never appeared in COLUMNS",
                    first < 0 ? std::string(tok_[0]) : std::string(tok_[1])));
    return false;
  }
  double value = 0.0;
  if (!parse_double(tok_[2], &value)) {
    *error = reader_.error_at(fmt::format("'{}' is not a number", tok_[2]));
    return false;
  }
  if (!std::isfinite(value)) {
    *error = reader_.error_at(fmt::format(
        "QCMATRIX entry for ('{}', '{}') is {}; it must be finite", tok_[0], tok_[1], value));
    return false;
  }
  // (i, j) and (j, i) are DIFFERENT entries of the full matrix and both are expected; the
  // same ordered pair twice is a repeated entry.
  if (!qc_seen_entries_.insert({qc_row_, first, second}).second) {
    *error = reader_.error_at(fmt::format(
        "duplicate QCMATRIX entry ('{}', '{}') in row '{}'; MPS has no accumulate semantics",
        tok_[0], tok_[1], row_names_[static_cast<std::size_t>(qc_row_)]));
    return false;
  }
  if (value != 0.0) {
    qc_entries_.push_back({qc_row_, first, second, value});
  }
  return true;
}

void MpsParser::finish_quadratic_rows() {
  // x'Qx = sum_ij Q_ij x_i x_j, so the product x_a x_b (a < b) collects Q_ab + Q_ba and the
  // square x_a^2 collects Q_aa. Summed in a map so the result is ordered by (row, a, b).
  std::map<std::tuple<Index, Index, Index>, double> sums;
  for (const QuadraticTerm& entry : qc_entries_) {
    const Index a = std::min(entry.first, entry.second);
    const Index b = std::max(entry.first, entry.second);
    sums[{entry.row, a, b}] += entry.value;
  }
  qc_sink_->clear();
  for (const auto& [key, value] : sums) {
    // (i, j) = +c and (j, i) = -c cancel to an exact zero: that product is not in the row.
    if (value == 0.0) continue;
    qc_sink_->push_back({std::get<0>(key), std::get<1>(key), std::get<2>(key), value});
  }
}

ReadResult read_qcqp_mps(const std::string& path, QcqpModel* model, MpsFormat format) {
  const auto parse_with = [&](MpsFormat dialect) {
    MpsParser parser(&model->linear, dialect);
    parser.collect_quadratic_rows(&model->quadratic);
    return parser.parse(path);
  };
  model->quadratic.clear();
  ReadResult result;
  if (format != MpsFormat::kAuto) {
    result = parse_with(format);
  } else {
    // The same order and the same fallback rule as read_mps().
    result = parse_with(MpsFormat::kFree);
    if (!result.ok && !result.refused) {
      const ReadResult fixed = parse_with(MpsFormat::kFixed);
      if (fixed.ok) {
        default_logger().info("{}: parsed as fixed-format MPS", path);
        result = fixed;
      } else {
        result = ReadResult::failure(fmt::format("{} (the fixed-format reader also failed: {})",
                                                 result.error, fixed.error));
      }
    }
  }
  if (!result.ok) return result;
  const std::string problem = model->validate();
  if (!problem.empty()) {
    return ReadResult::failure(fmt::format("{}: model failed validation: {}", path, problem));
  }
  return result;
}

ReadResult read_qcqp_model(const std::string& path, QcqpModel* model) {
  const auto ends_with = [&](const char* suffix) {
    const std::string s(suffix);
    return path.size() >= s.size() && path.compare(path.size() - s.size(), s.size(), s) == 0;
  };
  if (ends_with(".lp") || ends_with(".LP") || ends_with(".lp.gz") || ends_with(".LP.gz")) {
    model->quadratic.clear();
    return read_lp(path, &model->linear);
  }
  return read_qcqp_mps(path, model);
}

}  // namespace sankhya::io
