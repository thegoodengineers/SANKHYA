// SPDX-License-Identifier: Apache-2.0
// SANKHYA - MPS special ordered sets and semi-continuous columns (#754).
//
// References: IBM, "MPS file format" and the CPLEX and Xpress reference manuals' MPS
// extensions (public documentation, read for interface compatibility only); Beale and
// Tomlin (1970) for what the two set types mean.
//
// Three spellings reach this file:
//
//   SC bound        " SC BND x 40" in BOUNDS: x = 0 or lower <= x <= 40 (mps_bounds.cpp
//                   records it through mark_semicontinuous()).
//   SOS / SETS      a section of its own, one header per set and one line per member:
//                       SOS
//                        S2 SOS       curve     1
//                           curve     lam0      1
//                           curve     lam1:2
//                   The header is S1 or S2, the word SOS, the set's name and an optional
//                   priority. A member line is the set's name (optional), the column and its
//                   weight, the last two as "column weight" or "column:weight".
//   marker form     inside COLUMNS, " S1 SOS1 'MARKER' 'SOSORG'" opens a set and
//                   " SOSEND 'MARKER' 'SOSEND'" closes it; every column listed in between is
//                   a member, weighted 1, 2, 3, ... in the order the columns first appear.
//
// Weights order the members, and SOS2's "adjacent" is adjacency in that order, so each set
// is sorted by weight at the end; two equal weights leave the order undefined and are refused
// by Model::validate().

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "core/sc_sos.hpp"
#include "sankhya/model.hpp"

#include "mps_parser.hpp"
#include "token.hpp"

namespace sankhya::io {
namespace {

/// "S1" or "S2" (any case) as a set type, 0 otherwise.
[[nodiscard]] std::uint8_t sos_type_of(std::string_view token) {
  const std::string t = to_upper(token);
  if (t == "S1") return 1;
  if (t == "S2") return 2;
  return 0;
}

}  // namespace

void MpsParser::mark_semicontinuous(Index col) {
  if (col_semicontinuous_.size() < col_cost_.size())
    col_semicontinuous_.resize(col_cost_.size(), 0);
  col_semicontinuous_[static_cast<std::size_t>(col)] = 1;
}

bool MpsParser::do_sos(std::string* error) {
  const std::uint8_t type = sos_type_of(tok_[0]);
  // A set may be named S1 or S2 itself; a line that starts with an existing set's name is a
  // member line unless the word SOS follows.
  const bool names_a_set =
      std::any_of(sos_.begin(), sos_.end(), [&](const SosSet& s) { return s.name == tok_[0]; });
  const bool says_sos = tok_.size() >= 2 && to_upper(tok_[1]) == "SOS";
  const bool header = type != 0 && (says_sos || !names_a_set);
  if (header) {
    // S1|S2 [SOS] [name] [priority]
    std::size_t k = 1;
    if (k < tok_.size() && to_upper(tok_[k]) == "SOS") ++k;
    SosSet set;
    set.type = type;
    if (k < tok_.size()) set.name = std::string(tok_[k++]);
    if (k < tok_.size()) {
      if (!parse_double(tok_[k], &set.priority)) {
        *error = reader_.error_at(fmt::format("SOS priority '{}' is not a number", tok_[k]));
        return false;
      }
      ++k;
    }
    if (k != tok_.size()) {
      *error = reader_.error_at("SOS header has fields past the name and the priority");
      return false;
    }
    if (set.name.empty()) set.name = fmt::format("SOS{}", sos_.size() + 1);
    for (const SosSet& other : sos_) {
      if (other.name == set.name) {
        *error = reader_.error_at(fmt::format("duplicate SOS name '{}'", set.name));
        return false;
      }
    }
    sos_.push_back(std::move(set));
    sos_open_ = static_cast<Index>(sos_.size() - 1);
    return true;
  }

  if (sos_open_ < 0) {
    *error = reader_.error_at("SOS member line before any S1 or S2 header");
    return false;
  }
  // Optional leading set name, then "column weight" or "column:weight". A leading name
  // selects that set, so members may name any set already declared.
  std::size_t k = 0;
  if (tok_.size() == 3 || (tok_.size() == 2 && tok_[1].find(':') != std::string_view::npos)) {
    const auto named = std::find_if(sos_.begin(), sos_.end(),
                                    [&](const SosSet& s) { return s.name == tok_[0]; });
    if (named == sos_.end()) {
      *error =
          reader_.error_at(fmt::format("SOS member line names set '{}', which has no "
                                       "S1 or S2 header",
                                       tok_[0]));
      return false;
    }
    sos_open_ = static_cast<Index>(named - sos_.begin());
    k = 1;
  }
  SosSet& set = sos_[static_cast<std::size_t>(sos_open_)];
  std::string_view column_text;
  std::string_view weight_text;
  if (tok_.size() - k == 1) {
    const std::size_t colon = tok_[k].rfind(':');
    if (colon == std::string_view::npos) {
      *error =
          reader_.error_at("SOS member needs a weight, as 'column weight' or 'column:weight'");
      return false;
    }
    column_text = tok_[k].substr(0, colon);
    weight_text = tok_[k].substr(colon + 1);
  } else if (tok_.size() - k == 2) {
    column_text = tok_[k];
    weight_text = tok_[k + 1];
  } else {
    *error = reader_.error_at(fmt::format("SOS member line has {} fields", tok_.size()));
    return false;
  }
  const Index col = find_column(column_text);
  if (col < 0) {
    *error = reader_.error_at(fmt::format(
        "SOS set '{}' names column '{}', which has no COLUMNS entry", set.name, column_text));
    return false;
  }
  double weight = 0.0;
  if (!parse_double(weight_text, &weight)) {
    *error = reader_.error_at(fmt::format("SOS weight '{}' is not a number", weight_text));
    return false;
  }
  set.columns.push_back(col);
  set.weights.push_back(weight);
  return true;
}

bool MpsParser::do_sos_marker(bool* handled, std::string* error) {
  bool opens = false;
  bool closes = false;
  std::uint8_t type = 0;
  std::string name;
  for (const std::string_view raw : tok_) {
    const std::string_view bare = unquote(raw);
    const std::string t = to_upper(bare);
    if (t == "SOSORG") {
      opens = true;
    } else if (t == "SOSEND") {
      closes = true;
    } else if (t == "MARKER") {
      continue;
    } else if (sos_type_of(bare) != 0 && type == 0) {
      type = sos_type_of(bare);
    } else if (name.empty()) {
      name = std::string(bare);
    }
  }
  *handled = opens || closes;
  if (!*handled) return true;
  if (opens) {
    if (sos_marker_active_) {
      *error = reader_.error_at("SOSORG marker inside a set that has not been closed");
      return false;
    }
    if (type == 0) {
      *error = reader_.error_at("SOSORG marker names no set type; expected S1 or S2");
      return false;
    }
    SosSet set;
    set.type = type;
    set.name = name.empty() ? fmt::format("SOS{}", sos_.size() + 1) : name;
    sos_.push_back(std::move(set));
    sos_open_ = static_cast<Index>(sos_.size() - 1);
    sos_marker_active_ = true;
    return true;
  }
  if (!sos_marker_active_) {
    *error = reader_.error_at("SOSEND marker with no set open");
    return false;
  }
  sos_marker_active_ = false;
  sos_open_ = -1;
  return true;
}

void MpsParser::note_sos_marker_member(Index col) {
  SosSet& set = sos_[static_cast<std::size_t>(sos_open_)];
  if (std::find(set.columns.begin(), set.columns.end(), col) != set.columns.end()) return;
  set.columns.push_back(col);
  set.weights.push_back(static_cast<double>(set.columns.size()));
}

void MpsParser::finish_sc_sos() {
  model_->semicontinuous.clear();
  for (std::size_t j = 0; j < col_semicontinuous_.size(); ++j) {
    if (col_semicontinuous_[j] != 0) model_->semicontinuous.push_back(static_cast<Index>(j));
  }
  model_->sos.clear();
  for (const SosSet& set : sos_) model_->sos.push_back(sorted_by_weight(set));
}

}  // namespace sankhya::io
