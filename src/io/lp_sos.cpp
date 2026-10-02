// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the LP format's semi-continuous and SOS sections (#754).
//
// Reference: the LP file format as documented in the public CPLEX and Gurobi reference
// manuals (interface compatibility, read from documentation); Beale and Tomlin (1970) for
// what the two set types mean.
//
//   Semi-Continuous            (also "semi", "semis", "semicontinuous")
//    x y                       each listed column takes 0 or a value in its Bounds range
//   SOS
//    s1: S1:: x1:1 x2:2 x3:3   a named set of type 1, members "column:weight"
//    S2:: l0:0 l1:10 l2:20     an unnamed set of type 2
//
// A set's statement runs until the next one begins, so members may wrap across lines. The
// members are sorted by weight at the end, as the MPS reader sorts them.

#include <cstdint>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "core/sc_sos.hpp"
#include "sankhya/model.hpp"

#include "lp_parser.hpp"
#include "token.hpp"

namespace sankhya::io::lp {

bool LpParser::parse_semicontinuous(const SectionSpan& span, std::string* error) {
  for (std::size_t i = span.begin; i < span.end; ++i) {
    if (tokens_[i].kind != TokKind::kIdent) {
      *error = at(i, "expected a variable name in the semi-continuous section");
      return false;
    }
    const auto col = static_cast<std::size_t>(intern_column(tokens_[i].text));
    if (col_semicontinuous_.size() < col_names_.size()) {
      col_semicontinuous_.resize(col_names_.size(), 0);
    }
    col_semicontinuous_[col] = 1;
  }
  return true;
}

bool LpParser::parse_sos(const SectionSpan& span, std::string* error) {
  const auto is_op = [&](std::size_t k, const char* text) {
    return k < span.end && tokens_[k].kind == TokKind::kOp && tokens_[k].text == text;
  };
  const auto set_type = [&](std::size_t k) -> std::uint8_t {
    if (k >= span.end || tokens_[k].kind != TokKind::kIdent) return 0;
    const std::string t = to_upper(tokens_[k].text);
    return t == "S1" ? 1 : (t == "S2" ? 2 : 0);
  };
  // "S1 : :" starting at k, and "name : S1 : :" starting at k.
  const auto unnamed_header = [&](std::size_t k) {
    return set_type(k) != 0 && is_op(k + 1, ":") && is_op(k + 2, ":");
  };
  const auto named_header = [&](std::size_t k) {
    return k < span.end && tokens_[k].kind == TokKind::kIdent && is_op(k + 1, ":") &&
           unnamed_header(k + 2);
  };

  std::size_t i = span.begin;
  while (i < span.end) {
    SosSet set;
    if (named_header(i)) {
      set.name = tokens_[i].text;
      set.type = set_type(i + 2);
      i += 5;
    } else if (unnamed_header(i)) {
      set.type = set_type(i);
      set.name = fmt::format("SOS{}", sos_.size() + 1);
      i += 3;
    } else {
      *error = at(i, "expected an SOS statement, 'name: S1:: column:weight ...'");
      return false;
    }
    for (const SosSet& other : sos_) {
      if (other.name == set.name) {
        *error = at(i, fmt::format("duplicate SOS name '{}'", set.name));
        return false;
      }
    }
    while (i < span.end && !named_header(i) && !unnamed_header(i)) {
      if (tokens_[i].kind != TokKind::kIdent) {
        *error = at(i, "expected a member 'column:weight' in an SOS statement");
        return false;
      }
      const Index col = intern_column(tokens_[i].text);
      ++i;
      double weight = static_cast<double>(set.columns.size() + 1);
      if (is_op(i, ":")) {
        ++i;
        double sign = 1.0;
        while (is_op(i, "-") || is_op(i, "+")) {
          if (tokens_[i].text == "-") sign = -sign;
          ++i;
        }
        if (i >= span.end || tokens_[i].kind != TokKind::kNumber) {
          *error = at(i, "expected a number after ':' in an SOS member");
          return false;
        }
        weight = sign * tokens_[i].value;
        ++i;
      }
      set.columns.push_back(col);
      set.weights.push_back(weight);
    }
    sos_.push_back(std::move(set));
  }
  return true;
}

void LpParser::finish_sc_sos() {
  model_->semicontinuous.clear();
  for (std::size_t j = 0; j < col_semicontinuous_.size(); ++j) {
    if (col_semicontinuous_[j] != 0) model_->semicontinuous.push_back(static_cast<Index>(j));
  }
  model_->sos.clear();
  for (const SosSet& set : sos_) model_->sos.push_back(sorted_by_weight(set));
}

}  // namespace sankhya::io::lp
