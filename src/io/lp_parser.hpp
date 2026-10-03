// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the LP-format parser state, shared between lp_reader.cpp (lexing, the sections
// split, the objective, constraints, bounds and integrality) and lp_sos.cpp (the
// semi-continuous and SOS sections, #754).
//
// Not a public header - internal to src/io/, exactly like mps_parser.hpp. Split out of
// lp_reader.cpp when #754's two sections would have taken it further past the project's ~600
// line guideline; no logic changed in the split.
#pragma once

#include <cctype>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"

namespace sankhya::io::lp {

enum class TokKind { kIdent, kNumber, kOp, kEof };

struct Tok {
  TokKind kind = TokKind::kEof;
  std::string text;    ///< identifier spelling, or the operator
  double value = 0.0;  ///< numeric value when kind == kNumber
  Count line = 0;
  bool first_on_line = false;
};

/// Relational operators, normalised so that "=<" and "<" both arrive as "<=".
[[nodiscard]] inline bool is_relop(const Tok& t) {
  return t.kind == TokKind::kOp && (t.text == "<=" || t.text == ">=" || t.text == "=");
}

[[nodiscard]] inline bool ident_start(char c) {
  return std::isalpha(static_cast<unsigned char>(c)) != 0 || c == '_' || c == '!' || c == '"' ||
         c == '#' || c == '$' || c == '%' || c == '&' || c == '(' || c == ')' || c == ',' ||
         c == ';' || c == '?' || c == '@' || c == '\'' || c == '`' || c == '|' || c == '~';
}

[[nodiscard]] inline bool ident_body(char c) {
  return ident_start(c) || std::isdigit(static_cast<unsigned char>(c)) != 0 || c == '.';
}

/// Section kinds, in the order they may appear. The semi-continuous and SOS sections (#754)
/// are read in lp_sos.cpp.
enum class LpSection {
  kObjective,
  kConstraints,
  kBounds,
  kGeneral,
  kBinary,
  kSemicontinuous,
  kSos,
  kEnd
};

struct SectionSpan {
  LpSection kind;
  std::size_t begin;  ///< first token of the section body
  std::size_t end;    ///< one past the last
  bool maximize = false;
};

class LpParser {
 public:
  explicit LpParser(Model* model) : model_(model) {}
  ReadResult parse(const std::string& path);

 private:
  [[nodiscard]] bool lex(const std::string& path, std::string* error);
  [[nodiscard]] bool split_sections(std::string* error);

  [[nodiscard]] bool parse_objective(const SectionSpan& span, std::string* error);
  [[nodiscard]] bool parse_constraints(const SectionSpan& span, std::string* error);
  [[nodiscard]] bool parse_bounds(const SectionSpan& span, std::string* error);
  [[nodiscard]] bool parse_integrality(const SectionSpan& span, bool binary,
                                       std::string* error);
  // #754, in lp_sos.cpp.
  [[nodiscard]] bool parse_semicontinuous(const SectionSpan& span, std::string* error);
  [[nodiscard]] bool parse_sos(const SectionSpan& span, std::string* error);
  /// Move the semi-continuous flags and the sets, sorted by weight, into the model.
  void finish_sc_sos();

  /// Parse a signed linear expression starting at `i`, stopping before the first token that
  /// cannot continue it. Coefficients accumulate into `terms` keyed by column, and any bare
  /// number accumulates into `constant`.
  [[nodiscard]] bool parse_expression(std::size_t* i, std::size_t end,
                                      std::unordered_map<Index, double>* terms,
                                      double* constant, std::string* error);

  [[nodiscard]] Index intern_column(const std::string& name);
  [[nodiscard]] std::string at(std::size_t i, const std::string& message) const;

  Model* model_;
  std::string path_;
  std::vector<Tok> tokens_;
  std::vector<SectionSpan> sections_;

  std::unordered_map<std::string, Index> col_index_;
  std::vector<std::string> col_names_;
  std::vector<double> col_cost_;
  std::vector<double> col_lower_;
  std::vector<double> col_upper_;
  std::vector<VarType> col_type_;

  std::vector<std::string> row_names_;
  std::vector<double> row_lower_;
  std::vector<double> row_upper_;
  std::vector<Index> tri_row_;
  std::vector<Index> tri_col_;
  std::vector<double> tri_value_;
  Count unnamed_rows_ = 0;

  std::vector<char> col_semicontinuous_;  ///< #754, sized lazily
  std::vector<SosSet> sos_;
};

}  // namespace sankhya::io::lp
