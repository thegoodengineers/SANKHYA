// SPDX-License-Identifier: Apache-2.0
// SANKHYA - reading an AMPL .nl file, text form (NLP stage 1). See nl_reader.hpp for what is
// read, what is refused, and the two documents it is written from. Section and table numbers
// below are those of D. M. Gay, "Writing .nl Files", SAND2005-7907P (2005).

#include "nlp/nl_reader.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <utility>

#include <fmt/format.h>

#include "io/line_reader.hpp"

namespace sankhya::nlp {
namespace {

using io::ReadResult;

/// Whitespace-separated tokens across lines, a '#' starting a comment to the end of the line.
/// The layout of a .nl file is one item per line, but nothing in the grammar depends on it,
/// so a reader that does not either accepts a file whose writer joined two lines.
class Tokens {
 public:
  explicit Tokens(io::LineReader* in) : in_(in) {}
  bool next(std::string* token) {
    for (;;) {
      while (pos_ < line_.size() && std::isspace(static_cast<unsigned char>(line_[pos_])))
        ++pos_;
      if (pos_ < line_.size() && line_[pos_] != '#') {
        const std::size_t start = pos_;
        while (pos_ < line_.size() && !std::isspace(static_cast<unsigned char>(line_[pos_]))) {
          ++pos_;
        }
        *token = line_.substr(start, pos_ - start);
        return true;
      }
      if (!in_->next(&line_)) return false;
      pos_ = 0;
    }
  }
  [[nodiscard]] std::string at(const std::string& message) const {
    return in_->error_at(message);
  }

 private:
  io::LineReader* in_;
  std::string line_;
  std::size_t pos_ = 0;
};

bool to_long(const std::string& text, long long* out) {
  if (text.empty()) return false;
  errno = 0;
  char* end = nullptr;
  const long long v = std::strtoll(text.c_str(), &end, 10);
  if (errno != 0 || end != text.c_str() + text.size()) return false;
  *out = v;
  return true;
}

bool to_double(const std::string& text, double* out) {
  if (text.empty()) return false;
  errno = 0;
  char* end = nullptr;
  const double v = std::strtod(text.c_str(), &end);
  if (errno == ERANGE && std::fabs(v) > 1.0) return false;  // overflow; underflow is fine
  if (end != text.c_str() + text.size()) return false;
  *out = v;
  return true;
}

/// Names of the operators this reader recognises and refuses (Tables 4, 6, 8, 11 and the
/// piecewise-linear operator of sec. 3), so a refusal can say which one it met.
const char* refused_operator(long long code) {
  static const std::map<long long, const char*> names = {
      {4, "rem"},     {6, "less"},       {11, "min"},       {12, "max"},
      {13, "floor"},  {14, "ceil"},      {15, "abs"},       {20, "or"},
      {21, "and"},    {22, "lt"},        {23, "le"},        {24, "eq"},
      {28, "ge"},     {29, "gt"},        {30, "ne"},        {34, "not"},
      {35, "if"},     {37, "tanh"},      {38, "tan"},       {40, "sinh"},
      {45, "cosh"},   {47, "atanh"},     {48, "atan2"},     {49, "atan"},
      {50, "asinh"},  {51, "asin"},      {52, "acosh"},     {53, "acos"},
      {55, "intdiv"}, {56, "precision"}, {57, "round"},     {58, "trunc"},
      {59, "count"},  {60, "numberof"},  {61, "numberofs"}, {64, "piecewise-linear term"},
      {65, "ifs"},    {70, "n-ary and"}, {71, "n-ary or"},  {72, "implies"},
      {73, "iff"},    {74, "alldiff"}};
  const auto it = names.find(code);
  return it == names.end() ? nullptr : it->second;
}

int arity_of(long long code) {
  switch (code) {
    case 16:
    case 39:
    case 41:
    case 42:
    case 43:
    case 44:
    case 46: return 1;
    case 0:
    case 1:
    case 2:
    case 3:
    case 5: return 2;
    case 54: return -1;  // n-ary: the count follows
    default: return 0;   // not accepted
  }
}

struct Frame {
  long long code = 0;
  std::size_t arity = 0;
  std::vector<ExprId> args;
};

class Parser {
 public:
  Parser(Tokens* tokens, ExpressionGraph* graph, Index n, std::vector<std::string>* notes)
      : t_(tokens), g_(graph), n_(n), notes_(notes) {}

  std::string error;
  bool refused = false;
  std::vector<ExprId> defined;  ///< defined variable n_var + k -> its expression (V segments)

  bool token(std::string* out, const char* what) {
    if (t_->next(out)) return true;
    return fail(fmt::format("the file ends where {} was expected", what));
  }
  bool integer(long long* out, const char* what) {
    std::string tok;
    if (!token(&tok, what)) return false;
    if (!to_long(tok, out)) return fail(fmt::format("'{}' is not an integer ({})", tok, what));
    return true;
  }
  bool number(double* out, const char* what) {
    std::string tok;
    if (!token(&tok, what)) return false;
    if (!to_double(tok, out) || !std::isfinite(*out)) {
      return fail(fmt::format("'{}' is not a finite number ({})", tok, what));
    }
    return true;
  }
  bool fail(const std::string& message) {
    if (error.empty()) error = t_->at(message);
    return false;
  }
  bool refuse(const std::string& message) {
    refused = true;
    return fail(message);
  }

  /// One expression graph in Polish prefix notation (sec. 3), iteratively: a deep chain of
  /// binary pluses is ordinary in a written file and must not become a deep recursion.
  bool expression(ExprId* out) {
    std::vector<Frame> stack;
    for (;;) {
      std::string tok;
      if (!token(&tok, "an expression")) return false;
      ExprId value = kNoExpr;
      const char kind = tok[0];
      const std::string rest = tok.substr(1);
      long long code = 0;
      if (kind == 'n') {
        double v = 0.0;
        if (!to_double(rest, &v) || !std::isfinite(v)) {
          return fail(fmt::format("'{}' is not a finite numeric constant", tok));
        }
        value = g_->constant(v);
      } else if (kind == 'v') {
        if (!to_long(rest, &code) || code < 0)
          return fail(fmt::format("bad variable '{}'", tok));
        if (code < n_) {
          value = g_->variable(static_cast<Index>(code));
        } else {
          const auto k = static_cast<std::size_t>(code - n_);
          if (k >= defined.size() || defined[k] == kNoExpr) {
            return fail(fmt::format("defined variable v{} is used before its V segment", code));
          }
          value = defined[k];
        }
      } else if (kind == 'o') {
        if (!to_long(rest, &code)) return fail(fmt::format("bad operator '{}'", tok));
        const int arity = arity_of(code);
        if (arity == 0) {
          const char* name = refused_operator(code);
          return refuse(name != nullptr
                            ? fmt::format("operator o{} ({}) is not smooth or not supported; "
                                          "the model is refused rather than approximated",
                                          code, name)
                            : fmt::format("operator o{} is not in the published .nl "
                                          "specification this reader follows",
                                          code));
        }
        Frame f;
        f.code = code;
        if (arity > 0) {
          f.arity = static_cast<std::size_t>(arity);
        } else {
          long long count = 0;
          if (!integer(&count, "the operand count of a sumlist")) return false;
          if (count < 0) return fail("a negative operand count");
          f.arity = static_cast<std::size_t>(count);
        }
        if (f.arity > 0) {
          stack.push_back(std::move(f));
          continue;
        }
        value = g_->constant(0.0);  // an empty sum
      } else if (kind == 'f') {
        return refuse("imported function calls (f) are not supported");
      } else if (kind == 'h') {
        return refuse("string arguments (h) are not supported");
      } else {
        return fail(fmt::format("'{}' does not start an expression", tok));
      }
      // Hand the value up until a frame is still waiting for operands.
      for (;;) {
        if (value == kNoExpr) return fail("the expression is malformed: " + g_->invalid());
        if (stack.empty()) {
          *out = value;
          return true;
        }
        Frame& top = stack.back();
        top.args.push_back(value);
        if (top.args.size() < top.arity) break;
        value = build(top);
        stack.pop_back();
      }
    }
  }

 private:
  ExprId build(const Frame& f) {
    const auto& a = f.args;
    switch (f.code) {
      case 0: return g_->add(a[0], a[1]);
      case 1: return g_->subtract(a[0], a[1]);
      case 2: return g_->multiply(a[0], a[1]);
      case 3: return g_->divide(a[0], a[1]);
      case 5: return power(a[0], a[1]);
      case 16: return g_->negate(a[0]);
      case 39: return g_->sqrt(a[0]);
      case 41: return g_->sin(a[0]);
      case 42: return g_->multiply(g_->log(a[0]), g_->constant(1.0 / std::log(10.0)));
      case 43: return g_->log(a[0]);
      case 44: return g_->exp(a[0]);
      case 46: return g_->cos(a[0]);
      case 54: return g_->sum(a);
      default: return kNoExpr;
    }
  }

  ExprId power(ExprId base, ExprId exponent) {
    const Node& e = g_->node(exponent);
    if (e.op == Op::kConstant) return g_->power(base, e.value);
    const Node& b = g_->node(base);
    if (b.op == Op::kConstant && b.value > 0.0) {
      return g_->exp(g_->multiply(exponent, g_->constant(std::log(b.value))));
    }
    if (notes_ != nullptr && !noted_power_) {
      notes_->push_back(
          "a power with a variable exponent is read as exp(exponent * log(base)), which is "
          "undefined where the base is not positive");
      noted_power_ = true;
    }
    return g_->exp(g_->multiply(exponent, g_->log(base)));
  }

  Tokens* t_;
  ExpressionGraph* g_;
  Index n_;
  std::vector<std::string>* notes_;
  bool noted_power_ = false;
};

/// A .col / .row file beside the model: one name per line (AMPL's `auxfiles rc`, sec. 1).
std::vector<std::string> names_beside(const std::string& path, const char* extension) {
  std::string stem = path;
  const auto ends = [&](const std::string& s) {
    return stem.size() >= s.size() && stem.compare(stem.size() - s.size(), s.size(), s) == 0;
  };
  if (ends(".gz")) stem.resize(stem.size() - 3);
  if (ends(".nl") || ends(".NL")) stem.resize(stem.size() - 3);
  std::ifstream in(stem + extension);
  std::vector<std::string> names;
  std::string line;
  while (in && std::getline(in, line)) {
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
    names.push_back(line);
  }
  return names;
}

}  // namespace

bool looks_like_nl(const std::string& path) {
  std::string lowered = path;
  for (char& c : lowered) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (lowered.size() >= 3 && lowered.compare(lowered.size() - 3, 3, ".gz") == 0) {
    lowered.resize(lowered.size() - 3);
  }
  return lowered.size() >= 3 && lowered.compare(lowered.size() - 3, 3, ".nl") == 0;
}

ReadResult read_nl(const std::string& path, std::unique_ptr<NonlinearModel>* out,
                   std::vector<std::string>* notes) {
  io::LineReader in;
  std::string open_error;
  if (!in.open(path, &open_error)) return ReadResult::failure(open_error);

  // ---- The header: ten lines of counts (sec. 2).
  std::vector<std::vector<long long>> header(10);
  std::string line;
  for (std::size_t h = 0; h < 10; ++h) {
    if (!in.next(&line))
      return ReadResult::failure(in.error_at("the header has fewer than 10 lines"));
    const std::size_t hash = line.find('#');
    if (hash != std::string::npos) line.resize(hash);
    if (h == 0) {
      if (!line.empty() && line[0] == 'b') {
        return ReadResult::refusal(
            in.error_at("binary .nl files are not supported; only the text ('g') form is"));
      }
      if (line.empty() || line[0] != 'g') {
        return ReadResult::failure(in.error_at("a .nl file starts with 'g' (text) or 'b'"));
      }
      continue;
    }
    std::size_t pos = 0;
    while (pos < line.size()) {
      while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos]))) ++pos;
      const std::size_t start = pos;
      while (pos < line.size() && !std::isspace(static_cast<unsigned char>(line[pos]))) ++pos;
      if (start == pos) break;
      long long v = 0;
      if (!to_long(line.substr(start, pos - start), &v) || v < 0) {
        return ReadResult::failure(in.error_at("a header count is not a non-negative integer"));
      }
      header[h].push_back(v);
    }
  }
  const auto get = [&](std::size_t h, std::size_t k, long long fallback) {
    return k < header[h].size() ? header[h][k] : fallback;
  };
  const long long n_var = get(1, 0, -1);
  const long long n_con = get(1, 1, -1);
  const long long n_obj = get(1, 2, -1);
  if (n_var < 0 || n_con < 0 || n_obj < 0 || n_var > kMaxNonzeros || n_con > kMaxNonzeros) {
    return ReadResult::failure(
        in.error_at("header line 2 needs the numbers of variables, "
                    "constraints and objectives"));
  }
  if (get(1, 5, 0) > 0)
    return ReadResult::refusal(in.error_at("logical constraints are not supported"));
  if (get(5, 1, 0) > 0)
    return ReadResult::refusal(in.error_at("imported functions are not supported"));
  const auto n = static_cast<Index>(n_var);

  // ---- Which columns are integer: "Hooking Your Solver to AMPL", Tables 3 and 4.
  const long long nlvc = get(4, 0, 0);
  const long long nlvo = get(4, 1, 0);
  const long long nlvb = get(4, 2, -1);
  const long long nbv = get(6, 0, 0);
  const long long niv = get(6, 1, 0);
  const long long nlvbi = get(6, 2, 0);
  const long long nlvci = get(6, 3, 0);
  const long long nlvoi = get(6, 4, 0);
  const long long nlv = std::max(nlvc, nlvo);
  std::vector<bool> integer(static_cast<std::size_t>(n), false);
  if (nlvbi + nlvci + nlvoi > 0 && nlvb < 0) {
    return ReadResult::refusal(
        in.error_at("a header from before 1993 does not say which "
                    "nonlinear variables are integer"));
  }
  const long long b1 = std::max(nlvb, 0LL);
  const long long blocks[3][3] = {{0, b1, nlvbi}, {b1, nlvc, nlvci}, {nlvc, nlv, nlvoi}};
  for (const auto& block : blocks) {
    const long long lo = block[0], hi = block[1], tail = block[2];
    if (tail == 0) continue;
    if (hi - lo < tail || lo < 0 || hi > n_var) {
      return ReadResult::failure(
          in.error_at("the header's integer counts do not fit its "
                      "nonlinear-variable counts"));
    }
    for (long long j = hi - tail; j < hi; ++j) integer[static_cast<std::size_t>(j)] = true;
  }
  if (nlv + nbv + niv > n_var) {
    return ReadResult::failure(
        in.error_at("the header counts more special variables than "
                    "there are variables"));
  }
  for (long long j = n_var - nbv - niv; j < n_var; ++j)
    integer[static_cast<std::size_t>(j)] = true;

  Model base;
  base.resize_columns(n);
  std::fill(base.col_lower.begin(), base.col_lower.end(), -kInfinity);
  for (Index j = 0; j < n; ++j) {
    if (integer[static_cast<std::size_t>(j)])
      base.col_type[static_cast<std::size_t>(j)] = VarType::kInteger;
  }
  base.matrix = SparseMatrix(0, n);
  base.matrix.finalize();
  auto model = std::make_unique<NonlinearModel>(std::move(base));

  // ---- The segments (sec. 4, Table 13).
  Tokens tokens(&in);
  Parser p(&tokens, &model->graph, n, notes);
  const auto m = static_cast<std::size_t>(n_con);
  std::vector<ExprId> con_expr(m, kNoExpr);
  std::vector<std::vector<std::pair<Index, double>>> con_linear(m);
  std::vector<double> lower(m, -kInfinity), upper(m, kInfinity);
  std::vector<ExprId> obj_expr(static_cast<std::size_t>(n_obj), kNoExpr);
  std::vector<int> obj_sense(static_cast<std::size_t>(n_obj), 0);
  std::vector<std::vector<std::pair<Index, double>>> obj_linear(
      static_cast<std::size_t>(n_obj));
  bool saw_ranges = false;
  const auto fail = [&]() {
    return p.refused ? ReadResult::refusal(p.error) : ReadResult::failure(p.error);
  };
  const auto index_in = [&](const std::string& rest, long long limit, const char* what,
                            long long* out_index) {
    if (!to_long(rest, out_index) || *out_index < 0 || *out_index >= limit) {
      return p.fail(fmt::format("'{}' is not a {} index", rest, what));
    }
    return true;
  };
  const auto pairs = [&](long long count, long long limit,
                         std::vector<std::pair<Index, double>>* into) {
    for (long long k = 0; k < count; ++k) {
      long long j = 0;
      double v = 0.0;
      if (!p.integer(&j, "an index") || !p.number(&v, "a value")) return false;
      if (j < 0 || j >= limit) return p.fail(fmt::format("index {} is out of range", j));
      into->emplace_back(static_cast<Index>(j), v);
    }
    return true;
  };
  const auto bound_line = [&](const char* what, double* lo, double* hi) {
    long long type = 0;
    if (!p.integer(&type, what)) return false;
    double a = 0.0, b = 0.0;
    switch (type) {  // Table 17
      case 0:
        if (!p.number(&a, what) || !p.number(&b, what)) return false;
        *lo = a;
        *hi = b;
        return true;
      case 1:
        if (!p.number(&b, what)) return false;
        *hi = b;
        return true;
      case 2:
        if (!p.number(&a, what)) return false;
        *lo = a;
        return true;
      case 3: return true;
      case 4:
        if (!p.number(&a, what)) return false;
        *lo = a;
        *hi = a;
        return true;
      case 5: return p.refuse("complementarity constraints are not supported");
      default: return p.fail(fmt::format("unknown bound type {}", type));
    }
  };

  std::string tok;
  while (tokens.next(&tok)) {
    const char key = tok[0];
    const std::string rest = tok.substr(1);
    long long i = 0, count = 0, extra = 0;
    std::vector<std::pair<Index, double>> scratch;
    ExprId e = kNoExpr;
    switch (key) {
      case 'C':
        if (!index_in(rest, n_con, "constraint", &i) || !p.expression(&e)) return fail();
        con_expr[static_cast<std::size_t>(i)] = e;
        break;
      case 'O':
        if (!index_in(rest, n_obj, "objective", &i) ||
            !p.integer(&extra, "the objective sense") || !p.expression(&e)) {
          return fail();
        }
        obj_expr[static_cast<std::size_t>(i)] = e;
        obj_sense[static_cast<std::size_t>(i)] = extra != 0 ? 1 : 0;
        break;
      case 'V': {
        if (!to_long(rest, &i) || i < n_var || !p.integer(&count, "a linear term count") ||
            !p.integer(&extra, "a V segment's use") || !pairs(count, n_var, &scratch) ||
            !p.expression(&e)) {
          return p.error.empty() ? ReadResult::failure(tokens.at("a malformed V segment"))
                                 : fail();
        }
        std::vector<ExprId> terms{e};
        for (const auto& [j, c] : scratch) {
          terms.push_back(
              model->graph.multiply(model->graph.constant(c), model->graph.variable(j)));
        }
        const auto k = static_cast<std::size_t>(i - n_var);
        if (p.defined.size() <= k) p.defined.resize(k + 1, kNoExpr);
        p.defined[k] = model->graph.sum(terms);
        break;
      }
      case 'J':
      case 'G': {
        const bool jac = key == 'J';
        if (!index_in(rest, jac ? n_con : n_obj, jac ? "constraint" : "objective", &i) ||
            !p.integer(&count, "a term count")) {
          return fail();
        }
        auto& into = jac ? con_linear[static_cast<std::size_t>(i)]
                         : obj_linear[static_cast<std::size_t>(i)];
        if (!pairs(count, n_var, &into)) return fail();
        break;
      }
      case 'r':
        saw_ranges = true;
        for (std::size_t r = 0; r < m; ++r) {
          if (!bound_line("a constraint range", &lower[r], &upper[r])) return fail();
        }
        break;
      case 'b':
        for (Index j = 0; j < n; ++j) {
          auto& b = model->base;
          if (!bound_line("a variable bound", &b.col_lower[static_cast<std::size_t>(j)],
                          &b.col_upper[static_cast<std::size_t>(j)])) {
            return fail();
          }
        }
        break;
      case 'x':
      case 'd': {
        if (!to_long(rest, &count) || count < 0)
          return ReadResult::failure(tokens.at("a bad count"));
        if (!pairs(count, key == 'x' ? n_var : n_con, &scratch)) return fail();
        if (key == 'x') {
          model->start.assign(static_cast<std::size_t>(n), 0.0);
          for (const auto& [j, v] : scratch) model->start[static_cast<std::size_t>(j)] = v;
        } else if (notes != nullptr) {
          notes->push_back("the initial dual values (d segment) are not used");
        }
        break;
      }
      case 'k':
        if (!to_long(rest, &count)) return ReadResult::failure(tokens.at("a bad k segment"));
        for (long long k = 0; k < count; ++k) {
          if (!p.integer(&extra, "a Jacobian column count")) return fail();
        }
        break;
      case 'S': {
        std::string name;
        if (!p.integer(&count, "a suffix count") || !p.token(&name, "a suffix name"))
          return fail();
        for (long long k = 0; k < 2 * count; ++k) {
          if (!p.token(&name, "a suffix value")) return fail();
        }
        break;
      }
      case 'F': return ReadResult::refusal(tokens.at("imported functions are not supported"));
      case 'L': return ReadResult::refusal(tokens.at("logical constraints are not supported"));
      default: return ReadResult::failure(tokens.at(fmt::format("unknown segment '{}'", tok)));
    }
  }
  if (m > 0 && !saw_ranges)
    return ReadResult::failure(in.error_at("no r segment for the constraints"));

  ExpressionGraph& g = model->graph;
  const std::vector<std::string> col_names = names_beside(path, ".col");
  const std::vector<std::string> row_names = names_beside(path, ".row");
  if (col_names.size() >= static_cast<std::size_t>(n) && n > 0) {
    model->base.col_names.assign(col_names.begin(), col_names.begin() + n);
  }
  for (std::size_t r = 0; r < m; ++r) {
    std::vector<ExprId> terms;
    if (con_expr[r] != kNoExpr) terms.push_back(con_expr[r]);
    for (const auto& [j, c] : con_linear[r]) {
      if (c != 0.0) terms.push_back(g.multiply(g.constant(c), g.variable(j)));
    }
    NonlinearConstraint c;
    c.expression = g.sum(terms);
    c.lower = lower[r];
    c.upper = upper[r];
    c.name = r < row_names.size() ? row_names[r] : fmt::format("c{}", r);
    model->constraints.push_back(std::move(c));
  }
  if (n_obj > 0) {
    if (n_obj > 1 && notes != nullptr) {
      notes->push_back(fmt::format("the file has {} objectives; the first is used", n_obj));
    }
    model->base.sense = obj_sense[0] != 0 ? ObjSense::kMaximize : ObjSense::kMinimize;
    for (const auto& [j, c] : obj_linear[0])
      model->base.col_cost[static_cast<std::size_t>(j)] += c;
    const ExprId e = obj_expr[0];
    if (e != kNoExpr && g.contains(e) && g.node(e).op == Op::kConstant) {
      model->base.objective_offset = g.node(e).value;
    } else if (e != kNoExpr) {
      model->objective = e;
    }
  }
  const std::string problem = model->validate();
  if (!problem.empty()) return ReadResult::failure(fmt::format("{}: {}", path, problem));
  *out = std::move(model);
  return ReadResult::success();
}

}  // namespace sankhya::nlp
