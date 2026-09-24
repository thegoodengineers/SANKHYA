// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the VIPR certificate writer (#518). See certificate_writer.hpp for what the
// certificate proves and the references.

#include "certificate_writer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <span>
#include <sstream>

#include <fmt/format.h>

#include "core/safe_bound.hpp"

namespace sankhya::mip {

std::string exact_decimal(double v) {
  if (v == 0.0) return "0";
  if (!std::isfinite(v)) return v > 0.0 ? "inf" : "-inf";
  if (v == std::trunc(v)) return fmt::format("{:.0f}", v);
  // The number of binary digits after the point: v * 2^k is an integer for this k and no
  // smaller one, and a binary fraction with k digits has exactly k decimal digits.
  int digits = 0;
  for (double t = v; t != std::trunc(t); t *= 2.0) ++digits;
  return fmt::format("{:.{}f}", v, digits);
}

using detail::write_claim;
using detail::write_model;

namespace {

/// What a subtree proves: a contradiction, or objective >= bound (minimise space).
struct Derived {
  Index index = -1;
  bool absurd = false;
  double bound = 0.0;
};

class Writer {
 public:
  Writer(const Model& model, const CertificateTree& tree, std::ostream& der)
      : model_(model), tree_(tree), der_(der), rows_(model.matrix) {
    sense_ = model.sense_multiplier();
    const auto n = static_cast<std::size_t>(model.num_cols());
    cost_.resize(n);
    for (std::size_t j = 0; j < n; ++j) cost_[j] = sense_ * model.col_cost[j];
    lower_ = model.col_lower;
    upper_ = model.col_upper;
    // CON layout (verify_certificate_model.py reproduces it): rows first, one E or a G
    // and/or an L each, then the finite column bounds.
    const auto m = static_cast<std::size_t>(model.num_rows());
    con_lower_.assign(m, -1);
    con_upper_.assign(m, -1);
    for (std::size_t i = 0; i < m; ++i) {
      const double lo = model.row_lower[i];
      const double hi = model.row_upper[i];
      if (is_finite_bound(lo) && lo == hi) {
        con_lower_[i] = con_upper_[i] = num_con_++;
        continue;
      }
      if (is_finite_bound(lo)) con_lower_[i] = num_con_++;
      if (is_finite_bound(hi)) con_upper_[i] = num_con_++;
    }
    lower_source_.assign(n, -1);
    upper_source_.assign(n, -1);
    for (std::size_t j = 0; j < n; ++j) {
      if (is_finite_bound(model.col_lower[j])) lower_source_[j] = num_con_++;
      if (is_finite_bound(model.col_upper[j])) upper_source_[j] = num_con_++;
    }
    next_ = num_con_;
  }

  [[nodiscard]] Index num_con() const { return num_con_; }
  [[nodiscard]] Count derivations() const { return next_ - num_con_; }
  [[nodiscard]] const std::string& error() const { return error_; }
  /// How the leaves were proved, for the log.
  struct LeafCounts {
    Count own_duals = 0;        ///< a bound from the leaf's own LP duals
    Count inherited_duals = 0;  ///< from the nearest ancestor's (pruned before its LP)
    Count farkas = 0;           ///< a contradiction from its infeasible LP
    Count farkas_failed = 0;    ///< an infeasible LP whose Farkas vector proved nothing here
    Count empty_box = 0;        ///< a contradiction between two bounds on one column
    Count implied_bounds = 0;   ///< row-implied column bounds derived for the leaves
  };
  [[nodiscard]] const LeafCounts& leaves() const { return leaves_; }

  /// Walk the tree in post-order, writing every derivation. False on a leaf with no proof.
  bool walk(Derived* root);
  /// Round the root's bound up to the objective's step when that is valid and helps.
  Derived round_to_step(const Derived& root, double step);

 private:
  Index emit(const std::string& line) {
    der_ << line << '\n';
    return next_++;
  }
  /// The assumption `change` of `node`, as a VIPR asm line.
  Index emit_assumption(Index node) {
    const CertificateTree::Node& c = tree_.nodes[static_cast<std::size_t>(node)];
    return emit(fmt::format("a{} {} {} 1 {} 1 {{ asm }} -1", node, c.is_upper ? 'L' : 'G',
                            exact_decimal(c.value), c.column));
  }
  /// Enter `node`: its change tightens the box, and the assumption on top of path_ (emitted
  /// by the parent just before) becomes the source of that side when it is tighter.
  void apply(Index node) {
    const CertificateTree::Node& c = tree_.nodes[static_cast<std::size_t>(node)];
    if (!c.has_change) return;
    const auto u = static_cast<std::size_t>(c.column);
    std::vector<double>& side = c.is_upper ? upper_ : lower_;
    std::vector<Index>& source = c.is_upper ? upper_source_ : lower_source_;
    undo_.push_back({u, c.is_upper, side[u], source[u]});
    const bool tighter = c.is_upper ? c.value < side[u] : c.value > side[u];
    if (tighter || source[u] < 0) {
      side[u] = c.value;
      source[u] = path_.back();
    }
  }
  void undo(Index node) {
    if (!tree_.nodes[static_cast<std::size_t>(node)].has_change) return;
    const Saved s = undo_.back();
    undo_.pop_back();
    (s.is_upper ? upper_ : lower_)[s.column] = s.value;
    (s.is_upper ? upper_source_ : lower_source_)[s.column] = s.source;
  }
  /// The assumptions a step over these columns may need: the one each side of each
  /// column's box currently comes from, when that is an assumption rather than a CON bound
  /// (which the checker has anyway). Referencing only these, not the whole path, keeps a
  /// leaf's reason proportional to what it uses rather than to the depth of the tree.
  template <typename Columns>
  void add_assumptions(const Columns& columns, std::vector<Index>* out) const {
    for (const Index j : columns) {
      const auto u = static_cast<std::size_t>(j);
      if (lower_source_[u] >= num_con_) out->push_back(lower_source_[u]);
      if (upper_source_[u] >= num_con_) out->push_back(upper_source_[u]);
    }
  }
  static std::string zero_references(std::vector<Index> indices, Index* k) {
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    std::string pairs;
    for (const Index a : indices) pairs += fmt::format(" {} 0", a);
    *k += static_cast<Index>(indices.size());
    return pairs;
  }
  /// The multipliers as a lin reason: the rows by the side each y_i prices, then, with
  /// multiplier 0 so the checker may use their bounds, the assumptions behind every column
  /// the combination or the objective touches, and every implied bound in `extra`.
  std::string lin_reason(const std::vector<double>& y, double sign,
                         const std::vector<Index>& extra, bool with_cost) const {
    std::string pairs;
    Index k = 0;
    for (std::size_t i = 0; i < y.size(); ++i) {
      if (y[i] == 0.0) continue;
      const Index con = y[i] > 0.0 ? con_lower_[i] : con_upper_[i];
      pairs += fmt::format(" {} {}", con, exact_decimal(sign * y[i]));
      ++k;
    }
    std::vector<Index> touched;
    for (Index j = 0; j < model_.num_cols(); ++j) {
      bool uses = with_cost && cost_[static_cast<std::size_t>(j)] != 0.0;
      const ColumnView col = model_.matrix.column(j);
      for (Index p = 0; p < col.size && !uses; ++p) {
        uses = y[static_cast<std::size_t>(col.rows[p])] != 0.0;
      }
      if (uses) touched.push_back(j);
    }
    std::vector<Index> references(extra);
    add_assumptions(touched, &references);
    pairs += zero_references(std::move(references), &k);
    return fmt::format("{{ lin {}{} }}", k, pairs);
  }
  /// Each row-implied bound the safe bound used, as a derivation of its own: the row with
  /// multiplier 1, completed over the box, proves coefficient * x_j <= rhs (or >=).
  std::vector<Index> emit_implied(Index node, const std::vector<ImpliedBound>& implied) {
    std::vector<Index> indices;
    for (const ImpliedBound& b : implied) {
      const auto row = static_cast<std::size_t>(b.row);
      const Index con = b.uses_row_upper ? con_upper_[row] : con_lower_[row];
      std::string pairs = fmt::format(" {} 1", con);
      const ColumnView row_view = rows_.row(b.row);
      std::vector<Index> references;
      add_assumptions(
          std::span<const Index>(row_view.rows, static_cast<std::size_t>(row_view.size)),
          &references);
      Index k = 1;
      pairs += zero_references(std::move(references), &k);
      indices.push_back(
          emit(fmt::format("i{}_{} {} {} 1 {} {} {{ lin {}{} }} -1", node, b.column,
                           b.uses_row_upper ? 'L' : 'G', exact_decimal(b.rhs), b.column,
                           exact_decimal(b.coefficient), k, pairs)));
      ++leaves_.implied_bounds;
    }
    return indices;
  }
  Derived emit_leaf(Index node);
  Derived emit_split(Index node, const Derived& down, Index asm_down, const Derived& up,
                     Index asm_up);
  std::string objective_line(Index node, char kind, double bound, const std::string& reason) {
    // Minimise: objective >= bound. Maximise: objective <= -bound (bound is minimise space).
    return sense_ > 0.0
               ? fmt::format("{}{} G {} OBJ {} -1", kind, node, exact_decimal(bound), reason)
               : fmt::format("{}{} L {} OBJ {} -1", kind, node, exact_decimal(-bound), reason);
  }

  struct Saved {
    std::size_t column;
    bool is_upper;
    double value;
    Index source;
  };

  const Model& model_;
  const CertificateTree& tree_;
  std::ostream& der_;
  CsrView rows_;
  double sense_ = 1.0;
  std::vector<double> cost_;
  std::vector<double> lower_;
  std::vector<double> upper_;
  std::vector<Index> con_lower_;
  std::vector<Index> con_upper_;
  /// The constraint each finite side of the box comes from: a CON bound or an assumption.
  std::vector<Index> lower_source_;
  std::vector<Index> upper_source_;
  Index num_con_ = 0;
  Index next_ = 0;
  std::vector<Saved> undo_;
  std::vector<Index> path_;                          ///< assumptions in force
  std::vector<const CertificateTree::Node*> duals_;  ///< nodes on the path with duals
  std::string error_;
  LeafCounts leaves_;
};

Derived Writer::emit_leaf(Index node) {
  const CertificateTree::Node& leaf = tree_.nodes[static_cast<std::size_t>(node)];
  const auto m = static_cast<std::size_t>(model_.num_rows());
  SafeBoundProblem problem;
  problem.matrix = &model_.matrix;
  problem.row_lower = model_.row_lower;
  problem.row_upper = model_.row_upper;
  problem.col_lower = lower_;
  problem.col_upper = upper_;
  // A box emptied by the branching itself (an integer column's fractional bound crossed by
  // a branch): lower - upper > 0 from the two single-variable constraints, 1 and -1.
  for (std::size_t j = 0; j < lower_.size(); ++j) {
    if (lower_source_[j] < 0 || upper_source_[j] < 0 || !(lower_[j] > upper_[j])) continue;
    const double gap = std::nextafter(lower_[j] - upper_[j], 0.0);  // at most the exact one
    if (!(gap > 0.0)) continue;
    Derived d;
    d.absurd = true;
    d.index = emit(fmt::format("n{} G {} 0 {{ lin 2 {} 1 {} -1 }} -1", node, exact_decimal(gap),
                               lower_source_[j], upper_source_[j]));
    ++leaves_.empty_box;
    return d;
  }
  std::vector<double> y(m, 0.0);
  SafeBoundDetail used;
  // An infeasible leaf: its Farkas multipliers against a zero objective, either sign.
  if (leaf.proof == CertificateTree::Proof::kFarkas) {
    for (const double sign : {1.0, -1.0}) {
      std::fill(y.begin(), y.end(), 0.0);
      for (const auto& [i, v] : leaf.y) y[static_cast<std::size_t>(i)] = sign * v;
      const SafeBound bound = safe_dual_bound(problem, y, true, &used);
      if (bound.value > 0.0) {
        const std::vector<Index> implied = emit_implied(node, used.implied);
        Derived d;
        d.absurd = true;
        d.index = emit(fmt::format("n{} G {} 0 {} -1", node, exact_decimal(bound.value),
                                   lin_reason(used.multipliers, 1.0, implied, false)));
        ++leaves_.farkas;
        return d;
      }
    }
  }
  // Otherwise a bound from the duals of this node's LP, or of its nearest ancestor's; and
  // when those leave a column unbounded over this box, of the next ancestor's up, a few deep.
  if (leaf.proof == CertificateTree::Proof::kFarkas) ++leaves_.farkas_failed;
  if (duals_.empty()) {
    error_ = fmt::format("node {} has no LP duals of its own or on its path", node);
    return {};
  }
  problem.cost = cost_;
  SafeBound bound;
  const CertificateTree::Node* source = nullptr;
  constexpr std::size_t kAncestorsTried = 8;
  for (std::size_t k = 0; k < std::min(kAncestorsTried, duals_.size()); ++k) {
    source = duals_[duals_.size() - 1 - k];
    std::fill(y.begin(), y.end(), 0.0);
    for (const auto& [i, v] : source->y) y[static_cast<std::size_t>(i)] = v;
    bound = safe_dual_bound(problem, y, true, &used);
    if (std::isfinite(bound.value)) break;
  }
  if (!std::isfinite(bound.value)) {
    error_ = fmt::format(
        "node {}: neither its duals nor its nearest ancestors' give a finite safe bound over "
        "the node's box ({} column(s) whose reduced cost needs a bound they lack)",
        node, bound.unbounded_columns);
    return {};
  }
  ++(source == &leaf ? leaves_.own_duals : leaves_.inherited_duals);
  const std::vector<Index> implied = emit_implied(node, used.implied);
  Derived d;
  d.bound = bound.value;
  d.index = emit(objective_line(node, 'n', bound.value,
                                lin_reason(used.multipliers, sense_, implied, true)));
  return d;
}

Derived Writer::emit_split(Index node, const Derived& down, Index asm_down, const Derived& up,
                           Index asm_up) {
  const std::string reason =
      fmt::format("{{ uns {} {} {} {} }}", down.index, asm_down, up.index, asm_up);
  Derived d;
  if (down.absurd && up.absurd) {
    d.absurd = true;
    d.index = emit(fmt::format("u{} G 1 0 {} -1", node, reason));
    return d;
  }
  d.bound = down.absurd ? up.bound : (up.absurd ? down.bound : std::min(down.bound, up.bound));
  d.index = emit(objective_line(node, 'u', d.bound, reason));
  return d;
}

bool Writer::walk(Derived* root) {
  struct Frame {
    explicit Frame(Index n) : node(n) {}
    Index node;
    int stage = 0;  ///< 0 entering, 1 waiting on the down child, 2 on the up child
    Index asm_down = -1;
    Index asm_up = -1;
    Derived down;
    bool pushed_duals = false;
  };
  std::vector<Frame> stack;
  stack.emplace_back(0);
  Derived result;
  while (!stack.empty()) {
    Frame& f = stack.back();
    const CertificateTree::Node& node = tree_.nodes[static_cast<std::size_t>(f.node)];
    const bool split = node.down >= 0 && node.up >= 0;
    if (f.stage == 0) {
      apply(f.node);
      if (node.proof == CertificateTree::Proof::kDual) {
        duals_.push_back(&node);
        f.pushed_duals = true;
      }
      if (split) {
        f.asm_down = emit_assumption(node.down);
        f.asm_up = emit_assumption(node.up);
        f.stage = 1;
        path_.push_back(f.asm_down);
        stack.emplace_back(node.down);
        continue;
      }
      result = emit_leaf(f.node);
      if (!error_.empty()) return false;
    } else if (f.stage == 1) {
      f.down = result;
      path_.pop_back();
      f.stage = 2;
      path_.push_back(f.asm_up);
      stack.emplace_back(node.up);
      continue;
    } else {
      path_.pop_back();
      result = emit_split(f.node, f.down, f.asm_down, result, f.asm_up);
    }
    // This node is finished: `result` is what it proves.
    if (f.pushed_duals) duals_.pop_back();
    undo(f.node);
    stack.pop_back();  // the parent resumes at its own stage with `result`
  }
  *root = result;
  return true;
}

Derived Writer::round_to_step(const Derived& root, double step) {
  // CHVATAL-GOMORY ON THE OBJECTIVE: when every column with a cost is integer and every cost
  // a multiple of an integer step s, the objective only takes multiples of s, so
  // (c / s) x >= L / s rounds up to (c / s) x >= ceil(L / s). The search pruned on exactly
  // this rounding (#221); without it a leaf at 13.2 against an incumbent of 14 would prove
  // only 13.2.
  if (root.absurd || !(step >= 1.0) || step != std::trunc(step) || step > 9e15) return root;
  const auto n = static_cast<std::size_t>(model_.num_cols());
  std::string coefficients;
  Index k = 0;
  for (std::size_t j = 0; j < n; ++j) {
    const double c = model_.col_cost[j];
    if (c == 0.0) continue;
    if (model_.col_type[j] != VarType::kInteger || std::fmod(c, step) != 0.0) return root;
    coefficients += fmt::format(" {} {}", j, exact_decimal(c / step));
    ++k;
  }
  // The bound in the model's sense, divided by the step, moved one ulp towards the weaker
  // side before the ceiling (or floor): the quotient is inexact in general.
  const double bound = sense_ * root.bound;
  const double quotient = bound / step;
  const double rounded = sense_ > 0.0 ? std::ceil(std::nextafter(quotient, -1e300))
                                      : std::floor(std::nextafter(quotient, 1e300));
  const bool improves = sense_ > 0.0 ? rounded * step > bound : rounded * step < bound;
  if (!improves || std::fabs(rounded * step) > 9e15) return root;
  const Index cg = emit(fmt::format("r0 {} {} {}{} {{ rnd 1 {} 1/{} }} -1",
                                    sense_ > 0.0 ? 'G' : 'L', exact_decimal(rounded), k,
                                    coefficients, root.index, exact_decimal(step)));
  Derived d;
  d.bound = sense_ * rounded * step;
  d.index = emit(objective_line(0, 's', d.bound,
                                fmt::format("{{ lin 1 {} {} }}", cg, exact_decimal(step))));
  return d;
}

}  // namespace

CertificateOutcome write_vipr_certificate(const std::string& path, const Model& model,
                                          const CertificateTree& tree,
                                          const std::vector<double>& incumbent,
                                          double objective_step) {
  CertificateOutcome outcome;
  if (tree.nodes.empty()) {
    outcome.message = "the search has no tree";
    return outcome;
  }
  // The derivations go to a side file first: the DER header needs their count.
  const std::string body_path = path + ".der.tmp";
  Derived root;
  Count derivations = 0;
  Index num_con = 0;
  {
    std::ofstream body(body_path, std::ios::binary | std::ios::trunc);
    if (!body) {
      outcome.message = "cannot open " + body_path;
      return outcome;
    }
    Writer w(model, tree, body);
    if (!w.walk(&root)) {
      outcome.message = w.error();
      body.close();
      std::remove(body_path.c_str());
      return outcome;
    }
    root = w.round_to_step(root, objective_step);
    derivations = w.derivations();
    num_con = w.num_con();
    const Writer::LeafCounts& c = w.leaves();
    outcome.leaves = fmt::format(
        "leaves: {} from their own LP duals, {} from an ancestor's, {} infeasible by Farkas, "
        "{} infeasible LPs bounded by an ancestor's duals instead, {} empty boxes; {} "
        "row-implied column bounds derived",
        c.own_duals, c.inherited_duals, c.farkas, c.farkas_failed, c.empty_box,
        c.implied_bounds);
    if (!body) {
      outcome.message = "writing " + body_path + " failed";
      return outcome;
    }
  }
  // Infeasible when the tree says so and nothing was found; a bound and no point when a
  // limit stopped a search that found nothing; a bound and a point otherwise.
  const bool infeasible_claim = root.absurd && incumbent.empty();
  if (root.absurd && !incumbent.empty()) {
    std::remove(body_path.c_str());
    outcome.message = "the tree proves infeasibility but an incumbent exists";
    return outcome;
  }
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    std::remove(body_path.c_str());
    outcome.message = "cannot open " + path;
    return outcome;
  }
  write_model(out, model, num_con);
  outcome.infeasible = infeasible_claim;
  if (infeasible_claim) {
    out << "RTP infeas\nSOL 0\n";
  } else {
    outcome.proved_bound = model.sense_multiplier() * root.bound;
    write_claim(out, model, incumbent, outcome.proved_bound);
  }
  out << "DER " << derivations << '\n';
  {
    std::ifstream body(body_path, std::ios::binary);
    out << body.rdbuf();
  }
  std::remove(body_path.c_str());
  if (!out) {
    outcome.message = "writing " + path + " failed";
    return outcome;
  }
  outcome.written = true;
  outcome.derivations = derivations;
  outcome.message = fmt::format("{} derivation(s) over {} constraint(s) written to {}",
                                derivations, num_con, path);
  return outcome;
}

}  // namespace sankhya::mip
