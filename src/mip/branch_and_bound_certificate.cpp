// SPDX-License-Identifier: Apache-2.0
// SANKHYA - branch and bound: recording the tree a certificate is written from (#518).
//
// With `write_certificate` set, every solved node keeps what its LP proves - the row duals
// of an optimal LP, the Farkas multipliers of an infeasible one - and every branching node
// its two children. At the end of the search src/mip/certificate_writer.cpp turns that into
// a VIPR derivation (Cheung, Gleixner and Steffy, IPCO 2017) that tools/verify_certificate.py
// checks in exact arithmetic. Root cut rows are certified as they are added (each with its
// derivation, cut_derivation.hpp) and written before the tree. Anything the certificate
// cannot express - a row it cannot derive, a split that is not one disjunction, a tree thrown
// away by a restart - makes the search give up on it and say why.

#include "branch_and_bound_internal.hpp"

#include "core/safe_bound.hpp"
#include "cut_derivation.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace sankhya::mip {

namespace {

CertificateTree::Node& grow_to(std::vector<CertificateTree::Node>* nodes, Index node) {
  const auto u = static_cast<std::size_t>(node);
  if (nodes->size() <= u) nodes->resize(u + 1);
  return (*nodes)[u];
}

}  // namespace

bool BranchAndBound::certificate_rows_match() {
  const auto cuts = static_cast<Index>(pool_cuts_.size());
  const bool derived = std::all_of(pool_cuts_.begin(), pool_cuts_.end(),
                                   [](const Cut& cut) { return cut.proof != nullptr; });
  if (working_.num_rows() == original_.num_rows() + cuts && derived) return true;
  certificate_refuse(
      "rows the model does not have, and the certificate cannot derive, were added to the "
      "search (symmetry or objective rows, or an underived cut)");
  return false;
}

bool BranchAndBound::farkas_proves(const std::vector<double>& y) const {
  if (y.size() != static_cast<std::size_t>(working_.num_rows())) return false;
  SafeBoundProblem problem;
  problem.matrix = &working_.matrix;
  problem.row_lower = working_.row_lower;
  problem.row_upper = working_.row_upper;
  problem.col_lower = working_.col_lower;
  problem.col_upper = working_.col_upper;
  std::vector<double> negated(y);
  for (double& v : negated) v = -v;
  return safe_dual_bound(problem, y).value > 0.0 ||
         safe_dual_bound(problem, negated).value > 0.0;
}

std::vector<CertificateTree::CutRow> BranchAndBound::certificate_cut_rows() const {
  std::vector<CertificateTree::CutRow> cuts;
  cuts.reserve(pool_cuts_.size());
  for (const Cut& cut : pool_cuts_) {
    CertificateTree::CutRow row;
    // The entries append_cut_rows() gave the row, and no others.
    for (std::size_t j = 0; j < cut.coeff.size(); ++j) {
      if (std::fabs(cut.coeff[j]) > tol::kZeroDrop) {
        row.coefficients.emplace_back(static_cast<Index>(j), cut.coeff[j]);
      }
    }
    row.rhs = cut.rhs;
    row.proof = cut.proof;
    cuts.push_back(std::move(row));
  }
  return cuts;
}

Model BranchAndBound::certificate_rows() const {
  return detail::with_cut_rows(original_, certificate_cut_rows());
}

void BranchAndBound::certify_round_cuts(std::vector<Cut>* accepted) {
  if (!certificate_mode() || accepted->empty()) return;
  if (certificate_cuts_derived_ + certificate_cuts_dropped_ == 0) {
    std::string off;
    for (const char* option : {"enable_clique_cuts", "enable_flow_cover_cuts",
                               "mip_implied_bound_cuts", "mir_cmir"}) {
      if (options_.get_bool(option))
        off += fmt::format("{}{}", off.empty() ? "" : ", ", option);
    }
    if (!off.empty()) {
      logger_.info("Certificate (#518): off in this mode, their cuts state no derivation: {}",
                   off);
    }
  }
  // Derived against the rows the certificate will state and the model's own column box,
  // which is what the checker has; a cut that needs a bound the search tightened fails.
  const Model rows = certificate_rows();
  const CutCertification outcome =
      certify_cuts(rows, original_.col_lower, original_.col_upper, accepted);
  certificate_cuts_derived_ += outcome.derived;
  certificate_cuts_dropped_ += outcome.dropped;
  logger_.info("Certificate (#518): cuts of this round: {}; {} right-hand side(s) relaxed{}",
               outcome.summary, outcome.relaxed,
               outcome.first_failure.empty()
                   ? std::string()
                   : fmt::format("; first drop: {}", outcome.first_failure));
}

void BranchAndBound::certificate_refuse(const std::string& why) {
  if (certificate_path_.empty() || !certificate_refusal_.empty()) return;
  certificate_refusal_ = why;
}

void BranchAndBound::certificate_record(Index node, const Solution& relaxation,
                                        CertificateTree::Proof proof) {
  if (certificate_path_.empty() || !certificate_refusal_.empty()) return;
  // The model's rows, then the certified cut rows (#518), and nothing else.
  const Index rows = working_.num_rows();
  if (!certificate_rows_match()) return;
  const std::vector<double>* source =
      proof == CertificateTree::Proof::kFarkas ? &relaxation.farkas_dual : &relaxation.row_dual;
  // AN INFEASIBLE NODE'S FARKAS VECTOR MUST PROVE IT EXACTLY. The dual simplex's is the row of
  // B^-1 it stalled on, which ignores columns whose pivot-row entry is below its pivot
  // tolerance, so the aggregate it names can miss a contradiction by a hair. When the safe
  // test (#519) finds neither sign proves anything, the node is solved once more, cold, by
  // the primal simplex, whose phase-1 duals price every column; that costs an LP per such
  // node, in this mode only.
  Solution cold;
  if (proof == CertificateTree::Proof::kFarkas && !farkas_proves(*source)) {
    cold = solve_primal_simplex(working_, node_options_, logger_, scaling_, control_);
    if (cold.status == SolveStatus::kInfeasible && farkas_proves(cold.farkas_dual)) {
      source = &cold.farkas_dual;
      ++certificate_farkas_resolves_;
    }
  }
  CertificateTree::Node& record = grow_to(&certificate_nodes_, node);
  record.y.clear();
  if (source->size() != static_cast<std::size_t>(rows)) {
    // An LP that reported no multipliers proves nothing here; the node falls back on its
    // nearest ancestor's duals, which is weaker but still sound.
    record.proof = CertificateTree::Proof::kNone;
    return;
  }
  // Duals in minimise space; a Farkas vector keeps the engine's sign, which the writer tries
  // both ways.
  const double scale = proof == CertificateTree::Proof::kDual ? sense_ : 1.0;
  for (Index i = 0; i < rows; ++i) {
    const double v = (*source)[static_cast<std::size_t>(i)];
    if (v != 0.0) record.y.emplace_back(i, scale * v);
  }
  record.proof = proof;
}

void BranchAndBound::certificate_children(Index node, Index down, Index up) {
  if (certificate_path_.empty() || !certificate_refusal_.empty()) return;
  for (const Index child : {down, up}) {
    const TreeNode& tree_node = nodes_[static_cast<std::size_t>(child)];
    if (tree_node.change.column >= original_.num_cols()) {
      certificate_refuse("the search branched on a row (objective branching)");
      return;
    }
  }
  CertificateTree::Node& record = grow_to(&certificate_nodes_, node);
  record.down = down;
  record.up = up;
}

void BranchAndBound::finish_certificate() {
  if (certificate_path_.empty()) return;
  if (quadratic_) certificate_refuse("the objective is quadratic");
  if (shared_ != nullptr || seed_ != nullptr) certificate_refuse("the search ran in parallel");
  if (!options_.get_string("resume").empty()) {
    certificate_refuse("the search resumed from a checkpoint");
  }
  (void)certificate_rows_match();
  if (!certificate_refusal_.empty()) {
    logger_.warning("Certificate (#518) not written: {}", certificate_refusal_);
    return;
  }
  CertificateTree tree;
  tree.nodes.resize(nodes_.size());
  for (std::size_t k = 0; k < nodes_.size(); ++k) {
    CertificateTree::Node& out = tree.nodes[k];
    if (k < certificate_nodes_.size()) out = std::move(certificate_nodes_[k]);
    const TreeNode& node = nodes_[k];
    out.parent = node.parent;
    out.has_change = node.has_change;
    out.column = node.change.column;
    out.is_upper = node.change.is_upper;
    out.value = node.change.value;
  }
  certificate_nodes_.clear();
  tree.cuts = certificate_cut_rows();
  const std::vector<double> none;
  const CertificateOutcome outcome =
      write_vipr_certificate(certificate_path_, original_, tree,
                             have_incumbent_ ? incumbent_x_ : none, objective_step_);
  if (!outcome.written) {
    logger_.warning("Certificate (#518) not written: {}", outcome.message);
    return;
  }
  if (outcome.infeasible) {
    logger_.info("Certificate (#518): {}; it proves the model infeasible", outcome.message);
  } else {
    logger_.info("Certificate (#518): {}; it proves the objective {} {:.10g}", outcome.message,
                 original_.sense == ObjSense::kMaximize ? "<=" : ">=",
                 outcome.proved_bound + original_.objective_offset);
  }
  logger_.info(
      "Certificate (#518) {}; {} infeasible node(s) re-solved by the primal simplex "
      "for a Farkas vector that proves them",
      outcome.leaves, certificate_farkas_resolves_);
}

}  // namespace sankhya::mip
