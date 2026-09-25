// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a checkable proof of a branch-and-bound answer, in the VIPR format (#518).
//
// A MILP "optimal" is a claim about every integer point of the model, and nothing in the
// solution lets anyone check it. A certificate does: the model, the incumbent, and a
// derivation in which every leaf of the search tree contributes an inequality proved from
// the rows (or a contradiction, for an infeasible leaf), and every branching node combines
// its two children's inequalities over the disjunction it branched on. The root's inequality
// is then "objective >= bound" under no assumption at all. tools/verify_certificate.py checks
// every step in exact rational arithmetic without linking any of this code.
//
// Format: W. Cheung, A. Gleixner and D. E. Steffy, "Verifying integer programming results",
// IPCO 2017, LNCS 10328 - the VIPR sections VAR, INT, OBJ, CON, RTP, SOL, DER and the
// reasons asm, lin, rnd and uns. Each leaf's inequality is the Neumaier-Shcherbina safe bound
// (src/core/safe_bound.hpp) of its LP duals over exactly the box the checker will use, so a
// floating-point dual never turns into a claim the exact arithmetic cannot confirm. The one
// extension to the format (a lin step's coefficients may be completed by the column bounds)
// is described in the checker's docstring.
//
// Every number is written as the EXACT decimal value of the double the solver holds, so the
// checker's rational arithmetic sees precisely the model and multipliers this code reasoned
// about: 0.1 in an MPS file is written as the 55-digit decimal of the double nearest 0.1.
#pragma once

#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "sankhya/model.hpp"

namespace sankhya::mip {

struct CutProof;  // cut_derivation.hpp

/// The search tree as the certificate needs it. Node 0 is the root.
struct CertificateTree {
  enum class Proof : std::uint8_t {
    kNone,    ///< no LP of its own: the node borrows its nearest ancestor's duals
    kDual,    ///< an optimal LP: `y` holds its row duals, minimise space
    kFarkas,  ///< an infeasible LP: `y` holds its Farkas multipliers, sign unknown
  };
  struct Node {
    Index parent = -1;
    /// The bound this node adds to its parent's box: column, side, value.
    bool has_change = false;
    Index column = -1;
    bool is_upper = false;
    double value = 0.0;
    /// Children, both or neither; a node with none is a leaf.
    Index down = -1;
    Index up = -1;
    Proof proof = Proof::kNone;
    std::vector<std::pair<Index, double>> y;  ///< sparse: the model's rows, then the cuts'
  };
  std::vector<Node> nodes;
  /// The cut rows every node LP carried after the model's, in order, each with its
  /// certified derivation (cut_derivation.hpp): sum coefficients x <= rhs.
  struct CutRow {
    std::vector<std::pair<Index, double>> coefficients;
    double rhs = 0.0;
    std::shared_ptr<const CutProof> proof;
  };
  std::vector<CutRow> cuts;
};

struct CertificateOutcome {
  bool written = false;
  /// Why nothing was written, or a one-line summary of what was.
  std::string message;
  /// The bound the derivation proves on the objective, in the model's sense (no offset).
  double proved_bound = 0.0;
  /// True when what it proves is that the model has no feasible point.
  bool infeasible = false;
  Count derivations = 0;
  /// How the leaves were proved, one line.
  std::string leaves;
};

/// Write the VIPR certificate for `tree` over `model` to `path`. `incumbent` is the best
/// point found (empty when there is none, and the claim is then infeasibility);
/// `objective_step` is the search's integral-objective step, or 0. Never throws.
[[nodiscard]] CertificateOutcome write_vipr_certificate(const std::string& path,
                                                        const Model& model,
                                                        const CertificateTree& tree,
                                                        const std::vector<double>& incumbent,
                                                        double objective_step);

/// The exact decimal value of `v`: an integer when it is one, otherwise every digit of its
/// finite binary expansion. Exposed for the tests.
[[nodiscard]] std::string exact_decimal(double v);

namespace detail {
/// VAR, INT, OBJ and CON (certificate_sections.cpp): the model, every number exact, rows
/// then column bounds, `num_con` constraints in all.
void write_model(std::ostream& out, const Model& model, Index num_con);
/// RTP and SOL for the incumbent (none: a bound with no point) and the proved bound, in the
/// model's sense.
void write_claim(std::ostream& out, const Model& model, const std::vector<double>& incumbent,
                 double proved);
/// The DER lines deriving each cut row (certificate_cuts.cpp): a Chvatal-Gomory `rnd`, or a
/// split's two assumptions, a `lin` under each and the `uns` that discharges them. `emit`
/// writes one line and returns its index; `con_lower` / `con_upper` map every row to the
/// constraint holding its side, and receive each cut row's (row model_rows + k).
void emit_cut_derivations(const std::vector<CertificateTree::CutRow>& cuts, Index model_rows,
                          const std::function<Index(const std::string&)>& emit,
                          std::vector<Index>* con_lower, std::vector<Index>* con_upper);
/// The model with the cut rows appended after its own, `<=` rows: what every node LP was.
[[nodiscard]] Model with_cut_rows(const Model& model,
                                  const std::vector<CertificateTree::CutRow>& cuts);
}  // namespace detail

}  // namespace sankhya::mip
