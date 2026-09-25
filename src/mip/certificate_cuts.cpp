// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the cut rows of a VIPR certificate (#518): each cut the node LPs carried is
// derived from the model before any leaf uses it. A Chvatal-Gomory cut is one `rnd` step; a
// split cut is two assumptions (the disjunction's sides), a `lin` step under each and the
// `uns` step that discharges them, exactly the derivation of split cuts in Cheung, Gleixner
// and Steffy, "Verifying integer programming results", IPCO 2017 (LNCS 10328), sec. 3. The
// multipliers are the ones cut_derivation.cpp certified with the safe bound, so the exact
// checker proves at least the right-hand side written.

#include <string>
#include <vector>

#include <fmt/format.h>

#include "certificate_writer.hpp"
#include "cut_derivation.hpp"

namespace sankhya::mip::detail {
namespace {

std::string coefficient_list(const std::vector<std::pair<Index, double>>& entries) {
  std::string out = fmt::format("{}", entries.size());
  for (const auto& [j, v] : entries) out += fmt::format(" {} {}", j, exact_decimal(v));
  return out;
}

}  // namespace

Model with_cut_rows(const Model& model, const std::vector<CertificateTree::CutRow>& cuts) {
  if (cuts.empty()) return model;
  Model out = model;
  const Index m = model.num_rows();
  const Index n = model.num_cols();
  SparseMatrix matrix(m + static_cast<Index>(cuts.size()), n);
  for (Index j = 0; j < n; ++j) {
    const ColumnView col = model.matrix.column(j);
    for (Index k = 0; k < col.size; ++k) matrix.add_entry(col.rows[k], j, col.values[k]);
  }
  for (std::size_t k = 0; k < cuts.size(); ++k) {
    for (const auto& [j, v] : cuts[k].coefficients) {
      matrix.add_entry(m + static_cast<Index>(k), j, v);
    }
  }
  matrix.finalize(0.0);  // nothing dropped: these are the rows the certificate states
  out.matrix = std::move(matrix);
  out.resize_rows(m + static_cast<Index>(cuts.size()));
  for (std::size_t k = 0; k < cuts.size(); ++k) {
    const auto row = static_cast<std::size_t>(m) + k;
    out.row_lower[row] = -kInfinity;
    out.row_upper[row] = cuts[k].rhs;
  }
  return out;
}

void emit_cut_derivations(const std::vector<CertificateTree::CutRow>& cuts, Index model_rows,
                          const std::function<Index(const std::string&)>& emit,
                          std::vector<Index>* con_lower, std::vector<Index>* con_upper) {
  for (std::size_t k = 0; k < cuts.size(); ++k) {
    const CertificateTree::CutRow& cut = cuts[k];
    const CutProof& proof = *cut.proof;
    const std::string target =
        fmt::format("L {} {}", exact_decimal(cut.rhs), coefficient_list(cut.coefficients));
    Index asm_side[2] = {-1, -1};
    if (proof.split) {
      const std::string pi = coefficient_list(proof.disjunction);
      asm_side[0] = emit(fmt::format("cut{}_a0 L {} {} {{ asm }} -1", k,
                                     exact_decimal(proof.disjunction_rhs), pi));
      asm_side[1] = emit(fmt::format("cut{}_a1 G {} {} {{ asm }} -1", k,
                                     exact_decimal(proof.disjunction_rhs + 1.0), pi));
    }
    // The constraint holding row i's side the multiplier y prices (y > 0: the lower side),
    // the disjunction row being this side's assumption.
    const auto con_of = [&](Index i, bool lower, int s) {
      if (i == proof.rows_seen) return asm_side[s];
      const auto u = static_cast<std::size_t>(i);
      return lower ? (*con_lower)[u] : (*con_upper)[u];
    };
    Index derived[2] = {-1, -1};
    for (int s = 0; s < (proof.split ? 2 : 1); ++s) {
      const CutProof::Side& side = proof.side[s];
      // Row-implied column bounds the completion relied on, each derived from its row with
      // multiplier 1 over the column box, as for a leaf.
      std::string references;
      Index count = 0;
      for (const ImpliedBound& b : side.implied) {
        const Index con = con_of(b.row, !b.uses_row_upper, s);
        const Index index =
            emit(fmt::format("cut{}_i{}_{} {} {} 1 {} {} {{ lin 1 {} 1 }} -1", k, s, b.column,
                             b.uses_row_upper ? 'L' : 'G', exact_decimal(b.rhs), b.column,
                             exact_decimal(b.coefficient), con));
        references += fmt::format(" {} 0", index);
        ++count;
      }
      std::string pairs;
      for (const auto& [i, y] : side.rows) {
        pairs += fmt::format(" {} {}", con_of(i, y > 0.0, s), exact_decimal(-y));
        ++count;
      }
      if (side.assumption != 0.0) {
        pairs += fmt::format(" {} {}", asm_side[s], exact_decimal(-side.assumption));
        ++count;
      }
      const char* reason = proof.split ? "lin" : "rnd";
      const std::string name =
          proof.split ? fmt::format("cut{}_d{}", k, s) : fmt::format("cut{}", k);
      derived[s] = emit(fmt::format("{} {} {{ {} {}{}{} }} -1", name, target, reason, count,
                                    pairs, references));
    }
    Index row_con = derived[0];
    if (proof.split) {
      row_con = emit(fmt::format("cut{} {} {{ uns {} {} {} {} }} -1", k, target, derived[0],
                                 asm_side[0], derived[1], asm_side[1]));
    }
    const auto row = static_cast<std::size_t>(model_rows) + k;
    (*con_upper)[row] = row_con;
    (*con_lower)[row] = -1;
  }
}

}  // namespace sankhya::mip::detail
