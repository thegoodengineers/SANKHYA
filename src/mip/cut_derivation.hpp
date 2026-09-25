// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a derivation for every cut row a certificate uses (#518).
//
// A MILP certificate (certificate_writer.hpp) proves each leaf's bound from the LP rows the
// leaf was solved over. A cut row is not a row of the model, so the proof must first derive
// it from the model, in a form an exact checker can repeat. The two forms used are the ones
// of the VIPR paper (Cheung, Gleixner and Steffy, "Verifying integer programming results",
// IPCO 2017, LNCS 10328, sec. 3):
//
//  * CHVATAL-GOMORY ROUNDING (VIPR `rnd`): a nonnegative combination of rows, completed over
//    the column bounds, proves a.x <= P for integer a on integer columns, hence
//    a.x <= floor(P). Knapsack covers are this with one row and multiplier 1 / max a_C
//    (the multiplier is chosen here as the minimiser of P over the row's breakpoints), and
//    {0,1/2}-cuts are this with multipliers 1/2 (Caprara and Fischetti, "{0,1/2}-Chvatal-
//    Gomory cuts", Math. Programming 74, 1996).
//
//  * A SPLIT (VIPR `asm`, `lin`, `uns`): the cut is derived twice, once under pi.x <= pi0
//    and once under pi.x >= pi0 + 1 (pi integer on integer columns, zero elsewhere), and the
//    two are combined over the disjunction. Gomory mixed-integer and MIR cuts are split cuts
//    (Nemhauser and Wolsey, "A recursive procedure to generate all cuts for 0-1 mixed integer
//    programs", Math. Programming 46, 1990; Cornuejols, "Valid inequalities for mixed
//    integer linear programs", Math. Programming 112, 2008, sec. 5): the generator states
//    the disjunction and the multipliers of each side.
//
// A generator only proposes multipliers (a CutDerivation). certify_cuts() turns them into a
// proof with the Neumaier-Shcherbina safe bound (core/safe_bound.hpp): the combination is
// evaluated with outward rounding over the model's own column box, so the right-hand side it
// certifies is never below what the exact combination proves. A cut the proof cannot reach
// is dropped from the round; one it reaches only up to rounding (at most
// tol::kCertificateCutSlack) has its right-hand side relaxed to the proved value before it
// becomes a row, so the LP and the certificate hold the same row.
#pragma once

#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "core/safe_bound.hpp"
#include "cuts.hpp"
#include "sankhya/model.hpp"

namespace sankhya::mip {

/// What a generator knows about how its cut follows from the rows. Row multipliers are in
/// the cut's own `<=` orientation: w > 0 on a row's upper side, w < 0 on its lower side, so
/// sum_i w_i (row i) is a `<=` inequality.
struct CutDerivation {
  enum class Kind : std::uint8_t {
    kRounding,           ///< floor of the combination `rows[0]`
    kSingleRowRounding,  ///< floor of lambda * (row `row`'s upper side), lambda chosen here
    kSplit,              ///< side s: rows[s] and mu[s] times the disjunction's side s
  };
  Kind kind = Kind::kRounding;
  std::vector<std::pair<Index, double>> rows[2];
  /// kSplit: multiplier on pi.x <= pi0 (side 0, >= 0) and on pi.x >= pi0 + 1 (side 1, <= 0).
  double mu[2] = {0.0, 0.0};
  std::vector<std::pair<Index, double>> disjunction;  ///< pi, integer values
  double disjunction_rhs = 0.0;                       ///< pi0, an integer
  Index row = -1;                                     ///< kSingleRowRounding
};

/// A certified derivation: what the certificate writes for the cut. Multipliers are in the
/// safe bound's minimise convention (y > 0 prices a row's lower side) over the rows the
/// derivation saw (the model's, then earlier cut rows); `rows_seen` is their number. An
/// implied bound whose row is `rows_seen` comes from the disjunction side of that proof.
struct CutProof {
  bool split = false;
  Index rows_seen = 0;
  struct Side {
    std::vector<std::pair<Index, double>> rows;
    double assumption = 0.0;
    std::vector<ImpliedBound> implied;
  };
  Side side[2];  ///< side[0] alone for a rounding
  std::vector<std::pair<Index, double>> disjunction;
  double disjunction_rhs = 0.0;
};

/// How a batch of cuts fared, per family, for the log.
struct CutCertification {
  Count derived = 0;
  Count relaxed = 0;  ///< right-hand side relaxed by at most kCertificateCutSlack
  Count dropped = 0;
  std::string summary;  ///< "gomory 12 derived, 1 dropped; knapsack_cover 2 derived"
  std::string first_failure;
};

/// Certify each cut of `cuts` (the `sum coeff x <= rhs` convention) against the rows of
/// `rows` (its matrix and row bounds: the model's rows, then earlier cut rows) over the
/// column box `col_lower`/`col_upper` - the box the certificate's checker has. A cut is
/// kept with `proof` set, or removed. Coefficients at or below tol::kZeroDrop are zeroed
/// first, as append_cut_rows would drop them.
CutCertification certify_cuts(const Model& rows, std::span<const double> col_lower,
                              std::span<const double> col_upper, std::vector<Cut>* cuts);

/// The split a Gomory mixed-integer cut from `tableau` rests on (see the file header), or
/// null when a column's status leaves it unstated. Recomputed from the tableau row with the
/// generator's own formulas; certify_cuts() checks whatever it proposes.
[[nodiscard]] std::shared_ptr<const CutDerivation> gmi_derivation(
    const Model& model, const detail::ReconstructedTableauRow& tableau);

}  // namespace sankhya::mip
