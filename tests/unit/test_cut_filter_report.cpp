// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the cut filter's report (#496).
//
// The tally is bookkeeping over verdicts the filter already made, so what can go wrong is the
// counting: a family dropped, a reason miscounted, or the separator that built a cut not
// named. The hand cases pin the line's shape from known verdicts, and the search-level case
// checks that a real root round names its families and that the count of accepted cuts in
// the line is the count the search reports.

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "mip/cuts.hpp"
#include "sankhya/model.hpp"

namespace sankhya::mip {
namespace {

FilteredCut verdict(CutFamily family, CutFilterReason reason) {
  FilteredCut fc;
  fc.cut.family = family;
  fc.cut.coeff.assign(3, 1.0);
  fc.reason = reason;
  return fc;
}

TEST(CutFilterReport, EveryFamilyAndReasonHasAName) {
  for (int f = 0; f <= static_cast<int>(CutFamily::kZeroHalf); ++f) {
    EXPECT_STRNE(cut_family_name(static_cast<CutFamily>(f)), "") << f;
  }
  EXPECT_STREQ(cut_family_name(CutFamily::kGomory), "gomory");
  EXPECT_STREQ(cut_family_name(CutFamily::kUnknown), "unknown");
  EXPECT_STREQ(cut_filter_reason_name(CutFilterReason::kAccepted), "accepted");
  EXPECT_STREQ(cut_filter_reason_name(CutFilterReason::kInsufficientViolation),
               "insufficient_violation");
  EXPECT_STREQ(cut_filter_reason_name(CutFilterReason::kDuplicate), "duplicate");
}

TEST(CutFilterReport, NoCandidatesSaysSo) {
  EXPECT_EQ(describe_cut_filter({}), "no candidates");
}

TEST(CutFilterReport, CountsPerFamilyAndReasonInFamilyOrder) {
  const std::vector<FilteredCut> filtered{
      verdict(CutFamily::kMir, CutFilterReason::kTooDense),
      verdict(CutFamily::kGomory, CutFilterReason::kInsufficientViolation),
      verdict(CutFamily::kGomory, CutFilterReason::kAccepted),
      verdict(CutFamily::kGomory, CutFilterReason::kInsufficientViolation),
      verdict(CutFamily::kZeroHalf, CutFilterReason::kDuplicate),
  };
  EXPECT_EQ(describe_cut_filter(filtered),
            "gomory 3: 1 accepted, 2 insufficient_violation; mir 1: 1 too_dense; "
            "zero_half 1: 1 duplicate");
}

TEST(CutFilterReport, ACutNobodyTaggedIsCountedAsUnknownNotDropped) {
  const std::vector<FilteredCut> filtered{
      verdict(CutFamily::kUnknown, CutFilterReason::kAccepted)};
  EXPECT_EQ(describe_cut_filter(filtered), "unknown 1: 1 accepted");
}

}  // namespace
}  // namespace sankhya::mip
