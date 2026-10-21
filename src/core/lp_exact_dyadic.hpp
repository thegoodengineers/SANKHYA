// SPDX-License-Identifier: Apache-2.0
// SANKHYA - dyadic rationals m * 2^e for the exact-dual refinement (lp_exact_dual.cpp, #763),
// and the refinement's step history (#1082). Internal to src/core: every operation is exact
// except scaled(), which says so.
#pragma once

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "exact/bigint.hpp"
#include "util/wide_mul.hpp"

namespace sankhya::dyadic_arith {

using exact::BigInt;
using Sz = std::size_t;
inline constexpr int kZero = INT_MIN;  // top() of zero

/// m * 2^e, exactly.
struct Dyadic {
  BigInt m;
  int e = 0;
};

inline Dyadic dyadic(double v) {
  if (v == 0.0) return {};
  int exponent = 0;
  const double mantissa = std::frexp(v, &exponent);
  return {BigInt(static_cast<long long>(std::ldexp(mantissa, 53))), exponent - 53};
}
inline Dyadic operator*(const Dyadic& a, const Dyadic& b) {
  if (a.m.is_zero() || b.m.is_zero()) return {};
  return {a.m * b.m, a.e + b.e};
}
inline Dyadic operator+(const Dyadic& a, const Dyadic& b) {
  if (a.m.is_zero()) return b;
  if (b.m.is_zero()) return a;
  if (a.e == b.e) return {a.m + b.m, a.e};
  if (a.e > b.e) return {a.m.shifted_left(a.e - b.e) + b.m, b.e};
  return {a.m + b.m.shifted_left(b.e - a.e), a.e};
}
inline Dyadic operator-(const Dyadic& a, const Dyadic& b) {
  return a + Dyadic{-b.m, b.e};
}
/// |value| < 2^top(value); kZero for zero.
inline int top(const Dyadic& a) {
  return a.m.is_zero() ? kZero : a.m.bits() + a.e;
}
/// value / 2^shift, to double precision or so - the one inexact operation here.
inline double scaled(const Dyadic& a, int shift) {
  if (a.m.is_zero()) return 0.0;
  std::uint64_t lead = 0;
  int ex = 0;
  a.m.leading_bits(&lead, &ex);
  const double magnitude = std::ldexp(static_cast<double>(lead), ex + a.e - shift);
  return a.m.sign() < 0 ? -magnitude : magnitude;
}
/// a with every bit below 2^position dropped (towards zero): an error under 2^position.
inline Dyadic truncated_below(const Dyadic& a, int position) {
  if (a.m.is_zero() || a.e >= position) return a;
  return {a.m.shifted_right(position - a.e), position};
}
/// The integer nearest to a (halves away from zero).
inline BigInt nearest(const Dyadic& a) {
  if (a.e >= 0) return a.m.shifted_left(a.e);
  // (|m| + 2^(-e-1)) >> -e: a shift, not a division, since the divisor is a power of two.
  const BigInt q = (a.m.abs() + BigInt(1).shifted_left(-a.e - 1)).shifted_right(-a.e);
  return a.m.sign() < 0 ? -q : q;
}

/// The refinement's corrections, kept as they come: step k added steps[k][i] * 2^exponent[k]
/// to x_i. x is the exact sum of them, assembled only when a reconstruction attempt needs it
/// (#1082). Adding each step to a growing BigInt instead rebuilt every x_i, by then
/// thousands of bits, on every iteration: on greenbeb that phase took 13 s of wall time on a
/// loaded machine, and the whole refinement now takes about 2 s of CPU time. The sum is the
/// same exact dyadic number either way, so nothing downstream sees a difference.
// ponytail: one int64 per row per iteration - about 86 MB for pilot87's 5,297 iterations
// on its 2,030 rows, at most about 320 MB at kExactDualMaxIterations on a 2,000-row
// basis; fold old steps into a running BigInt if a larger basis needs it.
class StepHistory {
 public:
  void add(std::vector<long long> steps, int exponent) {
    steps_.push_back(std::move(steps));
    exponents_.push_back(exponent);
  }

  /// x_i = sum_k steps[k][i] 2^exponent[k], exactly: the positive and negative steps are
  /// added into two word arrays at their bit offsets above the smallest exponent, with the
  /// carries propagated, and subtracted once at the end.
  [[nodiscard]] Dyadic value(Sz i) const {
    int low = INT_MAX;
    int high = INT_MIN;
    for (Sz k = 0; k < steps_.size(); ++k) {
      if (steps_[k][i] == 0) continue;
      low = std::min(low, exponents_[k]);
      high = std::max(high, exponents_[k]);
    }
    if (low == INT_MAX) return {};
    // Each |step| < 2^63, and at most steps_.size() of them overlap: two guard words cover
    // the carries out of the top of any sum this long.
    const auto words = static_cast<Sz>((high - low) / 64 + 3);
    std::vector<std::uint64_t> positive(words, 0);
    std::vector<std::uint64_t> negative(words, 0);
    for (Sz k = 0; k < steps_.size(); ++k) {
      const long long step = steps_[k][i];
      if (step == 0) continue;
      const std::uint64_t magnitude =
          step < 0 ? 0ULL - static_cast<std::uint64_t>(step) : static_cast<std::uint64_t>(step);
      std::vector<std::uint64_t>& acc = step < 0 ? negative : positive;
      const int shift = exponents_[k] - low;
      const auto word = static_cast<Sz>(shift / 64);
      const int bits = shift % 64;
      const std::uint64_t parts[2] = {magnitude << bits,
                                      bits == 0 ? 0 : magnitude >> (64 - bits)};
      unsigned carry = 0;
      for (Sz w = word; w < words && (w < word + 2 || carry != 0); ++w) {
        carry = add_carry(acc[w], w < word + 2 ? parts[w - word] : 0, carry, &acc[w]);
      }
    }
    return {BigInt::from_words(positive.data(), words, false) -
                BigInt::from_words(negative.data(), words, false),
            low};
  }

  [[nodiscard]] std::vector<Dyadic> values(Sz size) const {
    std::vector<Dyadic> x(size);
    for (Sz i = 0; i < size; ++i) x[i] = value(i);
    return x;
  }

 private:
  std::vector<std::vector<long long>> steps_;
  std::vector<int> exponents_;
};

}  // namespace sankhya::dyadic_arith
