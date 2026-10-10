// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a signed 128-bit integer for the reference oracle on compilers without __int128
// (MSVC, #748).
//
// TESTS ONLY, like the rest of the oracle. Two's complement in two 64-bit words, with the
// semantics of GCC's __int128: +, -, * and << wrap modulo 2^128, / and % truncate toward
// zero, the conversion to double rounds to nearest. The oracle never relies on wrapping -
// it calls add_overflow / mul_overflow, which report it, as __builtin_*_overflow does.
// Division is binary long division (Knuth, "The Art of Computer Programming", vol. 2, 3rd
// ed., section 4.3.1), with the 64-bit hardware divide when both operands fit. Everything
// here is compiled on every compiler and checked against __int128 where the compiler has
// it (tests/oracles/test_int128.cpp).
#pragma once

#include <bit>
#include <cmath>
#include <compare>
#include <cstdint>

#include "util/wide_mul.hpp"

namespace sankhya::oracle {

class Int128 {
 public:
  constexpr Int128() = default;
  constexpr Int128(long long v) noexcept  // NOLINT: implicit, as for the built-in integers
      : lo_(static_cast<std::uint64_t>(v)), hi_(v < 0 ? ~std::uint64_t{0} : 0) {}

  explicit constexpr operator long long() const noexcept { return static_cast<long long>(lo_); }

  /// The two's-complement words, for the test that compares this class with __int128.
  [[nodiscard]] constexpr std::uint64_t low_word() const noexcept { return lo_; }
  [[nodiscard]] constexpr std::uint64_t high_word() const noexcept { return hi_; }

  /// Nearest double, ties to even, as the built-in conversion.
  explicit operator double() const noexcept {
    const double m = to_double(magnitude(*this));
    return negative() ? -m : m;
  }

  friend constexpr bool operator==(const Int128& a, const Int128& b) noexcept {
    return a.lo_ == b.lo_ && a.hi_ == b.hi_;
  }
  friend constexpr std::strong_ordering operator<=>(const Int128& a, const Int128& b) noexcept {
    const auto ah = static_cast<long long>(a.hi_);
    const auto bh = static_cast<long long>(b.hi_);
    if (ah != bh) return ah <=> bh;
    return a.lo_ <=> b.lo_;
  }

  friend constexpr Int128 operator+(const Int128& a, const Int128& b) noexcept {
    const std::uint64_t lo = a.lo_ + b.lo_;
    return make(lo, a.hi_ + b.hi_ + (lo < a.lo_ ? 1 : 0));
  }
  friend constexpr Int128 operator-(const Int128& a) noexcept {
    return make(~a.lo_, ~a.hi_) + Int128(1);
  }
  friend constexpr Int128 operator-(const Int128& a, const Int128& b) noexcept {
    return a + (-b);
  }
  friend Int128 operator*(const Int128& a, const Int128& b) noexcept {
    const U128 p = mul_64x64(a.lo_, b.lo_);
    return make(p.lo, p.hi + a.lo_ * b.hi_ + a.hi_ * b.lo_);
  }
  friend constexpr Int128 operator<<(const Int128& a, int s) noexcept {
    if (s == 0) return a;
    if (s >= 64) return make(0, a.lo_ << (s - 64));
    return make(a.lo_ << s, (a.hi_ << s) | (a.lo_ >> (64 - s)));
  }
  friend Int128 operator/(const Int128& a, const Int128& b) noexcept {
    U128 q;
    U128 r;
    divmod(magnitude(a), magnitude(b), &q, &r);
    const Int128 m = make(q.lo, q.hi);
    return a.negative() != b.negative() ? -m : m;
  }
  friend Int128 operator%(const Int128& a, const Int128& b) noexcept {
    U128 q;
    U128 r;
    divmod(magnitude(a), magnitude(b), &q, &r);
    const Int128 m = make(r.lo, r.hi);
    return a.negative() ? -m : m;  // the remainder takes the dividend's sign
  }
  Int128& operator+=(const Int128& b) noexcept { return *this = *this + b; }
  Int128& operator-=(const Int128& b) noexcept { return *this = *this - b; }
  Int128& operator*=(const Int128& b) noexcept { return *this = *this * b; }
  Int128& operator/=(const Int128& b) noexcept { return *this = *this / b; }
  Int128& operator%=(const Int128& b) noexcept { return *this = *this % b; }

  /// *r = a + b modulo 2^128; true when the exact sum does not fit.
  static bool add_overflow(const Int128& a, const Int128& b, Int128* r) noexcept {
    *r = a + b;
    return a.negative() == b.negative() && r->negative() != a.negative();
  }

  /// *r = a * b modulo 2^128; true when the exact product does not fit.
  static bool mul_overflow(const Int128& a, const Int128& b, Int128* r) noexcept {
    *r = a * b;
    const U128 ua = magnitude(a);
    const U128 ub = magnitude(b);
    if (ua.hi != 0 && ub.hi != 0) return true;
    // At most one high word is non-zero: |a||b| = lo*lo + (cross << 64).
    U128 m = mul_64x64(ua.lo, ub.lo);
    const U128 cross = mul_64x64(ua.hi != 0 ? ua.hi : ub.hi, ua.hi != 0 ? ub.lo : ua.lo);
    if (cross.hi != 0) return true;
    if (add_carry(m.hi, cross.lo, 0, &m.hi) != 0) return true;
    // |product| must be below 2^127, or equal to it for a negative product.
    const bool negative_product = a.negative() != b.negative();
    if (m.hi < (std::uint64_t{1} << 63)) return false;
    return !(negative_product && m.hi == (std::uint64_t{1} << 63) && m.lo == 0);
  }

 private:
  static constexpr Int128 make(std::uint64_t lo, std::uint64_t hi) noexcept {
    Int128 r;
    r.lo_ = lo;
    r.hi_ = hi;
    return r;
  }
  [[nodiscard]] constexpr bool negative() const noexcept { return (hi_ >> 63) != 0; }

  /// |a| as an unsigned 128-bit value; exact for the most negative value too.
  static constexpr U128 magnitude(const Int128& a) noexcept {
    const Int128 m = a.negative() ? -a : a;
    return {m.lo_, m.hi_};
  }

  /// u = q v + r for unsigned 128-bit values, v non-zero. Division by zero is undefined for
  /// __int128 as well; the oracle never divides by zero.
  static void divmod(U128 u, U128 v, U128* q, U128* r) noexcept {
    if (u.hi == 0 && v.hi == 0) {
      *q = {u.lo / v.lo, 0};
      *r = {u.lo % v.lo, 0};
      return;
    }
    *q = {0, 0};
    *r = {0, 0};
    const int top = u.hi != 0 ? 127 - std::countl_zero(u.hi) : 63 - std::countl_zero(u.lo);
    for (int bit = top; bit >= 0; --bit) {
      // r = 2 r + bit of u
      r->hi = (r->hi << 1) | (r->lo >> 63);
      r->lo = (r->lo << 1) | ((bit >= 64 ? u.hi >> (bit - 64) : u.lo >> bit) & 1U);
      if (r->hi > v.hi || (r->hi == v.hi && r->lo >= v.lo)) {
        const std::uint64_t borrow = r->lo < v.lo ? 1 : 0;
        r->lo -= v.lo;
        r->hi -= v.hi + borrow;
        if (bit >= 64) {
          q->hi |= std::uint64_t{1} << (bit - 64);
        } else {
          q->lo |= std::uint64_t{1} << bit;
        }
      }
    }
  }

  static double to_double(U128 m) noexcept {
    if (m.hi == 0) return static_cast<double>(m.lo);
    // The top 64 bits, with every bit below them folded into the lowest one as a sticky bit:
    // that bit sits under the rounding position, so it breaks a tie exactly as the bits it
    // stands for would, and the one 64-bit conversion rounds correctly.
    const int n = std::countl_zero(m.hi);
    const std::uint64_t top = n == 0 ? m.hi : (m.hi << n) | (m.lo >> (64 - n));
    const std::uint64_t rest = n == 0 ? m.lo : m.lo << n;
    return std::ldexp(static_cast<double>(top | (rest != 0 ? 1U : 0U)), 64 - n);
  }

  std::uint64_t lo_ = 0;
  std::uint64_t hi_ = 0;
};

}  // namespace sankhya::oracle
