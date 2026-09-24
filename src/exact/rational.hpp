// SPDX-License-Identifier: Apache-2.0
// SANKHYA - exact rational arithmetic for basis verification (#521).
//
// A separate, independent copy of tests/oracles/rational.hpp's design (that file is TESTS
// ONLY by deliberate choice, documented there: nothing in src/ includes it, and changing that
// boundary is a bigger decision than this feature needs to make). Same representation and
// the same reason for it: a normalised fraction over __int128 with a strictly positive
// denominator, every operation overflow-checked and throwing rather than wrapping, because a
// wrapped intermediate here would manufacture a false proof of exactness - worse than no
// proof. See exact_verify.hpp for what OVERFLOW means in practice: this module declines
// rather than lies.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>

namespace sankhya::exact {

/// Thrown when an exact operation, or a double-to-exact conversion, cannot be represented in
/// __int128. Caught once, at the top of verify_basis_exact() (exact_verify.cpp), and turned
/// into a declined-not-failed verdict.
struct RationalOverflow : std::exception {
  [[nodiscard]] const char* what() const noexcept override {
    return "exact rational arithmetic overflowed __int128";
  }
};

class Rational {
 public:
  using Int = __int128;

  Rational() = default;
  Rational(Int numerator) : numerator_(numerator), denominator_(1) {}  // NOLINT: implicit
  Rational(Int numerator, Int denominator) : numerator_(numerator), denominator_(denominator) {
    normalize();
  }

  [[nodiscard]] Int numerator() const noexcept { return numerator_; }
  [[nodiscard]] Int denominator() const noexcept { return denominator_; }

  [[nodiscard]] bool is_zero() const noexcept { return numerator_ == 0; }
  [[nodiscard]] int sign() const noexcept {
    return numerator_ > 0 ? 1 : (numerator_ < 0 ? -1 : 0);
  }

  /// Nearest double. Used only to report a result or to compare against the float solver -
  /// never inside this module's own arithmetic.
  [[nodiscard]] double to_double() const noexcept {
    return static_cast<double>(numerator_) / static_cast<double>(denominator_);
  }

  /// The EXACT value of a finite double, bit for bit - not the decimal a human wrote, which
  /// this module never sees (the MPS reader already rounded it to the nearest double before
  /// the Model existed). A double is exactly mantissa * 2^exponent with a 53-bit mantissa
  /// (std::frexp normalises the mantissa to [0.5, 1), so scaling it by 2^53 is exactly an
  /// integer - no rounding happens in this function; the only way it loses information is if
  /// the caller already handed it a value that lost information becoming a double, which is
  /// not this function's problem to fix.
  [[nodiscard]] static Rational from_double(double value) {
    if (value == 0.0) return Rational(0);
    if (!std::isfinite(value)) throw RationalOverflow();
    int exponent = 0;
    const double mantissa = std::frexp(value, &exponent);
    constexpr int kMantissaBits = 53;
    const auto scaled_mantissa = static_cast<Int>(std::ldexp(mantissa, kMantissaBits));
    // value == scaled_mantissa * 2^(exponent - kMantissaBits)
    const int shift = exponent - kMantissaBits;
    if (shift >= 0) {
      // An integer: scaled_mantissa << shift. Bounded by the sign bit and the magnitude
      // already occupying up to 53 bits of scaled_mantissa.
      if (shift >= 127 - kMantissaBits) throw RationalOverflow();
      return Rational(shift_left_checked(scaled_mantissa, shift));
    }
    // A proper fraction: scaled_mantissa / 2^(-shift).
    const int negative_shift = -shift;
    if (negative_shift >= 127) throw RationalOverflow();
    return Rational(scaled_mantissa, shift_left_checked(Int(1), negative_shift));
  }

  Rational operator-() const {
    Rational r;
    r.numerator_ = negate(numerator_);
    r.denominator_ = denominator_;
    return r;
  }

  Rational operator+(const Rational& other) const {
    // a/b + c/d = (a*d + c*b) / (b*d)
    return Rational(
        add(multiply(numerator_, other.denominator_), multiply(other.numerator_, denominator_)),
        multiply(denominator_, other.denominator_));
  }

  Rational operator-(const Rational& other) const { return *this + (-other); }

  Rational operator*(const Rational& other) const {
    return Rational(multiply(numerator_, other.numerator_),
                    multiply(denominator_, other.denominator_));
  }

  Rational operator/(const Rational& other) const {
    if (other.numerator_ == 0) throw RationalOverflow();  // division by zero is a bug here
    return Rational(multiply(numerator_, other.denominator_),
                    multiply(denominator_, other.numerator_));
  }

  Rational& operator+=(const Rational& other) { return *this = *this + other; }
  Rational& operator-=(const Rational& other) { return *this = *this - other; }
  Rational& operator*=(const Rational& other) { return *this = *this * other; }

  /// Exact comparison. a/b < c/d with b, d > 0 is a*d < c*b, and the products are checked.
  bool operator<(const Rational& other) const {
    return multiply(numerator_, other.denominator_) < multiply(other.numerator_, denominator_);
  }
  bool operator>(const Rational& other) const { return other < *this; }
  bool operator<=(const Rational& other) const { return !(other < *this); }
  bool operator>=(const Rational& other) const { return !(*this < other); }
  bool operator==(const Rational& other) const {
    // Both sides are normalised, so equality is componentwise and needs no multiplication.
    return numerator_ == other.numerator_ && denominator_ == other.denominator_;
  }
  bool operator!=(const Rational& other) const { return !(*this == other); }

 private:
  static Int absolute(Int v) { return v < 0 ? negate(v) : v; }

  static Int negate(Int v) {
    if (v == kMin)
      throw RationalOverflow();  // the most negative value has no positive counterpart
    return -v;
  }

  static Int add(Int a, Int b) {
    Int result = 0;
    if (__builtin_add_overflow(a, b, &result)) throw RationalOverflow();
    return result;
  }

  static Int multiply(Int a, Int b) {
    Int result = 0;
    if (__builtin_mul_overflow(a, b, &result)) throw RationalOverflow();
    return result;
  }

  /// v * 2^shift, checked. Callers bound `shift` to at most 126 so `Int(1) << shift` is
  /// itself representable; the multiply below is where an over-large `v` is caught, reusing
  /// the same overflow check as every other operation here rather than a second, hand-rolled
  /// one for shifting.
  static Int shift_left_checked(Int v, int shift) {
    if (v == 0) return 0;
    return multiply(v, static_cast<Int>(1) << shift);
  }

  static Int greatest_common_divisor(Int a, Int b) {
    a = absolute(a);
    b = absolute(b);
    while (b != 0) {
      const Int t = a % b;
      a = b;
      b = t;
    }
    return a;
  }

  void normalize() {
    if (denominator_ == 0) throw RationalOverflow();  // 1/0 is a bug in the caller
    if (denominator_ < 0) {
      numerator_ = negate(numerator_);
      denominator_ = negate(denominator_);
    }
    if (numerator_ == 0) {
      denominator_ = 1;
      return;
    }
    const Int g = greatest_common_divisor(numerator_, denominator_);
    if (g > 1) {
      numerator_ /= g;
      denominator_ /= g;
    }
  }

  static constexpr Int kMin = static_cast<Int>(1) << 127;

  Int numerator_ = 0;
  Int denominator_ = 1;
};

}  // namespace sankhya::exact
