// SPDX-License-Identifier: Apache-2.0
// SANKHYA - exact rational arithmetic for basis verification (#521) and certified
// sensitivity (#757).
//
// A separate, independent copy of tests/oracles/rational.hpp's design (that file is TESTS
// ONLY by deliberate choice, documented there: nothing in src/ includes it, and changing that
// boundary is a bigger decision than this feature needs to make): a normalised fraction with
// a strictly positive denominator. Since #757 over the arbitrary-precision BigInt
// (bigint.hpp) rather than __int128, which overflowed on almost any decimal data and made
// the exact modules decline where they were most wanted.
#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <string>
#include <utility>

#include "exact/bigint.hpp"

namespace sankhya::exact {

/// Thrown on an operation with no exact rational result: a division by zero, or a double
/// that is not finite. The name is kept from the __int128 days, when overflow was the common
/// cause; callers turn it into a declined-not-failed verdict.
struct RationalOverflow : std::exception {
  [[nodiscard]] const char* what() const noexcept override {
    return "no exact rational result (a division by zero or a non-finite input)";
  }
};

/// Thrown when an exact computation runs past its time budget (option exact_seconds). The
/// callers turn it into a declined verdict: running out of time proves nothing either way.
struct ExactBudgetExceeded : std::exception {
  [[nodiscard]] const char* what() const noexcept override {
    return "the exact computation ran past its time budget";
  }
};

/// A wall-clock budget for the exact modules. Exact arithmetic on a dense basis is O(m^3)
/// operations on numbers that grow with the elimination, so a 220-row Netlib basis (brandy)
/// ran for over forty minutes after the solve itself had finished; the budget makes that a
/// stated decline instead of a hang.
class Deadline {
 public:
  explicit Deadline(double seconds)
      : unlimited_(!(seconds < 1e18)),
        end_(std::chrono::steady_clock::now() +
             std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                 std::chrono::duration<double>(unlimited_ ? 0.0 : seconds))) {}
  void check() const {
    if (!unlimited_ && std::chrono::steady_clock::now() > end_) throw ExactBudgetExceeded();
  }

 private:
  bool unlimited_;
  std::chrono::steady_clock::time_point end_;
};

class Rational {
 public:
  using Int = BigInt;

  Rational() = default;
  Rational(long long numerator) : numerator_(numerator), denominator_(1) {}       // NOLINT
  Rational(Int numerator) : numerator_(std::move(numerator)), denominator_(1) {}  // NOLINT
  Rational(Int numerator, Int denominator)
      : numerator_(std::move(numerator)), denominator_(std::move(denominator)) {
    normalize();
  }

  [[nodiscard]] const Int& numerator() const noexcept { return numerator_; }
  [[nodiscard]] const Int& denominator() const noexcept { return denominator_; }

  [[nodiscard]] bool is_zero() const noexcept { return numerator_.is_zero(); }
  [[nodiscard]] int sign() const noexcept { return numerator_.sign(); }

  /// Nearest double, near enough to report (the leading 64 bits of each side). Never used
  /// inside this module's own arithmetic.
  [[nodiscard]] double to_double() const noexcept {
    if (numerator_.is_zero()) return 0.0;
    std::uint64_t num = 0;
    std::uint64_t den = 0;
    int num_exp = 0;
    int den_exp = 0;
    numerator_.leading_bits(&num, &num_exp);
    denominator_.leading_bits(&den, &den_exp);
    const double ratio = static_cast<double>(num) / static_cast<double>(den);
    return (numerator_.sign() < 0 ? -1.0 : 1.0) * std::ldexp(ratio, num_exp - den_exp);
  }

  /// "numerator/denominator" in decimal, exact.
  [[nodiscard]] std::string to_string() const {
    return numerator_.to_string() + "/" + denominator_.to_string();
  }

  /// The EXACT value of a finite double, bit for bit - not the decimal a human wrote, which
  /// this module never sees (the MPS reader already rounded it to the nearest double before
  /// the Model existed). A double is exactly mantissa * 2^exponent with a 53-bit mantissa
  /// (std::frexp normalises the mantissa to [0.5, 1), so scaling it by 2^53 is exactly an
  /// integer), so no rounding happens in this function.
  [[nodiscard]] static Rational from_double(double value) {
    if (value == 0.0) return Rational(0);
    if (!std::isfinite(value)) throw RationalOverflow();
    int exponent = 0;
    const double mantissa = std::frexp(value, &exponent);
    constexpr int kMantissaBits = 53;
    const auto scaled_mantissa = static_cast<long long>(std::ldexp(mantissa, kMantissaBits));
    const int shift = exponent - kMantissaBits;  // value == scaled_mantissa * 2^shift
    if (shift >= 0) return Rational(Int(scaled_mantissa) * Int::power_of_two(shift));
    return Rational(Int(scaled_mantissa), Int::power_of_two(-shift));
  }

  Rational operator-() const {
    Rational r;
    r.numerator_ = -numerator_;
    r.denominator_ = denominator_;
    return r;
  }

  // Sums and products follow Knuth (TAOCP vol. 2, 3rd ed., section 4.5.1): the gcds are
  // taken of the operands' smaller parts before multiplying, so the result is already in
  // lowest terms and no gcd of the full-size product is ever needed. A gcd of 1 - the common
  // case - skips the divisions as well.
  Rational operator+(const Rational& other) const {
    if (is_zero()) return other;
    if (other.is_zero()) return *this;
    const Int g = Int::gcd(denominator_, other.denominator_);
    if (g.is_one()) {
      return raw(numerator_ * other.denominator_ + other.numerator_ * denominator_,
                 denominator_ * other.denominator_);
    }
    const Int mine = denominator_ / g;
    Int t = numerator_ * (other.denominator_ / g) + other.numerator_ * mine;
    if (t.is_zero()) return Rational(0);
    const Int g2 = Int::gcd(t, g);
    if (g2.is_one()) return raw(std::move(t), mine * other.denominator_);
    return raw(t / g2, mine * (other.denominator_ / g2));
  }
  Rational operator-(const Rational& other) const { return *this + (-other); }
  Rational operator*(const Rational& other) const {
    if (is_zero() || other.is_zero()) return Rational(0);
    const Int g1 = Int::gcd(numerator_, other.denominator_);
    const Int g2 = Int::gcd(other.numerator_, denominator_);
    return raw(reduced(numerator_, g1) * reduced(other.numerator_, g2),
               reduced(denominator_, g2) * reduced(other.denominator_, g1));
  }
  Rational operator/(const Rational& other) const {
    if (other.is_zero()) throw RationalOverflow();
    if (is_zero()) return Rational(0);
    const Int g1 = Int::gcd(numerator_, other.numerator_);
    const Int g2 = Int::gcd(other.denominator_, denominator_);
    Int num = reduced(numerator_, g1) * reduced(other.denominator_, g2);
    Int den = reduced(denominator_, g2) * reduced(other.numerator_, g1);
    if (den.sign() < 0) {
      num = -num;
      den = -den;
    }
    return raw(std::move(num), std::move(den));
  }

  Rational& operator+=(const Rational& other) { return *this = *this + other; }
  Rational& operator-=(const Rational& other) { return *this = *this - other; }
  Rational& operator*=(const Rational& other) { return *this = *this * other; }

  bool operator<(const Rational& other) const {
    if (sign() != other.sign()) return sign() < other.sign();
    if (denominator_ == other.denominator_) return numerator_ < other.numerator_;
    return numerator_ * other.denominator_ < other.numerator_ * denominator_;
  }
  bool operator>(const Rational& other) const { return other < *this; }
  bool operator<=(const Rational& other) const { return !(other < *this); }
  bool operator>=(const Rational& other) const { return !(*this < other); }
  bool operator==(const Rational& other) const {
    // Both sides are normalised, so equality is componentwise.
    return numerator_ == other.numerator_ && denominator_ == other.denominator_;
  }
  bool operator!=(const Rational& other) const { return !(*this == other); }

 private:
  /// Already in lowest terms with a positive denominator: no gcd.
  static Rational raw(Int numerator, Int denominator) {
    Rational r;
    r.numerator_ = std::move(numerator);
    r.denominator_ = std::move(denominator);
    return r;
  }
  static Int reduced(const Int& value, const Int& divisor) {
    return divisor.is_one() ? value : value / divisor;
  }

  void normalize() {
    if (denominator_.is_zero()) throw RationalOverflow();
    if (denominator_.sign() < 0) {
      numerator_ = -numerator_;
      denominator_ = -denominator_;
    }
    if (numerator_.is_zero()) {
      denominator_ = Int(1);
      return;
    }
    const Int g = Int::gcd(numerator_, denominator_);
    if (!g.is_one()) {
      numerator_ = numerator_ / g;
      denominator_ = denominator_ / g;
    }
  }

  Int numerator_;
  Int denominator_ = Int(1);
};

}  // namespace sankhya::exact
