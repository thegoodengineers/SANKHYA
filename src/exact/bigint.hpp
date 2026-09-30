// SPDX-License-Identifier: Apache-2.0
// SANKHYA - arbitrary-precision signed integers for exact rational arithmetic (#757).
//
// The exact modules first ran over __int128 and declined on overflow. That was honest and,
// on real data, almost always: a decimal coefficient such as 0.3 is a 53-bit numerator over
// 2^54 as a double, and the determinant of a 3x3 basis of such numbers already needs about
// 160 bits, so the crude blend demo itself could not be checked. Nothing here overflows; a
// number grows as its value needs.
//
// Sign and magnitude, the magnitude in 32-bit limbs, least significant first, with no
// leading zero limbs (zero is the empty vector, never negative). Schoolbook addition,
// subtraction and multiplication; division is Knuth's Algorithm D (Knuth, "The Art of
// Computer Programming", vol. 2, 3rd ed., section 4.3.1), written in the form of Warren,
// "Hacker's Delight", 2nd ed., section 9-2. Quadratic everywhere: the numbers this project
// meets are hundreds to a few thousand bits, where that is the fast choice.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace sankhya::exact {

class BigInt {
 public:
  BigInt() = default;
  BigInt(long long value) {  // NOLINT: implicit, as for the built-in integers
    neg_ = value < 0;
    unsigned long long magnitude = neg_ ? 0ULL - static_cast<unsigned long long>(value)
                                        : static_cast<unsigned long long>(value);
    while (magnitude != 0) {
      mag_.push_back(static_cast<std::uint32_t>(magnitude));
      magnitude >>= 32;
    }
  }

  /// 2^k.
  [[nodiscard]] static BigInt power_of_two(int k) {
    BigInt r;
    r.mag_.assign(static_cast<std::size_t>(k / 32) + 1, 0);
    r.mag_.back() = std::uint32_t{1} << (k % 32);
    return r;
  }

  [[nodiscard]] bool is_zero() const noexcept { return mag_.empty(); }
  [[nodiscard]] int sign() const noexcept { return mag_.empty() ? 0 : (neg_ ? -1 : 1); }
  [[nodiscard]] BigInt abs() const {
    BigInt r = *this;
    r.neg_ = false;
    return r;
  }

  BigInt operator-() const {
    BigInt r = *this;
    if (!r.mag_.empty()) r.neg_ = !r.neg_;
    return r;
  }

  friend BigInt operator+(const BigInt& a, const BigInt& b) {
    if (a.neg_ == b.neg_) return make(add_mag(a.mag_, b.mag_), a.neg_);
    const int c = cmp_mag(a.mag_, b.mag_);
    if (c == 0) return {};
    return c > 0 ? make(sub_mag(a.mag_, b.mag_), a.neg_)
                 : make(sub_mag(b.mag_, a.mag_), b.neg_);
  }
  friend BigInt operator-(const BigInt& a, const BigInt& b) { return a + (-b); }
  friend BigInt operator*(const BigInt& a, const BigInt& b) {
    return make(mul_mag(a.mag_, b.mag_), a.neg_ != b.neg_);
  }
  /// Truncating division, as for the built-in integers. `b` must be nonzero.
  friend BigInt operator/(const BigInt& a, const BigInt& b) {
    std::vector<std::uint32_t> q, r;
    divmod_mag(a.mag_, b.mag_, &q, &r);
    return make(std::move(q), a.neg_ != b.neg_);
  }
  /// The remainder of truncating division: the sign of `a`.
  friend BigInt operator%(const BigInt& a, const BigInt& b) {
    std::vector<std::uint32_t> q, r;
    divmod_mag(a.mag_, b.mag_, &q, &r);
    return make(std::move(r), a.neg_);
  }

  friend bool operator==(const BigInt& a, const BigInt& b) {
    return a.neg_ == b.neg_ && a.mag_ == b.mag_;
  }
  friend bool operator!=(const BigInt& a, const BigInt& b) { return !(a == b); }
  friend bool operator<(const BigInt& a, const BigInt& b) {
    if (a.neg_ != b.neg_) return a.neg_;
    const int c = cmp_mag(a.mag_, b.mag_);
    return a.neg_ ? c > 0 : c < 0;
  }
  friend bool operator>(const BigInt& a, const BigInt& b) { return b < a; }
  friend bool operator<=(const BigInt& a, const BigInt& b) { return !(b < a); }
  friend bool operator>=(const BigInt& a, const BigInt& b) { return !(a < b); }

  /// Greatest common divisor of the magnitudes (Euclid); gcd(0, 0) is 0.
  [[nodiscard]] static BigInt gcd(BigInt a, BigInt b) {
    a.neg_ = false;
    b.neg_ = false;
    while (!b.is_zero()) {
      BigInt t = a % b;
      a = std::move(b);
      b = std::move(t);
    }
    return a;
  }

  /// The value as top * 2^exponent, `top` holding the leading (at most) 64 bits, so a ratio
  /// of two huge numbers can be formed without overflowing a double.
  void leading_bits(std::uint64_t* top, int* exponent) const {
    *top = 0;
    *exponent = 0;
    if (mag_.empty()) return;
    const int bits = bit_length();
    const int shift = std::max(0, bits - 64);
    for (int b = bits - 1; b >= shift; --b) {
      *top = (*top << 1) | bit(b);
    }
    *exponent = shift;
  }

  [[nodiscard]] std::string to_string() const {
    if (mag_.empty()) return "0";
    std::vector<std::uint32_t> rest = mag_;
    std::string digits;
    constexpr std::uint64_t kChunk = 1000000000;  // nine decimal digits per step
    while (!rest.empty()) {
      std::uint64_t remainder = 0;
      for (std::size_t i = rest.size(); i-- > 0;) {
        const std::uint64_t cur = (remainder << 32) | rest[i];
        rest[i] = static_cast<std::uint32_t>(cur / kChunk);
        remainder = cur % kChunk;
      }
      trim(&rest);
      for (int k = 0; k < 9; ++k) {
        digits.push_back(static_cast<char>('0' + remainder % 10));
        remainder /= 10;
        if (rest.empty() && remainder == 0) break;
      }
    }
    if (neg_) digits.push_back('-');
    return {digits.rbegin(), digits.rend()};
  }

 private:
  using Limbs = std::vector<std::uint32_t>;

  static BigInt make(Limbs mag, bool negative) {
    BigInt r;
    trim(&mag);
    r.mag_ = std::move(mag);
    r.neg_ = negative && !r.mag_.empty();
    return r;
  }
  static void trim(Limbs* v) {
    while (!v->empty() && v->back() == 0) v->pop_back();
  }
  [[nodiscard]] int bit_length() const {
    if (mag_.empty()) return 0;
    int top = 0;
    for (std::uint32_t v = mag_.back(); v != 0; v >>= 1) ++top;
    return static_cast<int>(32 * (mag_.size() - 1)) + top;
  }
  [[nodiscard]] std::uint64_t bit(int b) const {
    return (mag_[static_cast<std::size_t>(b / 32)] >> (b % 32)) & 1U;
  }

  static int cmp_mag(const Limbs& a, const Limbs& b) {
    if (a.size() != b.size()) return a.size() < b.size() ? -1 : 1;
    for (std::size_t i = a.size(); i-- > 0;) {
      if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    }
    return 0;
  }
  static Limbs add_mag(const Limbs& a, const Limbs& b) {
    const Limbs& big = a.size() >= b.size() ? a : b;
    const Limbs& small = a.size() >= b.size() ? b : a;
    Limbs r(big.size() + 1, 0);
    std::uint64_t carry = 0;
    for (std::size_t i = 0; i < big.size(); ++i) {
      const std::uint64_t s = carry + big[i] + (i < small.size() ? small[i] : 0U);
      r[i] = static_cast<std::uint32_t>(s);
      carry = s >> 32;
    }
    r[big.size()] = static_cast<std::uint32_t>(carry);
    return r;
  }
  /// |a| - |b| for |a| >= |b|.
  static Limbs sub_mag(const Limbs& a, const Limbs& b) {
    Limbs r(a.size(), 0);
    std::int64_t borrow = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      std::int64_t d = static_cast<std::int64_t>(a[i]) - borrow -
                       static_cast<std::int64_t>(i < b.size() ? b[i] : 0U);
      borrow = d < 0 ? 1 : 0;
      if (d < 0) d += std::int64_t{1} << 32;
      r[i] = static_cast<std::uint32_t>(d);
    }
    return r;
  }
  static Limbs mul_mag(const Limbs& a, const Limbs& b) {
    if (a.empty() || b.empty()) return {};
    Limbs r(a.size() + b.size(), 0);
    for (std::size_t i = 0; i < a.size(); ++i) {
      std::uint64_t carry = 0;
      for (std::size_t j = 0; j < b.size(); ++j) {
        const std::uint64_t t = static_cast<std::uint64_t>(a[i]) * b[j] + r[i + j] + carry;
        r[i + j] = static_cast<std::uint32_t>(t);
        carry = t >> 32;
      }
      r[i + b.size()] = static_cast<std::uint32_t>(carry);
    }
    return r;
  }

  /// Knuth's Algorithm D on magnitudes: u = q v + r, 0 <= r < v. `v` nonzero.
  static void divmod_mag(const Limbs& u, const Limbs& v, Limbs* q, Limbs* r) {
    q->clear();
    r->clear();
    if (cmp_mag(u, v) < 0) {
      *r = u;
      return;
    }
    const std::size_t n = v.size();
    const std::size_t m = u.size();
    constexpr std::uint64_t kBase = std::uint64_t{1} << 32;
    if (n == 1) {
      q->assign(m, 0);
      std::uint64_t rem = 0;
      for (std::size_t i = m; i-- > 0;) {
        const std::uint64_t cur = (rem << 32) | u[i];
        (*q)[i] = static_cast<std::uint32_t>(cur / v[0]);
        rem = cur % v[0];
      }
      trim(q);
      if (rem != 0) r->push_back(static_cast<std::uint32_t>(rem));
      return;
    }
    // D1: normalise so the divisor's top limb has its high bit set.
    int s = 0;
    for (std::uint32_t top = v[n - 1]; (top & 0x80000000U) == 0; top <<= 1) ++s;
    Limbs vn(n), un(m + 1);
    for (std::size_t i = n - 1; i > 0; --i) {
      vn[i] = static_cast<std::uint32_t>((static_cast<std::uint64_t>(v[i]) << s) |
                                         (static_cast<std::uint64_t>(v[i - 1]) >> (32 - s)));
    }
    vn[0] = static_cast<std::uint32_t>(static_cast<std::uint64_t>(v[0]) << s);
    un[m] = static_cast<std::uint32_t>(static_cast<std::uint64_t>(u[m - 1]) >> (32 - s));
    for (std::size_t i = m - 1; i > 0; --i) {
      un[i] = static_cast<std::uint32_t>((static_cast<std::uint64_t>(u[i]) << s) |
                                         (static_cast<std::uint64_t>(u[i - 1]) >> (32 - s)));
    }
    un[0] = static_cast<std::uint32_t>(static_cast<std::uint64_t>(u[0]) << s);
    q->assign(m - n + 1, 0);
    for (std::size_t j = m - n + 1; j-- > 0;) {
      // D3: estimate the quotient limb from the top two limbs, correct it at most twice.
      const std::uint64_t num = (static_cast<std::uint64_t>(un[j + n]) << 32) | un[j + n - 1];
      std::uint64_t qhat = num / vn[n - 1];
      std::uint64_t rhat = num % vn[n - 1];
      while (qhat >= kBase || qhat * vn[n - 2] > ((rhat << 32) | un[j + n - 2])) {
        --qhat;
        rhat += vn[n - 1];
        if (rhat >= kBase) break;
      }
      // D4: multiply and subtract.
      std::int64_t k = 0;
      std::int64_t t = 0;
      for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t p = qhat * vn[i];
        t = static_cast<std::int64_t>(un[i + j]) - k -
            static_cast<std::int64_t>(p & 0xFFFFFFFFULL);
        un[i + j] = static_cast<std::uint32_t>(t);
        k = static_cast<std::int64_t>(p >> 32) - (t >> 32);
      }
      t = static_cast<std::int64_t>(un[j + n]) - k;
      un[j + n] = static_cast<std::uint32_t>(t);
      (*q)[j] = static_cast<std::uint32_t>(qhat);
      if (t < 0) {
        // D6: the estimate was one too large; add the divisor back.
        (*q)[j] -= 1;
        std::uint64_t carry = 0;
        for (std::size_t i = 0; i < n; ++i) {
          const std::uint64_t sum = static_cast<std::uint64_t>(un[i + j]) + vn[i] + carry;
          un[i + j] = static_cast<std::uint32_t>(sum);
          carry = sum >> 32;
        }
        un[j + n] = static_cast<std::uint32_t>(un[j + n] + carry);
      }
    }
    trim(q);
    // D8: the remainder, unnormalised.
    r->assign(n, 0);
    for (std::size_t i = 0; i + 1 < n; ++i) {
      (*r)[i] = static_cast<std::uint32_t>((static_cast<std::uint64_t>(un[i]) >> s) |
                                           (static_cast<std::uint64_t>(un[i + 1]) << (32 - s)));
    }
    (*r)[n - 1] = static_cast<std::uint32_t>(static_cast<std::uint64_t>(un[n - 1]) >> s);
    trim(r);
  }

  Limbs mag_;
  bool neg_ = false;
};

}  // namespace sankhya::exact
