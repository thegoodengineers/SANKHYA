// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the 64 x 64 -> 128-bit product and the add-with-carry that the exact arithmetic
// needs, with the same bits on every compiler (#748).
//
// GCC and Clang have unsigned __int128; MSVC has no 128-bit integer but has the _umul128
// intrinsic on x64. Anywhere else the product is formed from 32-bit halves, the schoolbook
// method of Knuth, "The Art of Computer Programming", vol. 2, 3rd ed., section 4.3.1,
// Algorithm M with base 2^32. All three give the exact product, so the result is identical
// bit for bit; tests/unit/test_wide_mul.cpp checks them against each other.
#pragma once

#include <cstdint>

#if !defined(__SIZEOF_INT128__) && defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

namespace sankhya {

/// An unsigned 128-bit value as two 64-bit words.
struct U128 {
  std::uint64_t lo = 0;
  std::uint64_t hi = 0;
};

/// a * b from 32-bit halves; exact on any conforming compiler.
constexpr U128 mul_64x64_portable(std::uint64_t a, std::uint64_t b) noexcept {
  constexpr std::uint64_t kLow = 0xFFFFFFFFULL;
  const std::uint64_t p00 = (a & kLow) * (b & kLow);
  const std::uint64_t p01 = (a & kLow) * (b >> 32);
  const std::uint64_t p10 = (a >> 32) * (b & kLow);
  const std::uint64_t p11 = (a >> 32) * (b >> 32);
  // Three terms under 2^32 each: no overflow.
  const std::uint64_t mid = (p00 >> 32) + (p01 & kLow) + (p10 & kLow);
  return {(mid << 32) | (p00 & kLow), p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32)};
}

/// a * b, exactly.
inline U128 mul_64x64(std::uint64_t a, std::uint64_t b) noexcept {
#if defined(__SIZEOF_INT128__)
  __extension__ using Wide = unsigned __int128;
  const Wide p = static_cast<Wide>(a) * b;
  return {static_cast<std::uint64_t>(p), static_cast<std::uint64_t>(p >> 64)};
#elif defined(_MSC_VER) && defined(_M_X64)
  U128 r;
  r.lo = _umul128(a, b, &r.hi);
  return r;
#else
  return mul_64x64_portable(a, b);
#endif
}

/// *sum = a + b + carry (carry 0 or 1), modulo 2^64; returns the carry out. Plain C++ is
/// exact everywhere and every compiler here turns it into add/adc, so no intrinsic.
inline unsigned add_carry(std::uint64_t a, std::uint64_t b, unsigned carry,
                          std::uint64_t* sum) noexcept {
  const std::uint64_t s = a + b;
  *sum = s + carry;
  return static_cast<unsigned>(s < a) | static_cast<unsigned>(*sum < s);
}

}  // namespace sankhya
