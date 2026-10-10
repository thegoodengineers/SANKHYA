// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the portable 128-bit product and carry (#748): every compiler's path agrees
// with the 32-bit schoolbook one, and with unsigned __int128 where the compiler has it.
#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

#include "util/wide_mul.hpp"

namespace {

using sankhya::U128;

std::vector<std::uint64_t> operands() {
  std::vector<std::uint64_t> v = {0,
                                  1,
                                  2,
                                  0xFFFFFFFFULL,
                                  0x100000000ULL,
                                  0x7FFFFFFFFFFFFFFFULL,
                                  0x8000000000000000ULL,
                                  0xFFFFFFFFFFFFFFFFULL,
                                  0xFFFFFFFF00000000ULL};
  std::mt19937_64 rng(748);
  for (int i = 0; i < 2000; ++i) v.push_back(rng() >> (rng() % 64));
  return v;
}

TEST(WideMul, MatchesSchoolbookAndInt128) {
  const std::vector<std::uint64_t> v = operands();
  for (std::size_t i = 0; i < v.size(); i += 7) {
    for (const std::uint64_t b : v) {
      const std::uint64_t a = v[i];
      const U128 fast = sankhya::mul_64x64(a, b);
      const U128 slow = sankhya::mul_64x64_portable(a, b);
      ASSERT_EQ(fast.lo, slow.lo) << a << " * " << b;
      ASSERT_EQ(fast.hi, slow.hi) << a << " * " << b;
#if defined(__SIZEOF_INT128__)
      __extension__ using Wide = unsigned __int128;
      const Wide p = static_cast<Wide>(a) * b;
      ASSERT_EQ(slow.lo, static_cast<std::uint64_t>(p));
      ASSERT_EQ(slow.hi, static_cast<std::uint64_t>(p >> 64));
#endif
    }
  }
  const U128 top = sankhya::mul_64x64(~0ULL, ~0ULL);  // (2^64 - 1)^2 = 2^128 - 2^65 + 1
  EXPECT_EQ(top.lo, 1ULL);
  EXPECT_EQ(top.hi, 0xFFFFFFFFFFFFFFFEULL);
}

TEST(WideMul, AddCarry) {
  std::uint64_t s = 0;
  EXPECT_EQ(sankhya::add_carry(~0ULL, 0, 1, &s), 1U);
  EXPECT_EQ(s, 0ULL);
  EXPECT_EQ(sankhya::add_carry(~0ULL, ~0ULL, 1, &s), 1U);
  EXPECT_EQ(s, ~0ULL);
  EXPECT_EQ(sankhya::add_carry(5, 7, 0, &s), 0U);
  EXPECT_EQ(s, 12ULL);
  EXPECT_EQ(sankhya::add_carry(~0ULL, 1, 0, &s), 1U);
  EXPECT_EQ(s, 0ULL);
  const std::vector<std::uint64_t> v = operands();
  for (std::size_t i = 0; i + 1 < v.size(); ++i) {
    for (unsigned c = 0; c < 2; ++c) {
      const unsigned out = sankhya::add_carry(v[i], v[i + 1], c, &s);
      // Recompute through 32-bit halves.
      const std::uint64_t low = (v[i] & 0xFFFFFFFFULL) + (v[i + 1] & 0xFFFFFFFFULL) + c;
      const std::uint64_t high = (v[i] >> 32) + (v[i + 1] >> 32) + (low >> 32);
      ASSERT_EQ(s, (high << 32) | (low & 0xFFFFFFFFULL));
      ASSERT_EQ(out, static_cast<unsigned>(high >> 32));
    }
  }
}

}  // namespace
