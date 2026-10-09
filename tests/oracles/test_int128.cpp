// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the oracle's portable Int128 (#748). Where the compiler has __int128 every
// operation is compared with it on random and edge operands; everywhere, fixed cases pin
// the values the oracle depends on.
#include "oracles/int128.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

namespace sankhya::oracle {
namespace {

TEST(OracleInt128, FixedValues) {
  const Int128 two_to_100 = Int128(1) << 100;
  EXPECT_EQ(static_cast<double>(two_to_100), 1267650600228229401496703205376.0);
  EXPECT_EQ(static_cast<long long>(Int128(-7) / Int128(2)), -3);
  EXPECT_EQ(static_cast<long long>(Int128(-7) % Int128(2)), -1);
  EXPECT_EQ(static_cast<long long>(Int128(7) % Int128(-2)), 1);
  EXPECT_EQ(static_cast<long long>((two_to_100 * Int128(3) + Int128(5)) % two_to_100), 5);
  EXPECT_EQ((two_to_100 * Int128(3)) / two_to_100, Int128(3));
  EXPECT_LT(Int128(-1), Int128(0));
  EXPECT_LT(-two_to_100, Int128(-1));
  EXPECT_GT(two_to_100, Int128(0x7FFFFFFFFFFFFFFFLL));

  const Int128 min = Int128(1) << 127;
  const Int128 max = min - Int128(1);
  Int128 r;
  EXPECT_TRUE(Int128::add_overflow(max, Int128(1), &r));
  EXPECT_FALSE(Int128::add_overflow(min, max, &r));
  EXPECT_EQ(r, Int128(-1));
  EXPECT_TRUE(Int128::add_overflow(min, Int128(-1), &r));
  EXPECT_FALSE(Int128::mul_overflow(Int128(1) << 63, Int128(-1) << 63, &r));  // -2^126
  EXPECT_FALSE(Int128::mul_overflow(Int128(1) << 64, Int128(-1) << 63, &r));  // -2^127 fits
  EXPECT_EQ(r, min);
  EXPECT_TRUE(Int128::mul_overflow(Int128(1) << 64, Int128(1) << 63, &r));  // 2^127 does not
  EXPECT_TRUE(Int128::mul_overflow(min, Int128(-1), &r));
  EXPECT_TRUE(Int128::mul_overflow(Int128(1) << 64, Int128(1) << 64, &r));
}

#if defined(__SIZEOF_INT128__)
__extension__ using Wide = __int128;

Wide wide(const Int128& v) {
  return static_cast<Wide>((static_cast<unsigned __int128>(v.high_word()) << 64) |
                           v.low_word());
}

Int128 narrow(Wide v) {
  const auto hi = static_cast<long long>(v >> 64);
  const auto lo = static_cast<unsigned long long>(v);
  return (Int128(hi) << 64) + (Int128(static_cast<long long>(lo >> 32)) << 32) +
         Int128(static_cast<long long>(lo & 0xFFFFFFFFULL));
}

TEST(OracleInt128, AgreesWithInt128) {
  std::mt19937_64 rng(748);
  std::vector<Wide> values = {0,
                              1,
                              -1,
                              2,
                              -2,
                              static_cast<Wide>(1) << 64,
                              -(static_cast<Wide>(1) << 64),
                              static_cast<Wide>(1) << 126,
                              static_cast<Wide>(static_cast<unsigned __int128>(1) << 127),
                              static_cast<Wide>(~(static_cast<unsigned __int128>(1) << 127))};
  for (int i = 0; i < 400; ++i) {
    const auto raw = static_cast<Wide>((static_cast<unsigned __int128>(rng()) << 64) | rng());
    values.push_back(raw >> (rng() % 127));
  }
  for (const Wide a : values) {
    const Int128 x = narrow(a);
    ASSERT_EQ(wide(x), a);
    ASSERT_EQ(static_cast<double>(x), static_cast<double>(a));
    ASSERT_EQ(static_cast<long long>(x), static_cast<long long>(a));
    for (int s : {0, 1, 31, 63, 64, 65, 100, 127}) {
      ASSERT_EQ(wide(x << s), static_cast<Wide>(static_cast<unsigned __int128>(a) << s));
    }
    for (std::size_t j = 0; j < values.size(); j += 3) {
      const Wide b = values[j];
      const Int128 y = narrow(b);
      ASSERT_EQ(x < y, a < b);
      ASSERT_EQ(x == y, a == b);
      ASSERT_EQ(wide(x + y), static_cast<Wide>(static_cast<unsigned __int128>(a) +
                                               static_cast<unsigned __int128>(b)));
      ASSERT_EQ(wide(x * y), static_cast<Wide>(static_cast<unsigned __int128>(a) *
                                               static_cast<unsigned __int128>(b)));
      Wide expected = 0;
      Int128 got;
      ASSERT_EQ(Int128::add_overflow(x, y, &got), __builtin_add_overflow(a, b, &expected));
      ASSERT_EQ(wide(got), expected);
      ASSERT_EQ(Int128::mul_overflow(x, y, &got), __builtin_mul_overflow(a, b, &expected));
      ASSERT_EQ(wide(got), expected);
      const Wide min = static_cast<Wide>(static_cast<unsigned __int128>(1) << 127);
      if (b != 0 && !(a == min && b == -1)) {
        ASSERT_EQ(wide(x / y), a / b);
        ASSERT_EQ(wide(x % y), a % b);
      }
    }
  }
}
#endif

}  // namespace
}  // namespace sankhya::oracle
