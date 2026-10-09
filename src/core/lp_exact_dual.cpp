// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the safe bound from the EXACT duals of the reported basis (#763).
//
// THE SYSTEM. B'y = c_B, B the reported basis: the basic structural columns and, for a basic
// row, its logical column -e_i (so that row's multiplier is exactly zero). Every entry is a
// double, so every entry is a dyadic rational m * 2^e, and so is the solution's numerator
// over det(B) - but det(B) runs to thousands of bits on Netlib (greenbea: 15,457).
//
// THE SOLVE: numeric-symbolic iterative refinement (Wan, "An algorithm to solve integer
// linear systems exactly using numerical methods", J. Symbolic Computation 41, 2006; the
// same scheme for LP in Gleixner, Steffy and Wolter, "Iterative refinement for linear
// programming", INFORMS J. Computing 28(3), 2016). The residual r = c_B - B'x is kept
// EXACTLY, as dyadic rationals; each step solves B'z = r in floating point with the LU the
// simplex also uses, keeps kExactDualStepBits leading bits of z as an exact dyadic
// correction, adds it to x and subtracts B' times it from r, exactly. Each step removes
// about as many bits of the residual as the basis's conditioning allows, and nothing in it
// is ever rounded: x is an exact dyadic number whose distance from y is |B'^-1 r|.
//
// THE RATIONALS: continued fractions. Once x is within 2^-2k of y, every y_i = p_i / q with
// q < 2^k is a convergent of x_i (Legendre; Hardy and Wright, "An Introduction to the
// Theory of Numbers", thm. 184). The common denominator is read off ONE continued fraction,
// that of a random small-integer combination of the components, whose denominator is the
// common one except with a probability of about one over its smallest prime factor
// (Monagan, "Maximal quotient rational reconstruction", ISSAC 2004); a component the
// result leaves away from an integer supplies a small missing factor by its own continued
// fraction. The numerators are the nearest integers to x_i q, and the candidate is then
// checked EXACTLY against B'y = c_B: a wrong guess can only fail the check and ask for more
// bits, never produce a wrong y. The whole computation is quadratic in the size of the
// numbers, with the ones that grow (the residual updates and the roundings) kept to shifts
// and fixed-width products.
//
// THE BOUND: Neumaier and Shcherbina (Math. Programming 99, 2004), evaluated in exact
// arithmetic with y = p / q: every basic reduced cost is exactly zero and needs no bound;
// a nonbasic one needs the bound its sign asks for, and the basis must therefore be exactly
// dual feasible over the box, or no finite bound is proved. The exact value is rounded down
// to a double at the end, checked exactly.

#include "core/lp_exact_dual.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "exact/bigint.hpp"
#include "exact/rational.hpp"
#include "la/lu.hpp"
#include "sankhya/tolerances.hpp"
#include "sankhya/types.hpp"
#include "util/wide_mul.hpp"

namespace sankhya {
namespace {

using exact::BigInt;
using exact::Rational;
using Sz = std::size_t;
constexpr int kZero = INT_MIN;  // top() of zero

/// m * 2^e, exactly.
struct Dyadic {
  BigInt m;
  int e = 0;
};

Dyadic dyadic(double v) {
  if (v == 0.0) return {};
  int exponent = 0;
  const double mantissa = std::frexp(v, &exponent);
  return {BigInt(static_cast<long long>(std::ldexp(mantissa, 53))), exponent - 53};
}
Dyadic operator*(const Dyadic& a, const Dyadic& b) {
  if (a.m.is_zero() || b.m.is_zero()) return {};
  return {a.m * b.m, a.e + b.e};
}
Dyadic operator+(const Dyadic& a, const Dyadic& b) {
  if (a.m.is_zero()) return b;
  if (b.m.is_zero()) return a;
  if (a.e == b.e) return {a.m + b.m, a.e};
  if (a.e > b.e) return {a.m.shifted_left(a.e - b.e) + b.m, b.e};
  return {a.m + b.m.shifted_left(b.e - a.e), a.e};
}
Dyadic operator-(const Dyadic& a, const Dyadic& b) {
  return a + Dyadic{-b.m, b.e};
}
/// |value| < 2^top(value); kZero for zero.
int top(const Dyadic& a) {
  return a.m.is_zero() ? kZero : a.m.bits() + a.e;
}
/// value / 2^shift, to double precision or so.
double scaled(const Dyadic& a, int shift) {
  if (a.m.is_zero()) return 0.0;
  std::uint64_t lead = 0;
  int ex = 0;
  a.m.leading_bits(&lead, &ex);
  const double magnitude = std::ldexp(static_cast<double>(lead), ex + a.e - shift);
  return a.m.sign() < 0 ? -magnitude : magnitude;
}
/// a with every bit below 2^position dropped (towards zero): an error under 2^position.
Dyadic truncated_below(const Dyadic& a, int position) {
  if (a.m.is_zero() || a.e >= position) return a;
  return {a.m.shifted_right(position - a.e), position};
}
/// The integer nearest to a (halves away from zero).
BigInt nearest(const Dyadic& a) {
  if (a.e >= 0) return a.m.shifted_left(a.e);
  // (|m| + 2^(-e-1)) >> -e: a shift, not a division, since the divisor is a power of two.
  const BigInt q = (a.m.abs() + BigInt(1).shifted_left(-a.e - 1)).shifted_right(-a.e);
  return a.m.sign() < 0 ? -q : q;
}

/// The last convergent h/k of the continued fraction of t with k < 2^k_bits.
std::optional<std::pair<BigInt, BigInt>> convergent(const Dyadic& t, int k_bits) {
  BigInt n = t.e >= 0 ? t.m.shifted_left(t.e) : t.m;
  BigInt d = t.e >= 0 ? BigInt(1) : BigInt(1).shifted_left(-t.e);
  const bool negative = n.sign() < 0;
  if (negative) n = -n;
  BigInt h2(0), h1(1), k2(1), k1(0);
  while (!d.is_zero()) {
    const BigInt a = n / d;
    BigInt h = a * h1 + h2;
    BigInt k = a * k1 + k2;
    if (k.bits() > k_bits) break;
    h2 = std::move(h1);
    h1 = std::move(h);
    k2 = std::move(k1);
    k1 = std::move(k);
    BigInt rest = n - a * d;
    n = std::move(d);
    d = std::move(rest);
  }
  if (k1.is_zero()) return std::nullopt;
  return std::make_pair(negative ? -h1 : h1, k1);
}

struct RationalVector {
  std::vector<BigInt> p;
  BigInt q;
};

/// Rationals p_i / q within 2^error_exponent of x_i (see the file header): the denominator
/// from ONE continued fraction, of a random small-integer combination of the components,
/// whose lowest-terms denominator is the common one except with a probability of about
/// 1 / (the smallest prime factor of it) per attempt (Monagan, "Maximal quotient rational
/// reconstruction", ISSAC 2004, uses the same combination); the numerators are then the
/// nearest integers to x_i q. A wrong guess fails the caller's exact check, and the next
/// attempt draws new weights. nullopt when x is not yet precise enough for any denominator.
std::optional<RationalVector> reconstruct(const std::vector<Dyadic>& x, int error_exponent,
                                          std::mt19937_64& rng) {
  constexpr int kWeightBits = 16;
  constexpr int kSlack = 16;  // bits kept beyond what a rounding needs
  std::uniform_int_distribution<long long> weight(1, (1LL << kWeightBits) - 1);
  // |t - T| < sum(weights) 2^error_exponent < 2^(error_exponent + kWeightBits + bits(m));
  // the bits of t below that carry nothing and are dropped before the continued fraction.
  int size_bits = 0;
  for (Sz n = x.size(); n != 0; n >>= 1) ++size_bits;
  const int error_bits = error_exponent + kWeightBits + size_bits + 1;
  const int q_bits = (-error_bits - 1) / 2;
  if (q_bits < 1) return std::nullopt;
  Dyadic t;
  for (const Dyadic& xi : x) {
    if (xi.m.is_zero()) continue;
    t = t + truncated_below(xi, error_bits - kSlack) * Dyadic{BigInt(weight(rng)), 0};
  }
  const auto c = convergent(t, q_bits);
  if (!c) return std::nullopt;
  BigInt q = c->second;
  // The combination misses a factor of the common denominator with a probability of about
  // one over that factor: a component that x_i q leaves away from an integer supplies it,
  // by its own continued fraction, in the room the precision still leaves.
  const auto scaled = [&](const Dyadic& xi, const BigInt& denominator) {
    return truncated_below(xi, -(denominator.bits() + kSlack)) * Dyadic{denominator, 0};
  };
  // The missing factor is small (a power of two, mostly), so the search for it is kept to
  // kFixupBits of denominator, at the precision a convergent of that size needs.
  constexpr int kFixupBits = 64;
  constexpr int kMaxFixups = 4;  // more than that and the denominator was never right
  int fixups = 0;
  for (const Dyadic& xi : x) {
    if (xi.m.is_zero()) continue;
    const Dyadic t_i = scaled(xi, q);
    const Dyadic away = t_i - Dyadic{nearest(t_i), 0};
    if (away.m.is_zero() || top(away) < -kSlack / 2) continue;
    if (++fixups > kMaxFixups) return std::nullopt;
    const int room = std::min(kFixupBits, q_bits - q.bits());
    if (room < 1) return std::nullopt;
    const Dyadic precise = truncated_below(xi, -(q.bits() + 2 * room + kSlack)) * Dyadic{q, 0};
    const auto extra = convergent(precise, room);
    if (!extra) return std::nullopt;
    q = q * extra->second;
  }
  RationalVector out;
  out.p.reserve(x.size());
  for (const Dyadic& xi : x) out.p.push_back(nearest(scaled(xi, q)));
  out.q = std::move(q);
  return out;
}

}  // namespace

ExactDualBound exact_dual_bound(const Model& model, std::span<const double> dual_cost,
                                std::span<const double> cost, std::span<const double> lower,
                                std::span<const double> upper, const Solution& basis,
                                double seconds) {
  ExactDualBound out;
  const Index n = model.num_cols();
  const Index m = model.num_rows();
  const auto sm = static_cast<Sz>(m);
  if (basis.col_status.size() != static_cast<Sz>(n) || basis.row_status.size() != sm) {
    out.message = "no basis";
    return out;
  }
  // Position p of the basis: a structural column j >= 0, or row i's logical as -1 - i.
  std::vector<Index> variable;
  std::vector<LuColumn> columns;
  std::vector<Index> logical_row;
  logical_row.reserve(sm);  // never reallocates: the columns point into it
  static const double kMinusOne = -1.0;
  for (Index j = 0; j < n; ++j) {
    if (basis.col_status[static_cast<Sz>(j)] != BasisStatus::kBasic) continue;
    const ColumnView col = model.matrix.column(j);
    columns.push_back({col.rows, col.values, col.size});
    variable.push_back(j);
  }
  for (Index i = 0; i < m; ++i) {
    if (basis.row_status[static_cast<Sz>(i)] != BasisStatus::kBasic) continue;
    logical_row.push_back(i);
    columns.push_back({&logical_row.back(), &kMinusOne, 1});
    variable.push_back(-1 - i);
  }
  if (columns.size() != sm) {
    out.message =
        fmt::format("the basis has {} basic variables for {} rows", columns.size(), m);
    return out;
  }
  SparseLu lu;
  if (!lu.factorize(columns, m, tol::kPivotTolerance, tol::kMarkowitzThreshold)) {
    out.message = "the basis is singular to working precision";
    return out;
  }
  const auto rhs = [&](Sz p) {
    return variable[p] >= 0 ? dyadic(dual_cost[static_cast<Sz>(variable[p])]) : Dyadic{};
  };
  // The structural basis columns' entries as mantissa * 2^exponent, once, for the residual
  // update below; a column takes the 256-bit path when its exponents span at most
  // kAccumulatorSpread bits (its products, under 2^105 each, then sit under 2^225).
  constexpr int kAccumulatorSpread = 120;
  std::vector<long long> entry_mantissa;
  std::vector<int> entry_shift;  // the entry's exponent above the column's smallest
  std::vector<Sz> column_start(sm + 1, 0);
  std::vector<int> column_exponent(sm, 0);
  std::vector<char> column_fits(sm, 0);
  for (Sz p = 0; p < sm; ++p) {
    column_start[p] = entry_mantissa.size();
    if (variable[p] < 0) continue;
    const ColumnView col = model.matrix.column(variable[p]);
    int lowest = INT_MAX;
    int highest = INT_MIN;
    for (Index k = 0; k < col.size; ++k) {
      int exponent = 0;
      const double normalised = std::frexp(col.values[k], &exponent);
      const auto mantissa = static_cast<long long>(std::ldexp(normalised, 53));
      exponent -= 53;
      entry_mantissa.push_back(mantissa);
      entry_shift.push_back(exponent);
      if (mantissa != 0) {
        lowest = std::min(lowest, exponent);
        highest = std::max(highest, exponent);
      }
    }
    if (lowest == INT_MAX) lowest = highest = 0;
    column_exponent[p] = lowest;
    column_fits[p] = highest - lowest <= kAccumulatorSpread ? 1 : 0;
    for (Sz k = column_start[p]; k < entry_mantissa.size(); ++k) entry_shift[k] -= lowest;
  }
  column_start[sm] = entry_mantissa.size();

  // column_p . v, exactly.
  const auto apply = [&](Sz p, const std::vector<Dyadic>& v) {
    const Index var = variable[p];
    if (var < 0) {
      const Dyadic& vi = v[static_cast<Sz>(-1 - var)];
      return Dyadic{-vi.m, vi.e};
    }
    Dyadic sum;
    const ColumnView col = model.matrix.column(var);
    for (Index k = 0; k < col.size; ++k) {
      const Dyadic& vi = v[static_cast<Sz>(col.rows[k])];
      if (!vi.m.is_zero()) sum = sum + dyadic(col.values[k]) * vi;
    }
    return sum;
  };

  exact::Deadline deadline(seconds);
  try {
    std::vector<Dyadic> x(sm);
    std::vector<Dyadic> r(sm);
    for (Sz p = 0; p < sm; ++p) r[p] = rhs(p);
    std::vector<double> work(sm);
    std::optional<RationalVector> y;
    int gained = 0;
    int next_attempt = tol::kExactDualFirstAttemptBits;
    std::mt19937_64 rng(763);  // fixed: the same run gives the same answer
    int stalls = 0;
    for (int iteration = 0; iteration < tol::kExactDualMaxIterations && !y; ++iteration) {
      deadline.check();
      int t = kZero;
      for (const Dyadic& rp : r) t = std::max(t, top(rp));
      int error_exponent = kZero;
      if (t != kZero) {
        for (Sz p = 0; p < sm; ++p) work[p] = scaled(r[p], t);
        lu.solve_transpose(work.data());
        double z_max = 0.0;
        for (const double v : work) z_max = std::max(z_max, std::fabs(v));
        if (!(z_max > 0.0) || !std::isfinite(z_max)) {
          out.message = "the floating-point solve broke down";
          return out;
        }
        int ez = 0;
        (void)std::frexp(z_max, &ez);
        std::vector<Dyadic> dx(sm);
        std::vector<long long> steps(sm, 0);
        const int step_exponent = ez - tol::kExactDualStepBits + t;
        for (Sz i = 0; i < sm; ++i) {
          const long long step =
              std::llround(std::ldexp(work[i], tol::kExactDualStepBits - ez));
          if (step == 0) continue;
          steps[i] = step;
          dx[i] = Dyadic{BigInt(step), step_exponent};
          x[i] = x[i] + dx[i];
        }
        // r -= B' dx, exactly. Every dx_i is step_i 2^step_exponent, so a structural
        // column's sum is sum_k mantissa_k step_i 2^shift_k times 2^(exponent + step
        // exponent): products under 2^105 shifted by at most kAccumulatorSpread, summed in
        // 256 bits - one BigInt per column instead of three per entry.
        for (Sz p = 0; p < sm; ++p) {
          if (variable[p] < 0) {
            const Dyadic& vi = dx[static_cast<Sz>(-1 - variable[p])];
            if (!vi.m.is_zero()) r[p] = r[p] + vi;
            continue;
          }
          if (column_fits[p] != 0) {
            std::uint64_t magnitude[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
            const ColumnView col = model.matrix.column(variable[p]);
            for (Sz k = column_start[p], e = 0; k < column_start[p + 1]; ++k, ++e) {
              const long long step = steps[static_cast<Sz>(col.rows[e])];
              const long long mantissa = entry_mantissa[k];
              if (step == 0 || mantissa == 0) continue;
              const U128 product =
                  mul_64x64(static_cast<std::uint64_t>(mantissa < 0 ? -mantissa : mantissa),
                            static_cast<std::uint64_t>(step < 0 ? -step : step));
              std::uint64_t* acc = magnitude[(mantissa < 0) != (step < 0) ? 1 : 0];
              const int shift = entry_shift[k];
              const int word = shift / 64;
              const int bits = shift % 64;
              std::uint64_t words[3];
              if (bits == 0) {
                words[0] = product.lo;
                words[1] = product.hi;
                words[2] = 0;
              } else {
                words[0] = product.lo << bits;
                words[1] = (product.lo >> (64 - bits)) | (product.hi << bits);
                words[2] = product.hi >> (64 - bits);
              }
              unsigned carry = 0;
              for (int w = word; w < 4; ++w) {
                carry = add_carry(acc[w], w - word < 3 ? words[w - word] : 0, carry, &acc[w]);
              }
            }
            // positive - negative, as a signed 256-bit magnitude.
            int cmp = 0;
            for (int w = 3; w >= 0 && cmp == 0; --w) {
              cmp = magnitude[0][w] == magnitude[1][w]  ? 0
                    : magnitude[0][w] > magnitude[1][w] ? 1
                                                        : -1;
            }
            if (cmp == 0) continue;
            const std::uint64_t* big = magnitude[cmp > 0 ? 0 : 1];
            const std::uint64_t* small = magnitude[cmp > 0 ? 1 : 0];
            std::uint64_t difference[4];
            std::uint64_t borrow = 0;
            for (int w = 0; w < 4; ++w) {
              const std::uint64_t subtrahend = small[w] + borrow;
              borrow =
                  (small[w] == ~std::uint64_t{0} && borrow != 0) || big[w] < subtrahend ? 1 : 0;
              difference[w] = big[w] - subtrahend;
            }
            r[p] = r[p] - Dyadic{BigInt::from_words(difference, 4, cmp < 0),
                                 column_exponent[p] + step_exponent};
            continue;
          }
          r[p] = r[p] - apply(p, dx);
        }
        int new_t = kZero;
        for (const Dyadic& rp : r) new_t = std::max(new_t, top(rp));
        const int progress = new_t == kZero ? tol::kExactDualStepBits : t - new_t;
        if (progress < tol::kExactDualMinProgress) {
          if (++stalls >= tol::kExactDualMaxStalls) {
            out.message = "the refinement stopped converging";
            return out;
          }
        } else {
          stalls = 0;
        }
        gained += std::max(progress, 0);
        // |x - y| = |B'^-1 r| <~ |z| 2^new_t, and |z| < 2^ez for a residual of size 2^t.
        error_exponent = new_t == kZero ? kZero : new_t + ez + tol::kExactDualErrorMargin;
      }
      if (error_exponent == kZero) {
        // x is exact: its common denominator is a power of two.
        int low = 0;
        for (const Dyadic& xi : x) {
          if (!xi.m.is_zero()) low = std::min(low, xi.e);
        }
        RationalVector exact_x;
        exact_x.q = BigInt(1).shifted_left(-low);
        for (const Dyadic& xi : x) exact_x.p.push_back(nearest(xi * Dyadic{exact_x.q, 0}));
        y = std::move(exact_x);
      } else if (gained >= next_attempt) {
        next_attempt = gained + gained / 4;
        y = reconstruct(x, error_exponent, rng);
      } else {
        continue;
      }
      if (!y) continue;
      // The exact check: column_p . p == c_p q for every position.
      std::vector<Dyadic> p_dyadic(sm);
      for (Sz i = 0; i < sm; ++i) p_dyadic[i] = Dyadic{y->p[i], 0};
      const Dyadic q_dyadic{y->q, 0};
      for (Sz p = 0; p < sm && y; ++p) {
        if (!(apply(p, p_dyadic) - rhs(p) * q_dyadic).m.is_zero()) y.reset();
      }
    }
    if (!y) {
      out.message = "the refinement did not reach the exact duals within its iteration cap";
      return out;
    }

    // ---- The bound, exactly, times q: sum_i p_i side_i + sum_j S_j bound_j, with
    // S_j = q c_j - a_j'p the reduced cost times q.
    const Dyadic q_dyadic{y->q, 0};
    std::vector<Dyadic> used(sm);
    Dyadic total;
    // A multiplier that prices a side the row has not got - a basis that is exactly dual
    // infeasible on that row by a rounding residue - is priced at the row's activity bound
    // over the box on that side, exactly; that bound holds for every feasible x, so the term
    // is valid. Only when the box leaves that unbounded too is the multiplier dropped, as
    // safe_dual_bound() does. tools/verify_solution.py applies the same rule.
    for (Sz i = 0; i < sm; ++i) {
      const BigInt& pi = y->p[i];
      if (pi.is_zero()) continue;
      const bool want_min = pi.sign() > 0;
      const double side = want_min ? model.row_lower[i] : model.row_upper[i];
      Dyadic side_value;
      if (is_finite_bound(side)) {
        side_value = dyadic(side);
      } else {
        deadline.check();
        bool bounded = true;
        for (Index j = 0; j < n && bounded; ++j) {
          const ColumnView col = model.matrix.column(j);
          for (Index k = 0; k < col.size; ++k) {
            if (static_cast<Sz>(col.rows[k]) != i) continue;
            const double v = col.values[k];
            const auto uj = static_cast<Sz>(j);
            const double at = (v > 0.0) == want_min ? lower[uj] : upper[uj];
            if (!is_finite_bound(at)) {
              bounded = false;
              break;
            }
            side_value = side_value + dyadic(v) * dyadic(at);
          }
        }
        if (!bounded) continue;  // priced at zero
      }
      used[i] = Dyadic{pi, 0};
      total = total + used[i] * side_value;
    }
    for (Index j = 0; j < n; ++j) {
      if ((j & 255) == 0) deadline.check();
      const auto uj = static_cast<Sz>(j);
      Dyadic s = dyadic(cost[uj]) * q_dyadic;
      const ColumnView col = model.matrix.column(j);
      for (Index k = 0; k < col.size; ++k) {
        const Dyadic& pi = used[static_cast<Sz>(col.rows[k])];
        if (!pi.m.is_zero()) s = s - dyadic(col.values[k]) * pi;
      }
      if (s.m.is_zero()) continue;
      const double bound = s.m.sign() > 0 ? lower[uj] : upper[uj];
      if (!is_finite_bound(bound)) {
        out.message = fmt::format("the exact basis is not dual feasible for column {}", j);
        return out;
      }
      total = total + s * dyadic(bound);
    }
    const Rational value = total.e >= 0 ? Rational(total.m.shifted_left(total.e), y->q)
                                        : Rational(total.m, y->q.shifted_left(-total.e));
    double rounded = value.to_double();
    if (!std::isfinite(rounded)) {
      out.message = "the exact bound is outside the range of a double";
      return out;
    }
    while (Rational::from_double(rounded) > value) {
      rounded = std::nextafter(rounded, -std::numeric_limits<double>::infinity());
    }
    const std::string denominator = y->q.to_string();
    out.multipliers.reserve(sm);
    for (const BigInt& pi : y->p) out.multipliers.push_back(pi.to_string() + "/" + denominator);
    out.value = rounded;
    out.proved = true;
  } catch (const exact::ExactBudgetExceeded&) {
    out.message = "the exact duals ran past their time budget";
    out.timed_out = true;
  } catch (const exact::RationalOverflow&) {
    out.message = "a non-finite value reached the exact arithmetic";
  }
  return out;
}

}  // namespace sankhya
