// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the local rules of every expression operation: its value, its partial derivative
// with respect to each child, and that partial's derivative along a direction (NLP stage 1).
//
// ONE SOURCE OF TRUTH. ExpressionGraph (derivatives.cpp) evaluates over node ids and the
// compiled tapes the NLP solver runs on (tape.cpp) evaluate over local positions; both call
// these functions, so the two cannot disagree about what d sqrt(a)/da is. The rules are the
// elementary derivatives of Griewank and Walther, "Evaluating Derivatives", 2nd ed., SIAM
// 2008, ch. 2 and 3; the second-order ("adjoint-of-tangent") form is their ch. 5.
//
// A rule is a pure function of the node and its children's values; nothing here allocates.
#pragma once

#include <cmath>
#include <cstddef>
#include <string>

#include "nlp/expression.hpp"

namespace sankhya::nlp::detail {

inline bool is_integer_exponent(double value) {
  return std::isfinite(value) && std::floor(value) == value;
}

/// The node's value from its children's, `child[k]` for child k. False, with `why` set,
/// outside the operation's domain (the caller turns that into EvalError::kDomain); a result
/// that overflows is returned as it is and judged by the caller.
inline bool apply(const Node& n, const double* child, double* result, std::string* why) {
  switch (n.op) {
    case Op::kConstant: *result = n.value; return true;
    case Op::kVariable: return true;  // the caller supplies the column's value
    case Op::kSum: {
      double s = 0.0;
      for (std::size_t k = 0; k < n.children.size(); ++k) s += child[k];
      *result = s;
      return true;
    }
    case Op::kProduct: *result = child[0] * child[1]; return true;
    case Op::kNegate: *result = -child[0]; return true;
    case Op::kDivide:
      if (child[1] == 0.0) {
        *why = "division by zero";
        return false;
      }
      *result = child[0] / child[1];
      return true;
    case Op::kPower: {
      const double base = child[0];
      if (base < 0.0 && !is_integer_exponent(n.value)) {
        *why = "a negative number raised to a non-integer power";
        return false;
      }
      if (base == 0.0 && n.value < 0.0) {
        *why = "0 raised to a negative power";
        return false;
      }
      *result = std::pow(base, n.value);
      return true;
    }
    case Op::kExp: *result = std::exp(child[0]); return true;
    case Op::kLog:
      if (child[0] <= 0.0) {
        *why = "log of a non-positive number";
        return false;
      }
      *result = std::log(child[0]);
      return true;
    case Op::kSqrt:
      if (child[0] < 0.0) {
        *why = "sqrt of a negative number";
        return false;
      }
      *result = std::sqrt(child[0]);
      return true;
    case Op::kSin: *result = std::sin(child[0]); return true;
    case Op::kCos: *result = std::cos(child[0]); return true;
  }
  *why = "an unknown operation";
  return false;
}

/// d(node)/d(child k) into `d`, and - when `child_dot` is given - that partial's derivative
/// along the direction whose child tangents it holds, into `d_dot`. `self` is the node's own
/// value. False when the derivative does not exist here (sqrt at 0, x^0.5 at 0).
inline bool partial(const Node& n, std::size_t k, double self, const double* child,
                    const double* child_dot, double* d, double* d_dot) {
  const auto dot = [&](std::size_t c) { return child_dot == nullptr ? 0.0 : child_dot[c]; };
  double first = 0.0;
  double second_along = 0.0;
  switch (n.op) {
    case Op::kConstant:
    case Op::kVariable: return true;  // no children
    case Op::kSum: first = 1.0; break;
    case Op::kNegate: first = -1.0; break;
    case Op::kProduct:
      // d(ab)/da = b, whose derivative along the direction is b-dot; and symmetrically. For
      // x*x both children are x, and the two contributions sum to 2x - the rule needs no case.
      first = child[1 - k];
      second_along = dot(1 - k);
      break;
    case Op::kDivide: {
      const double a = child[0];
      const double b = child[1];
      if (k == 0) {
        first = 1.0 / b;
        second_along = -dot(1) / (b * b);
      } else {
        first = -a / (b * b);
        second_along = -dot(0) / (b * b) + 2.0 * a * dot(1) / (b * b * b);
      }
      break;
    }
    case Op::kPower: {
      const double a = child[0];
      const double p = n.value;
      first = p == 1.0 ? 1.0 : p * std::pow(a, p - 1.0);
      if (child_dot != nullptr && dot(0) != 0.0) {
        const double curvature = p == 2.0 ? 2.0 : p * (p - 1.0) * std::pow(a, p - 2.0);
        second_along = curvature * dot(0);
      }
      break;
    }
    case Op::kExp:
      first = self;
      second_along = self * dot(0);
      break;
    case Op::kLog:
      first = 1.0 / child[0];
      second_along = -dot(0) / (child[0] * child[0]);
      break;
    case Op::kSqrt:
      first = 0.5 / self;
      second_along = -dot(0) / (4.0 * self * self * self);
      break;
    // d sin(a)/da = cos(a), whose derivative along the direction is -sin(a) a-dot; and
    // d cos(a)/da = -sin(a), whose derivative is -cos(a) a-dot. `self` is sin(a), cos(a).
    case Op::kSin:
      first = std::cos(child[0]);
      second_along = -self * dot(0);
      break;
    case Op::kCos:
      first = -std::sin(child[0]);
      second_along = -self * dot(0);
      break;
  }
  if (!std::isfinite(first) || !std::isfinite(second_along)) return false;
  *d = first;
  if (d_dot != nullptr) *d_dot = second_along;
  return true;
}

}  // namespace sankhya::nlp::detail
