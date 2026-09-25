// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the structural sparsity of an expression's Hessian (NLP stage 1).
//
// WHY STRUCTURE AND NOT VALUES. ExpressionGraph::hessian() omits an entry that is exactly zero
// at the point it was asked about: x * y at y = 0 has d2/dx dy = 1 but x * exp(y) - x at
// y = 0 has a zero there. A solver that factorizes the Hessian of the Lagrangian fixes the
// pattern ONCE (the symbolic analysis of src/la/ldl.cpp) and refills the values every
// iteration, so it needs every position that can EVER be nonzero, independent of x.
//
// THE RULE. The Hessian of a composite is, by the chain rule applied twice,
//
//     d2 F = sum over nodes v of  adjoint(v) * sum_{k,l} d2 phi_v / du_k du_l  grad u_k grad
//     u_l'
//
// (Griewank and Walther, "Evaluating Derivatives", 2nd ed., SIAM 2008, ch. 7), and the
// pattern of grad u_k is contained in the set of columns u_k depends on, its INDEX DOMAIN.
// So the pattern is contained in the union, over every node whose local second derivative
// d2 phi / du_k du_l is not identically zero, of domain(u_k) x domain(u_l). That is the
// nonlinear-interaction propagation of Walther, "Computing sparse Hessians with automatic
// differentiation", ACM TOMS 34(1), 2008:
//
//   sum, negate              linear in every child: no interaction
//   a * b                    d2/da db = 1:           domain(a) x domain(b)
//   a / b                    d2/da db, d2/db2 != 0:  domain(a) x domain(b), domain(b)^2
//   a ^ p  (p != 0, 1)       domain(a)^2
//   exp, log, sqrt, sin, cos domain(a)^2
//
// A constant child has an empty domain, so 3 * x and x / 3 contribute nothing, as they must.
// The result is a SUPERSET of every hessian() pattern - the adjoint of a node may vanish at a
// point - and never a subset, which is the direction a fixed factorization pattern needs.

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <unordered_set>

#include "nlp/expression.hpp"

namespace sankhya::nlp {
namespace {

/// Every (i, j) with i >= j from a x b, both directions of the symmetric pair folded into the
/// lower triangle, keyed as one 64-bit integer.
void add_block(const std::vector<Index>& a, const std::vector<Index>& b,
               std::unordered_set<std::uint64_t>* pairs) {
  for (const Index i : a) {
    for (const Index j : b) {
      const Index row = std::max(i, j);
      const Index col = std::min(i, j);
      pairs->insert((static_cast<std::uint64_t>(static_cast<std::uint32_t>(col)) << 32) |
                    static_cast<std::uint32_t>(row));
    }
  }
}

std::vector<Index> merged(const std::vector<Index>& a, const std::vector<Index>& b) {
  std::vector<Index> out;
  out.reserve(a.size() + b.size());
  std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));
  return out;
}

}  // namespace

std::vector<std::pair<Index, Index>> ExpressionGraph::hessian_pattern(ExprId root) const {
  if (!contains(root)) return {};
  const std::vector<ExprId> order = tape(root);
  // The index domain of every node on the tape, children before parents.
  std::vector<std::vector<Index>> domain(size());
  std::unordered_set<std::uint64_t> pairs;
  for (const ExprId id : order) {
    const Node& n = node(id);
    const auto kid = [&](std::size_t k) -> const std::vector<Index>& {
      return domain[static_cast<std::size_t>(n.children[k])];
    };
    std::vector<Index>& mine = domain[static_cast<std::size_t>(id)];
    if (n.op == Op::kVariable) {
      mine = {n.variable};
      continue;
    }
    for (std::size_t k = 0; k < n.children.size(); ++k) mine = merged(mine, kid(k));
    switch (n.op) {
      case Op::kConstant:
      case Op::kVariable:
      case Op::kSum:
      case Op::kNegate: break;
      case Op::kProduct: add_block(kid(0), kid(1), &pairs); break;
      case Op::kDivide:
        if (!kid(1).empty()) {
          add_block(kid(0), kid(1), &pairs);
          add_block(kid(1), kid(1), &pairs);
        }
        break;
      case Op::kPower:
        if (n.value != 0.0 && n.value != 1.0) add_block(kid(0), kid(0), &pairs);
        break;
      case Op::kExp:
      case Op::kLog:
      case Op::kSqrt:
      case Op::kSin:
      case Op::kCos: add_block(kid(0), kid(0), &pairs); break;
    }
  }
  std::vector<std::pair<Index, Index>> out;
  out.reserve(pairs.size());
  for (const std::uint64_t key : pairs) {
    out.emplace_back(static_cast<Index>(key & 0xffffffffU), static_cast<Index>(key >> 32));
  }
  // By column, then row: the CSC order a lower-triangle matrix is assembled in.
  std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
    return a.second != b.second ? a.second < b.second : a.first < b.first;
  });
  return out;
}

}  // namespace sankhya::nlp
