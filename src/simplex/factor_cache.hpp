// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the LU factors of starting bases, kept across node solves (#501).
//
// WHY. Branch and bound solves many LPs from the SAME starting basis: every strong-branching
// probe of a node starts from that node's optimal basis, and so do both of its children.
// Each of those solves builds a fresh simplex and factorizes that basis from scratch. The
// profile in #501 put that first factorization at 10 to 19 percent of node LP time on
// MIPLIB instances with many cheap nodes (p0201, gen-ip054, pk1, markshare_4_0, ran12x21);
// the per-node copy of the scaled model the issue first suspected was 1 percent.
//
// WHAT IS REUSED, AND WHY IT IS THE SAME ANSWER. A factorization is a pure function of
// the basis matrix (which columns, in which positions, of which scaled matrix) and of the
// factorization's own settings; the simplex builds a fresh SparseLu for every solve and
// factorizes it once in prepare(). This cache keeps a copy of that SparseLu taken right
// after a first factorization that succeeded at the default Markowitz threshold without
// repair, keyed by everything the factorization reads, and hands the copy back to a later
// solve whose starting basis is identical. The later solve then continues from factors
// bit-for-bit equal to the ones it would have computed, so the pivots, the answer and the
// tree are unchanged; only the time differs. Achterberg, Constraint Integer Programming (PhD
// thesis, TU Berlin 2007), on warm-starting the node LPs of a tree from the parent's basis; the
// factorization is the part of that warm start this keeps.
//
// WHAT IS NOT REUSED. The parent's FINAL factors: they describe the same basis in the
// parent's pivot order, not in the order the child seeds it, so they are a different
// (equally valid) factorization and could change rounding. Reusing them is a behaviour
// change that needs its own A/B, and is not done here.
//
// The matrix is identified by NodeScaling::id, which build_node_scaling() draws from a
// process-wide counter: the simplex factorizes the cache's own scaled matrix, so one id is
// one matrix for as long as any copy of that NodeScaling exists.
//
// NOT THREAD SAFE. One cache per search; a parallel tree gives each worker its own.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "sankhya/types.hpp"

#include "../la/lu.hpp"

namespace sankhya {

class NodeFactorCache {
 public:
  /// At most `capacity` factorizations are kept, least recently used dropped first.
  explicit NodeFactorCache(std::size_t capacity) : capacity_(capacity) {}

  /// Everything a first factorization reads besides the matrix values, which `matrix_id`
  /// stands for.
  struct Key {
    std::uint64_t matrix_id = 0;
    Index rows = 0;
    Index cols = 0;
    bool forrest_tomlin = false;
    const std::vector<Index>* basis = nullptr;  ///< variable in each basis position
  };

  /// The factors stored under `key`, or nullptr. A hit becomes the most recently used.
  /// The pointer is valid until the next call on this cache.
  [[nodiscard]] const SparseLu* find(const Key& key);
  /// Keep a copy of `lu`, the fresh factorization of `key`'s basis. A key with matrix_id
  /// 0 is never stored: it names no matrix.
  void store(const Key& key, const SparseLu& lu);

  [[nodiscard]] std::int64_t hits() const noexcept { return hits_; }
  [[nodiscard]] std::int64_t misses() const noexcept { return misses_; }

 private:
  struct Entry {
    std::uint64_t matrix_id = 0;
    Index rows = 0;
    Index cols = 0;
    bool forrest_tomlin = false;
    std::uint64_t hash = 0;
    std::vector<Index> basis;
    SparseLu lu;
  };
  [[nodiscard]] static std::uint64_t hash_of(const Key& key);
  [[nodiscard]] static bool matches(const Entry& entry, const Key& key, std::uint64_t hash);

  std::size_t capacity_;
  std::vector<Entry> entries_;  ///< least recently used first
  std::int64_t hits_ = 0;
  std::int64_t misses_ = 0;
};

}  // namespace sankhya
