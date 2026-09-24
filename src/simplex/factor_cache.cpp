// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the LU factors of starting bases, kept across node solves (#501). The design
// and the argument that a hit changes no answer are in factor_cache.hpp.

#include "factor_cache.hpp"
#include "simplex_core.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <utility>

namespace sankhya {

std::uint64_t NodeFactorCache::hash_of(const Key& key) {
  // FNV-1a over the basis, then the scalars. A hash only filters: a hit is decided by the
  // full comparison in matches(), so a collision costs a comparison, never an answer.
  std::uint64_t h = 1469598103934665603ULL;
  const auto mix = [&h](std::uint64_t v) {
    h ^= v;
    h *= 1099511628211ULL;
  };
  for (const Index k : *key.basis) mix(static_cast<std::uint64_t>(k));
  mix(key.matrix_id);
  mix(static_cast<std::uint64_t>(key.rows));
  mix(static_cast<std::uint64_t>(key.cols));
  mix(key.forrest_tomlin ? 1U : 0U);
  return h;
}

bool NodeFactorCache::matches(const Entry& entry, const Key& key, std::uint64_t hash) {
  return entry.hash == hash && entry.matrix_id == key.matrix_id && entry.rows == key.rows &&
         entry.cols == key.cols && entry.forrest_tomlin == key.forrest_tomlin &&
         entry.basis == *key.basis;
}

const SparseLu* NodeFactorCache::find(const Key& key) {
  if (key.matrix_id == 0 || key.basis == nullptr || capacity_ == 0) return nullptr;
  const std::uint64_t hash = hash_of(key);
  for (std::size_t i = entries_.size(); i-- > 0;) {
    if (!matches(entries_[i], key, hash)) continue;
    ++hits_;
    // Most recently used goes last; the relative order of the others is kept.
    std::rotate(entries_.begin() + static_cast<std::ptrdiff_t>(i),
                entries_.begin() + static_cast<std::ptrdiff_t>(i) + 1, entries_.end());
    return &entries_.back().lu;
  }
  ++misses_;
  return nullptr;
}

void NodeFactorCache::store(const Key& key, const SparseLu& lu) {
  if (key.matrix_id == 0 || key.basis == nullptr || capacity_ == 0) return;
  const std::uint64_t hash = hash_of(key);
  for (const Entry& entry : entries_) {
    if (matches(entry, key, hash)) return;  // already held; find() would have returned it
  }
  if (entries_.size() >= capacity_) entries_.erase(entries_.begin());
  Entry entry;
  entry.matrix_id = key.matrix_id;
  entry.rows = key.rows;
  entry.cols = key.cols;
  entry.forrest_tomlin = key.forrest_tomlin;
  entry.hash = hash;
  entry.basis = *key.basis;
  entry.lu = lu;
  entries_.push_back(std::move(entry));
}

}  // namespace sankhya

namespace sankhya::detail {

bool Simplex::first_factorization() {
  if (factor_cache_ == nullptr || factor_cache_matrix_ == 0) return refactorize();
  // THE SAME STATE refactorize() WOULD LEAVE (#501, factor_cache.hpp). A hit is only ever a
  // copy of a fresh SparseLu factorized from this very basis of this very matrix at the
  // default threshold without repair, so the factors are the ones refactorize() would
  // compute; what refactorize() sets besides the factors is set here the same way.
  const NodeFactorCache::Key key{factor_cache_matrix_, m_, n_, lu_.forrest_tomlin(), &basis_};
  if (const SparseLu* cached = factor_cache_->find(key)) {
    eta_work_since_refactor_ = 0.0;
    std::fill(numerically_dependent_.begin(), numerically_dependent_.end(), 0);
    lu_ = *cached;
    // The copy carries the cached LU's own hyper-sparse flag; this solve's option decides
    // (#464).
    lu_.use_hyper_sparse(options_.get_bool("lu_hyper_sparse"));
    basis_needed_stricter_threshold_ = false;
    last_factorization_plain_ = true;
    const double pivot = lu_.smallest_pivot();
    if (pivot > 0.0 && (worst_basis_pivot_ == 0.0 || pivot < worst_basis_pivot_)) {
      worst_basis_pivot_ = pivot;
    }
    logger_.verbose("reused the factors of this starting basis: smallest pivot {:.3e}", pivot);
    return true;
  }
  if (!refactorize()) return false;
  if (last_factorization_plain_) factor_cache_->store(key, lu_);
  return true;
}

}  // namespace sankhya::detail
