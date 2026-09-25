// SPDX-License-Identifier: Apache-2.0
// SANKHYA - branch and bound: many nodes, or many strong-branching children, bounded in one
// batched PDHG run (#520).
//
// A single node LP is too small for a GPU; a batch of them is not. The nodes share the
// matrix and differ only in their column boxes, so K of them are one sparse matrix times a
// dense n x K block per iteration (src/pdhg/batch_pdhg.hpp, src/gpu/pdhg_batch.cu). After a
// fixed budget, each LP's bound is the Neumaier-Shcherbina bound of its duals against ITS
// OWN box (core/safe_bound.hpp), which is a valid lower bound whatever the duals are. So
// the batch is used only to prune and to order - never to accept a point as optimal:
//
//  * gpu_batch_nodes: between nodes, up to gpu_batch_size open nodes are bounded together.
//    A node's box is the global bounds with its chain of branching changes applied, without
//    the propagation it gets when entered: a larger box, so a weaker but still valid bound
//    for everything below the node. The stored bound becomes the larger of the two, and a
//    node that can then be pruned leaves the open list without a simplex solve.
//
//  * gpu_batch_strong_branching: the 2K children of the K strong-branching candidates are
//    one batch, started from the node's own point and duals. A child whose bound already
//    prunes it is a closed side, as an infeasible probe is (#502). Under "score" the other
//    children's batched bounds are their scores (the issue's use 1); under "filter" they are
//    probed by the dual simplex as before, and the batch only spares the closed ones.
//
// The simplex stays the node solver for every node that survives.
//
// References: Neumaier & Shcherbina (2004) for the bound; Applegate et al. (2021) and Lu &
// Yang (2023) for the method; Achterberg, Koch & Martin, "Branching rules revisited", ORL 33
// (2005) for strong branching and the product score. PR #661 proposed the n x K block
// layout, reused here.

#include "branch_and_bound_internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "../util/profiler.hpp"
#include "batch_audit.hpp"
#include "pdhg/batch_pdhg.hpp"

namespace sankhya::mip {

BatchPruneHook& batch_prune_audit_for_testing() {
  static BatchPruneHook hook;
  return hook;
}

namespace {

pdhg::BatchSettings batch_settings(Count iterations, bool device, double time_limit,
                                   double elapsed) {
  pdhg::BatchSettings settings;
  settings.iterations = iterations;
  settings.backend = device ? pdhg::BatchBackendKind::kDevice : pdhg::BatchBackendKind::kCpu;
  if (time_limit > 0.0 && std::isfinite(time_limit)) {
    settings.time_limit = std::max(0.0, time_limit - elapsed);
  }
  return settings;
}

}  // namespace

void BranchAndBound::init_batch() {
  batch_nodes_ = options_.get_bool("gpu_batch_nodes");
  const std::string& strong = options_.get_string("gpu_batch_strong_branching");
  batch_strong_ = strong != "off";
  batch_score_ = strong == "score";
  batch_size_ = static_cast<Index>(options_.get_int("gpu_batch_size"));
  batch_iterations_ = options_.get_int("gpu_batch_iterations");
  batch_device_ = options_.get_string("gpu_batch_backend") == "auto";
  if (!batch_nodes_ && !batch_strong_) return;
  // A QP node's bound is not an LP's; a certificate proves each leaf from its own node LP,
  // and a node closed by a batch has none; a parallel worker shares its open list.
  const char* declined = quadratic_           ? "the relaxation is a QP"
                         : certificate_mode() ? "a certificate is being written"
                         : shared_ != nullptr ? "the search is parallel"
                                              : nullptr;
  if (declined != nullptr) {
    logger_.info("Batched PDHG bounds (#520) off: {}", declined);
    batch_nodes_ = false;
    batch_strong_ = false;
    batch_score_ = false;
  }
}

void BranchAndBound::batch_remember_root(const Solution& relaxation) {
  if (!batch_nodes_) return;
  batch_root_x_ = relaxation.col_value;
  batch_root_y_.resize(relaxation.row_dual.size());
  for (std::size_t i = 0; i < batch_root_y_.size(); ++i) {
    batch_root_y_[i] = sense_ * relaxation.row_dual[i];
  }
}

void BranchAndBound::batch_audit(bool strong_branching, const std::vector<double>& lower,
                                 const std::vector<double>& upper, double bound) const {
  const BatchPruneHook& hook = batch_prune_audit_for_testing();
  if (!hook) return;
  BatchPruneRecord record;
  record.strong_branching = strong_branching;
  record.col_lower = lower;
  record.col_upper = upper;
  record.bound = bound;
  record.cutoff = pool_complete_ ? pool_cutoff() : incumbent_internal_;
  record.margin = absolute_gap_target_;
  record.objective_step = objective_step_;
  hook(record);
}

void BranchAndBound::batch_bound_open_nodes() {
  // Between nodes only: working_'s column bounds must be the global ones.
  if (!batch_nodes_ || !saved_.empty() || batch_size_ <= 0) return;
  if (!have_incumbent_ && !pool_complete_) return;  // nothing to prune against yet

  // The most recently opened nodes that have not been batched, once K of them are waiting
  // or every open node is one.
  const auto k_max = static_cast<std::size_t>(batch_size_);
  std::vector<Index> picked;
  std::size_t waiting = 0;
  for (auto it = open_.rbegin(); it != open_.rend(); ++it) {
    if (nodes_[static_cast<std::size_t>(*it)].batch_bounded) continue;
    ++waiting;
    if (picked.size() < k_max) picked.push_back(*it);
  }
  if (picked.empty() || (waiting < k_max && waiting < open_.size())) return;

  // Each node's box: the global bounds with its chain applied, as enter() applies it. A
  // node whose chain moves a row bound (objective branching, #418) or whose box is empty
  // is left to the ordinary search.
  const Index n = original_.num_cols();
  const auto un = static_cast<std::size_t>(n);
  std::vector<Index> bounded;
  pdhg::BatchProblem problem;
  problem.model = &working_;
  problem.cost.resize(un);
  for (std::size_t j = 0; j < un; ++j) problem.cost[j] = sense_ * working_.col_cost[j];
  for (const Index node_index : picked) {
    nodes_[static_cast<std::size_t>(node_index)].batch_bounded = true;
    std::vector<double> lower = working_.col_lower;
    std::vector<double> upper = working_.col_upper;
    bool usable = true;
    for (Index walk = node_index; walk >= 0 && usable;) {
      const TreeNode& node = nodes_[static_cast<std::size_t>(walk)];
      if (node.has_change) {
        if (node.change.column >= n) {
          usable = false;
          break;
        }
        const auto u = static_cast<std::size_t>(node.change.column);
        if (node.change.is_upper) {
          upper[u] = std::min(upper[u], node.change.value);
        } else {
          lower[u] = std::max(lower[u], node.change.value);
        }
      }
      walk = node.parent;
    }
    for (std::size_t j = 0; j < un && usable; ++j) usable = lower[j] <= upper[j];
    if (!usable) continue;
    bounded.push_back(node_index);
    problem.col_lower.insert(problem.col_lower.end(), lower.begin(), lower.end());
    problem.col_upper.insert(problem.col_upper.end(), upper.begin(), upper.end());
  }
  if (bounded.empty()) return;
  problem.count = static_cast<Index>(bounded.size());
  if (batch_root_x_.size() == un) problem.primal_start = batch_root_x_;
  problem.dual_start = batch_root_y_;

  pdhg::BatchResult result;
  {
    ProfileScope timed(logger_.profiler(), "batched node bounds", ProfileMode::kDetailed);
    result = pdhg::solve_batch(problem, batch_settings(batch_iterations_, batch_device_,
                                                       time_limit_, timer_.elapsed_seconds()));
  }
  ++batch_calls_;
  if (result.on_device) ++batch_device_calls_;
  batch_lps_ += problem.count;
  batch_seconds_ += result.seconds;

  bool changed = false;
  std::vector<char> drop(nodes_.size(), 0);
  for (std::size_t k = 0; k < bounded.size(); ++k) {
    TreeNode& node = nodes_[static_cast<std::size_t>(bounded[k])];
    const double bound = result.bound[k];
    if (!(bound > node.bound)) continue;
    const bool prunable_before = can_prune(node.bound);
    node.bound = bound;
    ++batch_bounds_raised_;
    changed = true;
    if (!can_prune(bound)) continue;
    drop[static_cast<std::size_t>(bounded[k])] = 1;
    ++nodes_pruned_;
    if (prunable_before) continue;
    ++batch_node_prunes_;
    const std::size_t offset = k * un;
    batch_audit(false,
                std::vector<double>(
                    problem.col_lower.begin() + static_cast<std::ptrdiff_t>(offset),
                    problem.col_lower.begin() + static_cast<std::ptrdiff_t>(offset + un)),
                std::vector<double>(
                    problem.col_upper.begin() + static_cast<std::ptrdiff_t>(offset),
                    problem.col_upper.begin() + static_cast<std::ptrdiff_t>(offset + un)),
                bound);
  }
  if (!changed) return;
  open_.erase(std::remove_if(open_.begin(), open_.end(),
                             [&](Index i) { return drop[static_cast<std::size_t>(i)] != 0; }),
              open_.end());
  if (open_is_heap()) rebuild_open_heap();
}

bool BranchAndBound::batch_strong_branch(const std::vector<double>& x,
                                         const std::vector<Index>& columns,
                                         std::vector<double>* down, std::vector<double>* up) {
  if (columns.empty()) return false;
  const auto un = static_cast<std::size_t>(original_.num_cols());
  pdhg::BatchProblem problem;
  problem.model = &working_;
  problem.cost.resize(un);
  for (std::size_t j = 0; j < un; ++j) problem.cost[j] = sense_ * working_.col_cost[j];
  problem.count = static_cast<Index>(2 * columns.size());
  problem.col_lower.reserve(un * 2 * columns.size());
  problem.col_upper.reserve(un * 2 * columns.size());
  for (const Index column : columns) {
    const auto u = static_cast<std::size_t>(column);
    for (int direction = 0; direction < 2; ++direction) {
      const std::size_t at = problem.col_lower.size() + u;
      problem.col_lower.insert(problem.col_lower.end(), working_.col_lower.begin(),
                               working_.col_lower.end());
      problem.col_upper.insert(problem.col_upper.end(), working_.col_upper.begin(),
                               working_.col_upper.end());
      if (direction == 0) {
        problem.col_upper[at] = std::min(problem.col_upper[at], std::floor(x[u]));
      } else {
        problem.col_lower[at] = std::max(problem.col_lower[at], std::floor(x[u]) + 1.0);
      }
    }
  }
  if (x.size() == un) problem.primal_start = x;
  problem.dual_start = batch_branch_duals_;

  pdhg::BatchResult result;
  {
    ProfileScope timed(logger_.profiler(), "batched strong branching", ProfileMode::kDetailed);
    result = pdhg::solve_batch(problem, batch_settings(batch_iterations_, batch_device_,
                                                       time_limit_, timer_.elapsed_seconds()));
  }
  ++batch_calls_;
  if (result.on_device) ++batch_device_calls_;
  batch_lps_ += problem.count;
  batch_seconds_ += result.seconds;

  down->assign(columns.size(), -std::numeric_limits<double>::infinity());
  up->assign(columns.size(), -std::numeric_limits<double>::infinity());
  for (std::size_t i = 0; i < columns.size(); ++i) {
    for (std::size_t direction = 0; direction < 2; ++direction) {
      const std::size_t k = 2 * i + direction;
      const double bound = result.bound[k];
      (direction == 0 ? *down : *up)[i] = bound;
      if (!can_prune(bound)) continue;
      ++batch_children_closed_;
      const std::size_t offset = k * un;
      batch_audit(true,
                  std::vector<double>(
                      problem.col_lower.begin() + static_cast<std::ptrdiff_t>(offset),
                      problem.col_lower.begin() + static_cast<std::ptrdiff_t>(offset + un)),
                  std::vector<double>(
                      problem.col_upper.begin() + static_cast<std::ptrdiff_t>(offset),
                      problem.col_upper.begin() + static_cast<std::ptrdiff_t>(offset + un)),
                  bound);
    }
  }
  return true;
}

void BranchAndBound::report_batch() const {
  if (!batch_nodes_ && !batch_strong_) return;
  logger_.info(
      "Batched PDHG bounds (#520): {} batch(es) of {} LP(s) in {:.3f}s, {} on the device; {} "
      "node bound(s) raised, {} node(s) pruned without a simplex solve, {} strong-branching "
      "child(ren) closed",
      batch_calls_, batch_lps_, batch_seconds_, batch_device_calls_, batch_bounds_raised_,
      batch_node_prunes_, batch_children_closed_);
  Profiler* profiler = logger_.profiler();
  if (profiler == nullptr) return;
  // So an A/B can tell "no effect" from "never ran".
  profiler->count("batch calls", static_cast<std::int64_t>(batch_calls_));
  profiler->count("batch device calls", static_cast<std::int64_t>(batch_device_calls_));
  profiler->count("batch LPs", static_cast<std::int64_t>(batch_lps_));
  profiler->count("batch bounds raised", static_cast<std::int64_t>(batch_bounds_raised_));
  profiler->count("batch nodes pruned", static_cast<std::int64_t>(batch_node_prunes_));
  profiler->count("batch children closed", static_cast<std::int64_t>(batch_children_closed_));
}

}  // namespace sankhya::mip
