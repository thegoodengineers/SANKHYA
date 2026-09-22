// SPDX-License-Identifier: Apache-2.0
// SANKHYA - parallel tree search (#222): the state the workers share, and nothing else.
//
// THE DESIGN IN ONE PARAGRAPH. Each worker thread runs the ordinary, sequential
// BranchAndBound on one SUBTREE at a time: its own working model, its own node LPs, its own
// open list. A subtree is a chain of bound changes from the root, which is all a node is in
// this tree (branch_and_bound_internal.hpp), so handing one to another thread copies a few
// dozen numbers and nothing else. What the workers share is exactly what the issue names: the
// incumbent (so every worker prunes against the best point anyone found), the node count (so
// node_limit means the whole search), the queue of subtrees waiting for a worker, and the stop
// flag. A worker with at least eight open nodes (kMinOpenToDonate) gives its smallest-bound
// ones away, never its two newest, whenever another is idle and the queue is empty, which is
// how the root's subtree fans out to the other threads at the start and how the load stays
// level later.
//
// WHY THE ANSWER CANNOT DEPEND ON THE THREADS. A subtree is closed by the same code that
// closes the sequential tree, and a subtree that stops early (a limit, or its own gap test)
// hands back the smallest bound among its open nodes. The optimum is proved only when every
// subtree closed or stopped within the gap target, and the reported bound is the smallest of
// everything left open anywhere, so no thread's early exit can turn into a claim. The tree a
// run explores does vary with timing; the objective and the status do not.
//
// THE SCHEME HAS A NAME. This is the subtree-parallel branch and bound of Ralphs, Shinano,
// Berthold and Koch, "Parallel Solvers for Mixed Integer Linear Optimization", in Hamadi and
// Sais (eds.), Handbook of Parallel Constraint Reasoning, Springer, 2018: each worker owns a
// subtree, the incumbent and the bound are the shared state, and idle workers are fed by
// taking work from the busy ones. docs/PROVENANCE.md carries the row.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

#include "branch_and_bound_internal.hpp"
#include "solution_pool.hpp"

namespace sankhya::mip {

/// One subtree waiting for a worker: the node's whole domain as bound changes from the root,
/// and what the node carried in the tree it came from.
struct SubtreeSpec {
  bool is_root = false;             ///< the whole problem: the root node, its cuts and its dive
  std::vector<DomainChange> chain;  ///< root first; the last entry is the node's own change
  double bound = -std::numeric_limits<double>::infinity();
  double estimate = -std::numeric_limits<double>::infinity();
  double fraction = 0.0;
  Index depth = 0;
  WarmStart warm;  ///< empty when the donor's rows differ from the model's (its cut rows)
};

class SharedSearch {
 public:
  SharedSearch(int workers, Count node_limit, SolutionPool pool)
      : workers_(workers), node_limit_(node_limit), pool_(std::move(pool)) {}

  // ---- The queue ----------------------------------------------------------------------

  void push(SubtreeSpec spec);
  /// Block until there is a subtree to take (true) or the search is over (false): stopped,
  /// or the queue empty with no worker left that could refill it.
  bool take(SubtreeSpec* spec);
  /// The worker finished the subtree it took.
  void finished_subtree();
  /// A worker is waiting and the queue is empty: a busy worker should give some nodes away.
  [[nodiscard]] bool hungry() const noexcept {
    return idle_.load(std::memory_order_relaxed) > 0 &&
           queued_.load(std::memory_order_relaxed) == 0 && !stopped();
  }
  [[nodiscard]] int idle() const noexcept { return idle_.load(std::memory_order_relaxed); }
  /// Subtrees waiting for a worker: what the driver can report as open, since the workers'
  /// own open lists are theirs alone until they leave.
  [[nodiscard]] std::size_t queued() const noexcept {
    return queued_.load(std::memory_order_relaxed);
  }
  /// Every worker has left take() for good.
  [[nodiscard]] bool wait_until_done(std::chrono::milliseconds timeout);
  void worker_exited();
  /// The smallest bound among subtrees still queued (the search stopped before they ran).
  [[nodiscard]] double queued_bound();

  // ---- The incumbent, in minimise space ------------------------------------------------

  [[nodiscard]] double best() const noexcept { return best_.load(std::memory_order_acquire); }
  /// Offer a verified point; true when it became the incumbent.
  bool publish(double objective, const std::vector<double>& x);
  /// Copy the incumbent out; false when there is none.
  bool incumbent(double* objective, std::vector<double>* x);
  /// Every verified point goes to the one shared pool, as in the sequential search (#225).
  void offer_to_pool(double objective, const std::vector<double>& x);
  [[nodiscard]] SolutionPool& pool() { return pool_; }

  // ---- Stopping -------------------------------------------------------------------------

  void stop(LimitReason why);
  [[nodiscard]] bool stopped() const noexcept { return stop_.load(std::memory_order_acquire); }
  [[nodiscard]] LimitReason reason();
  /// A worker's answer that ends the whole search: unbounded, a numerical failure.
  void fail(Solution answer);
  [[nodiscard]] bool failed();
  [[nodiscard]] Solution failure();
  void record_exception(std::exception_ptr error);
  [[nodiscard]] std::exception_ptr exception();

  // ---- Counters -------------------------------------------------------------------------

  /// Add explored nodes; true when the global node limit is now reached.
  bool add_nodes(Count delta);
  [[nodiscard]] Count nodes() const noexcept { return nodes_.load(std::memory_order_relaxed); }
  [[nodiscard]] Count node_limit() const noexcept { return node_limit_; }
  /// A subtree that ended with nodes still open (a limit or its gap test) leaves their
  /// smallest bound here; gap_met records that some subtree stopped on the gap target.
  void add_residual(double bound, bool gap_met);
  [[nodiscard]] double residual();
  [[nodiscard]] bool gap_met();
  /// The objective step of #221, as the workers detected it; 0 when none is known.
  void set_objective_step(double step) {
    objective_step_.store(step, std::memory_order_relaxed);
  }
  [[nodiscard]] double objective_step() const {
    return objective_step_.load(std::memory_order_relaxed);
  }
  std::atomic<Count> subtrees{0};
  std::atomic<Count> donations{0};

  // ---- Pseudocosts, pooled across workers ------------------------------------------------

  struct Pseudocosts {
    std::vector<double> down_sum;
    std::vector<double> up_sum;
    std::vector<Count> down_count;
    std::vector<Count> up_count;
  };
  [[nodiscard]] Pseudocosts pseudocosts();

  /// The node scaling, computed once by the driver before any worker starts and read-only
  /// after, so it needs no lock.
  void set_scaling(NodeScaling scaling) {
    scaling_ = std::move(scaling);
    has_scaling_ = true;
  }
  [[nodiscard]] const NodeScaling* scaling() const {
    return has_scaling_ ? &scaling_ : nullptr;
  }
  /// Add what a worker observed on top of the snapshot it started from.
  void merge_pseudocosts(const Pseudocosts& start, const Pseudocosts& end);

 private:
  const int workers_;
  const Count node_limit_;

  std::mutex queue_mutex_;
  std::condition_variable queue_changed_;
  std::vector<SubtreeSpec> queue_;
  int busy_ = 0;
  int exited_ = 0;
  std::atomic<int> idle_{0};
  std::atomic<std::size_t> queued_{0};

  std::mutex incumbent_mutex_;
  std::atomic<double> best_{std::numeric_limits<double>::infinity()};
  std::vector<double> best_x_;
  SolutionPool pool_;

  std::atomic<bool> stop_{false};
  std::mutex state_mutex_;
  LimitReason reason_ = LimitReason::kNone;
  bool failed_ = false;
  Solution failure_;
  std::exception_ptr exception_;
  double residual_ = std::numeric_limits<double>::infinity();
  bool gap_met_ = false;

  std::atomic<Count> nodes_{0};
  std::atomic<double> objective_step_{0.0};

  std::mutex pseudocost_mutex_;
  Pseudocosts pseudocosts_;

  NodeScaling scaling_;
  bool has_scaling_ = false;
};

/// How many tree workers `mip_threads` asks for on this model: 1 when it asks for one, and
/// also, with a note in the log, for a model or mode the parallel search does not take.
[[nodiscard]] int parallel_threads(const Model& model, const Options& options, Logger& logger);

/// The parallel search on `threads` workers. Called by solve_branch_and_bound when
/// mip_threads asks for more than one and the model is one the parallel search takes.
[[nodiscard]] Solution solve_branch_and_bound_parallel(const Model& model,
                                                       const Options& options, Logger& logger,
                                                       SolveControl* control, int threads);

}  // namespace sankhya::mip
