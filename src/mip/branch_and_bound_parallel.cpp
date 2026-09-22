// SPDX-License-Identifier: Apache-2.0
// SANKHYA - parallel tree search (#222). See parallel_search.hpp for the design and for why
// the answer cannot depend on the number of threads.

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "branch_and_bound_internal.hpp"
#include "parallel_search.hpp"
#include "util/threads.hpp"

namespace sankhya::mip {

// =========================================================================================
// SharedSearch
// =========================================================================================

void SharedSearch::push(SubtreeSpec spec) {
  {
    const std::lock_guard<std::mutex> lock(queue_mutex_);
    queue_.push_back(std::move(spec));
    queued_.store(queue_.size(), std::memory_order_relaxed);
  }
  // All, not one: the driver waits on the same condition, and a wake-up it swallowed would
  // leave a worker asleep beside a full queue.
  queue_changed_.notify_all();
}

bool SharedSearch::take(SubtreeSpec* spec) {
  std::unique_lock<std::mutex> lock(queue_mutex_);
  idle_.fetch_add(1, std::memory_order_relaxed);
  // Wait while there is nothing to take but a busy worker could still give something away.
  // Counting busy workers, not only the queue's length, is what keeps a worker from leaving
  // while another is about to push the subtrees it would have taken.
  queue_changed_.wait(lock, [&] { return !queue_.empty() || busy_ == 0 || stopped(); });
  idle_.fetch_sub(1, std::memory_order_relaxed);
  if (stopped() || queue_.empty()) {
    queue_changed_.notify_all();
    return false;
  }
  // The smallest bound first: that subtree holds the global bound, and closing it is what
  // moves the proof.
  std::size_t pick = 0;
  for (std::size_t k = 1; k < queue_.size(); ++k) {
    if (queue_[k].bound < queue_[pick].bound) pick = k;
  }
  *spec = std::move(queue_[pick]);
  queue_.erase(queue_.begin() + static_cast<std::ptrdiff_t>(pick));
  queued_.store(queue_.size(), std::memory_order_relaxed);
  ++busy_;
  subtrees.fetch_add(1, std::memory_order_relaxed);
  return true;
}

void SharedSearch::finished_subtree() {
  {
    const std::lock_guard<std::mutex> lock(queue_mutex_);
    --busy_;
  }
  queue_changed_.notify_all();
}

void SharedSearch::worker_exited() {
  {
    const std::lock_guard<std::mutex> lock(queue_mutex_);
    ++exited_;
  }
  queue_changed_.notify_all();
}

bool SharedSearch::wait_until_done(std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(queue_mutex_);
  return queue_changed_.wait_for(lock, timeout, [&] { return exited_ == workers_; });
}

double SharedSearch::queued_bound() {
  const std::lock_guard<std::mutex> lock(queue_mutex_);
  double bound = std::numeric_limits<double>::infinity();
  for (const SubtreeSpec& spec : queue_) bound = std::min(bound, spec.bound);
  return bound;
}

bool SharedSearch::publish(double objective, const std::vector<double>& x) {
  const std::lock_guard<std::mutex> lock(incumbent_mutex_);
  if (objective >= best_.load(std::memory_order_relaxed) - 1e-12) return false;
  best_x_ = x;
  best_.store(objective, std::memory_order_release);
  return true;
}

bool SharedSearch::incumbent(double* objective, std::vector<double>* x) {
  const std::lock_guard<std::mutex> lock(incumbent_mutex_);
  if (best_x_.empty()) return false;
  *objective = best_.load(std::memory_order_relaxed);
  *x = best_x_;
  return true;
}

void SharedSearch::offer_to_pool(double objective, const std::vector<double>& x) {
  const std::lock_guard<std::mutex> lock(incumbent_mutex_);
  pool_.offer(objective, x);
}

void SharedSearch::stop(LimitReason why) {
  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    if (reason_ == LimitReason::kNone) reason_ = why;  // the first reason is the one reported
  }
  stop_.store(true, std::memory_order_release);
  const std::lock_guard<std::mutex> lock(queue_mutex_);
  queue_changed_.notify_all();
}

LimitReason SharedSearch::reason() {
  const std::lock_guard<std::mutex> lock(state_mutex_);
  return reason_;
}

void SharedSearch::fail(Solution answer) {
  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    if (!failed_) {
      failed_ = true;
      failure_ = std::move(answer);
    }
  }
  stop(LimitReason::kNone);
}

bool SharedSearch::failed() {
  const std::lock_guard<std::mutex> lock(state_mutex_);
  return failed_;
}

Solution SharedSearch::failure() {
  const std::lock_guard<std::mutex> lock(state_mutex_);
  return failure_;
}

void SharedSearch::record_exception(std::exception_ptr error) {
  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    if (!exception_) exception_ = std::move(error);
  }
  stop(LimitReason::kNone);
}

std::exception_ptr SharedSearch::exception() {
  const std::lock_guard<std::mutex> lock(state_mutex_);
  return exception_;
}

bool SharedSearch::add_nodes(Count delta) {
  const Count total = nodes_.fetch_add(delta, std::memory_order_relaxed) + delta;
  return node_limit_ >= 0 && total >= node_limit_;
}

void SharedSearch::add_residual(double bound, bool gap_met) {
  const std::lock_guard<std::mutex> lock(state_mutex_);
  residual_ = std::min(residual_, bound);
  gap_met_ = gap_met_ || gap_met;
}

double SharedSearch::residual() {
  const std::lock_guard<std::mutex> lock(state_mutex_);
  return residual_;
}

bool SharedSearch::gap_met() {
  const std::lock_guard<std::mutex> lock(state_mutex_);
  return gap_met_;
}

SharedSearch::Pseudocosts SharedSearch::pseudocosts() {
  const std::lock_guard<std::mutex> lock(pseudocost_mutex_);
  return pseudocosts_;
}

void SharedSearch::merge_pseudocosts(const Pseudocosts& start, const Pseudocosts& end) {
  const std::lock_guard<std::mutex> lock(pseudocost_mutex_);
  const std::size_t n = end.down_sum.size();
  if (pseudocosts_.down_sum.size() != n) {
    pseudocosts_.down_sum.assign(n, 0.0);
    pseudocosts_.up_sum.assign(n, 0.0);
    pseudocosts_.down_count.assign(n, 0);
    pseudocosts_.up_count.assign(n, 0);
  }
  const bool has_start = start.down_sum.size() == n;
  for (std::size_t j = 0; j < n; ++j) {
    pseudocosts_.down_sum[j] += end.down_sum[j] - (has_start ? start.down_sum[j] : 0.0);
    pseudocosts_.up_sum[j] += end.up_sum[j] - (has_start ? start.up_sum[j] : 0.0);
    pseudocosts_.down_count[j] += end.down_count[j] - (has_start ? start.down_count[j] : 0);
    pseudocosts_.up_count[j] += end.up_count[j] - (has_start ? start.up_count[j] : 0);
  }
}

// =========================================================================================
// The worker's side of BranchAndBound
// =========================================================================================

void BranchAndBound::plant_seed() {
  // The chain as ordinary tree nodes under the root, each one's parent the one before, so
  // enter() walks it exactly as it walks any node's ancestry. Only the last is open.
  Index parent = 0;
  for (std::size_t k = 0; k < seed_->chain.size(); ++k) {
    // Built in place, never copied: a copy of a node with an empty basis is what GCC's
    // -Wnull-dereference misreads under -fsanitize=thread.
    TreeNode& node = nodes_.emplace_back();
    node.parent = parent;
    node.change = seed_->chain[k];
    node.has_change = true;
    node.bound = seed_->bound;
    node.estimate = seed_->estimate;
    node.depth = static_cast<Index>(k + 1);
    parent = static_cast<Index>(nodes_.size() - 1);
  }
  TreeNode& leaf = nodes_.back();
  leaf.depth = seed_->depth;
  leaf.fraction = seed_->fraction;
  // A basis is only a basis of the same rows; the seed's is dropped if they differ.
  if (!seed_->warm.row_status.empty() &&
      static_cast<Index>(seed_->warm.row_status.size()) == working_.num_rows()) {
    leaf.warm.col_status.assign(seed_->warm.col_status.begin(), seed_->warm.col_status.end());
    leaf.warm.row_status.assign(seed_->warm.row_status.begin(), seed_->warm.row_status.end());
  }
  open_.push_back(parent);

  // Start from what every worker has learned about the columns so far.
  const SharedSearch::Pseudocosts shared = shared_->pseudocosts();
  if (shared.down_sum.size() == pseudo_down_sum_.size()) {
    pseudo_down_sum_ = shared.down_sum;
    pseudo_up_sum_ = shared.up_sum;
    pseudo_down_count_ = shared.down_count;
    pseudo_up_count_ = shared.up_count;
    pseudo_start_down_sum_ = shared.down_sum;
    pseudo_start_up_sum_ = shared.up_sum;
    pseudo_start_down_count_ = shared.down_count;
    pseudo_start_up_count_ = shared.up_count;
  }
}

bool BranchAndBound::sync_with_shared(LimitReason* why) {
  if (shared_->stopped()) {
    *why = shared_->reason();
    if (*why == LimitReason::kNone) *why = LimitReason::kInterrupt;  // another worker failed
    return false;
  }
  if (nodes_explored_ > nodes_reported_) {
    const bool reached = shared_->add_nodes(nodes_explored_ - nodes_reported_);
    nodes_reported_ = nodes_explored_;
    if (reached) {
      shared_->stop(LimitReason::kNodes);
      *why = LimitReason::kNodes;
      return false;
    }
  }
  // Pseudocosts both ways, every few nodes (#222). Reliability branching strong-branches on a
  // column until it has eight observations, and those observations are made wherever the
  // column's children are solved - often in another worker. Merged only when a subtree
  // ended, a long-lived subtree would keep strong-branching on columns the others had long
  // since priced.
  if (nodes_explored_ >= next_pseudocost_sync_) {
    next_pseudocost_sync_ = nodes_explored_ + kPseudocostSyncNodes;
    sync_pseudocosts();
  }
  // Another worker's incumbent prunes here as if this worker had found it.
  if (shared_->best() < incumbent_internal_ - 1e-12) {
    double objective = 0.0;
    std::vector<double> x;
    if (shared_->incumbent(&objective, &x) && objective < incumbent_internal_) {
      have_incumbent_ = true;
      incumbent_internal_ = objective;
      incumbent_x_ = std::move(x);
    }
  }
  // A backlog worth splitting, not merely two children: every subtree given away pays for a
  // model copy and a cold start, and a two-node subtree does not repay it.
  if (shared_->hungry() && open_.size() >= kMinOpenToDonate) donate_open_nodes();
  return true;
}

void BranchAndBound::sync_pseudocosts() {
  const SharedSearch::Pseudocosts start{pseudo_start_down_sum_, pseudo_start_up_sum_,
                                        pseudo_start_down_count_, pseudo_start_up_count_};
  const SharedSearch::Pseudocosts end{pseudo_down_sum_, pseudo_up_sum_, pseudo_down_count_,
                                      pseudo_up_count_};
  shared_->merge_pseudocosts(start, end);
  const SharedSearch::Pseudocosts merged = shared_->pseudocosts();
  if (merged.down_sum.size() != pseudo_down_sum_.size()) return;
  pseudo_down_sum_ = pseudo_start_down_sum_ = merged.down_sum;
  pseudo_up_sum_ = pseudo_start_up_sum_ = merged.up_sum;
  pseudo_down_count_ = pseudo_start_down_count_ = merged.down_count;
  pseudo_up_count_ = pseudo_start_up_count_ = merged.up_count;
}

void BranchAndBound::donate_open_nodes() {
  // The smallest-bound nodes, never the two newest: those are the children the hybrid's
  // dive is about to take, and the dive is what keeps the dual simplex's warm starts short.
  const std::size_t keep = 2;
  const auto give =
      std::min<std::size_t>(static_cast<std::size_t>(shared_->idle()), open_.size() - keep);
  for (std::size_t given = 0; given < give; ++given) {
    std::size_t pick = 0;
    for (std::size_t k = 1; k + keep < open_.size(); ++k) {
      if (nodes_[static_cast<std::size_t>(open_[k])].bound <
          nodes_[static_cast<std::size_t>(open_[pick])].bound) {
        pick = k;
      }
    }
    const Index index = open_[pick];
    open_.erase(open_.begin() + static_cast<std::ptrdiff_t>(pick));
    TreeNode& node = nodes_[static_cast<std::size_t>(index)];

    SubtreeSpec spec;
    for (Index walk = index; walk >= 0;) {
      const TreeNode& step = nodes_[static_cast<std::size_t>(walk)];
      if (step.has_change) spec.chain.push_back(step.change);
      walk = step.parent;
    }
    std::reverse(spec.chain.begin(), spec.chain.end());
    spec.bound = node.bound;
    spec.estimate = node.estimate;
    spec.fraction = node.fraction;
    spec.depth = node.depth;
    // Cut rows are this worker's own; a basis that counts them is no basis elsewhere.
    if (working_.num_rows() == original_.num_rows()) spec.warm = std::move(node.warm);
    node.warm = WarmStart{};
    shared_->push(std::move(spec));
    shared_->donations.fetch_add(1, std::memory_order_relaxed);
  }
}

void BranchAndBound::leave_shared(bool limit_hit, LimitReason why, bool gap_target_met) {
  if (nodes_explored_ > nodes_reported_) {
    (void)shared_->add_nodes(nodes_explored_ - nodes_reported_);
    nodes_reported_ = nodes_explored_;
  }
  // The same objective step in every worker: it is a property of the model (#221).
  shared_->set_objective_step(objective_step_);
  if (!open_.empty()) {
    double bound = std::numeric_limits<double>::infinity();
    for (const Index index : open_) {
      bound = std::min(bound, integral_bound(nodes_[static_cast<std::size_t>(index)].bound));
    }
    shared_->add_residual(bound, gap_target_met);
  }
  if (limit_hit && why != LimitReason::kNone) shared_->stop(why);
  sync_pseudocosts();
}

// =========================================================================================
// The driver
// =========================================================================================

int parallel_threads(const Model& model, const Options& options, Logger& logger) {
  auto threads = static_cast<int>(options.get_int("mip_threads"));
  if (threads == 0)
    threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  if (threads <= 1) return 1;
  const char* declined = nullptr;
  if (model.has_quadratic_objective()) {
    declined = "an MIQP's node relaxations are QPs, which the parallel search does not run";
  } else if (options.get_bool("pool_complete")) {
    declined = "pool_complete prunes on the pool's cutoff at every node";
  } else if (options.get_bool("deterministic")) {
    declined = "deterministic mode promises the same tree, and a parallel tree varies";
  } else if (!options.get_string("checkpoint").empty() ||
             !options.get_string("resume").empty()) {
    declined = "a checkpoint holds one search's tree, not several workers' subtrees";
  }
  if (declined != nullptr) {
    logger.info("mip_threads={} ignored: {}; searching on one thread", threads, declined);
    return 1;
  }
  if (!options.get_string("conflict_out").empty()) {
    logger.info(
        "mip_threads={}: conflict_out is not written by the parallel search; each worker's "
        "conflicts stay its own",
        threads);
  }
  return threads;
}

namespace {

void run_worker(const Model& model, const Options& options, SharedSearch* shared,
                SolveControl* stop_signal, const Timer& clock, double time_limit,
                Solution* root_answer) {
  Logger silent(nullptr);
  // OpenMP's thread count is per thread, and a new thread starts at the runtime's default of
  // one per core, not at what solve() set on the calling thread. Without this every column
  // loop in every worker forked a team of eight: on f2gap40400 one worker took 17 s where the
  // same search on the calling thread takes 2.
  {
    Options loops = options;
    loops.set_int("threads", 1);
    apply_thread_option(loops, silent);
  }
  SubtreeSpec spec;
  while (shared->take(&spec)) {
    try {
      Options worker_options = options;
      worker_options.set_bool("log_to_console", false);
      worker_options.set_int("threads", 1);      // one tree worker per core, not one per loop
      worker_options.set_int("node_limit", -1);  // the node limit is the shared count's
      // Every worker would write its own subtree's conflicts over the same file.
      worker_options.set_string("conflict_out", "");
      if (time_limit >= 0.0) {
        worker_options.set_double("time_limit",
                                  std::max(0.0, time_limit - clock.elapsed_seconds()));
      }
      BranchAndBound search(model, worker_options, silent, stop_signal);
      search.attach(shared, spec.is_root ? nullptr : &spec);
      Solution answer = search.run();
      if (answer.status == SolveStatus::kUnbounded ||
          answer.status == SolveStatus::kNumericalError ||
          answer.status == SolveStatus::kModelError) {
        shared->fail(std::move(answer));
      } else if (spec.is_root) {
        *root_answer = std::move(answer);
      }
    } catch (...) {
      // Out of memory, most likely. The first one is rethrown on the calling thread, where
      // the solve's out-of-memory guard turns it into a status (#246).
      shared->record_exception(std::current_exception());
    }
    shared->finished_subtree();
  }
  shared->worker_exited();
}

/// Joins the workers on every way out of the driver. The caller's progress callback runs on
/// the driver thread and may throw, and so may a thread's creation; an exception unwinding
/// past a joinable std::thread is std::terminate, not a status. On that path the workers are
/// told to stop first, so the join is short.
class WorkerJoiner {
 public:
  WorkerJoiner(std::vector<std::thread>* workers, SharedSearch* shared,
               SolveControl* stop_signal)
      : workers_(workers), shared_(shared), stop_signal_(stop_signal) {}
  WorkerJoiner(const WorkerJoiner&) = delete;
  WorkerJoiner& operator=(const WorkerJoiner&) = delete;
  ~WorkerJoiner() {
    if (joined_) return;
    shared_->stop(LimitReason::kInterrupt);
    stop_signal_->interrupt();
    join();
  }
  void join() {
    for (std::thread& worker : *workers_) {
      if (worker.joinable()) worker.join();
    }
    joined_ = true;
  }

 private:
  std::vector<std::thread>* workers_;
  SharedSearch* shared_;
  SolveControl* stop_signal_;
  bool joined_ = false;
};

/// The sequential search writes a node line every 20 nodes; the driver keeps the cadence.
constexpr Count kNodesPerProgressLine = 20;

}  // namespace

Solution solve_branch_and_bound_parallel(const Model& model, const Options& options,
                                         Logger& logger, SolveControl* control, int threads) {
  const Timer clock;
  const double sense = model.sense_multiplier();
  const auto reported = [&](double internal) {
    return sense * internal + model.objective_offset;
  };

  std::vector<Index> integer_columns;
  for (Index j = 0; j < model.num_cols(); ++j) {
    if (model.col_type[static_cast<std::size_t>(j)] == VarType::kInteger) {
      integer_columns.push_back(j);
    }
  }
  const double pool_gap = options.get_double("pool_gap");
  SharedSearch shared(
      threads, options.get_int("node_limit"),
      SolutionPool(integer_columns, static_cast<std::size_t>(options.get_int("pool_size")),
                   options.get_bool("pool_diversity")));
  const double time_limit = options.get_double("time_limit");
  const bool has_time_limit = std::isfinite(time_limit) && time_limit >= 0.0;

  logger.info("Branch and bound: parallel tree search on {} threads (#222)", threads);
  {
    Options quiet = options;
    quiet.set_bool("log_to_console", false);
    shared.set_scaling(build_node_scaling(model, quiet));
  }
  SubtreeSpec root;
  root.is_root = true;
  shared.push(std::move(root));

  // The workers answer to their own control; this thread forwards the caller's interrupt
  // and calls the caller's progress callback, which was never promised to be thread-safe.
  SolveControl stop_signal;
  Solution root_answer;
  std::vector<std::thread> workers;
  workers.reserve(static_cast<std::size_t>(threads));
  WorkerJoiner joiner(&workers, &shared, &stop_signal);
  for (int t = 0; t < threads; ++t) {
    workers.emplace_back(run_worker, std::cref(model), std::cref(options), &shared,
                         &stop_signal, std::cref(clock), has_time_limit ? time_limit : -1.0,
                         &root_answer);
  }
  // What this thread can see is the incumbent and the queue. The workers' open lists, and
  // so the bound and the gap, are theirs alone until they leave, and both are reported as
  // UNKNOWN - the weakest bound and an infinite gap, the sequential search's own convention
  // for "none yet" - never as a zero a reader would take for a closed gap (#289).
  const double unknown_bound = model.sense == ObjSense::kMaximize ? kInfinity : -kInfinity;
  Count last_logged = 0;
  bool logged_table = false;
  while (!shared.wait_until_done(std::chrono::milliseconds(50))) {
    const Count nodes = shared.nodes();
    const double best = shared.best();
    const double incumbent_report = std::isfinite(best) ? reported(best) : kInfinity;
    const auto queued = static_cast<Count>(shared.queued());
    if (nodes > 0 && (last_logged == 0 || nodes >= last_logged + kNodesPerProgressLine)) {
      if (!logged_table) {
        logger.begin_node_table();
        logged_table = true;
      }
      logger.node(nodes, queued, incumbent_report, unknown_bound, kInfinity,
                  clock.elapsed_seconds());
      last_logged = nodes;
    }
    bool interrupt = control != nullptr && control->interruption_requested();
    if (!interrupt && control != nullptr && control->progress_callback &&
        control->callback_due()) {
      Progress progress;
      progress.phase = Progress::Phase::kTree;
      progress.nodes = nodes;
      progress.open_nodes = queued;
      progress.objective = incumbent_report;
      progress.best_bound = unknown_bound;
      progress.gap = kInfinity;
      progress.elapsed_seconds = clock.elapsed_seconds();
      if (control->progress_callback(progress) != 0) {
        control->interrupt();
        interrupt = true;
      }
    }
    if (interrupt) {
      shared.stop(LimitReason::kInterrupt);
      stop_signal.interrupt();
    }
  }
  joiner.join();
  if (const std::exception_ptr error = shared.exception(); error) std::rethrow_exception(error);

  if (shared.failed()) {
    Solution failure = shared.failure();
    failure.nodes = shared.nodes();
    failure.solve_seconds = clock.elapsed_seconds();
    return failure;
  }

  // ---- The answer, from what every subtree left behind ---------------------------------
  Solution solution;
  solution.allocate_for(model);
  solution.algorithm = "branch-and-bound";
  solution.nodes = shared.nodes();
  solution.root_bound = root_answer.root_bound;
  solution.root_bound_after_cuts = root_answer.root_bound_after_cuts;
  solution.cuts_applied = root_answer.cuts_applied;

  const LimitReason why = shared.reason();
  const bool stopped = why != LimitReason::kNone;
  // Subtrees still queued carry raw node bounds; they are rounded with the step the workers
  // found, as the sequential search rounds every bound it reports (#221).
  const double residual = std::min(
      shared.residual(), round_up_to_step(shared.queued_bound(), shared.objective_step()));
  double best = std::numeric_limits<double>::infinity();
  std::vector<double> best_x;
  const bool found = shared.incumbent(&best, &best_x);
  const std::string summary =
      fmt::format("parallel tree search on {} threads: {} subtrees, {} given away", threads,
                  shared.subtrees.load(), shared.donations.load());

  if (!found) {
    const double nothing = model.sense == ObjSense::kMaximize ? -kInfinity : kInfinity;
    solution.objective = nothing;
    solution.absolute_gap = kInfinity;
    solution.relative_gap = kInfinity;
    if (stopped) {
      solution.status = status_for(why);
      solution.stopped_by = why;
      solution.dual_bound = reported(residual);
      solution.message =
          fmt::format("{} with no integer feasible point; {}", to_string(why), summary);
    } else {
      solution.status = SolveStatus::kInfeasible;
      solution.dual_bound = nothing;
      solution.message =
          fmt::format("the search closed with no integer feasible point after {} nodes; {}",
                      solution.nodes, summary);
    }
    solution.solve_seconds = clock.elapsed_seconds();
    logger.info("{}", solution.message);
    return solution;
  }

  solution.col_value = best_x;
  const double bound = std::min(best, residual);
  solution.dual_bound = reported(bound);
  if (!stopped) {
    // Every subtree closed, or stopped within the gap target against an incumbent no worse
    // than this one: proved, to the target.
    solution.status = SolveStatus::kOptimal;
    solution.message =
        shared.gap_met()
            ? fmt::format(
                  "optimal within the gap target after {} nodes; the bound is the best "
                  "open node's, the tree was not exhausted; {}",
                  solution.nodes, summary)
            : summary;
  } else {
    solution.status = SolveStatus::kFeasible;
    solution.stopped_by = why;
    solution.message = fmt::format("stopped by the {}; {}", to_string(why), summary);
  }
  solution.solve_seconds = clock.elapsed_seconds();
  solution.recompute_quality(model);

  SolutionPool& pool = shared.pool();
  if (pool.enabled()) {
    for (SolutionPool::Entry& entry : pool.finish(best, best_x, pool_gap)) {
      Solution::PoolEntry member;
      member.objective = model.evaluate_objective(entry.x.data());
      member.col_value = std::move(entry.x);
      solution.pool.push_back(std::move(member));
    }
    solution.pool.front().objective = solution.objective;
  }
  logger.info("Status: {}   objective {:.10g}   bound {:.10g}   nodes {}   time {:.3f}s",
              to_string(solution.status), solution.objective, solution.dual_bound,
              solution.nodes, solution.solve_seconds);
  logger.info("{}", solution.message);
  return solution;
}

}  // namespace sankhya::mip
