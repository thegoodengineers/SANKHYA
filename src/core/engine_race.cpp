// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the verifier-gated engine race (#476). See engine_race.hpp.

#include "core/engine_race.hpp"

#include <chrono>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "core/kkt_check.hpp"
#include "core/presolve_pipeline.hpp"
#include "core/resource_limits.hpp"
#include "core/status_guard.hpp"
#include "sankhya/certificate.hpp"
#include "solver_engine/solver_engine.hpp"
#include "solver_engine/solver_registry.hpp"
#include "util/threads.hpp"

namespace sankhya {

EngineRaceTestHooks& engine_race_hooks_for_testing() {
  static EngineRaceTestHooks hooks;
  return hooks;
}

bool engine_race_applies(const Options& options, bool warm_start) {
  return options.get_string("algorithm") == "auto" && options.get_bool("engine_race") &&
         !warm_start;
}

namespace {

/// One entrant: the registry's engine and the name the log gives it.
struct Racer {
  std::string name;
  const engine::SolverEngine* engine = nullptr;
};

/// The entrants, in the FIXED order the sequential mode runs them: the dual simplex (the
/// measured default below 20,000 rows), the interior point with crossover, then PDHG - on the
/// GPU when the build has one, whose engine falls back to the CPU on its own when the device
/// or the model does not suit it.
std::vector<Racer> entrants(const Model& model) {
  const engine::SolverRegistry& registry = engine::SolverRegistry::builtin();
  const std::string pdhg = registry.find("pdhg-gpu") != nullptr ? "pdhg-gpu" : "pdhg";
  std::vector<Racer> racers;
  for (const std::string& name : {std::string("dual-simplex"), std::string("ipm"), pdhg}) {
    const engine::SolverEngine* found = registry.find(name);
    if (found != nullptr && found->supports(model)) racers.push_back({name, found});
  }
  return racers;
}

enum class Verdict { kPending, kAccepted, kRejected, kNoClaim, kNotRun };

struct Entry {
  Solution answer;
  Verdict verdict = Verdict::kPending;
  std::string why;
  double finished_at = 0.0;
};

/// The gate. An optimal answer must pass the KKT check against the ORIGINAL model at the
/// solve's own tolerances; an infeasibility verdict must carry a Farkas certificate that
/// checks out against it (either sign, as verify_and_keep_certificate reads one). Every other
/// status makes no claim a check could accept, so it cannot win - an unbounded verdict
/// included, whose ray proves unboundedness only beside a feasible point.
Verdict judge(const Model& model, const Options& options, const Solution& answer,
              std::string* why) {
  if (answer.status == SolveStatus::kOptimal) {
    KktTolerances tolerances;
    tolerances.primal = options.get_double("primal_feasibility_tolerance");
    tolerances.dual = options.get_double("dual_feasibility_tolerance");
    const KktVerdict verdict = check_lp_optimality(model, answer, tolerances);
    if (verdict.passed) return Verdict::kAccepted;
    *why = fmt::format("{}: {}", verdict.check, verdict.detail);
    return Verdict::kRejected;
  }
  if (answer.status == SolveStatus::kInfeasible) {
    if (answer.farkas_dual.empty()) {
      *why = "no infeasibility certificate to check";
      return Verdict::kRejected;
    }
    std::vector<double> flipped = answer.farkas_dual;
    for (double& value : flipped) value = -value;
    if (farkas_proves_infeasible(model, answer.farkas_dual, why) ||
        farkas_proves_infeasible(model, flipped, why)) {
      return Verdict::kAccepted;
    }
    return Verdict::kRejected;
  }
  return Verdict::kNoClaim;
}

/// One engine, start to finish: presolve, the engine on what the time limit has left,
/// postsolve, all on a logger of its own, inside the out-of-memory guard that turns an
/// exhausted allocation into this engine's declined status (#437). Anything else thrown is
/// this engine's failure too: on a worker thread an escaping exception is std::terminate.
Solution run_racer(const Racer& racer, const Model& model, const Options& options,
                   const Timer& timer, SolveControl* control) {
  Logger silent(nullptr);
  Options racer_options = options;
  racer_options.set_bool("log_to_console", false);
  racer_options.set_string("progress_out", "");  // one file cannot take three engines' lines
  const ResourceLimits limits(options, silent);
  const std::function<Solution(const Model&)> run_engine = [&](const Model& target) {
    Options engine_options = racer_options;
    if (limits.has_time_limit()) {
      engine_options.set_double("time_limit",
                                limits.remaining_seconds(timer.elapsed_seconds()));
    }
    return racer.engine->solve_verified(target, engine_options, silent, control);
  };
  // The test seam runs INSIDE the guards, so a test can throw std::bad_alloc from it and see
  // what an engine that runs out of memory does to the race.
  const EngineRaceTestHooks& hooks = engine_race_hooks_for_testing();
  try {
    return run_declining_on_out_of_memory(
        [&] {
          if (hooks.before_engine) hooks.before_engine(racer.name);
          return run_with_presolve(model, racer_options, silent, timer,
                                   engine::ProblemClass::kLp, run_engine)
              .solution;
        },
        racer.name, timer, silent);
  } catch (const std::exception& error) {
    Solution failed;
    failed.status = SolveStatus::kNumericalError;
    failed.algorithm = racer.name;
    failed.message = fmt::format("the {} threw: {}", racer.name, error.what());
    return failed;
  } catch (...) {
    Solution failed;
    failed.status = SolveStatus::kNumericalError;
    failed.algorithm = racer.name;
    failed.message =
        fmt::format("the {} threw an exception that is not a std::exception", racer.name);
    return failed;
  }
}

/// Everything the entrants' threads share, behind one mutex.
struct Track {
  std::mutex mutex;
  std::condition_variable changed;
  std::vector<Entry> entries;
  std::size_t finished = 0;
  int winner = -1;
};

void race_one(std::size_t k, const Racer& racer, const Model& model, const Options& options,
              const Timer& timer, SolveControl* race_control, Track* track) {
  Logger silent(nullptr);
  // OpenMP's team size is per thread and a new thread starts at one per core (#222).
  apply_thread_option(options, silent);
  const EngineRaceTestHooks& hooks = engine_race_hooks_for_testing();
  Solution answer = run_racer(racer, model, options, timer, race_control);
  if (hooks.after_engine) hooks.after_engine(racer.name, &answer);
  std::string why;
  const Verdict verdict = judge(model, options, answer, &why);
  const double at = timer.elapsed_seconds();
  {
    const std::lock_guard<std::mutex> lock(track->mutex);
    Entry& entry = track->entries[k];
    entry.answer = std::move(answer);
    entry.verdict = verdict;
    entry.why = std::move(why);
    entry.finished_at = at;
    ++track->finished;
    if (verdict == Verdict::kAccepted && track->winner < 0) {
      track->winner = static_cast<int>(k);
      race_control->interrupt();  // the others stop at their next safe point
    }
  }
  track->changed.notify_all();
}

/// Joins the entrants on every way out, as the parallel tree search's joiner does (#222): the
/// caller's progress callback runs on this thread and may throw, and an exception unwinding
/// past a joinable std::thread is std::terminate. On that path the entrants are stopped first.
class Joiner {
 public:
  Joiner(std::vector<std::thread>* threads, SolveControl* race_control)
      : threads_(threads), race_control_(race_control) {}
  Joiner(const Joiner&) = delete;
  Joiner& operator=(const Joiner&) = delete;
  ~Joiner() {
    if (joined_) return;
    race_control_->interrupt();
    join();
  }
  void join() {
    for (std::thread& thread : *threads_) {
      if (thread.joinable()) thread.join();
    }
    joined_ = true;
  }

 private:
  std::vector<std::thread>* threads_;
  SolveControl* race_control_;
  bool joined_ = false;
};

void race_concurrently(const std::vector<Racer>& racers, const Model& model,
                       const Options& options, const Timer& timer, SolveControl* control,
                       Track* track) {
  SolveControl race_control;
  std::vector<std::thread> threads;
  threads.reserve(racers.size());
  Joiner joiner(&threads, &race_control);
  for (std::size_t k = 0; k < racers.size(); ++k) {
    threads.emplace_back(race_one, k, std::cref(racers[k]), std::cref(model),
                         std::cref(options), std::cref(timer), &race_control, track);
  }
  // The caller's interrupt and progress callback are this thread's, never an entrant's: the
  // callback was never promised to be thread-safe (#222).
  std::unique_lock<std::mutex> lock(track->mutex);
  while (track->finished < racers.size()) {
    track->changed.wait_for(lock, std::chrono::milliseconds(50));
    lock.unlock();
    bool interrupt = control != nullptr && control->interruption_requested();
    if (!interrupt && control != nullptr && control->progress_callback &&
        control->callback_due()) {
      Progress progress;
      progress.phase = Progress::Phase::kLp;
      progress.elapsed_seconds = timer.elapsed_seconds();
      if (control->progress_callback(progress) != 0) {
        control->interrupt();
        interrupt = true;
      }
    }
    if (interrupt) race_control.interrupt();
    lock.lock();
  }
  lock.unlock();
  joiner.join();
}

void race_in_order(const std::vector<Racer>& racers, const Model& model, const Options& options,
                   const Timer& timer, SolveControl* control, Track* track) {
  for (std::size_t k = 0; k < racers.size(); ++k) {
    Entry& entry = track->entries[k];
    if (track->winner >= 0) {
      entry.verdict = Verdict::kNotRun;
      continue;
    }
    const EngineRaceTestHooks& hooks = engine_race_hooks_for_testing();
    entry.answer = run_racer(racers[k], model, options, timer, control);
    if (hooks.after_engine) hooks.after_engine(racers[k].name, &entry.answer);
    entry.verdict = judge(model, options, entry.answer, &entry.why);
    entry.finished_at = timer.elapsed_seconds();
    ++track->finished;
    if (entry.verdict == Verdict::kAccepted) track->winner = static_cast<int>(k);
  }
}

std::string describe(const Racer& racer, const Entry& entry, bool won, bool someone_won) {
  switch (entry.verdict) {
    case Verdict::kNotRun:
      return fmt::format("{}: not run, an earlier engine's answer was accepted", racer.name);
    case Verdict::kAccepted:
      return fmt::format("{}: {} at {:.2f}s, checked and accepted{}", racer.name,
                         to_string(entry.answer.status), entry.finished_at,
                         won ? " - the winner" : ", after the winner");
    case Verdict::kRejected:
      return fmt::format(
          "{}: {} at {:.2f}s, REJECTED by the check ({}); it did not win and did "
          "not stop the others",
          racer.name, to_string(entry.answer.status), entry.finished_at, entry.why);
    case Verdict::kNoClaim:
    case Verdict::kPending: break;
  }
  if (someone_won && entry.answer.status == SolveStatus::kInterrupted) {
    return fmt::format("{}: cancelled at {:.2f}s when the winner was accepted", racer.name,
                       entry.finished_at);
  }
  return fmt::format("{}: {} at {:.2f}s, nothing a check could accept", racer.name,
                     to_string(entry.answer.status), entry.finished_at);
}

}  // namespace

Solution run_engine_race(const Model& model, const Options& options, SolveControl* control,
                         Logger& logger, const Timer& timer, const std::string& rule_engine,
                         std::string* engine_ran, bool* accepted) {
  *accepted = false;
  const std::vector<Racer> racers = entrants(model);
  Track track;
  track.entries.resize(racers.size());
  const unsigned cores = std::thread::hardware_concurrency();
  const bool in_order = options.get_bool("deterministic") || cores <= 1;
  logger.info("Engine race (#476): {} {}", racers.size(),
              in_order ? (options.get_bool("deterministic")
                              ? "engines in a fixed order, deterministic mode"
                              : "engines in a fixed order, one core")
                       : "engines at once, the first answer that passes its check wins");
  if (racers.empty()) {
    Solution none;
    none.allocate_for(model);
    none.status = SolveStatus::kNotSolved;
    none.algorithm = "none";
    none.message = "engine race: no registered LP engine accepts this model";
    return none;
  }
  if (in_order) {
    race_in_order(racers, model, options, timer, control, &track);
  } else {
    race_concurrently(racers, model, options, timer, control, &track);
  }

  std::vector<std::string> log;
  log.reserve(racers.size());
  for (std::size_t k = 0; k < racers.size(); ++k) {
    log.push_back(describe(racers[k], track.entries[k], static_cast<int>(k) == track.winner,
                           track.winner >= 0));
    logger.info("Engine race: {}", log.back());
  }
  std::string race_log;
  for (const std::string& line : log) race_log += (race_log.empty() ? "" : "; ") + line;

  Solution answer;
  if (track.winner >= 0) {
    const auto k = static_cast<std::size_t>(track.winner);
    answer = std::move(track.entries[k].answer);
    *engine_ran = racers[k].name;
    *accepted = true;
    answer.message = fmt::format("{}{}engine race (#476), {} won: {}", answer.message,
                                 answer.message.empty() ? "" : "; ", racers[k].name, race_log);
    return answer;
  }

  // NO ANSWER PASSED ITS CHECK: the rule table's engine's answer is returned, as the solve
  // without the race would have returned it - except that an optimal claim the check rejected
  // is not reported as one. A point outside its bounds or rows is no point; one that is
  // feasible but whose duals failed is a feasible point, not a proof.
  std::size_t fallback = 0;
  const auto is_pdhg = [](const std::string& name) { return name.rfind("pdhg", 0) == 0; };
  for (std::size_t k = 0; k < racers.size(); ++k) {
    if (racers[k].name == rule_engine || (is_pdhg(racers[k].name) && is_pdhg(rule_engine))) {
      fallback = k;
    }
  }
  Entry& entry = track.entries[fallback];
  answer = std::move(entry.answer);
  *engine_ran = racers[fallback].name;
  if (answer.status == SolveStatus::kOptimal && entry.verdict == Verdict::kRejected) {
    const bool primal = entry.why.rfind("structure", 0) == 0 ||
                        entry.why.rfind("column bounds", 0) == 0 ||
                        entry.why.rfind("row activity", 0) == 0;
    answer.status = primal ? SolveStatus::kNumericalError : SolveStatus::kFeasible;
    answer.message = fmt::format(
        "{}{}the in-process check rejected this optimal claim ({}), so "
        "it is reported {}",
        answer.message, answer.message.empty() ? "" : "; ", entry.why,
        to_string(answer.status));
  }
  answer.message = fmt::format(
      "{}{}engine race (#476): no answer passed its check, so {}'s, the "
      "rule table's choice, is reported: {}",
      answer.message, answer.message.empty() ? "" : "; ", racers[fallback].name, race_log);
  return answer;
}

}  // namespace sankhya
