// SPDX-License-Identifier: Apache-2.0
// SANKHYA - Benders decomposition of an LP over linking columns (#525). See benders.hpp for the
// method, the two kinds of cut and why the answer is measured rather than trusted.

#include "decomp/benders.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "sankhya/sparse.hpp"
#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::decomp {
namespace {

constexpr std::size_t at(Index i) {
  return static_cast<std::size_t>(i);
}

/// One block: its rows and owned columns as a small LP, and how its rows touch the linking
/// columns.
struct Block {
  std::vector<Index> rows;     ///< global rows, ascending
  std::vector<Index> columns;  ///< global columns it owns, ascending
  Model sub;                   ///< rows x owned columns, the model's own row bounds
  /// Per local row, the entries on linking columns: (slot, value).
  std::vector<Index> couple_start, couple_slot;
  std::vector<double> couple_value;
  /// Row bounds with the coupling relaxed over the linking columns' own bounds.
  std::vector<double> relaxed_lower, relaxed_upper;
};

struct Setup {
  std::vector<Block> blocks;
  std::vector<Index> slot_column;  ///< linking slot -> global column (linking, then unattached)
  std::vector<Index> master_rows;  ///< global rows whose columns are all linking
  /// The master's own rows, as (master row, slot, value).
  std::vector<Index> master_entry_row, master_entry_slot;
  std::vector<double> master_entry_value;
};

/// A cut as the master holds it, with the block duals it was built from (for the dual vector).
struct Cut {
  Index block = 0;
  bool feasibility = false;
  double rhs = 0.0;
  std::vector<Index> slots;
  std::vector<double> slopes;
  std::vector<double> dual;
};

struct BlockAnswer {
  enum class Kind { kOptimal, kInfeasible, kFailed } kind = Kind::kFailed;
  double value = 0.0;  ///< Q_b, or V_b (the least total violation) when infeasible
  std::vector<double> x;
  std::vector<double> dual;
  std::string failure;
  LimitReason limit = LimitReason::kNone;
};

/// The shift a row's linking entries make at `y`.
double coupling(const Block& block, Index row, const std::vector<double>& y) {
  double s = 0.0;
  for (Index k = block.couple_start[at(row)]; k < block.couple_start[at(row) + 1]; ++k) {
    s += block.couple_value[at(k)] * y[at(block.couple_slot[at(k)])];
  }
  return s;
}

double shifted(double bound, double shift) {
  return is_finite_bound(bound) ? bound - shift : bound;
}

/// Splits `model` along `structure`, checking the structure instead of trusting it: every
/// column a block owns must lie in that block's rows only. Returns the reason on failure.
std::string prepare(const Model& model, const BlockStructure& structure, Setup* out) {
  const Index m = model.num_rows();
  const Index n = model.num_cols();
  const Index blocks = structure.blocks;
  out->blocks.assign(at(blocks), Block{});
  std::vector<Index> row_local(at(m), -1);
  for (Index i = 0; i < m; ++i) {
    const Index b = structure.row_block[at(i)];
    if (b >= 0) {
      row_local[at(i)] = static_cast<Index>(out->blocks[at(b)].rows.size());
      out->blocks[at(b)].rows.push_back(i);
    } else if (b == kLinking) {
      row_local[at(i)] = static_cast<Index>(out->master_rows.size());
      out->master_rows.push_back(i);
    } else if (model.row_lower[at(i)] > 0.0 || model.row_upper[at(i)] < 0.0) {
      return fmt::format("row {} is empty and excludes zero", i);
    }
  }
  std::vector<Index> column_slot(at(n), -1);
  std::vector<Index> column_local(at(n), -1);
  for (Index j = 0; j < n; ++j) {
    const Index b = structure.col_block[at(j)];
    if (b >= 0) {
      column_local[at(j)] = static_cast<Index>(out->blocks[at(b)].columns.size());
      out->blocks[at(b)].columns.push_back(j);
    } else {
      column_slot[at(j)] = static_cast<Index>(out->slot_column.size());
      out->slot_column.push_back(j);
    }
  }
  struct Triplet {
    Index row, col;
    double value;
  };
  std::vector<std::vector<Triplet>> sub_entries(at(blocks)), couple(at(blocks));
  for (Index j = 0; j < n; ++j) {
    const Index owner = structure.col_block[at(j)];
    const ColumnView column = model.matrix.column(j);
    for (Index k = 0; k < column.size; ++k) {
      if (column.values[k] == 0.0) continue;
      const Index i = column.rows[k];
      const Index rb = structure.row_block[at(i)];
      if (owner >= 0) {
        if (rb != owner)
          return fmt::format("column {} is owned by block {} but is in row {}", j, owner, i);
        sub_entries[at(owner)].push_back(
            {row_local[at(i)], column_local[at(j)], column.values[k]});
      } else if (rb >= 0) {
        couple[at(rb)].push_back({row_local[at(i)], column_slot[at(j)], column.values[k]});
      } else if (rb == kLinking) {
        out->master_entry_row.push_back(row_local[at(i)]);
        out->master_entry_slot.push_back(column_slot[at(j)]);
        out->master_entry_value.push_back(column.values[k]);
      } else {
        return fmt::format("row {} has an entry but no block", i);
      }
    }
  }
  for (Index b = 0; b < blocks; ++b) {
    Block& block = out->blocks[at(b)];
    const Index rows = static_cast<Index>(block.rows.size());
    const Index cols = static_cast<Index>(block.columns.size());
    Model sub;
    sub.resize_columns(cols);
    sub.resize_rows(rows);
    sub.sense = ObjSense::kMinimize;
    for (Index c = 0; c < cols; ++c) {
      const auto g = at(block.columns[at(c)]);
      sub.col_cost[at(c)] = model.col_cost[g];
      sub.col_lower[at(c)] = model.col_lower[g];
      sub.col_upper[at(c)] = model.col_upper[g];
    }
    for (Index r = 0; r < rows; ++r) {
      sub.row_lower[at(r)] = model.row_lower[at(block.rows[at(r)])];
      sub.row_upper[at(r)] = model.row_upper[at(block.rows[at(r)])];
    }
    SparseMatrix matrix(rows, cols);
    for (const Triplet& t : sub_entries[at(b)]) matrix.add_entry(t.row, t.col, t.value);
    matrix.finalize(0.0);
    sub.matrix = std::move(matrix);
    block.sub = std::move(sub);

    block.couple_start.assign(at(rows) + 1, 0);
    for (const Triplet& t : couple[at(b)]) ++block.couple_start[at(t.row) + 1];
    for (Index r = 0; r < rows; ++r) block.couple_start[at(r) + 1] += block.couple_start[at(r)];
    block.couple_slot.assign(couple[at(b)].size(), 0);
    block.couple_value.assign(couple[at(b)].size(), 0.0);
    std::vector<Index> next(block.couple_start.begin(), block.couple_start.end() - 1);
    for (const Triplet& t : couple[at(b)]) {
      const auto at_k = at(next[at(t.row)]++);
      block.couple_slot[at_k] = t.col;
      block.couple_value[at_k] = t.value;
    }
    block.relaxed_lower.assign(at(rows), 0.0);
    block.relaxed_upper.assign(at(rows), 0.0);
    for (Index r = 0; r < rows; ++r) {
      double smin = 0.0;
      double smax = 0.0;
      for (Index k = block.couple_start[at(r)]; k < block.couple_start[at(r) + 1]; ++k) {
        const auto g = at(out->slot_column[at(block.couple_slot[at(k)])]);
        const double a = block.couple_value[at(k)];
        smin += a * (a > 0.0 ? model.col_lower[g] : model.col_upper[g]);
        smax += a * (a > 0.0 ? model.col_upper[g] : model.col_lower[g]);
      }
      const auto g = at(block.rows[at(r)]);
      block.relaxed_lower[at(r)] = shifted(model.row_lower[g], smax);
      block.relaxed_upper[at(r)] = shifted(model.row_upper[g], smin);
    }
  }
  return {};
}

/// The LP with the coupling at `y`: rows shifted, columns the block's own. `elastic` adds a
/// surplus and a slack to every row and minimises their total.
Model block_model(const Block& block, const std::vector<double>& y, bool elastic) {
  Model m = block.sub;
  const Index rows = m.num_rows();
  for (Index r = 0; r < rows; ++r) {
    const double s = coupling(block, r, y);
    m.row_lower[at(r)] = shifted(m.row_lower[at(r)], s);
    m.row_upper[at(r)] = shifted(m.row_upper[at(r)], s);
  }
  if (!elastic) return m;
  const Index cols = m.num_cols();
  m.resize_columns(cols + 2 * rows);
  std::fill(m.col_cost.begin(), m.col_cost.end(), 0.0);
  SparseMatrix matrix(rows, cols + 2 * rows);
  for (Index j = 0; j < cols; ++j) {
    const ColumnView column = block.sub.matrix.column(j);
    for (Index k = 0; k < column.size; ++k)
      matrix.add_entry(column.rows[k], j, column.values[k]);
  }
  for (Index r = 0; r < rows; ++r) {
    for (const Index k : {cols + r, cols + rows + r}) {
      m.col_cost[at(k)] = 1.0;
      m.col_lower[at(k)] = 0.0;
      m.col_upper[at(k)] = kInfinity;
    }
    matrix.add_entry(r, cols + r, 1.0);
    matrix.add_entry(r, cols + rows + r, -1.0);
  }
  matrix.finalize(0.0);
  m.matrix = std::move(matrix);
  return m;
}

BlockAnswer solve_block(const Block& block, const std::vector<double>& y,
                        const Options& options, SolveControl* control) {
  BlockAnswer answer;
  const Index rows = block.sub.num_rows();
  const auto stopped = [&](const Solution& s, BlockAnswer* a) {
    if (s.status == SolveStatus::kTimeLimit || s.status == SolveStatus::kInterrupted) {
      a->limit =
          s.status == SolveStatus::kTimeLimit ? LimitReason::kTime : LimitReason::kInterrupt;
      a->failure = "stopped by a limit";
      return true;
    }
    return false;
  };
  try {
    const Solution s = solve(block_model(block, y, false), options, control);
    if (s.status == SolveStatus::kOptimal) {
      if (static_cast<Index>(s.row_dual.size()) != rows) {
        answer.failure = "a block LP returned no row duals";
        return answer;
      }
      answer.kind = BlockAnswer::Kind::kOptimal;
      answer.value = s.objective;
      answer.x = s.col_value;
      answer.dual = s.row_dual;
      return answer;
    }
    if (stopped(s, &answer)) return answer;
    if (s.status == SolveStatus::kUnbounded) {
      answer.failure = "a block LP is unbounded";
      return answer;
    }
    // Infeasible, or an engine that could not say so with a proof (numerical_error): either
    // way the elastic LP is the test. It is always feasible, and its optimum being positive is
    // itself the proof that the block is empty at y; being zero means the block LP was
    // feasible and the engine failed, which is a failure of this method and no cut.
    const Solution e = solve(block_model(block, y, true), options, control);
    if (stopped(e, &answer)) return answer;
    if (e.status != SolveStatus::kOptimal || static_cast<Index>(e.row_dual.size()) != rows) {
      answer.failure =
          fmt::format("the elastic LP of an infeasible block ended {}", to_string(e.status));
      return answer;
    }
    if (e.objective <= tol::kBendersFeasibilityCut) {
      answer.failure = fmt::format("a block LP ended {} but its elastic LP has no violation",
                                   to_string(s.status));
      return answer;
    }
    answer.kind = BlockAnswer::Kind::kInfeasible;
    answer.value = e.objective;
    answer.dual = e.row_dual;
    return answer;
  } catch (const std::exception& error) {
    answer.failure = fmt::format("a block solve threw: {}", error.what());
    return answer;
  }
}

/// The subgradient of a block's value in the linking slots: g = - A_{b,y}' lambda.
void slopes_of(const Block& block, const std::vector<double>& dual, std::vector<double>* g,
               Index slots) {
  g->assign(at(slots), 0.0);
  for (Index r = 0; r < block.sub.num_rows(); ++r) {
    for (Index k = block.couple_start[at(r)]; k < block.couple_start[at(r) + 1]; ++k) {
      (*g)[at(block.couple_slot[at(k)])] -= block.couple_value[at(k)] * dual[at(r)];
    }
  }
}

Model build_master(const Model& model, const Setup& setup, const std::vector<char>& has_cut,
                   const std::vector<double>& theta_lower, const std::vector<Cut>& cuts,
                   bool free_theta) {
  const Index slots = static_cast<Index>(setup.slot_column.size());
  const Index blocks = static_cast<Index>(setup.blocks.size());
  const Index master_rows = static_cast<Index>(setup.master_rows.size());
  const Index total = master_rows + static_cast<Index>(cuts.size());
  Model master;
  master.sense = ObjSense::kMinimize;
  master.objective_offset = model.objective_offset;
  master.resize_columns(slots + blocks);
  master.resize_rows(total);
  for (Index s = 0; s < slots; ++s) {
    const auto g = at(setup.slot_column[at(s)]);
    master.col_cost[at(s)] = model.col_cost[g];
    master.col_lower[at(s)] = model.col_lower[g];
    master.col_upper[at(s)] = model.col_upper[g];
  }
  for (Index b = 0; b < blocks; ++b) {
    const auto t = at(slots + b);
    master.col_cost[t] = 1.0;
    // The start bound is a valid lower bound on the block's value for every y, so keeping it is
    // always a relaxation - and it keeps the master bounded while a linking column is unbounded
    // and no cut yet says which way the block's value turns. It is dropped only in the last
    // rounds, for blocks that have an optimality cut: a free epigraph column is what makes the
    // multipliers on the block's cuts sum to one at a master optimum, which the dual vector
    // read from them needs.
    master.col_lower[t] = (free_theta && has_cut[at(b)] != 0) ? -kInfinity : theta_lower[at(b)];
    master.col_upper[t] = kInfinity;
  }
  for (Index k = 0; k < master_rows; ++k) {
    const auto g = at(setup.master_rows[at(k)]);
    master.row_lower[at(k)] = model.row_lower[g];
    master.row_upper[at(k)] = model.row_upper[g];
  }
  SparseMatrix matrix(total, slots + blocks);
  for (std::size_t k = 0; k < setup.master_entry_row.size(); ++k) {
    matrix.add_entry(setup.master_entry_row[k], setup.master_entry_slot[k],
                     setup.master_entry_value[k]);
  }
  for (std::size_t c = 0; c < cuts.size(); ++c) {
    const Index row = master_rows + static_cast<Index>(c);
    const Cut& cut = cuts[c];
    if (!cut.feasibility) {
      matrix.add_entry(row, slots + cut.block, 1.0);  // theta_b - g'y >= rhs
      for (std::size_t q = 0; q < cut.slots.size(); ++q) {
        matrix.add_entry(row, cut.slots[q], -cut.slopes[q]);
      }
      master.row_lower[at(row)] = cut.rhs;
    } else {
      for (std::size_t q = 0; q < cut.slots.size(); ++q) {
        matrix.add_entry(row, cut.slots[q], cut.slopes[q]);  // g'y <= rhs
      }
      master.row_upper[at(row)] = cut.rhs;
    }
  }
  matrix.finalize(0.0);
  master.matrix = std::move(matrix);
  return master;
}

Options sub_options(const Options& options, double seconds_left) {
  Options sub = options;
  sub.set_string("decomposition", "off");
  sub.set_bool("log_to_console", false);
  sub.set_int("threads", 1);
  // The algorithm is the user's (`auto` by default), exactly as for the monolithic solve: a
  // forced dual simplex skips the robust default pipeline and was an order of magnitude slower
  // on the refinery ladder's blocks. A first-order engine's inexact multipliers give inexact
  // cuts, and the measurement of the answer is what catches that.
  sub.set_double("time_limit", seconds_left);
  return sub;
}

Solution limit_solution(LimitReason reason, Count iterations, double seconds) {
  Solution s;
  s.algorithm = "benders";
  s.status = status_for(reason);
  s.stopped_by = reason;
  s.iterations = iterations;
  s.solve_seconds = seconds;
  s.message = fmt::format("stopped by the {} during Benders decomposition after {} round(s)",
                          to_string(reason), iterations);
  return s;
}

}  // namespace

BendersOutcome solve_benders(const Model& model, const BlockStructure& structure,
                             const Options& options, SolveControl* control, Logger& logger) {
  BendersOutcome outcome;
  const Timer timer;
  const auto decline = [&](const std::string& why) {
    outcome.declined = why;
    return outcome;
  };
  if (structure.kind != Linking::kColumns)
    return decline("only linking columns are decomposed");
  Setup setup;
  if (const std::string bad = prepare(model, structure, &setup); !bad.empty()) {
    return decline("the structure does not hold: " + bad);
  }
  const Index blocks = static_cast<Index>(setup.blocks.size());
  const Index slots = static_cast<Index>(setup.slot_column.size());
  const double time_limit = options.get_double("time_limit");
  // Half of a finite limit is all the decomposition may spend. A method that has not converged
  // by then declines, and the monolithic engines get the rest; one that used every second and
  // then declined would leave them none (measured on the refinery ladder's largest model).
  const bool limited = time_limit < std::numeric_limits<double>::max();
  const double budget = limited ? 0.5 * time_limit : time_limit;
  const Count max_rounds = options.get_int("decomposition_max_iterations");
  const double gap_target = tol::kBendersGap;
  Index threads = static_cast<Index>(options.get_int("decomposition_threads"));
  if (threads <= 0)
    threads = std::max<Index>(1, static_cast<Index>(std::thread::hardware_concurrency()));
  threads = std::min(threads, blocks);
  const auto left = [&] { return std::max(0.0, budget - timer.elapsed_seconds()); };
  const auto stop_reason = [&]() {
    if (control != nullptr && control->interruption_requested()) return LimitReason::kInterrupt;
    if (timer.elapsed_seconds() > budget) return LimitReason::kTime;
    return LimitReason::kNone;
  };
  // An interrupt is final; the budget running out is a decline.
  const auto stopped_outcome = [&](LimitReason reason, Count rounds) {
    if (reason == LimitReason::kInterrupt) {
      outcome.solution = limit_solution(reason, rounds, timer.elapsed_seconds());
      return outcome;
    }
    return decline(fmt::format(
        "not converged after {} round(s) in the {:.1f} s it is given (half of the {:.1f} s "
        "limit), so the monolithic engines get the rest",
        rounds, budget, time_limit));
  };

  // The start: each block's value over every y within the linking columns' own bounds.
  std::vector<double> theta_lower(at(blocks), 0.0);
  for (Index b = 0; b < blocks; ++b) {
    Model relaxed = setup.blocks[at(b)].sub;
    relaxed.row_lower = setup.blocks[at(b)].relaxed_lower;
    relaxed.row_upper = setup.blocks[at(b)].relaxed_upper;
    const Solution s = solve(relaxed, sub_options(options, left()), control);
    if (s.status == SolveStatus::kTimeLimit || s.status == SolveStatus::kInterrupted) {
      return stopped_outcome(
          s.status == SolveStatus::kTimeLimit ? LimitReason::kTime : LimitReason::kInterrupt,
          0);
    }
    if (s.status != SolveStatus::kOptimal) {
      return decline(
          fmt::format("block {}'s relaxed LP ended {}, so there is no starting bound", b,
                      to_string(s.status)));
    }
    theta_lower[at(b)] = s.objective - tol::kBendersGap * std::max(1.0, std::fabs(s.objective));
  }

  std::vector<Cut> cuts;
  std::vector<char> has_cut(at(blocks), 0);
  bool free_theta = false;
  Count rounds = 0;
  while (true) {
    if (const LimitReason reason = stop_reason(); reason != LimitReason::kNone) {
      return stopped_outcome(reason, rounds);
    }
    if (rounds >= max_rounds)
      return decline(fmt::format("no convergence in {} rounds", rounds));
    ++rounds;

    const Model master = build_master(model, setup, has_cut, theta_lower, cuts, free_theta);
    const Solution ms = solve(master, sub_options(options, left()), control);
    if (ms.status == SolveStatus::kTimeLimit || ms.status == SolveStatus::kInterrupted) {
      return stopped_outcome(
          ms.status == SolveStatus::kTimeLimit ? LimitReason::kTime : LimitReason::kInterrupt,
          rounds);
    }
    if (ms.status != SolveStatus::kOptimal) {
      return decline(fmt::format("the master{} ended {}",
                                 free_theta ? " with free epigraph columns" : "",
                                 to_string(ms.status)));
    }
    const Index master_rows = static_cast<Index>(setup.master_rows.size());
    if (static_cast<Index>(ms.row_dual.size()) !=
            master_rows + static_cast<Index>(cuts.size()) ||
        static_cast<Index>(ms.col_value.size()) != slots + blocks) {
      return decline("the master returned no duals");
    }
    const double lower = ms.objective;
    const std::vector<double> y(ms.col_value.begin(), ms.col_value.begin() + slots);

    // The blocks at this y, in parallel; the answers are read in block order, so the cuts and
    // the result do not depend on which thread finished first.
    std::vector<BlockAnswer> answers(at(blocks));
    {
      const Options opts = sub_options(options, left());
      std::atomic<Index> next{0};
      const auto work = [&] {
        for (Index b = next.fetch_add(1); b < blocks; b = next.fetch_add(1)) {
          answers[at(b)] = solve_block(setup.blocks[at(b)], y, opts, control);
        }
      };
      if (threads <= 1) {
        work();
      } else {
        std::vector<std::thread> pool;
        for (Index w = 0; w < threads; ++w) pool.emplace_back(work);
        for (std::thread& t : pool) t.join();
      }
    }
    bool all_optimal = true;
    double upper = model.objective_offset;
    for (Index s = 0; s < slots; ++s)
      upper += model.col_cost[at(setup.slot_column[at(s)])] * y[at(s)];
    for (Index b = 0; b < blocks; ++b) {
      const BlockAnswer& a = answers[at(b)];
      if (a.limit != LimitReason::kNone) return stopped_outcome(a.limit, rounds);
      if (a.kind == BlockAnswer::Kind::kFailed) return decline(a.failure);
      if (a.kind == BlockAnswer::Kind::kInfeasible) {
        all_optimal = false;
      } else {
        upper += a.value;
      }
    }

    if (all_optimal) {
      logger.verbose("Benders round {}: bound {:.12g}, value at y {:.12g}, {} cut(s)", rounds,
                     lower, upper, cuts.size());
      const double gap = upper - lower;
      if (gap < -gap_target * std::max(1.0, std::fabs(upper))) {
        return decline(fmt::format(
            "the master's bound {:.12g} exceeds a feasible value {:.12g}: a cut was not valid",
            lower, upper));
      }
      if (gap <= gap_target * std::max(1.0, std::fabs(upper))) {
        // The dual vector below needs every block's cut multipliers to sum to one, which a free
        // epigraph column guarantees - and a block that never needed a cut (it does not touch
        // the linking columns, or its starting bound was already its value) still has the
        // bounded one. Give it its cut, a satisfied one, and read the multipliers after the
        // master has been solved with it.
        bool completed = false;
        for (Index b = 0; b < blocks; ++b) {
          if (has_cut[at(b)] != 0) continue;
          std::vector<double> g;
          slopes_of(setup.blocks[at(b)], answers[at(b)].dual, &g, slots);
          double at_y = 0.0;
          for (Index s = 0; s < slots; ++s) at_y += g[at(s)] * y[at(s)];
          Cut cut;
          cut.block = b;
          cut.dual = answers[at(b)].dual;
          cut.rhs = answers[at(b)].value - at_y;
          for (Index s = 0; s < slots; ++s) {
            if (g[at(s)] == 0.0) continue;
            cut.slots.push_back(s);
            cut.slopes.push_back(g[at(s)]);
          }
          has_cut[at(b)] = 1;
          cuts.push_back(std::move(cut));
          completed = true;
        }
        if (completed) continue;
        // Converged with the bounded epigraph columns, which prove the bound but do not give
        // multipliers that sum to one. One more round with them free, and the pair read from
        // that round - blocks and master at the same y - is the answer.
        if (!free_theta) {
          free_theta = true;
          continue;
        }
        // Converged: primal from this round's blocks, duals from this round's master.
        const Index n = model.num_cols();
        const Index m = model.num_rows();
        Solution out;
        out.algorithm = "benders";
        out.col_value.assign(at(n), 0.0);
        for (Index s = 0; s < slots; ++s)
          out.col_value[at(setup.slot_column[at(s)])] = y[at(s)];
        for (Index b = 0; b < blocks; ++b) {
          const Block& block = setup.blocks[at(b)];
          for (std::size_t c = 0; c < block.columns.size(); ++c) {
            out.col_value[at(block.columns[c])] = answers[at(b)].x[c];
          }
        }
        out.row_dual.assign(at(m), 0.0);
        for (Index k = 0; k < master_rows; ++k) {
          out.row_dual[at(setup.master_rows[at(k)])] = ms.row_dual[at(k)];
        }
        for (std::size_t c = 0; c < cuts.size(); ++c) {
          const Cut& cut = cuts[c];
          const double multiplier = ms.row_dual[at(master_rows) + c];
          const double weight = cut.feasibility ? -multiplier : multiplier;
          const Block& block = setup.blocks[at(cut.block)];
          for (std::size_t r = 0; r < block.rows.size(); ++r) {
            out.row_dual[at(block.rows[r])] += weight * cut.dual[r];
          }
        }
        out.col_dual.assign(at(n), 0.0);
        for (Index j = 0; j < n; ++j) {
          double reduced = model.col_cost[at(j)];
          const ColumnView column = model.matrix.column(j);
          for (Index k = 0; k < column.size; ++k) {
            reduced -= column.values[k] * out.row_dual[at(column.rows[k])];
          }
          out.col_dual[at(j)] = reduced;
        }
        out.status = SolveStatus::kOptimal;
        out.dual_bound = lower;
        out.iterations = rounds;
        out.solve_seconds = timer.elapsed_seconds();
        out.message = fmt::format(
            "Benders decomposition: {} block(s), {} linking column(s), {} round(s), {} cut(s)",
            blocks, slots, rounds, cuts.size());
        out.recompute_quality(model);
        outcome.solution = std::move(out);
        return outcome;
      }
    }

    // The cuts this y calls for.
    std::size_t added = 0;
    for (Index b = 0; b < blocks; ++b) {
      const BlockAnswer& a = answers[at(b)];
      std::vector<double> g;
      slopes_of(setup.blocks[at(b)], a.dual, &g, slots);
      double at_y = 0.0;
      for (Index s = 0; s < slots; ++s) at_y += g[at(s)] * y[at(s)];
      Cut cut;
      cut.block = b;
      cut.dual = a.dual;
      if (a.kind == BlockAnswer::Kind::kOptimal) {
        const double theta = ms.col_value[at(slots + b)];
        if (theta >= a.value - tol::kBendersCutViolation * std::max(1.0, std::fabs(a.value))) {
          continue;
        }
        cut.rhs = a.value - at_y;
      } else {
        cut.feasibility = true;
        cut.rhs = at_y - a.value;
      }
      for (Index s = 0; s < slots; ++s) {
        if (g[at(s)] == 0.0) continue;
        cut.slots.push_back(s);
        cut.slopes.push_back(g[at(s)]);
      }
      if (!cut.feasibility) has_cut[at(b)] = 1;
      cuts.push_back(std::move(cut));
      ++added;
    }
    if (added == 0)
      return decline("no cut is violated at the master's point, yet the gap is open");
  }
}

}  // namespace sankhya::decomp
