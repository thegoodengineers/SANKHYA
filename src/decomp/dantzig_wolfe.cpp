// SPDX-License-Identifier: Apache-2.0
// SANKHYA - Dantzig-Wolfe decomposition of an LP over linking rows (#1048). See
// dantzig_wolfe.hpp for the method, the two phases and why the answer is measured.

#include "decomp/dantzig_wolfe.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <string>
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

/// A sparse vector over the master's linking rows.
struct Linked {
  std::vector<Index> rows;
  std::vector<double> values;
};

struct Block {
  std::vector<Index> rows;     ///< global rows it owns, ascending
  std::vector<Index> columns;  ///< global columns it owns, ascending
  Model sub;                   ///< its rows x its columns, true costs
  std::vector<Linked> link;    ///< per local column, its entries in the linking rows
};

struct Setup {
  std::vector<Block> blocks;
  std::vector<Index> linking_rows;  ///< master row -> global row
  std::vector<Index> direct;        ///< global columns kept in the master as they are
  std::vector<Linked> direct_link;  ///< their entries in the linking rows
};

/// A point (or a ray) of one block, as the master holds it.
struct Proposal {
  Index block = 0;
  bool ray = false;
  std::vector<double> x;  ///< over the block's columns
  double cost = 0.0;      ///< c_b'x
  Linked activity;        ///< A_0b x
};

struct Priced {
  enum class Kind { kPoint, kRay, kFailed } kind = Kind::kFailed;
  double value = 0.0;  ///< the block LP's optimum (a point), or c~'d (a ray)
  std::vector<double> x;
  std::vector<double> dual;
  std::string failure;
  LimitReason limit = LimitReason::kNone;
};

/// Splits `model` along `structure`, checking it: a column a block owns may be in that block's
/// rows and in linking rows, nowhere else.
std::string prepare(const Model& model, const BlockStructure& structure, Setup* out) {
  const Index m = model.num_rows();
  const Index n = model.num_cols();
  out->blocks.assign(at(structure.blocks), Block{});
  std::vector<Index> row_local(at(m), -1);
  for (Index i = 0; i < m; ++i) {
    const Index b = structure.row_block[at(i)];
    if (b >= 0) {
      row_local[at(i)] = static_cast<Index>(out->blocks[at(b)].rows.size());
      out->blocks[at(b)].rows.push_back(i);
    } else if (b == kLinking) {
      row_local[at(i)] = static_cast<Index>(out->linking_rows.size());
      out->linking_rows.push_back(i);
    } else if (model.row_lower[at(i)] > 0.0 || model.row_upper[at(i)] < 0.0) {
      return fmt::format("row {} is empty and excludes zero", i);
    }
  }
  struct Triplet {
    Index row, col;
    double value;
  };
  std::vector<std::vector<Triplet>> entries(at(structure.blocks));
  for (Index j = 0; j < n; ++j) {
    const Index owner = structure.col_block[at(j)];
    Linked link;
    Index local = -1;
    if (owner >= 0) {
      local = static_cast<Index>(out->blocks[at(owner)].columns.size());
      out->blocks[at(owner)].columns.push_back(j);
    }
    const ColumnView column = model.matrix.column(j);
    for (Index k = 0; k < column.size; ++k) {
      if (column.values[k] == 0.0) continue;
      const Index i = column.rows[k];
      const Index rb = structure.row_block[at(i)];
      if (rb == kLinking) {
        link.rows.push_back(row_local[at(i)]);
        link.values.push_back(column.values[k]);
      } else if (rb >= 0 && rb == owner) {
        entries[at(owner)].push_back({row_local[at(i)], local, column.values[k]});
      } else {
        return fmt::format("column {} of block {} is in row {} of block {}", j, owner, i, rb);
      }
    }
    if (owner >= 0) {
      out->blocks[at(owner)].link.push_back(std::move(link));
    } else {
      out->direct.push_back(j);
      out->direct_link.push_back(std::move(link));
    }
  }
  for (Index b = 0; b < structure.blocks; ++b) {
    Block& block = out->blocks[at(b)];
    const Index rows = static_cast<Index>(block.rows.size());
    const Index cols = static_cast<Index>(block.columns.size());
    Model& sub = block.sub;
    sub.sense = ObjSense::kMinimize;
    sub.resize_columns(cols);
    sub.resize_rows(rows);
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
    for (const Triplet& t : entries[at(b)]) matrix.add_entry(t.row, t.col, t.value);
    matrix.finalize(0.0);
    sub.matrix = std::move(matrix);
  }
  return {};
}

/// The block's costs priced against the linking rows' duals: c_b - A_0b' pi (phase two), or
/// - A_0b' pi alone (phase one).
std::vector<double> priced_costs(const Block& block, const std::vector<double>& pi,
                                 bool phase_one) {
  std::vector<double> cost =
      phase_one ? std::vector<double>(block.columns.size(), 0.0) : block.sub.col_cost;
  for (std::size_t c = 0; c < block.columns.size(); ++c) {
    const Linked& link = block.link[c];
    for (std::size_t k = 0; k < link.rows.size(); ++k) {
      cost[c] -= link.values[k] * pi[at(link.rows[k])];
    }
  }
  return cost;
}

Priced price_block(const Block& block, std::vector<double> cost, const Options& options,
                   SolveControl* control) {
  Priced out;
  try {
    Model sub = block.sub;
    sub.col_cost = std::move(cost);
    const Solution s = solve(sub, options, control);
    if (s.status == SolveStatus::kTimeLimit || s.status == SolveStatus::kInterrupted) {
      out.limit =
          s.status == SolveStatus::kTimeLimit ? LimitReason::kTime : LimitReason::kInterrupt;
      return out;
    }
    if (s.status == SolveStatus::kOptimal && s.row_dual.size() == block.rows.size() &&
        s.col_value.size() == block.columns.size()) {
      out.kind = Priced::Kind::kPoint;
      out.value = s.objective;
      out.x = s.col_value;
      out.dual = s.row_dual;
      return out;
    }
    if (s.status == SolveStatus::kUnbounded && s.primal_ray.size() == block.columns.size()) {
      out.kind = Priced::Kind::kRay;
      out.x = s.primal_ray;
      double value = 0.0;
      for (std::size_t c = 0; c < out.x.size(); ++c) value += sub.col_cost[c] * out.x[c];
      out.value = value;
      if (value < 0.0) return out;
      out.kind = Priced::Kind::kFailed;
    }
    out.failure =
        fmt::format("a block LP ended {}{}", to_string(s.status),
                    s.status == SolveStatus::kUnbounded ? " without a usable ray" : "");
    return out;
  } catch (const std::exception& error) {
    out.failure = fmt::format("a block solve threw: {}", error.what());
    return out;
  }
}

Proposal make_proposal(const Block& block, Index b, const Priced& priced) {
  Proposal p;
  p.block = b;
  p.ray = priced.kind == Priced::Kind::kRay;
  p.x = priced.x;
  std::vector<double> dense;
  for (std::size_t c = 0; c < p.x.size(); ++c) {
    const double v = p.x[c];
    if (v == 0.0) continue;
    p.cost += block.sub.col_cost[c] * v;
    const Linked& link = block.link[c];
    for (std::size_t k = 0; k < link.rows.size(); ++k) {
      const auto r = at(link.rows[k]);
      if (dense.size() <= r) dense.resize(r + 1, 0.0);
      dense[r] += link.values[k] * v;
    }
  }
  for (std::size_t r = 0; r < dense.size(); ++r) {
    if (dense[r] == 0.0) continue;
    p.activity.rows.push_back(static_cast<Index>(r));
    p.activity.values.push_back(dense[r]);
  }
  return p;
}

/// The restricted master: direct columns, then one column per proposal, then (phase one) a
/// surplus and a slack per linking row; the linking rows, then one convexity row per block.
Model build_master(const Model& model, const Setup& setup, const std::vector<Proposal>& props,
                   bool phase_one) {
  const Index direct = static_cast<Index>(setup.direct.size());
  const Index count = static_cast<Index>(props.size());
  const Index linking = static_cast<Index>(setup.linking_rows.size());
  const Index blocks = static_cast<Index>(setup.blocks.size());
  const Index artificial = phase_one ? 2 * linking : 0;
  const Index cols = direct + count + artificial;
  Model master;
  master.sense = ObjSense::kMinimize;
  master.objective_offset = phase_one ? 0.0 : model.objective_offset;
  master.resize_columns(cols);
  master.resize_rows(linking + blocks);
  SparseMatrix matrix(linking + blocks, cols);
  for (Index d = 0; d < direct; ++d) {
    const auto g = at(setup.direct[at(d)]);
    master.col_cost[at(d)] = phase_one ? 0.0 : model.col_cost[g];
    master.col_lower[at(d)] = model.col_lower[g];
    master.col_upper[at(d)] = model.col_upper[g];
    const Linked& link = setup.direct_link[at(d)];
    for (std::size_t k = 0; k < link.rows.size(); ++k)
      matrix.add_entry(link.rows[k], d, link.values[k]);
  }
  for (Index p = 0; p < count; ++p) {
    const Proposal& prop = props[at(p)];
    const Index col = direct + p;
    master.col_cost[at(col)] = phase_one ? 0.0 : prop.cost;
    master.col_lower[at(col)] = 0.0;
    master.col_upper[at(col)] = kInfinity;
    for (std::size_t k = 0; k < prop.activity.rows.size(); ++k)
      matrix.add_entry(prop.activity.rows[k], col, prop.activity.values[k]);
    if (!prop.ray) matrix.add_entry(linking + prop.block, col, 1.0);
  }
  for (Index r = 0; r < linking; ++r) {
    const auto g = at(setup.linking_rows[at(r)]);
    master.row_lower[at(r)] = model.row_lower[g];
    master.row_upper[at(r)] = model.row_upper[g];
    if (!phase_one) continue;
    for (const Index k : {direct + count + r, direct + count + linking + r}) {
      master.col_cost[at(k)] = 1.0;
      master.col_lower[at(k)] = 0.0;
      master.col_upper[at(k)] = kInfinity;
    }
    matrix.add_entry(r, direct + count + r, 1.0);
    matrix.add_entry(r, direct + count + linking + r, -1.0);
  }
  for (Index b = 0; b < blocks; ++b) {
    master.row_lower[at(linking + b)] = 1.0;
    master.row_upper[at(linking + b)] = 1.0;
  }
  matrix.finalize(0.0);
  master.matrix = std::move(matrix);
  return master;
}

Solution limit_solution(LimitReason reason, Count rounds, double seconds) {
  Solution s;
  s.algorithm = "dantzig-wolfe";
  s.status = status_for(reason);
  s.stopped_by = reason;
  s.iterations = rounds;
  s.solve_seconds = seconds;
  s.message =
      fmt::format("stopped by the {} during Dantzig-Wolfe decomposition after {} round(s)",
                  to_string(reason), rounds);
  return s;
}

}  // namespace

BendersOutcome solve_dantzig_wolfe(const Model& model, const BlockStructure& structure,
                                   const Options& options, SolveControl* control,
                                   Logger& logger) {
  BendersOutcome outcome;
  const Timer timer;
  const auto decline = [&](const std::string& why) {
    outcome.declined = why;
    return outcome;
  };
  if (structure.kind != Linking::kRows)
    return decline("Dantzig-Wolfe decomposes linking rows only");
  Setup setup;
  if (const std::string bad = prepare(model, structure, &setup); !bad.empty())
    return decline("the structure does not hold: " + bad);
  const Index blocks = static_cast<Index>(setup.blocks.size());
  const Index direct = static_cast<Index>(setup.direct.size());
  const Index linking = static_cast<Index>(setup.linking_rows.size());
  // The same budget rule as Benders: half of a finite limit, the rest for the monolithic solve
  // should this decline.
  const double time_limit = options.get_double("time_limit");
  const bool limited = time_limit < std::numeric_limits<double>::max();
  const double budget = limited ? 0.5 * time_limit : time_limit;
  const Count max_rounds = options.get_int("decomposition_max_iterations");
  Index threads = static_cast<Index>(options.get_int("decomposition_threads"));
  if (threads <= 0)
    threads = std::max<Index>(1, static_cast<Index>(std::thread::hardware_concurrency()));
  threads = std::min(threads, blocks);
  // The progress callback is not thread-safe (#222): with one attached, price on this thread.
  if (control != nullptr && control->progress_callback) threads = 1;
  const auto left = [&] { return std::max(0.0, budget - timer.elapsed_seconds()); };
  const auto stopped = [&](LimitReason reason, Count rounds) {
    if (reason == LimitReason::kInterrupt) {
      outcome.solution = limit_solution(reason, rounds, timer.elapsed_seconds());
      return outcome;
    }
    return decline(fmt::format(
        "not converged after {} round(s) in the {:.1f} s it is given (half of the {:.1f} s "
        "limit), so the monolithic engines get the rest",
        rounds, budget, time_limit));
  };
  const auto limit_of = [](const Solution& s) {
    if (s.status == SolveStatus::kTimeLimit) return LimitReason::kTime;
    if (s.status == SolveStatus::kInterrupted) return LimitReason::kInterrupt;
    return LimitReason::kNone;
  };

  // The blocks priced at `pi` (all of them, in parallel; read in block order, so the result
  // does not depend on which thread finished first).
  std::vector<Priced> priced(at(blocks));
  const auto price_all = [&](const std::vector<double>& pi, bool phase_one, bool own_costs) {
    std::atomic<Index> next{0};
    const auto work = [&] {
      for (Index b = next.fetch_add(1); b < blocks; b = next.fetch_add(1)) {
        const Block& block = setup.blocks[at(b)];
        std::vector<double> cost =
            own_costs ? block.sub.col_cost : priced_costs(block, pi, phase_one);
        priced[at(b)] =
            price_block(block, std::move(cost), block_solve_options(options, left()), control);
      }
    };
    if (threads <= 1) {
      work();
    } else {
      std::vector<std::thread> pool;
      for (Index w = 0; w < threads; ++w) pool.emplace_back(work);
      for (std::thread& t : pool) t.join();
    }
  };

  // The start: every block's optimum for its own costs.
  std::vector<Proposal> props;
  price_all({}, false, true);
  for (Index b = 0; b < blocks; ++b) {
    const Priced& p = priced[at(b)];
    if (p.limit != LimitReason::kNone) return stopped(p.limit, 0);
    if (p.kind == Priced::Kind::kFailed)
      return decline(fmt::format("block {}: {}", b, p.failure));
    if (p.kind == Priced::Kind::kRay) {
      // Unbounded for its own costs: the ray goes in, and a point of the block is still needed
      // for its convexity row, which any objective gives - zero is the simplest.
      props.push_back(make_proposal(setup.blocks[at(b)], b, p));
      Model sub = setup.blocks[at(b)].sub;
      std::fill(sub.col_cost.begin(), sub.col_cost.end(), 0.0);
      const Solution s = solve(sub, block_solve_options(options, left()), control);
      if (limit_of(s) != LimitReason::kNone) return stopped(limit_of(s), 0);
      if (s.status != SolveStatus::kOptimal)
        return decline(fmt::format("block {} has no point ({})", b, to_string(s.status)));
      Priced point;
      point.kind = Priced::Kind::kPoint;
      point.x = s.col_value;
      props.push_back(make_proposal(setup.blocks[at(b)], b, point));
      continue;
    }
    props.push_back(make_proposal(setup.blocks[at(b)], b, p));
  }

  bool phase_one = true;
  Count rounds = 0;
  while (true) {
    if (control != nullptr && control->interruption_requested())
      return stopped(LimitReason::kInterrupt, rounds);
    if (timer.elapsed_seconds() > budget) return stopped(LimitReason::kTime, rounds);
    if (rounds >= max_rounds)
      return decline(fmt::format("no convergence in {} rounds", rounds));
    ++rounds;

    // The restricted master, by the dual simplex.
    Options master_options = block_solve_options(options, left());
    master_options.set_string("algorithm", "dual-simplex");
    const Model master = build_master(model, setup, props, phase_one);
    const Solution ms = solve(master, master_options, control);
    if (limit_of(ms) != LimitReason::kNone) return stopped(limit_of(ms), rounds);
    if (ms.status != SolveStatus::kOptimal)
      return decline(fmt::format("the restricted master (phase {}) ended {}", phase_one ? 1 : 2,
                                 to_string(ms.status)));
    if (ms.row_dual.size() != at(linking + blocks) ||
        ms.col_value.size() != at(master.num_cols()))
      return decline("the restricted master returned no duals");
    const double z = ms.objective;
    if (phase_one && z <= tol::kPrimalFeasibility) {
      // Feasible: the second phase starts from these weights, with the true costs.
      phase_one = false;
      logger.verbose("Dantzig-Wolfe round {}: the master is feasible, phase two", rounds);
      continue;
    }
    const std::vector<double> pi(ms.row_dual.begin(), ms.row_dual.begin() + linking);
    price_all(pi, phase_one, false);

    // Reduced costs; the Lagrangian bound is the master's value plus the negative ones.
    const double scale = std::max(1.0, std::fabs(z));
    double shortfall = 0.0;
    std::size_t added = 0;
    for (Index b = 0; b < blocks; ++b) {
      const Priced& p = priced[at(b)];
      if (p.limit != LimitReason::kNone) return stopped(p.limit, rounds);
      if (p.kind == Priced::Kind::kFailed)
        return decline(fmt::format("block {}: {}", b, p.failure));
      const double reduced =
          p.kind == Priced::Kind::kRay ? p.value : p.value - ms.row_dual[at(linking + b)];
      if (p.kind == Priced::Kind::kRay) {
        shortfall = -kInfinity;  // the bound is minus infinity until no ray prices out
      } else if (reduced < 0.0) {
        shortfall += reduced;
      }
      if (reduced < -tol::kDantzigWolfeReducedCost * scale) {
        props.push_back(make_proposal(setup.blocks[at(b)], b, p));
        ++added;
      }
    }
    const double lower = z + shortfall;
    logger.verbose(
        "Dantzig-Wolfe round {} (phase {}): master {:.12g}, bound {:.12g}, {} column(s)",
        rounds, phase_one ? 1 : 2, z, lower, props.size());
    if (shortfall >= -tol::kDantzigWolfeGap * scale) {
      if (phase_one) {
        return decline(
            fmt::format("the first phase ends with the linking rows violated by {:.3g} in "
                        "total, so the LP is "
                        "infeasible; the monolithic engines prove it",
                        z));
      }
      // Converged: the point from the master's weights, the duals from pi and the blocks.
      const Index n = model.num_cols();
      const Index m = model.num_rows();
      Solution out;
      out.algorithm = "dantzig-wolfe";
      out.col_value.assign(at(n), 0.0);
      for (Index d = 0; d < direct; ++d)
        out.col_value[at(setup.direct[at(d)])] = ms.col_value[at(d)];
      for (std::size_t q = 0; q < props.size() && q < ms.col_value.size() - at(direct); ++q) {
        const double weight = ms.col_value[at(direct) + q];
        if (weight == 0.0) continue;
        const Proposal& prop = props[q];
        const Block& block = setup.blocks[at(prop.block)];
        for (std::size_t c = 0; c < block.columns.size(); ++c)
          out.col_value[at(block.columns[c])] += weight * prop.x[c];
      }
      out.row_dual.assign(at(m), 0.0);
      for (Index r = 0; r < linking; ++r)
        out.row_dual[at(setup.linking_rows[at(r)])] = pi[at(r)];
      for (Index b = 0; b < blocks; ++b) {
        const Block& block = setup.blocks[at(b)];
        for (std::size_t r = 0; r < block.rows.size(); ++r)
          out.row_dual[at(block.rows[r])] = priced[at(b)].dual[r];
      }
      out.col_dual.assign(at(n), 0.0);
      for (Index j = 0; j < n; ++j) {
        double reduced = model.col_cost[at(j)];
        const ColumnView column = model.matrix.column(j);
        for (Index k = 0; k < column.size; ++k)
          reduced -= column.values[k] * out.row_dual[at(column.rows[k])];
        out.col_dual[at(j)] = reduced;
      }
      out.status = SolveStatus::kOptimal;
      out.dual_bound = lower;
      out.iterations = rounds;
      out.solve_seconds = timer.elapsed_seconds();
      out.message = fmt::format(
          "Dantzig-Wolfe decomposition: {} block(s), {} linking row(s), {} round(s), {} "
          "column(s)",
          blocks, linking, rounds, props.size());
      out.recompute_quality(model);
      outcome.solution = std::move(out);
      return outcome;
    }
    if (added == 0)
      return decline("no column prices out at the master's duals, yet the gap is open");
  }
}

}  // namespace sankhya::decomp
