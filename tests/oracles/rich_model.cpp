// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the general-form model of the oracle fuzz: its two readings. TESTS ONLY.
//
// Reference for the rewrite onto the oracle's form: Chvatal, "Linear Programming" (1983),
// ch. 8 (bounded, free and reflected variables). For the unbounded mixed-integer case:
// Meyer, "On the existence of optimal solutions to integer and mixed-integer programming
// problems", Mathematical Programming 7 (1974).

#include "oracles/rich_model.hpp"

#include <sstream>
#include <utility>

namespace sankhya::oracle {
namespace {

using Int = Rational::Int;

std::size_t at(Index i) {
  return static_cast<std::size_t>(i);
}

Int gcd_of(Int a, Int b) {
  if (a < Int(0)) a = -a;
  if (b < Int(0)) b = -b;
  while (b != Int(0)) {
    const Int t = a % b;
    a = b;
    b = t;
  }
  return a;
}

/// *common becomes the least common multiple of itself and r's denominator.
void fold_denominator(const Rational& r, Int* common) {
  const Int d = r.denominator();
  *common = *common / gcd_of(*common, d) * d;
}

Rational floor_of(const Rational& r) {
  const Int n = r.numerator();
  const Int d = r.denominator();  // always positive
  Int q = n / d;                  // truncates toward zero
  if (n % d != Int(0) && n < Int(0)) q = q - Int(1);
  return Rational(q);
}

Rational ceil_of(const Rational& r) {
  return -floor_of(-r);
}

/// An integer the oracle's int64 fields can hold with room to multiply, or RationalOverflow,
/// which the caller reports as the oracle abstaining.
std::int64_t to_int64(const Rational& r) {
  const Int limit = Int(std::int64_t{1} << 62);
  if (r.denominator() != Int(1) || r.numerator() > limit || r.numerator() < -limit) {
    throw RationalOverflow();
  }
  return static_cast<std::int64_t>(r.numerator());
}

std::string text(const Rational& r) {
  std::string s = std::to_string(static_cast<long long>(r.numerator()));
  if (r.denominator() != Int(1)) {
    s += "/" + std::to_string(static_cast<long long>(r.denominator()));
  }
  return s;
}

/// The instance on the oracle's form, and the way back:
///   x_j = base_j + x'[plus_j] - x'[minus_j]              (an absent part is -1)
///   objective = sign * (oracle objective) / objective_divisor + objective_constant
struct Normal {
  GeneratedLp lp;
  /// A column whose domain is empty, or a row 0 >= b with b > 0: no oracle run needed.
  bool infeasible = false;
  std::vector<Rational> base;
  std::vector<Index> plus;
  std::vector<Index> minus;
  Rational objective_divisor;
  Rational objective_constant;
};

/// Append  sum coefficients[k] x'[k] >= rhs,  multiplied through to integers.
void append_row(Normal* n, const std::vector<Rational>& coefficients, const Rational& rhs) {
  Int common = rhs.denominator();
  bool any = false;
  for (const Rational& v : coefficients) {
    fold_denominator(v, &common);
    any = any || !v.is_zero();
  }
  if (!any) {
    if (rhs.is_positive()) n->infeasible = true;  // 0 >= rhs
    return;
  }
  const Rational scale(common);
  std::vector<std::int64_t> row;
  row.reserve(coefficients.size());
  for (const Rational& v : coefficients) row.push_back(to_int64(v * scale));
  n->lp.a.push_back(std::move(row));
  n->lp.b.push_back(to_int64(rhs * scale));
  ++n->lp.num_rows;
}

Normal normalize(const RichModel& m) {
  Normal n;
  const std::size_t cols = at(m.num_cols);
  n.base.assign(cols, Rational());
  n.plus.assign(cols, -1);
  n.minus.assign(cols, -1);
  // An upper bound on an oracle column that is not an integer goes in as a row.
  std::vector<std::pair<Index, Rational>> fractional_upper;

  const auto add_column = [&n](bool integral) {
    n.lp.upper.push_back(kNoUpperBound);
    n.lp.integral.push_back(integral ? 1 : 0);
    return n.lp.num_cols++;
  };
  for (std::size_t j = 0; j < cols; ++j) {
    const bool integral = m.integral[j] != 0;
    Bound lower = m.col_lower[j];
    Bound upper = m.col_upper[j];
    // An integer column's domain is the integers inside its bounds.
    if (integral && lower) lower = ceil_of(*lower);
    if (integral && upper) upper = floor_of(*upper);
    if (lower && upper && *upper < *lower) {
      n.infeasible = true;
      return n;
    }
    if (lower) {
      n.base[j] = *lower;
      n.plus[j] = add_column(integral);
      if (upper) {
        const Rational width = *upper - *lower;
        if (width.denominator() == Int(1)) {
          n.lp.upper.back() = to_int64(width);
        } else {
          fractional_upper.emplace_back(n.plus[j], width);
        }
      }
    } else if (upper) {
      n.base[j] = *upper;
      n.minus[j] = add_column(integral);
    } else {
      n.plus[j] = add_column(integral);
      n.minus[j] = add_column(integral);
    }
  }

  const std::size_t width = at(n.lp.num_cols);
  for (const auto& [column, bound] : fractional_upper) {
    std::vector<Rational> row(width);
    row[at(column)] = rat(-1);
    append_row(&n, row, -bound);
  }
  for (std::size_t i = 0; i < at(m.num_rows); ++i) {
    std::vector<Rational> row(width);
    Rational constant;
    for (std::size_t j = 0; j < cols; ++j) {
      const Rational& v = m.a[i][j];
      if (v.is_zero()) continue;
      constant += v * n.base[j];
      if (n.plus[j] >= 0) row[at(n.plus[j])] = v;
      if (n.minus[j] >= 0) row[at(n.minus[j])] = -v;
    }
    if (m.row_lower[i]) append_row(&n, row, *m.row_lower[i] - constant);
    if (m.row_upper[i]) {
      for (Rational& v : row) v = -v;
      append_row(&n, row, constant - *m.row_upper[i]);
    }
  }

  std::vector<Rational> costs(width);
  Int common = Int(1);
  n.objective_constant = m.offset;
  for (std::size_t j = 0; j < cols; ++j) {
    const Rational c = m.maximize ? -m.cost[j] : m.cost[j];
    n.objective_constant += m.cost[j] * n.base[j];
    if (n.plus[j] >= 0) costs[at(n.plus[j])] = c;
    if (n.minus[j] >= 0) costs[at(n.minus[j])] = -c;
    fold_denominator(c, &common);
  }
  n.objective_divisor = Rational(common);
  n.lp.c.reserve(width);
  for (const Rational& c : costs) n.lp.c.push_back(to_int64(c * n.objective_divisor));
  return n;
}

RichModel without_row(RichModel m, Index i) {
  m.a.erase(m.a.begin() + i);
  m.row_lower.erase(m.row_lower.begin() + i);
  m.row_upper.erase(m.row_upper.begin() + i);
  --m.num_rows;
  return m;
}

RichModel without_column(RichModel m, Index j) {
  for (auto& row : m.a) row.erase(row.begin() + j);
  m.col_lower.erase(m.col_lower.begin() + j);
  m.col_upper.erase(m.col_upper.begin() + j);
  m.integral.erase(m.integral.begin() + j);
  m.cost.erase(m.cost.begin() + j);
  --m.num_cols;
  return m;
}

}  // namespace

Rational rat(std::int64_t numerator, std::int64_t denominator) {
  return Rational(Int(static_cast<long long>(numerator)),
                  Int(static_cast<long long>(denominator)));
}

bool RichModel::has_integers() const {
  for (const char flag : integral) {
    if (flag != 0) return true;
  }
  return false;
}

std::string RichModel::to_text() const {
  std::ostringstream out;
  out << (maximize ? "maximize  " : "minimize  ") << text(offset);
  for (std::size_t j = 0; j < at(num_cols); ++j) {
    if (!cost[j].is_zero()) out << " + " << text(cost[j]) << " x" << j;
  }
  out << "\nsubject to\n";
  for (std::size_t i = 0; i < at(num_rows); ++i) {
    out << "  ";
    if (row_lower[i]) out << text(*row_lower[i]) << " <= ";
    bool first = true;
    for (std::size_t j = 0; j < at(num_cols); ++j) {
      if (a[i][j].is_zero()) continue;
      out << (first ? "" : " + ") << text(a[i][j]) << " x" << j;
      first = false;
    }
    if (first) out << "0";
    if (row_upper[i]) out << " <= " << text(*row_upper[i]);
    out << "\n";
  }
  out << "columns\n";
  for (std::size_t j = 0; j < at(num_cols); ++j) {
    out << "  " << (col_lower[j] ? text(*col_lower[j]) : std::string("-inf")) << " <= x" << j
        << " <= " << (col_upper[j] ? text(*col_upper[j]) : std::string("inf"))
        << (integral[j] != 0 ? "  integer" : "") << "\n";
  }
  return out.str();
}

Model to_model(const RichModel& rich) {
  Model model;
  model.name = "rich";
  model.sense = rich.maximize ? ObjSense::kMaximize : ObjSense::kMinimize;
  model.objective_offset = rich.offset.to_double();
  model.resize_columns(rich.num_cols);
  model.resize_rows(rich.num_rows);
  for (std::size_t j = 0; j < at(rich.num_cols); ++j) {
    model.col_cost[j] = rich.cost[j].to_double();
    model.col_lower[j] = rich.col_lower[j] ? rich.col_lower[j]->to_double() : -kInfinity;
    model.col_upper[j] = rich.col_upper[j] ? rich.col_upper[j]->to_double() : kInfinity;
    model.col_type[j] = rich.integral[j] != 0 ? VarType::kInteger : VarType::kContinuous;
  }
  model.matrix.reset(rich.num_rows, rich.num_cols);
  for (Index i = 0; i < rich.num_rows; ++i) {
    model.row_lower[at(i)] =
        rich.row_lower[at(i)] ? rich.row_lower[at(i)]->to_double() : -kInfinity;
    model.row_upper[at(i)] =
        rich.row_upper[at(i)] ? rich.row_upper[at(i)]->to_double() : kInfinity;
    for (Index j = 0; j < rich.num_cols; ++j) {
      const Rational& v = rich.a[at(i)][at(j)];
      if (!v.is_zero()) model.matrix.add_entry(i, j, v.to_double());
    }
  }
  model.matrix.finalize();
  return model;
}

RichExact solve_exact(const RichModel& rich, std::int64_t node_limit) {
  RichExact out;
  try {
    const Normal n = normalize(rich);
    if (n.infeasible) {
      out.status = OracleStatus::kInfeasible;
      return out;
    }
    const bool mixed_integer = rich.has_integers();
    OracleResult r = mixed_integer ? solve_exact_milp(n.lp, node_limit) : solve_exact(n.lp);
    if (mixed_integer && r.status == OracleStatus::kUnbounded) {
      // solve_exact_milp says unbounded as soon as a relaxation is. With rational data that
      // is the MILP's answer exactly when the MILP has a point (Meyer 1974), so look for one.
      GeneratedLp any_point = n.lp;
      any_point.c.assign(any_point.c.size(), 0);
      const OracleResult found = solve_exact_milp(any_point, node_limit / 10);
      if (found.status != OracleStatus::kOptimal) r.status = found.status;
    }
    out.status = r.status;
    if (r.status != OracleStatus::kOptimal) return out;

    out.x.reserve(n.base.size());
    for (std::size_t j = 0; j < n.base.size(); ++j) {
      Rational value = n.base[j];
      if (n.plus[j] >= 0) value += r.x[at(n.plus[j])];
      if (n.minus[j] >= 0) value -= r.x[at(n.minus[j])];
      out.x.push_back(value);
    }
    const Rational in_minimization = r.objective / n.objective_divisor;
    out.objective = (rich.maximize ? -in_minimization : in_minimization) + n.objective_constant;
  } catch (const RationalOverflow&) {
    out.status = OracleStatus::kOverflow;
  }
  return out;
}

std::string exact_violation(const RichModel& rich, const std::vector<Rational>& x,
                            const Rational& objective) {
  if (x.size() != at(rich.num_cols)) return "the exact point has the wrong length";
  try {
    Rational value = rich.offset;
    for (std::size_t j = 0; j < x.size(); ++j) {
      const std::string name = "x" + std::to_string(j) + " = " + text(x[j]);
      if (rich.col_lower[j] && x[j] < *rich.col_lower[j]) return name + " is below its bound";
      if (rich.col_upper[j] && x[j] > *rich.col_upper[j]) return name + " is above its bound";
      if (rich.integral[j] != 0 && x[j].denominator() != Int(1)) return name + " is fractional";
      value += rich.cost[j] * x[j];
    }
    for (std::size_t i = 0; i < at(rich.num_rows); ++i) {
      Rational activity;
      for (std::size_t j = 0; j < x.size(); ++j) activity += rich.a[i][j] * x[j];
      const std::string name = "row " + std::to_string(i) + " = " + text(activity);
      if (rich.row_lower[i] && activity < *rich.row_lower[i])
        return name + " is below its bound";
      if (rich.row_upper[i] && activity > *rich.row_upper[i])
        return name + " is above its bound";
    }
    if (value != objective) {
      return "the point is worth " + text(value) + ", not " + text(objective);
    }
  } catch (const RationalOverflow&) {
    return "";  // the check itself overflowed; it says nothing either way
  }
  return "";
}

RichModel shrink(RichModel rich, const std::function<bool(const RichModel&)>& fails) {
  int budget = 4000;  // evaluations of `fails`, each an exact solve and a float solve
  const auto keep = [&](RichModel candidate) {
    if (budget <= 0) return false;
    --budget;
    if (!fails(candidate)) return false;
    rich = std::move(candidate);
    return true;
  };
  // Every accepted step removes something (a row, a column, an entry, a bound, an integer
  // mark, a cost, the constant, the sense), so the loop ends.
  for (bool changed = true; changed && budget > 0;) {
    changed = false;
    for (Index i = rich.num_rows - 1; i >= 0; --i) {
      changed = keep(without_row(rich, i)) || changed;
    }
    for (Index j = rich.num_cols - 1; j >= 0 && rich.num_cols > 1; --j) {
      changed = keep(without_column(rich, j)) || changed;
    }
    for (std::size_t i = 0; i < at(rich.num_rows); ++i) {
      for (std::size_t j = 0; j < at(rich.num_cols); ++j) {
        if (rich.a[i][j].is_zero()) continue;
        RichModel candidate = rich;
        candidate.a[i][j] = Rational();
        changed = keep(std::move(candidate)) || changed;
      }
      if (!rich.row_lower[i] || !rich.row_upper[i]) continue;
      RichModel no_lower = rich;
      no_lower.row_lower[i].reset();
      if (keep(std::move(no_lower))) {
        changed = true;
        continue;
      }
      RichModel no_upper = rich;
      no_upper.row_upper[i].reset();
      changed = keep(std::move(no_upper)) || changed;
    }
    for (std::size_t j = 0; j < at(rich.num_cols); ++j) {
      if (rich.integral[j] != 0) {
        RichModel candidate = rich;
        candidate.integral[j] = 0;
        changed = keep(std::move(candidate)) || changed;
      }
      if (rich.col_lower[j]) {
        RichModel candidate = rich;
        candidate.col_lower[j].reset();
        changed = keep(std::move(candidate)) || changed;
      }
      if (rich.col_upper[j]) {
        RichModel candidate = rich;
        candidate.col_upper[j].reset();
        changed = keep(std::move(candidate)) || changed;
      }
      if (!rich.cost[j].is_zero()) {
        RichModel candidate = rich;
        candidate.cost[j] = Rational();
        changed = keep(std::move(candidate)) || changed;
      }
    }
    if (!rich.offset.is_zero()) {
      RichModel candidate = rich;
      candidate.offset = Rational();
      changed = keep(std::move(candidate)) || changed;
    }
    if (rich.maximize) {
      RichModel candidate = rich;
      candidate.maximize = false;
      for (Rational& c : candidate.cost) c = -c;
      changed = keep(std::move(candidate)) || changed;
    }
  }
  return rich;
}

}  // namespace sankhya::oracle
