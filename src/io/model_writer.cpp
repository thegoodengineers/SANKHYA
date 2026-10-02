// SPDX-License-Identifier: Apache-2.0
// SANKHYA - MPS and LP model writers.
//
// These writers are the inverse of the readers in mps_reader.cpp and lp_reader.cpp.
// Round-trip identity (read -> write -> read -> same model) is the acceptance test:
// every supported field must survive the cycle at full double precision.
//
// MPS format reference: IBM, "MPS file format" (the de-facto specification).
// LP format reference: CPLEX LP dialect as documented in the public CPLEX reference manual.
// Both are the same dialects the readers accept, so the round-trip is testable end-to-end.
//
// The RANGES and BOUNDS inverse mappings here are the exact mirror of the forward mappings
// in mps_reader.cpp's finish_rows() and mps_bounds.cpp's do_bounds(). They are kept in
// close correspondence; a change to either side should touch both.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/sparse.hpp"
#include "sankhya/types.hpp"

#include "core/sc_sos.hpp"
#include "model_writer_sos.hpp"

namespace sankhya::io {
namespace {

/// The names a file will carry, decided once per write. A real name is kept as given; an
/// absent one is generated, and every generated name is made unique against every real name
/// and every earlier generated one - an unnamed column 3 beside a real column named "C3"
/// used to write two "C3" entries, which a reader binds to one column: a different model,
/// silently. Both formats split on whitespace and neither quotes, so a real name that
/// contains whitespace, or two real names that are equal, is refused with a message rather
/// than written wrong.
struct Names {
  std::vector<std::string> col;
  std::vector<std::string> row;
};

[[nodiscard]] bool build_names(const std::vector<std::string>& given, Index count, char prefix,
                               const char* what, std::vector<std::string>* out,
                               std::string* error) {
  out->assign(static_cast<std::size_t>(count), std::string());
  std::unordered_set<std::string> taken;
  for (Index k = 0; k < count; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (u >= given.size() || given[u].empty()) continue;
    const std::string& name = given[u];
    const bool has_whitespace = std::any_of(
        name.begin(), name.end(), [](unsigned char c) { return std::isspace(c) != 0; });
    if (has_whitespace) {
      if (error != nullptr) {
        *error = fmt::format(
            "{} {} is named \"{}\", which contains whitespace; neither MPS nor LP can carry "
            "such a name",
            what, k, name);
      }
      return false;
    }
    if (!taken.insert(name).second) {
      if (error != nullptr) {
        *error = fmt::format(
            "{} name \"{}\" is used twice; a file with duplicate names reads back as a "
            "different model",
            what, name);
      }
      return false;
    }
    (*out)[u] = name;
  }
  for (Index k = 0; k < count; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (!(*out)[u].empty()) continue;
    std::string candidate = fmt::format("{}{}", prefix, k);
    for (int suffix = 1; !taken.insert(candidate).second; ++suffix) {
      candidate = fmt::format("{}{}_{}", prefix, k, suffix);
    }
    (*out)[u] = candidate;
  }
  return true;
}

[[nodiscard]] bool build_names(const Model& model, Names* names, std::string* error) {
  return build_names(model.col_names, model.num_cols(), 'C', "column", &names->col, error) &&
         build_names(model.row_names, model.num_rows(), 'R', "row", &names->row, error);
}

/// Round-trip-safe decimal representation (17 significant digits guarantees bit-for-bit
/// round-trip for any IEEE 754 double; fmt's default {} gives the shortest such repr).
[[nodiscard]] std::string xfmt(double v) {
  if (v == kInfinity) return "inf";
  if (v == -kInfinity) return "-inf";
  double nv = (v == 0.0) ? 0.0 : v;  // strip negative zero
  return fmt::format("{:.17g}", nv);
}

/// Pick an objective row name that does not collide with any constraint row's name.
[[nodiscard]] std::string pick_obj_row(const Names& names) {
  static const char* kCandidates[] = {"obj", "COST", "OBJ", "OBJROW", "_obj"};
  for (const char* candidate : kCandidates) {
    bool clash = false;
    for (const auto& name : names.row) {
      if (name == candidate) {
        clash = true;
        break;
      }
    }
    if (!clash) return candidate;
  }
  return "OBJROW_SANKHYA";
}

// -----------------------------------------------------------------------------------------
// MPS row type classification (mirror of finish_rows in mps_reader.cpp)
// -----------------------------------------------------------------------------------------

enum class MpsRowKind { kFree, kLe, kGe, kEq };

[[nodiscard]] MpsRowKind mps_kind(const Model& model, Index i) {
  const auto u = static_cast<std::size_t>(i);
  const double lo = model.row_lower[u];
  const double hi = model.row_upper[u];
  if (lo == -kInfinity && hi == kInfinity) return MpsRowKind::kFree;
  if (std::isfinite(lo) && lo == hi) return MpsRowKind::kEq;
  if (lo == -kInfinity) return MpsRowKind::kLe;
  return MpsRowKind::kGe;  // >= or ranged; both written as G with RANGES for the upper bound
}

// -----------------------------------------------------------------------------------------
// LP expression helper
// -----------------------------------------------------------------------------------------

/// Append a signed term "+/- coef var" to an LP expression buffer.
/// `first` tracks whether we have already written any terms (controls the leading sign).
void append_lp_term(std::string& buf, double coef, const std::string& var, bool& first) {
  if (coef == 0.0) return;
  if (first) {
    if (coef == 1.0) {
      buf += var;
    } else if (coef == -1.0) {
      buf += "-";
      buf += var;
    } else {
      buf += fmt::format("{:.17g} {}", coef, var);
    }
    first = false;
  } else {
    if (coef == 1.0) {
      buf += fmt::format(" + {}", var);
    } else if (coef == -1.0) {
      buf += fmt::format(" - {}", var);
    } else if (coef > 0.0) {
      buf += fmt::format(" + {:.17g} {}", coef, var);
    } else {
      buf += fmt::format(" - {:.17g} {}", -coef, var);
    }
  }
}

}  // namespace

// -----------------------------------------------------------------------------------------
// MPS writer
// -----------------------------------------------------------------------------------------

bool write_mps(const std::string& path, const Model& model, std::string* error) {
  Names names;
  if (!build_names(model, &names, error)) return false;
  std::FILE* out = std::fopen(path.c_str(), "wb");
  if (out == nullptr) {
    if (error != nullptr) *error = fmt::format("{}: cannot open for writing", path);
    return false;
  }

  const Index m = model.num_rows();
  const Index n = model.num_cols();
  const std::string obj_row = pick_obj_row(names);

  // MPS convention treats every N row beyond the objective as a free row and readers drop
  // them (mps_reader.cpp). Warn so the caller knows the round-trip loses these rows.
  Index n_free_mps = 0;
  for (Index i = 0; i < m; ++i) {
    if (mps_kind(model, i) == MpsRowKind::kFree) ++n_free_mps;
  }
  if (n_free_mps > 0) {
    fmt::print(stderr,
               "write_mps: warning: {} free row(s) written as extra N rows, which MPS readers "
               "drop; the round-trip loses them\n",
               n_free_mps);
  }

  // ---- NAME ----
  fmt::print(out, "NAME          {}\n", model.name.empty() ? "UNNAMED" : model.name);

  // ---- OBJSENSE (only for maximization; minimize is the implicit default) ----
  if (model.sense == ObjSense::kMaximize) {
    fmt::print(out, "OBJSENSE\n    MAX\n");
  }

  // ---- ROWS ----
  fmt::print(out, "ROWS\n N  {}\n", obj_row);
  for (Index i = 0; i < m; ++i) {
    char type;
    switch (mps_kind(model, i)) {
      case MpsRowKind::kFree: type = 'N'; break;
      case MpsRowKind::kLe: type = 'L'; break;
      case MpsRowKind::kGe: type = 'G'; break;  // also ranged rows
      case MpsRowKind::kEq: type = 'E'; break;
    }
    fmt::print(out, " {}  {}\n", type, names.row[static_cast<std::size_t>(i)]);
  }

  // ---- COLUMNS ----
  // Integer columns are wrapped with MARKER INTORG / INTEND records. The counter in
  // the marker names just needs to be unique within the file; the reader scans for the
  // 'MARKER' token and ignores the name.
  fmt::print(out, "COLUMNS\n");
  bool in_integer = false;
  int marker_count = 0;

  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const bool is_int = (model.col_type[u] == VarType::kInteger);

    if (is_int && !in_integer) {
      fmt::print(out, "    M{:07d}  'MARKER'                 'INTORG'\n", marker_count++);
      in_integer = true;
    } else if (!is_int && in_integer) {
      fmt::print(out, "    M{:07d}  'MARKER'                 'INTEND'\n", marker_count++);
      in_integer = false;
    }

    const std::string cname = names.col[static_cast<std::size_t>(j)];

    // Build the list of (row_name, value) pairs for this column.
    std::vector<std::pair<std::string, double>> entries;
    if (model.col_cost[u] != 0.0) {
      entries.emplace_back(obj_row, model.col_cost[u]);
    }
    const ColumnView cv = model.matrix.column(j);
    for (Index k = 0; k < cv.size; ++k) {
      if (cv.values[k] != 0.0) {
        entries.emplace_back(names.row[static_cast<std::size_t>(cv.rows[k])], cv.values[k]);
      }
    }

    // MPS convention allows up to two row/value pairs per COLUMNS line.
    // A column with no entries still needs an appearance; write a zero on the obj row.
    if (entries.empty()) {
      fmt::print(out, "    {}  {}  0\n", cname, obj_row);
    } else {
      for (std::size_t e = 0; e < entries.size(); e += 2) {
        if (e + 1 < entries.size()) {
          fmt::print(out, "    {}  {}  {}  {}  {}\n", cname, entries[e].first,
                     xfmt(entries[e].second), entries[e + 1].first,
                     xfmt(entries[e + 1].second));
        } else {
          fmt::print(out, "    {}  {}  {}\n", cname, entries[e].first, xfmt(entries[e].second));
        }
      }
    }
  }

  if (in_integer) {
    fmt::print(out, "    M{:07d}  'MARKER'                 'INTEND'\n", marker_count++);
  }

  // ---- RHS ----
  // Objective row: RHS = -objective_offset (model stores +offset; reader negates it).
  // Constraint rows: RHS = the active bound (lower for G/E, upper for L).
  fmt::print(out, "RHS\n");
  if (model.objective_offset != 0.0) {
    fmt::print(out, "    RHS  {}  {}\n", obj_row, xfmt(-model.objective_offset));
  }
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double lo = model.row_lower[u];
    const double hi = model.row_upper[u];
    double rhs = 0.0;
    switch (mps_kind(model, i)) {
      case MpsRowKind::kFree: continue;
      case MpsRowKind::kLe: rhs = hi; break;
      case MpsRowKind::kGe: rhs = lo; break;  // also ranged: RHS = lower bound
      case MpsRowKind::kEq: rhs = lo; break;
    }
    if (rhs != 0.0) {
      fmt::print(out, "    RHS  {}  {}\n", names.row[static_cast<std::size_t>(i)], xfmt(rhs));
    }
  }

  // ---- RANGES ----
  // For G rows with a finite upper bound (ranged rows): range = upper - lower.
  // The reader reconstructs [lo, lo + |range|] from (G, lo, range), which equals [lo, hi].
  bool any_range = false;
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double lo = model.row_lower[u];
    const double hi = model.row_upper[u];
    if (std::isfinite(lo) && std::isfinite(hi) && lo < hi) {
      if (!any_range) {
        fmt::print(out, "RANGES\n");
        any_range = true;
      }
      fmt::print(out, "    RNG  {}  {}\n", names.row[static_cast<std::size_t>(i)],
                 xfmt(hi - lo));
    }
  }

  // ---- BOUNDS ----
  // Default is [0, +inf) for continuous and integer columns.
  // See mps_bounds.cpp for the inverse of every case written here.
  bool any_bounds = false;
  auto bounds_header = [&]() {
    if (!any_bounds) {
      fmt::print(out, "BOUNDS\n");
      any_bounds = true;
    }
  };

  const std::vector<char> semicontinuous = semicontinuous_mask(model);  // #754
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double lo = model.col_lower[u];
    const double hi = model.col_upper[u];
    const bool is_int = (model.col_type[u] == VarType::kInteger);
    const std::string cname = names.col[static_cast<std::size_t>(j)];

    if (!semicontinuous.empty() && semicontinuous[u] != 0) {
      bounds_header();
      write_mps_semicontinuous_bounds(out, cname, lo, hi, is_int);
      continue;
    }
    if (is_int) {
      // Binary: [0, 1] integer — single BV line overrides both bounds and marks integer.
      if (lo == 0.0 && hi == 1.0) {
        bounds_header();
        fmt::print(out, " BV BND  {}\n", cname);
        continue;
      }
      // Default integer [0, +inf): already covered by INTORG/INTEND; no BOUNDS line needed.
      if (lo == 0.0 && hi == kInfinity) continue;
      // Fixed integer.
      if (lo == hi) {
        bounds_header();
        fmt::print(out, " FX BND  {}  {}\n", cname, xfmt(lo));
        continue;
      }
      // Free integer.
      if (lo == -kInfinity && hi == kInfinity) {
        bounds_header();
        fmt::print(out, " FR BND  {}\n", cname);
        continue;
      }
      // Integer with non-standard lower bound.
      if (lo != 0.0) {
        bounds_header();
        if (lo == -kInfinity) {
          fmt::print(out, " MI BND  {}\n", cname);
        } else {
          fmt::print(out, " LI BND  {}  {}\n", cname, xfmt(lo));
        }
      }
      // Integer with finite upper bound.
      if (hi != kInfinity) {
        bounds_header();
        fmt::print(out, " UI BND  {}  {}\n", cname, xfmt(hi));
      }
    } else {
      // Continuous default: [0, +inf).
      if (lo == 0.0 && hi == kInfinity) continue;
      // Fixed.
      if (lo == hi) {
        bounds_header();
        fmt::print(out, " FX BND  {}  {}\n", cname, xfmt(lo));
        continue;
      }
      // Free continuous.
      if (lo == -kInfinity && hi == kInfinity) {
        bounds_header();
        fmt::print(out, " FR BND  {}\n", cname);
        continue;
      }
      // Lower bound is -inf: write MI, then UP if upper is finite.
      if (lo == -kInfinity) {
        bounds_header();
        fmt::print(out, " MI BND  {}\n", cname);
        if (hi != kInfinity) {
          fmt::print(out, " UP BND  {}  {}\n", cname, xfmt(hi));
        }
        continue;
      }
      // Non-default lower bound.
      if (lo != 0.0) {
        bounds_header();
        fmt::print(out, " LO BND  {}  {}\n", cname, xfmt(lo));
      }
      // Non-default upper bound.
      if (hi != kInfinity) {
        bounds_header();
        fmt::print(out, " UP BND  {}  {}\n", cname, xfmt(hi));
      }
    }
  }

  // ---- QUADOBJ ----
  // Lower-triangular Hessian entries, one per line: col col value.
  // The reader normalises (max, min) column order, so the triangle convention is preserved.
  if (model.has_quadratic_objective()) {
    fmt::print(out, "QUADOBJ\n");
    const Index hcols = model.hessian.num_cols();
    for (Index j = 0; j < hcols; ++j) {
      const ColumnView cv = model.hessian.column(j);
      const std::string cj = names.col[static_cast<std::size_t>(j)];
      for (Index k = 0; k < cv.size; ++k) {
        if (cv.values[k] != 0.0) {
          // Hessian is stored lower-triangular: col.rows[k] >= j.
          fmt::print(out, "    {}  {}  {}\n", names.col[static_cast<std::size_t>(cv.rows[k])],
                     cj, xfmt(cv.values[k]));
        }
      }
    }
  }

  write_mps_sos(out, model, names.col);  // #754

  fmt::print(out, "ENDATA\n");
  const bool ok = (std::fclose(out) == 0);
  if (!ok && error != nullptr) *error = fmt::format("{}: write failed", path);
  return ok;
}

// -----------------------------------------------------------------------------------------
// LP writer
// -----------------------------------------------------------------------------------------

bool write_lp(const std::string& path, const Model& model, std::string* error) {
  if (model.has_quadratic_objective()) {
    if (error != nullptr)
      *error = "the LP writer cannot encode a quadratic objective; use write_mps instead";
    return false;
  }

  Names names;
  if (!build_names(model, &names, error)) return false;
  std::FILE* out = std::fopen(path.c_str(), "wb");
  if (out == nullptr) {
    if (error != nullptr) *error = fmt::format("{}: cannot open for writing", path);
    return false;
  }

  const Index m = model.num_rows();
  const Index n = model.num_cols();

  // Count and warn about free rows; they have no LP syntax and are silently skipped.
  Index n_free_lp = 0;
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    if (model.row_lower[u] == -kInfinity && model.row_upper[u] == kInfinity) ++n_free_lp;
  }
  if (n_free_lp > 0) {
    fmt::print(stderr,
               "write_lp: warning: dropped {} free row(s), which neither MPS nor "
               "LP format can express\n",
               n_free_lp);
  }

  // ---- Objective ----
  fmt::print(out, "{}\n", model.sense == ObjSense::kMaximize ? "Maximize" : "Minimize");

  {
    std::string expr;
    bool first = true;
    for (Index j = 0; j < n; ++j) {
      const auto u = static_cast<std::size_t>(j);
      append_lp_term(expr, model.col_cost[u], names.col[static_cast<std::size_t>(j)], first);
    }
    // Objective offset: write as a trailing constant.
    if (model.objective_offset != 0.0) {
      if (first) {
        expr += fmt::format("{:.17g}", model.objective_offset);
        first = false;
      } else if (model.objective_offset > 0.0) {
        expr += fmt::format(" + {:.17g}", model.objective_offset);
      } else {
        expr += fmt::format(" - {:.17g}", -model.objective_offset);
      }
    }
    if (expr.empty()) expr = "0";
    fmt::print(out, "  obj: {}\n", expr);
  }

  // ---- Subject To ----
  // Build a CSR (row-major) view once so the constraint loop is O(m + nnz), not
  // O(m * nnz) as it would be if we scanned every CSC column for each row.
  const CsrView by_row(model.matrix);

  fmt::print(out, "Subject To\n");
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double lo = model.row_lower[u];
    const double hi = model.row_upper[u];

    // Free rows (N type) are not constraints.
    if (lo == -kInfinity && hi == kInfinity) continue;

    const std::string rname = names.row[static_cast<std::size_t>(i)];

    // Build expression using the row view; column indices are in rv.rows[k].
    std::string expr;
    bool first = true;
    const ColumnView rv = by_row.row(i);
    for (Index k = 0; k < rv.size; ++k) {
      append_lp_term(expr, rv.values[k], names.col[static_cast<std::size_t>(rv.rows[k])],
                     first);
    }
    if (expr.empty()) expr = "0";

    if (std::isfinite(lo) && std::isfinite(hi) && lo < hi) {
      // Ranged: lo <= expr <= hi (LP dialect supports this syntax; see lp_reader.cpp).
      fmt::print(out, "  {}: {:.17g} <= {} <= {:.17g}\n", rname, lo, expr, hi);
    } else if (std::isfinite(lo) && lo == hi) {
      // Equality.
      fmt::print(out, "  {}: {} = {:.17g}\n", rname, expr, lo);
    } else if (lo == -kInfinity) {
      // Upper bounded.
      fmt::print(out, "  {}: {} <= {:.17g}\n", rname, expr, hi);
    } else {
      // Lower bounded (no finite upper, or standard G).
      fmt::print(out, "  {}: {} >= {:.17g}\n", rname, expr, lo);
    }
  }

  // ---- Bounds ----
  // Default is [0, +inf) for all variable types. Only non-default bounds are written.
  bool any_bounds = false;
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double lo = model.col_lower[u];
    const double hi = model.col_upper[u];
    if (lo == 0.0 && hi == kInfinity) continue;  // default

    if (!any_bounds) {
      fmt::print(out, "Bounds\n");
      any_bounds = true;
    }
    const std::string cname = names.col[static_cast<std::size_t>(j)];

    if (lo == -kInfinity && hi == kInfinity) {
      fmt::print(out, "  {} free\n", cname);
    } else if (lo == hi) {
      fmt::print(out, "  {:.17g} <= {} <= {:.17g}\n", lo, cname, hi);
    } else if (lo == -kInfinity) {
      fmt::print(out, "  -{} <= {} <= {:.17g}\n", "inf", cname, hi);
    } else if (hi == kInfinity) {
      fmt::print(out, "  {:.17g} <= {}\n", lo, cname);
    } else {
      fmt::print(out, "  {:.17g} <= {} <= {:.17g}\n", lo, cname, hi);
    }
  }

  // ---- General (integer) and Binary sections ----
  std::string general_names;
  std::string binary_names;
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (model.col_type[u] == VarType::kInteger) {
      const std::string cname = names.col[static_cast<std::size_t>(j)];
      if (model.col_lower[u] == 0.0 && model.col_upper[u] == 1.0) {
        binary_names += "  ";
        binary_names += cname;
        binary_names += "\n";
      } else {
        general_names += "  ";
        general_names += cname;
        general_names += "\n";
      }
    }
  }
  if (!general_names.empty()) {
    fmt::print(out, "General\n{}", general_names);
  }
  if (!binary_names.empty()) {
    fmt::print(out, "Binary\n{}", binary_names);
  }
  write_lp_sc_sos(out, model, names.col);  // #754

  fmt::print(out, "End\n");
  const bool ok = (std::fclose(out) == 0);
  if (!ok && error != nullptr) *error = fmt::format("{}: write failed", path);
  return ok;
}

// -----------------------------------------------------------------------------------------
// Format-dispatching writer
// -----------------------------------------------------------------------------------------

bool write_model(const std::string& path, const Model& model, std::string* error) {
  // Pick the writer from the extension: .lp -> LP, everything else -> MPS.
  const std::size_t dot = path.rfind('.');
  if (dot != std::string::npos) {
    const std::string ext = path.substr(dot);
    // Case-insensitive comparison for common casings.
    std::string lower_ext = ext;
    for (char& c : lower_ext)
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (lower_ext == ".lp") return write_lp(path, model, error);
  }
  return write_mps(path, model, error);
}

}  // namespace sankhya::io
