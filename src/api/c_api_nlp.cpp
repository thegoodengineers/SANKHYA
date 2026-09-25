// SPDX-License-Identifier: Apache-2.0
// SANKHYA - C API for nonlinear models (NLP stage 1): include/sankhya/sankhya_nonlinear.h.
//
// As in c_api.cpp, every entry point validates its arguments, calls the C++ core and turns
// the outcome into a status; the expression layer itself (src/nlp/expression.hpp) is what
// folds, interns and records a malformed construction, so a handle cannot become an
// expression this file invented.

#include "sankhya/sankhya_nonlinear.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "api/c_api_internal.hpp"
#include "nlp/nl_reader.hpp"
#include "nlp/nonlinear_model.hpp"

namespace sankhya::capi {
namespace {

/// The handle's nonlinear parts, created on first use, over as many columns as it has now.
NonlinearParts& parts(sankhya_model* model) {
  if (!model->nonlinear) model->nonlinear = std::make_unique<NonlinearParts>();
  model->nonlinear->graph.extend_variables(static_cast<Index>(model->model.col_cost.size()));
  return *model->nonlinear;
}

/// A builder's result: the handle, or -1 with the graph's first problem as the error.
int result_of(const NonlinearParts& p, nlp::ExprId id) {
  if (id == nlp::kNoExpr) {
    fail(SANKHYA_ERROR_ARGUMENT,
         p.graph.invalid().empty() ? "the expression could not be built" : p.graph.invalid());
    return -1;
  }
  ok();
  return static_cast<int>(id);
}

/// A handle from C: -1 stays -1 (an earlier failure), anything else must be in the graph.
nlp::ExprId as_id(int handle) {
  return handle < 0 ? nlp::kNoExpr : static_cast<nlp::ExprId>(handle);
}

/// The whole model the handle describes, over `built`.
nlp::NonlinearModel assemble(const sankhya_model& model, const Model& built) {
  nlp::NonlinearModel out(built);
  const NonlinearParts& p = *model.nonlinear;
  out.graph = p.graph;
  out.graph.extend_variables(built.num_cols());
  out.objective = p.objective;
  out.constraints = p.rows;
  out.start = p.start;
  return out;
}

template <typename Build>
int build(sankhya_model* model, Build&& body) {
  if (model == nullptr) {
    fail(SANKHYA_ERROR_ARGUMENT, "model is null");
    return -1;
  }
  int out = -1;
  const sankhya_status status = guarded([&]() -> sankhya_status {
    NonlinearParts& p = parts(model);
    out = result_of(p, body(p.graph));
    return out < 0 ? SANKHYA_ERROR_ARGUMENT : SANKHYA_OK;
  });
  return status == SANKHYA_OK ? out : -1;
}

}  // namespace

std::string validate_nonlinear(const sankhya_model& model, const Model& built) {
  if (!model.nonlinear) return {};
  return assemble(model, built).validate();
}

sankhya_status read_nl_into(sankhya_model* model, const char* path) {
  return guarded([&]() -> sankhya_status {
    std::unique_ptr<nlp::NonlinearModel> read;
    const io::ReadResult result = nlp::read_nl(path, &read);
    if (!result.ok) return fail(SANKHYA_ERROR_IO, result.error);
    // Only on success, as for the linear readers.
    auto p = std::make_unique<NonlinearParts>();
    p->graph = std::move(read->graph);
    p->objective = read->objective;
    p->rows = std::move(read->constraints);
    p->start = std::move(read->start);
    model->model = std::move(read->base);
    model->entries.clear();
    model->quadratic.clear();
    model->nonlinear = std::move(p);
    return ok();
  });
}

bool solve_nonlinear(sankhya_model* model, const Model& built, const Options& options,
                     SolveControl* control, Solution* out) {
  (void)options;
  (void)control;
  if (!model->nonlinear || model->nonlinear->empty()) return false;
  const nlp::NonlinearModel whole = assemble(*model, built);
  const std::string problem = whole.validate();
  out->status = SolveStatus::kModelError;
  if (!problem.empty()) {
    out->message = problem;
    return true;
  }
  // Stage 1 builds and reads nonlinear models; the engine that solves a general one is not in
  // this build. Saying so is the whole answer: solving the linear part alone would be solving
  // a different problem.
  out->status = SolveStatus::kNotSolved;
  out->message = fmt::format(
      "model class {}: this build represents and differentiates it but has no engine for it",
      nlp::to_string(whole.classify()));
  return true;
}

}  // namespace sankhya::capi

using sankhya::capi::fail;
using sankhya::capi::guarded;
using sankhya::capi::ok;

extern "C" {

int sankhya_expr_constant(sankhya_model* model, double value) {
  return sankhya::capi::build(model, [&](auto& g) { return g.constant(value); });
}

int sankhya_expr_variable(sankhya_model* model, int col) {
  return sankhya::capi::build(model, [&](auto& g) { return g.variable(col); });
}

int sankhya_expr_unary(sankhya_model* model, int op, int argument) {
  return sankhya::capi::build(model, [&](auto& g) -> sankhya::nlp::ExprId {
    const auto a = sankhya::capi::as_id(argument);
    switch (op) {
      case SANKHYA_EXPR_NEG: return g.negate(a);
      case SANKHYA_EXPR_EXP: return g.exp(a);
      case SANKHYA_EXPR_LOG: return g.log(a);
      case SANKHYA_EXPR_SQRT: return g.sqrt(a);
      case SANKHYA_EXPR_SIN: return g.sin(a);
      case SANKHYA_EXPR_COS: return g.cos(a);
      default: return sankhya::nlp::kNoExpr;
    }
  });
}

int sankhya_expr_binary(sankhya_model* model, int op, int left, int right) {
  return sankhya::capi::build(model, [&](auto& g) -> sankhya::nlp::ExprId {
    const auto a = sankhya::capi::as_id(left);
    const auto b = sankhya::capi::as_id(right);
    switch (op) {
      case SANKHYA_EXPR_ADD: return g.add(a, b);
      case SANKHYA_EXPR_SUB: return g.subtract(a, b);
      case SANKHYA_EXPR_MUL: return g.multiply(a, b);
      case SANKHYA_EXPR_DIV: return g.divide(a, b);
      default: return sankhya::nlp::kNoExpr;
    }
  });
}

int sankhya_expr_power(sankhya_model* model, int base, double exponent) {
  return sankhya::capi::build(
      model, [&](auto& g) { return g.power(sankhya::capi::as_id(base), exponent); });
}

int sankhya_expr_sum(sankhya_model* model, int count, const int* terms) {
  if (count < 0 || (count > 0 && terms == nullptr)) {
    fail(SANKHYA_ERROR_ARGUMENT, "sum: a negative count or a null term array");
    return -1;
  }
  return sankhya::capi::build(model, [&](auto& g) {
    std::vector<sankhya::nlp::ExprId> ids;
    for (int k = 0; k < count; ++k) ids.push_back(sankhya::capi::as_id(terms[k]));
    return g.sum(std::move(ids));
  });
}

sankhya_status sankhya_model_set_nonlinear_objective(sankhya_model* model, int expression) {
  if (model == nullptr) return fail(SANKHYA_ERROR_ARGUMENT, "model is null");
  return guarded([&]() -> sankhya_status {
    sankhya::capi::NonlinearParts& p = sankhya::capi::parts(model);
    if (expression >= 0 && !p.graph.contains(expression)) {
      return fail(SANKHYA_ERROR_ARGUMENT,
                  fmt::format("{} is not an expression of this model", expression));
    }
    p.objective = sankhya::capi::as_id(expression);
    return ok();
  });
}

sankhya_status sankhya_model_add_nonlinear_row(sankhya_model* model, int expression,
                                               double lower, double upper, const char* name,
                                               int* index) {
  if (model == nullptr) return fail(SANKHYA_ERROR_ARGUMENT, "model is null");
  return guarded([&]() -> sankhya_status {
    sankhya::capi::NonlinearParts& p = sankhya::capi::parts(model);
    if (!p.graph.contains(expression)) {
      return fail(SANKHYA_ERROR_ARGUMENT,
                  fmt::format("{} is not an expression of this model", expression));
    }
    const int position = static_cast<int>(p.rows.size());
    p.rows.push_back({expression, sankhya::normalize_infinity(lower),
                      sankhya::normalize_infinity(upper),
                      name != nullptr ? std::string(name) : fmt::format("N{}", position + 1)});
    if (index != nullptr) *index = position;
    return ok();
  });
}

int sankhya_model_num_nonlinear_rows(const sankhya_model* model) {
  return model == nullptr || !model->nonlinear
             ? 0
             : static_cast<int>(model->nonlinear->rows.size());
}

sankhya_status sankhya_model_set_start(sankhya_model* model, const double* x, int count) {
  if (model == nullptr || x == nullptr)
    return fail(SANKHYA_ERROR_ARGUMENT, "model or x is null");
  if (count != static_cast<int>(model->model.col_cost.size())) {
    return fail(SANKHYA_ERROR_ARGUMENT, fmt::format("a start of {} entries for {} columns",
                                                    count, model->model.col_cost.size()));
  }
  return guarded([&]() -> sankhya_status {
    sankhya::capi::parts(model).start.assign(x, x + count);
    return ok();
  });
}

sankhya_status sankhya_expr_evaluate(const sankhya_model* model, int expression,
                                     const double* x, int count, double* value) {
  if (model == nullptr || x == nullptr || value == nullptr) {
    return fail(SANKHYA_ERROR_ARGUMENT, "model, x or value is null");
  }
  return guarded([&]() -> sankhya_status {
    if (!model->nonlinear || !model->nonlinear->graph.contains(expression)) {
      return fail(SANKHYA_ERROR_ARGUMENT,
                  fmt::format("{} is not an expression of this model", expression));
    }
    const sankhya::nlp::ExpressionGraph& graph = model->nonlinear->graph;
    const int columns = static_cast<int>(model->model.col_cost.size());
    if (count != columns) {
      return fail(SANKHYA_ERROR_ARGUMENT,
                  fmt::format("a point of {} entries for {} columns", count, columns));
    }
    // The graph spans the columns that existed when it last grew; a column added since cannot
    // appear in any of its expressions, so the point's leading entries are all it reads.
    const sankhya::nlp::Evaluation e =
        graph.evaluate(expression, std::vector<double>(x, x + graph.num_variables()));
    if (!e.ok()) return fail(SANKHYA_ERROR_ARGUMENT, e.message);
    *value = e.value;
    return ok();
  });
}

}  // extern "C"
