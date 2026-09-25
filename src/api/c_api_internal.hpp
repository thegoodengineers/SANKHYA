// SPDX-License-Identifier: Apache-2.0
// SANKHYA - what the C API's implementation files share: the model handle and the error
// convention. Not a public header; include/sankhya/sankhya.h keeps the handles opaque.
//
// Split out of c_api.cpp when the nonlinear entry points (c_api_nlp.cpp, NLP stage 1) needed
// the same handle and the same thread-local error, and c_api.cpp was already past the
// project's file-size limit.
#pragma once

#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "nlp/nonlinear_model.hpp"
#include "sankhya/model.hpp"
#include "sankhya/sankhya.h"
#include "sankhya/solve_control.hpp"

namespace sankhya::capi {

/// What a C caller built beyond the frozen Model: the expression graph over its columns and
/// the objective term and rows made from it (NLP stage 1). Absent - a null pointer on the
/// handle - until the first expression call or the first .nl read, so a linear model's handle
/// carries nothing it does not use.
struct NonlinearParts {
  nlp::ExpressionGraph graph{0};
  nlp::ExprId objective = nlp::kNoExpr;
  std::vector<nlp::NonlinearConstraint> rows;
  std::vector<double> start;

  [[nodiscard]] bool empty() const noexcept {
    return objective == nlp::kNoExpr && rows.empty();
  }
};

/// Record `message` as this thread's last error and return `code`.
sankhya_status fail(sankhya_status code, std::string message);
/// Clear this thread's last error and return SANKHYA_OK.
sankhya_status ok();

/// Run `body`, converting any exception into a status code.
///
/// An exception crossing into C is undefined behaviour, so every escape route has to be
/// closed - including ones this file does not know about, hence the bare `catch (...)`.
///
/// A template rather than the macro this started as. A function-like macro splits its
/// argument on commas, so a body containing `entries[{row, col}]` arrives as two arguments
/// and the preprocessor rejects it - which it duly did, in four places.
template <typename Body>
sankhya_status guarded(Body&& body) {
  try {
    return body();
  } catch (const std::bad_alloc&) {
    return fail(SANKHYA_ERROR_MEMORY, "out of memory");
  } catch (const std::exception& error) {
    return fail(SANKHYA_ERROR_INTERNAL, error.what());
  } catch (...) {
    return fail(SANKHYA_ERROR_INTERNAL, "an unknown exception crossed the C API");
  }
}

}  // namespace sankhya::capi

// The handle. Defined here so the header can keep it opaque.
struct sankhya_model {
  sankhya::Model model;
  // (row, col) -> value, pending until the matrix is materialised.
  std::map<std::pair<int, int>, double> entries;
  std::map<std::pair<int, int>, double> quadratic;
  /// Null for a model with no expression in it (see NonlinearParts).
  std::unique_ptr<sankhya::capi::NonlinearParts> nonlinear;

  std::mutex control_mutex;
  std::shared_ptr<sankhya::SolveControl> active_control;
  sankhya::ProgressCallback progress_callback;

  sankhya_model() = default;
};

namespace sankhya::capi {

/// The handle's nonlinear parts over `built`, validated: empty when there are none or they
/// are well formed, else the first problem (c_api_nlp.cpp).
std::string validate_nonlinear(const sankhya_model& model, const sankhya::Model& built);

/// Read a .nl file into the handle, replacing what it held (c_api_nlp.cpp).
sankhya_status read_nl_into(sankhya_model* model, const char* path);

/// The solve of a handle whose nonlinear parts are not empty, given the materialised linear
/// part `built` (c_api_nlp.cpp). Returns false when the handle has no nonlinear parts, and
/// the caller solves `built` as it always did.
bool solve_nonlinear(sankhya_model* model, const sankhya::Model& built, const Options& options,
                     SolveControl* control, Solution* out);

}  // namespace sankhya::capi
