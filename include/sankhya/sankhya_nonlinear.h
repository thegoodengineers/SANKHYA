/* SPDX-License-Identifier: Apache-2.0 */
/*
 * SANKHYA - C API for nonlinear models (NLP stage 1).
 *
 * A model built through sankhya.h can also carry EXPRESSIONS over its columns: a nonlinear
 * objective term added to its linear and quadratic objective, and nonlinear rows
 *     lower <= expression(x) <= upper.
 * Expressions are built bottom-up and named by an int handle (>= 0). The operations are the
 * ones the solver evaluates and differentiates exactly: constants, columns, + - * /, a power
 * with a constant exponent, a sum of many terms, exp, log (natural), sqrt, sin, cos.
 *
 * ERRORS ARE STICKY. A builder given a column out of range, a non-finite constant or a handle
 * that is not an expression of this model returns -1 and sets sankhya_last_error(); every
 * builder given -1 returns -1, and the model then fails sankhya_model_validate() and
 * sankhya_solve() with the FIRST problem. A malformed expression cannot become a plausible
 * one on the way to the solver.
 *
 * COLUMNS FIRST OR LATER. A column added after an expression was built is available to the
 * next one; handles already returned stay valid.
 *
 * sankhya_model_read() of a ".nl" file (AMPL's text format; see src/nlp/nl_reader.hpp) fills
 * the same parts.
 */
#ifndef SANKHYA_NONLINEAR_H
#define SANKHYA_NONLINEAR_H

#include "sankhya/sankhya.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Operations of one argument. */
typedef enum {
  SANKHYA_EXPR_NEG = 0,
  SANKHYA_EXPR_EXP = 1,
  SANKHYA_EXPR_LOG = 2,
  SANKHYA_EXPR_SQRT = 3,
  SANKHYA_EXPR_SIN = 4,
  SANKHYA_EXPR_COS = 5
} sankhya_unary_op;

/** Operations of two arguments: a + b, a - b, a * b, a / b. */
typedef enum {
  SANKHYA_EXPR_ADD = 0,
  SANKHYA_EXPR_SUB = 1,
  SANKHYA_EXPR_MUL = 2,
  SANKHYA_EXPR_DIV = 3
} sankhya_binary_op;

/* Builders: each returns the new expression's handle, or -1 with sankhya_last_error() set. */
int sankhya_expr_constant(sankhya_model* model, double value);
int sankhya_expr_variable(sankhya_model* model, int col);
int sankhya_expr_unary(sankhya_model* model, int op, int argument);
int sankhya_expr_binary(sankhya_model* model, int op, int left, int right);
/** base ^ exponent, the exponent a constant. */
int sankhya_expr_power(sankhya_model* model, int base, double exponent);
/** terms[0] + ... + terms[count - 1]; count 0 is the constant 0. */
int sankhya_expr_sum(sankhya_model* model, int count, const int* terms);

/** Add `expression` to the objective (in the model's sense). -1 removes the term. */
sankhya_status sankhya_model_set_nonlinear_objective(sankhya_model* model, int expression);

/**
 * Add the row lower <= expression <= upper. Use +/- sankhya_infinity() for an absent side.
 * `name` may be NULL. Nonlinear rows are numbered after the linear rows in every solution
 * vector indexed by row.
 */
sankhya_status sankhya_model_add_nonlinear_row(sankhya_model* model, int expression,
                                               double lower, double upper, const char* name,
                                               int* index);

/** The number of nonlinear rows. */
int sankhya_model_num_nonlinear_rows(const sankhya_model* model);

/**
 * A starting point for a local method, `count` = the number of columns. A nonlinear solve
 * starts here; without one it starts from the column bounds' interior.
 */
sankhya_status sankhya_model_set_start(sankhya_model* model, const double* x, int count);

/**
 * The value of `expression` at x (`count` = the number of columns). A point outside the
 * expression's domain - log of a non-positive number, say - is SANKHYA_ERROR_ARGUMENT with
 * the reason, never a NaN in `value`.
 */
sankhya_status sankhya_expr_evaluate(const sankhya_model* model, int expression,
                                     const double* x, int count, double* value);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SANKHYA_NONLINEAR_H */
