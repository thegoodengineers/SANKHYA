// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the elementwise halves of a PDHG step over the thread pool (#487).
//
// With pdhg_parallel_spmv both sparse products are gathers over a fixed static partition
// (A^T y always was; A x since #587), so what stayed serial in a CPU iteration was
// everything else: the projected primal and dual updates, the extrapolation, the movement
// and interaction sums of the adaptive step rule and the running sums of the average. On a
// model the size of the refinery year those passes over n + m entries are a large share of
// an iteration. pdhg_parallel_updates runs them over the same OpenMP workers.
//
// THE ANSWER DOES NOT DEPEND ON THE THREAD COUNT. The elementwise updates write one entry
// per index, so any partition gives the same bits. The three sums are the only place the
// order matters, and they are taken in fixed chunks of tol::kPdhgParallelChunk entries -
// a chunk size that is a constant, not a function of the thread count - each chunk summed
// left to right, then the chunk totals summed left to right on one thread. At 1, 2, 4 or 8
// threads the same numbers are added in the same order. Against the serial loops, which
// sum all n (or m) entries in one running total, the chunked order differs, so the two
// paths agree to rounding, not to the bit (Higham, *Accuracy and Stability of Numerical
// Algorithms*, 2nd ed., SIAM 2002, section 4.1, on summation order; the static-partition
// argument is #57's).
//
// The iteration is [CP11] Algorithm 1 with the [PDLP] section 3.1 step rule, exactly as the
// serial loops in pdhg.cpp compute it; see the references there.
#pragma once

#include <vector>

#include "../la/scaling.hpp"

namespace sankhya::pdhg {

/// x_next = proj_X(x - tau (c + A^T y)); when `extrapolated` is non-null also 2 x_next - x,
/// and when `dx` is non-null also x_next - x. Returns sum_j 0.5 omega (x_next_j - x_j)^2.
[[nodiscard]] double parallel_primal_step(const Scaling& scaling, const std::vector<double>& x,
                                          const std::vector<double>& at_y, double tau,
                                          double omega, std::vector<double>& x_next,
                                          std::vector<double>* extrapolated,
                                          std::vector<double>* dx);

/// y_next = v - sigma proj_C(v / sigma) with v = y + sigma a_x. Returns
/// sum_i 0.5 (y_next_i - y_i)^2 / omega.
[[nodiscard]] double parallel_dual_step(const Scaling& scaling, const std::vector<double>& y,
                                        const std::vector<double>& a_x, double sigma,
                                        double omega, std::vector<double>& y_next);

/// sum_i (y_next_i - y_i) * (a_i - b_i), with b null meaning zero: the interaction term,
/// from A dx directly or from two cached products (#479).
[[nodiscard]] double parallel_interaction(const std::vector<double>& y_next,
                                          const std::vector<double>& y,
                                          const std::vector<double>& a,
                                          const std::vector<double>* b);

/// out = 2 next - cached (#479's A xbar from A x_{k+1} and A x_k).
void parallel_extrapolate_product(const std::vector<double>& next,
                                  const std::vector<double>& cached, std::vector<double>& out);

/// sum += v, entry by entry.
void parallel_accumulate(std::vector<double>& sum, const std::vector<double>& v);

}  // namespace sankhya::pdhg
