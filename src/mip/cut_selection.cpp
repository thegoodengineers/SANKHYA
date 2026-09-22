// SPDX-License-Identifier: Apache-2.0
// SANKHYA - cut selection (#415). References and the score on the declaration.

#include "cut_selection.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "sankhya/tolerances.hpp"

namespace sankhya::mip {
namespace {

double norm_of(const std::vector<double>& v) {
  double sum = 0.0;
  for (const double x : v) sum += x * x;
  return std::sqrt(sum);
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double sum = 0.0;
  const std::size_t n = std::min(a.size(), b.size());
  for (std::size_t k = 0; k < n; ++k) sum += a[k] * b[k];
  return sum;
}

}  // namespace

CutScore score_cut(const Model& model, const std::vector<double>& point, const Cut& cut) {
  CutScore s;
  const double a_norm = norm_of(cut.coeff);
  if (a_norm <= tol::kZeroDrop) return s;
  s.efficacy = (dot(cut.coeff, point) - cut.rhs) / a_norm;
  // Parallelism is sign-free, so the model's sense does not enter.
  const double c_norm = norm_of(model.col_cost);
  if (c_norm > tol::kZeroDrop) {
    s.objective_parallelism = std::fabs(dot(cut.coeff, model.col_cost)) / (a_norm * c_norm);
  }
  Index nonzeros = 0;
  Index on_integers = 0;
  for (std::size_t j = 0; j < cut.coeff.size(); ++j) {
    if (std::fabs(cut.coeff[j]) <= tol::kZeroDrop) continue;
    ++nonzeros;
    if (j < model.col_type.size() && model.col_type[j] == VarType::kInteger) ++on_integers;
  }
  if (nonzeros > 0) {
    s.integer_support = static_cast<double>(on_integers) / static_cast<double>(nonzeros);
  }
  s.score = tol::kCutSelectionEfficacyWeight * s.efficacy +
            tol::kCutSelectionObjectiveWeight * s.objective_parallelism +
            tol::kCutSelectionIntegerSupportWeight * s.integer_support;
  return s;
}

CutSelection select_cuts(const Model& model, const std::vector<double>& point,
                         std::vector<Cut> candidates, Index max_per_round,
                         double max_parallelism) {
  CutSelection out;
  const std::size_t n = candidates.size();
  std::vector<CutScore> scores(n);
  std::vector<double> norms(n, 0.0);
  for (std::size_t k = 0; k < n; ++k) {
    scores[k] = score_cut(model, point, candidates[k]);
    norms[k] = norm_of(candidates[k].coeff);
  }
  // Best score first; a tie keeps the candidates' own order, so a rerun selects the same.
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&scores](std::size_t p, std::size_t q) {
    return scores[p].score > scores[q].score;
  });

  std::vector<std::size_t> taken;
  std::vector<std::size_t> waiting;
  for (const std::size_t k : order) {
    if (norms[k] <= tol::kZeroDrop) continue;  // no direction: nothing to add or defer
    bool parallel = false;
    if (static_cast<Index>(taken.size()) < max_per_round) {
      for (const std::size_t t : taken) {
        const double cosine =
            std::fabs(dot(candidates[k].coeff, candidates[t].coeff)) / (norms[k] * norms[t]);
        if (cosine > max_parallelism) {
          parallel = true;
          break;
        }
      }
      if (!parallel) {
        taken.push_back(k);
        continue;
      }
    }
    waiting.push_back(k);
  }
  out.selected.reserve(taken.size());
  out.selected_scores.reserve(taken.size());
  for (const std::size_t k : taken) {
    out.selected.push_back(std::move(candidates[k]));
    out.selected_scores.push_back(scores[k]);
  }
  const std::size_t keep =
      std::min(waiting.size(), static_cast<std::size_t>(tol::kCutWaitingLimit));
  out.deferred.reserve(keep);
  for (std::size_t w = 0; w < keep; ++w)
    out.deferred.push_back(std::move(candidates[waiting[w]]));
  return out;
}

}  // namespace sankhya::mip
