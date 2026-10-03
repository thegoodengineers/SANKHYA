// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the learned engine selection's decision tree (#477). PLACEHOLDER until the
// training run: bench/runners/engine_selection_cart.py --emit overwrites this file with the
// trained tree. An empty domain means the tree is never consulted, so
// algorithm_selection=learned falls back to the rule table and says so.

#include "core/engine_selection_tree.hpp"

namespace sankhya {

LearnedTreeDomain learned_tree_domain() {
  return LearnedTreeDomain{0, 0};
}

LearnedTreeChoice learned_engine_tree(const EngineFeatures& f) {
  (void)f;
  LearnedTreeChoice c;
  c.algorithm = "dual-simplex";
  return c;
}

}  // namespace sankhya
