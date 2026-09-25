// SPDX-License-Identifier: Apache-2.0
// SANKHYA - reading a nonlinear model from an AMPL .nl file, text form (NLP stage 1).
//
// WHY .nl AND NOT A FORMAT OF OUR OWN. A nonlinear model needs a file format for the same
// reason a linear one does: so a judge can hand the solver a model it has never seen, and so
// tools/verify_solution.py can re-read that model independently. A private format would
// satisfy neither - nobody else writes it. The .nl format is what AMPL writes, what Pyomo
// writes (its `nl` writer), and what MINLPLib and other public collections publish their
// instances in, so a reader for it is the one that lets the models people already have
// reach the engine. Its specification is public: D. M. Gay, "Writing .nl Files", Sandia
// report SAND2005-7907P (2005), for the header, the segments and the operator tables, and
// D. M. Gay, "Hooking Your Solver to AMPL" (1993, rev. 1997), Tables 3 and 4, for the order
// of the variables, which is what says which ones are integer. This reader is written from
// those two documents; no solver's .nl reader, and not the AMPL/solver interface library
// either, was read (docs/PROVENANCE.md, judgement calls).
//
// WHAT IS READ. The text ('g') form only; the binary ('b') form is refused by name. Every
// segment of the specification is understood, and these operators are accepted because the
// expression layer represents them exactly (src/nlp/expression.hpp): plus, minus, mult, div,
// pow, neg, sumlist, sqrt, exp, log, log10, sin, cos. Everything else - abs, min, max,
// if-then-else, the logical and relational operators, the other trigonometric and
// hyperbolic functions, piecewise-linear terms, imported functions, logical and
// complementarity constraints - is REFUSED with the operator's name, never approximated: a
// non-smooth operator handed to a smooth method gives an answer the method cannot vouch for.
//
// pow with a variable exponent. `a ^ c` with c constant is ExpressionGraph::power; `c ^ b`
// with c > 0 constant is exp(b log c), which is the same function everywhere; `a ^ b` with
// both variable is exp(b log a), the same function where a > 0 and undefined elsewhere - so
// a model that needs (-2) ^ 3 through a VARIABLE exponent is reported as a domain error at
// that point rather than evaluated. That is the one place this reader narrows a function's
// domain, and it is stated here and in the result's notes.
//
// THE MODEL IT BUILDS. Every algebraic constraint becomes a NonlinearConstraint, in the
// file's own order - the linear ones as their linear expression - so constraint i of the
// file is constraint i of the model and multiplier i of any solution. base holds the
// columns (bounds, integrality, names from a .col file beside the model when there is one)
// and the linear part of the FIRST objective as its costs; the objective's nonlinear part
// is `objective`. A file with several objectives is read for the first, and the notes say
// so. The `x` segment is the model's `start`.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "nlp/nonlinear_model.hpp"
#include "sankhya/io.hpp"

namespace sankhya::nlp {

/// Read `path` (optionally gzip-compressed). On success `*out` holds the model and `notes`,
/// when given, what the reader decided that a caller should know (a second objective
/// ignored, a variable-exponent power). On failure `*out` is untouched.
io::ReadResult read_nl(const std::string& path, std::unique_ptr<NonlinearModel>* out,
                       std::vector<std::string>* notes = nullptr);

/// True when the path names a .nl file (".nl" or ".nl.gz", any case).
[[nodiscard]] bool looks_like_nl(const std::string& path);

}  // namespace sankhya::nlp
