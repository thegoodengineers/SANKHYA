// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the .nl reader (NLP stage 1), on files written by hand from D. M. Gay, "Writing
// .nl Files", SAND2005-7907P (2005). Every expected value is derived in a comment from the
// file text, not from the reader.

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "nlp/nl_reader.hpp"
#include "nlp/nlp_problem.hpp"
#include "sankhya/io.hpp"

namespace sankhya::nlp {
namespace {

namespace fs = std::filesystem;

std::string write_file(const std::string& name, const std::string& text) {
  const fs::path path = fs::temp_directory_path() / name;
  std::ofstream(path) << text;
  return path.string();
}

/// The standard ten-line header with the given lines 2..10 (sec. 2, Figure 2).
std::string header(const std::string& body) {
  return "g3 1 1 0\t# problem test\n" + body;
}

// HS071 (Hock and Schittkowski 1981, problem 71): min x0 x3 (x0+x1+x2) + x2 subject to
// x0 x1 x2 x3 >= 25, sum x_i^2 = 40, 1 <= x <= 5, start (1, 5, 5, 1). All four variables
// are nonlinear in both the objective and the constraints (nlvc = nlvo = nlvb = 4).
const char* kHs071 =
    " 4 2 1 0 1\t# vars, constraints, objectives, ranges, eqns\n"
    " 2 1\t# nonlinear constraints, objectives\n"
    " 0 0\t# network constraints: nonlinear, linear\n"
    " 4 4 4\t# nonlinear vars in constraints, objectives, both\n"
    " 0 0 0 1\t# linear network variables; functions; arith, flags\n"
    " 0 0 0 0 0\t# discrete variables: binary, integer, nonlinear (b,c,o)\n"
    " 8 4\t# nonzeros in Jacobian, gradients\n"
    " 0 0\t# max name lengths: constraints, variables\n"
    " 0 0 0 0 0\t# common exprs: b,c,o,c1,o1\n"
    "C0\no2\no2\no2\nv0\nv1\nv2\nv3\n"
    "C1\no54\n4\no5\nv0\nn2\no5\nv1\nn2\no5\nv2\nn2\no5\nv3\nn2\n"
    "O0 0\no2\no2\nv0\nv3\no54\n3\nv0\nv1\nv2\n"
    "x4\n0 1\n1 5\n2 5\n3 1\n"
    "r\n2 25\n4 40\n"
    "b\n0 1 5\n0 1 5\n0 1 5\n0 1 5\n"
    "k3\n2\n4\n6\n"
    "J0 4\n0 0\n1 0\n2 0\n3 0\n"
    "J1 4\n0 0\n1 0\n2 0\n3 0\n"
    "G0 4\n0 0\n1 0\n2 1\n3 0\n";

TEST(NlReader, Hs071IsReadAsWritten) {
  const std::string path = write_file("sankhya_hs071.nl", header(kHs071));
  std::unique_ptr<NonlinearModel> m;
  const io::ReadResult r = read_nl(path, &m);
  ASSERT_TRUE(r.ok) << r.error;
  EXPECT_EQ(m->base.num_cols(), 4);
  EXPECT_EQ(m->base.num_rows(), 0)
      << "every constraint is a NonlinearConstraint, in file order";
  ASSERT_EQ(m->constraints.size(), 2u);
  EXPECT_EQ(m->constraints[0].lower, 25.0);  // r type 2: lower bound only
  EXPECT_TRUE(std::isinf(m->constraints[0].upper));
  EXPECT_EQ(m->constraints[1].lower, 40.0);  // r type 4: equality
  EXPECT_EQ(m->constraints[1].upper, 40.0);
  for (Index j = 0; j < 4; ++j) {
    EXPECT_EQ(m->base.col_lower[static_cast<std::size_t>(j)], 1.0);
    EXPECT_EQ(m->base.col_upper[static_cast<std::size_t>(j)], 5.0);
    EXPECT_EQ(m->base.col_type[static_cast<std::size_t>(j)], VarType::kContinuous);
  }
  EXPECT_EQ(m->start, (std::vector<double>{1, 5, 5, 1}));
  EXPECT_EQ(m->base.col_cost, (std::vector<double>{0, 0, 1, 0})) << "the G segment";
  EXPECT_EQ(m->classify(), ProblemClass::kNlp);

  // At the start: f = 1*1*(1+5+5) + 5 = 16; g = (1*5*5*1, 1+25+25+1) = (25, 52).
  NlpProblem p;
  ASSERT_EQ(NlpProblem::build(*m, &p), "");
  Evaluation err;
  double f = 0.0;
  ASSERT_TRUE(p.objective(m->start, &f, &err)) << err.message;
  EXPECT_EQ(f, 16.0);
  std::vector<double> g;
  ASSERT_TRUE(p.constraints(m->start, &g, &err));
  EXPECT_EQ(g, (std::vector<double>{25.0, 52.0}));
}

TEST(NlReader, IntegerColumnsFollowTheOrderingTables) {
  // 6 variables: nlvc = 3, nlvo = 2, nlvb = 2, nlvbi = 1, nlvci = 1 (Hooking, Table 4):
  //   0 continuous both, 1 integer both, 2 integer just-in-constraints,
  // then linear: 3 continuous, 4 binary (nbv = 1), 5 integer (niv = 1) (Table 3).
  const std::string text = header(
      " 6 1 1 0 0\n 1 1\n 0 0\n 3 2 2\n 0 0 0 1\n 1 1 1 1 0\n 6 3\n 0 0\n 0 0 0 0 0\n"
      "C0\no54\n3\no5\nv0\nn2\no5\nv1\nn2\no5\nv2\nn2\n"
      "O0 1\no2\nv0\nv1\n"
      "r\n1 10\n"
      "b\n3\n0 0 3\n0 0 3\n2 0\n0 0 1\n0 -2 2\n"
      "J0 6\n0 0\n1 0\n2 0\n3 1\n4 1\n5 1\n"
      "G0 3\n0 0\n1 0\n3 -1\n");
  std::unique_ptr<NonlinearModel> m;
  const io::ReadResult r = read_nl(write_file("sankhya_ints.nl", text), &m);
  ASSERT_TRUE(r.ok) << r.error;
  const std::vector<VarType> want{VarType::kContinuous, VarType::kInteger, VarType::kInteger,
                                  VarType::kContinuous, VarType::kInteger, VarType::kInteger};
  EXPECT_EQ(m->base.col_type, want);
  EXPECT_EQ(m->base.sense, ObjSense::kMaximize) << "O0 1";
  EXPECT_TRUE(std::isinf(m->base.col_lower[0]) && m->base.col_lower[0] < 0) << "b type 3: free";
  EXPECT_EQ(m->base.col_lower[3], 0.0);  // b type 2
  EXPECT_TRUE(std::isinf(m->base.col_upper[3]));
  EXPECT_EQ(m->classify(), ProblemClass::kMinlp);
  // Constraint 0 = x0^2 + x1^2 + x2^2 + x3 + x4 + x5 <= 10: at x = 1..6, 1+4+9+4+5+6 = 29.
  NlpProblem p;
  ASSERT_EQ(NlpProblem::build(*m, &p), "");
  std::vector<double> g;
  Evaluation err;
  ASSERT_TRUE(p.constraints({1, 2, 3, 4, 5, 6}, &g, &err));
  EXPECT_EQ(g[0], 29.0);
}

TEST(NlReader, DefinedVariablesLog10PowAndNames) {
  // V2 = 3 x0 + x1^2 (a defined variable, sec. 4 V segment), used twice:
  // f = log10(V2) + 2^x1 ; c0: V2 * x0 >= 1. Names come from a .row / .col beside the file.
  const std::string text = header(
      " 2 1 1 0 0\n 1 1\n 0 0\n 2 2 2\n 0 0 0 1\n 0 0 0 0 0\n 2 2\n 0 0\n 1 0 0 0 0\n"
      "V2 1 0\n0 3\no5\nv1\nn2\n"
      "C0\no2\nv2\nv0\n"
      "O0 0\no0\no42\nv2\no5\nn2\nv1\n"
      "r\n2 1\n"
      "b\n0 0.1 10\n0 -3 3\n"
      "J0 2\n0 0\n1 0\n"
      "G0 2\n0 0\n1 0\n");
  const std::string path = write_file("sankhya_defined.nl", text);
  std::ofstream(fs::temp_directory_path() / "sankhya_defined.col") << "alpha\nbeta\n";
  std::ofstream(fs::temp_directory_path() / "sankhya_defined.row") << "cap\nobj\n";
  std::unique_ptr<NonlinearModel> m;
  std::vector<std::string> notes;
  const io::ReadResult r = read_nl(path, &m, &notes);
  ASSERT_TRUE(r.ok) << r.error;
  EXPECT_EQ(m->base.col_names, (std::vector<std::string>{"alpha", "beta"}));
  EXPECT_EQ(m->constraints[0].name, "cap");
  NlpProblem p;
  ASSERT_EQ(NlpProblem::build(*m, &p), "");
  // At x = (2, 1): V2 = 6 + 1 = 7; f = log10(7) + 2^1; g0 = 7 * 2 = 14.
  Evaluation err;
  double f = 0.0;
  ASSERT_TRUE(p.objective({2.0, 1.0}, &f, &err)) << err.message;
  EXPECT_NEAR(f, std::log10(7.0) + 2.0, 1e-15);
  std::vector<double> g;
  ASSERT_TRUE(p.constraints({2.0, 1.0}, &g, &err));
  EXPECT_NEAR(g[0], 14.0, 1e-15);
  EXPECT_TRUE(notes.empty()) << "2^x1 has a constant positive base: exact, nothing to note";
}

TEST(NlReader, RefusesWhatItCannotRepresentByName) {
  const std::string lines =
      " 1 1 1 0 0\n 1 1\n 0 0\n 1 1 1\n 0 0 0 1\n 0 0 0 0 0\n 1 1\n 0 0\n 0 0 0 0 0\n";
  const auto read = [&](const std::string& name, const std::string& body) {
    std::unique_ptr<NonlinearModel> m;
    return read_nl(write_file(name, header(lines) + body), &m);
  };
  const std::string tail = "O0 0\nv0\nr\n3\nb\n3\nJ0 1\n0 0\n";
  io::ReadResult r = read("sankhya_abs.nl", "C0\no15\nv0\n" + tail);
  EXPECT_FALSE(r.ok);
  EXPECT_TRUE(r.refused);
  EXPECT_NE(r.error.find("abs"), std::string::npos) << r.error;
  r = read("sankhya_if.nl", "C0\no35\no28\nv0\nn0\nv0\nn1\n" + tail);
  EXPECT_NE(r.error.find("if"), std::string::npos) << r.error;
  r = read("sankhya_compl.nl", "C0\nv0\nO0 0\nv0\nr\n5 1 1\nb\n3\nJ0 1\n0 0\n");
  EXPECT_NE(r.error.find("complementarity"), std::string::npos) << r.error;
  r = read("sankhya_unknown_op.nl", "C0\no99\nv0\n" + tail);
  EXPECT_FALSE(r.ok);
  EXPECT_NE(r.error.find("o99"), std::string::npos) << r.error;

  std::unique_ptr<NonlinearModel> m;
  r = read_nl(write_file("sankhya_binary.nl", "b3 1 1 0\n" + lines), &m);
  EXPECT_TRUE(r.refused);
  EXPECT_NE(r.error.find("binary"), std::string::npos) << r.error;
  EXPECT_TRUE(m == nullptr) << "nothing is handed back from a failed read";

  // The linear readers refuse a .nl file by name instead of misreading it as MPS.
  Model linear;
  r = io::read_model(write_file("sankhya_linear_view.nl", header(kHs071)), &linear);
  EXPECT_TRUE(r.refused);
  EXPECT_NE(r.error.find("nonlinear"), std::string::npos) << r.error;
}

TEST(NlReader, AVariableExponentIsNotedAndItsDomainIsTheLogs) {
  // f = x0 ^ x1 is read as exp(x1 log x0): 2^3 = 8 at (2, 3), and undefined at x0 = -2.
  const std::string text = header(
      " 2 0 1 0 0\n 0 1\n 0 0\n 0 2 0\n 0 0 0 1\n 0 0 0 0 0\n 0 2\n 0 0\n 0 0 0 0 0\n"
      "O0 0\no5\nv0\nv1\nb\n3\n3\nG0 2\n0 0\n1 0\n");
  std::unique_ptr<NonlinearModel> m;
  std::vector<std::string> notes;
  ASSERT_TRUE(read_nl(write_file("sankhya_varpow.nl", text), &m, &notes).ok);
  ASSERT_EQ(notes.size(), 1u);
  EXPECT_NE(notes[0].find("exp(exponent * log(base))"), std::string::npos);
  NlpProblem p;
  ASSERT_EQ(NlpProblem::build(*m, &p), "");
  double f = 0.0;
  Evaluation err;
  ASSERT_TRUE(p.objective({2.0, 3.0}, &f, &err));
  EXPECT_NEAR(f, 8.0, 1e-13);
  EXPECT_FALSE(p.objective({-2.0, 3.0}, &f, &err));
  EXPECT_EQ(err.error, EvalError::kDomain);
}

}  // namespace
}  // namespace sankhya::nlp
