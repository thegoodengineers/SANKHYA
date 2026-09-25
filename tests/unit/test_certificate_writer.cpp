// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the VIPR certificate writer (#518).
//
// The certificate's arithmetic is checked by tools/verify_certificate.py, which never links
// this code (tools/test_verify_certificate.py runs it end to end against the built CLI).
// These tests hold the C++ side's own promises: every number is written exactly, a search
// that added rows the model does not have writes nothing and says why, and the claim written
// is the one the search proved.

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#include "mip/certificate_writer.hpp"
#include "support/temp_file.hpp"

namespace sankhya::mip {
namespace {

TEST(CertificateWriter, NumbersAreWrittenExactly) {
  EXPECT_EQ(exact_decimal(0.0), "0");
  EXPECT_EQ(exact_decimal(3.0), "3");
  EXPECT_EQ(exact_decimal(-2.5), "-2.5");
  EXPECT_EQ(exact_decimal(0.1), "0.1000000000000000055511151231257827021181583404541015625");
  EXPECT_EQ(exact_decimal(std::ldexp(1.0, 70)), "1180591620717411303424");
  EXPECT_EQ(exact_decimal(std::ldexp(1.0, -20)), "0.00000095367431640625");
  // Every digit of the smallest subnormal: 1074 of them after the point, and it reads back
  // as itself.
  const double tiny = std::nextafter(0.0, 1.0);
  const std::string text = exact_decimal(tiny);
  EXPECT_EQ(text.size(), std::string("0.").size() + 1074);
  EXPECT_EQ(std::strtod(text.c_str(), nullptr), tiny);
}

/// min -5 x1 - 4 x2 - 3 x3  s.t. three knapsack rows, x integer in [0, 10]: optimum -13.
Model knapsack() {
  Model model;
  model.resize_columns(3);
  model.col_cost = {-5.0, -4.0, -3.0};
  for (auto& t : model.col_type) t = VarType::kInteger;
  model.col_upper = {10.0, 10.0, 10.0};
  model.resize_rows(3);
  model.row_upper = {5.0, 11.0, 8.0};
  model.matrix.reset(3, 3);
  const double a[3][3] = {{2, 3, 1}, {4, 1, 2}, {3, 4, 2}};
  for (Index i = 0; i < 3; ++i) {
    for (Index j = 0; j < 3; ++j) {
      model.matrix.add_entry(i, j, a[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
    }
  }
  model.matrix.finalize();
  return model;
}

std::string read_all(const std::string& path) {
  std::ifstream in(path);
  std::stringstream s;
  s << in.rdbuf();
  return s.str();
}

TEST(CertificateWriter, TheSearchWritesTheClaimItProved) {
  const testing::TempFile file("", ".vipr");
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("enable_root_cuts", false);
  options.set_string("write_certificate", file.path());
  const Solution solved = solve(knapsack(), options);
  ASSERT_EQ(solved.status, SolveStatus::kOptimal);
  EXPECT_DOUBLE_EQ(solved.objective, -13.0);
  const std::string text = read_all(file.path());
  ASSERT_FALSE(text.empty());
  EXPECT_NE(text.find("VER 1.0"), std::string::npos);
  EXPECT_NE(text.find("OBJ min"), std::string::npos);
  // The objective is integral with step 1, so the proof ends on the optimum itself.
  EXPECT_NE(text.find("RTP range -13 -13"), std::string::npos) << text.substr(0, 600);
  EXPECT_NE(text.find("DER "), std::string::npos);
}

TEST(CertificateWriter, SymmetryRowsAreLeftOutOfACertifiedSearch) {
  // Two identical columns: formulation symmetry (#413) appends an ordering row, a row the
  // model does not have and the certificate cannot derive. With symmetry on by default, a
  // search asked for a proof runs without those rows and is certified, rather than refused
  // on every symmetric model.
  Model model = knapsack();
  model.col_cost[1] = model.col_cost[0];
  model.matrix.reset(3, 3);
  const double a[3][3] = {{2, 2, 1}, {4, 4, 2}, {3, 3, 2}};
  for (Index i = 0; i < 3; ++i) {
    for (Index j = 0; j < 3; ++j) {
      model.matrix.add_entry(i, j, a[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
    }
  }
  model.matrix.finalize();
  const testing::TempFile file("", ".vipr");
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("enable_root_cuts", false);
  options.set_bool("mip_symmetry", true);
  // Without a certificate the model is symmetric and the ordering rows are added.
  const Solution plain = solve(model, options);
  ASSERT_EQ(plain.status, SolveStatus::kOptimal);
  ASSERT_GT(plain.symmetry_generators, 0);
  // With one, the search runs without them and the answer is certified.
  options.set_string("write_certificate", file.path());
  const Solution solved = solve(model, options);
  ASSERT_EQ(solved.status, SolveStatus::kOptimal);
  EXPECT_EQ(solved.symmetry_generators, 0);
  EXPECT_DOUBLE_EQ(solved.objective, plain.objective);
  EXPECT_NE(read_all(file.path()).find("RTP range"), std::string::npos);
}

}  // namespace
}  // namespace sankhya::mip
