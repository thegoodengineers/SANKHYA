// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the engine-selection features (#477) on models small enough to count by hand.

#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include "core/engine_features.hpp"
#include "sankhya/model.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya {
namespace {

// 3 rows x 4 columns:
//   row 0: x0 + x1 + x3        = 1   (equality)
//   row 1:      x1 + x3       <= 2
//   row 2: x2 + x3            >= 0
// column counts 1, 2, 1, 3; bounds: x0 in [0, 1] boxed, x1 free, x2 fixed at 2, x3 >= 0,
// x3 integer.
Model hand_model() {
  Model m;
  m.resize_columns(4);
  m.resize_rows(3);
  m.matrix.reset(3, 4);
  m.matrix.add_entry(0, 0, 1.0);
  m.matrix.add_entry(0, 1, 1.0);
  m.matrix.add_entry(1, 1, 1.0);
  m.matrix.add_entry(2, 2, 1.0);
  m.matrix.add_entry(0, 3, 1.0);
  m.matrix.add_entry(1, 3, 1.0);
  m.matrix.add_entry(2, 3, 1.0);
  m.matrix.finalize();
  m.col_lower = {0.0, -kInfinity, 2.0, 0.0};
  m.col_upper = {1.0, kInfinity, 2.0, kInfinity};
  m.col_type[3] = VarType::kInteger;
  m.row_lower = {1.0, -kInfinity, 0.0};
  m.row_upper = {1.0, 2.0, kInfinity};
  return m;
}

TEST(EngineFeatures, HandCountedModel) {
  const EngineFeatures f = compute_engine_features(hand_model());
  EXPECT_EQ(f.rows, 3);
  EXPECT_EQ(f.columns, 4);
  EXPECT_EQ(f.nonzeros, 7);
  EXPECT_DOUBLE_EQ(f.density, 7.0 / 12.0);
  EXPECT_EQ(f.max_column_count, 3);
  EXPECT_EQ(f.dense_columns, 0);  // 3 entries at most, the threshold is 10 sqrt(3)
  EXPECT_DOUBLE_EQ(f.rows_per_column, 0.75);
  EXPECT_DOUBLE_EQ(f.equality_row_share, 1.0 / 3.0);
  EXPECT_DOUBLE_EQ(f.integer_share, 0.25);
  EXPECT_DOUBLE_EQ(f.boxed_column_share, 0.25);
  EXPECT_DOUBLE_EQ(f.free_column_share, 0.25);
  EXPECT_DOUBLE_EQ(f.fixed_column_share, 0.25);
  // 1 + 4 + 1 + 9 outer-product entries.
  EXPECT_DOUBLE_EQ(f.normal_equations_nnz_bound, 15.0);
  EXPECT_DOUBLE_EQ(f.normal_equations_ratio, 15.0 / 7.0);
}

TEST(EngineFeatures, OneLongColumnIsDense) {
  // 200 singleton columns plus one column touching every row: 200 entries against the
  // interior point's threshold kIpmDenseColumnFactor * sqrt(200) = 141.4.
  const Index rows = 200;
  Model m;
  m.resize_columns(rows + 1);
  m.resize_rows(rows);
  m.matrix.reset(rows, rows + 1);
  for (Index i = 0; i < rows; ++i) m.matrix.add_entry(i, i, 1.0);
  for (Index i = 0; i < rows; ++i) m.matrix.add_entry(i, rows, 1.0);
  m.matrix.finalize();
  const EngineFeatures f = compute_engine_features(m);
  EXPECT_EQ(f.max_column_count, rows);
  EXPECT_EQ(f.dense_columns, 1);
  EXPECT_DOUBLE_EQ(f.normal_equations_nnz_bound, rows + static_cast<double>(rows) * rows);
}

TEST(EngineFeatures, DenseColumnsAreTheInteriorPointsDenseColumns) {
  // 100 rows: the threshold is kIpmDenseColumnFactor * sqrt(100) = 100, so a column over
  // every row is exactly at it, and the test is strict as in find_dense_columns: not dense.
  const Index rows = 100;
  Model m;
  m.resize_columns(rows + 1);
  m.resize_rows(rows);
  m.matrix.reset(rows, rows + 1);
  for (Index i = 0; i < rows; ++i) m.matrix.add_entry(i, i, 1.0);
  for (Index i = 0; i < rows; ++i) m.matrix.add_entry(i, rows, 1.0);
  m.matrix.finalize();
  EXPECT_DOUBLE_EQ(tol::kIpmDenseColumnFactor * std::sqrt(static_cast<double>(rows)), 100.0);
  EXPECT_EQ(compute_engine_features(m).dense_columns, 0);
}

TEST(EngineFeatures, EmptyModelHasNoDivisionByZero) {
  Model m;
  m.matrix.reset(0, 0);
  m.matrix.finalize();
  const EngineFeatures f = compute_engine_features(m);
  EXPECT_EQ(f.rows, 0);
  EXPECT_EQ(f.density, 0.0);
  EXPECT_EQ(f.normal_equations_ratio, 0.0);
}

TEST(EngineFeatures, JsonCarriesEveryField) {
  const std::string json = format_engine_features_json(compute_engine_features(hand_model()));
  for (const char* key :
       {"\"rows\": 3", "\"columns\": 4", "\"nonzeros\": 7", "\"max_column_count\": 3",
        "\"dense_columns\": 0", "\"normal_equations_nnz_bound\": 15", "\"density\"",
        "\"equality_row_share\"", "\"integer_share\"", "\"boxed_column_share\"",
        "\"free_column_share\"", "\"fixed_column_share\"", "\"rows_per_column\"",
        "\"normal_equations_ratio\""}) {
    EXPECT_NE(json.find(key), std::string::npos) << key << " in " << json;
  }
  EXPECT_EQ(json.front(), '{');
  EXPECT_EQ(json.back(), '}');
}

}  // namespace
}  // namespace sankhya
