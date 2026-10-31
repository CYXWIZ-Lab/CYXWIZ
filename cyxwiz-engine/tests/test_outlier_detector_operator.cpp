// The Outlier Detector step (TOFIX129 D7a G2): learns each column's bounds
// from the Train rows (the pipeline run's "__partition__" column), tests
// every row, and flags, clips or drops. The bounds themselves are checked
// against PyTorch in test_computation_truth_data_transform; here the
// table contract: which rows, which columns, which values.

#include "../src/core/node_executors/preprocessing_operators.h"
#include "../src/core/split_partitioning.h"

#include <arrow/api.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

using cyxwiz::OutlierDetectorOperator;

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

// value: 8 Train rows 1..8 plus Validation / Test rows holding the extremes,
// so bounds learned from every row would differ from the Train rows' ones.
//   rows:      0  1  2  3  4  5  6  7   8    9   10
//   value:     1  2  3  4  5  6  7  8  100  -50   4.5   (row 10 is null in `other`)
//   partition: 0  0  0  0  0  0  0  0   1    2    1
// Train IQR: q1 = 2.75, q3 = 6.25, IQR 3.5, fences at 1.5: -2.5 .. 11.5.
std::shared_ptr<arrow::Table> Table(bool partitioned) {
    arrow::DoubleBuilder value, other;
    arrow::StringBuilder name;
    arrow::Int8Builder partition;
    const std::vector<double> v = {1, 2, 3, 4, 5, 6, 7, 8, 100, -50, 4.5};
    const std::vector<int8_t> p = {0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1};
    for (size_t i = 0; i < v.size(); ++i) {
        Check(value.Append(v[i]).ok(), "append");
        if (i == 10) Check(other.AppendNull().ok(), "append null");
        else Check(other.Append(10.0 * static_cast<double>(i)).ok(), "append other");
        Check(name.Append("r" + std::to_string(i)).ok(), "append name");
        Check(partition.Append(p[i]).ok(), "append partition");
    }
    std::shared_ptr<arrow::Array> va, oa, na, pa;
    Check(value.Finish(&va).ok() && other.Finish(&oa).ok() && name.Finish(&na).ok() && partition.Finish(&pa).ok(), "finish");
    std::vector<std::shared_ptr<arrow::Field>> fields = {arrow::field("value", arrow::float64()), arrow::field("other", arrow::float64()),
                                                         arrow::field("name", arrow::utf8())};
    std::vector<std::shared_ptr<arrow::Array>> arrays = {va, oa, na};
    if (partitioned) {
        fields.push_back(arrow::field(cyxwiz::kSplitPartitionColumn, arrow::int8()));
        arrays.push_back(pa);
    }
    return arrow::Table::Make(arrow::schema(fields), arrays);
}

std::shared_ptr<arrow::Table> Run(const std::map<std::string, std::string>& params, bool partitioned, const std::string& what) {
    OutlierDetectorOperator op;
    std::string error;
    Check(op.Configure(params, error), what + ": configure: " + error);
    auto result = op.Apply(Table(partitioned));
    Check(result.ok(), what + ": apply: " + result.status().ToString());
    return *result;
}

std::vector<double> Column(const std::shared_ptr<arrow::Table>& t, const std::string& name) {
    std::vector<double> out;
    const auto column = t->GetColumnByName(name);
    Check(column != nullptr, "column " + name);
    for (const auto& chunk : column->chunks()) {
        if (chunk->type_id() == arrow::Type::INT32) {
            const auto a = std::static_pointer_cast<arrow::Int32Array>(chunk);
            for (int64_t i = 0; i < a->length(); ++i) out.push_back(a->Value(i));
        } else if (chunk->type_id() == arrow::Type::INT8) {
            const auto a = std::static_pointer_cast<arrow::Int8Array>(chunk);
            for (int64_t i = 0; i < a->length(); ++i) out.push_back(a->Value(i));
        } else {
            const auto a = std::static_pointer_cast<arrow::DoubleArray>(chunk);
            for (int64_t i = 0; i < a->length(); ++i) out.push_back(a->IsNull(i) ? NAN : a->Value(i));
        }
    }
    return out;
}

}  // namespace

int main() {
    // Flag: an is_outlier column, rows unchanged, bounds from the Train rows
    // (-2.5 .. 11.5): rows 8 and 9 are outside; row 10 (4.5) is not.
    {
        const auto t = Run({{"columns", "value"}, {"method", "iqr"}, {"threshold", "1.5"}, {"action", "flag"}}, true, "flag");
        Check(t->num_rows() == 11 && t->num_columns() == 5 && t->schema()->GetFieldIndex("is_outlier") == 4, "flag: a column added");
        const auto flags = Column(t, "is_outlier");
        Check(flags == std::vector<double>({0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0}), "flag: rows 8 and 9 outside the Train bounds");
        Check(Column(t, "value")[8] == 100.0, "flag: values untouched");
    }
    // Without a partition column every row teaches the bounds: q1 = 2.5, q3 = 6.5,
    // fences -3.5 .. 12.5, so only 100 and -50 are outside (the same rows here).
    {
        const auto t = Run({{"columns", "value"}, {"method", "iqr"}, {"threshold", "1.5"}}, false, "flag unpartitioned");
        Check(t->num_columns() == 4 && Column(t, "is_outlier") == std::vector<double>({0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0}),
              "unpartitioned: learned from every row, flag is the default action");
    }
    // Clip: the values pulled into the Train bounds, no column added, the
    // column float64, the null stays null.
    {
        const auto t = Run({{"columns", "value,other"}, {"method", "iqr"}, {"threshold", "1.5"}, {"action", "clip"}}, true, "clip");
        Check(t->num_rows() == 11 && t->num_columns() == 4, "clip: no column added, no row dropped");
        const auto v = Column(t, "value");
        Check(v[8] == 11.5 && v[9] == -2.5 && v[0] == 1.0 && v[10] == 4.5, "clip: 100 -> 11.5, -50 -> -2.5, the rest as they were");
        // other: Train rows 0..70 (q1 17.5, q3 52.5, IQR 35): -35 .. 105; 80 and 90 stay, the null stays null.
        const auto o = Column(t, "other");
        Check(o[8] == 80.0 && o[9] == 90.0 && std::isnan(o[10]), "clip: the other column keeps its values and its null");
        Check(t->schema()->GetFieldByName("value")->type()->id() == arrow::Type::DOUBLE, "clip: float64 column");
    }
    // Remove: the outlier rows gone, the other columns and the partition kept with the rows.
    {
        const auto t = Run({{"columns", "value"}, {"method", "iqr"}, {"threshold", "1.5"}, {"action", "remove"}}, true, "remove");
        Check(t->num_rows() == 9 && t->num_columns() == 4, "remove: 2 rows dropped, no column added");
        const auto v = Column(t, "value");
        Check(v.size() == 9 && v[8] == 4.5, "remove: the rows outside are gone, row 10 stays");
        const auto p = Column(t, cyxwiz::kSplitPartitionColumn);
        Check(p.size() == 9 && p[8] == 1, "remove: the partition column stays with the rows");
        const auto names = std::static_pointer_cast<arrow::StringArray>(t->GetColumnByName("name")->chunk(0));
        Check(names->GetString(8) == "r10", "remove: the text column follows the rows");
    }
    // Modified Z and Z-score are accepted; a rule with no spread flags nothing.
    {
        const auto t = Run({{"columns", "value"}, {"method", "modified_z"}, {"threshold", "3.5"}}, true, "modified z");
        const auto flags = Column(t, "is_outlier");
        // Train median 4.5, MAD 2: bounds 4.5 -+ 3.5 * 2 / 0.6745 = -5.88 .. 14.88.
        Check(flags == std::vector<double>({0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0}), "modified z: the same two rows");
        const auto z = Run({{"columns", "value"}, {"method", "zscore"}, {"threshold", "1.0"}}, true, "zscore");
        // Train mean 4.5, population std 2.291: 2.21 .. 6.79 -> rows 0, 1, 6, 7, 8, 9 outside.
        Check(Column(z, "is_outlier") == std::vector<double>({1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0}), "zscore: six rows outside one std");
    }
    // Refusals: an unknown rule or action, a threshold at or below 0.
    {
        OutlierDetectorOperator op;
        std::string error;
        Check(!op.Configure({{"method", "lof"}}, error) && error.find("modified_z") != std::string::npos, "unknown method refused: " + error);
        Check(!op.Configure({{"action", "winsor"}}, error) && error.find("remove") != std::string::npos, "unknown action refused: " + error);
        Check(!op.Configure({{"threshold", "0"}}, error) && error.find("> 0") != std::string::npos, "zero threshold refused: " + error);
    }
    std::cout << "outlier detector operator: flag / clip / remove, bounds from the Train rows, partition kept, refusals. OK\n";
    return 0;
}
