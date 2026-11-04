#include "linear_map.h"

#include "../split_partitioning.h"
#include "text_column_utils.h"  // ReadLabelColumnAsInt

#include <cyxwiz/linear_algebra.h>

#include <cmath>

namespace cyxwiz {

namespace {

// 1 x n of ones times x: the column sums.
bool ColumnSums(const LinearMatrix& x, std::vector<double>& sums, std::string& error) {
    const MatrixResult ones = LinearAlgebra::Ones(1, static_cast<int>(x.size()));
    const MatrixResult s = LinearAlgebra::Multiply(ones.matrix, x);
    if (!s.success) return error = s.error_message, false;
    sums = s.matrix[0];
    return true;
}

}  // namespace

bool Centred(const LinearMatrix& x, const std::vector<double>& mean, LinearMatrix& out, std::string& error) {
    const MatrixResult column = LinearAlgebra::Ones(static_cast<int>(x.size()), 1);
    const MatrixResult spread = LinearAlgebra::Multiply(column.matrix, LinearMatrix{mean});
    const MatrixResult centred = LinearAlgebra::Subtract(x, spread.matrix);
    if (!centred.success) return error = centred.error_message, false;
    out = centred.matrix;
    return true;
}

bool ColumnMoments(const LinearMatrix& x, std::vector<double>& mean, std::vector<double>& sd, std::string& error) {
    if (x.empty()) return error = "no rows", false;
    if (!ColumnSums(x, mean, error)) return false;
    const double n = static_cast<double>(x.size());
    for (double& m : mean) m /= n;
    LinearMatrix c;
    if (!Centred(x, mean, c, error)) return false;
    // The diagonal of c^T c: the squared deviations of each column.
    const MatrixResult t = LinearAlgebra::Transpose(c);
    const MatrixResult cross = LinearAlgebra::Multiply(t.matrix, c);
    if (!cross.success) return error = cross.error_message, false;
    sd.assign(mean.size(), 0.0);
    for (size_t j = 0; j < sd.size(); ++j) sd[j] = std::sqrt(std::max(0.0, cross.matrix[j][j]) / n);
    return true;
}

bool ApplyLinearMap(const LinearMap& map, const LinearMatrix& rows, LinearMatrix& out, std::string& error) {
    if (rows.empty()) return out.clear(), true;
    LinearMatrix x;
    if (!Centred(rows, map.shift, x, error)) return false;
    // Dividing each column by its divisor is a product with diag(1 / divisor), folded into W.
    LinearMatrix w = map.w;
    for (size_t j = 0; j < w.size(); ++j)
        for (double& v : w[j]) v /= map.divisor[j];
    const MatrixResult p = LinearAlgebra::Multiply(x, w);
    if (!p.success) return error = p.error_message, false;
    out = p.matrix;
    return true;
}

LinearMatrix TrainSubset(const std::shared_ptr<arrow::Table>& table, const FeatureRows& fit) {
    const std::vector<bool> train = TrainRowMask(table);
    if (train.empty()) return fit.data;
    LinearMatrix out;
    for (size_t i = 0; i < fit.rows.size(); ++i)
        if (train[static_cast<size_t>(fit.rows[i])]) out.push_back(fit.data[i]);
    return out;
}

arrow::Result<std::shared_ptr<arrow::Table>> LinearMapTable(const std::shared_ptr<arrow::Table>& input, const LinearMap& map,
                                                            const FeatureRows& fit, const LinearMatrix& projected,
                                                            bool keep_columns, const std::string& label_col, const std::string& op_name) {
    if (keep_columns) return AppendFittedColumns(input, map.outputs, fit.rows, projected);
    const int64_t n = static_cast<int64_t>(fit.rows.size());
    const size_t k = map.outputs.size();
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    for (size_t c = 0; c < k; ++c) {
        arrow::FloatBuilder b;
        ARROW_RETURN_NOT_OK(b.Reserve(n));
        for (int64_t r = 0; r < n; ++r) b.UnsafeAppend(static_cast<float>(projected[static_cast<size_t>(r)][c]));
        std::shared_ptr<arrow::Array> a;
        ARROW_RETURN_NOT_OK(b.Finish(&a));
        fields.push_back(arrow::field(map.outputs[c], arrow::float32()));
        arrays.push_back(a);
    }
    if (!label_col.empty()) {
        const auto column = input->GetColumnByName(label_col);
        if (!column) return arrow::Status::KeyError(op_name + ": label column '" + label_col + "' not found");
        std::vector<int> labels;
        std::vector<std::string> class_names;
        std::string bad;
        if (!ReadLabelColumnAsInt(column, labels, class_names, bad))
            return arrow::Status::TypeError(op_name + ": label column '" + label_col + "' has unsupported type '" + bad + "'");
        arrow::Int32Builder b;
        ARROW_RETURN_NOT_OK(b.Reserve(n));
        for (const int64_t row : fit.rows) b.UnsafeAppend(labels[static_cast<size_t>(row)]);
        std::shared_ptr<arrow::Array> a;
        ARROW_RETURN_NOT_OK(b.Finish(&a));
        fields.push_back(arrow::field("y", arrow::int32()));
        arrays.push_back(a);
    }
    // The split the step learned from, for the rows kept.
    const std::vector<int8_t> parts = PartitionValues(input);
    if (!parts.empty()) {
        arrow::Int8Builder b;
        ARROW_RETURN_NOT_OK(b.Reserve(n));
        for (const int64_t row : fit.rows) b.UnsafeAppend(parts[static_cast<size_t>(row)]);
        std::shared_ptr<arrow::Array> a;
        ARROW_RETURN_NOT_OK(b.Finish(&a));
        fields.push_back(arrow::field(kSplitPartitionColumn, arrow::int8()));
        arrays.push_back(a);
    }
    return arrow::Table::Make(arrow::schema(fields), arrays, n);
}

FittedPreprocessingState LinearMapState(const LinearMap& map, const std::string& op_name, int64_t fit_rows,
                                        const std::shared_ptr<arrow::Schema>& schema,
                                        const std::map<std::string, std::string>& configuration) {
    FittedPreprocessingState state;
    state.operator_name = op_name;
    state.fit_rows = fit_rows;
    state.input_schema_fingerprint = FingerprintPreprocessingSchema(schema);
    state.configuration = configuration;
    std::string outputs;
    for (const auto& name : map.outputs) outputs += (outputs.empty() ? "" : ",") + name;
    state.configuration["outputs"] = outputs;
    for (size_t j = 0; j < map.inputs.size(); ++j) {
        PreprocessingFeatureState f;
        f.name = map.inputs[j];
        f.data_type = "double";
        f.numeric_values["shift"] = map.shift[j];
        f.numeric_values["divisor"] = map.divisor[j];
        for (size_t c = 0; c < map.outputs.size(); ++c) f.numeric_values["w" + std::to_string(c)] = map.w[j][c];
        state.features.push_back(std::move(f));
    }
    return state;
}

bool LinearMapFromState(const FittedPreprocessingState& state, LinearMap& map, std::string& error) {
    map = LinearMap{};
    const auto outputs = state.configuration.find("outputs");
    if (outputs == state.configuration.end() || outputs->second.empty()) return error = "the saved values name no output columns", false;
    size_t start = 0;
    const std::string& list = outputs->second;
    while (start <= list.size()) {
        const size_t comma = list.find(',', start);
        map.outputs.push_back(list.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    for (const auto& f : state.features) {
        const auto value = [&](const std::string& key, double& v) {
            const auto it = f.numeric_values.find(key);
            if (it == f.numeric_values.end()) return false;
            v = it->second;
            return true;
        };
        double shift = 0, divisor = 1;
        std::vector<double> row(map.outputs.size());
        bool ok = value("shift", shift) && value("divisor", divisor) && divisor != 0.0;
        for (size_t c = 0; ok && c < row.size(); ++c) ok = value("w" + std::to_string(c), row[c]);
        if (!ok) return error = "the saved values of '" + f.name + "' are incomplete", false;
        map.inputs.push_back(f.name);
        map.shift.push_back(shift);
        map.divisor.push_back(divisor);
        map.w.push_back(row);
    }
    if (map.inputs.empty()) return error = "the saved values hold no column", false;
    return true;
}

}  // namespace cyxwiz
