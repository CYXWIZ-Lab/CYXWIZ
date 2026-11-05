#include "linear_map.h"

#include "../split_partitioning.h"
#include "text_column_utils.h"  // ReadLabelColumnAsInt

#include <cyxwiz/linear_algebra.h>

#include <arrow/compute/api.h>

#include <algorithm>
#include <cmath>

namespace cyxwiz {

namespace {

bool IsNumberColumn(const std::shared_ptr<arrow::ChunkedArray>& col) {
    if (!col) return false;
    switch (col->type()->id()) {
        case arrow::Type::DOUBLE:
        case arrow::Type::FLOAT:
        case arrow::Type::INT64:
        case arrow::Type::INT32:
        case arrow::Type::INT16:
        case arrow::Type::INT8:
        case arrow::Type::UINT64:
        case arrow::Type::UINT32:
        case arrow::Type::UINT16:
        case arrow::Type::UINT8: return true;
        default: return false;
    }
}

}  // namespace

std::vector<std::string> ParseColumnList(const std::string& text) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t comma = text.find(',', start);
        std::string name = text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        const size_t first = name.find_first_not_of(" \t"), last = name.find_last_not_of(" \t");
        if (first != std::string::npos) out.push_back(name.substr(first, last - first + 1));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return out;
}

bool ReadBoolParameter(const std::map<std::string, std::string>& params, const char* key, bool fallback, bool& out,
                       const std::string& op_name, std::string& error) {
    const auto it = params.find(key);
    if (it == params.end() || it->second.empty()) return out = fallback, true;
    if (it->second == "true" || it->second == "false") return out = it->second == "true", true;
    error = op_name + ": '" + key + "' must be 'true' or 'false' (got '" + it->second + "')";
    return false;
}

arrow::Status ResolveNumberFeatures(const std::shared_ptr<arrow::Table>& input, const std::vector<std::string>& wanted,
                                    const std::string& label_col, const std::string& op_name, std::vector<std::string>& out) {
    out.clear();
    if (!wanted.empty()) {
        for (const auto& name : wanted) {
            const auto col = input->GetColumnByName(name);
            if (!col) return arrow::Status::KeyError(op_name + ": feature column '" + name + "' not found");
            if (!IsNumberColumn(col)) return arrow::Status::TypeError(op_name + ": feature column '" + name + "' is not numeric");
            out.push_back(name);
        }
        return arrow::Status::OK();
    }
    for (const auto& field : input->schema()->fields()) {
        const std::string& name = field->name();
        if (name == label_col || name.rfind("__", 0) == 0) continue;
        if (IsNumberColumn(input->GetColumnByName(name))) out.push_back(name);
    }
    if (out.empty())
        return arrow::Status::Invalid(op_name + ": no numeric feature columns found (auto-detect). Specify feature_cols explicitly.");
    return arrow::Status::OK();
}

LinearMatrix SecondMoments(const ColumnSummaryResult& summary, bool uncentred, const std::vector<double>& divisor, double denominator) {
    LinearMatrix m = summary.cross;
    const size_t d = m.size();
    const double n = static_cast<double>(summary.rows);
    for (size_t i = 0; i < d; ++i)
        for (size_t j = 0; j < d; ++j) {
            if (uncentred) m[i][j] += n * summary.mean[i] * summary.mean[j];  // about 0: + n mean_i mean_j
            if (!divisor.empty()) m[i][j] /= divisor[i] * divisor[j];
            m[i][j] /= denominator;
        }
    return m;
}

bool TopDirections(const LinearMatrix& m, size_t k, LinearMatrix& vt, std::vector<double>& values, std::string& error) {
    const size_t d = m.size();
    const SVDResult s = d > 256 && k * 4 <= d ? LinearAlgebra::LeadingEigenvectors(m, static_cast<int>(k)) : LinearAlgebra::SVD(m, false);
    if (!s.success) return error = s.error_message, false;
    vt.assign(s.Vt.begin(), s.Vt.begin() + static_cast<std::ptrdiff_t>(std::min(k, s.Vt.size())));
    values.assign(s.S.begin(), s.S.begin() + static_cast<std::ptrdiff_t>(std::min(k, s.S.size())));
    for (auto& row : vt) {  // the largest-magnitude weight positive
        size_t peak = 0;
        for (size_t j = 1; j < row.size(); ++j)
            if (std::fabs(row[j]) > std::fabs(row[peak])) peak = j;
        if (!row.empty() && row[peak] < 0)
            for (double& v : row) v = -v;
    }
    return true;
}

arrow::Status ReadNumberColumns(const std::shared_ptr<arrow::Table>& table, const std::vector<std::string>& names,
                                const std::string& op_name, NumberColumns& out, const std::function<bool()>& cancelled) {
    out = NumberColumns{};
    if (!table) return arrow::Status::Invalid(op_name + ": input table is null");
    const int64_t n = table->num_rows();
    out.total = n;
    out.cols = names.size();
    // Each feature as float64 (a cast keeps nulls; NaN counts as missing), and which rows have every one.
    std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;
    std::vector<uint8_t> complete(static_cast<size_t>(n), 1);
    for (const auto& name : names) {
        const auto column = table->GetColumnByName(name);
        if (!column) return arrow::Status::KeyError(op_name + ": feature column '" + name + "' not found");
        ARROW_ASSIGN_OR_RAISE(auto cast, arrow::compute::Cast(arrow::Datum(column), arrow::float64()));
        const auto chunked = cast.chunked_array();
        int64_t r = 0;
        for (const auto& chunk : chunked->chunks()) {
            const auto values = std::static_pointer_cast<arrow::DoubleArray>(chunk);
            const double* raw = values->raw_values();
            for (int64_t i = 0; i < values->length(); ++i, ++r)
                if (values->IsNull(i) || !std::isfinite(raw[i])) complete[static_cast<size_t>(r)] = 0;
        }
        if (r != n) return arrow::Status::Invalid(op_name + ": feature '" + name + "' has a different row count");
        columns.push_back(chunked);
        if (cancelled && cancelled()) return arrow::Status::Cancelled(op_name + ": cancelled while reading features");
    }
    for (int64_t r = 0; r < n; ++r)
        if (complete[static_cast<size_t>(r)]) out.rows.push_back(r);
    const size_t kept = out.rows.size();
    out.data.resize(kept * out.cols);
    const bool all = kept == static_cast<size_t>(n);
    for (size_t j = 0; j < out.cols; ++j) {
        double* dest = out.data.data() + j * kept;
        size_t k = 0;
        int64_t r = 0;
        for (const auto& chunk : columns[j]->chunks()) {
            const auto values = std::static_pointer_cast<arrow::DoubleArray>(chunk);
            const double* raw = values->raw_values();
            if (all) {
                std::copy(raw, raw + values->length(), dest + k);
                k += static_cast<size_t>(values->length());
                r += values->length();
                continue;
            }
            for (int64_t i = 0; i < values->length(); ++i, ++r)
                if (complete[static_cast<size_t>(r)]) dest[k++] = raw[i];
        }
    }
    return arrow::Status::OK();
}

NumberColumns TrainColumns(const std::shared_ptr<arrow::Table>& table, const NumberColumns& all) {
    const std::vector<bool> train = TrainRowMask(table);
    if (train.empty()) return all;
    NumberColumns out;
    out.total = all.total;
    out.cols = all.cols;
    std::vector<size_t> picked;
    for (size_t i = 0; i < all.rows.size(); ++i)
        if (train[static_cast<size_t>(all.rows[i])]) {
            picked.push_back(i);
            out.rows.push_back(all.rows[i]);
        }
    const size_t n = all.rows.size(), kept = picked.size();
    out.data.resize(kept * all.cols);
    for (size_t j = 0; j < all.cols; ++j)
        for (size_t k = 0; k < kept; ++k) out.data[j * kept + k] = all.data[j * n + picked[k]];
    return out;
}

bool ApplyLinearMap(const LinearMap& map, const NumberColumns& rows, std::vector<double>& out, std::string& error) {
    if (rows.rows.empty()) return out.clear(), true;
    return LinearAlgebra::AffineProjection(rows.View(), map.shift, map.divisor, map.w, out, error);
}

arrow::Result<std::shared_ptr<arrow::Table>> LinearMapTable(const std::shared_ptr<arrow::Table>& input, const LinearMap& map,
                                                            const NumberColumns& rows, const std::vector<double>& mapped,
                                                            bool keep_columns, const std::string& label_col, const std::string& op_name) {
    const int64_t n = static_cast<int64_t>(rows.rows.size());
    const size_t k = map.outputs.size();
    if (keep_columns) {
        // Float64 beside the input, null on the rows left out.
        for (const auto& name : map.outputs)
            if (input->schema()->GetFieldIndex(name) >= 0)
                return arrow::Status::Invalid(op_name + ": the input already has a column '" + name +
                                              "' (an earlier step's output?): turn Keep columns off, or give the outputs another name");
        std::shared_ptr<arrow::Table> out = input;
        for (size_t c = 0; c < k; ++c) {
            arrow::DoubleBuilder b;
            ARROW_RETURN_NOT_OK(b.Reserve(rows.total));
            const double* column = mapped.data() + c * static_cast<size_t>(n);
            size_t next = 0;
            for (int64_t r = 0; r < rows.total; ++r) {
                if (next < rows.rows.size() && rows.rows[next] == r) b.UnsafeAppend(column[next++]);
                else b.UnsafeAppendNull();
            }
            std::shared_ptr<arrow::Array> a;
            ARROW_RETURN_NOT_OK(b.Finish(&a));
            ARROW_ASSIGN_OR_RAISE(out, out->AddColumn(out->num_columns(), arrow::field(map.outputs[c], arrow::float64()),
                                                      std::make_shared<arrow::ChunkedArray>(a)));
        }
        return out;
    }
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    for (size_t c = 0; c < k; ++c) {
        arrow::FloatBuilder b;
        ARROW_RETURN_NOT_OK(b.Reserve(n));
        const double* column = mapped.data() + c * static_cast<size_t>(n);
        for (int64_t r = 0; r < n; ++r) b.UnsafeAppend(static_cast<float>(column[r]));
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
        for (const int64_t row : rows.rows) b.UnsafeAppend(labels[static_cast<size_t>(row)]);
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
        for (const int64_t row : rows.rows) b.UnsafeAppend(parts[static_cast<size_t>(row)]);
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
