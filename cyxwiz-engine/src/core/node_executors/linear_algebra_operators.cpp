#include "linear_algebra_operators.h"

#include "../sparse_feature_dataset.h"
#include "../split_partitioning.h"

#include <cyxwiz/linear_algebra.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace cyxwiz {

namespace {

bool ReadInt(const std::map<std::string, std::string>& params, const char* key, int fallback, int min, int& out,
             const std::string& op_name, std::string& error) {
    const auto it = params.find(key);
    out = fallback;
    if (it == params.end() || it->second.empty()) return true;
    char* end = nullptr;
    const long v = std::strtol(it->second.c_str(), &end, 10);
    if (!end || *end != '\0') return error = op_name + ": '" + key + "' is not a whole number: " + it->second, false;
    if (v < min) return error = op_name + ": " + key + " must be >= " + std::to_string(min) + " (got " + it->second + ")", false;
    out = static_cast<int>(v);
    return true;
}

// The rows of Vt [k × d] signed so each one's largest-magnitude entry is positive.
void SignComponents(LinearMatrix& vt) {
    for (auto& row : vt) {
        size_t peak = 0;
        for (size_t j = 1; j < row.size(); ++j)
            if (std::fabs(row[j]) > std::fabs(row[peak])) peak = j;
        if (!row.empty() && row[peak] < 0)
            for (double& v : row) v = -v;
    }
}

}  // namespace

// ---- the shared step

bool FittedLinearOperator::Configure(const std::map<std::string, std::string>& params, std::string& error) {
    const auto fc = params.find("feature_cols");
    feature_cols_ = ParseColumnList(fc != params.end() ? fc->second : "");
    const auto lc = params.find("label_col");
    label_col_ = lc != params.end() ? lc->second : "";
    if (!ReadBoolParameter(params, "keep_columns", false, keep_columns_, GetName(), error)) return false;
    if (!ParseFittedPreprocessingOptions(params, GetName(), options_, error)) return false;
    return ConfigureOwn(params, error);
}

arrow::Result<std::shared_ptr<arrow::Table>> FittedLinearOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    const std::string name = GetName();
    if (!input) return arrow::Status::Invalid(name + ": input table is null");
    const auto progress = [&](const char* stage, const std::string& message, float value) {
        if (!progress_callback_) return;
        PipelineOperatorProgress event;
        event.stage = stage;
        event.message = message;
        event.status = "running";
        event.progress = value;
        progress_callback_(event);
    };
    std::vector<std::string> features;
    ARROW_RETURN_NOT_OK(ResolveNumberFeatures(input, feature_cols_, label_col_, name, features));
    progress("Reading feature columns", "Reading " + std::to_string(features.size()) + " number columns", 0.1f);
    FeatureRows rows;
    ARROW_RETURN_NOT_OK(ReadFeatureRows(input, features, name, false, rows, GetCancellationQuery()));
    ARROW_RETURN_NOT_OK(CheckCancellation(name));

    LinearMap map;
    int64_t fit_rows = 0;
    if (options_.IsTransformOnly()) {
        FittedPreprocessingState state;
        std::string error;
        if (!LoadFittedPreprocessingState(options_.state_path, name, state, error) ||
            !ValidateFittedPreprocessingConfiguration(state, Settings(), error) || !LinearMapFromState(state, map, error))
            return arrow::Status::Invalid(name + ": " + error);
        if (map.inputs != features)
            return arrow::Status::Invalid(name + ": the feature columns differ from the saved values' columns. Keep the columns used when fitting.");
        fit_rows = state.fit_rows;
    } else {
        // Fitted on the Train rows (all rows when the run marks none).
        const LinearMatrix train = TrainSubset(input, rows);
        if (train.size() < 2) return arrow::Status::Invalid(name + ": fewer than 2 Train rows have every feature");
        progress("Fitting", "Fitting on " + std::to_string(train.size()) + " Train rows", 0.4f);
        ARROW_RETURN_NOT_OK(Fit(train, features, map));
        fit_rows = static_cast<int64_t>(train.size());
    }
    ARROW_RETURN_NOT_OK(CheckCancellation(name));
    progress("Mapping rows", "Mapping " + std::to_string(rows.rows.size()) + " rows on the device", 0.7f);
    LinearMatrix mapped;
    std::string error;
    if (!ApplyLinearMap(map, rows.data, mapped, error)) return arrow::Status::ExecutionError(name + ": " + error);
    if (!options_.IsTransformOnly() && options_.save_state) {
        if (!SaveFittedPreprocessingState(options_.state_path, LinearMapState(map, name, fit_rows, input->schema(), Settings()),
                                          options_.state_overwrite, error))
            return arrow::Status::IOError(name + ": failed to save the fitted values: " + error);
    }
    spdlog::info("{}: {} rows x {} features -> {} columns ({} on {} Train rows)", name, rows.rows.size(), features.size(),
                 map.outputs.size(), options_.IsTransformOnly() ? "saved values fitted" : "fitted", fit_rows);
    progress("Complete", name + " complete", 1.0f);
    return LinearMapTable(input, map, rows, mapped, keep_columns_, label_col_, name);
}

// ---- Truncated SVD

bool TruncatedSVDOperator::ConfigureOwn(const std::map<std::string, std::string>& params, std::string& error) {
    return ReadInt(params, "n_components", 2, 1, n_components_, GetName(), error);
}

std::map<std::string, std::string> TruncatedSVDOperator::Settings() const { return {{"n_components", std::to_string(n_components_)}}; }

namespace {

// The map x V_k from the right singular vectors Vt (rows), signed.
void SvdMap(const LinearMatrix& right, const std::vector<std::string>& features, size_t k, LinearMap& map) {
    const size_t d = features.size();
    LinearMatrix vt(right.begin(), right.begin() + static_cast<std::ptrdiff_t>(k));
    SignComponents(vt);
    map.inputs = features;
    map.shift.assign(d, 0.0);
    map.divisor.assign(d, 1.0);
    map.w.assign(d, std::vector<double>(k));
    for (size_t c = 0; c < k; ++c) {
        map.outputs.push_back("svd_" + std::to_string(c));
        for (size_t j = 0; j < d; ++j) map.w[j][c] = vt[c][j];
    }
}

}  // namespace

arrow::Status TruncatedSVDOperator::Fit(const LinearMatrix& train, const std::vector<std::string>& features, LinearMap& map) {
    const size_t d = features.size(), k = static_cast<size_t>(n_components_);
    if (k > std::min(train.size(), d))
        return arrow::Status::Invalid("TruncatedSVD: n_components (" + std::to_string(k) + ") is above min(Train rows, features) = " +
                                      std::to_string(std::min(train.size(), d)));
    const SVDResult svd = LinearAlgebra::SVD(train, false);
    if (!svd.success) return arrow::Status::ExecutionError("TruncatedSVD: " + svd.error_message);
    SvdMap(svd.Vt, features, k, map);
    return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::Table>> TruncatedSVDOperator::ApplySparseInput(const SparseFeatureDataset& input) {
    const std::string name = GetName();
    if (keep_columns_) return arrow::Status::Invalid(name + ": Keep columns needs a table; sparse features have no columns to keep");
    std::vector<std::string> features = input.GetFeatureNames();
    if (features.size() != static_cast<size_t>(input.GetNumFeatures())) {
        features.clear();
        for (int64_t j = 0; j < input.GetNumFeatures(); ++j) features.push_back("f" + std::to_string(j));
    }
    const auto& offsets = input.GetRowOffsets();
    const auto& columns = input.GetColumnIndices();
    const auto& values = input.GetValues();
    const auto& partition = input.GetPartition();
    CsrView all;
    all.rows = static_cast<int>(input.GetNumRows());
    all.cols = static_cast<int>(input.GetNumFeatures());
    all.row_offsets = offsets.data();
    all.column_indices = columns.data();
    all.values = values.data();
    LinearMap map;
    int64_t fit_rows = 0;
    if (options_.IsTransformOnly()) {
        FittedPreprocessingState state;
        std::string error;
        if (!LoadFittedPreprocessingState(options_.state_path, name, state, error) ||
            !ValidateFittedPreprocessingConfiguration(state, Settings(), error) || !LinearMapFromState(state, map, error))
            return arrow::Status::Invalid(name + ": " + error);
        if (map.inputs != features)
            return arrow::Status::Invalid(name + ": the features differ from the saved values' (fit the vectorizer and this step together)");
        fit_rows = state.fit_rows;
    } else {
        // The Train rows' CSR (all rows when the run marks none).
        std::vector<int> train_offsets{0}, train_columns;
        std::vector<float> train_values;
        CsrView train = all;
        if (!partition.empty()) {
            for (int r = 0; r < all.rows; ++r) {
                if (partition[static_cast<size_t>(r)] != 0) continue;
                for (int k = offsets[static_cast<size_t>(r)]; k < offsets[static_cast<size_t>(r) + 1]; ++k) {
                    train_columns.push_back(columns[static_cast<size_t>(k)]);
                    train_values.push_back(values[static_cast<size_t>(k)]);
                }
                train_offsets.push_back(static_cast<int>(train_columns.size()));
            }
            train.rows = static_cast<int>(train_offsets.size()) - 1;
            train.row_offsets = train_offsets.data();
            train.column_indices = train_columns.data();
            train.values = train_values.data();
        }
        const size_t k = static_cast<size_t>(n_components_);
        if (train.rows < 2) return arrow::Status::Invalid(name + ": fewer than 2 Train rows");
        if (k > std::min<size_t>(static_cast<size_t>(train.rows), features.size()))
            return arrow::Status::Invalid(name + ": n_components (" + std::to_string(k) + ") is above min(Train rows, features) = " +
                                          std::to_string(std::min<size_t>(static_cast<size_t>(train.rows), features.size())));
        // The right singular vectors of X are the eigenvectors of X^T X.
        const MatrixResult gram = LinearAlgebra::SparseGram(train);
        if (!gram.success) return arrow::Status::ExecutionError(name + ": " + gram.error_message);
        ARROW_RETURN_NOT_OK(CheckCancellation(name));
        const SVDResult svd = LinearAlgebra::SVD(gram.matrix, false);
        if (!svd.success) return arrow::Status::ExecutionError(name + ": " + svd.error_message);
        SvdMap(svd.Vt, features, k, map);
        fit_rows = train.rows;
    }
    ARROW_RETURN_NOT_OK(CheckCancellation(name));
    const MatrixResult mapped = LinearAlgebra::SparseMultiply(all, map.w);
    if (!mapped.success) return arrow::Status::ExecutionError(name + ": " + mapped.error_message);
    if (!options_.IsTransformOnly() && options_.save_state) {
        std::string error;
        FittedPreprocessingState state = LinearMapState(map, name, fit_rows, nullptr, Settings());
        if (!SaveFittedPreprocessingState(options_.state_path, state, options_.state_overwrite, error))
            return arrow::Status::IOError(name + ": failed to save the fitted values: " + error);
    }
    // svd_* (Float32), the label as y, the split.
    const int64_t n = input.GetNumRows();
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::ChunkedArray>> arrays;
    for (size_t c = 0; c < map.outputs.size(); ++c) {
        arrow::FloatBuilder b;
        ARROW_RETURN_NOT_OK(b.Reserve(n));
        for (int64_t r = 0; r < n; ++r) b.UnsafeAppend(static_cast<float>(mapped.matrix[static_cast<size_t>(r)][c]));
        std::shared_ptr<arrow::Array> a;
        ARROW_RETURN_NOT_OK(b.Finish(&a));
        fields.push_back(arrow::field(map.outputs[c], arrow::float32()));
        arrays.push_back(std::make_shared<arrow::ChunkedArray>(a));
    }
    if (const auto& labels = input.GetLabels(); labels && labels->length() == n) {
        fields.push_back(arrow::field("y", labels->type()));
        arrays.push_back(std::make_shared<arrow::ChunkedArray>(labels->chunks(), labels->type()));
    }
    if (static_cast<int64_t>(partition.size()) == n) {
        arrow::Int8Builder b;
        ARROW_RETURN_NOT_OK(b.AppendValues(partition));
        std::shared_ptr<arrow::Array> a;
        ARROW_RETURN_NOT_OK(b.Finish(&a));
        fields.push_back(arrow::field(kSplitPartitionColumn, arrow::int8()));
        arrays.push_back(std::make_shared<arrow::ChunkedArray>(a));
    }
    spdlog::info("{}: {} rows x {} sparse features -> {} components (fitted on {} Train rows)", name, n, features.size(),
                 map.outputs.size(), fit_rows);
    return arrow::Table::Make(arrow::schema(fields), arrays, n);
}

// ---- Whitening

bool WhiteningOperator::ConfigureOwn(const std::map<std::string, std::string>& params, std::string& error) {
    const auto m = params.find("method");
    method_ = m == params.end() || m->second.empty() ? "pca" : m->second;
    if (method_ != "pca" && method_ != "zca") return error = "Whitening: method must be 'pca' or 'zca' (got '" + method_ + "')", false;
    if (!ReadInt(params, "n_components", 0, 0, n_components_, GetName(), error)) return false;
    epsilon_ = 0.0;
    if (const auto e = params.find("epsilon"); e != params.end() && !e->second.empty()) {
        char* end = nullptr;
        epsilon_ = std::strtod(e->second.c_str(), &end);
        if (!end || *end != '\0' || !(epsilon_ >= 0.0) || !std::isfinite(epsilon_))
            return error = "Whitening: epsilon must be a number of 0 or more (got '" + e->second + "')", false;
    }
    return true;
}

std::map<std::string, std::string> WhiteningOperator::Settings() const {
    char eps[32];
    std::snprintf(eps, sizeof(eps), "%.17g", epsilon_);
    return {{"method", method_}, {"n_components", std::to_string(n_components_)}, {"epsilon", eps}};
}

arrow::Status WhiteningOperator::Fit(const LinearMatrix& train, const std::vector<std::string>& features, LinearMap& map) {
    const size_t d = features.size(), n = train.size();
    std::vector<double> mean, sd;
    LinearMatrix centred;
    std::string error;
    if (!ColumnMoments(train, mean, sd, error) || !Centred(train, mean, centred, error))
        return arrow::Status::ExecutionError("Whitening: " + error);
    const SVDResult svd = LinearAlgebra::SVD(centred, false);
    if (!svd.success) return arrow::Status::ExecutionError("Whitening: " + svd.error_message);
    LinearMatrix vt = svd.Vt;
    SignComponents(vt);
    // Each direction's variance (divides by n - 1, as PCA's explained variance).
    std::vector<double> scale(svd.S.size());
    const double top = svd.S.empty() ? 0.0 : svd.S[0] * svd.S[0] / static_cast<double>(n - 1);
    const size_t k = method_ == "zca" ? svd.S.size() : (n_components_ == 0 ? svd.S.size() : static_cast<size_t>(n_components_));
    if (k > svd.S.size())
        return arrow::Status::Invalid("Whitening: n_components (" + std::to_string(k) + ") is above min(Train rows, features) = " +
                                      std::to_string(svd.S.size()));
    for (size_t c = 0; c < svd.S.size(); ++c) {
        const double variance = svd.S[c] * svd.S[c] / static_cast<double>(n - 1);
        if (c < k && epsilon_ == 0.0 && !(variance > 1e-12 * std::max(1.0, top)))
            return arrow::Status::Invalid("Whitening: direction " + std::to_string(c + 1) +
                                          " of the Train rows has no variance, so it cannot be scaled to 1: " +
                                          (method_ == "pca" ? "keep fewer components, or " : "") + "set epsilon above 0.");
        scale[c] = 1.0 / std::sqrt(variance + epsilon_);
    }
    if (method_ == "zca" && svd.S.size() < d)
        return arrow::Status::Invalid("Whitening: ZCA needs more Train rows (" + std::to_string(n) + ") than features (" + std::to_string(d) + ")");
    map.inputs = features;
    map.shift = mean;
    map.divisor.assign(d, 1.0);
    if (method_ == "pca") {
        map.w.assign(d, std::vector<double>(k));
        for (size_t c = 0; c < k; ++c) {
            map.outputs.push_back("white_" + std::to_string(c));
            for (size_t j = 0; j < d; ++j) map.w[j][c] = vt[c][j] * scale[c];
        }
        return arrow::Status::OK();
    }
    // ZCA: V diag(scale) V^T, on the device.
    LinearMatrix scaled = vt;  // diag(scale) V^T
    for (size_t c = 0; c < scaled.size(); ++c)
        for (double& v : scaled[c]) v *= scale[c];
    const MatrixResult w = LinearAlgebra::Multiply(LinearAlgebra::Transpose(vt).matrix, scaled);
    if (!w.success) return arrow::Status::ExecutionError("Whitening: " + w.error_message);
    map.w = w.matrix;
    for (const auto& f : features) map.outputs.push_back(f + "_white");
    return arrow::Status::OK();
}

// ---- Linear Solve

namespace {

std::shared_ptr<arrow::Table> XTable(const std::vector<std::string>& a_cols, const std::vector<std::string>& b_cols, const LinearMatrix& x) {
    arrow::StringBuilder names;
    for (const auto& n : a_cols) (void)names.Append(n);
    std::shared_ptr<arrow::Array> name_array;
    (void)names.Finish(&name_array);
    std::vector<std::shared_ptr<arrow::Field>> fields = {arrow::field("column", arrow::utf8())};
    std::vector<std::shared_ptr<arrow::Array>> arrays = {name_array};
    for (size_t c = 0; c < b_cols.size(); ++c) {
        arrow::DoubleBuilder b;
        for (const auto& row : x) (void)b.Append(row[c]);
        std::shared_ptr<arrow::Array> a;
        (void)b.Finish(&a);
        fields.push_back(arrow::field("x_" + b_cols[c], arrow::float64()));
        arrays.push_back(a);
    }
    return arrow::Table::Make(arrow::schema(fields), arrays);
}

// Strings of a text column (null: "").
std::vector<std::string> TextColumn(const std::shared_ptr<arrow::ChunkedArray>& column) {
    std::vector<std::string> out;
    for (const auto& chunk : column->chunks()) {
        if (chunk->type_id() == arrow::Type::STRING) {
            const auto s = std::static_pointer_cast<arrow::StringArray>(chunk);
            for (int64_t i = 0; i < s->length(); ++i) out.push_back(s->IsNull(i) ? "" : s->GetString(i));
        } else if (chunk->type_id() == arrow::Type::LARGE_STRING) {
            const auto s = std::static_pointer_cast<arrow::LargeStringArray>(chunk);
            for (int64_t i = 0; i < s->length(); ++i) out.push_back(s->IsNull(i) ? "" : s->GetString(i));
        }
    }
    return out;
}

}  // namespace

bool LinearSolveOperator::Configure(const std::map<std::string, std::string>& params, std::string& error) {
    const auto list = [&](const char* key) {
        const auto it = params.find(key);
        return ParseColumnList(it != params.end() ? it->second : "");
    };
    a_cols_ = list("a_cols");
    b_cols_ = list("b_cols");
    if (b_cols_.empty()) return error = "LinearSolve: b_cols names b's column(s)", false;
    const auto m = params.find("method");
    method_ = m == params.end() || m->second.empty() ? "auto" : m->second;
    if (method_ != "auto" && method_ != "solve" && method_ != "least_squares")
        return error = "LinearSolve: method must be 'auto', 'solve' or 'least_squares' (got '" + method_ + "')", false;
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>> LinearSolveOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    if (!input) return arrow::Status::Invalid("LinearSolve: input table is null");
    const bool two_tables = b_table_ != nullptr;
    const std::shared_ptr<arrow::Table> b_source = two_tables ? b_table_ : input;
    // A's columns: given, or every number column of the input but b's.
    std::vector<std::string> a_cols;
    if (!a_cols_.empty()) {
        ARROW_RETURN_NOT_OK(ResolveNumberFeatures(input, a_cols_, "", GetName(), a_cols));
    } else {
        std::vector<std::string> all;
        ARROW_RETURN_NOT_OK(ResolveNumberFeatures(input, {}, "", GetName(), all));
        for (const auto& c : all)
            if (two_tables || std::find(b_cols_.begin(), b_cols_.end(), c) == b_cols_.end()) a_cols.push_back(c);
        if (a_cols.empty()) return arrow::Status::Invalid("LinearSolve: no number column is left for A");
    }
    std::vector<std::string> b_cols;
    ARROW_RETURN_NOT_OK(ResolveNumberFeatures(b_source, b_cols_, "", GetName(), b_cols));
    LinearMatrix a, b;
    if (two_tables) {
        // Two tables line up row by row: the same count, no row left out of one side.
        if (b_table_->num_rows() != input->num_rows())
            return arrow::Status::Invalid("LinearSolve: A has " + std::to_string(input->num_rows()) + " rows and b has " +
                                          std::to_string(b_table_->num_rows()) + ": two tables line up row by row");
        FeatureRows ar, br;
        ARROW_RETURN_NOT_OK(ReadFeatureRows(input, a_cols, GetName(), false, ar, GetCancellationQuery()));
        ARROW_RETURN_NOT_OK(ReadFeatureRows(b_table_, b_cols, GetName(), false, br, GetCancellationQuery()));
        if (ar.rows.size() != static_cast<size_t>(ar.total) || br.rows.size() != static_cast<size_t>(br.total))
            return arrow::Status::Invalid(std::string("LinearSolve: A and b come from two tables and ") +
                                          (ar.rows.size() != static_cast<size_t>(ar.total) ? "A" : "b") +
                                          " has rows with a missing value: the rows cannot be lined up with one side left out. Fill or remove them first.");
        const std::vector<bool> train = TrainRowMask(input);
        for (size_t i = 0; i < ar.data.size(); ++i)
            if (train.empty() || train[i]) {
                a.push_back(ar.data[i]);
                b.push_back(br.data[i]);
            }
    } else {
        std::vector<std::string> both = a_cols;
        both.insert(both.end(), b_cols.begin(), b_cols.end());
        FeatureRows rows;
        ARROW_RETURN_NOT_OK(ReadFeatureRows(input, both, GetName(), false, rows, GetCancellationQuery()));
        for (const auto& row : TrainSubset(input, rows)) {
            a.emplace_back(row.begin(), row.begin() + static_cast<std::ptrdiff_t>(a_cols.size()));
            b.emplace_back(row.begin() + static_cast<std::ptrdiff_t>(a_cols.size()), row.end());
        }
    }
    ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
    if (a.empty()) return arrow::Status::Invalid("LinearSolve: no row has every column of A and b");
    const ScalarResult rank = LinearAlgebra::Rank(a);
    if (!rank.success) return arrow::Status::ExecutionError("LinearSolve: " + rank.error_message);
    const bool square = a.size() == a_cols.size();
    const bool full = static_cast<size_t>(std::lround(rank.value)) == a_cols.size();
    const bool solve = method_ == "solve" || (method_ == "auto" && square && full);
    if (solve && !(square && full))
        return arrow::Status::Invalid("LinearSolve: A x = b needs a square A of full rank; A is " + std::to_string(a.size()) + " × " +
                                      std::to_string(a_cols.size()) + " of rank " + std::to_string(std::lround(rank.value)) +
                                      ". Least squares takes any A.");
    const MatrixResult x = solve ? LinearAlgebra::Solve(a, b) : LinearAlgebra::LeastSquares(a, b);
    if (!x.success) return arrow::Status::ExecutionError("LinearSolve: " + x.error_message);
    spdlog::info("LinearSolve: {} rows, A {} columns of rank {}, b {} columns: {}", a.size(), a_cols.size(), std::lround(rank.value),
                 b_cols.size(), solve ? "the one solution" : "least squares");
    return XTable(a_cols, b_cols, x.matrix);
}

// ---- Linear Projection

bool LinearProjectionOperator::Configure(const std::map<std::string, std::string>& params, std::string& error) {
    const auto fc = params.find("feature_cols");
    feature_cols_ = ParseColumnList(fc != params.end() ? fc->second : "");
    const auto lc = params.find("label_col");
    label_col_ = lc != params.end() ? lc->second : "";
    const auto px = params.find("prefix");
    prefix_ = px != params.end() ? px->second : "";
    return ReadBoolParameter(params, "keep_columns", false, keep_columns_, GetName(), error);
}

arrow::Result<std::shared_ptr<arrow::Table>> LinearProjectionOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    if (!input) return arrow::Status::Invalid("LinearProjection: input table is null");
    if (!matrix_) return arrow::Status::Invalid("LinearProjection: link a table to the Matrix input (its number columns are the outputs)");
    // The matrix: its number columns are the outputs; a text column names the data column of each row.
    std::vector<std::string> outputs, names;
    for (const auto& field : matrix_->schema()->fields()) {
        const auto id = field->type()->id();
        if (id == arrow::Type::STRING || id == arrow::Type::LARGE_STRING) {
            if (names.empty()) names = TextColumn(matrix_->GetColumnByName(field->name()));
        } else if (field->name().rfind("__", 0) != 0) {
            outputs.push_back(field->name());
        }
    }
    std::vector<std::string> checked;
    ARROW_RETURN_NOT_OK(ResolveNumberFeatures(matrix_, outputs, "", GetName(), checked));
    FeatureRows m;
    ARROW_RETURN_NOT_OK(ReadFeatureRows(matrix_, outputs, GetName(), false, m));
    if (m.rows.size() != static_cast<size_t>(m.total)) return arrow::Status::Invalid("LinearProjection: the matrix has a missing value");
    std::vector<std::string> features;
    if (!names.empty()) {
        ARROW_RETURN_NOT_OK(ResolveNumberFeatures(input, names, "", GetName(), features));
    } else {
        ARROW_RETURN_NOT_OK(ResolveNumberFeatures(input, feature_cols_, label_col_, GetName(), features));
    }
    if (features.size() != m.data.size())
        return arrow::Status::Invalid("LinearProjection: the matrix has " + std::to_string(m.data.size()) + " rows and the data " +
                                      std::to_string(features.size()) + " columns: one row per data column (or name them in a text column)");
    LinearMap map;
    map.inputs = features;
    map.shift.assign(features.size(), 0.0);
    map.divisor.assign(features.size(), 1.0);
    map.w = m.data;
    for (const auto& o : outputs) map.outputs.push_back(prefix_ + o);
    FeatureRows rows;
    ARROW_RETURN_NOT_OK(ReadFeatureRows(input, features, GetName(), false, rows, GetCancellationQuery()));
    LinearMatrix mapped;
    std::string error;
    if (!ApplyLinearMap(map, rows.data, mapped, error)) return arrow::Status::ExecutionError("LinearProjection: " + error);
    return LinearMapTable(input, map, rows, mapped, keep_columns_, label_col_, GetName());
}

}  // namespace cyxwiz
