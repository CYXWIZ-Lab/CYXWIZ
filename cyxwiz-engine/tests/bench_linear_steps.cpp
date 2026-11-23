// How long the linear-algebra steps take on large tables (TOFIX129 D9):
// PCA, Truncated SVD, Whitening, Linear Projection and Linear Solve on dense
// tables (fitted on the Train rows, 80 %), and Truncated SVD on sparse
// TF-IDF-shaped features, with the time and the process's peak memory of
// each. Not a check: it prints, for the ticket and the docs; the same shapes
// through scikit-learn: bench_linear_steps_sklearn.py.
//
//   bench_linear_steps [rows] [columns]   (default: 200000 50, then 1000000 20)

#include "../src/core/node_executors/linear_algebra_operators.h"
#include <cyxwiz/linear_algebra.h>
#include "../src/core/node_executors/pca_operator.h"
#include "../src/core/sparse_feature_dataset.h"
#include "../src/core/split_partitioning.h"

#include <arrow/api.h>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace cyxwiz;

namespace {

double PeakMB() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    return static_cast<double>(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0);
#else
    return 0.0;
#endif
}

// rows × columns of correlated normals (each column a mix of 5 factors and noise), the split column 80 / 10 / 10.
std::shared_ptr<arrow::Table> Dense(int64_t rows, int columns) {
    std::mt19937 rng(7);
    std::normal_distribution<double> z(0.0, 1.0);
    std::vector<std::vector<double>> mix(static_cast<size_t>(columns), std::vector<double>(5));
    for (auto& m : mix)
        for (double& v : m) v = z(rng);
    std::vector<arrow::DoubleBuilder> builders(static_cast<size_t>(columns));
    for (auto& b : builders) (void)b.Reserve(rows);
    arrow::Int8Builder part;
    (void)part.Reserve(rows);
    std::vector<double> factor(5);
    for (int64_t r = 0; r < rows; ++r) {
        for (double& f : factor) f = z(rng);
        for (int c = 0; c < columns; ++c) {
            double v = 0.3 * z(rng);
            for (int k = 0; k < 5; ++k) v += mix[static_cast<size_t>(c)][static_cast<size_t>(k)] * factor[static_cast<size_t>(k)];
            builders[static_cast<size_t>(c)].UnsafeAppend(v);
        }
        part.UnsafeAppend(static_cast<int8_t>(r % 10 < 8 ? 0 : (r % 10 == 8 ? 1 : 2)));
    }
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    for (int c = 0; c < columns; ++c) {
        std::shared_ptr<arrow::Array> a;
        (void)builders[static_cast<size_t>(c)].Finish(&a);
        fields.push_back(arrow::field("x" + std::to_string(c), arrow::float64()));
        arrays.push_back(a);
    }
    std::shared_ptr<arrow::Array> p;
    (void)part.Finish(&p);
    fields.push_back(arrow::field(kSplitPartitionColumn, arrow::int8()));
    arrays.push_back(p);
    return arrow::Table::Make(arrow::schema(fields), arrays);
}

// A columns × outputs matrix table (one row per data column).
std::shared_ptr<arrow::Table> Matrix(int columns, int outputs) {
    std::mt19937 rng(3);
    std::normal_distribution<double> z(0.0, 1.0);
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    for (int o = 0; o < outputs; ++o) {
        arrow::DoubleBuilder b;
        for (int c = 0; c < columns; ++c) (void)b.Append(z(rng));
        std::shared_ptr<arrow::Array> a;
        (void)b.Finish(&a);
        fields.push_back(arrow::field("p" + std::to_string(o), arrow::float64()));
        arrays.push_back(a);
    }
    return arrow::Table::Make(arrow::schema(fields), arrays);
}

void Time(const char* what, IPipelineOperator& op, const std::map<std::string, std::string>& params,
          const std::shared_ptr<arrow::Table>& input) {
    std::string error;
    if (!op.Configure(params, error)) {
        std::cout << "  " << what << ": configure failed: " << error << '\n';
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const auto out = op.Apply(input);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("  %-34s %8.2f s   peak %7.0f MB   %s\n", what, s, PeakMB(),
                out.ok() ? (std::to_string((*out)->num_rows()) + " rows x " + std::to_string((*out)->num_columns()) + " columns").c_str()
                         : out.status().ToString().c_str());
}

void DenseRun(int64_t rows, int columns) {
    std::printf("dense: %lld rows x %d columns (fitted on the 80 %% Train rows)\n", static_cast<long long>(rows), columns);
    const auto table = Dense(rows, columns);
    const std::string k = std::to_string(std::min(10, columns));
    PCAOperator pca;
    Time("PCA (10)", pca, {{"n_components", k}}, table);
    TruncatedSVDOperator svd;
    Time("Truncated SVD (10)", svd, {{"n_components", k}}, table);
    WhiteningOperator white;
    Time("Whitening PCA (10)", white, {{"n_components", k}}, table);
    WhiteningOperator zca;
    Time("Whitening ZCA", zca, {{"method", "zca"}}, table);
    LinearProjectionOperator projection;
    projection.SetSecondTable(Matrix(columns, std::min(10, columns)));
    Time("Linear Projection (10)", projection, {}, table);
    LinearSolveOperator solve;
    Time("Linear Solve (least squares, 1 b)", solve, {{"b_cols", "x0"}}, table);
}

void SparseRun(int64_t rows, int vocabulary, int per_row, int k) {
    std::printf("sparse: %lld rows x %d features, %d entries a row (TF-IDF-shaped)\n", static_cast<long long>(rows), vocabulary, per_row);
    std::mt19937 rng(11);
    std::uniform_int_distribution<int> term(0, vocabulary - 1);
    std::uniform_real_distribution<float> weight(0.01f, 1.0f);
    SparseFeatureDataset::Contents c;
    c.name = "bench_sparse";
    c.num_rows = rows;
    c.num_features = vocabulary;
    c.row_offsets.push_back(0);
    std::vector<int> row;
    for (int64_t r = 0; r < rows; ++r) {
        row.clear();
        for (int e = 0; e < per_row; ++e) row.push_back(term(rng));
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
        for (const int t : row) {
            c.column_indices.push_back(t);
            c.values.push_back(weight(rng));
        }
        c.row_offsets.push_back(static_cast<int32_t>(c.column_indices.size()));
        c.partition.push_back(static_cast<int8_t>(r % 10 < 8 ? 0 : (r % 10 == 8 ? 1 : 2)));
    }
    for (int j = 0; j < vocabulary; ++j) c.feature_names.push_back("t" + std::to_string(j));
    auto dataset = SparseFeatureDataset::Create(std::move(c));
    if (!dataset.ok()) {
        std::cout << "  sparse dataset: " << dataset.status().ToString() << '\n';
        return;
    }
    TruncatedSVDOperator svd;
    std::string error;
    (void)svd.Configure({{"n_components", std::to_string(k)}}, error);
    const auto t0 = std::chrono::steady_clock::now();
    const auto out = svd.ApplySparseInput(**dataset);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("  %-34s %8.2f s   peak %7.0f MB   %s\n", ("Truncated SVD (" + std::to_string(k) + ") sparse").c_str(), s, PeakMB(),
                out.ok() ? (std::to_string((*out)->num_rows()) + " rows").c_str() : out.status().ToString().c_str());
}

}  // namespace

int main(int argc, char** argv) {
    // The device starts once (CUDA's context, ArrayFire's kernels) before anything is timed.
    const LinearMatrix one = {{1.0}};
    (void)LinearAlgebra::Multiply(one, one);
    (void)LinearAlgebra::SVD(LinearMatrix{{1.0, 2.0}, {3.0, 4.0}}, false);
    if (argc >= 3) {
        DenseRun(std::atoll(argv[1]), std::atoi(argv[2]));
        return 0;
    }
    DenseRun(200000, 50);
    DenseRun(1000000, 20);
    SparseRun(100000, 2000, 50, 50);
    return 0;
}
