// Shared analysis device helpers (analysis_device.h), used by clustering_*.cpp and
// dimensionality_reduction.cpp; all run on ArrayFire on the active device, its CPU
// backend being the CPU option (TOFIX140).

#ifdef _WIN32
#define NOMINMAX
#endif

#include "analysis_device.h"

#include <algorithm>
#include <random>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace cyxwiz::analysis_detail {

std::string MatrixRefusal(const std::vector<std::vector<double>>& data) {
    if (data.empty()) return "Empty data";
    const size_t features = data[0].size();
    if (features == 0) return "Data has no features";
    for (const auto& row : data) {
        if (row.size() != features) return "All rows need the same number of features";
    }
    return {};
}

bool IsKnownMetric(const std::string& metric) {
    return metric == "euclidean" || metric == "manhattan" || metric == "cosine";
}

std::string NoArrayFire(const char* operation) {
    return std::string(operation) + " runs on ArrayFire, and this build has no ArrayFire";
}

#ifdef CYXWIZ_HAS_ARRAYFIRE

namespace {
constexpr dim_t kBlockElements = dim_t{1} << 22;
}

af::array MatrixToDevice(const Matrix& data) {
    const size_t rows = data.size(), cols = data[0].size();
    std::vector<double> flat(rows * cols);
    for (size_t c = 0; c < cols; ++c) {
        for (size_t r = 0; r < rows; ++r) flat[c * rows + r] = data[r][c];
    }
    return af::moddims(HostValuesToDevice(flat), static_cast<dim_t>(rows), static_cast<dim_t>(cols));
}

Matrix MatrixFromDevice(const af::array& source, const char* operation) {
    const size_t rows = static_cast<size_t>(source.dims(0)), cols = static_cast<size_t>(source.dims(1));
    const std::vector<double> values = DeviceToHostValues(source, operation);
    Matrix result(rows, std::vector<double>(cols));
    for (size_t c = 0; c < cols; ++c) {
        for (size_t r = 0; r < rows; ++r) result[r][c] = values[c * rows + r];
    }
    return result;
}

std::vector<int> LabelsFromDevice(const af::array& source, const char* operation) {
    std::vector<int> labels(static_cast<size_t>(source.elements()));
    if (labels.empty()) return labels;
    af::array values = source.as(s32);
    values.eval();
    MaterializeArrayFireToHost(values, labels.data(), ArrayFireHostSyncCategory::OutputMaterialization, operation,
                               "arrayfire_native");
    return labels;
}

dim_t RowBlock(dim_t n) {
    return std::max<dim_t>(1, kBlockElements / std::max<dim_t>(1, n));
}

af::array PairwiseDistances(const af::array& rows, const af::array& all, const std::string& metric) {
    const dim_t b = rows.dims(0), n = all.dims(0), d = rows.dims(1);
    if (metric == "cosine") {
        const auto unit = [](const af::array& x) {
            const af::array norms = af::sqrt(af::sum(x * x, 1));
            const af::array safe = af::max(norms, af::constant(1e-300, norms.dims(), norms.type()));
            return x / af::tile(safe, 1, static_cast<unsigned>(x.dims(1)));
        };
        return 1.0 - af::matmul(unit(rows), unit(all), AF_MAT_NONE, AF_MAT_TRANS);
    }
    const bool manhattan = metric == "manhattan";
    af::array total = af::constant(0.0, b, n, rows.type());
    for (dim_t f = 0; f < d; ++f) {
        const af::array diff = af::tile(rows(af::span, static_cast<int>(f)), 1, static_cast<unsigned>(n)) -
                               af::tile(all(af::span, static_cast<int>(f)).T(), static_cast<unsigned>(b), 1);
        total += manhattan ? af::abs(diff) : diff * diff;
        total.eval();  // one small kernel per feature, not one per matrix
    }
    return manhattan ? total : af::sqrt(total);
}

af::array SquaredDistances(const af::array& rows, const af::array& centers) {
    const dim_t b = rows.dims(0), k = centers.dims(0), d = rows.dims(1);
    af::array total = af::constant(0.0, b, k, rows.type());
    for (dim_t f = 0; f < d; ++f) {
        const af::array diff = af::tile(rows(af::span, static_cast<int>(f)), 1, static_cast<unsigned>(k)) -
                               af::tile(centers(af::span, static_cast<int>(f)).T(), static_cast<unsigned>(b), 1);
        total += diff * diff;
        total.eval();
    }
    return total;
}

af::randomEngine SeededEngine(unsigned int seed) {
    const unsigned long long value = seed != 0 ? seed : std::random_device{}();
    return af::randomEngine(AF_RANDOM_ENGINE_DEFAULT, value);
}

af::array KMeansPlusPlus(const af::array& data, int k, af::randomEngine& engine) {
    const dim_t n = data.dims(0);
    const auto draw_index = [&](const af::array& weights) {
        // Smallest index whose cumulative weight reaches a uniform draw in [0, total).
        const double total = af::sum<double>(weights);
        const double u = af::randu(1, f32, engine).scalar<float>();
        if (!(total > 0.0)) return std::min<dim_t>(static_cast<dim_t>(u * n), n - 1);
        const double reach = u * total;
        const dim_t below = static_cast<dim_t>(af::count<double>(af::accum(weights) <= reach));
        return std::min<dim_t>(below, n - 1);
    };
    const af::array uniform = af::constant(1.0, n, data.type());
    af::array centers = data(static_cast<int>(draw_index(uniform)), af::span);
    af::array nearest = SquaredDistances(data, centers);
    for (int c = 1; c < k; ++c) {
        const af::array next = data(static_cast<int>(draw_index(nearest)), af::span);
        centers = af::join(0, centers, next);
        nearest = af::min(nearest, SquaredDistances(data, next));
        nearest.eval();
    }
    return centers;
}

#endif

}  // namespace cyxwiz::analysis_detail
