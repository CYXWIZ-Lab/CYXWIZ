#include "metric_learning_mining.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#include "algorithms/arrayfire_host_values.h"
#endif

namespace cyxwiz {

#ifdef CYXWIZ_HAS_ARRAYFIRE
namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

template <typename Fn>
auto OnDevice(const char* operation, Fn&& fn) {
    try {
        return fn();
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError(operation, e));
    }
}

// Per row of an [N, M] condition: the first column where it holds, or M.
af::array FirstColumnWhere(const af::array& condition) {
    const dim_t columns = condition.dims(1);
    const af::array index = af::range(condition.dims(), 1, s32);
    return af::flat((af::min)(af::select(condition, index, static_cast<double>(columns)), 1)).as(s32);
}

// Per row: the first column attaining the masked maximum (or minimum).
af::array FirstExtremeColumn(const af::array& values, const af::array& mask, bool maximum) {
    const unsigned columns = static_cast<unsigned>(values.dims(1));
    const af::array masked = af::select(mask, values, maximum ? -kInfinity : kInfinity);
    const af::array extreme = maximum ? (af::max)(masked, 1) : (af::min)(masked, 1);
    return FirstColumnWhere(mask && (masked == af::tile(extreme, 1, columns)));
}

Tensor IndexTensor(const af::array& values) {
    af::array indices = af::flat(values).as(s32);
    indices.eval();
    return Tensor::FromRowMajorFlatArray(indices, {static_cast<size_t>(indices.elements())});
}

struct RowMasks {
    af::array distances;  // [N, N], [a, j]
    af::array positive;   // same class, not the row itself
    af::array negative;   // other class
    dim_t n = 0;
};

RowMasks Masks(const Tensor& class_ids, const Tensor& distances) {
    RowMasks m;
    m.distances = distances.GetArrayRowMajor2D();
    m.n = m.distances.dims(0);
    const af::array id = af::flat(class_ids.GetRowMajorFlatArray());
    if (id.elements() != m.n || m.distances.dims(1) != m.n) {
        throw std::invalid_argument("metric mining: class ids and distances disagree on the batch size");
    }
    const unsigned n = static_cast<unsigned>(m.n);
    const af::array same = af::tile(id, 1, n) == af::tile(id.T(), n, 1);
    m.positive = same && !af::identity(m.n, m.n, b8);
    m.negative = !same;
    return m;
}

}  // namespace
#endif

Tensor BatchClassIds(const Tensor& labels, size_t rows) {
    const std::string builder = "The Pair / Triplet Dataset Builder";
    if (labels.NumElements() != rows) {
        throw std::runtime_error(builder + " needs one class id per row; the labels have " +
                                 std::to_string(labels.NumElements()) + " values for " + std::to_string(rows) +
                                 " rows");
    }
    const DataType dtype = labels.GetDataType();
    if (dtype != DataType::Int32 && dtype != DataType::Int64 && dtype != DataType::Float32) {
        throw std::runtime_error(builder + " needs Int32, Int64 or Float32 class ids");
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    return OnDevice("BatchClassIds", [&]() {
        const af::array ids = af::flat(labels.GetRowMajorFlatArray());
        if (dtype == DataType::Float32) {
            const af::array bad = af::where(af::isNaN(ids) || af::isInf(ids) || (ids != af::round(ids)));
            if (bad.elements() > 0) {
                const Tensor value = Tensor::FromRowMajorFlatArray(ids(af::array(bad(0))).as(f64), {1});
                throw std::runtime_error(builder + " needs whole-number class ids; a label is " +
                                         std::to_string(value.ReadData<double>()[0]));
            }
        }
        af::array result = ids.as(s64);
        result.eval();
        return Tensor::FromRowMajorFlatArray(result, {rows});
    });
#else
    throw std::runtime_error("Metric mining runs on ArrayFire, and this build has no ArrayFire");
#endif
}

std::vector<int64_t> ReadBatchClassIds(const Tensor& labels, size_t rows) {
    const Tensor ids = BatchClassIds(labels, rows);
    const int64_t* values = ids.ReadData<int64_t>();
    return std::vector<int64_t>(values, values + rows);
}

Tensor MiningDistances(const Tensor& embeddings, MiningDistance kind) {
    const auto& shape = embeddings.Shape();
    if (shape.size() != 2 || shape[0] == 0) {
        throw std::invalid_argument("metric mining needs [N, D] embeddings");
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    return OnDevice("MiningDistances", [&]() {
        const af::array x = embeddings.GetArrayRowMajor2D().as(DeviceHasDouble() ? f64 : f32);
        const dim_t n = x.dims(0);
        const unsigned rows = static_cast<unsigned>(n);
        af::array distances;
        if (kind == MiningDistance::Euclidean) {
            distances = af::constant(0.0, n, n, x.type());
            for (dim_t f = 0; f < x.dims(1); ++f) {
                const af::array column = x(af::span, static_cast<int>(f));
                const af::array diff = af::tile(column, 1, rows) - af::tile(column.T(), rows, 1);
                distances += diff * diff;
                distances.eval();  // one small kernel per feature
            }
            distances = af::sqrt(distances);
        } else {
            // As the backend Cosine Embedding loss.
            constexpr double kCosineEpsilon = 1.0e-12;
            const af::array dot = af::matmul(x, x, AF_MAT_NONE, AF_MAT_TRANS);
            const af::array squares = af::sum(x * x, 1) + kCosineEpsilon;
            distances = 1.0 - dot / af::sqrt(af::tile(squares, 1, rows) * af::tile(squares.T(), rows, 1));
        }
        distances = af::select(af::identity(n, n, b8), 0.0, distances);
        distances.eval();
        return Tensor::FromSemanticArray(distances, {shape[0], shape[0]});
    });
#else
    (void)kind;
    throw std::runtime_error("Metric mining runs on ArrayFire, and this build has no ArrayFire");
#endif
}

MinedTriplets MineTriplets(const Tensor& class_ids, const Tensor& distances, MetricMining mining) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    return OnDevice("MineTriplets", [&]() {
        const RowMasks m = Masks(class_ids, distances);
        const af::array has_negative = af::flat(af::anyTrue(m.negative, 1));
        MinedTriplets out;
        if (mining == MetricMining::Hard) {
            const af::array valid = af::flat(af::anyTrue(m.positive, 1)) && has_negative;
            const af::array rows = af::where(valid);
            out.count = static_cast<size_t>(rows.elements());
            if (out.count == 0) return out;
            out.anchors = IndexTensor(rows);
            out.positives = IndexTensor(FirstExtremeColumn(m.distances, m.positive, true)(rows));
            out.negatives = IndexTensor(FirstExtremeColumn(m.distances, m.negative, false)(rows));
            return out;
        }
        // Semi-hard: every (anchor, positive) pair against every negative, an
        // anchor block at a time so the [N, N, block] cube stays bounded.
        const dim_t n = m.n;
        const unsigned nu = static_cast<unsigned>(n);
        const af::array farthest_negative = FirstExtremeColumn(m.distances, m.negative, true);
        const dim_t block = (std::max)(dim_t{1}, static_cast<dim_t>((1 << 24) / (n * n)));
        af::array anchors, positives, negatives;
        for (dim_t start = 0; start < n; start += block) {
            const dim_t end = (std::min)(n, start + block);
            const dim_t b = end - start;
            const af::seq span(static_cast<double>(start), static_cast<double>(end - 1));
            const af::array d = m.distances(span, af::span);      // [b, N]: [a, j]
            const af::array neg = m.negative(span, af::span);
            const af::array pos = m.positive(span, af::span);
            // Cube [j, p, a]: d(a, j), d(a, p), negative(a, j).
            const af::array d_aj = af::tile(af::moddims(d.T(), n, 1, b), 1, nu, 1);
            const af::array d_ap = af::tile(af::moddims(d.T(), 1, n, b), nu, 1, 1);
            const af::array neg_aj = af::tile(af::moddims(neg.T(), n, 1, b), 1, nu, 1);
            const af::array farther = neg_aj && (d_aj > d_ap);
            const af::array values = af::select(farther, d_aj, kInfinity);
            const af::array closest = af::tile((af::min)(values, 0), nu, 1, 1);
            const af::array j_index = af::range(farther.dims(), 0, s32);
            af::array semi = af::moddims(
                (af::min)(af::select(farther && (values == closest), j_index, static_cast<double>(n)), 0), n, b);
            const af::array fallback = af::tile(af::moddims(farthest_negative(span), 1, b), nu, 1);
            semi = af::select(semi >= static_cast<double>(n), fallback, semi);  // [p, a]
            const af::array candidates = pos.T() && af::tile(af::moddims(has_negative(span), 1, b), nu, 1);
            const af::array k = af::where(af::flat(candidates)).as(s64);   // p + N * a_local, anchor order
            if (k.elements() == 0) continue;
            const af::array p = af::mod(k, static_cast<double>(n));
            const af::array a = (k - p) / static_cast<double>(n) + static_cast<double>(start);
            const af::array chosen = af::flat(semi)(k);
            anchors = anchors.isempty() ? a : af::join(0, anchors, a);
            positives = positives.isempty() ? p : af::join(0, positives, p);
            negatives = negatives.isempty() ? chosen : af::join(0, negatives, chosen);
        }
        if (anchors.isempty()) return out;
        out.count = static_cast<size_t>(anchors.elements());
        out.anchors = IndexTensor(anchors);
        out.positives = IndexTensor(positives);
        out.negatives = IndexTensor(negatives);
        return out;
    });
#else
    (void)class_ids;
    (void)distances;
    (void)mining;
    throw std::runtime_error("Metric mining runs on ArrayFire, and this build has no ArrayFire");
#endif
}

MinedPairs MinePairs(const Tensor& class_ids, const Tensor& distances) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    return OnDevice("MinePairs", [&]() {
        const RowMasks m = Masks(class_ids, distances);
        const dim_t n = m.n;
        // [2, N]: per row, the positive pair then the negative pair.
        const af::array present = af::join(0, af::moddims(af::anyTrue(m.positive, 1), 1, n),
                                           af::moddims(af::anyTrue(m.negative, 1), 1, n));
        const af::array partners = af::join(0, af::moddims(FirstExtremeColumn(m.distances, m.positive, true), 1, n),
                                            af::moddims(FirstExtremeColumn(m.distances, m.negative, false), 1, n));
        const af::array k = af::where(af::flat(present)).as(s64);  // type + 2 * row, row order
        MinedPairs out;
        out.count = static_cast<size_t>(k.elements());
        if (out.count == 0) return out;
        const af::array type = af::mod(k, 2.0);
        out.firsts = IndexTensor((k - type) / 2.0);
        out.seconds = IndexTensor(af::flat(partners)(k));
        af::array similar = (type == 0.0).as(f32);
        similar.eval();
        out.similar = Tensor::FromRowMajorFlatArray(similar, {out.count});
        return out;
    });
#else
    (void)class_ids;
    (void)distances;
    throw std::runtime_error("Metric mining runs on ArrayFire, and this build has no ArrayFire");
#endif
}

Tensor GatherRows(const Tensor& matrix, const Tensor& rows) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    return OnDevice("GatherRows", [&]() {
        af::array gathered = af::lookup(matrix.GetArrayRowMajor2D(), af::flat(rows.GetRowMajorFlatArray()), 0);
        gathered.eval();
        return Tensor::FromSemanticArray(gathered, {rows.NumElements(), matrix.Shape()[1]});
    });
#else
    (void)matrix;
    (void)rows;
    throw std::runtime_error("Metric mining runs on ArrayFire, and this build has no ArrayFire");
#endif
}

Tensor ScatterAddRows(const Tensor& stacked, const Tensor& source_rows, size_t rows) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    return OnDevice("ScatterAddRows", [&]() {
        const af::array source = af::moddims(af::flat(source_rows.GetRowMajorFlatArray()), 1,
                                             static_cast<dim_t>(source_rows.NumElements()));
        // selection[row, j] = 1 when stacked row j came from `row`.
        const af::array selection =
            (af::range(af::dim4(static_cast<dim_t>(rows), source.dims(1)), 0, s32) ==
             af::tile(source, static_cast<unsigned>(rows), 1)).as(f32);
        af::array summed = af::matmul(selection, stacked.GetArrayRowMajor2D().as(f32));
        summed.eval();
        return Tensor::FromSemanticArray(summed, {rows, stacked.Shape()[1]});
    });
#else
    (void)stacked;
    (void)source_rows;
    (void)rows;
    throw std::runtime_error("Metric mining runs on ArrayFire, and this build has no ArrayFire");
#endif
}

Tensor DistancesAt(const Tensor& distances, const Tensor& rows, const Tensor& columns) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    return OnDevice("DistancesAt", [&]() {
        const af::array d = distances.GetArrayRowMajor2D();
        const af::array r = af::flat(rows.GetRowMajorFlatArray()).as(s64);
        const af::array c = af::flat(columns.GetRowMajorFlatArray()).as(s64);
        // Column-major storage: element (r, c) sits at r + N * c.
        af::array values = af::flat(d)(r + c * static_cast<double>(d.dims(0)));
        values.eval();
        return Tensor::FromRowMajorFlatArray(values, {rows.NumElements()});
    });
#else
    (void)distances;
    (void)rows;
    (void)columns;
    throw std::runtime_error("Metric mining runs on ArrayFire, and this build has no ArrayFire");
#endif
}

}  // namespace cyxwiz
