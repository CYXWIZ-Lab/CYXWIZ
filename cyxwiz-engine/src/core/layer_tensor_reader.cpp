#include "layer_tensor_reader.h"

#include "algorithms/arrayfire_host_materialization.h"

#include <arrayfire.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace cyxwiz::layer_view {

namespace {

dim_t D(size_t v) { return static_cast<dim_t>(v); }
// An index into one dimension (af::index takes an int).
int I(size_t v) { return static_cast<int>(v); }
// An af::seq bound (af::seq takes doubles).
double S(size_t v) { return static_cast<double>(v); }

std::vector<float> ToHost(const af::array& values, const char* what) {
    std::vector<float> host(static_cast<size_t>(values.elements()));
    if (host.empty()) return host;
    MaterializeArrayFireToHost(values.as(f32), host.data(), ArrayFireHostSyncCategory::DebugSampleDump,
                               "LayerView::Read", "arrayfire_column_major", "bounded_debug_readback", what);
    return host;
}

// A spatial sample [H, W, C] as ArrayFire (C, W * H): column w + h * W is a
// pixel in row-major order, row c a channel.
af::array ChannelsByPixel(const Tensor& spatial_sample) {
    const auto& s = spatial_sample.Shape();
    if (s.size() != 3) throw std::invalid_argument("a spatial sample is [H, W, C]");
    return af::moddims(spatial_sample.GetRowMajorFlatArray(), D(s[2]), D(s[0] * s[1]));
}

}  // namespace

Tensor SampleOf(const Tensor& activation, size_t sample) {
    const auto& s = activation.Shape();
    const af::array flat = activation.GetRowMajorFlatArray();
    if (s.size() == 4) {
        // Row-major [H, W, C, N] is ArrayFire (N, C, W, H): the sample is a
        // row, and (C, W, H) column-major is row-major [H, W, C].
        if (sample >= s[3]) throw std::out_of_range("sample past the batch");
        const af::array all = af::moddims(flat, D(s[3]), D(s[2]), D(s[1]), D(s[0]));
        const af::array one = af::moddims(all(I(sample), af::span, af::span, af::span), D(s[0] * s[1] * s[2]));
        return Tensor::FromRowMajorFlatArray(one, {s[0], s[1], s[2]});
    }
    if (s.size() >= 2) {
        if (sample >= s[0]) throw std::out_of_range("sample past the batch");
        const size_t per = activation.NumElements() / s[0];
        const af::array rows = af::moddims(flat, D(per), D(s[0]));
        return Tensor::FromRowMajorFlatArray(rows(af::span, I(sample)), std::vector<size_t>(s.begin() + 1, s.end()));
    }
    return activation;
}

ChannelSummary ReadChannelSummary(const Tensor& spatial_sample) {
    const af::array x = ChannelsByPixel(spatial_sample);
    const af::array joined = af::join(1, af::mean(x, 1), af::max(x, 1));
    const std::vector<float> host = ToHost(joined, "channel means and maxima");
    ChannelSummary summary;
    const size_t channels = static_cast<size_t>(x.dims(0));
    summary.means.assign(host.begin(), host.begin() + static_cast<std::ptrdiff_t>(channels));
    summary.maxima.assign(host.begin() + static_cast<std::ptrdiff_t>(channels), host.end());
    return summary;
}

std::vector<float> ReadChannelImages(const Tensor& spatial_sample, const std::vector<size_t>& channels) {
    if (channels.empty()) return {};
    const af::array x = ChannelsByPixel(spatial_sample);
    std::vector<unsigned> index(channels.size());
    std::transform(channels.begin(), channels.end(), index.begin(), [](size_t c) { return static_cast<unsigned>(c); });
    const af::array picked = af::lookup(x, af::array(D(index.size()), index.data()), 0);
    // (K, pixels) transposed: each channel's pixels contiguous.
    return ToHost(af::transpose(picked), "shown feature maps");
}

std::vector<float> ReadValues(const Tensor& tensor) {
    return ToHost(tensor.GetRowMajorFlatArray(), "shown values");
}

Histogram ReadHistogram(const Tensor& tensor, float low, float high, size_t bins) {
    Histogram histogram;
    histogram.low = low;
    histogram.high = high;
    histogram.counts.assign(bins, 0);
    if (bins == 0 || tensor.NumElements() == 0) return histogram;
    const af::array values = af::flat(tensor.GetArray());
    const af::array finite = values(af::where(!(af::isNaN(values) || af::isInf(values))));
    if (finite.elements() == 0) return histogram;
    // af::histogram needs a range wider than nothing.
    const double top = high > low ? high : low + 1.0;
    const af::array counts = af::histogram(finite, static_cast<unsigned>(bins), low, top);
    MaterializeArrayFireToHost(counts.as(u32), histogram.counts.data(), ArrayFireHostSyncCategory::DebugSampleDump,
                               "LayerView::Read", "arrayfire_aggregate_vector", "bounded_debug_readback",
                               "value histogram");
    return histogram;
}

std::vector<float> ReadMatrixRows(const TensorInfo& info, const Tensor& tensor, size_t row_begin, size_t row_count) {
    const MatrixLayout layout = LayoutAsMatrix(info);
    if (layout.rows == 0 || layout.columns == 0 || row_begin >= layout.rows) return {};
    row_count = std::min(row_count, layout.rows - row_begin);
    const af::array flat = tensor.GetRowMajorFlatArray();
    const af::seq rows(S(row_begin), S(row_begin + row_count - 1));
    const auto& s = info.shape;
    const bool conv_weights = s.size() == 4 && (info.role == CapturedTensorRole::Parameter ||
                                                info.role == CapturedTensorRole::Gradient);
    if (conv_weights) {
        // Flat index = column * cout + row: ArrayFire (cout, columns); the
        // page's rows transposed so each row is contiguous.
        const af::array m = af::moddims(flat, D(layout.rows), D(layout.columns));
        return ToHost(af::transpose(m(rows, af::span)), "shown heatmap rows");
    }
    // Flat index = row * columns + column: ArrayFire (columns, rows).
    const af::array m = af::moddims(flat, D(layout.columns), D(layout.rows));
    return ToHost(m(af::span, rows), "shown heatmap rows");
}

}  // namespace cyxwiz::layer_view
