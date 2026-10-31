#include "image_channel_stats.h"

#include "algorithms/arrayfire_host_materialization.h"

#include <arrayfire.h>

#include <algorithm>
#include <cmath>
#include <set>
#include <string>

namespace cyxwiz {

namespace {
constexpr size_t kChunk = 2048;  // images per upload
}

ImageChannelStats MeasureImageChannelStats(const PreparedImageSet& set,
                                           const std::function<void(size_t, size_t)>& progress,
                                           const std::atomic<bool>* cancel) {
    ImageChannelStats stats;
    const size_t channels = set.Channels(), row = set.RowBytes(), total = set.Count();
    if (channels == 0 || row == 0 || total == 0) return stats;
    const std::set<std::string> unreadable(set.Unreadable().begin(), set.Unreadable().end());

    std::vector<double> sum(channels, 0.0), squares(channels, 0.0);
    std::vector<uint8_t> chunk;
    for (size_t begin = 0; begin < total; begin += kChunk) {
        if (cancel && cancel->load()) return {};
        const size_t end = std::min(total, begin + kChunk);
        chunk.clear();
        for (size_t i = begin; i < end; ++i) {
            if (!unreadable.empty() && unreadable.count(set.Files()[i])) continue;
            chunk.insert(chunk.end(), set.Row(i), set.Row(i) + row);
            ++stats.images;
        }
        if (!chunk.empty()) {
            // [channels, pixels]: rows are [H, W, C] with the channel fastest.
            const dim_t pixels = static_cast<dim_t>(chunk.size() / channels);
            const af::array bytes(static_cast<dim_t>(channels), pixels, chunk.data());
            const af::array x = bytes.as(f32) / 255.0f;
            // f32 on the device (some devices have no f64), doubles on the host.
            const af::array s = af::sum(x, 1);
            const af::array q = af::sum(x * x, 1);
            std::vector<float> hs(channels), hq(channels);
            MaterializeArrayFireToHost(s, hs.data(), ArrayFireHostSyncCategory::MetricScalarReadback,
                                       "ImageChannelStats::Measure", "arrayfire_aggregate_vector",
                                       "bounded_metric_readback", "per-channel pixel sum");
            MaterializeArrayFireToHost(q, hq.data(), ArrayFireHostSyncCategory::MetricScalarReadback,
                                       "ImageChannelStats::Measure", "arrayfire_aggregate_vector",
                                       "bounded_metric_readback", "per-channel pixel sum of squares");
            for (size_t c = 0; c < channels; ++c) {
                sum[c] += hs[c];
                squares[c] += hq[c];
            }
        }
        if (progress) progress(end, total);
    }
    const double n = static_cast<double>(stats.images) * static_cast<double>(set.Height() * set.Width());
    if (n <= 0.0) return stats;
    for (size_t c = 0; c < channels; ++c) {
        const double mean = sum[c] / n;
        stats.mean.push_back(mean);
        stats.std_dev.push_back(std::sqrt(std::max(0.0, squares[c] / n - mean * mean)));
    }
    return stats;
}

}  // namespace cyxwiz
