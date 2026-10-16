#include "plot_scales.h"

#include <algorithm>
#include <cmath>

namespace cyxwiz::plot {

const std::vector<ScaleInfo>& Scales() {
    static const std::vector<ScaleInfo> scales = {
#include "plot_scales.inc"
    };
    return scales;
}

const ScaleInfo* FindScale(const std::string& id) {
    for (const auto& s : Scales())
        if (id == s.id) return &s;
    return nullptr;
}

std::array<float, 3> SampleScale(const ScaleInfo& scale, double t, bool reverse) {
    if (!std::isfinite(t)) t = 0.0;
    t = std::clamp(t, 0.0, 1.0);
    if (reverse) t = 1.0 - t;
    const double f = t * 15.0;
    const int i = std::min(14, static_cast<int>(f));
    const float u = static_cast<float>(f - i);
    std::array<float, 3> out{};
    for (int c = 0; c < 3; ++c)
        out[static_cast<size_t>(c)] = scale.stops[static_cast<size_t>(i)][static_cast<size_t>(c)] * (1.0f - u) +
                                      scale.stops[static_cast<size_t>(i + 1)][static_cast<size_t>(c)] * u;
    return out;
}

}  // namespace cyxwiz::plot
