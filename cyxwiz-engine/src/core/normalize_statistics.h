#pragma once

// Normalize's statistics per image channel (TOFIX178 1.4): the stored form
// "0.5282,0.4924,0.4528", the ImageNet values and the channels' names, shared
// by the compiler, Properties and Data Studio's Profile. No ImGui.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace cyxwiz {

// torchvision's ImageNet statistics (RGB).
inline const std::vector<float>& ImageNetChannelMean() {
    static const std::vector<float> values = {0.485f, 0.456f, 0.406f};
    return values;
}
inline const std::vector<float>& ImageNetChannelStd() {
    static const std::vector<float> values = {0.229f, 0.224f, 0.225f};
    return values;
}

// "0.5282, 0.4924, 0.4528" -> the numbers; false when one is not a number
// (or there are none).
inline bool ParseChannelValues(const std::string& text, std::vector<float>& out) {
    std::string spaced = text;
    std::replace(spaced.begin(), spaced.end(), ',', ' ');
    std::istringstream in(spaced);
    out.clear();
    for (float v; in >> v;) out.push_back(v);
    return in.eof() && !out.empty() && std::all_of(out.begin(), out.end(), [](float v) { return std::isfinite(v); });
}

// The stored form: "%.6g" each, comma-separated.
inline std::string FormatChannelValues(const std::vector<float>& values) {
    std::string text;
    char one[32];
    for (size_t i = 0; i < values.size(); ++i) {
        std::snprintf(one, sizeof(one), "%.6g", values[i]);
        text += (i ? "," : "") + std::string(one);
    }
    return text;
}

// "per channel, mean 0.528 / 0.492 / 0.453, std 0.235 / 0.227 / 0.230" (the
// compile report and the Data Input dialog); `what` names the statistics.
inline std::string ChannelSummary(const std::string& what, const std::vector<float>& mean,
                                  const std::vector<float>& std_dev) {
    const auto list = [](const std::vector<float>& values) {
        std::string text;
        char one[32];
        for (size_t i = 0; i < values.size(); ++i) {
            std::snprintf(one, sizeof(one), "%.3f", values[i]);
            text += (i ? " / " : "") + std::string(one);
        }
        return text;
    };
    return what + ", mean " + list(mean) + ", std " + list(std_dev);
}

// R, G, B for three channels, grey for one.
inline std::string ChannelName(size_t index, size_t channels) {
    if (channels == 3 && index < 3) return std::string(1, "RGB"[index]);
    if (channels == 1) return "grey";
    return "channel " + std::to_string(index + 1);
}

}  // namespace cyxwiz
