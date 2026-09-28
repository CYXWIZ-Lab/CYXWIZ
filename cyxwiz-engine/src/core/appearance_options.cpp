#include "appearance_options.h"

#include <cmath>
#include <cstdlib>

namespace cyxwiz::appearance {

int UiTextIndex(int pixels) {
    int best = 1;
    int best_distance = 1 << 30;
    for (int i = 0; i < static_cast<int>(kUiTextSizes.size()); ++i) {
        const int distance = std::abs(kUiTextSizes[i].pixels - pixels);
        if (distance < best_distance) {
            best = i;
            best_distance = distance;
        }
    }
    return best;
}

int NormalizeUiTextPx(int pixels) { return kUiTextSizes[UiTextIndex(pixels)].pixels; }

int CodeScaleIndex(float scale) {
    if (!std::isfinite(scale)) return 1;
    int best = 1;
    float best_distance = 1e9f;
    for (int i = 0; i < static_cast<int>(kCodeTextSizes.size()); ++i) {
        const float distance = std::fabs(kCodeTextSizes[i].scale - scale);
        if (distance < best_distance) {
            best = i;
            best_distance = distance;
        }
    }
    return best;
}

float NormalizeCodeScale(float scale) { return kCodeTextSizes[CodeScaleIndex(scale)].scale; }

int NormalizeThemePreset(int preset, int count) {
    return preset >= 0 && preset < count ? preset : 0;
}

float Luminance(float r, float g, float b) {
    const auto linear = [](float c) {
        return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
    };
    return 0.2126f * linear(r) + 0.7152f * linear(g) + 0.0722f * linear(b);
}

bool IsLightBackground(float r, float g, float b) { return Luminance(r, g, b) > 0.5f; }

}  // namespace cyxwiz::appearance
