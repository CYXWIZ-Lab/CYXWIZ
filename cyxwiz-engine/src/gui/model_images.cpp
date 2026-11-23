#include "model_images.h"

#include "panels/output_renderer.h"
#include "ui_tokens.h"

#include <algorithm>

namespace cyxwiz::model_images {

ImVec4 Mix(const ImVec4& a, const ImVec4& b, float t) {
    return ImVec4(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, 1.0f);
}

ImVec4 Sequential(float v) {
    const auto& stops = ui::CurrentTokens().scale_sequential;
    const float x = std::clamp(v, 0.0f, 1.0f) * 3.0f;
    const int i = std::min(2, static_cast<int>(x));
    return Mix(stops[i], stops[i + 1], x - static_cast<float>(i));
}

ImVec4 Diverging(float v) {
    const auto& stops = ui::CurrentTokens().scale_diverging;
    const float x = std::clamp(v, -1.0f, 1.0f);
    return x < 0.0f ? Mix(stops[1], stops[0], -x) : Mix(stops[1], stops[2], x);
}

Rgba Picture(size_t height, size_t width, int scale, const std::function<ImVec4(size_t)>& colour) {
    Rgba out;
    out.width = static_cast<int>(width) * scale;
    out.height = static_cast<int>(height) * scale;
    out.pixels.resize(static_cast<size_t>(out.width) * out.height * 4);
    for (int y = 0; y < out.height; ++y) {
        for (int x = 0; x < out.width; ++x) {
            const ImVec4 c = colour(static_cast<size_t>(y / scale) * width + static_cast<size_t>(x / scale));
            unsigned char* p = &out.pixels[(static_cast<size_t>(y) * out.width + x) * 4];
            p[0] = static_cast<unsigned char>(std::clamp(c.x, 0.0f, 1.0f) * 255.0f + 0.5f);
            p[1] = static_cast<unsigned char>(std::clamp(c.y, 0.0f, 1.0f) * 255.0f + 0.5f);
            p[2] = static_cast<unsigned char>(std::clamp(c.z, 0.0f, 1.0f) * 255.0f + 0.5f);
            p[3] = 255;
        }
    }
    return out;
}

std::function<ImVec4(size_t)> InputPixels(const std::vector<float>& values, size_t channels) {
    const size_t c = std::max<size_t>(1, channels);
    float lo = 0.0f, hi = 1.0f;
    if (!values.empty()) {
        const auto [lo_it, hi_it] = std::minmax_element(values.begin(), values.end());
        lo = *lo_it;
        hi = *hi_it;
    }
    const float span = std::max(1e-12f, hi - lo);
    return [&values, c, lo, span](size_t i) {
        if (c >= 3)
            return ImVec4((values[i * c] - lo) / span, (values[i * c + 1] - lo) / span, (values[i * c + 2] - lo) / span, 1.0f);
        const float g = (values[i * c] - lo) / span;
        return ImVec4(g, g, g, 1.0f);
    };
}

Rgba InputPicture(const std::vector<float>& values, size_t height, size_t width, size_t channels, int side) {
    if (height == 0 || width == 0 || values.size() < height * width * std::max<size_t>(1, channels)) return {};
    const int scale = std::max(1, side / static_cast<int>(std::max(height, width)));
    return Picture(height, width, scale, InputPixels(values, channels));
}

unsigned int Upload(const Rgba& picture) {
    if (picture.pixels.empty()) return 0;
    return OutputRenderer::CreateTextureFromRGBA(picture.pixels.data(), picture.width, picture.height);
}

void ReleaseTexture(unsigned int& texture) {
    if (texture) OutputRenderer::DeleteTexture(texture);
    texture = 0;
}

}  // namespace cyxwiz::model_images
