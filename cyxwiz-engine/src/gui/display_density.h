#pragma once

#include <algorithm>
#include <cmath>

namespace cyxwiz::gui {

inline constexpr float kMinimumFontRasterizerDensity = 1.0f;
inline constexpr float kMaximumFontRasterizerDensity = 4.0f;
inline constexpr float kFontDensityChangeThreshold = 0.05f;

inline float CalculateFramebufferDensity(int window_width, int window_height,
                                         int framebuffer_width,
                                         int framebuffer_height) {
  if (window_width <= 0 || window_height <= 0 || framebuffer_width <= 0 ||
      framebuffer_height <= 0) {
    return kMinimumFontRasterizerDensity;
  }

  const float scale_x = static_cast<float>(framebuffer_width) /
                        static_cast<float>(window_width);
  const float scale_y = static_cast<float>(framebuffer_height) /
                        static_cast<float>(window_height);
  const float density = std::max(scale_x, scale_y);
  if (!std::isfinite(density)) {
    return kMinimumFontRasterizerDensity;
  }

  return std::clamp(density, kMinimumFontRasterizerDensity,
                    kMaximumFontRasterizerDensity);
}

inline bool HasMaterialFontDensityChange(float current_density,
                                         float new_density) {
  return std::abs(current_density - new_density) >=
         kFontDensityChangeThreshold;
}

} // namespace cyxwiz::gui
