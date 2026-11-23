#pragma once

// Pictures of model inputs and maps (TOFIX129 D11): the Model window's
// Explain and Data Studio's Results > Test > Errors draw a test row's input
// and the explanation maps with these, in the theme's scales.

#include <imgui.h>

#include <cstddef>
#include <functional>
#include <vector>

namespace cyxwiz::model_images {

struct Rgba {
    int width = 0, height = 0;
    std::vector<unsigned char> pixels;
};

ImVec4 Mix(const ImVec4& a, const ImVec4& b, float t);
// The theme's sequential scale at 0..1 and diverging scale at -1..1.
ImVec4 Sequential(float v);
ImVec4 Diverging(float v);

// height x width colours, each pixel `scale` times (nearest), RGBA.
Rgba Picture(size_t height, size_t width, int scale, const std::function<ImVec4(size_t)>& colour);

// The colour of pixel i of an [H, W, C] row, scaled by the row's own range
// (grey for one channel, RGB for three or more).
std::function<ImVec4(size_t)> InputPixels(const std::vector<float>& values, size_t channels);

// A model input as a picture about `side` px wide.
Rgba InputPicture(const std::vector<float>& values, size_t height, size_t width, size_t channels, int side);

// An OpenGL texture of a picture (0 when empty); free it with ReleaseTexture.
unsigned int Upload(const Rgba& picture);
void ReleaseTexture(unsigned int& texture);

}  // namespace cyxwiz::model_images
