#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gui {

/**
 * Image preview for both raw files and dataset samples
 * Supports zoom and pan
 */
class ImagePreviewRenderer {
public:
    ImagePreviewRenderer() = default;

    // Render from raw file
    void RenderFile(const std::string& filepath, float zoom = 1.0f);

    // Render single dataset sample
    void RenderSample(const std::vector<float>& data, int width, int height, int channels,
                       float zoom = 1.0f);

    // Render image info bar
    static void RenderImageInfo(int width, int height, int channels, const float* data = nullptr, size_t data_size = 0);

private:
    // File-based texture cache
    uint32_t file_texture_ = 0;
    std::string file_texture_path_;
    int file_tex_w_ = 0, file_tex_h_ = 0;
};

} // namespace gui
