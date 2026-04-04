#include "texture_manager.h"
#include <glad/glad.h>
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

// stb_image.h is already included with STB_IMAGE_IMPLEMENTATION in stb_image_impl.cpp
#include <stb_image.h>

namespace cyxwiz {

TextureManager& TextureManager::Instance() {
    static TextureManager instance;
    return instance;
}

TextureManager::~TextureManager() {
    // Do nothing - cleanup is done in CyxWizApp::Shutdown() before OpenGL context is destroyed.
    // This destructor runs during static destruction when the context is already gone.
}

uint32_t TextureManager::CreateTextureFromFloatData(const float* data, int width, int height, int channels) {
    if (!data || width <= 0 || height <= 0 || channels <= 0) {
        return 0;
    }

    // Convert float data (0-1) to uint8 (0-255)
    std::vector<unsigned char> pixels(width * height * channels);
    for (int i = 0; i < width * height * channels; ++i) {
        float val = std::clamp(data[i], 0.0f, 1.0f);
        pixels[i] = static_cast<unsigned char>(val * 255.0f);
    }

    return CreateTextureFromUint8Data(pixels.data(), width, height, channels);
}

uint32_t TextureManager::CreateTextureFromUint8Data(const unsigned char* data, int width, int height, int channels) {
    if (!data || width <= 0 || height <= 0 || channels <= 0) {
        return 0;
    }

    GLuint texture_id;
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);

    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);  // Nearest for pixel-perfect preview
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Determine format based on channels
    GLenum format;
    GLenum internal_format;
    switch (channels) {
        case 1:
            format = GL_RED;
            internal_format = GL_R8;
            break;
        case 2:
            format = GL_RG;
            internal_format = GL_RG8;
            break;
        case 3:
            format = GL_RGB;
            internal_format = GL_RGB8;
            break;
        case 4:
        default:
            format = GL_RGBA;
            internal_format = GL_RGBA8;
            break;
    }

    // For grayscale images, we need to set up swizzle to make it display correctly
    if (channels == 1) {
        GLint swizzle_mask[] = {GL_RED, GL_RED, GL_RED, GL_ONE};
        glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzle_mask);
    }

    // Upload texture data
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, format, GL_UNSIGNED_BYTE, data);

    glBindTexture(GL_TEXTURE_2D, 0);

    // Track texture - O(1) insert
    all_textures_.insert(texture_id);
    memory_usage_ += width * height * channels;

    return texture_id;
}

uint32_t TextureManager::LoadTextureFromFile(const std::string& filepath, int* out_width, int* out_height) {
    int width, height, channels;
    unsigned char* data = stbi_load(filepath.c_str(), &width, &height, &channels, 0);

    if (!data) {
        spdlog::warn("Failed to load image: {}", filepath);
        return 0;
    }

    uint32_t texture_id = CreateTextureFromUint8Data(data, width, height, channels);

    if (out_width) *out_width = width;
    if (out_height) *out_height = height;

    stbi_image_free(data);
    return texture_id;
}

bool TextureManager::UpdateTexture(uint32_t texture_id, const float* data, int width, int height, int channels) {
    if (texture_id == 0 || !data) return false;

    // Convert float to uint8
    std::vector<unsigned char> pixels(width * height * channels);
    for (int i = 0; i < width * height * channels; ++i) {
        float val = std::clamp(data[i], 0.0f, 1.0f);
        pixels[i] = static_cast<unsigned char>(val * 255.0f);
    }

    glBindTexture(GL_TEXTURE_2D, texture_id);

    GLenum format;
    switch (channels) {
        case 1: format = GL_RED; break;
        case 2: format = GL_RG; break;
        case 3: format = GL_RGB; break;
        default: format = GL_RGBA; break;
    }

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, format, GL_UNSIGNED_BYTE, pixels.data());

    glBindTexture(GL_TEXTURE_2D, 0);
    return true;
}

void TextureManager::DeleteTexture(uint32_t texture_id) {
    if (texture_id == 0) return;

    GLuint id = texture_id;
    glDeleteTextures(1, &id);

    // Remove from tracking - O(1) erase
    all_textures_.erase(texture_id);

    // Remove from cache if present
    for (auto cache_it = texture_cache_.begin(); cache_it != texture_cache_.end(); ) {
        if (cache_it->second == texture_id) {
            cache_it = texture_cache_.erase(cache_it);
        } else {
            ++cache_it;
        }
    }
}

void TextureManager::DeleteAllTextures() {
    for (uint32_t tex_id : all_textures_) {
        GLuint id = tex_id;
        glDeleteTextures(1, &id);
    }
    all_textures_.clear();
    texture_cache_.clear();
    cache_order_.clear();
    memory_usage_ = 0;
}

uint32_t TextureManager::GetOrCreateCachedTexture(const std::string& cache_key,
                                                   const float* data, int width, int height, int channels) {
    // Check if already cached
    auto it = texture_cache_.find(cache_key);
    if (it != texture_cache_.end()) {
        // Move to back of LRU - O(n) but only for cache_order_ which is typically small
        // Could optimize further with std::list + unordered_map if needed
        auto order_it = std::find(cache_order_.begin(), cache_order_.end(), cache_key);
        if (order_it != cache_order_.end()) {
            cache_order_.erase(order_it);
            cache_order_.push_back(cache_key);
        }
        return it->second;
    }

    // Evict if cache is full
    while (texture_cache_.size() >= max_cache_size_) {
        EvictOldest();
    }

    // Create new texture
    uint32_t texture_id = CreateTextureFromFloatData(data, width, height, channels);
    if (texture_id != 0) {
        texture_cache_[cache_key] = texture_id;
        cache_order_.push_back(cache_key);
    }

    return texture_id;
}

void TextureManager::ClearCache() {
    for (const auto& [key, tex_id] : texture_cache_) {
        // Only delete if in general tracking - O(1) find
        if (all_textures_.find(tex_id) != all_textures_.end()) {
            GLuint id = tex_id;
            glDeleteTextures(1, &id);
            all_textures_.erase(tex_id);
        }
    }
    texture_cache_.clear();
    cache_order_.clear();
}

size_t TextureManager::GetMemoryUsage() const {
    return memory_usage_;
}

void TextureManager::EvictOldest() {
    if (cache_order_.empty()) return;

    std::string oldest_key = cache_order_.front();
    cache_order_.pop_front();  // O(1) pop from front using deque

    auto it = texture_cache_.find(oldest_key);
    if (it != texture_cache_.end()) {
        DeleteTexture(it->second);
        texture_cache_.erase(it);
    }
}

// Helper function implementations

void RenderImageWithTexture(const float* data, int width, int height, int channels,
                            float display_width, float display_height) {
    if (!data || width <= 0 || height <= 0) return;

    auto& tm = TextureManager::Instance();

    // Create a unique cache key based on data pointer and dimensions
    // This allows reuse across frames when the same image is displayed
    std::string cache_key = "render_temp_" +
                           std::to_string(reinterpret_cast<uintptr_t>(data)) + "_" +
                           std::to_string(width) + "x" + std::to_string(height) + "x" +
                           std::to_string(channels);

    // Use the existing cache mechanism instead of static variable
    uint32_t texture_id = tm.GetOrCreateCachedTexture(cache_key, data, width, height, channels);

    if (texture_id == 0) {
        ImGui::Text("Failed to create texture");
        return;
    }

    // Calculate display size
    if (display_width <= 0 || display_height <= 0) {
        // Auto-calculate while maintaining aspect ratio
        float scale = 4.0f;  // Scale up small images
        if (width < 64 || height < 64) {
            scale = std::max(4.0f, 128.0f / std::max(width, height));
        }
        display_width = width * scale;
        display_height = height * scale;
    }

    // Render with ImGui
    ImGui::Image((ImTextureID)(intptr_t)texture_id, ImVec2(display_width, display_height));
}

void RenderImageGrid(const std::vector<std::vector<float>>& images,
                     int width, int height, int channels,
                     int images_per_row, float display_size) {
    if (images.empty()) return;

    auto& tm = TextureManager::Instance();

    for (size_t i = 0; i < images.size(); ++i) {
        if (i > 0 && i % images_per_row != 0) {
            ImGui::SameLine();
        }

        // Create unique cache key
        std::string cache_key = "grid_" + std::to_string(i) + "_" +
                                std::to_string(width) + "x" + std::to_string(height);

        uint32_t tex_id = tm.GetOrCreateCachedTexture(cache_key, images[i].data(),
                                                       width, height, channels);
        if (tex_id != 0) {
            ImGui::Image((ImTextureID)(intptr_t)tex_id, ImVec2(display_size, display_size));
        }
    }
}

} // namespace cyxwiz
