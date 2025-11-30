#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <string>

namespace cyxwiz {

/**
 * TextureManager - Manages OpenGL textures for image preview
 *
 * Handles creation, caching, and cleanup of textures used for
 * displaying dataset samples in ImGui.
 */
class TextureManager {
public:
    // Singleton access
    static TextureManager& Instance();

    // Prevent copying
    TextureManager(const TextureManager&) = delete;
    TextureManager& operator=(const TextureManager&) = delete;

    /**
     * Create a texture from float image data
     * @param data Float pixel data (normalized 0-1)
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels (1=grayscale, 3=RGB, 4=RGBA)
     * @return OpenGL texture ID (cast to ImTextureID)
     */
    uint32_t CreateTextureFromFloatData(const float* data, int width, int height, int channels);

    /**
     * Create a texture from uint8 image data
     * @param data Uint8 pixel data (0-255)
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels (1=grayscale, 3=RGB, 4=RGBA)
     * @return OpenGL texture ID
     */
    uint32_t CreateTextureFromUint8Data(const unsigned char* data, int width, int height, int channels);

    /**
     * Load texture from file
     * @param filepath Path to image file
     * @param out_width Output width
     * @param out_height Output height
     * @return OpenGL texture ID, 0 on failure
     */
    uint32_t LoadTextureFromFile(const std::string& filepath, int* out_width = nullptr, int* out_height = nullptr);

    /**
     * Update an existing texture with new data
     * @param texture_id Existing texture ID
     * @param data Float pixel data
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels
     * @return true if successful
     */
    bool UpdateTexture(uint32_t texture_id, const float* data, int width, int height, int channels);

    /**
     * Delete a specific texture
     * @param texture_id Texture to delete
     */
    void DeleteTexture(uint32_t texture_id);

    /**
     * Delete all textures
     */
    void DeleteAllTextures();

    /**
     * Get a cached texture for a dataset sample
     * Creates it if not exists
     * @param cache_key Unique key for this sample
     * @param data Float pixel data
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels
     * @return Texture ID
     */
    uint32_t GetOrCreateCachedTexture(const std::string& cache_key,
                                       const float* data, int width, int height, int channels);

    /**
     * Clear texture cache
     */
    void ClearCache();

    /**
     * Get memory usage estimate
     * @return Bytes used by textures
     */
    size_t GetMemoryUsage() const;

    /**
     * Set maximum cache size
     * @param max_textures Maximum number of cached textures
     */
    void SetMaxCacheSize(size_t max_textures) { max_cache_size_ = max_textures; }

private:
    TextureManager() = default;
    ~TextureManager();

    // Evict oldest textures when cache is full
    void EvictOldest();

    // Cache for dataset preview textures
    std::unordered_map<std::string, uint32_t> texture_cache_;
    std::vector<std::string> cache_order_;  // LRU order

    // All managed textures
    std::vector<uint32_t> all_textures_;

    // Settings
    size_t max_cache_size_ = 100;
    size_t memory_usage_ = 0;
};

/**
 * Helper function to render an image preview with ImGui
 * @param data Float pixel data (normalized 0-1)
 * @param width Image width
 * @param height Image height
 * @param channels Number of channels
 * @param display_width Display width in pixels (0 = auto)
 * @param display_height Display height in pixels (0 = auto)
 */
void RenderImageWithTexture(const float* data, int width, int height, int channels,
                            float display_width = 0.0f, float display_height = 0.0f);

/**
 * Render an image grid (multiple images in a row)
 * @param images Vector of image data
 * @param width Width of each image
 * @param height Height of each image
 * @param channels Number of channels
 * @param images_per_row Number of images per row
 * @param display_size Display size for each image
 */
void RenderImageGrid(const std::vector<std::vector<float>>& images,
                     int width, int height, int channels,
                     int images_per_row = 8, float display_size = 64.0f);

} // namespace cyxwiz
