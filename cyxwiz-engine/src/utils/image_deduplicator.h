#pragma once

#include <vector>
#include <string>
#include <functional>
#include <cstdint>

namespace cyxwiz {

/**
 * Perceptual hashing algorithm types
 */
enum class HashType {
    pHash,  // Perceptual Hash (DCT-based, most robust)
    aHash,  // Average Hash (mean-based, fastest)
    dHash   // Difference Hash (gradient-based)
};

/**
 * Hash result for a single image
 */
struct ImageHash {
    uint64_t hash_value = 0;
    HashType type = HashType::pHash;
    size_t sample_index = 0;  // Index in dataset
};

/**
 * Group of duplicate/near-duplicate images
 */
struct DuplicateGroup {
    std::vector<size_t> indices;  // Sample indices in dataset
    float similarity = 0.0f;      // 0-1 (1 = identical)
    bool exact_duplicate = false; // Hamming distance = 0
};

/**
 * Image duplicate detector using perceptual hashing
 */
class ImageDeduplicator {
public:
    /**
     * Compute perceptual hash for a single image
     * @param image_data Float image data (0-1 range) [HWC format]
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels (1=grayscale, 3=RGB)
     * @param type Hash algorithm to use
     * @return 64-bit perceptual hash
     */
    static uint64_t ComputeHash(
        const std::vector<float>& image_data,
        int width, int height, int channels,
        HashType type = HashType::pHash
    );

    /**
     * Compute hashes for a batch of images
     * @param images Vector of image data
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels
     * @param type Hash algorithm to use
     * @param progress_callback Optional progress callback (progress, message)
     * @return Vector of image hashes
     */
    static std::vector<ImageHash> ComputeHashes(
        const std::vector<std::vector<float>>& images,
        int width, int height, int channels,
        HashType type = HashType::pHash,
        std::function<void(float progress, const std::string& msg)> progress_callback = nullptr
    );

    /**
     * Find duplicate groups based on Hamming distance
     * @param hashes Vector of image hashes
     * @param hamming_threshold Maximum Hamming distance (0-64 bits)
     *                          0 = exact duplicates only
     *                          5 = near-duplicates (recommended)
     *                          10 = very similar images
     * @return Vector of duplicate groups
     */
    static std::vector<DuplicateGroup> FindDuplicates(
        const std::vector<ImageHash>& hashes,
        int hamming_threshold = 5
    );

    /**
     * Compute Hamming distance between two hashes
     * @param hash1 First hash
     * @param hash2 Second hash
     * @return Number of differing bits (0-64)
     */
    static int HammingDistance(uint64_t hash1, uint64_t hash2);

    /**
     * Compute similarity score from Hamming distance
     * @param hash1 First hash
     * @param hash2 Second hash
     * @return Similarity score (0-1, 1 = identical)
     */
    static float ComputeSimilarity(uint64_t hash1, uint64_t hash2);

    /**
     * Convert hash type to string
     */
    static std::string HashTypeToString(HashType type);

private:
    // Perceptual Hash (DCT-based, most robust to minor changes)
    static uint64_t ComputePHash(const std::vector<float>& image, int width, int height, int channels);

    // Average Hash (mean-based, fastest but less robust)
    static uint64_t ComputeAHash(const std::vector<float>& image, int width, int height, int channels);

    // Difference Hash (gradient-based, good for detecting crops)
    static uint64_t ComputeDHash(const std::vector<float>& image, int width, int height, int channels);
};

} // namespace cyxwiz
