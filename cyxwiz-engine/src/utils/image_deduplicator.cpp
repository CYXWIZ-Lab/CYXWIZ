#include "image_deduplicator.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <bitset>
#include <unordered_map>

namespace cyxwiz {

// ============================================================================
// Public API
// ============================================================================

uint64_t ImageDeduplicator::ComputeHash(
    const std::vector<float>& image_data,
    int width, int height, int channels,
    HashType type
) {
    if (image_data.empty() || width <= 0 || height <= 0 || channels <= 0) {
        spdlog::error("ImageDeduplicator::ComputeHash: Invalid input parameters");
        return 0;
    }

    size_t expected_size = static_cast<size_t>(width) * height * channels;
    if (image_data.size() != expected_size) {
        spdlog::error("ImageDeduplicator::ComputeHash: Image data size mismatch (expected {}, got {})",
                      expected_size, image_data.size());
        return 0;
    }

    try {
        switch (type) {
            case HashType::pHash:
                return ComputePHash(image_data, width, height, channels);
            case HashType::aHash:
                return ComputeAHash(image_data, width, height, channels);
            case HashType::dHash:
                return ComputeDHash(image_data, width, height, channels);
            default:
                spdlog::error("ImageDeduplicator::ComputeHash: Unknown hash type");
                return 0;
        }
    } catch (const cv::Exception& e) {
        spdlog::error("ImageDeduplicator::ComputeHash: OpenCV exception: {}", e.what());
        return 0;
    } catch (const std::exception& e) {
        spdlog::error("ImageDeduplicator::ComputeHash: Exception: {}", e.what());
        return 0;
    }
}

std::vector<ImageHash> ImageDeduplicator::ComputeHashes(
    const std::vector<std::vector<float>>& images,
    int width, int height, int channels,
    HashType type,
    std::function<void(float progress, const std::string& msg)> progress_callback
) {
    std::vector<ImageHash> hashes;
    hashes.reserve(images.size());

    if (progress_callback) {
        progress_callback(0.0f, "Computing perceptual hashes...");
    }

    for (size_t i = 0; i < images.size(); ++i) {
        ImageHash hash;
        hash.hash_value = ComputeHash(images[i], width, height, channels, type);
        hash.type = type;
        hash.sample_index = i;
        hashes.push_back(hash);

        if (progress_callback && (i % 10 == 0 || i == images.size() - 1)) {
            float progress = static_cast<float>(i + 1) / images.size();
            std::string msg = std::format("Hashed {}/{} images", i + 1, images.size());
            progress_callback(progress, msg);
        }
    }

    if (progress_callback) {
        progress_callback(1.0f, "Hash computation complete!");
    }

    return hashes;
}

std::vector<DuplicateGroup> ImageDeduplicator::FindDuplicates(
    const std::vector<ImageHash>& hashes,
    int hamming_threshold
) {
    std::vector<DuplicateGroup> groups;

    if (hashes.empty()) {
        return groups;
    }

    // Track which images have been grouped
    std::vector<bool> grouped(hashes.size(), false);

    for (size_t i = 0; i < hashes.size(); ++i) {
        if (grouped[i]) continue;

        DuplicateGroup group;
        group.indices.push_back(hashes[i].sample_index);
        grouped[i] = true;

        // Find all images similar to this one
        for (size_t j = i + 1; j < hashes.size(); ++j) {
            if (grouped[j]) continue;

            int distance = HammingDistance(hashes[i].hash_value, hashes[j].hash_value);
            if (distance <= hamming_threshold) {
                group.indices.push_back(hashes[j].sample_index);
                grouped[j] = true;

                // Update similarity (average of all pairs)
                float similarity = ComputeSimilarity(hashes[i].hash_value, hashes[j].hash_value);
                group.similarity = (group.similarity * (group.indices.size() - 2) + similarity) / (group.indices.size() - 1);

                if (distance == 0) {
                    group.exact_duplicate = true;
                }
            }
        }

        // Only add groups with 2+ images
        if (group.indices.size() >= 2) {
            if (group.similarity == 0.0f) {
                // Calculate average similarity for the group
                float total_similarity = 0.0f;
                int pair_count = 0;
                for (size_t a = 0; a < group.indices.size(); ++a) {
                    for (size_t b = a + 1; b < group.indices.size(); ++b) {
                        size_t idx_a = 0, idx_b = 0;
                        for (size_t k = 0; k < hashes.size(); ++k) {
                            if (hashes[k].sample_index == group.indices[a]) idx_a = k;
                            if (hashes[k].sample_index == group.indices[b]) idx_b = k;
                        }
                        total_similarity += ComputeSimilarity(hashes[idx_a].hash_value, hashes[idx_b].hash_value);
                        pair_count++;
                    }
                }
                group.similarity = total_similarity / pair_count;
            }
            groups.push_back(group);
        }
    }

    return groups;
}

int ImageDeduplicator::HammingDistance(uint64_t hash1, uint64_t hash2) {
    // XOR to find differing bits
    uint64_t diff = hash1 ^ hash2;

    // Count set bits (popcount)
    return std::bitset<64>(diff).count();
}

float ImageDeduplicator::ComputeSimilarity(uint64_t hash1, uint64_t hash2) {
    int distance = HammingDistance(hash1, hash2);
    return 1.0f - (static_cast<float>(distance) / 64.0f);
}

std::string ImageDeduplicator::HashTypeToString(HashType type) {
    switch (type) {
        case HashType::pHash: return "pHash (DCT)";
        case HashType::aHash: return "aHash (Average)";
        case HashType::dHash: return "dHash (Difference)";
        default: return "Unknown";
    }
}

// ============================================================================
// Perceptual Hashing Algorithms
// ============================================================================

uint64_t ImageDeduplicator::ComputePHash(
    const std::vector<float>& image,
    int width, int height, int channels
) {
    // Convert to OpenCV Mat
    cv::Mat img;
    if (channels == 1) {
        img = cv::Mat(height, width, CV_32FC1, const_cast<float*>(image.data())).clone();
    } else {
        cv::Mat color_img = cv::Mat(height, width, CV_32FC3, const_cast<float*>(image.data())).clone();
        cv::cvtColor(color_img, img, cv::COLOR_RGB2GRAY);
    }

    // Convert to 8-bit
    img.convertTo(img, CV_8UC1, 255.0);

    // 1. Resize to 32x32
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(32, 32), 0, 0, cv::INTER_LINEAR);

    // 2. Convert to float for DCT
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32F);

    // 3. Compute DCT
    cv::Mat dct_img;
    cv::dct(float_img, dct_img);

    // 4. Extract top-left 8x8 (low frequencies)
    cv::Mat low_freq = dct_img(cv::Rect(0, 0, 8, 8));

    // 5. Compute median of the 8x8 block
    std::vector<float> values;
    values.reserve(64);
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            values.push_back(low_freq.at<float>(i, j));
        }
    }
    std::sort(values.begin(), values.end());
    float median = values[32]; // Middle value for 64 elements

    // 6. Create 64-bit hash (1 if > median, 0 otherwise)
    uint64_t hash = 0;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            if (low_freq.at<float>(i, j) > median) {
                hash |= (1ULL << (i * 8 + j));
            }
        }
    }

    return hash;
}

uint64_t ImageDeduplicator::ComputeAHash(
    const std::vector<float>& image,
    int width, int height, int channels
) {
    // Convert to OpenCV Mat
    cv::Mat img;
    if (channels == 1) {
        img = cv::Mat(height, width, CV_32FC1, const_cast<float*>(image.data())).clone();
    } else {
        cv::Mat color_img = cv::Mat(height, width, CV_32FC3, const_cast<float*>(image.data())).clone();
        cv::cvtColor(color_img, img, cv::COLOR_RGB2GRAY);
    }

    // Convert to 8-bit
    img.convertTo(img, CV_8UC1, 255.0);

    // 1. Resize to 8x8
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(8, 8), 0, 0, cv::INTER_LINEAR);

    // 2. Compute mean pixel value
    cv::Scalar mean_scalar = cv::mean(resized);
    float mean = static_cast<float>(mean_scalar[0]);

    // 3. Create 64-bit hash (1 if > mean, 0 otherwise)
    uint64_t hash = 0;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            if (resized.at<uint8_t>(i, j) > mean) {
                hash |= (1ULL << (i * 8 + j));
            }
        }
    }

    return hash;
}

uint64_t ImageDeduplicator::ComputeDHash(
    const std::vector<float>& image,
    int width, int height, int channels
) {
    // Convert to OpenCV Mat
    cv::Mat img;
    if (channels == 1) {
        img = cv::Mat(height, width, CV_32FC1, const_cast<float*>(image.data())).clone();
    } else {
        cv::Mat color_img = cv::Mat(height, width, CV_32FC3, const_cast<float*>(image.data())).clone();
        cv::cvtColor(color_img, img, cv::COLOR_RGB2GRAY);
    }

    // Convert to 8-bit
    img.convertTo(img, CV_8UC1, 255.0);

    // 1. Resize to 9x8 (one extra column for gradient)
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(9, 8), 0, 0, cv::INTER_LINEAR);

    // 2. Create 64-bit hash based on horizontal gradients
    // For each row, compare adjacent pixels (8 comparisons per row)
    uint64_t hash = 0;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            // If left pixel > right pixel, set bit to 1
            if (resized.at<uint8_t>(i, j) > resized.at<uint8_t>(i, j + 1)) {
                hash |= (1ULL << (i * 8 + j));
            }
        }
    }

    return hash;
}

} // namespace cyxwiz
