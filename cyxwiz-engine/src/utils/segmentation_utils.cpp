#include "segmentation_utils.h"
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <cstring>
#include <algorithm>

namespace cyxwiz {

// ============================================================================
// Enum String Conversions
// ============================================================================

std::string GrabCutMaskToString(GrabCutMask mask) {
    switch (mask) {
        case GrabCutMask::Background:     return "background";
        case GrabCutMask::Foreground:     return "foreground";
        case GrabCutMask::ProbBackground: return "prob_background";
        case GrabCutMask::ProbForeground: return "prob_foreground";
        default:                          return "background";
    }
}

std::string GrabCutModeToString(GrabCutMode mode) {
    switch (mode) {
        case GrabCutMode::WithRect: return "with_rect";
        case GrabCutMode::WithMask: return "with_mask";
        default:                    return "with_rect";
    }
}

// ============================================================================
// SegmentationUtils Implementation
// ============================================================================

bool SegmentationUtils::ValidateInput(const std::vector<float>& image_data,
                                       int width, int height, int channels) {
    if (width <= 0 || height <= 0 || channels <= 0) {
        spdlog::error("SegmentationUtils: Invalid dimensions ({}x{}, {} channels)",
                      width, height, channels);
        return false;
    }

    size_t expected_size = static_cast<size_t>(width) * height * channels;
    if (image_data.size() != expected_size) {
        spdlog::error("SegmentationUtils: Data size mismatch. Expected {}, got {}",
                      expected_size, image_data.size());
        return false;
    }

    return true;
}

std::vector<WatershedRegion> SegmentationUtils::Watershed(
    const std::vector<float>& image_data,
    int width, int height, int channels,
    const std::vector<int32_t>& markers,
    std::vector<int32_t>& output_labels
) {
    std::vector<WatershedRegion> regions;

    if (!ValidateInput(image_data, width, height, channels)) {
        return regions;
    }

    if (channels != 3) {
        spdlog::error("SegmentationUtils: Watershed requires 3-channel RGB image");
        return regions;
    }

    if (markers.size() != static_cast<size_t>(width) * height) {
        spdlog::error("SegmentationUtils: Markers size mismatch");
        return regions;
    }

    try {
        // Convert float image to 8-bit BGR
        cv::Mat img(height, width, CV_32FC3, const_cast<float*>(image_data.data()));
        cv::Mat img8u;
        img.convertTo(img8u, CV_8UC3, 255.0);

        // Create markers Mat
        cv::Mat marker_mat(height, width, CV_32S, const_cast<int32_t*>(markers.data()));
        cv::Mat marker_copy = marker_mat.clone();

        // Apply watershed
        cv::watershed(img8u, marker_copy);

        // Copy to output
        output_labels.resize(static_cast<size_t>(width) * height);
        std::memcpy(output_labels.data(), marker_copy.data,
                    output_labels.size() * sizeof(int32_t));

        // Collect region info
        std::map<int, WatershedRegion> region_map;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int label = marker_copy.at<int32_t>(y, x);
                if (label > 0) {
                    auto& region = region_map[label];
                    region.label = label;
                    region.area++;
                } else if (label == -1) {
                    // Boundary pixel - find adjacent regions
                    // Store boundary pixels for region -1
                    if (region_map.find(-1) == region_map.end()) {
                        region_map[-1].label = -1;
                    }
                    region_map[-1].boundary_pixels.push_back({x, y});
                }
            }
        }

        for (auto& [label, region] : region_map) {
            regions.push_back(region);
        }

        spdlog::debug("SegmentationUtils: Watershed found {} regions", regions.size());
        return regions;

    } catch (const cv::Exception& e) {
        spdlog::error("SegmentationUtils: OpenCV exception in Watershed: {}", e.what());
        return regions;
    }
}

std::vector<int32_t> SegmentationUtils::GenerateWatershedMarkers(
    const std::vector<float>& binary_mask,
    int width, int height,
    float dist_threshold
) {
    std::vector<int32_t> markers(static_cast<size_t>(width) * height, 0);

    if (binary_mask.size() != static_cast<size_t>(width) * height) {
        spdlog::error("SegmentationUtils: Binary mask size mismatch");
        return markers;
    }

    try {
        // Convert to 8-bit binary
        cv::Mat binary(height, width, CV_32FC1, const_cast<float*>(binary_mask.data()));
        cv::Mat binary8u;
        binary.convertTo(binary8u, CV_8UC1, 255.0);

        // Distance transform
        cv::Mat dist;
        cv::distanceTransform(binary8u, dist, cv::DIST_L2, 3);

        // Normalize and threshold
        double max_dist;
        cv::minMaxLoc(dist, nullptr, &max_dist);
        if (max_dist > 0) {
            dist = dist / max_dist;
        }

        cv::Mat fg_markers;
        cv::threshold(dist, fg_markers, dist_threshold, 255, cv::THRESH_BINARY);
        fg_markers.convertTo(fg_markers, CV_8UC1);

        // Find connected components as markers
        cv::Mat labels;
        int num_labels = cv::connectedComponents(fg_markers, labels, 8, CV_32S);

        // Background marker (sure background = dilated - foreground)
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::Mat dilated;
        cv::dilate(binary8u, dilated, kernel, cv::Point(-1, -1), 3);
        cv::Mat sure_bg;
        cv::threshold(dilated, sure_bg, 1, 255, cv::THRESH_BINARY);

        // Unknown region
        cv::Mat unknown = sure_bg - fg_markers;

        // Increment labels (so background is 1, not 0)
        labels = labels + 1;

        // Mark unknown region as 0
        labels.setTo(0, unknown == 255);

        // Copy to output
        std::memcpy(markers.data(), labels.data, markers.size() * sizeof(int32_t));

        spdlog::debug("SegmentationUtils: Generated {} watershed markers", num_labels);
        return markers;

    } catch (const cv::Exception& e) {
        spdlog::error("SegmentationUtils: OpenCV exception in GenerateWatershedMarkers: {}", e.what());
        return markers;
    }
}

std::vector<float> SegmentationUtils::GrabCut(
    const std::vector<float>& image_data,
    int width, int height, int channels,
    int rect_x, int rect_y, int rect_width, int rect_height,
    int iterations,
    GrabCutMode mode
) {
    std::vector<float> mask(static_cast<size_t>(width) * height, 0.0f);

    if (!ValidateInput(image_data, width, height, channels)) {
        return mask;
    }

    if (channels != 3) {
        spdlog::error("SegmentationUtils: GrabCut requires 3-channel RGB image");
        return mask;
    }

    // Validate rectangle
    rect_x = std::max(0, rect_x);
    rect_y = std::max(0, rect_y);
    rect_width = std::min(rect_width, width - rect_x);
    rect_height = std::min(rect_height, height - rect_y);

    if (rect_width <= 0 || rect_height <= 0) {
        spdlog::error("SegmentationUtils: Invalid GrabCut rectangle");
        return mask;
    }

    try {
        // Convert float image to 8-bit BGR
        cv::Mat img(height, width, CV_32FC3, const_cast<float*>(image_data.data()));
        cv::Mat img8u;
        img.convertTo(img8u, CV_8UC3, 255.0);

        // GrabCut mask and models
        cv::Mat gc_mask(height, width, CV_8UC1, cv::GC_BGD);
        cv::Mat bgd_model, fgd_model;
        cv::Rect rect(rect_x, rect_y, rect_width, rect_height);

        // Apply GrabCut
        cv::grabCut(img8u, gc_mask, rect, bgd_model, fgd_model, iterations,
                    mode == GrabCutMode::WithRect ? cv::GC_INIT_WITH_RECT : cv::GC_INIT_WITH_MASK);

        // Create binary mask (foreground = 1, background = 0)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                uint8_t val = gc_mask.at<uint8_t>(y, x);
                if (val == cv::GC_FGD || val == cv::GC_PR_FGD) {
                    mask[y * width + x] = 1.0f;
                }
            }
        }

        spdlog::debug("SegmentationUtils: GrabCut completed ({} iterations)", iterations);
        return mask;

    } catch (const cv::Exception& e) {
        spdlog::error("SegmentationUtils: OpenCV exception in GrabCut: {}", e.what());
        return mask;
    }
}

std::vector<float> SegmentationUtils::GrabCutWithMask(
    const std::vector<float>& image_data,
    int width, int height, int channels,
    const std::vector<uint8_t>& initial_mask,
    int iterations
) {
    std::vector<float> mask(static_cast<size_t>(width) * height, 0.0f);

    if (!ValidateInput(image_data, width, height, channels)) {
        return mask;
    }

    if (channels != 3) {
        spdlog::error("SegmentationUtils: GrabCut requires 3-channel RGB image");
        return mask;
    }

    if (initial_mask.size() != static_cast<size_t>(width) * height) {
        spdlog::error("SegmentationUtils: Initial mask size mismatch");
        return mask;
    }

    try {
        // Convert float image to 8-bit BGR
        cv::Mat img(height, width, CV_32FC3, const_cast<float*>(image_data.data()));
        cv::Mat img8u;
        img.convertTo(img8u, CV_8UC3, 255.0);

        // Create mask from input
        cv::Mat gc_mask(height, width, CV_8UC1, const_cast<uint8_t*>(initial_mask.data()));
        gc_mask = gc_mask.clone();  // Make a copy

        cv::Mat bgd_model, fgd_model;
        cv::Rect rect;  // Not used with mask mode

        // Apply GrabCut
        cv::grabCut(img8u, gc_mask, rect, bgd_model, fgd_model, iterations, cv::GC_INIT_WITH_MASK);

        // Create binary mask
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                uint8_t val = gc_mask.at<uint8_t>(y, x);
                if (val == cv::GC_FGD || val == cv::GC_PR_FGD) {
                    mask[y * width + x] = 1.0f;
                }
            }
        }

        spdlog::debug("SegmentationUtils: GrabCutWithMask completed ({} iterations)", iterations);
        return mask;

    } catch (const cv::Exception& e) {
        spdlog::error("SegmentationUtils: OpenCV exception in GrabCutWithMask: {}", e.what());
        return mask;
    }
}

std::vector<ConnectedComponent> SegmentationUtils::FindConnectedComponents(
    const std::vector<float>& binary_mask,
    int width, int height,
    int connectivity,
    std::vector<int32_t>* output_labels
) {
    std::vector<ConnectedComponent> components;

    if (binary_mask.size() != static_cast<size_t>(width) * height) {
        spdlog::error("SegmentationUtils: Binary mask size mismatch");
        return components;
    }

    if (connectivity != 4 && connectivity != 8) {
        spdlog::warn("SegmentationUtils: Invalid connectivity {}, using 8", connectivity);
        connectivity = 8;
    }

    try {
        // Convert to 8-bit binary
        cv::Mat binary(height, width, CV_32FC1, const_cast<float*>(binary_mask.data()));
        cv::Mat binary8u;
        binary.convertTo(binary8u, CV_8UC1, 255.0);

        // Find connected components with stats
        cv::Mat labels, stats, centroids;
        int num_labels = cv::connectedComponentsWithStats(binary8u, labels, stats, centroids,
                                                           connectivity, CV_32S);

        // Copy labels if requested
        if (output_labels) {
            output_labels->resize(static_cast<size_t>(width) * height);
            std::memcpy(output_labels->data(), labels.data,
                        output_labels->size() * sizeof(int32_t));
        }

        // Collect component info (skip background label 0)
        for (int i = 1; i < num_labels; ++i) {
            ConnectedComponent comp;
            comp.label = i;
            comp.area = stats.at<int>(i, cv::CC_STAT_AREA);
            comp.bbox_x = stats.at<int>(i, cv::CC_STAT_LEFT);
            comp.bbox_y = stats.at<int>(i, cv::CC_STAT_TOP);
            comp.bbox_width = stats.at<int>(i, cv::CC_STAT_WIDTH);
            comp.bbox_height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
            comp.centroid_x = static_cast<float>(centroids.at<double>(i, 0));
            comp.centroid_y = static_cast<float>(centroids.at<double>(i, 1));
            components.push_back(comp);
        }

        spdlog::debug("SegmentationUtils: Found {} connected components", components.size());
        return components;

    } catch (const cv::Exception& e) {
        spdlog::error("SegmentationUtils: OpenCV exception in FindConnectedComponents: {}", e.what());
        return components;
    }
}

int SegmentationUtils::FilterComponentsByArea(
    std::vector<float>& binary_mask,
    int width, int height,
    int min_area,
    int max_area
) {
    if (binary_mask.size() != static_cast<size_t>(width) * height) {
        spdlog::error("SegmentationUtils: Binary mask size mismatch");
        return 0;
    }

    try {
        // Find components
        std::vector<int32_t> labels;
        auto components = FindConnectedComponents(binary_mask, width, height, 8, &labels);

        // Filter by area
        std::set<int> keep_labels;
        for (const auto& comp : components) {
            bool keep = true;
            if (min_area > 0 && comp.area < min_area) keep = false;
            if (max_area > 0 && comp.area > max_area) keep = false;
            if (keep) {
                keep_labels.insert(comp.label);
            }
        }

        // Update mask
        for (size_t i = 0; i < binary_mask.size(); ++i) {
            if (labels[i] > 0 && keep_labels.find(labels[i]) == keep_labels.end()) {
                binary_mask[i] = 0.0f;
            }
        }

        spdlog::debug("SegmentationUtils: Kept {} of {} components after area filter",
                      keep_labels.size(), components.size());
        return static_cast<int>(keep_labels.size());

    } catch (const cv::Exception& e) {
        spdlog::error("SegmentationUtils: OpenCV exception in FilterComponentsByArea: {}", e.what());
        return 0;
    }
}

int SegmentationUtils::FloodFill(
    std::vector<float>& image_data,
    int width, int height, int channels,
    int seed_x, int seed_y,
    float new_value,
    float lo_diff,
    float up_diff
) {
    if (!ValidateInput(image_data, width, height, channels)) {
        return 0;
    }

    if (seed_x < 0 || seed_x >= width || seed_y < 0 || seed_y >= height) {
        spdlog::error("SegmentationUtils: Seed point ({}, {}) out of bounds", seed_x, seed_y);
        return 0;
    }

    try {
        cv::Rect filled_rect;
        int area = 0;

        if (channels == 1) {
            cv::Mat img(height, width, CV_32FC1, image_data.data());
            cv::Scalar new_val(new_value);
            cv::Scalar lo(lo_diff);
            cv::Scalar up(up_diff);
            area = cv::floodFill(img, cv::Point(seed_x, seed_y), new_val, &filled_rect, lo, up);
        } else if (channels == 3) {
            cv::Mat img(height, width, CV_32FC3, image_data.data());
            cv::Scalar new_val(new_value, new_value, new_value);
            cv::Scalar lo(lo_diff, lo_diff, lo_diff);
            cv::Scalar up(up_diff, up_diff, up_diff);
            area = cv::floodFill(img, cv::Point(seed_x, seed_y), new_val, &filled_rect, lo, up);
        }

        spdlog::debug("SegmentationUtils: FloodFill filled {} pixels", area);
        return area;

    } catch (const cv::Exception& e) {
        spdlog::error("SegmentationUtils: OpenCV exception in FloodFill: {}", e.what());
        return 0;
    }
}

std::vector<float> SegmentationUtils::FloodFillMask(
    const std::vector<float>& image_data,
    int width, int height, int channels,
    int seed_x, int seed_y,
    float lo_diff,
    float up_diff
) {
    std::vector<float> mask(static_cast<size_t>(width) * height, 0.0f);

    if (!ValidateInput(image_data, width, height, channels)) {
        return mask;
    }

    if (seed_x < 0 || seed_x >= width || seed_y < 0 || seed_y >= height) {
        spdlog::error("SegmentationUtils: Seed point ({}, {}) out of bounds", seed_x, seed_y);
        return mask;
    }

    try {
        // Create mask with 2-pixel border (required by floodFill with mask)
        cv::Mat fill_mask = cv::Mat::zeros(height + 2, width + 2, CV_8UC1);

        if (channels == 1) {
            cv::Mat img(height, width, CV_32FC1, const_cast<float*>(image_data.data()));
            cv::Mat img_copy = img.clone();
            cv::Scalar lo(lo_diff);
            cv::Scalar up(up_diff);
            cv::floodFill(img_copy, fill_mask, cv::Point(seed_x, seed_y),
                         cv::Scalar(1.0), nullptr, lo, up,
                         cv::FLOODFILL_MASK_ONLY | (255 << 8));
        } else if (channels == 3) {
            cv::Mat img(height, width, CV_32FC3, const_cast<float*>(image_data.data()));
            cv::Mat img_copy = img.clone();
            cv::Scalar lo(lo_diff, lo_diff, lo_diff);
            cv::Scalar up(up_diff, up_diff, up_diff);
            cv::floodFill(img_copy, fill_mask, cv::Point(seed_x, seed_y),
                         cv::Scalar(1.0, 1.0, 1.0), nullptr, lo, up,
                         cv::FLOODFILL_MASK_ONLY | (255 << 8));
        }

        // Extract mask (remove 2-pixel border)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (fill_mask.at<uint8_t>(y + 1, x + 1) > 0) {
                    mask[y * width + x] = 1.0f;
                }
            }
        }

        return mask;

    } catch (const cv::Exception& e) {
        spdlog::error("SegmentationUtils: OpenCV exception in FloodFillMask: {}", e.what());
        return mask;
    }
}

} // namespace cyxwiz
