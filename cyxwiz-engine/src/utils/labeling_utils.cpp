#include "labeling_utils.h"
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <set>

namespace cyxwiz {

// ============================================================================
// Enum String Conversions
// ============================================================================

std::string ContourModeToString(ContourMode mode) {
    switch (mode) {
        case ContourMode::External: return "external";
        case ContourMode::List:     return "list";
        case ContourMode::CComp:    return "ccomp";
        case ContourMode::Tree:     return "tree";
        default:                    return "list";
    }
}

ContourMode StringToContourMode(const std::string& str) {
    if (str == "external") return ContourMode::External;
    if (str == "list")     return ContourMode::List;
    if (str == "ccomp")    return ContourMode::CComp;
    if (str == "tree")     return ContourMode::Tree;
    return ContourMode::List;
}

std::string ContourApproxToString(ContourApprox approx) {
    switch (approx) {
        case ContourApprox::None:      return "none";
        case ContourApprox::Simple:    return "simple";
        case ContourApprox::TC89_L1:   return "tc89_l1";
        case ContourApprox::TC89_KCOS: return "tc89_kcos";
        default:                       return "simple";
    }
}

ContourApprox StringToContourApprox(const std::string& str) {
    if (str == "none")      return ContourApprox::None;
    if (str == "simple")    return ContourApprox::Simple;
    if (str == "tc89_l1")   return ContourApprox::TC89_L1;
    if (str == "tc89_kcos") return ContourApprox::TC89_KCOS;
    return ContourApprox::Simple;
}

// ============================================================================
// LabelingUtils Implementation
// ============================================================================

bool LabelingUtils::ValidateInput(const std::vector<float>& data,
                                   int width, int height, int channels) {
    if (width <= 0 || height <= 0 || channels <= 0) {
        spdlog::error("LabelingUtils: Invalid dimensions ({}x{}, {} channels)",
                      width, height, channels);
        return false;
    }

    size_t expected_size = static_cast<size_t>(width) * height * channels;
    if (data.size() != expected_size) {
        spdlog::error("LabelingUtils: Data size mismatch. Expected {}, got {}",
                      expected_size, data.size());
        return false;
    }

    return true;
}

std::vector<Contour> LabelingUtils::ExtractContours(
    const std::vector<float>& binary_mask,
    int width, int height,
    ContourMode mode,
    ContourApprox approx
) {
    std::vector<Contour> result;

    if (!ValidateInput(binary_mask, width, height, 1)) {
        return result;
    }

    try {
        // Convert to 8-bit
        cv::Mat mask(height, width, CV_32FC1, const_cast<float*>(binary_mask.data()));
        cv::Mat mask8u;
        mask.convertTo(mask8u, CV_8UC1, 255.0);

        // Map modes
        int cv_mode;
        switch (mode) {
            case ContourMode::External: cv_mode = cv::RETR_EXTERNAL; break;
            case ContourMode::List:     cv_mode = cv::RETR_LIST; break;
            case ContourMode::CComp:    cv_mode = cv::RETR_CCOMP; break;
            case ContourMode::Tree:     cv_mode = cv::RETR_TREE; break;
            default: cv_mode = cv::RETR_LIST;
        }

        int cv_approx;
        switch (approx) {
            case ContourApprox::None:      cv_approx = cv::CHAIN_APPROX_NONE; break;
            case ContourApprox::Simple:    cv_approx = cv::CHAIN_APPROX_SIMPLE; break;
            case ContourApprox::TC89_L1:   cv_approx = cv::CHAIN_APPROX_TC89_L1; break;
            case ContourApprox::TC89_KCOS: cv_approx = cv::CHAIN_APPROX_TC89_KCOS; break;
            default: cv_approx = cv::CHAIN_APPROX_SIMPLE;
        }

        // Find contours
        std::vector<std::vector<cv::Point>> cv_contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(mask8u, cv_contours, hierarchy, cv_mode, cv_approx);

        // Convert to our format
        for (size_t i = 0; i < cv_contours.size(); ++i) {
            Contour c;

            // Convert points
            for (const auto& pt : cv_contours[i]) {
                c.points.push_back({static_cast<float>(pt.x), static_cast<float>(pt.y)});
            }

            // Hierarchy: [next, previous, child, parent]
            if (i < hierarchy.size()) {
                c.next = hierarchy[i][0];
                c.prev = hierarchy[i][1];
                c.child = hierarchy[i][2];
                c.parent = hierarchy[i][3];

                // Calculate level by traversing parent chain
                int level = 0;
                int p = c.parent;
                while (p >= 0 && p < static_cast<int>(hierarchy.size())) {
                    level++;
                    p = hierarchy[p][3];
                }
                c.level = level;
            }

            result.push_back(c);
        }

        spdlog::debug("LabelingUtils: Extracted {} contours", result.size());
        return result;

    } catch (const cv::Exception& e) {
        spdlog::error("LabelingUtils: OpenCV exception in ExtractContours: {}", e.what());
        return result;
    }
}

std::vector<RegionProposal> LabelingUtils::SelectiveSearch(
    const std::vector<float>& image_data,
    int width, int height, int channels
) {
    std::vector<RegionProposal> proposals;

    if (!ValidateInput(image_data, width, height, channels)) {
        return proposals;
    }

    if (channels != 3) {
        spdlog::error("LabelingUtils: SelectiveSearch requires 3-channel RGB image");
        return proposals;
    }

    try {
        // Convert to 8-bit BGR
        cv::Mat img(height, width, CV_32FC3, const_cast<float*>(image_data.data()));
        cv::Mat img8u;
        img.convertTo(img8u, CV_8UC3, 255.0);

        // SelectiveSearch requires the ximgproc module from opencv_contrib
        // For now, we'll use a simple alternative based on superpixels or sliding window
        // This is a simplified fallback implementation

        spdlog::warn("LabelingUtils: SelectiveSearch requires opencv_contrib (ximgproc). "
                    "Using fallback sliding window approach.");

        // Fallback: multi-scale sliding window
        std::vector<int> scales = {32, 64, 128, 256};
        float aspect_ratios[] = {0.5f, 1.0f, 2.0f};

        for (int scale : scales) {
            for (float ar : aspect_ratios) {
                int w = scale;
                int h = static_cast<int>(scale / ar);

                for (int y = 0; y < height - h; y += scale / 2) {
                    for (int x = 0; x < width - w; x += scale / 2) {
                        RegionProposal prop;
                        prop.x = x;
                        prop.y = y;
                        prop.width = w;
                        prop.height = h;
                        prop.score = 0.5f;  // Placeholder score
                        proposals.push_back(prop);
                    }
                }
            }
        }

        spdlog::debug("LabelingUtils: Generated {} region proposals (fallback)", proposals.size());
        return proposals;

    } catch (const cv::Exception& e) {
        spdlog::error("LabelingUtils: OpenCV exception in SelectiveSearch: {}", e.what());
        return proposals;
    }
}

std::vector<RegionProposal> LabelingUtils::EdgeBoxes(
    const std::vector<float>& image_data,
    int width, int height, int channels,
    int max_boxes
) {
    std::vector<RegionProposal> proposals;

    if (!ValidateInput(image_data, width, height, channels)) {
        return proposals;
    }
    if (max_boxes <= 0) {
        spdlog::error("LabelingUtils: EdgeBoxes max_boxes must be positive");
        return proposals;
    }

    try {
        // EdgeBoxes also requires opencv_contrib (ximgproc)
        // Fallback: use contour-based proposals

        spdlog::warn("LabelingUtils: EdgeBoxes requires opencv_contrib (ximgproc). "
                    "Using contour-based fallback.");

        // Convert to grayscale and detect edges
        cv::Mat img(height, width, channels == 1 ? CV_32FC1 : CV_32FC3,
                   const_cast<float*>(image_data.data()));
        cv::Mat gray;

        if (channels == 3) {
            cv::Mat img8u;
            img.convertTo(img8u, CV_8UC3, 255.0);
            cv::cvtColor(img8u, gray, cv::COLOR_BGR2GRAY);
        } else {
            img.convertTo(gray, CV_8UC1, 255.0);
        }

        // Edge detection
        cv::Mat edges;
        cv::Canny(gray, edges, 50, 150);

        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Convert contour bounding boxes to proposals
        for (const auto& contour : contours) {
            cv::Rect bbox = cv::boundingRect(contour);

            // Filter small regions
            if (bbox.width < 10 || bbox.height < 10) continue;

            RegionProposal prop;
            prop.x = bbox.x;
            prop.y = bbox.y;
            prop.width = bbox.width;
            prop.height = bbox.height;
            prop.score = static_cast<float>(cv::contourArea(contour)) /
                        (bbox.width * bbox.height);  // Extent as score
            proposals.push_back(prop);

            if (static_cast<int>(proposals.size()) >= max_boxes) break;
        }

        // Sort by score
        std::sort(proposals.begin(), proposals.end(),
                  [](const RegionProposal& a, const RegionProposal& b) {
                      return a.score > b.score;
                  });

        // Limit to max_boxes
        if (static_cast<int>(proposals.size()) > max_boxes) {
            proposals.resize(max_boxes);
        }

        spdlog::debug("LabelingUtils: Generated {} edge-based proposals", proposals.size());
        return proposals;

    } catch (const cv::Exception& e) {
        spdlog::error("LabelingUtils: OpenCV exception in EdgeBoxes: {}", e.what());
        return proposals;
    }
}

std::vector<float> LabelingUtils::CreateMaskFromContour(
    const std::vector<Point2D>& contour,
    int width, int height,
    bool fill, int thickness
) {
    std::vector<float> mask(static_cast<size_t>(width) * height, 0.0f);

    if (contour.size() < 3) {
        return mask;
    }

    try {
        cv::Mat mask_mat = cv::Mat::zeros(height, width, CV_8UC1);

        // Convert points
        std::vector<cv::Point> cv_contour;
        for (const auto& p : contour) {
            cv_contour.push_back(cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)));
        }

        std::vector<std::vector<cv::Point>> contours = {cv_contour};

        if (fill) {
            cv::drawContours(mask_mat, contours, 0, cv::Scalar(255), cv::FILLED);
        } else {
            cv::drawContours(mask_mat, contours, 0, cv::Scalar(255), thickness);
        }

        // Convert to float
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                mask[y * width + x] = mask_mat.at<uint8_t>(y, x) / 255.0f;
            }
        }

        return mask;

    } catch (const cv::Exception& e) {
        spdlog::error("LabelingUtils: OpenCV exception in CreateMaskFromContour: {}", e.what());
        return mask;
    }
}

std::vector<float> LabelingUtils::CreateMaskFromContours(
    const std::vector<std::vector<Point2D>>& contours,
    int width, int height,
    bool fill
) {
    std::vector<float> mask(static_cast<size_t>(width) * height, 0.0f);

    if (contours.empty()) {
        return mask;
    }

    try {
        cv::Mat mask_mat = cv::Mat::zeros(height, width, CV_8UC1);

        // Convert all contours
        std::vector<std::vector<cv::Point>> cv_contours;
        for (const auto& contour : contours) {
            std::vector<cv::Point> cv_contour;
            for (const auto& p : contour) {
                cv_contour.push_back(cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)));
            }
            if (cv_contour.size() >= 3) {
                cv_contours.push_back(cv_contour);
            }
        }

        cv::drawContours(mask_mat, cv_contours, -1, cv::Scalar(255),
                        fill ? cv::FILLED : 1);

        // Convert to float
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                mask[y * width + x] = mask_mat.at<uint8_t>(y, x) / 255.0f;
            }
        }

        return mask;

    } catch (const cv::Exception& e) {
        spdlog::error("LabelingUtils: OpenCV exception in CreateMaskFromContours: {}", e.what());
        return mask;
    }
}

std::vector<float> LabelingUtils::CreateMaskFromBBox(
    const BoundingBox& bbox,
    int width, int height,
    bool fill
) {
    std::vector<float> mask(static_cast<size_t>(width) * height, 0.0f);

    try {
        cv::Mat mask_mat = cv::Mat::zeros(height, width, CV_8UC1);

        if (std::abs(bbox.angle) < 0.01f) {
            // Axis-aligned box
            cv::Rect rect(static_cast<int>(bbox.x), static_cast<int>(bbox.y),
                         static_cast<int>(bbox.width), static_cast<int>(bbox.height));
            if (fill) {
                cv::rectangle(mask_mat, rect, cv::Scalar(255), cv::FILLED);
            } else {
                cv::rectangle(mask_mat, rect, cv::Scalar(255), 1);
            }
        } else {
            // Rotated box - use rotated rectangle
            cv::RotatedRect rr(
                cv::Point2f(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2),
                cv::Size2f(bbox.width, bbox.height),
                bbox.angle
            );
            cv::Point2f corners[4];
            rr.points(corners);

            std::vector<cv::Point> pts;
            for (int i = 0; i < 4; ++i) {
                pts.push_back(cv::Point(static_cast<int>(corners[i].x),
                                       static_cast<int>(corners[i].y)));
            }

            std::vector<std::vector<cv::Point>> contours = {pts};
            cv::drawContours(mask_mat, contours, 0, cv::Scalar(255),
                           fill ? cv::FILLED : 1);
        }

        // Convert to float
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                mask[y * width + x] = mask_mat.at<uint8_t>(y, x) / 255.0f;
            }
        }

        return mask;

    } catch (const cv::Exception& e) {
        spdlog::error("LabelingUtils: OpenCV exception in CreateMaskFromBBox: {}", e.what());
        return mask;
    }
}

BoundingBox LabelingUtils::ContourToBBox(
    const std::vector<Point2D>& contour,
    bool rotated
) {
    BoundingBox bbox;

    if (contour.size() < 3) {
        return bbox;
    }

    try {
        std::vector<cv::Point2f> cv_points;
        for (const auto& p : contour) {
            cv_points.push_back(cv::Point2f(p.x, p.y));
        }

        if (rotated) {
            cv::RotatedRect rr = cv::minAreaRect(cv_points);
            bbox.x = rr.center.x - rr.size.width / 2;
            bbox.y = rr.center.y - rr.size.height / 2;
            bbox.width = rr.size.width;
            bbox.height = rr.size.height;
            bbox.angle = rr.angle;
        } else {
            cv::Rect rect = cv::boundingRect(cv_points);
            bbox.x = static_cast<float>(rect.x);
            bbox.y = static_cast<float>(rect.y);
            bbox.width = static_cast<float>(rect.width);
            bbox.height = static_cast<float>(rect.height);
            bbox.angle = 0.0f;
        }

        return bbox;

    } catch (const cv::Exception& e) {
        spdlog::error("LabelingUtils: OpenCV exception in ContourToBBox: {}", e.what());
        return bbox;
    }
}

std::vector<BoundingBox> LabelingUtils::MaskToBBoxes(
    const std::vector<float>& binary_mask,
    int width, int height,
    int min_area
) {
    std::vector<BoundingBox> boxes;

    if (!ValidateInput(binary_mask, width, height, 1)) {
        return boxes;
    }

    try {
        cv::Mat mask(height, width, CV_32FC1, const_cast<float*>(binary_mask.data()));
        cv::Mat mask8u;
        mask.convertTo(mask8u, CV_8UC1, 255.0);

        // Find connected components
        cv::Mat labels, stats, centroids;
        int num_labels = cv::connectedComponentsWithStats(mask8u, labels, stats, centroids, 8);

        for (int i = 1; i < num_labels; ++i) {  // Skip background (0)
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area < min_area) continue;

            BoundingBox bbox;
            bbox.x = static_cast<float>(stats.at<int>(i, cv::CC_STAT_LEFT));
            bbox.y = static_cast<float>(stats.at<int>(i, cv::CC_STAT_TOP));
            bbox.width = static_cast<float>(stats.at<int>(i, cv::CC_STAT_WIDTH));
            bbox.height = static_cast<float>(stats.at<int>(i, cv::CC_STAT_HEIGHT));
            boxes.push_back(bbox);
        }

        return boxes;

    } catch (const cv::Exception& e) {
        spdlog::error("LabelingUtils: OpenCV exception in MaskToBBoxes: {}", e.what());
        return boxes;
    }
}

std::vector<int> LabelingUtils::NonMaxSuppression(
    const std::vector<BoundingBox>& boxes,
    float iou_threshold
) {
    std::vector<int> keep;

    if (boxes.empty()) {
        return keep;
    }

    // Sort by confidence (or area if no confidence)
    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&boxes](int a, int b) {
        return boxes[a].confidence > boxes[b].confidence;
    });

    std::vector<bool> suppressed(boxes.size(), false);

    for (size_t i = 0; i < order.size(); ++i) {
        int idx = order[i];
        if (suppressed[idx]) continue;

        keep.push_back(idx);

        for (size_t j = i + 1; j < order.size(); ++j) {
            int jdx = order[j];
            if (suppressed[jdx]) continue;

            float iou = CalculateIoU(boxes[idx], boxes[jdx]);
            if (iou > iou_threshold) {
                suppressed[jdx] = true;
            }
        }
    }

    return keep;
}

float LabelingUtils::CalculateIoU(const BoundingBox& box1, const BoundingBox& box2) {
    // Calculate intersection
    float x1 = std::max(box1.x, box2.x);
    float y1 = std::max(box1.y, box2.y);
    float x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    float y2 = std::min(box1.y + box1.height, box2.y + box2.height);

    if (x2 <= x1 || y2 <= y1) {
        return 0.0f;
    }

    float intersection = (x2 - x1) * (y2 - y1);
    float area1 = box1.width * box1.height;
    float area2 = box2.width * box2.height;
    float union_area = area1 + area2 - intersection;

    return union_area > 0 ? intersection / union_area : 0.0f;
}

void LabelingUtils::DilateMask(
    std::vector<float>& mask,
    int width, int height,
    int kernel_size,
    int iterations
) {
    if (!ValidateInput(mask, width, height, 1)) return;

    try {
        cv::Mat mask_mat(height, width, CV_32FC1, mask.data());
        cv::Mat mask8u;
        mask_mat.convertTo(mask8u, CV_8UC1, 255.0);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                    cv::Size(kernel_size, kernel_size));
        cv::dilate(mask8u, mask8u, kernel, cv::Point(-1, -1), iterations);

        mask8u.convertTo(mask_mat, CV_32FC1, 1.0 / 255.0);

    } catch (const cv::Exception& e) {
        spdlog::error("LabelingUtils: OpenCV exception in DilateMask: {}", e.what());
    }
}

void LabelingUtils::ErodeMask(
    std::vector<float>& mask,
    int width, int height,
    int kernel_size,
    int iterations
) {
    if (!ValidateInput(mask, width, height, 1)) return;

    try {
        cv::Mat mask_mat(height, width, CV_32FC1, mask.data());
        cv::Mat mask8u;
        mask_mat.convertTo(mask8u, CV_8UC1, 255.0);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                    cv::Size(kernel_size, kernel_size));
        cv::erode(mask8u, mask8u, kernel, cv::Point(-1, -1), iterations);

        mask8u.convertTo(mask_mat, CV_32FC1, 1.0 / 255.0);

    } catch (const cv::Exception& e) {
        spdlog::error("LabelingUtils: OpenCV exception in ErodeMask: {}", e.what());
    }
}

void LabelingUtils::SmoothMaskBoundary(
    std::vector<float>& mask,
    int width, int height,
    int kernel_size,
    float threshold
) {
    if (!ValidateInput(mask, width, height, 1)) return;

    try {
        cv::Mat mask_mat(height, width, CV_32FC1, mask.data());

        // Gaussian blur
        cv::GaussianBlur(mask_mat, mask_mat, cv::Size(kernel_size, kernel_size), 0);

        // Threshold back to binary
        cv::threshold(mask_mat, mask_mat, threshold, 1.0, cv::THRESH_BINARY);

    } catch (const cv::Exception& e) {
        spdlog::error("LabelingUtils: OpenCV exception in SmoothMaskBoundary: {}", e.what());
    }
}

void LabelingUtils::FillHoles(
    std::vector<float>& mask,
    int width, int height
) {
    if (!ValidateInput(mask, width, height, 1)) return;

    try {
        cv::Mat mask_mat(height, width, CV_32FC1, mask.data());
        cv::Mat mask8u;
        mask_mat.convertTo(mask8u, CV_8UC1, 255.0);

        // Find contours and fill
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask8u, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(mask8u, contours, -1, cv::Scalar(255), cv::FILLED);

        mask8u.convertTo(mask_mat, CV_32FC1, 1.0 / 255.0);

    } catch (const cv::Exception& e) {
        spdlog::error("LabelingUtils: OpenCV exception in FillHoles: {}", e.what());
    }
}

std::vector<int> LabelingUtils::PolygonToRLE(
    const std::vector<Point2D>& contour,
    int width, int height
) {
    std::vector<int> rle;

    if (contour.size() < 3) {
        return rle;
    }

    // Create mask from contour
    auto mask = CreateMaskFromContour(contour, width, height, true);

    // Convert to RLE (column-major order for COCO format)
    bool prev_val = false;
    int count = 0;

    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            bool val = mask[y * width + x] > 0.5f;

            if (val != prev_val) {
                rle.push_back(count);
                count = 0;
                prev_val = val;
            }
            count++;
        }
    }
    rle.push_back(count);

    return rle;
}

std::vector<float> LabelingUtils::RLEToMask(
    const std::vector<int>& rle,
    int width, int height
) {
    std::vector<float> mask(static_cast<size_t>(width) * height, 0.0f);

    if (rle.empty()) {
        return mask;
    }

    // Decode RLE (column-major order)
    int idx = 0;
    bool val = false;  // Start with background

    for (size_t i = 0; i < rle.size() && idx < width * height; ++i) {
        int count = rle[i];
        for (int j = 0; j < count && idx < width * height; ++j) {
            int x = idx / height;
            int y = idx % height;
            if (val) {
                mask[y * width + x] = 1.0f;
            }
            idx++;
        }
        val = !val;
    }

    return mask;
}

} // namespace cyxwiz
