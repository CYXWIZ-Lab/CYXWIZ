/**
 * OpenCV Integration Tests
 *
 * Tests all 4 phases of OpenCV integration:
 * - Phase 1: Preprocessing transforms (Morphology, Blur, Perspective, Pyramid)
 * - Phase 2: Dataset utilities (Segmentation, Shape, Labeling)
 *
 * Run: test_opencv_integration.exe [output_dir]
 * Outputs test images to verify transforms work correctly.
 */

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>
#include <cmath>

// Phase 1: Preprocessing
#include "preprocessing/morphology_transform.h"
#include "preprocessing/blur_transform.h"
#include "preprocessing/perspective_transform.h"
#include "preprocessing/pyramid_transform.h"

// Phase 2: Utilities
#include "utils/segmentation_utils.h"
#include "utils/shape_utils.h"
#include "utils/labeling_utils.h"

// Core utilities
#include "core/image_utils.h"

namespace fs = std::filesystem;

// Test result tracking
struct TestResult {
    std::string name;
    bool passed;
    std::string message;
    double time_ms;
};

std::vector<TestResult> g_results;

void RecordTest(const std::string& name, bool passed, const std::string& msg, double ms) {
    g_results.push_back({name, passed, msg, ms});
    std::cout << (passed ? "[PASS]" : "[FAIL]") << " " << name;
    if (!passed) std::cout << " - " << msg;
    std::cout << " (" << ms << "ms)" << std::endl;
}

// Create a test image with patterns for testing transforms
std::vector<float> CreateTestImage(int width, int height, int channels) {
    std::vector<float> img(width * height * channels);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * channels;

            // Create distinct regions for testing
            bool in_circle = std::pow(x - width/2, 2) + std::pow(y - height/2, 2) < std::pow(width/4, 2);
            bool in_rect = x > width/4 && x < 3*width/4 && y > height/4 && y < 3*height/4;

            if (channels == 1) {
                img[idx] = in_circle ? 1.0f : (float)x / width * 0.5f;
            } else {
                img[idx + 0] = in_circle ? 1.0f : 0.1f;
                img[idx + 1] = in_rect ? 0.8f : 0.1f;
                img[idx + 2] = (float)y / height * 0.8f;
            }
        }
    }
    return img;
}

// Create binary test mask
std::vector<uint8_t> CreateBinaryMask(int width, int height) {
    std::vector<uint8_t> mask(width * height, 0);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (std::pow(x - width/2, 2) + std::pow(y - height/2, 2) < std::pow(width/4, 2)) {
                mask[y * width + x] = 255;
            }
        }
    }
    return mask;
}

// ============================================================================
// Phase 1 Tests: Preprocessing Transforms
// ============================================================================

void TestMorphologyTransform(const std::string& output_dir) {
    auto img = CreateTestImage(128, 128, 1);
    int w = 128, h = 128, c = 1;

    // Test Erode
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto eroded = img;
        bool ok = cyxwiz::MorphologyTransform::Erode(eroded, w, h, c, 3, 1);
        bool passed = ok && eroded.size() == img.size();

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        RecordTest("Morphology - Erode", passed, passed ? "" : "Erode failed", ms);

        if (passed && !output_dir.empty()) {
            cyxwiz::ImageUtils::SaveMask(output_dir + "/morph_erode.png", eroded, w, h, false);
        }
    }

    // Test Dilate
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto dilated = img;
        bool ok = cyxwiz::MorphologyTransform::Dilate(dilated, w, h, c, 3, 1);
        bool passed = ok;

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        RecordTest("Morphology - Dilate", passed, passed ? "" : "Dilate failed", ms);

        if (passed && !output_dir.empty()) {
            cyxwiz::ImageUtils::SaveMask(output_dir + "/morph_dilate.png", dilated, w, h, false);
        }
    }

    // Test Open
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto opened = img;
        bool ok = cyxwiz::MorphologyTransform::Open(opened, w, h, c, 5);

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        RecordTest("Morphology - Open", ok, ok ? "" : "Open failed", ms);
    }

    // Test Close
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto closed = img;
        bool ok = cyxwiz::MorphologyTransform::Close(closed, w, h, c, 5);

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        RecordTest("Morphology - Close", ok, ok ? "" : "Close failed", ms);
    }

    // Test Gradient
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto gradient = img;
        bool ok = cyxwiz::MorphologyTransform::Gradient(gradient, w, h, c, 3);

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        RecordTest("Morphology - Gradient", ok, ok ? "" : "Gradient failed", ms);

        if (ok && !output_dir.empty()) {
            cyxwiz::ImageUtils::SaveMask(output_dir + "/morph_gradient.png", gradient, w, h, false);
        }
    }
}

void TestBlurTransform(const std::string& output_dir) {
    auto img = CreateTestImage(128, 128, 3);
    int w = 128, h = 128, c = 3;

    // Test Gaussian
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto blurred = img;
        bool ok = cyxwiz::BlurTransform::GaussianBlur(blurred, w, h, c, 5, 1.5f);
        bool passed = ok && blurred.size() == img.size();

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        RecordTest("Blur - Gaussian", passed, passed ? "" : "GaussianBlur failed", ms);

        if (passed && !output_dir.empty()) {
            cyxwiz::ImageUtils::SaveImage(output_dir + "/blur_gaussian.png", blurred, w, h, c);
        }
    }

    // Test Median
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto median = img;
        bool ok = cyxwiz::BlurTransform::MedianBlur(median, w, h, c, 5);

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        RecordTest("Blur - Median", ok, ok ? "" : "MedianBlur failed", ms);

        if (ok && !output_dir.empty()) {
            cyxwiz::ImageUtils::SaveImage(output_dir + "/blur_median.png", median, w, h, c);
        }
    }

    // Test Bilateral
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto bilateral = img;
        bool ok = cyxwiz::BlurTransform::BilateralFilter(bilateral, w, h, c, 9, 75.0f, 75.0f);

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        RecordTest("Blur - Bilateral", ok, ok ? "" : "BilateralFilter failed", ms);

        if (ok && !output_dir.empty()) {
            cyxwiz::ImageUtils::SaveImage(output_dir + "/blur_bilateral.png", bilateral, w, h, c);
        }
    }

    // Test Box
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto box = img;
        bool ok = cyxwiz::BlurTransform::BoxBlur(box, w, h, c, 5);

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        RecordTest("Blur - Box", ok, ok ? "" : "BoxBlur failed", ms);
    }
}

void TestPerspectiveTransform(const std::string& output_dir) {
    auto img = CreateTestImage(128, 128, 3);
    int w = 128, h = 128, c = 3;

    // Test Rotate
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto rotated = img;
        bool ok = cyxwiz::PerspectiveTransform::Rotate(rotated, w, h, c, 45.0f, 1.0f, true);

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        RecordTest("Perspective - Rotate", ok, ok ? "" : "Rotate failed", ms);

        if (ok && !output_dir.empty()) {
            cyxwiz::ImageUtils::SaveImage(output_dir + "/perspective_rotate45.png", rotated, w, h, c);
        }
    }

    // Test LogPolar
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto polar = img;
        bool ok = cyxwiz::PerspectiveTransform::LogPolar(polar, w, h, c, 0.5f, 0.5f, 0.4f);

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        RecordTest("Perspective - LogPolar", ok, ok ? "" : "LogPolar failed", ms);

        if (ok && !output_dir.empty()) {
            cyxwiz::ImageUtils::SaveImage(output_dir + "/perspective_logpolar.png", polar, w, h, c);
        }
    }
}

void TestPyramidTransform(const std::string& output_dir) {
    auto img = CreateTestImage(128, 128, 3);
    int w = 128, h = 128, c = 3;

    // Test PyrDown
    std::vector<float> down;
    int dw = 0, dh = 0;
    {
        auto start = std::chrono::high_resolution_clock::now();
        down = img;
        dw = w; dh = h;
        bool ok = cyxwiz::PyramidTransform::PyrDown(down, dw, dh, c);
        bool passed = ok && dw == 64 && dh == 64 && down.size() == 64 * 64 * 3;

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        RecordTest("Pyramid - PyrDown", passed, passed ? "" : "PyrDown failed or wrong size", ms);

        if (passed && !output_dir.empty()) {
            cyxwiz::ImageUtils::SaveImage(output_dir + "/pyramid_down.png", down, dw, dh, c);
        }
    }

    // Test PyrUp
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto up = down;
        int uw = dw, uh = dh;
        bool ok = cyxwiz::PyramidTransform::PyrUp(up, uw, uh, c);
        bool passed = ok && uw == 128 && uh == 128;

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        RecordTest("Pyramid - PyrUp", passed, passed ? "" : "PyrUp failed or wrong size", ms);

        if (passed && !output_dir.empty()) {
            cyxwiz::ImageUtils::SaveImage(output_dir + "/pyramid_up.png", up, uw, uh, c);
        }
    }
}

// ============================================================================
// Phase 2 Tests: Dataset Utilities
// ============================================================================

void TestSegmentationUtils(const std::string& output_dir) {
    auto mask_u8 = CreateBinaryMask(128, 128);
    int w = 128, h = 128;

    // Convert to float mask
    std::vector<float> mask(mask_u8.begin(), mask_u8.end());
    for (auto& v : mask) v /= 255.0f;

    // Test ConnectedComponents
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<int32_t> labels;
        auto components = cyxwiz::SegmentationUtils::FindConnectedComponents(mask, w, h, 8, &labels);
        bool passed = labels.size() == mask.size() && components.size() >= 1;

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        RecordTest("Segmentation - ConnectedComponents", passed,
                   passed ? "" : "No components found", ms);

        if (passed) {
            std::cout << "    Found " << components.size() << " components" << std::endl;
        }
    }

    // Test FloodFill
    {
        auto img = CreateTestImage(128, 128, 3);
        auto start = std::chrono::high_resolution_clock::now();

        auto flood_mask = cyxwiz::SegmentationUtils::FloodFillMask(img, w, h, 3, 64, 64, 0.2f, 0.2f);
        bool passed = flood_mask.size() == (size_t)(w * h) && flood_mask[64 * 128 + 64] > 0;

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        RecordTest("Segmentation - FloodFill", passed,
                   passed ? "" : "FloodFill failed or center not filled", ms);

        if (passed && !output_dir.empty()) {
            std::vector<float> mask_float(flood_mask.begin(), flood_mask.end());
            for (auto& v : mask_float) v /= 255.0f;
            cyxwiz::ImageUtils::SaveMask(output_dir + "/segment_flood.png", mask_float, w, h, true);
        }
    }
}

void TestShapeUtils(const std::string& output_dir) {
    // Test ConvexHull
    {
        std::vector<cyxwiz::Point2D> contour = {
            {10, 10}, {50, 10}, {50, 50}, {10, 50}, {30, 30}
        };

        auto start = std::chrono::high_resolution_clock::now();
        auto hull = cyxwiz::ShapeUtils::ConvexHull(contour);
        bool passed = hull.size() == 4;

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        RecordTest("Shape - ConvexHull", passed,
                   passed ? "" : "Hull should have 4 points", ms);
    }

    // Test MinAreaRect
    {
        std::vector<cyxwiz::Point2D> contour = {
            {10, 10}, {50, 10}, {50, 50}, {10, 50}
        };

        auto start = std::chrono::high_resolution_clock::now();
        auto rect = cyxwiz::ShapeUtils::MinAreaRect(contour);
        bool passed = rect.width > 0 && rect.height > 0;

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        RecordTest("Shape - MinAreaRect", passed,
                   passed ? "" : "Invalid rect dimensions", ms);

        if (passed) {
            std::cout << "    MinAreaRect: " << rect.width << "x" << rect.height
                      << " at angle " << rect.angle << std::endl;
        }
    }

    // Test Descriptors
    {
        std::vector<cyxwiz::Point2D> circle_pts;
        for (int i = 0; i < 36; ++i) {
            float angle = i * 10.0f * 3.14159f / 180.0f;
            circle_pts.push_back({50.0f + 30.0f * std::cos(angle),
                                  50.0f + 30.0f * std::sin(angle)});
        }

        auto start = std::chrono::high_resolution_clock::now();
        auto desc = cyxwiz::ShapeUtils::ComputeDescriptors(circle_pts);
        bool passed = desc.circularity > 0.8f;

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        RecordTest("Shape - Descriptors", passed,
                   passed ? "" : "Circle should have high circularity", ms);

        if (passed) {
            std::cout << "    Circularity: " << desc.circularity
                      << ", Aspect: " << desc.aspect_ratio << std::endl;
        }
    }
}

void TestLabelingUtils(const std::string& output_dir) {
    auto mask = CreateBinaryMask(128, 128);
    int w = 128, h = 128;

    // Convert to float mask
    std::vector<float> float_mask(mask.begin(), mask.end());
    for (auto& v : float_mask) v /= 255.0f;

    std::vector<cyxwiz::Contour> contours;

    // Test ExtractContours
    {
        auto start = std::chrono::high_resolution_clock::now();
        contours = cyxwiz::LabelingUtils::ExtractContours(float_mask, w, h,
            cyxwiz::ContourMode::External, cyxwiz::ContourApprox::Simple);
        bool passed = contours.size() >= 1;

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        RecordTest("Labeling - ExtractContours", passed,
                   passed ? "" : "No contours found", ms);

        if (passed) {
            std::cout << "    Found " << contours.size() << " contours" << std::endl;
        }
    }

    // Test ContourToBBox
    if (!contours.empty()) {
        auto start = std::chrono::high_resolution_clock::now();
        auto bbox = cyxwiz::LabelingUtils::ContourToBBox(contours[0].points, false);
        bool passed = bbox.width > 0 && bbox.height > 0;

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        RecordTest("Labeling - ContourToBBox", passed,
                   passed ? "" : "Invalid bbox", ms);

        if (passed) {
            std::cout << "    BBox: " << bbox.x << "," << bbox.y << " "
                      << bbox.width << "x" << bbox.height << std::endl;
        }
    }

    // Test CreateMaskFromContour
    if (!contours.empty()) {
        auto start = std::chrono::high_resolution_clock::now();
        auto new_mask = cyxwiz::LabelingUtils::CreateMaskFromContour(
            contours[0].points, w, h, true);
        bool passed = new_mask.size() == (size_t)(w * h);

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        RecordTest("Labeling - CreateMaskFromContour", passed,
                   passed ? "" : "Wrong mask size", ms);

        if (passed && !output_dir.empty()) {
            // new_mask is already float [0,1]
            cyxwiz::ImageUtils::SaveMask(output_dir + "/label_mask.png", new_mask, w, h, true);
        }
    }

    // Test IoU
    {
        cyxwiz::BoundingBox box1;
        box1.x = 0; box1.y = 0; box1.width = 100; box1.height = 100;
        cyxwiz::BoundingBox box2;
        box2.x = 50; box2.y = 50; box2.width = 100; box2.height = 100;

        auto start = std::chrono::high_resolution_clock::now();
        float iou = cyxwiz::LabelingUtils::CalculateIoU(box1, box2);
        bool passed = iou > 0.1f && iou < 0.2f;

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        RecordTest("Labeling - IoU", passed,
                   passed ? "" : "IoU calculation wrong", ms);

        std::cout << "    IoU: " << iou << " (expected ~0.14)" << std::endl;
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "CyxWiz OpenCV Integration Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    std::string output_dir;
    if (argc > 1) {
        output_dir = argv[1];
        fs::create_directories(output_dir);
        std::cout << "Output directory: " << output_dir << std::endl;
    } else {
        std::cout << "No output directory specified, skipping image saves" << std::endl;
        std::cout << "Usage: test_opencv_integration.exe <output_dir>" << std::endl;
    }
    std::cout << std::endl;

    // Save original test image
    if (!output_dir.empty()) {
        auto test_img = CreateTestImage(128, 128, 3);
        cyxwiz::ImageUtils::SaveImage(output_dir + "/original.png", test_img, 128, 128, 3);

        auto test_gray = CreateTestImage(128, 128, 1);
        cyxwiz::ImageUtils::SaveMask(output_dir + "/original_gray.png", test_gray, 128, 128, false);
        std::cout << "Saved test images to " << output_dir << std::endl << std::endl;
    }

    std::cout << "--- Phase 1: Preprocessing Transforms ---" << std::endl;
    TestMorphologyTransform(output_dir);
    TestBlurTransform(output_dir);
    TestPerspectiveTransform(output_dir);
    TestPyramidTransform(output_dir);

    std::cout << std::endl << "--- Phase 2: Dataset Utilities ---" << std::endl;
    TestSegmentationUtils(output_dir);
    TestShapeUtils(output_dir);
    TestLabelingUtils(output_dir);

    // Summary
    std::cout << std::endl << "========================================" << std::endl;
    int passed = 0, failed = 0;
    for (const auto& r : g_results) {
        if (r.passed) passed++;
        else failed++;
    }

    std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;

    if (failed > 0) {
        std::cout << std::endl << "Failed tests:" << std::endl;
        for (const auto& r : g_results) {
            if (!r.passed) {
                std::cout << "  - " << r.name << ": " << r.message << std::endl;
            }
        }
    }

    std::cout << "========================================" << std::endl;

    return failed > 0 ? 1 : 0;
}
