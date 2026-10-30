#pragma once

// The OpenCV image steps (TOFIX178 1.3f): Image Contrast (CLAHE), Image
// Denoise, Image Sharpen, Image Edges, Image Smooth, Image Warp and Image
// Pyramid. Each gives every image the same result in every epoch, so it is
// done once when the images are prepared, on the CPU workers, in canvas
// order with the device transforms that are done once (Center Crop,
// Grayscale, Gaussian Blur, Morphology). The prepared set is uint8, so each
// step reads and writes uint8 [H, W, C] images. No OpenCV in this header;
// ApplyCvImageStep lives in image_cv_steps_apply.cpp.

#include <cyxwiz/image_augmentation.h>

#include <nlohmann/json.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace cyxwiz {

enum class CvImageStepKind { Contrast, Denoise, Sharpen, Edges, Smooth, Warp, Pyramid };
enum class CvContrastColour { Lightness, EachChannel };
enum class CvEdgeMethod { Canny, Sobel, Laplacian, Scharr };
enum class CvSmoothMethod { Box, Median, Bilateral };
enum class CvWarpMethod { FourCorners, ThreePoints, LogPolar, LinearPolar };
enum class CvWarpInterpolation { Nearest, Linear, Cubic, Lanczos };
enum class CvPyramidOperation { Down, Up, Detail };

struct CvImageStep {
    CvImageStepKind kind = CvImageStepKind::Contrast;
    // The device transforms compiled before it (TrainingConfiguration::
    // image_augmentation.ops): where it sits among them in canvas order.
    size_t position = 0;
    int node_id = -1;
    std::string node_name;

    // Contrast: cv::createCLAHE.
    float clip_limit = 2.0f;
    int tile_grid = 8;
    CvContrastColour colour = CvContrastColour::Lightness;
    // Denoise: fastNlMeansDenoising(Colored).
    float strength = 10.0f;
    float colour_strength = 10.0f;
    int patch_size = 7;
    int search_size = 21;
    // Sharpen: unsharp mask.
    float amount = 1.0f;
    float radius = 1.0f;
    float threshold = 0.0f;
    // Edges.
    CvEdgeMethod edge_method = CvEdgeMethod::Canny;
    float threshold1 = 50.0f;
    float threshold2 = 150.0f;
    int edge_kernel = 3;
    // Smooth.
    CvSmoothMethod smooth_method = CvSmoothMethod::Median;
    int kernel_size = 5;
    float sigma_color = 75.0f;
    float sigma_space = 75.0f;
    // Warp: points are fractions of the image, x then y; three points use the first six.
    CvWarpMethod warp_method = CvWarpMethod::FourCorners;
    CvWarpInterpolation interpolation = CvWarpInterpolation::Linear;
    std::array<float, 8> source_points = {0, 0, 1, 0, 1, 1, 0, 1};
    std::array<float, 8> target_points = {0, 0, 1, 0, 1, 1, 0, 1};
    float center_x = 0.5f, center_y = 0.5f, max_radius = 0.5f;
    bool inverse = false;
    int out_width = 0, out_height = 0;  // 0: the input's size
    // Pyramid.
    CvPyramidOperation pyramid = CvPyramidOperation::Down;
    int levels = 1;
};

// "" when the step can run on `input`, otherwise why not.
std::string ValidateCvImageStep(const CvImageStep& step, const image::ImageShape& input);
image::ImageShape CvImageShapeAfter(const CvImageStep& step, const image::ImageShape& input);
// "Image Contrast (CLAHE) 2 / 8", "Image Edges (Canny 50 / 150)".
std::string CvImageStepText(const CvImageStep& step);
// Every setting the result depends on, for the prepared set's key; the
// header keeps it, with "text" for Profile's cached sets.
nlohmann::json CvImageStepIdentity(const CvImageStep& step);
// One image through the step: `in` is uint8 [H, W, C] of `input`, `out` gets
// CvImageShapeAfter(step, input).Size() bytes. False, with the reason, when
// OpenCV fails. Thread-safe.
bool ApplyCvImageStep(const CvImageStep& step, const image::ImageShape& input, const uint8_t* in, uint8_t* out,
                      std::string* error = nullptr);

// A step done once with the images: a device transform or an OpenCV step.
using FixedImageStep = std::variant<image::ImageOp, CvImageStep>;

// What a device transform is, in words: "Center Crop 64 x 64", "Gaussian
// Blur 5 x 5, sigma 1", "Horizontal Flip (p 0.5)", "Rotate up to 15 degrees".
std::string ImageOpText(const image::ImageOp& op);
// A device transform's ImageOpText, an OpenCV step's CvImageStepText.
std::string FixedImageStepText(const FixedImageStep& step);

// The steps of a compiled graph in canvas order: the device transforms with
// the OpenCV steps placed at their positions.
std::vector<FixedImageStep> MergeImageSteps(const image::ImageAugmentation& compiled,
                                            const std::vector<CvImageStep>& cv_steps);
// The [H, W, C] a model receives from `decoded` through the compiled steps.
image::ImageShape CompiledImageShape(const image::ImageAugmentation& compiled,
                                     const std::vector<CvImageStep>& cv_steps, const image::ImageShape& decoded);

}  // namespace cyxwiz
