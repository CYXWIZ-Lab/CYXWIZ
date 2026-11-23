#pragma once

// The OpenCV image steps (TOFIX178 1.3f): Image Contrast (CLAHE), Image
// Denoise, Image Sharpen, Image Edges, Image Smooth, Image Warp, Image
// Pyramid and Image Filter (1.10a). Each gives every image the same result in every epoch, so it is
// done once when the images are prepared, on the CPU workers, in canvas
// order with the device transforms that are done once (Center Crop,
// Grayscale, Gaussian Blur, Morphology). The prepared set is uint8, so each
// step reads and writes uint8 [H, W, C] images. C may be any count: on images
// that are neither grey nor RGB a step runs on each channel alone (Image
// Filter always does). No OpenCV in this header;
// ApplyCvImageStep lives in image_cv_steps_apply.cpp.

#include "image_channel_layout.h"

#include <cyxwiz/image_augmentation.h>

#include <nlohmann/json.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace cyxwiz {

enum class CvImageStepKind { Contrast, Denoise, Sharpen, Edges, Smooth, Warp, Pyramid, Filter, Gabor };
enum class CvContrastColour { Lightness, EachChannel };
enum class CvEdgeMethod { Canny, Sobel, Laplacian, Scharr, LoG, DoG };
// Sobel and Scharr: both directions' mean size, one direction, or the angle.
enum class CvEdgeDirection { Magnitude, X, Y, Angle };
enum class CvSmoothMethod { Box, Median, Bilateral };
enum class CvWarpMethod { FourCorners, ThreePoints, LogPolar, LinearPolar };
enum class CvWarpInterpolation { Nearest, Linear, Cubic, Lanczos };
enum class CvPyramidOperation { Down, Up, Detail };
// Image Filter: a typed kernel or a classic one; Kirsch and Robinson have
// eight directions (the side an edge is brighter on).
enum class CvFilterPreset {
    Custom, Identity, Box, Sharpen, Emboss, Outline, PrewittX, PrewittY, RobertsX, RobertsY, Kirsch, Robinson
};
enum class CvFilterDirection { N, NE, E, SE, S, SW, W, NW };
enum class CvFilterBorder { Reflect101, Reflect, Replicate, Zero };
// What becomes of a response outside [0, 1]: cut, its size, or around mid-grey.
enum class CvFilterOutput { Clip, Absolute, Offset };
// Image Gabor: both phases' size (the texture's strength whatever its phase),
// or one phase stored around mid-grey.
enum class CvGaborResponse { Energy, Even, Odd };

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
    CvEdgeDirection edge_direction = CvEdgeDirection::Magnitude;
    float edge_sigma = 1.0f;   // LoG: the smoothing; DoG: the finer blur
    float edge_sigma2 = 1.6f;  // DoG: the coarser blur
    bool edge_keep_image = false;  // the image's channels, then the edge map as one more
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
    // Filter: cv::filter2D (correlation, the kernel's centre on the pixel).
    CvFilterPreset filter_preset = CvFilterPreset::Sharpen;
    CvFilterDirection filter_direction = CvFilterDirection::N;
    std::string filter_kernel = "0 -1 0; -1 5 -1; 0 -1 0";  // Custom: rows split by ';'
    bool filter_normalize = false;  // divide by the kernel's sum when it is not 0
    CvFilterBorder filter_border = CvFilterBorder::Reflect101;
    CvFilterOutput filter_output = CvFilterOutput::Clip;
    // Gabor: a bank of kernels over the grey image, one channel each.
    int gabor_orientations = 4;              // 0, 180 / n, ... degrees
    std::string gabor_wavelengths = "3, 5";  // pixels per cycle
    float gabor_bandwidth = 1.0f;            // octaves: sets the envelope's sigma from the wavelength
    float gabor_aspect = 0.5f;               // the envelope's width across the stripes over its length
    CvGaborResponse gabor_response = CvGaborResponse::Energy;
    bool gabor_keep_image = false;           // the image's channels, then the bank's
};

// The kernel an Image Filter step applies: its preset (turned to its
// direction) or its typed text, divided by its sum when the step says so.
struct FilterKernel {
    int rows = 0;
    int cols = 0;
    std::vector<float> values;  // row-major
    float Sum() const;
};
inline constexpr int kFilterKernelMaxSide = 31;
// False, with the reason, when the typed kernel cannot be read.
bool ResolveFilterKernel(const CvImageStep& step, FilterKernel& kernel, std::string* why = nullptr);

// One filter of an Image Gabor step's bank. Its even (cosine) and odd (sine)
// kernels each have zero mean and absolute values that sum to 1, so a
// response to pixels in [0, 1] stays within -0.5..0.5.
struct GaborFilter {
    float degrees = 0.0f;     // the stripes' normal: 0 answers to vertical stripes
    float wavelength = 0.0f;  // pixels per cycle
    float sigma = 0.0f;       // the window across the stripes; along them it is sigma / aspect
    int side = 0;             // the kernel's rows and columns: two of the longer sigma each way
    FilterKernel even;        // empty until the kernels are built
    FilterKernel odd;
};
inline constexpr size_t kGaborMaxChannels = 32;
inline constexpr int kGaborKernelMaxSide = 255;
// The bank in channel order (each wavelength, its orientations in turn),
// without its kernels: cheap, for names, shapes and checks. False, with the
// reason, when the settings cannot be read.
bool DescribeGaborBank(const CvImageStep& step, std::vector<GaborFilter>& bank, std::string* why = nullptr);
// The bank with its kernels built, for running it.
bool ResolveGaborBank(const CvImageStep& step, std::vector<GaborFilter>& bank, std::string* why = nullptr);
// "gabor 45 deg 4 px" per channel; empty when the settings cannot be read.
std::vector<std::string> GaborChannelNames(const CvImageStep& step);

// "" when the step can run on `input`, otherwise why not.
std::string ValidateCvImageStep(const CvImageStep& step, const image::ImageShape& input);
image::ImageShape CvImageShapeAfter(const CvImageStep& step, const image::ImageShape& input);
// "CLAHE 2.0 / 8", "Edges (Canny 50 / 150)", "Filter (Sharpen 3 x 3)".
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
// The layout after one step that turns `input` into `output_channels` channels.
ImageChannelLayout ChannelLayoutAfter(const FixedImageStep& step, const ImageChannelLayout& input,
                                      size_t output_channels);
// The layout a model receives from `decoded` through the compiled steps.
ImageChannelLayout CompiledImageChannels(const image::ImageAugmentation& compiled,
                                         const std::vector<CvImageStep>& cv_steps,
                                         const image::ImageShape& decoded);
// The [H, W, C] a model receives from `decoded` through the compiled steps.
image::ImageShape CompiledImageShape(const image::ImageAugmentation& compiled,
                                     const std::vector<CvImageStep>& cv_steps, const image::ImageShape& decoded);

}  // namespace cyxwiz
