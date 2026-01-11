#pragma once

#include <vector>
#include <array>
#include <nlohmann/json.hpp>

namespace cyxwiz {

/**
 * Coordinate transform types
 */
enum class CoordinateTransform {
    Perspective,  // 4-point perspective warp (homography)
    Affine,       // 3-point affine warp (preserves parallel lines)
    LogPolar,     // Log-polar coordinate transform (rotation-invariant features)
    LinearPolar   // Linear-polar coordinate transform
};

/**
 * Interpolation methods for warping
 */
enum class WarpInterpolation {
    Nearest,   // cv::INTER_NEAREST - Fastest, pixelated
    Linear,    // cv::INTER_LINEAR - Good balance (default)
    Cubic,     // cv::INTER_CUBIC - Higher quality, slower
    Lanczos    // cv::INTER_LANCZOS4 - Highest quality, slowest
};

/**
 * Configuration for perspective/coordinate transforms
 */
struct PerspectiveConfig {
    bool enabled = false;
    CoordinateTransform type = CoordinateTransform::Perspective;
    WarpInterpolation interpolation = WarpInterpolation::Linear;

    // For Perspective/Affine: normalized source/destination points (0-1 range)
    // Format: [x0, y0, x1, y1, x2, y2, x3, y3] for 4 corners (perspective)
    // Format: [x0, y0, x1, y1, x2, y2, 0, 0] for 3 points (affine)
    std::array<float, 8> src_points = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
    std::array<float, 8> dst_points = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};

    // For Polar transforms: center point (normalized 0-1)
    float center_x = 0.5f;
    float center_y = 0.5f;
    float max_radius = 0.5f;  // Maximum radius as fraction of min(width, height)
    bool inverse_polar = false;  // If true, convert from polar back to Cartesian

    // Output dimensions (0 = same as input)
    int output_width = 0;
    int output_height = 0;

    nlohmann::json ToJson() const;
    static PerspectiveConfig FromJson(const nlohmann::json& j);

    bool operator==(const PerspectiveConfig& other) const;
    bool operator!=(const PerspectiveConfig& other) const { return !(*this == other); }
};

/**
 * Perspective and coordinate transform operations using OpenCV
 *
 * These transforms are useful for:
 * - Document scanning (perspective correction)
 * - Data augmentation (random perspective distortions)
 * - Rotation-invariant feature extraction (polar transforms)
 * - Image registration
 */
class PerspectiveTransform {
public:
    /**
     * Apply transform using config
     * @param image_data Float image data (0-1 range) [modified in-place]
     * @param width Input/output image width (may change)
     * @param height Input/output image height (may change)
     * @param channels Number of channels (1=grayscale, 3=RGB)
     * @param config Transform configuration
     * @return true if successful, false otherwise
     */
    static bool Apply(
        std::vector<float>& image_data,
        int& width, int& height, int channels,
        const PerspectiveConfig& config
    );

    /**
     * Warp perspective using 4 corner points
     * @param src_pts Source corner points [x0,y0, x1,y1, x2,y2, x3,y3] (normalized 0-1)
     * @param dst_pts Destination corner points (normalized 0-1)
     * @param out_w Output width (0 = same as input)
     * @param out_h Output height (0 = same as input)
     */
    static bool WarpPerspective(
        std::vector<float>& image_data,
        int& width, int& height, int channels,
        const std::array<float, 8>& src_pts,
        const std::array<float, 8>& dst_pts,
        int out_w = 0, int out_h = 0,
        WarpInterpolation interp = WarpInterpolation::Linear
    );

    /**
     * Warp affine using 3 points
     * @param src_pts Source points [x0,y0, x1,y1, x2,y2, 0,0] (normalized 0-1)
     * @param dst_pts Destination points (normalized 0-1)
     */
    static bool WarpAffine(
        std::vector<float>& image_data,
        int& width, int& height, int channels,
        const std::array<float, 8>& src_pts,
        const std::array<float, 8>& dst_pts,
        int out_w = 0, int out_h = 0,
        WarpInterpolation interp = WarpInterpolation::Linear
    );

    /**
     * Apply log-polar coordinate transform
     * Useful for rotation-invariant features (rotation becomes translation)
     * @param center_x Center X (normalized 0-1)
     * @param center_y Center Y (normalized 0-1)
     * @param max_radius Maximum radius as fraction of min(w,h)
     * @param inverse If true, convert from polar back to Cartesian
     */
    static bool LogPolar(
        std::vector<float>& image_data,
        int& width, int& height, int channels,
        float center_x = 0.5f, float center_y = 0.5f,
        float max_radius = 0.5f, bool inverse = false
    );

    /**
     * Apply linear-polar coordinate transform
     * Similar to log-polar but with linear radius mapping
     */
    static bool LinearPolar(
        std::vector<float>& image_data,
        int& width, int& height, int channels,
        float center_x = 0.5f, float center_y = 0.5f,
        float max_radius = 0.5f, bool inverse = false
    );

    /**
     * Rotate image by arbitrary angle
     * @param angle Rotation angle in degrees (positive = counter-clockwise)
     * @param scale Scale factor (1.0 = no scaling)
     * @param crop If true, crop to original size; if false, expand canvas
     */
    static bool Rotate(
        std::vector<float>& image_data,
        int& width, int& height, int channels,
        float angle, float scale = 1.0f, bool crop = true
    );

    /**
     * Auto-detect document corners using edge detection and Hough lines
     * Returns 4 corner points in normalized coordinates, or identity if detection fails
     */
    static std::array<float, 8> DetectDocumentCorners(
        const std::vector<float>& image_data,
        int width, int height, int channels
    );

private:
    static bool ValidateInput(const std::vector<float>& image_data,
                              int width, int height, int channels);
    static int GetCVInterpolation(WarpInterpolation interp);
};

// Enum string conversions
std::string CoordinateTransformToString(CoordinateTransform type);
CoordinateTransform StringToCoordinateTransform(const std::string& str);
std::string WarpInterpolationToString(WarpInterpolation interp);
WarpInterpolation StringToWarpInterpolation(const std::string& str);

} // namespace cyxwiz
