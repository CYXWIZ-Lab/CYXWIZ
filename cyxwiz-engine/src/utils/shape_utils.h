#pragma once

#include <vector>
#include <string>
#include <cstdint>

namespace cyxwiz {

/**
 * 2D Point structure
 */
struct Point2D {
    float x = 0.0f;
    float y = 0.0f;
};

/**
 * Rotated rectangle (for minAreaRect, fitEllipse)
 */
struct RotatedRect {
    float center_x = 0.0f;
    float center_y = 0.0f;
    float width = 0.0f;      // Full width of rectangle
    float height = 0.0f;     // Full height of rectangle
    float angle = 0.0f;      // Rotation angle in degrees

    // Get corner points
    std::vector<Point2D> GetCorners() const;
};

/**
 * Ellipse parameters
 */
struct Ellipse {
    float center_x = 0.0f;
    float center_y = 0.0f;
    float axis_major = 0.0f;   // Major axis length
    float axis_minor = 0.0f;   // Minor axis length
    float angle = 0.0f;        // Rotation angle in degrees
};

/**
 * Circle parameters (from HoughCircles)
 */
struct Circle {
    float center_x = 0.0f;
    float center_y = 0.0f;
    float radius = 0.0f;
};

/**
 * Line segment (from HoughLines)
 */
struct LineSegment {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;

    // Line equation: rho = x*cos(theta) + y*sin(theta)
    float rho = 0.0f;
    float theta = 0.0f;  // In radians
};

/**
 * Contour shape descriptors
 */
struct ShapeDescriptor {
    float area = 0.0f;
    float perimeter = 0.0f;
    float circularity = 0.0f;     // 4*pi*area / perimeter^2 (1.0 = perfect circle)
    float aspect_ratio = 0.0f;    // width / height of bounding rect
    float extent = 0.0f;          // area / bounding_rect_area
    float solidity = 0.0f;        // area / convex_hull_area
    float compactness = 0.0f;     // perimeter^2 / area
    bool is_convex = false;
    int vertex_count = 0;         // After polygon approximation
};

/**
 * Moments of a contour/region
 */
struct ShapeMoments {
    // Spatial moments
    double m00 = 0, m10 = 0, m01 = 0;
    double m20 = 0, m11 = 0, m02 = 0;
    double m30 = 0, m21 = 0, m12 = 0, m03 = 0;

    // Central moments
    double mu20 = 0, mu11 = 0, mu02 = 0;
    double mu30 = 0, mu21 = 0, mu12 = 0, mu03 = 0;

    // Normalized central moments (scale invariant)
    double nu20 = 0, nu11 = 0, nu02 = 0;
    double nu30 = 0, nu21 = 0, nu12 = 0, nu03 = 0;

    // Hu moments (rotation, scale, translation invariant)
    double hu[7] = {0};

    // Centroid
    float centroid_x = 0.0f;
    float centroid_y = 0.0f;
};

/**
 * Shape analysis utilities using OpenCV
 *
 * Provides tools for:
 * - Convex hull computation
 * - Ellipse and rectangle fitting
 * - Hough line and circle detection
 * - Shape descriptors and moments
 * - Polygon approximation
 */
class ShapeUtils {
public:
    /**
     * Compute convex hull of a contour
     * @param contour Input contour points
     * @param clockwise If true, output hull is clockwise
     * @return Convex hull points
     */
    static std::vector<Point2D> ConvexHull(
        const std::vector<Point2D>& contour,
        bool clockwise = false
    );

    /**
     * Fit ellipse to a contour (requires >= 5 points)
     * @param contour Input contour points
     * @return Fitted ellipse parameters
     */
    static Ellipse FitEllipse(const std::vector<Point2D>& contour);

    /**
     * Find minimum area bounding rectangle
     * @param contour Input contour points
     * @return Minimum area rotated rectangle
     */
    static RotatedRect MinAreaRect(const std::vector<Point2D>& contour);

    /**
     * Find minimum enclosing circle
     * @param contour Input contour points
     * @return Minimum enclosing circle
     */
    static Circle MinEnclosingCircle(const std::vector<Point2D>& contour);

    /**
     * Detect lines using Hough transform
     * @param edge_image Binary edge image (0-1 values)
     * @param width Image width
     * @param height Image height
     * @param rho Distance resolution in pixels
     * @param theta Angle resolution in radians
     * @param threshold Accumulator threshold
     * @param min_line_length Minimum line length (for probabilistic)
     * @param max_line_gap Maximum gap between line segments
     * @param probabilistic Use probabilistic Hough transform
     * @return Detected line segments
     */
    static std::vector<LineSegment> HoughLines(
        const std::vector<float>& edge_image,
        int width, int height,
        double rho = 1.0,
        double theta = 0.017453,  // ~1 degree
        int threshold = 50,
        double min_line_length = 50.0,
        double max_line_gap = 10.0,
        bool probabilistic = true
    );

    /**
     * Detect circles using Hough transform
     * @param gray_image Grayscale image (0-1 values)
     * @param width Image width
     * @param height Image height
     * @param dp Inverse ratio of accumulator resolution
     * @param min_dist Minimum distance between circle centers
     * @param param1 Higher Canny threshold
     * @param param2 Accumulator threshold
     * @param min_radius Minimum circle radius
     * @param max_radius Maximum circle radius (0 = no max)
     * @return Detected circles
     */
    static std::vector<Circle> HoughCircles(
        const std::vector<float>& gray_image,
        int width, int height,
        double dp = 1.0,
        double min_dist = 20.0,
        double param1 = 100.0,
        double param2 = 30.0,
        int min_radius = 0,
        int max_radius = 0
    );

    /**
     * Approximate contour with polygon
     * @param contour Input contour points
     * @param epsilon Approximation accuracy (% of arc length, default 2%)
     * @param closed If true, approximated curve is closed
     * @return Approximated polygon vertices
     */
    static std::vector<Point2D> ApproxPolygon(
        const std::vector<Point2D>& contour,
        double epsilon = 0.02,
        bool closed = true
    );

    /**
     * Compute shape descriptors for a contour
     * @param contour Input contour points
     * @return Shape descriptor struct
     */
    static ShapeDescriptor ComputeDescriptors(const std::vector<Point2D>& contour);

    /**
     * Compute moments for a contour
     * @param contour Input contour points
     * @param binary_image If true, all non-zero pixels are treated as 1
     * @return Shape moments including Hu moments
     */
    static ShapeMoments ComputeMoments(const std::vector<Point2D>& contour);

    /**
     * Compute moments for a binary mask
     * @param binary_mask Binary mask (0-1 values)
     * @param width Image width
     * @param height Image height
     * @return Shape moments including Hu moments
     */
    static ShapeMoments ComputeMoments(
        const std::vector<float>& binary_mask,
        int width, int height
    );

    /**
     * Match two shapes using Hu moments
     * @param contour1 First contour
     * @param contour2 Second contour
     * @param method Comparison method (1, 2, or 3)
     * @return Match score (lower = more similar)
     */
    static double MatchShapes(
        const std::vector<Point2D>& contour1,
        const std::vector<Point2D>& contour2,
        int method = 1
    );

    /**
     * Compute contour area
     * @param contour Input contour points
     * @param oriented If true, returns signed area
     * @return Area in pixels
     */
    static double ContourArea(const std::vector<Point2D>& contour, bool oriented = false);

    /**
     * Compute contour perimeter (arc length)
     * @param contour Input contour points
     * @param closed If true, contour is closed
     * @return Perimeter in pixels
     */
    static double ContourPerimeter(const std::vector<Point2D>& contour, bool closed = true);

    /**
     * Check if a point is inside a contour
     * @param contour Input contour points
     * @param point Point to test
     * @return >0 inside, <0 outside, =0 on contour
     */
    static double PointPolygonTest(const std::vector<Point2D>& contour, Point2D point);

    /**
     * Check if contour is convex
     * @param contour Input contour points
     * @return true if convex
     */
    static bool IsContourConvex(const std::vector<Point2D>& contour);
};

} // namespace cyxwiz
