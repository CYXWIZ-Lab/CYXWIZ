#include "shape_utils.h"
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <cmath>
#include <algorithm>

namespace cyxwiz {

// ============================================================================
// RotatedRect Implementation
// ============================================================================

std::vector<Point2D> RotatedRect::GetCorners() const {
    std::vector<Point2D> corners(4);
    float angle_rad = angle * static_cast<float>(CV_PI) / 180.0f;
    float cos_a = std::cos(angle_rad);
    float sin_a = std::sin(angle_rad);
    float hw = width / 2.0f;
    float hh = height / 2.0f;

    // Rotate corner offsets
    corners[0] = {center_x + (-hw * cos_a - (-hh) * sin_a),
                  center_y + (-hw * sin_a + (-hh) * cos_a)};
    corners[1] = {center_x + (hw * cos_a - (-hh) * sin_a),
                  center_y + (hw * sin_a + (-hh) * cos_a)};
    corners[2] = {center_x + (hw * cos_a - hh * sin_a),
                  center_y + (hw * sin_a + hh * cos_a)};
    corners[3] = {center_x + (-hw * cos_a - hh * sin_a),
                  center_y + (-hw * sin_a + hh * cos_a)};

    return corners;
}

// ============================================================================
// Helper Functions (file-local)
// ============================================================================

namespace {

std::vector<cv::Point2f> ToCV(const std::vector<Point2D>& points) {
    std::vector<cv::Point2f> cv_points;
    cv_points.reserve(points.size());
    for (const auto& p : points) {
        cv_points.emplace_back(p.x, p.y);
    }
    return cv_points;
}

std::vector<Point2D> FromCV(const std::vector<cv::Point2f>& points) {
    std::vector<Point2D> result;
    result.reserve(points.size());
    for (const auto& p : points) {
        result.push_back({p.x, p.y});
    }
    return result;
}

} // anonymous namespace

// ============================================================================
// ShapeUtils Implementation
// ============================================================================

std::vector<Point2D> ShapeUtils::ConvexHull(
    const std::vector<Point2D>& contour,
    bool clockwise
) {
    if (contour.size() < 3) {
        spdlog::warn("ShapeUtils: ConvexHull requires at least 3 points");
        return contour;
    }

    try {
        auto cv_points = ToCV(contour);
        std::vector<cv::Point2f> hull;
        cv::convexHull(cv_points, hull, clockwise);
        return FromCV(hull);
    } catch (const cv::Exception& e) {
        spdlog::error("ShapeUtils: OpenCV exception in ConvexHull: {}", e.what());
        return {};
    }
}

Ellipse ShapeUtils::FitEllipse(const std::vector<Point2D>& contour) {
    Ellipse result;

    if (contour.size() < 5) {
        spdlog::warn("ShapeUtils: FitEllipse requires at least 5 points");
        return result;
    }

    try {
        auto cv_points = ToCV(contour);
        cv::RotatedRect ellipse = cv::fitEllipse(cv_points);

        result.center_x = ellipse.center.x;
        result.center_y = ellipse.center.y;
        result.axis_major = std::max(ellipse.size.width, ellipse.size.height);
        result.axis_minor = std::min(ellipse.size.width, ellipse.size.height);
        result.angle = ellipse.angle;

        return result;
    } catch (const cv::Exception& e) {
        spdlog::error("ShapeUtils: OpenCV exception in FitEllipse: {}", e.what());
        return result;
    }
}

RotatedRect ShapeUtils::MinAreaRect(const std::vector<Point2D>& contour) {
    RotatedRect result;

    if (contour.size() < 3) {
        spdlog::warn("ShapeUtils: MinAreaRect requires at least 3 points");
        return result;
    }

    try {
        auto cv_points = ToCV(contour);
        cv::RotatedRect rect = cv::minAreaRect(cv_points);

        result.center_x = rect.center.x;
        result.center_y = rect.center.y;
        result.width = rect.size.width;
        result.height = rect.size.height;
        result.angle = rect.angle;

        return result;
    } catch (const cv::Exception& e) {
        spdlog::error("ShapeUtils: OpenCV exception in MinAreaRect: {}", e.what());
        return result;
    }
}

Circle ShapeUtils::MinEnclosingCircle(const std::vector<Point2D>& contour) {
    Circle result;

    if (contour.size() < 3) {
        spdlog::warn("ShapeUtils: MinEnclosingCircle requires at least 3 points");
        return result;
    }

    try {
        auto cv_points = ToCV(contour);
        cv::Point2f center;
        float radius;
        cv::minEnclosingCircle(cv_points, center, radius);

        result.center_x = center.x;
        result.center_y = center.y;
        result.radius = radius;

        return result;
    } catch (const cv::Exception& e) {
        spdlog::error("ShapeUtils: OpenCV exception in MinEnclosingCircle: {}", e.what());
        return result;
    }
}

std::vector<LineSegment> ShapeUtils::HoughLines(
    const std::vector<float>& edge_image,
    int width, int height,
    double rho, double theta, int threshold,
    double min_line_length, double max_line_gap,
    bool probabilistic
) {
    std::vector<LineSegment> lines;

    if (edge_image.size() != static_cast<size_t>(width) * height) {
        spdlog::error("ShapeUtils: Edge image size mismatch");
        return lines;
    }

    try {
        // Convert to 8-bit
        cv::Mat edges(height, width, CV_32FC1, const_cast<float*>(edge_image.data()));
        cv::Mat edges8u;
        edges.convertTo(edges8u, CV_8UC1, 255.0);

        if (probabilistic) {
            // Probabilistic Hough Transform - returns line segments
            std::vector<cv::Vec4i> cv_lines;
            cv::HoughLinesP(edges8u, cv_lines, rho, theta, threshold,
                           min_line_length, max_line_gap);

            for (const auto& l : cv_lines) {
                LineSegment seg;
                seg.x1 = static_cast<float>(l[0]);
                seg.y1 = static_cast<float>(l[1]);
                seg.x2 = static_cast<float>(l[2]);
                seg.y2 = static_cast<float>(l[3]);

                // Calculate rho and theta for the line
                float dx = seg.x2 - seg.x1;
                float dy = seg.y2 - seg.y1;
                seg.theta = std::atan2(dy, dx);
                seg.rho = seg.x1 * std::cos(seg.theta) + seg.y1 * std::sin(seg.theta);

                lines.push_back(seg);
            }
        } else {
            // Standard Hough Transform - returns (rho, theta) pairs
            std::vector<cv::Vec2f> cv_lines;
            cv::HoughLines(edges8u, cv_lines, rho, theta, threshold);

            for (const auto& l : cv_lines) {
                LineSegment seg;
                seg.rho = l[0];
                seg.theta = l[1];

                // Convert to line segment endpoints (extend to image bounds)
                float a = std::cos(seg.theta);
                float b = std::sin(seg.theta);
                float x0 = a * seg.rho;
                float y0 = b * seg.rho;
                float scale = 1000.0f;

                seg.x1 = x0 + scale * (-b);
                seg.y1 = y0 + scale * a;
                seg.x2 = x0 - scale * (-b);
                seg.y2 = y0 - scale * a;

                lines.push_back(seg);
            }
        }

        spdlog::debug("ShapeUtils: HoughLines found {} lines", lines.size());
        return lines;

    } catch (const cv::Exception& e) {
        spdlog::error("ShapeUtils: OpenCV exception in HoughLines: {}", e.what());
        return lines;
    }
}

std::vector<Circle> ShapeUtils::HoughCircles(
    const std::vector<float>& gray_image,
    int width, int height,
    double dp, double min_dist,
    double param1, double param2,
    int min_radius, int max_radius
) {
    std::vector<Circle> circles;

    if (gray_image.size() != static_cast<size_t>(width) * height) {
        spdlog::error("ShapeUtils: Gray image size mismatch");
        return circles;
    }

    try {
        // Convert to 8-bit
        cv::Mat gray(height, width, CV_32FC1, const_cast<float*>(gray_image.data()));
        cv::Mat gray8u;
        gray.convertTo(gray8u, CV_8UC1, 255.0);

        std::vector<cv::Vec3f> cv_circles;
        cv::HoughCircles(gray8u, cv_circles, cv::HOUGH_GRADIENT, dp, min_dist,
                        param1, param2, min_radius, max_radius);

        for (const auto& c : cv_circles) {
            Circle circle;
            circle.center_x = c[0];
            circle.center_y = c[1];
            circle.radius = c[2];
            circles.push_back(circle);
        }

        spdlog::debug("ShapeUtils: HoughCircles found {} circles", circles.size());
        return circles;

    } catch (const cv::Exception& e) {
        spdlog::error("ShapeUtils: OpenCV exception in HoughCircles: {}", e.what());
        return circles;
    }
}

std::vector<Point2D> ShapeUtils::ApproxPolygon(
    const std::vector<Point2D>& contour,
    double epsilon,
    bool closed
) {
    if (contour.size() < 3) {
        return contour;
    }

    try {
        auto cv_points = ToCV(contour);

        // Calculate epsilon based on arc length
        double arc_length = cv::arcLength(cv_points, closed);
        double eps = epsilon * arc_length;

        std::vector<cv::Point2f> approx;
        cv::approxPolyDP(cv_points, approx, eps, closed);

        return FromCV(approx);

    } catch (const cv::Exception& e) {
        spdlog::error("ShapeUtils: OpenCV exception in ApproxPolygon: {}", e.what());
        return {};
    }
}

ShapeDescriptor ShapeUtils::ComputeDescriptors(const std::vector<Point2D>& contour) {
    ShapeDescriptor desc;

    if (contour.size() < 3) {
        return desc;
    }

    try {
        auto cv_points = ToCV(contour);

        // Area and perimeter
        desc.area = static_cast<float>(cv::contourArea(cv_points));
        desc.perimeter = static_cast<float>(cv::arcLength(cv_points, true));

        // Circularity: 4*pi*area / perimeter^2
        if (desc.perimeter > 0) {
            desc.circularity = 4.0f * static_cast<float>(CV_PI) * desc.area /
                              (desc.perimeter * desc.perimeter);
        }

        // Bounding rect
        cv::Rect bbox = cv::boundingRect(cv_points);
        if (bbox.height > 0) {
            desc.aspect_ratio = static_cast<float>(bbox.width) / bbox.height;
        }

        // Extent: area / bounding_rect_area
        float bbox_area = static_cast<float>(bbox.width * bbox.height);
        if (bbox_area > 0) {
            desc.extent = desc.area / bbox_area;
        }

        // Convex hull for solidity
        std::vector<cv::Point2f> hull;
        cv::convexHull(cv_points, hull);
        float hull_area = static_cast<float>(cv::contourArea(hull));
        if (hull_area > 0) {
            desc.solidity = desc.area / hull_area;
        }

        // Compactness: perimeter^2 / area
        if (desc.area > 0) {
            desc.compactness = (desc.perimeter * desc.perimeter) / desc.area;
        }

        // Convexity test
        desc.is_convex = cv::isContourConvex(cv_points);

        // Vertex count after approximation
        auto approx = ApproxPolygon(contour, 0.02, true);
        desc.vertex_count = static_cast<int>(approx.size());

        return desc;

    } catch (const cv::Exception& e) {
        spdlog::error("ShapeUtils: OpenCV exception in ComputeDescriptors: {}", e.what());
        return desc;
    }
}

ShapeMoments ShapeUtils::ComputeMoments(const std::vector<Point2D>& contour) {
    ShapeMoments result;

    if (contour.size() < 3) {
        return result;
    }

    try {
        auto cv_points = ToCV(contour);
        cv::Moments m = cv::moments(cv_points);

        // Spatial moments
        result.m00 = m.m00;
        result.m10 = m.m10;
        result.m01 = m.m01;
        result.m20 = m.m20;
        result.m11 = m.m11;
        result.m02 = m.m02;
        result.m30 = m.m30;
        result.m21 = m.m21;
        result.m12 = m.m12;
        result.m03 = m.m03;

        // Central moments
        result.mu20 = m.mu20;
        result.mu11 = m.mu11;
        result.mu02 = m.mu02;
        result.mu30 = m.mu30;
        result.mu21 = m.mu21;
        result.mu12 = m.mu12;
        result.mu03 = m.mu03;

        // Normalized central moments
        result.nu20 = m.nu20;
        result.nu11 = m.nu11;
        result.nu02 = m.nu02;
        result.nu30 = m.nu30;
        result.nu21 = m.nu21;
        result.nu12 = m.nu12;
        result.nu03 = m.nu03;

        // Hu moments
        cv::HuMoments(m, result.hu);

        // Centroid
        if (m.m00 > 0) {
            result.centroid_x = static_cast<float>(m.m10 / m.m00);
            result.centroid_y = static_cast<float>(m.m01 / m.m00);
        }

        return result;

    } catch (const cv::Exception& e) {
        spdlog::error("ShapeUtils: OpenCV exception in ComputeMoments: {}", e.what());
        return result;
    }
}

ShapeMoments ShapeUtils::ComputeMoments(
    const std::vector<float>& binary_mask,
    int width, int height
) {
    ShapeMoments result;

    if (binary_mask.size() != static_cast<size_t>(width) * height) {
        spdlog::error("ShapeUtils: Binary mask size mismatch");
        return result;
    }

    try {
        cv::Mat mask(height, width, CV_32FC1, const_cast<float*>(binary_mask.data()));
        cv::Mat mask8u;
        mask.convertTo(mask8u, CV_8UC1, 255.0);

        cv::Moments m = cv::moments(mask8u, true);  // binary_image = true

        // Copy all moments
        result.m00 = m.m00;
        result.m10 = m.m10;
        result.m01 = m.m01;
        result.m20 = m.m20;
        result.m11 = m.m11;
        result.m02 = m.m02;
        result.m30 = m.m30;
        result.m21 = m.m21;
        result.m12 = m.m12;
        result.m03 = m.m03;

        result.mu20 = m.mu20;
        result.mu11 = m.mu11;
        result.mu02 = m.mu02;
        result.mu30 = m.mu30;
        result.mu21 = m.mu21;
        result.mu12 = m.mu12;
        result.mu03 = m.mu03;

        result.nu20 = m.nu20;
        result.nu11 = m.nu11;
        result.nu02 = m.nu02;
        result.nu30 = m.nu30;
        result.nu21 = m.nu21;
        result.nu12 = m.nu12;
        result.nu03 = m.nu03;

        cv::HuMoments(m, result.hu);

        if (m.m00 > 0) {
            result.centroid_x = static_cast<float>(m.m10 / m.m00);
            result.centroid_y = static_cast<float>(m.m01 / m.m00);
        }

        return result;

    } catch (const cv::Exception& e) {
        spdlog::error("ShapeUtils: OpenCV exception in ComputeMoments (mask): {}", e.what());
        return result;
    }
}

double ShapeUtils::MatchShapes(
    const std::vector<Point2D>& contour1,
    const std::vector<Point2D>& contour2,
    int method
) {
    if (contour1.size() < 3 || contour2.size() < 3) {
        return std::numeric_limits<double>::max();
    }

    try {
        auto cv_points1 = ToCV(contour1);
        auto cv_points2 = ToCV(contour2);

        int cv_method;
        switch (method) {
            case 1: cv_method = cv::CONTOURS_MATCH_I1; break;
            case 2: cv_method = cv::CONTOURS_MATCH_I2; break;
            case 3: cv_method = cv::CONTOURS_MATCH_I3; break;
            default: cv_method = cv::CONTOURS_MATCH_I1;
        }

        return cv::matchShapes(cv_points1, cv_points2, cv_method, 0.0);

    } catch (const cv::Exception& e) {
        spdlog::error("ShapeUtils: OpenCV exception in MatchShapes: {}", e.what());
        return std::numeric_limits<double>::max();
    }
}

double ShapeUtils::ContourArea(const std::vector<Point2D>& contour, bool oriented) {
    if (contour.size() < 3) return 0.0;

    try {
        auto cv_points = ToCV(contour);
        return cv::contourArea(cv_points, oriented);
    } catch (const cv::Exception& e) {
        spdlog::error("ShapeUtils: OpenCV exception in ContourArea: {}", e.what());
        return 0.0;
    }
}

double ShapeUtils::ContourPerimeter(const std::vector<Point2D>& contour, bool closed) {
    if (contour.size() < 2) return 0.0;

    try {
        auto cv_points = ToCV(contour);
        return cv::arcLength(cv_points, closed);
    } catch (const cv::Exception& e) {
        spdlog::error("ShapeUtils: OpenCV exception in ContourPerimeter: {}", e.what());
        return 0.0;
    }
}

double ShapeUtils::PointPolygonTest(const std::vector<Point2D>& contour, Point2D point) {
    if (contour.size() < 3) return -1.0;

    try {
        auto cv_points = ToCV(contour);
        return cv::pointPolygonTest(cv_points, cv::Point2f(point.x, point.y), true);
    } catch (const cv::Exception& e) {
        spdlog::error("ShapeUtils: OpenCV exception in PointPolygonTest: {}", e.what());
        return -1.0;
    }
}

bool ShapeUtils::IsContourConvex(const std::vector<Point2D>& contour) {
    if (contour.size() < 3) return false;

    try {
        auto cv_points = ToCV(contour);
        return cv::isContourConvex(cv_points);
    } catch (const cv::Exception& e) {
        spdlog::error("ShapeUtils: OpenCV exception in IsContourConvex: {}", e.what());
        return false;
    }
}

} // namespace cyxwiz
