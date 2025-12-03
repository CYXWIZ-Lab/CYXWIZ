#pragma once

#include "transform.h"
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cyxwiz {
namespace transforms {

// ============================================================================
// Resize Transform
// ============================================================================

enum class InterpolationMode {
    Nearest,
    Bilinear,
    Bicubic
};

class Resize : public Transform {
public:
    Resize(int width, int height, InterpolationMode mode = InterpolationMode::Bilinear)
        : target_width_(width), target_height_(height), mode_(mode) {
        params_["width"] = width;
        params_["height"] = height;
        params_["mode"] = static_cast<int>(mode);
    }

    Image apply(const Image& input) const override {
        if (!input.isValid()) return input;
        if (input.width == target_width_ && input.height == target_height_) {
            return input;
        }

        Image output(target_width_, target_height_, input.channels);

        switch (mode_) {
            case InterpolationMode::Nearest:
                resizeNearest(input, output);
                break;
            case InterpolationMode::Bilinear:
                resizeBilinear(input, output);
                break;
            case InterpolationMode::Bicubic:
                resizeBicubic(input, output);
                break;
        }

        return output;
    }

    std::string name() const override { return "Resize"; }
    std::string category() const override { return "Geometric"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<Resize>(target_width_, target_height_, mode_);
    }

private:
    int target_width_;
    int target_height_;
    InterpolationMode mode_;

    void resizeNearest(const Image& src, Image& dst) const {
        float x_ratio = static_cast<float>(src.width) / dst.width;
        float y_ratio = static_cast<float>(src.height) / dst.height;

        for (int y = 0; y < dst.height; ++y) {
            for (int x = 0; x < dst.width; ++x) {
                int src_x = std::min(static_cast<int>(x * x_ratio), src.width - 1);
                int src_y = std::min(static_cast<int>(y * y_ratio), src.height - 1);
                for (int c = 0; c < dst.channels; ++c) {
                    dst.at(x, y, c) = src.at(src_x, src_y, c);
                }
            }
        }
    }

    void resizeBilinear(const Image& src, Image& dst) const {
        float x_ratio = static_cast<float>(src.width - 1) / dst.width;
        float y_ratio = static_cast<float>(src.height - 1) / dst.height;

        for (int y = 0; y < dst.height; ++y) {
            for (int x = 0; x < dst.width; ++x) {
                float src_x = x * x_ratio;
                float src_y = y * y_ratio;

                int x0 = static_cast<int>(src_x);
                int y0 = static_cast<int>(src_y);
                int x1 = std::min(x0 + 1, src.width - 1);
                int y1 = std::min(y0 + 1, src.height - 1);

                float dx = src_x - x0;
                float dy = src_y - y0;

                for (int c = 0; c < dst.channels; ++c) {
                    float v00 = src.at(x0, y0, c);
                    float v01 = src.at(x0, y1, c);
                    float v10 = src.at(x1, y0, c);
                    float v11 = src.at(x1, y1, c);

                    float value = v00 * (1 - dx) * (1 - dy) +
                                  v10 * dx * (1 - dy) +
                                  v01 * (1 - dx) * dy +
                                  v11 * dx * dy;

                    dst.at(x, y, c) = value;
                }
            }
        }
    }

    void resizeBicubic(const Image& src, Image& dst) const {
        // Simplified bicubic - use bilinear for now
        // Full bicubic requires more complex kernel
        resizeBilinear(src, dst);
    }
};

// ============================================================================
// Crop Transforms
// ============================================================================

class CenterCrop : public Transform {
public:
    CenterCrop(int width, int height) : crop_width_(width), crop_height_(height) {
        params_["width"] = width;
        params_["height"] = height;
    }

    Image apply(const Image& input) const override {
        if (!input.isValid()) return input;

        int start_x = (input.width - crop_width_) / 2;
        int start_y = (input.height - crop_height_) / 2;

        start_x = std::max(0, start_x);
        start_y = std::max(0, start_y);

        int actual_width = std::min(crop_width_, input.width - start_x);
        int actual_height = std::min(crop_height_, input.height - start_y);

        Image output(actual_width, actual_height, input.channels);

        for (int y = 0; y < actual_height; ++y) {
            for (int x = 0; x < actual_width; ++x) {
                for (int c = 0; c < input.channels; ++c) {
                    output.at(x, y, c) = input.at(start_x + x, start_y + y, c);
                }
            }
        }

        return output;
    }

    std::string name() const override { return "CenterCrop"; }
    std::string category() const override { return "Geometric"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<CenterCrop>(crop_width_, crop_height_);
    }

private:
    int crop_width_;
    int crop_height_;
};

class RandomCrop : public Transform {
public:
    RandomCrop(int width, int height, int padding = 0)
        : crop_width_(width), crop_height_(height), padding_(padding) {
        params_["width"] = width;
        params_["height"] = height;
        params_["padding"] = padding;
    }

    Image apply(const Image& input) const override {
        if (!input.isValid()) return input;

        // Apply padding if needed
        Image padded = input;
        if (padding_ > 0) {
            padded = applyPadding(input);
        }

        // Random crop position
        int max_x = std::max(0, padded.width - crop_width_);
        int max_y = std::max(0, padded.height - crop_height_);

        int start_x = max_x > 0 ? randomInt(0, max_x) : 0;
        int start_y = max_y > 0 ? randomInt(0, max_y) : 0;

        int actual_width = std::min(crop_width_, padded.width - start_x);
        int actual_height = std::min(crop_height_, padded.height - start_y);

        Image output(actual_width, actual_height, padded.channels);

        for (int y = 0; y < actual_height; ++y) {
            for (int x = 0; x < actual_width; ++x) {
                for (int c = 0; c < padded.channels; ++c) {
                    output.at(x, y, c) = padded.at(start_x + x, start_y + y, c);
                }
            }
        }

        return output;
    }

    std::string name() const override { return "RandomCrop"; }
    std::string category() const override { return "Geometric"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<RandomCrop>(crop_width_, crop_height_, padding_);
    }

private:
    int crop_width_;
    int crop_height_;
    int padding_;

    Image applyPadding(const Image& input) const {
        int new_width = input.width + 2 * padding_;
        int new_height = input.height + 2 * padding_;
        Image output(new_width, new_height, input.channels);

        // Fill with zeros (black padding)
        std::fill(output.data.begin(), output.data.end(), 0.0f);

        // Copy original image to center
        for (int y = 0; y < input.height; ++y) {
            for (int x = 0; x < input.width; ++x) {
                for (int c = 0; c < input.channels; ++c) {
                    output.at(padding_ + x, padding_ + y, c) = input.at(x, y, c);
                }
            }
        }

        return output;
    }
};

class RandomResizedCrop : public Transform {
public:
    RandomResizedCrop(int size, float scale_min = 0.08f, float scale_max = 1.0f,
                      float ratio_min = 0.75f, float ratio_max = 1.33f)
        : size_(size), scale_min_(scale_min), scale_max_(scale_max),
          ratio_min_(ratio_min), ratio_max_(ratio_max) {
        params_["size"] = size;
        params_["scale_min"] = scale_min;
        params_["scale_max"] = scale_max;
        params_["ratio_min"] = ratio_min;
        params_["ratio_max"] = ratio_max;
    }

    Image apply(const Image& input) const override {
        if (!input.isValid()) return input;

        int area = input.width * input.height;

        // Try to find valid crop parameters
        for (int attempt = 0; attempt < 10; ++attempt) {
            float target_area = area * randomFloat(scale_min_, scale_max_);
            float aspect_ratio = std::exp(randomFloat(
                std::log(ratio_min_), std::log(ratio_max_)));

            int w = static_cast<int>(std::sqrt(target_area * aspect_ratio));
            int h = static_cast<int>(std::sqrt(target_area / aspect_ratio));

            if (w <= input.width && h <= input.height) {
                int x = randomInt(0, input.width - w);
                int y = randomInt(0, input.height - h);

                // Crop
                Image cropped(w, h, input.channels);
                for (int cy = 0; cy < h; ++cy) {
                    for (int cx = 0; cx < w; ++cx) {
                        for (int c = 0; c < input.channels; ++c) {
                            cropped.at(cx, cy, c) = input.at(x + cx, y + cy, c);
                        }
                    }
                }

                // Resize to target size
                Resize resize(size_, size_, InterpolationMode::Bilinear);
                return resize.apply(cropped);
            }
        }

        // Fallback: center crop and resize
        CenterCrop center(std::min(input.width, input.height),
                         std::min(input.width, input.height));
        Resize resize(size_, size_, InterpolationMode::Bilinear);
        return resize.apply(center.apply(input));
    }

    std::string name() const override { return "RandomResizedCrop"; }
    std::string category() const override { return "Geometric"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<RandomResizedCrop>(
            size_, scale_min_, scale_max_, ratio_min_, ratio_max_);
    }

private:
    int size_;
    float scale_min_, scale_max_;
    float ratio_min_, ratio_max_;
};

// ============================================================================
// Flip Transforms
// ============================================================================

class HorizontalFlip : public Transform {
public:
    HorizontalFlip() = default;

    Image apply(const Image& input) const override {
        if (!input.isValid()) return input;

        Image output(input.width, input.height, input.channels);

        for (int y = 0; y < input.height; ++y) {
            for (int x = 0; x < input.width; ++x) {
                int flip_x = input.width - 1 - x;
                for (int c = 0; c < input.channels; ++c) {
                    output.at(x, y, c) = input.at(flip_x, y, c);
                }
            }
        }

        return output;
    }

    std::string name() const override { return "HorizontalFlip"; }
    std::string category() const override { return "Geometric"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<HorizontalFlip>();
    }
};

class VerticalFlip : public Transform {
public:
    VerticalFlip() = default;

    Image apply(const Image& input) const override {
        if (!input.isValid()) return input;

        Image output(input.width, input.height, input.channels);

        for (int y = 0; y < input.height; ++y) {
            int flip_y = input.height - 1 - y;
            for (int x = 0; x < input.width; ++x) {
                for (int c = 0; c < input.channels; ++c) {
                    output.at(x, y, c) = input.at(x, flip_y, c);
                }
            }
        }

        return output;
    }

    std::string name() const override { return "VerticalFlip"; }
    std::string category() const override { return "Geometric"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<VerticalFlip>();
    }
};

class RandomHorizontalFlip : public Transform {
public:
    RandomHorizontalFlip(float p = 0.5f) {
        probability_ = p;
        params_["p"] = p;
    }

    Image apply(const Image& input) const override {
        if (!shouldApply()) return input;
        return HorizontalFlip().apply(input);
    }

    std::string name() const override { return "RandomHorizontalFlip"; }
    std::string category() const override { return "Geometric"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<RandomHorizontalFlip>(probability_);
    }
};

class RandomVerticalFlip : public Transform {
public:
    RandomVerticalFlip(float p = 0.5f) {
        probability_ = p;
        params_["p"] = p;
    }

    Image apply(const Image& input) const override {
        if (!shouldApply()) return input;
        return VerticalFlip().apply(input);
    }

    std::string name() const override { return "RandomVerticalFlip"; }
    std::string category() const override { return "Geometric"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<RandomVerticalFlip>(probability_);
    }
};

// ============================================================================
// Rotation Transforms
// ============================================================================

class Rotate : public Transform {
public:
    Rotate(float degrees) : degrees_(degrees) {
        params_["degrees"] = degrees;
    }

    Image apply(const Image& input) const override {
        if (!input.isValid()) return input;

        float radians = degrees_ * static_cast<float>(M_PI) / 180.0f;
        float cos_theta = std::cos(radians);
        float sin_theta = std::sin(radians);

        // Calculate output size (same as input for now)
        int out_width = input.width;
        int out_height = input.height;

        float cx = input.width / 2.0f;
        float cy = input.height / 2.0f;

        Image output(out_width, out_height, input.channels);

        for (int y = 0; y < out_height; ++y) {
            for (int x = 0; x < out_width; ++x) {
                // Transform to center-origin coordinates
                float dx = x - cx;
                float dy = y - cy;

                // Inverse rotation to find source pixel
                float src_x = dx * cos_theta + dy * sin_theta + cx;
                float src_y = -dx * sin_theta + dy * cos_theta + cy;

                // Bilinear interpolation
                if (src_x >= 0 && src_x < input.width - 1 &&
                    src_y >= 0 && src_y < input.height - 1) {

                    int x0 = static_cast<int>(src_x);
                    int y0 = static_cast<int>(src_y);
                    float fx = src_x - x0;
                    float fy = src_y - y0;

                    for (int c = 0; c < input.channels; ++c) {
                        float v00 = input.at(x0, y0, c);
                        float v10 = input.at(x0 + 1, y0, c);
                        float v01 = input.at(x0, y0 + 1, c);
                        float v11 = input.at(x0 + 1, y0 + 1, c);

                        output.at(x, y, c) = v00 * (1 - fx) * (1 - fy) +
                                             v10 * fx * (1 - fy) +
                                             v01 * (1 - fx) * fy +
                                             v11 * fx * fy;
                    }
                }
                // else: pixel stays black (0)
            }
        }

        return output;
    }

    std::string name() const override { return "Rotate"; }
    std::string category() const override { return "Geometric"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<Rotate>(degrees_);
    }

private:
    float degrees_;
};

class RandomRotation : public Transform {
public:
    RandomRotation(float degrees) : max_degrees_(std::abs(degrees)) {
        params_["degrees"] = degrees;
    }

    Image apply(const Image& input) const override {
        if (!shouldApply()) return input;

        float angle = randomFloat(-max_degrees_, max_degrees_);
        return Rotate(angle).apply(input);
    }

    std::string name() const override { return "RandomRotation"; }
    std::string category() const override { return "Geometric"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<RandomRotation>(max_degrees_);
    }

private:
    float max_degrees_;
};

// ============================================================================
// Affine Transform
// ============================================================================

class RandomAffine : public Transform {
public:
    RandomAffine(float degrees = 0.0f,
                 std::pair<float, float> translate = {0.0f, 0.0f},
                 std::pair<float, float> scale = {1.0f, 1.0f},
                 std::pair<float, float> shear = {0.0f, 0.0f})
        : degrees_(degrees), translate_(translate), scale_(scale), shear_(shear) {
        params_["degrees"] = degrees;
        params_["translate"] = translate;
        params_["scale"] = scale;
        params_["shear"] = shear;
    }

    Image apply(const Image& input) const override {
        if (!input.isValid()) return input;

        // Random parameters
        float angle = randomFloat(-degrees_, degrees_);
        float tx = randomFloat(-translate_.first, translate_.first) * input.width;
        float ty = randomFloat(-translate_.second, translate_.second) * input.height;
        float s = randomFloat(scale_.first, scale_.second);
        float shear_x = randomFloat(-shear_.first, shear_.first);
        float shear_y = randomFloat(-shear_.second, shear_.second);

        float radians = angle * static_cast<float>(M_PI) / 180.0f;
        float cos_a = std::cos(radians);
        float sin_a = std::sin(radians);

        // Build affine matrix (inverse for backward mapping)
        // M = T * R * S * Sh
        float cx = input.width / 2.0f;
        float cy = input.height / 2.0f;

        Image output(input.width, input.height, input.channels);

        for (int y = 0; y < input.height; ++y) {
            for (int x = 0; x < input.width; ++x) {
                // Center coordinates
                float dx = x - cx - tx;
                float dy = y - cy - ty;

                // Apply inverse transforms
                // Inverse shear
                float sx = dx - shear_x * dy;
                float sy = dy - shear_y * dx;

                // Inverse scale
                sx /= s;
                sy /= s;

                // Inverse rotation
                float src_x = sx * cos_a + sy * sin_a + cx;
                float src_y = -sx * sin_a + sy * cos_a + cy;

                // Sample with bilinear interpolation
                if (src_x >= 0 && src_x < input.width - 1 &&
                    src_y >= 0 && src_y < input.height - 1) {

                    int x0 = static_cast<int>(src_x);
                    int y0 = static_cast<int>(src_y);
                    float fx = src_x - x0;
                    float fy = src_y - y0;

                    for (int c = 0; c < input.channels; ++c) {
                        float v00 = input.at(x0, y0, c);
                        float v10 = input.at(x0 + 1, y0, c);
                        float v01 = input.at(x0, y0 + 1, c);
                        float v11 = input.at(x0 + 1, y0 + 1, c);

                        output.at(x, y, c) = v00 * (1 - fx) * (1 - fy) +
                                             v10 * fx * (1 - fy) +
                                             v01 * (1 - fx) * fy +
                                             v11 * fx * fy;
                    }
                }
            }
        }

        return output;
    }

    std::string name() const override { return "RandomAffine"; }
    std::string category() const override { return "Geometric"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<RandomAffine>(degrees_, translate_, scale_, shear_);
    }

private:
    float degrees_;
    std::pair<float, float> translate_;
    std::pair<float, float> scale_;
    std::pair<float, float> shear_;
};

} // namespace transforms
} // namespace cyxwiz
