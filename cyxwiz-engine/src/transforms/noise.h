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
// Gaussian Noise
// ============================================================================

class GaussianNoise : public Transform {
public:
    GaussianNoise(float mean = 0.0f, float std = 0.1f)
        : mean_(mean), std_(std) {
        params_["mean"] = mean;
        params_["std"] = std;
    }

    Image apply(const Image& input) const override {
        if (!input.isValid()) return input;

        Image output = input.clone();
        std::normal_distribution<float> dist(mean_, std_);

        for (size_t i = 0; i < output.data.size(); ++i) {
            output.data[i] = std::clamp(output.data[i] + dist(rng_), 0.0f, 1.0f);
        }

        return output;
    }

    std::string name() const override { return "GaussianNoise"; }
    std::string category() const override { return "Noise"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<GaussianNoise>(mean_, std_);
    }

private:
    float mean_;
    float std_;
};

// ============================================================================
// Salt and Pepper Noise
// ============================================================================

class SaltPepperNoise : public Transform {
public:
    SaltPepperNoise(float amount = 0.05f, float salt_vs_pepper = 0.5f)
        : amount_(amount), salt_ratio_(salt_vs_pepper) {
        params_["amount"] = amount;
        params_["salt_vs_pepper"] = salt_vs_pepper;
    }

    Image apply(const Image& input) const override {
        if (!input.isValid()) return input;

        Image output = input.clone();
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        int total_pixels = output.width * output.height;
        int affected_pixels = static_cast<int>(total_pixels * amount_);

        for (int i = 0; i < affected_pixels; ++i) {
            int x = randomInt(0, output.width - 1);
            int y = randomInt(0, output.height - 1);
            float value = dist(rng_) < salt_ratio_ ? 1.0f : 0.0f;

            for (int c = 0; c < output.channels; ++c) {
                output.at(x, y, c) = value;
            }
        }

        return output;
    }

    std::string name() const override { return "SaltPepperNoise"; }
    std::string category() const override { return "Noise"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<SaltPepperNoise>(amount_, salt_ratio_);
    }

private:
    float amount_;
    float salt_ratio_;
};

// ============================================================================
// Gaussian Blur
// ============================================================================

class GaussianBlur : public Transform {
public:
    GaussianBlur(int kernel_size = 3, float sigma = 1.0f)
        : kernel_size_(kernel_size | 1), sigma_(sigma) {  // Ensure odd kernel size
        params_["kernel_size"] = kernel_size_;
        params_["sigma"] = sigma;
        buildKernel();
    }

    Image apply(const Image& input) const override {
        if (!input.isValid()) return input;

        Image output(input.width, input.height, input.channels);
        int half = kernel_size_ / 2;

        for (int y = 0; y < input.height; ++y) {
            for (int x = 0; x < input.width; ++x) {
                for (int c = 0; c < input.channels; ++c) {
                    float sum = 0.0f;
                    float weight_sum = 0.0f;

                    for (int ky = -half; ky <= half; ++ky) {
                        for (int kx = -half; kx <= half; ++kx) {
                            int sx = std::clamp(x + kx, 0, input.width - 1);
                            int sy = std::clamp(y + ky, 0, input.height - 1);
                            float weight = kernel_[(ky + half) * kernel_size_ + (kx + half)];
                            sum += input.at(sx, sy, c) * weight;
                            weight_sum += weight;
                        }
                    }

                    output.at(x, y, c) = sum / weight_sum;
                }
            }
        }

        return output;
    }

    std::string name() const override { return "GaussianBlur"; }
    std::string category() const override { return "Noise"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<GaussianBlur>(kernel_size_, sigma_);
    }

private:
    int kernel_size_;
    float sigma_;
    std::vector<float> kernel_;

    void buildKernel() {
        kernel_.resize(kernel_size_ * kernel_size_);
        int half = kernel_size_ / 2;
        float sum = 0.0f;

        for (int y = -half; y <= half; ++y) {
            for (int x = -half; x <= half; ++x) {
                float value = std::exp(-(x * x + y * y) / (2.0f * sigma_ * sigma_));
                kernel_[(y + half) * kernel_size_ + (x + half)] = value;
                sum += value;
            }
        }

        // Normalize
        for (auto& v : kernel_) v /= sum;
    }
};

class RandomGaussianBlur : public Transform {
public:
    RandomGaussianBlur(std::pair<int, int> kernel_size_range = {3, 7},
                       std::pair<float, float> sigma_range = {0.1f, 2.0f},
                       float p = 0.5f)
        : kernel_range_(kernel_size_range), sigma_range_(sigma_range) {
        probability_ = p;
        params_["kernel_min"] = kernel_size_range.first;
        params_["kernel_max"] = kernel_size_range.second;
        params_["sigma_min"] = sigma_range.first;
        params_["sigma_max"] = sigma_range.second;
        params_["p"] = p;
    }

    Image apply(const Image& input) const override {
        if (!shouldApply()) return input;

        int kernel_size = randomInt(kernel_range_.first / 2, kernel_range_.second / 2) * 2 + 1;
        float sigma = randomFloat(sigma_range_.first, sigma_range_.second);

        return GaussianBlur(kernel_size, sigma).apply(input);
    }

    std::string name() const override { return "RandomGaussianBlur"; }
    std::string category() const override { return "Noise"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<RandomGaussianBlur>(kernel_range_, sigma_range_, probability_);
    }

private:
    std::pair<int, int> kernel_range_;
    std::pair<float, float> sigma_range_;
};

// ============================================================================
// Box Blur (Mean Filter)
// ============================================================================

class BoxBlur : public Transform {
public:
    BoxBlur(int kernel_size = 3) : kernel_size_(kernel_size | 1) {
        params_["kernel_size"] = kernel_size_;
    }

    Image apply(const Image& input) const override {
        if (!input.isValid()) return input;

        Image output(input.width, input.height, input.channels);
        int half = kernel_size_ / 2;

        for (int y = 0; y < input.height; ++y) {
            for (int x = 0; x < input.width; ++x) {
                for (int c = 0; c < input.channels; ++c) {
                    float sum = 0.0f;
                    int count = 0;

                    for (int ky = -half; ky <= half; ++ky) {
                        for (int kx = -half; kx <= half; ++kx) {
                            int sx = x + kx;
                            int sy = y + ky;
                            if (sx >= 0 && sx < input.width && sy >= 0 && sy < input.height) {
                                sum += input.at(sx, sy, c);
                                count++;
                            }
                        }
                    }

                    output.at(x, y, c) = sum / count;
                }
            }
        }

        return output;
    }

    std::string name() const override { return "BoxBlur"; }
    std::string category() const override { return "Noise"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<BoxBlur>(kernel_size_);
    }

private:
    int kernel_size_;
};

// ============================================================================
// Sharpen
// ============================================================================

class Sharpen : public Transform {
public:
    Sharpen(float factor = 1.0f) : factor_(factor) {
        params_["factor"] = factor;
    }

    Image apply(const Image& input) const override {
        if (!input.isValid()) return input;

        // First apply blur to get low-frequency
        BoxBlur blur(3);
        Image blurred = blur.apply(input);

        // High-frequency = original - blurred
        // Sharpened = original + factor * high-frequency
        Image output = input.clone();

        for (int y = 0; y < input.height; ++y) {
            for (int x = 0; x < input.width; ++x) {
                for (int c = 0; c < input.channels; ++c) {
                    float high_freq = input.at(x, y, c) - blurred.at(x, y, c);
                    float sharpened = input.at(x, y, c) + factor_ * high_freq;
                    output.at(x, y, c) = std::clamp(sharpened, 0.0f, 1.0f);
                }
            }
        }

        return output;
    }

    std::string name() const override { return "Sharpen"; }
    std::string category() const override { return "Noise"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<Sharpen>(factor_);
    }

private:
    float factor_;
};

class RandomAdjustSharpness : public Transform {
public:
    RandomAdjustSharpness(float factor = 2.0f, float p = 0.5f)
        : factor_(factor) {
        probability_ = p;
        params_["factor"] = factor;
        params_["p"] = p;
    }

    Image apply(const Image& input) const override {
        if (!shouldApply()) return input;
        return Sharpen(factor_).apply(input);
    }

    std::string name() const override { return "RandomAdjustSharpness"; }
    std::string category() const override { return "Noise"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<RandomAdjustSharpness>(factor_, probability_);
    }

private:
    float factor_;
};

// ============================================================================
// Random Erasing (Cutout)
// ============================================================================

class RandomErasing : public Transform {
public:
    RandomErasing(float p = 0.5f,
                  std::pair<float, float> scale = {0.02f, 0.33f},
                  std::pair<float, float> ratio = {0.3f, 3.3f},
                  float value = 0.0f)
        : scale_(scale), ratio_(ratio), fill_value_(value) {
        probability_ = p;
        params_["p"] = p;
        params_["scale"] = scale;
        params_["ratio"] = ratio;
        params_["value"] = value;
    }

    Image apply(const Image& input) const override {
        if (!shouldApply()) return input;

        Image output = input.clone();
        int area = input.width * input.height;

        // Try to find valid erase parameters
        for (int attempt = 0; attempt < 10; ++attempt) {
            float target_area = area * randomFloat(scale_.first, scale_.second);
            float aspect_ratio = randomFloat(ratio_.first, ratio_.second);

            int h = static_cast<int>(std::sqrt(target_area / aspect_ratio));
            int w = static_cast<int>(std::sqrt(target_area * aspect_ratio));

            if (w < input.width && h < input.height) {
                int x = randomInt(0, input.width - w);
                int y = randomInt(0, input.height - h);

                // Fill with value
                for (int ey = y; ey < y + h; ++ey) {
                    for (int ex = x; ex < x + w; ++ex) {
                        for (int c = 0; c < input.channels; ++c) {
                            output.at(ex, ey, c) = fill_value_;
                        }
                    }
                }

                return output;
            }
        }

        return output;
    }

    std::string name() const override { return "RandomErasing"; }
    std::string category() const override { return "Noise"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<RandomErasing>(probability_, scale_, ratio_, fill_value_);
    }

private:
    std::pair<float, float> scale_;
    std::pair<float, float> ratio_;
    float fill_value_;
};

class Cutout : public Transform {
public:
    Cutout(int n_holes = 1, int length = 16)
        : n_holes_(n_holes), length_(length) {
        params_["n_holes"] = n_holes;
        params_["length"] = length;
    }

    Image apply(const Image& input) const override {
        if (!input.isValid()) return input;

        Image output = input.clone();

        for (int i = 0; i < n_holes_; ++i) {
            int cx = randomInt(0, input.width - 1);
            int cy = randomInt(0, input.height - 1);

            int x1 = std::max(0, cx - length_ / 2);
            int y1 = std::max(0, cy - length_ / 2);
            int x2 = std::min(input.width, cx + length_ / 2);
            int y2 = std::min(input.height, cy + length_ / 2);

            for (int y = y1; y < y2; ++y) {
                for (int x = x1; x < x2; ++x) {
                    for (int c = 0; c < input.channels; ++c) {
                        output.at(x, y, c) = 0.0f;
                    }
                }
            }
        }

        return output;
    }

    std::string name() const override { return "Cutout"; }
    std::string category() const override { return "Noise"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<Cutout>(n_holes_, length_);
    }

private:
    int n_holes_;
    int length_;
};

} // namespace transforms
} // namespace cyxwiz
