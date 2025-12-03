#pragma once

#include "transform.h"
#include <cmath>
#include <algorithm>

namespace cyxwiz {
namespace transforms {

// ============================================================================
// Normalize Transform
// ============================================================================

class Normalize : public Transform {
public:
    Normalize(std::vector<float> mean, std::vector<float> std)
        : mean_(std::move(mean)), std_(std::move(std)) {
        params_["mean"] = mean_;
        params_["std"] = std_;
    }

    // Common presets
    static Normalize ImageNet() {
        return Normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f});
    }
    static Normalize MNIST() {
        return Normalize({0.1307f}, {0.3081f});
    }
    static Normalize CIFAR10() {
        return Normalize({0.4914f, 0.4822f, 0.4465f}, {0.2470f, 0.2435f, 0.2616f});
    }

    Image apply(const Image& input) const override {
        if (!input.isValid()) return input;

        Image output = input.clone();

        for (int y = 0; y < output.height; ++y) {
            for (int x = 0; x < output.width; ++x) {
                for (int c = 0; c < output.channels; ++c) {
                    float m = c < static_cast<int>(mean_.size()) ? mean_[c] : 0.0f;
                    float s = c < static_cast<int>(std_.size()) ? std_[c] : 1.0f;
                    output.at(x, y, c) = (output.at(x, y, c) - m) / s;
                }
            }
        }

        return output;
    }

    std::string name() const override { return "Normalize"; }
    std::string category() const override { return "Color"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<Normalize>(mean_, std_);
    }

private:
    std::vector<float> mean_;
    std::vector<float> std_;
};

// ============================================================================
// Brightness/Contrast/Saturation Transforms
// ============================================================================

class AdjustBrightness : public Transform {
public:
    AdjustBrightness(float factor) : factor_(factor) {
        params_["factor"] = factor;
    }

    Image apply(const Image& input) const override {
        if (!input.isValid()) return input;

        Image output = input.clone();

        for (size_t i = 0; i < output.data.size(); ++i) {
            output.data[i] = std::clamp(output.data[i] * factor_, 0.0f, 1.0f);
        }

        return output;
    }

    std::string name() const override { return "AdjustBrightness"; }
    std::string category() const override { return "Color"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<AdjustBrightness>(factor_);
    }

private:
    float factor_;
};

class AdjustContrast : public Transform {
public:
    AdjustContrast(float factor) : factor_(factor) {
        params_["factor"] = factor;
    }

    Image apply(const Image& input) const override {
        if (!input.isValid()) return input;

        Image output = input.clone();

        // Calculate mean per channel
        std::vector<float> means(input.channels, 0.0f);
        int pixels = input.width * input.height;

        for (int y = 0; y < input.height; ++y) {
            for (int x = 0; x < input.width; ++x) {
                for (int c = 0; c < input.channels; ++c) {
                    means[c] += input.at(x, y, c);
                }
            }
        }
        for (auto& m : means) m /= pixels;

        // Apply contrast
        for (int y = 0; y < output.height; ++y) {
            for (int x = 0; x < output.width; ++x) {
                for (int c = 0; c < output.channels; ++c) {
                    float val = output.at(x, y, c);
                    val = means[c] + factor_ * (val - means[c]);
                    output.at(x, y, c) = std::clamp(val, 0.0f, 1.0f);
                }
            }
        }

        return output;
    }

    std::string name() const override { return "AdjustContrast"; }
    std::string category() const override { return "Color"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<AdjustContrast>(factor_);
    }

private:
    float factor_;
};

class AdjustSaturation : public Transform {
public:
    AdjustSaturation(float factor) : factor_(factor) {
        params_["factor"] = factor;
    }

    Image apply(const Image& input) const override {
        if (!input.isValid() || input.channels < 3) return input;

        Image output = input.clone();

        for (int y = 0; y < output.height; ++y) {
            for (int x = 0; x < output.width; ++x) {
                float r = output.at(x, y, 0);
                float g = output.at(x, y, 1);
                float b = output.at(x, y, 2);

                // Convert to grayscale
                float gray = 0.299f * r + 0.587f * g + 0.114f * b;

                // Interpolate between grayscale and original
                output.at(x, y, 0) = std::clamp(gray + factor_ * (r - gray), 0.0f, 1.0f);
                output.at(x, y, 1) = std::clamp(gray + factor_ * (g - gray), 0.0f, 1.0f);
                output.at(x, y, 2) = std::clamp(gray + factor_ * (b - gray), 0.0f, 1.0f);
            }
        }

        return output;
    }

    std::string name() const override { return "AdjustSaturation"; }
    std::string category() const override { return "Color"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<AdjustSaturation>(factor_);
    }

private:
    float factor_;
};

class AdjustHue : public Transform {
public:
    AdjustHue(float factor) : factor_(std::clamp(factor, -0.5f, 0.5f)) {
        params_["factor"] = factor_;
    }

    Image apply(const Image& input) const override {
        if (!input.isValid() || input.channels < 3) return input;

        Image output = input.clone();

        for (int y = 0; y < output.height; ++y) {
            for (int x = 0; x < output.width; ++x) {
                float r = output.at(x, y, 0);
                float g = output.at(x, y, 1);
                float b = output.at(x, y, 2);

                // Convert RGB to HSV
                float h, s, v;
                rgbToHsv(r, g, b, h, s, v);

                // Adjust hue
                h += factor_;
                if (h > 1.0f) h -= 1.0f;
                if (h < 0.0f) h += 1.0f;

                // Convert back to RGB
                hsvToRgb(h, s, v, r, g, b);

                output.at(x, y, 0) = r;
                output.at(x, y, 1) = g;
                output.at(x, y, 2) = b;
            }
        }

        return output;
    }

    std::string name() const override { return "AdjustHue"; }
    std::string category() const override { return "Color"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<AdjustHue>(factor_);
    }

private:
    float factor_;

    void rgbToHsv(float r, float g, float b, float& h, float& s, float& v) const {
        float max_val = std::max({r, g, b});
        float min_val = std::min({r, g, b});
        float delta = max_val - min_val;

        v = max_val;

        if (max_val == 0.0f) {
            s = 0.0f;
            h = 0.0f;
            return;
        }

        s = delta / max_val;

        if (delta == 0.0f) {
            h = 0.0f;
        } else if (max_val == r) {
            h = (g - b) / delta;
            if (h < 0.0f) h += 6.0f;
        } else if (max_val == g) {
            h = 2.0f + (b - r) / delta;
        } else {
            h = 4.0f + (r - g) / delta;
        }
        h /= 6.0f;
    }

    void hsvToRgb(float h, float s, float v, float& r, float& g, float& b) const {
        if (s == 0.0f) {
            r = g = b = v;
            return;
        }

        h *= 6.0f;
        int i = static_cast<int>(h);
        float f = h - i;
        float p = v * (1.0f - s);
        float q = v * (1.0f - s * f);
        float t = v * (1.0f - s * (1.0f - f));

        switch (i % 6) {
            case 0: r = v; g = t; b = p; break;
            case 1: r = q; g = v; b = p; break;
            case 2: r = p; g = v; b = t; break;
            case 3: r = p; g = q; b = v; break;
            case 4: r = t; g = p; b = v; break;
            case 5: r = v; g = p; b = q; break;
        }
    }
};

// ============================================================================
// ColorJitter - Combined random color augmentation
// ============================================================================

class ColorJitter : public Transform {
public:
    ColorJitter(float brightness = 0.0f, float contrast = 0.0f,
                float saturation = 0.0f, float hue = 0.0f)
        : brightness_(brightness), contrast_(contrast),
          saturation_(saturation), hue_(hue) {
        params_["brightness"] = brightness;
        params_["contrast"] = contrast;
        params_["saturation"] = saturation;
        params_["hue"] = hue;
    }

    Image apply(const Image& input) const override {
        if (!input.isValid()) return input;

        Image result = input;

        // Random order for transforms
        std::vector<int> order = {0, 1, 2, 3};
        std::shuffle(order.begin(), order.end(), rng_);

        for (int idx : order) {
            switch (idx) {
                case 0:
                    if (brightness_ > 0) {
                        float factor = randomFloat(
                            std::max(0.0f, 1.0f - brightness_),
                            1.0f + brightness_);
                        result = AdjustBrightness(factor).apply(result);
                    }
                    break;
                case 1:
                    if (contrast_ > 0) {
                        float factor = randomFloat(
                            std::max(0.0f, 1.0f - contrast_),
                            1.0f + contrast_);
                        result = AdjustContrast(factor).apply(result);
                    }
                    break;
                case 2:
                    if (saturation_ > 0) {
                        float factor = randomFloat(
                            std::max(0.0f, 1.0f - saturation_),
                            1.0f + saturation_);
                        result = AdjustSaturation(factor).apply(result);
                    }
                    break;
                case 3:
                    if (hue_ > 0) {
                        float factor = randomFloat(-hue_, hue_);
                        result = AdjustHue(factor).apply(result);
                    }
                    break;
            }
        }

        return result;
    }

    std::string name() const override { return "ColorJitter"; }
    std::string category() const override { return "Color"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<ColorJitter>(brightness_, contrast_, saturation_, hue_);
    }

private:
    float brightness_;
    float contrast_;
    float saturation_;
    float hue_;
};

// ============================================================================
// Grayscale Transforms
// ============================================================================

class Grayscale : public Transform {
public:
    Grayscale(int num_output_channels = 1) : num_channels_(num_output_channels) {
        params_["num_output_channels"] = num_output_channels;
    }

    Image apply(const Image& input) const override {
        if (!input.isValid()) return input;

        Image output(input.width, input.height, num_channels_);

        for (int y = 0; y < input.height; ++y) {
            for (int x = 0; x < input.width; ++x) {
                float gray;
                if (input.channels >= 3) {
                    gray = 0.299f * input.at(x, y, 0) +
                           0.587f * input.at(x, y, 1) +
                           0.114f * input.at(x, y, 2);
                } else {
                    gray = input.at(x, y, 0);
                }

                for (int c = 0; c < num_channels_; ++c) {
                    output.at(x, y, c) = gray;
                }
            }
        }

        return output;
    }

    std::string name() const override { return "Grayscale"; }
    std::string category() const override { return "Color"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<Grayscale>(num_channels_);
    }

private:
    int num_channels_;
};

class RandomGrayscale : public Transform {
public:
    RandomGrayscale(float p = 0.1f) {
        probability_ = p;
        params_["p"] = p;
    }

    Image apply(const Image& input) const override {
        if (!shouldApply()) return input;
        return Grayscale(input.channels).apply(input);
    }

    std::string name() const override { return "RandomGrayscale"; }
    std::string category() const override { return "Color"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<RandomGrayscale>(probability_);
    }
};

// ============================================================================
// Invert/Solarize/Posterize
// ============================================================================

class Invert : public Transform {
public:
    Invert() = default;

    Image apply(const Image& input) const override {
        if (!input.isValid()) return input;

        Image output = input.clone();
        for (size_t i = 0; i < output.data.size(); ++i) {
            output.data[i] = 1.0f - output.data[i];
        }
        return output;
    }

    std::string name() const override { return "Invert"; }
    std::string category() const override { return "Color"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<Invert>();
    }
};

class RandomInvert : public Transform {
public:
    RandomInvert(float p = 0.5f) {
        probability_ = p;
        params_["p"] = p;
    }

    Image apply(const Image& input) const override {
        if (!shouldApply()) return input;
        return Invert().apply(input);
    }

    std::string name() const override { return "RandomInvert"; }
    std::string category() const override { return "Color"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<RandomInvert>(probability_);
    }
};

class Solarize : public Transform {
public:
    Solarize(float threshold = 0.5f) : threshold_(threshold) {
        params_["threshold"] = threshold;
    }

    Image apply(const Image& input) const override {
        if (!input.isValid()) return input;

        Image output = input.clone();
        for (size_t i = 0; i < output.data.size(); ++i) {
            if (output.data[i] >= threshold_) {
                output.data[i] = 1.0f - output.data[i];
            }
        }
        return output;
    }

    std::string name() const override { return "Solarize"; }
    std::string category() const override { return "Color"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<Solarize>(threshold_);
    }

private:
    float threshold_;
};

class Posterize : public Transform {
public:
    Posterize(int bits = 4) : bits_(std::clamp(bits, 1, 8)) {
        params_["bits"] = bits_;
    }

    Image apply(const Image& input) const override {
        if (!input.isValid()) return input;

        Image output = input.clone();
        int levels = 1 << bits_;
        float scale = static_cast<float>(levels - 1);

        for (size_t i = 0; i < output.data.size(); ++i) {
            // Quantize to bits_ levels
            int quantized = static_cast<int>(output.data[i] * scale + 0.5f);
            output.data[i] = static_cast<float>(quantized) / scale;
        }
        return output;
    }

    std::string name() const override { return "Posterize"; }
    std::string category() const override { return "Color"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<Posterize>(bits_);
    }

private:
    int bits_;
};

} // namespace transforms
} // namespace cyxwiz
