#pragma once

#include <vector>
#include <string>
#include <memory>
#include <random>
#include <functional>
#include <map>
#include <variant>
#include <optional>

namespace cyxwiz {
namespace transforms {

/**
 * Image data structure for transforms
 * Stores image as float vector with dimensions
 */
struct Image {
    std::vector<float> data;
    int width = 0;
    int height = 0;
    int channels = 3;

    Image() = default;
    Image(int w, int h, int c = 3) : width(w), height(h), channels(c) {
        data.resize(w * h * c, 0.0f);
    }
    Image(std::vector<float> d, int w, int h, int c = 3)
        : data(std::move(d)), width(w), height(h), channels(c) {}

    // Access pixel at (x, y, channel)
    float& at(int x, int y, int c = 0) {
        return data[(y * width + x) * channels + c];
    }
    const float& at(int x, int y, int c = 0) const {
        return data[(y * width + x) * channels + c];
    }

    // Check validity
    bool isValid() const {
        return width > 0 && height > 0 && channels > 0 &&
               data.size() == static_cast<size_t>(width * height * channels);
    }

    // Clone
    Image clone() const {
        return Image(data, width, height, channels);
    }
};

/**
 * Transform parameter types
 */
using ParamValue = std::variant<int, float, double, bool, std::string,
                                 std::vector<float>, std::pair<float, float>>;

/**
 * Base Transform class
 * All augmentation transforms inherit from this
 */
class Transform {
public:
    virtual ~Transform() = default;

    // Apply transform to image
    virtual Image apply(const Image& input) const = 0;

    // Get transform name
    virtual std::string name() const = 0;

    // Get transform category
    virtual std::string category() const = 0;

    // Get/Set parameters
    virtual std::map<std::string, ParamValue> getParams() const { return params_; }
    virtual void setParam(const std::string& name, const ParamValue& value) {
        params_[name] = value;
    }

    // Enable/disable
    bool isEnabled() const { return enabled_; }
    void setEnabled(bool enabled) { enabled_ = enabled; }

    // Probability of applying (for random transforms)
    float getProbability() const { return probability_; }
    void setProbability(float p) { probability_ = std::clamp(p, 0.0f, 1.0f); }

    // Clone transform
    virtual std::unique_ptr<Transform> clone() const = 0;

    // Serialize to JSON-like string (for saving pipelines)
    virtual std::string serialize() const {
        std::string result = "{\"name\":\"" + name() + "\"";
        result += ",\"enabled\":" + std::string(enabled_ ? "true" : "false");
        result += ",\"probability\":" + std::to_string(probability_);
        result += "}";
        return result;
    }

protected:
    std::map<std::string, ParamValue> params_;
    bool enabled_ = true;
    float probability_ = 1.0f;

    // Random number generator (mutable for const apply)
    mutable std::mt19937 rng_{std::random_device{}()};

    // Helper: should apply based on probability
    bool shouldApply() const {
        if (probability_ >= 1.0f) return true;
        if (probability_ <= 0.0f) return false;
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        return dist(rng_) < probability_;
    }

    // Helper: get random float in range
    float randomFloat(float min, float max) const {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(rng_);
    }

    // Helper: get random int in range
    int randomInt(int min, int max) const {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(rng_);
    }
};

/**
 * Compose - Chain multiple transforms together
 */
class Compose : public Transform {
public:
    Compose() = default;
    Compose(std::vector<std::unique_ptr<Transform>> transforms)
        : transforms_(std::move(transforms)) {}

    Image apply(const Image& input) const override {
        Image result = input;
        for (const auto& t : transforms_) {
            if (t && t->isEnabled()) {
                result = t->apply(result);
            }
        }
        return result;
    }

    std::string name() const override { return "Compose"; }
    std::string category() const override { return "Utility"; }

    // Add transform to pipeline
    void add(std::unique_ptr<Transform> transform) {
        transforms_.push_back(std::move(transform));
    }

    // Insert at position
    void insert(size_t index, std::unique_ptr<Transform> transform) {
        if (index >= transforms_.size()) {
            transforms_.push_back(std::move(transform));
        } else {
            transforms_.insert(transforms_.begin() + index, std::move(transform));
        }
    }

    // Remove at position
    void remove(size_t index) {
        if (index < transforms_.size()) {
            transforms_.erase(transforms_.begin() + index);
        }
    }

    // Move transform (reorder)
    void move(size_t from, size_t to) {
        if (from >= transforms_.size() || to >= transforms_.size()) return;
        auto t = std::move(transforms_[from]);
        transforms_.erase(transforms_.begin() + from);
        transforms_.insert(transforms_.begin() + to, std::move(t));
    }

    // Get transform at index
    Transform* get(size_t index) {
        return index < transforms_.size() ? transforms_[index].get() : nullptr;
    }
    const Transform* get(size_t index) const {
        return index < transforms_.size() ? transforms_[index].get() : nullptr;
    }

    // Get number of transforms
    size_t size() const { return transforms_.size(); }

    // Clear all transforms
    void clear() { transforms_.clear(); }

    std::unique_ptr<Transform> clone() const override {
        auto result = std::make_unique<Compose>();
        for (const auto& t : transforms_) {
            if (t) result->add(t->clone());
        }
        return result;
    }

private:
    std::vector<std::unique_ptr<Transform>> transforms_;
};

/**
 * RandomChoice - Randomly select one transform from a list
 */
class RandomChoice : public Transform {
public:
    RandomChoice() = default;
    RandomChoice(std::vector<std::unique_ptr<Transform>> transforms)
        : transforms_(std::move(transforms)) {}

    Image apply(const Image& input) const override {
        if (transforms_.empty()) return input;
        int idx = randomInt(0, static_cast<int>(transforms_.size()) - 1);
        return transforms_[idx]->apply(input);
    }

    std::string name() const override { return "RandomChoice"; }
    std::string category() const override { return "Utility"; }

    void add(std::unique_ptr<Transform> transform) {
        transforms_.push_back(std::move(transform));
    }

    std::unique_ptr<Transform> clone() const override {
        auto result = std::make_unique<RandomChoice>();
        for (const auto& t : transforms_) {
            if (t) result->add(t->clone());
        }
        return result;
    }

private:
    std::vector<std::unique_ptr<Transform>> transforms_;
};

/**
 * RandomApply - Apply transform with given probability
 */
class RandomApply : public Transform {
public:
    RandomApply(std::unique_ptr<Transform> transform, float p = 0.5f)
        : transform_(std::move(transform)) {
        probability_ = p;
    }

    Image apply(const Image& input) const override {
        if (!transform_ || !shouldApply()) return input;
        return transform_->apply(input);
    }

    std::string name() const override {
        return "RandomApply(" + (transform_ ? transform_->name() : "null") + ")";
    }
    std::string category() const override { return "Utility"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<RandomApply>(
            transform_ ? transform_->clone() : nullptr,
            probability_
        );
    }

private:
    std::unique_ptr<Transform> transform_;
};

} // namespace transforms
} // namespace cyxwiz
