#pragma once

#include "transform.h"
#include "geometric.h"
#include "color.h"
#include "noise.h"
#include <cmath>
#include <algorithm>
#include <numeric>  // for std::iota

namespace cyxwiz {
namespace transforms {

// ============================================================================
// Mixup - Mixes two images with random interpolation
// ============================================================================

struct MixupResult {
    Image image;
    float lambda;  // Mixing coefficient
    int label1;
    int label2;

    // Soft label: lambda * one_hot(label1) + (1-lambda) * one_hot(label2)
    std::vector<float> getSoftLabel(int num_classes) const {
        std::vector<float> soft(num_classes, 0.0f);
        if (label1 >= 0 && label1 < num_classes) soft[label1] = lambda;
        if (label2 >= 0 && label2 < num_classes) soft[label2] = 1.0f - lambda;
        return soft;
    }
};

class Mixup {
public:
    Mixup(float alpha = 1.0f) : alpha_(alpha) {}

    MixupResult apply(const Image& img1, int label1,
                      const Image& img2, int label2) const {
        MixupResult result;
        result.label1 = label1;
        result.label2 = label2;

        // Sample lambda from Beta distribution (approximation using gamma)
        result.lambda = sampleBeta(alpha_, alpha_);

        // Ensure img1 and img2 have same dimensions
        if (img1.width != img2.width || img1.height != img2.height ||
            img1.channels != img2.channels) {
            // Return first image if sizes don't match
            result.image = img1;
            result.lambda = 1.0f;
            return result;
        }

        result.image = Image(img1.width, img1.height, img1.channels);

        for (size_t i = 0; i < img1.data.size(); ++i) {
            result.image.data[i] = result.lambda * img1.data[i] +
                                   (1.0f - result.lambda) * img2.data[i];
        }

        return result;
    }

private:
    float alpha_;
    mutable std::mt19937 rng_{std::random_device{}()};

    float sampleBeta(float a, float b) const {
        // Simple approximation: use uniform if alpha=1 (standard Mixup)
        if (a == 1.0f && b == 1.0f) {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            return dist(rng_);
        }

        // Gamma sampling for general case
        std::gamma_distribution<float> gamma_a(a, 1.0f);
        std::gamma_distribution<float> gamma_b(b, 1.0f);

        float x = gamma_a(rng_);
        float y = gamma_b(rng_);

        return x / (x + y);
    }
};

// ============================================================================
// CutMix - Cuts and pastes patches between images
// ============================================================================

struct CutMixResult {
    Image image;
    float lambda;  // Area ratio of image1
    int label1;
    int label2;

    std::vector<float> getSoftLabel(int num_classes) const {
        std::vector<float> soft(num_classes, 0.0f);
        if (label1 >= 0 && label1 < num_classes) soft[label1] = lambda;
        if (label2 >= 0 && label2 < num_classes) soft[label2] = 1.0f - lambda;
        return soft;
    }
};

class CutMix {
public:
    CutMix(float alpha = 1.0f) : alpha_(alpha) {}

    CutMixResult apply(const Image& img1, int label1,
                       const Image& img2, int label2) const {
        CutMixResult result;
        result.label1 = label1;
        result.label2 = label2;

        if (img1.width != img2.width || img1.height != img2.height ||
            img1.channels != img2.channels) {
            result.image = img1;
            result.lambda = 1.0f;
            return result;
        }

        // Sample lambda
        float lam = sampleBeta(alpha_, alpha_);

        // Get random box
        int cx = randomInt(0, img1.width - 1);
        int cy = randomInt(0, img1.height - 1);

        float cut_ratio = std::sqrt(1.0f - lam);
        int cut_w = static_cast<int>(img1.width * cut_ratio);
        int cut_h = static_cast<int>(img1.height * cut_ratio);

        int x1 = std::max(0, cx - cut_w / 2);
        int y1 = std::max(0, cy - cut_h / 2);
        int x2 = std::min(img1.width, cx + cut_w / 2);
        int y2 = std::min(img1.height, cy + cut_h / 2);

        // Copy img1 and paste patch from img2
        result.image = img1.clone();

        for (int y = y1; y < y2; ++y) {
            for (int x = x1; x < x2; ++x) {
                for (int c = 0; c < img1.channels; ++c) {
                    result.image.at(x, y, c) = img2.at(x, y, c);
                }
            }
        }

        // Adjust lambda to reflect actual box area
        int box_area = (x2 - x1) * (y2 - y1);
        int img_area = img1.width * img1.height;
        result.lambda = 1.0f - static_cast<float>(box_area) / img_area;

        return result;
    }

private:
    float alpha_;
    mutable std::mt19937 rng_{std::random_device{}()};

    float sampleBeta(float a, float b) const {
        if (a == 1.0f && b == 1.0f) {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            return dist(rng_);
        }

        std::gamma_distribution<float> gamma_a(a, 1.0f);
        std::gamma_distribution<float> gamma_b(b, 1.0f);

        float x = gamma_a(rng_);
        float y = gamma_b(rng_);

        return x / (x + y);
    }

    int randomInt(int min, int max) const {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(rng_);
    }
};

// ============================================================================
// RandAugment - Randomly apply N transforms from a pool
// ============================================================================

class RandAugment : public Transform {
public:
    RandAugment(int n = 2, int m = 9) : n_(n), magnitude_(m) {
        params_["n"] = n;
        params_["m"] = m;
        buildTransformPool();
    }

    Image apply(const Image& input) const override {
        if (!input.isValid()) return input;

        Image result = input;

        // Randomly select N transforms
        std::vector<int> indices(transform_pool_.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng_);

        for (int i = 0; i < std::min(n_, static_cast<int>(indices.size())); ++i) {
            result = transform_pool_[indices[i]]->apply(result);
        }

        return result;
    }

    std::string name() const override { return "RandAugment"; }
    std::string category() const override { return "Advanced"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<RandAugment>(n_, magnitude_);
    }

private:
    int n_;  // Number of transforms to apply
    int magnitude_;  // Strength (0-30, typically 9)
    std::vector<std::unique_ptr<Transform>> transform_pool_;

    void buildTransformPool() {
        float mag = magnitude_ / 30.0f;  // Normalize to 0-1

        // Geometric
        transform_pool_.push_back(std::make_unique<RandomHorizontalFlip>(0.5f));
        transform_pool_.push_back(std::make_unique<RandomVerticalFlip>(0.5f));
        transform_pool_.push_back(std::make_unique<RandomRotation>(30.0f * mag));
        transform_pool_.push_back(std::make_unique<RandomAffine>(
            0.0f, std::make_pair(0.3f * mag, 0.3f * mag),
            std::make_pair(1.0f, 1.0f), std::make_pair(0.0f, 0.0f)));

        // Color
        transform_pool_.push_back(std::make_unique<ColorJitter>(
            0.9f * mag, 0.9f * mag, 0.9f * mag, 0.1f * mag));
        transform_pool_.push_back(std::make_unique<RandomGrayscale>(0.3f));
        transform_pool_.push_back(std::make_unique<RandomInvert>(0.3f));
        transform_pool_.push_back(std::make_unique<Solarize>(1.0f - 0.5f * mag));
        transform_pool_.push_back(std::make_unique<Posterize>(
            static_cast<int>(8 - 4 * mag)));

        // Noise/Blur
        transform_pool_.push_back(std::make_unique<RandomGaussianBlur>(
            std::make_pair(3, 7), std::make_pair(0.1f, 2.0f), 0.5f));
        transform_pool_.push_back(std::make_unique<Sharpen>(1.0f + mag));
        transform_pool_.push_back(std::make_unique<RandomErasing>(
            0.5f, std::make_pair(0.02f, 0.2f * mag + 0.02f),
            std::make_pair(0.3f, 3.3f), 0.0f));
    }
};

// ============================================================================
// AutoAugment - Pre-defined augmentation policies
// ============================================================================

enum class AutoAugmentPolicy {
    ImageNet,
    CIFAR10,
    SVHN
};

class AutoAugment : public Transform {
public:
    AutoAugment(AutoAugmentPolicy policy = AutoAugmentPolicy::ImageNet)
        : policy_(policy) {
        params_["policy"] = static_cast<int>(policy);
        buildPolicy();
    }

    Image apply(const Image& input) const override {
        if (!input.isValid() || sub_policies_.empty()) return input;

        // Randomly select a sub-policy
        int idx = randomInt(0, static_cast<int>(sub_policies_.size()) - 1);
        return sub_policies_[idx]->apply(input);
    }

    std::string name() const override { return "AutoAugment"; }
    std::string category() const override { return "Advanced"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<AutoAugment>(policy_);
    }

private:
    AutoAugmentPolicy policy_;
    std::vector<std::unique_ptr<Compose>> sub_policies_;

    int randomInt(int min, int max) const {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(rng_);
    }

    void buildPolicy() {
        // Simplified ImageNet policy
        // Full policy would have 25 sub-policies with specific parameters

        auto makeSubPolicy = [](std::unique_ptr<Transform> t1, float p1,
                                std::unique_ptr<Transform> t2, float p2) {
            auto compose = std::make_unique<Compose>();
            t1->setProbability(p1);
            t2->setProbability(p2);
            compose->add(std::move(t1));
            compose->add(std::move(t2));
            return compose;
        };

        switch (policy_) {
            case AutoAugmentPolicy::ImageNet:
                sub_policies_.push_back(makeSubPolicy(
                    std::make_unique<Posterize>(4), 0.4f,
                    std::make_unique<RandomRotation>(30), 0.6f));
                sub_policies_.push_back(makeSubPolicy(
                    std::make_unique<Solarize>(0.5f), 0.6f,
                    std::make_unique<ColorJitter>(0.3f, 0.3f, 0.3f, 0.0f), 0.6f));
                sub_policies_.push_back(makeSubPolicy(
                    std::make_unique<RandomHorizontalFlip>(), 0.8f,
                    std::make_unique<RandomRotation>(15), 0.6f));
                sub_policies_.push_back(makeSubPolicy(
                    std::make_unique<ColorJitter>(0.5f, 0.5f, 0.5f, 0.1f), 0.8f,
                    std::make_unique<Sharpen>(1.5f), 0.4f));
                break;

            case AutoAugmentPolicy::CIFAR10:
                sub_policies_.push_back(makeSubPolicy(
                    std::make_unique<RandomHorizontalFlip>(), 0.5f,
                    std::make_unique<Cutout>(1, 8), 0.5f));
                sub_policies_.push_back(makeSubPolicy(
                    std::make_unique<ColorJitter>(0.4f, 0.4f, 0.4f, 0.0f), 0.7f,
                    std::make_unique<RandomRotation>(15), 0.5f));
                break;

            case AutoAugmentPolicy::SVHN:
                sub_policies_.push_back(makeSubPolicy(
                    std::make_unique<RandomRotation>(10), 0.9f,
                    std::make_unique<Solarize>(0.6f), 0.3f));
                sub_policies_.push_back(makeSubPolicy(
                    std::make_unique<ColorJitter>(0.2f, 0.2f, 0.2f, 0.0f), 0.6f,
                    std::make_unique<Invert>(), 0.2f));
                break;
        }
    }
};

// ============================================================================
// TrivialAugmentWide - Simple but effective augmentation
// ============================================================================

class TrivialAugmentWide : public Transform {
public:
    TrivialAugmentWide() {
        buildTransformPool();
    }

    Image apply(const Image& input) const override {
        if (!input.isValid() || transforms_.empty()) return input;

        // Apply single randomly selected transform with random magnitude
        int idx = randomInt(0, static_cast<int>(transforms_.size()) - 1);
        return transforms_[idx]->apply(input);
    }

    std::string name() const override { return "TrivialAugmentWide"; }
    std::string category() const override { return "Advanced"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<TrivialAugmentWide>();
    }

private:
    std::vector<std::unique_ptr<Transform>> transforms_;

    int randomInt(int min, int max) const {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(rng_);
    }

    void buildTransformPool() {
        // Add transforms with wide range of magnitudes
        transforms_.push_back(std::make_unique<RandomHorizontalFlip>(1.0f));
        transforms_.push_back(std::make_unique<RandomRotation>(45.0f));
        transforms_.push_back(std::make_unique<ColorJitter>(0.9f, 0.9f, 0.9f, 0.1f));
        transforms_.push_back(std::make_unique<Sharpen>(2.0f));
        transforms_.push_back(std::make_unique<Posterize>(2));
        transforms_.push_back(std::make_unique<Solarize>(0.3f));
        transforms_.push_back(std::make_unique<RandomGrayscale>(1.0f));
        transforms_.push_back(std::make_unique<GaussianBlur>(5, 1.5f));
        transforms_.push_back(std::make_unique<RandomErasing>(1.0f));
    }
};

} // namespace transforms
} // namespace cyxwiz
