#pragma once

#include "../layer.h"
#include "../tensor.h"
#include "../api_export.h"
#include <string>
#include <map>

namespace cyxwiz {

/**
 * @brief Linear (fully-connected / dense) layer
 *
 * Performs: output = input @ weight^T + bias
 *
 * Parameters:
 *   - weight: shape [out_features, in_features]
 *   - bias: shape [out_features] (optional)
 *
 * Input: [batch_size, in_features] or [in_features]
 * Output: [batch_size, out_features] or [out_features]
 */
class CYXWIZ_API LinearLayer : public Layer {
public:
    /**
     * @brief Construct a Linear layer
     * @param in_features Number of input features
     * @param out_features Number of output features
     * @param use_bias Whether to include bias term (default: true)
     */
    LinearLayer(size_t in_features, size_t out_features, bool use_bias = true);

    ~LinearLayer() override = default;

    /**
     * @brief Forward pass
     * @param input Input tensor [batch, in_features] or [in_features]
     * @return Output tensor [batch, out_features] or [out_features]
     */
    Tensor Forward(const Tensor& input) override;

    /**
     * @brief Backward pass (compute gradients)
     * @param grad_output Gradient from next layer
     * @return Gradient w.r.t input
     *
     * Also computes and stores gradients for weight and bias
     */
    Tensor Backward(const Tensor& grad_output) override;

    /**
     * @brief Get layer parameters
     * @return Map of parameter name -> tensor
     */
    std::map<std::string, Tensor> GetParameters() override;

    /**
     * @brief Set layer parameters
     * @param params Map of parameter name -> tensor
     */
    void SetParameters(const std::map<std::string, Tensor>& params) override;

    /**
     * @brief Get parameter gradients
     * @return Map of parameter name -> gradient tensor
     */
    std::map<std::string, Tensor> GetGradients();

    /**
     * @brief Initialize weights using Xavier/Glorot initialization
     */
    void InitializeWeights();

    // Accessors
    size_t GetInFeatures() const { return in_features_; }
    size_t GetOutFeatures() const { return out_features_; }
    bool HasBias() const { return use_bias_; }

private:
    size_t in_features_;
    size_t out_features_;
    bool use_bias_;

    // Parameters
    Tensor weight_;  // [out_features, in_features]
    Tensor bias_;    // [out_features]

    // Gradients (computed during backward pass)
    Tensor weight_grad_;  // [out_features, in_features]
    Tensor bias_grad_;    // [out_features]

    // Cache for backward pass
    Tensor input_cache_;  // Cached input from forward pass
};

} // namespace cyxwiz
