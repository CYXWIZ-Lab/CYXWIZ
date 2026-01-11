#pragma once

#include "api_export.h"
#include "tensor.h"
#include <memory>
#include <string>

namespace cyxwiz {

// ============================================================================
// Activation Types
// ============================================================================

enum class ActivationType {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    LeakyReLU,
    ELU,
    GELU,
    Swish,
    SiLU,      // Same as Swish (x * sigmoid(x))
    Mish,      // x * tanh(softplus(x))
    Hardswish, // PyTorch-style hardswish
    SELU,      // Scaled Exponential Linear Unit
    PReLU      // Parametric ReLU (learnable alpha)
};

// ============================================================================
// Base Activation Class
// ============================================================================

class CYXWIZ_API Activation {
public:
    virtual ~Activation() = default;
    virtual Tensor Forward(const Tensor& input) = 0;
    virtual Tensor Backward(const Tensor& grad_output, const Tensor& input) = 0;
    virtual std::string GetName() const { return "Activation"; }
};

// Factory function to create activation by type
CYXWIZ_API std::unique_ptr<Activation> CreateActivation(ActivationType type, float alpha = 0.01f);

// ============================================================================
// ReLU - Rectified Linear Unit
// ============================================================================

class CYXWIZ_API ReLUActivation : public Activation {
public:
    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output, const Tensor& input) override;
    std::string GetName() const override { return "ReLU"; }
};

// ============================================================================
// LeakyReLU - Leaky Rectified Linear Unit
// ============================================================================

class CYXWIZ_API LeakyReLUActivation : public Activation {
public:
    explicit LeakyReLUActivation(float alpha = 0.01f) : alpha_(alpha) {}

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output, const Tensor& input) override;
    std::string GetName() const override { return "LeakyReLU"; }

    float GetAlpha() const { return alpha_; }

private:
    float alpha_;
};

// ============================================================================
// ELU - Exponential Linear Unit
// ============================================================================

class CYXWIZ_API ELUActivation : public Activation {
public:
    explicit ELUActivation(float alpha = 1.0f) : alpha_(alpha) {}

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output, const Tensor& input) override;
    std::string GetName() const override { return "ELU"; }

    float GetAlpha() const { return alpha_; }

private:
    float alpha_;
};

// ============================================================================
// GELU - Gaussian Error Linear Unit
// ============================================================================

class CYXWIZ_API GELUActivation : public Activation {
public:
    GELUActivation() = default;

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output, const Tensor& input) override;
    std::string GetName() const override { return "GELU"; }
};

// ============================================================================
// Swish / SiLU - Self-Gated Activation
// ============================================================================

class CYXWIZ_API SwishActivation : public Activation {
public:
    SwishActivation() = default;

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output, const Tensor& input) override;
    std::string GetName() const override { return "Swish"; }
};

// Alias for Swish (PyTorch naming)
using SiLUActivation = SwishActivation;

// ============================================================================
// Sigmoid
// ============================================================================

class CYXWIZ_API SigmoidActivation : public Activation {
public:
    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output, const Tensor& input) override;
    std::string GetName() const override { return "Sigmoid"; }
};

// ============================================================================
// Tanh
// ============================================================================

class CYXWIZ_API TanhActivation : public Activation {
public:
    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output, const Tensor& input) override;
    std::string GetName() const override { return "Tanh"; }
};

// ============================================================================
// Softmax
// ============================================================================

class CYXWIZ_API SoftmaxActivation : public Activation {
public:
    explicit SoftmaxActivation(int axis = -1) : axis_(axis) {}

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output, const Tensor& input) override;
    std::string GetName() const override { return "Softmax"; }

private:
    int axis_;
    Tensor cached_output_;  // Store for backward pass
};

// ============================================================================
// Mish - Self Regularized Non-Monotonic Activation
// ============================================================================

class CYXWIZ_API MishActivation : public Activation {
public:
    MishActivation() = default;

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output, const Tensor& input) override;
    std::string GetName() const override { return "Mish"; }
};

// ============================================================================
// Hardswish - Efficient approximation of Swish
// ============================================================================

class CYXWIZ_API HardswishActivation : public Activation {
public:
    HardswishActivation() = default;

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output, const Tensor& input) override;
    std::string GetName() const override { return "Hardswish"; }
};


// ============================================================================
// SELU - Scaled Exponential Linear Unit
// ============================================================================

class CYXWIZ_API SELUActivation : public Activation {
public:
    SELUActivation() = default;

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output, const Tensor& input) override;
    std::string GetName() const override { return "SELU"; }

    // SELU constants (for self-normalization)
    static constexpr float ALPHA = 1.6732632423543772848170429916717f;
    static constexpr float SCALE = 1.0507009873554804934193349852946f;
};

// ============================================================================
// PReLU - Parametric ReLU (Learnable slope)
// ============================================================================

class CYXWIZ_API PReLUActivation : public Activation {
public:
    /**
     * Create PReLU activation with learnable parameters
     * @param num_parameters Number of learnable alpha parameters (1 or num_channels)
     * @param init Initial value for alpha (default: 0.25)
     */
    explicit PReLUActivation(int num_parameters = 1, float init = 0.25f);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output, const Tensor& input) override;
    std::string GetName() const override { return "PReLU"; }

    // Access learnable parameters
    Tensor GetAlpha() const { return alpha_; }
    void SetAlpha(const Tensor& alpha) { alpha_ = alpha; }
    Tensor GetAlphaGradient() const { return grad_alpha_; }

private:
    int num_parameters_;
    Tensor alpha_;       // Learnable negative slope [num_parameters]
    Tensor grad_alpha_;  // Gradient for alpha
};

} // namespace cyxwiz
