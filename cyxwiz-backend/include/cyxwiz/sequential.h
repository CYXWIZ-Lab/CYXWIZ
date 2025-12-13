#pragma once

#include "api_export.h"
#include "tensor.h"
#include "layer.h"
#include "activation.h"
#include "optimizer.h"
#include "layers/linear.h"
#include "activations/relu.h"
#include "activations/sigmoid.h"
#include "activations/tanh.h"
#include <vector>
#include <memory>
#include <string>
#include <variant>
#include <functional>

namespace cyxwiz {

/**
 * @brief Module types that can be added to a SequentialModel
 */
enum class ModuleType {
    Linear,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    Dropout,
    BatchNorm,
    Flatten,
    LeakyReLU,
    ELU,
    GELU,
    Swish,
    Mish
};

/**
 * @brief A wrapper for any layer or activation that provides a uniform interface
 */
class CYXWIZ_API Module {
public:
    virtual ~Module() = default;

    /**
     * @brief Forward pass
     * @param input Input tensor
     * @return Output tensor
     */
    virtual Tensor Forward(const Tensor& input) = 0;

    /**
     * @brief Backward pass
     * @param grad_output Gradient from next layer
     * @return Gradient w.r.t input
     */
    virtual Tensor Backward(const Tensor& grad_output) = 0;

    /**
     * @brief Get trainable parameters
     * @return Map of parameter name -> tensor (empty if no parameters)
     */
    virtual std::map<std::string, Tensor> GetParameters() { return {}; }

    /**
     * @brief Set trainable parameters
     * @param params Map of parameter name -> tensor
     */
    virtual void SetParameters(const std::map<std::string, Tensor>& params) {}

    /**
     * @brief Get parameter gradients
     * @return Map of parameter name -> gradient tensor (empty if no parameters)
     */
    virtual std::map<std::string, Tensor> GetGradients() { return {}; }

    /**
     * @brief Check if module has trainable parameters
     */
    virtual bool HasParameters() const { return false; }

    /**
     * @brief Get module name for debugging
     */
    virtual std::string GetName() const = 0;

    /**
     * @brief Set training mode (affects Dropout, BatchNorm)
     */
    virtual void SetTraining(bool training) { is_training_ = training; }

    bool IsTraining() const { return is_training_; }

    /**
     * @brief Set trainable state for transfer learning
     * @param trainable If false, parameters won't be updated during training
     */
    void SetTrainable(bool trainable) { trainable_ = trainable; }

    /**
     * @brief Check if module is trainable
     */
    bool IsTrainable() const { return trainable_; }

    /**
     * @brief Freeze the module (disable parameter updates)
     */
    void Freeze() { trainable_ = false; }

    /**
     * @brief Unfreeze the module (enable parameter updates)
     */
    void Unfreeze() { trainable_ = true; }

protected:
    bool is_training_ = true;
    bool trainable_ = true;  // For transfer learning - frozen layers won't update
    Tensor input_cache_;  // Cached input for backward pass
};

/**
 * @brief Wrapper for LinearLayer
 */
class CYXWIZ_API LinearModule : public Module {
public:
    LinearModule(size_t in_features, size_t out_features, bool use_bias = true);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::map<std::string, Tensor> GetGradients() override;
    bool HasParameters() const override { return true; }
    std::string GetName() const override;

private:
    std::unique_ptr<LinearLayer> layer_;
    size_t in_features_;
    size_t out_features_;
};

/**
 * @brief Wrapper for ReLU activation
 */
class CYXWIZ_API ReLUModule : public Module {
public:
    ReLUModule();

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::string GetName() const override { return "ReLU"; }

private:
    std::unique_ptr<ReLU> activation_;
};

/**
 * @brief Wrapper for Sigmoid activation
 */
class CYXWIZ_API SigmoidModule : public Module {
public:
    SigmoidModule();

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::string GetName() const override { return "Sigmoid"; }

private:
    std::unique_ptr<Sigmoid> activation_;
};

/**
 * @brief Wrapper for Tanh activation
 */
class CYXWIZ_API TanhModule : public Module {
public:
    TanhModule();

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::string GetName() const override { return "Tanh"; }

private:
    std::unique_ptr<Tanh> activation_;
};

/**
 * @brief Softmax activation module
 */
class CYXWIZ_API SoftmaxModule : public Module {
public:
    SoftmaxModule(int dim = -1);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::string GetName() const override { return "Softmax"; }

private:
    int dim_;
    Tensor output_cache_;  // Cache softmax output for backward
};

/**
 * @brief Dropout module for regularization
 */
class CYXWIZ_API DropoutModule : public Module {
public:
    DropoutModule(float p = 0.5f);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::string GetName() const override;

private:
    float p_;  // Dropout probability
    Tensor mask_;  // Dropout mask for backward
};

/**
 * @brief Flatten module - reshapes input to [batch, features]
 */
class CYXWIZ_API FlattenModule : public Module {
public:
    FlattenModule(int start_dim = 1);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::string GetName() const override { return "Flatten"; }

private:
    int start_dim_;
    std::vector<size_t> original_shape_;  // For backward reshape
};

/**
 * @brief Wrapper for LeakyReLU activation
 */
class CYXWIZ_API LeakyReLUModule : public Module {
public:
    LeakyReLUModule(float negative_slope = 0.01f);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::string GetName() const override;

private:
    std::unique_ptr<LeakyReLUActivation> activation_;
    float negative_slope_;
};

/**
 * @brief Wrapper for ELU activation
 */
class CYXWIZ_API ELUModule : public Module {
public:
    ELUModule(float alpha = 1.0f);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::string GetName() const override;

private:
    std::unique_ptr<ELUActivation> activation_;
    float alpha_;
};

/**
 * @brief Wrapper for GELU activation
 */
class CYXWIZ_API GELUModule : public Module {
public:
    GELUModule();

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::string GetName() const override { return "GELU"; }

private:
    std::unique_ptr<GELUActivation> activation_;
};

/**
 * @brief Wrapper for Swish activation (SiLU)
 */
class CYXWIZ_API SwishModule : public Module {
public:
    SwishModule();

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::string GetName() const override { return "Swish"; }

private:
    std::unique_ptr<SwishActivation> activation_;
};

/**
 * @brief Wrapper for Mish activation
 */
class CYXWIZ_API MishModule : public Module {
public:
    MishModule();

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::string GetName() const override { return "Mish"; }

private:
    std::unique_ptr<MishActivation> activation_;
};

/**
 * @brief Sequential model - a container for ordered layers
 *
 * Example:
 *   SequentialModel model;
 *   model.Add<LinearModule>(784, 128);
 *   model.Add<ReLUModule>();
 *   model.Add<LinearModule>(128, 10);
 */
class CYXWIZ_API SequentialModel {
public:
    SequentialModel() = default;
    ~SequentialModel() = default;

    // Non-copyable
    SequentialModel(const SequentialModel&) = delete;
    SequentialModel& operator=(const SequentialModel&) = delete;

    // Movable
    SequentialModel(SequentialModel&&) = default;
    SequentialModel& operator=(SequentialModel&&) = default;

    /**
     * @brief Add a module to the sequence
     * @tparam T Module type (LinearModule, ReLUModule, etc.)
     * @tparam Args Constructor arguments for the module
     */
    template<typename T, typename... Args>
    void Add(Args&&... args) {
        modules_.push_back(std::make_unique<T>(std::forward<Args>(args)...));
    }

    /**
     * @brief Add a pre-created module
     */
    void AddModule(std::unique_ptr<Module> module) {
        modules_.push_back(std::move(module));
    }

    /**
     * @brief Forward pass through all layers
     * @param input Input tensor
     * @return Output tensor
     */
    Tensor Forward(const Tensor& input);

    /**
     * @brief Backward pass through all layers (reverse order)
     * @param grad_output Gradient from loss function
     * @return Gradient w.r.t input (usually not needed)
     */
    Tensor Backward(const Tensor& grad_output);

    /**
     * @brief Get all trainable parameters
     * @return Map of "layer_idx.param_name" -> tensor
     */
    std::map<std::string, Tensor> GetParameters();

    /**
     * @brief Set all trainable parameters
     * @param params Map of "layer_idx.param_name" -> tensor
     */
    void SetParameters(const std::map<std::string, Tensor>& params);

    /**
     * @brief Get all parameter gradients
     * @return Map of "layer_idx.param_name" -> gradient tensor
     */
    std::map<std::string, Tensor> GetGradients();

    /**
     * @brief Apply optimizer to all parameters
     * @param optimizer Optimizer to use
     */
    void UpdateParameters(Optimizer* optimizer);

    /**
     * @brief Set training mode for all modules
     */
    void SetTraining(bool training);

    /**
     * @brief Get number of modules
     */
    size_t Size() const { return modules_.size(); }

    /**
     * @brief Get module at index
     */
    Module* GetModule(size_t index) {
        return index < modules_.size() ? modules_[index].get() : nullptr;
    }

    /**
     * @brief Get module at index (const version)
     */
    const Module* GetModule(size_t index) const {
        return index < modules_.size() ? modules_[index].get() : nullptr;
    }

    /**
     * @brief Print model summary
     */
    void Summary() const;

    // ==================== Serialization ====================

    /**
     * @brief Save model to file
     * @param path Base path (will create .json and .bin files)
     * @return true if successful
     */
    bool Save(const std::string& path) const;

    /**
     * @brief Load model weights from file
     * @param path Base path (expects .json and .bin files)
     * @return true if successful
     * @note Model architecture must already be set up before loading
     */
    bool Load(const std::string& path);

    /**
     * @brief Set model name (for metadata)
     */
    void SetName(const std::string& name) { model_name_ = name; }

    /**
     * @brief Get model name
     */
    const std::string& GetName() const { return model_name_; }

    /**
     * @brief Set model description (for metadata)
     */
    void SetDescription(const std::string& desc) { model_description_ = desc; }

    /**
     * @brief Get model description
     */
    const std::string& GetDescription() const { return model_description_; }

    // ==================== Transfer Learning ====================

    /**
     * @brief Freeze a specific layer by index
     * @param layer_idx Index of the layer to freeze
     */
    void FreezeLayer(size_t layer_idx);

    /**
     * @brief Freeze all layers up to (but not including) the given index
     * @param layer_idx First layer that remains trainable
     */
    void FreezeUpTo(size_t layer_idx);

    /**
     * @brief Freeze all layers except the last N layers
     * @param n Number of layers to keep trainable at the end
     */
    void FreezeExceptLast(size_t n);

    /**
     * @brief Unfreeze all layers
     */
    void UnfreezeAll();

    /**
     * @brief Check if a layer is trainable
     * @param layer_idx Index of the layer
     * @return true if the layer is trainable, false if frozen
     */
    bool IsLayerTrainable(size_t layer_idx) const;

private:
    std::vector<std::unique_ptr<Module>> modules_;
    std::vector<Tensor> intermediate_outputs_;  // Cached for backward pass
    std::string model_name_;
    std::string model_description_;
};

/**
 * @brief Factory function to create a module from type enum
 */
CYXWIZ_API std::unique_ptr<Module> CreateModule(
    ModuleType type,
    const std::map<std::string, std::string>& params = {}
);

} // namespace cyxwiz
