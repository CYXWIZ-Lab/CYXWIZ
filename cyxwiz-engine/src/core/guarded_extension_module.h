#pragma once

// The module the model builder puts around every module an extension creates
// (TOFIX125 P2). It checks what custom code returns and names the node in
// every error, so a wrong shape from an extension stops training at that node
// instead of surfacing two layers later in a built-in module.
// GuardedExtensionLoss does the same for an extension's loss (P7).

#include <cyxwiz/losses/loss_base.h>
#include <cyxwiz/sequential.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

class GuardedExtensionModule : public Module {
public:
    // expected_sample_shape: the output shape the compiler inferred, per
    // sample (without the batch dimension). lease keeps the provider loaded
    // while this module exists.
    GuardedExtensionModule(std::unique_ptr<Module> inner, std::string node_name,
                           std::string type_id, std::vector<size_t> expected_sample_shape,
                           std::shared_ptr<void> lease);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;

    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::map<std::string, Tensor> GetGradients() override;
    bool HasParameters() const override;
    std::string GetName() const override;
    void SetTraining(bool training) override;

    IncrementalDecoding GetIncrementalDecoding() const override;
    Tensor ForwardIncremental(const Tensor& input, size_t position_offset) override;
    void ResetIncrementalState() override;

    const Module& Inner() const { return *inner_; }
    const std::string& TypeId() const { return type_id_; }
    const std::string& NodeName() const { return node_name_; }  // as on the canvas

private:
    [[noreturn]] void Fail(const std::string& what) const;
    void CheckOutput(const Tensor& input, const Tensor& output) const;

    std::unique_ptr<Module> inner_;
    std::string node_name_;
    std::string type_id_;
    std::vector<size_t> expected_sample_shape_;
    std::shared_ptr<void> lease_;
    std::vector<size_t> last_input_shape_;
    bool gradients_checked_ = false;
};

// Around a loss an extension creates: Forward must give one value and
// Backward a gradient with the predictions' shape; errors name the node.
// The reduction is the inner loss's, so the trainer weighs it the same way.
class GuardedExtensionLoss : public Loss {
public:
    GuardedExtensionLoss(std::unique_ptr<Loss> inner, std::string node_name,
                         std::string type_id, std::shared_ptr<void> lease);

    Tensor Forward(const Tensor& predictions, const Tensor& targets) override;
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override;
    std::string GetName() const override;
    const Tensor* GetLastMeanReductionDenominator() const override;

    const Loss& Inner() const { return *inner_; }
    const std::string& TypeId() const { return type_id_; }

private:
    [[noreturn]] void Fail(const std::string& what) const;

    std::unique_ptr<Loss> inner_;
    std::string node_name_;
    std::string type_id_;
    std::shared_ptr<void> lease_;
};

// "[2, 8, 32]".
std::string FormatTensorShape(const std::vector<size_t>& shape);

}  // namespace cyxwiz
