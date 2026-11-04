#pragma once

#include <cyxwiz/optimizer.h>
#include <cyxwiz/sequential.h>
#include <cyxwiz/tensor.h>

#include <functional>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>

namespace cyxwiz {

/**
 * Narrow training-runtime model interface.
 *
 * Existing training still owns SequentialModel directly. This interface is the
 * compatibility bridge for future graph execution, where a model may need to
 * execute by node/pin plan instead of a single ordered module list.
 */
class IExecutableModel {
public:
    // Called after each layer module's forward, in run order: the module's
    // index in the model, its (first) input, its output and the time it took
    // (Local Debug's per-layer trace, TOFIX178 6.0a). While unset nothing is
    // called and the forward runs as usual.
    using LayerForwardObserver = std::function<void(
        size_t module_index, const Tensor& input, const Tensor& output, float forward_ms)>;

    virtual ~IExecutableModel() = default;

    virtual Tensor Forward(const Tensor& input) = 0;
    virtual Tensor Backward(const Tensor& grad_output) = 0;
    virtual bool SupportsSparseCsrInput() const { return false; }
    virtual Tensor ForwardSparseCsr(
        const LinearSparseCsrBatchView& /*input*/) {
        throw std::runtime_error(
            "Executable model does not support sparse CSR input");
    }
    virtual void BackwardSparseCsr(
        const LinearSparseCsrBatchView& /*input*/,
        const Tensor& /*grad_output*/) {
        throw std::runtime_error(
            "Executable model does not support sparse CSR input");
    }
    virtual void SetTraining(bool training) = 0;

    virtual std::map<std::string, Tensor> GetParameters() = 0;
    virtual void SetParameters(const std::map<std::string, Tensor>& params) = 0;
    virtual std::map<std::string, Tensor> GetGradients() = 0;
    // The GetParameters() keys that are state, not trained (BatchNorm's
    // running statistics): no gradient is expected for them.
    virtual std::set<std::string> GetStateParameterNames() const { return {}; }
    virtual void UpdateParameters(Optimizer* optimizer) = 0;
    // Layers that hold parameters (graph operations such as Add are not counted).
    virtual size_t LayerCount() const = 0;
    // The layer module at a module index (BuiltModuleProvenance::module_index).
    virtual Module* LayerModule(size_t module_index) = 0;

    void SetLayerForwardObserver(LayerForwardObserver observer) {
        layer_forward_observer_ = std::move(observer);
    }

    // For checkpoint / export / save paths that still need a real
    // SequentialModel; a graph model (GraphExecutableModel) returns nullptr.
    virtual SequentialModel* AsSequentialModel() { return nullptr; }
    virtual const SequentialModel* AsSequentialModel() const { return nullptr; }

protected:
    LayerForwardObserver layer_forward_observer_;
};

class SequentialExecutableModel final : public IExecutableModel {
public:
    // Owns the model, or shares it (a test of a model someone else holds).
    explicit SequentialExecutableModel(std::shared_ptr<SequentialModel> model);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    bool SupportsSparseCsrInput() const override;
    Tensor ForwardSparseCsr(
        const LinearSparseCsrBatchView& input) override;
    void BackwardSparseCsr(
        const LinearSparseCsrBatchView& input,
        const Tensor& grad_output) override;
    void SetTraining(bool training) override;

    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::map<std::string, Tensor> GetGradients() override;
    std::set<std::string> GetStateParameterNames() const override { return model_->GetStateParameterNames(); }
    void UpdateParameters(Optimizer* optimizer) override;
    size_t LayerCount() const override { return model_->Size(); }
    Module* LayerModule(size_t module_index) override { return model_->GetModule(module_index); }

    SequentialModel* AsSequentialModel() override { return model_.get(); }
    const SequentialModel* AsSequentialModel() const override { return model_.get(); }

private:
    std::shared_ptr<SequentialModel> model_;
};

} // namespace cyxwiz
