#pragma once

#include <cyxwiz/optimizer.h>
#include <cyxwiz/sequential.h>
#include <cyxwiz/tensor.h>

#include <map>
#include <memory>
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
    virtual void UpdateParameters(Optimizer* optimizer) = 0;

    // Transitional compatibility for checkpoint/export paths that still need
    // a real SequentialModel. GraphExecutableModel will return nullptr here.
    virtual SequentialModel* AsSequentialModel() { return nullptr; }
    virtual const SequentialModel* AsSequentialModel() const { return nullptr; }
    virtual std::unique_ptr<SequentialModel> ReleaseSequentialModel() { return nullptr; }
};

class SequentialExecutableModel final : public IExecutableModel {
public:
    explicit SequentialExecutableModel(std::unique_ptr<SequentialModel> model);

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
    void UpdateParameters(Optimizer* optimizer) override;

    SequentialModel* AsSequentialModel() override { return model_.get(); }
    const SequentialModel* AsSequentialModel() const override { return model_.get(); }
    std::unique_ptr<SequentialModel> ReleaseSequentialModel() override;

private:
    std::unique_ptr<SequentialModel> model_;
};

} // namespace cyxwiz
