#pragma once

#include "compiled_graph_plan.h"
#include "executable_model.h"

#include <cyxwiz/sequential.h>

#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

/**
 * Parity-only graph executable.
 *
 * This is the first executable consumer of CompiledGraphPlan. It accepts only
 * a single tensor chain from Data to Loss and delegates module execution to the
 * existing SequentialModel backend. Fan-in nodes remain blocked until this
 * class grows real node/pin tensor storage.
 */
class GraphExecutableModel final : public IExecutableModel {
public:
    GraphExecutableModel(std::unique_ptr<SequentialModel> model,
                         CompiledGraphPlan plan,
                         std::vector<int> layer_node_ids);

    static bool CanRunLinearPlan(const CompiledGraphPlan& plan,
                                 const std::vector<int>& layer_node_ids,
                                 std::string* reason = nullptr);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    void SetTraining(bool training) override;

    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::map<std::string, Tensor> GetGradients() override;
    void UpdateParameters(Optimizer* optimizer) override;

    const CompiledGraphPlan& Plan() const { return plan_; }
    const std::vector<int>& LayerNodeIds() const { return layer_node_ids_; }

private:
    std::unique_ptr<SequentialModel> model_;
    CompiledGraphPlan plan_;
    std::vector<int> layer_node_ids_;
};

} // namespace cyxwiz
