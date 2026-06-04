#pragma once

#include "compiled_graph_plan.h"
#include "executable_model.h"

#include <cyxwiz/sequential.h>

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

/**
 * Pin-aware graph executable.
 *
 * Consumes CompiledGraphPlan and executes selected layer nodes plus explicitly
 * enabled graph-op fan-in nodes. Graph ops stay opt-in via graph_op_node_ids so
 * unsupported multi-input/linalg nodes remain blocked at compile/build time.
 */
class GraphExecutableModel final : public IExecutableModel {
public:
    GraphExecutableModel(std::unique_ptr<SequentialModel> model,
                         CompiledGraphPlan plan,
                         std::vector<int> layer_node_ids);
    GraphExecutableModel(std::unique_ptr<SequentialModel> model,
                         CompiledGraphPlan plan,
                         std::vector<int> layer_node_ids,
                         std::vector<int> graph_op_node_ids);

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
    const std::vector<int>& GraphOpNodeIds() const { return graph_op_node_ids_; }
    const Tensor* FindCachedTensor(int node_id, int pin_id) const;

private:
    bool IsLayerNode(int node_id, size_t* module_index = nullptr) const;
    bool IsGraphOpNode(int node_id) const;
    void CacheTensor(int node_id, int pin_id, const Tensor& tensor);

    std::unique_ptr<SequentialModel> model_;
    CompiledGraphPlan plan_;
    std::vector<int> layer_node_ids_;
    std::vector<int> graph_op_node_ids_;
    std::map<std::pair<int, int>, Tensor> tensor_cache_;
};

} // namespace cyxwiz
