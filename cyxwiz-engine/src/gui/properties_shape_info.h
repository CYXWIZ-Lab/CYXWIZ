#pragma once

#include "node_editor.h"
#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace gui::properties_shape {

struct LayerParameters {
    std::vector<size_t> weight_shape;
    std::vector<size_t> bias_shape;
    size_t weight_count = 0;
    size_t bias_count = 0;
    size_t total_params = 0;
    bool has_parameters = false;
};

struct NodeShapeInfo {
    std::vector<size_t> input_shape;
    std::vector<size_t> output_shape;
    size_t input_size = 0;
    size_t output_size = 0;
    bool is_valid = false;
    std::string error;
    LayerParameters params;
};

std::string FormatShape(const std::vector<size_t>& shape);
std::string FormatShapeMatrix(const std::vector<size_t>& shape, size_t batch_size);
size_t GetBatchSize(NodeEditor* node_editor);
std::vector<size_t> GetInputShapeFromDataset(NodeEditor* node_editor);
std::vector<size_t> InferOutputShape(
    NodeEditor* node_editor,
    NodeType type,
    const std::vector<size_t>& input_shape,
    const std::map<std::string, std::string>& params);
LayerParameters ComputeLayerParameters(
    NodeType type,
    const std::vector<size_t>& input_shape,
    const std::map<std::string, std::string>& params);
NodeShapeInfo ComputeNodeShape(NodeEditor* node_editor, int node_id);

} // namespace gui::properties_shape
