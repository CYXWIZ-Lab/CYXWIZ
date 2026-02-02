#include "onnx_exporter.h"
#include <fstream>
#include <cstring>
#include <spdlog/spdlog.h>

namespace cyxwiz::plugin::rl {

// ONNX protobuf field numbers (from onnx.proto3)
// TensorProto: dims=1, data_type=2, float_data=4, name=8, raw_data=13
// ValueInfoProto: name=1, type=2
// TypeProto: tensor_type=1
// TypeProto.Tensor: elem_type=1, shape=2
// TensorShapeProto: dim=1
// TensorShapeProto.Dimension: dim_value=1
// NodeProto: input=1, output=2, name=3, op_type=4
// GraphProto: node=1, name=2, initializer=5, input=11, output=12
// ModelProto: ir_version=1, opset_import=8, graph=7, producer_name=2
// OperatorSetIdProto: version=2

// Wire types: 0=varint, 1=64bit, 2=length-delimited, 5=32bit

void ONNXExporter::WriteVarint(std::vector<uint8_t>& buf, uint64_t value) {
    while (value > 0x7F) {
        buf.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
        value >>= 7;
    }
    buf.push_back(static_cast<uint8_t>(value));
}

void ONNXExporter::WriteTag(std::vector<uint8_t>& buf, int field, int wire_type) {
    WriteVarint(buf, static_cast<uint64_t>((field << 3) | wire_type));
}

void ONNXExporter::WriteString(std::vector<uint8_t>& buf, int field, const std::string& value) {
    WriteTag(buf, field, 2);  // length-delimited
    WriteVarint(buf, value.size());
    buf.insert(buf.end(), value.begin(), value.end());
}

void ONNXExporter::WriteBytes(std::vector<uint8_t>& buf, int field, const std::vector<uint8_t>& data) {
    WriteTag(buf, field, 2);
    WriteVarint(buf, data.size());
    buf.insert(buf.end(), data.begin(), data.end());
}

void ONNXExporter::WriteInt64(std::vector<uint8_t>& buf, int field, int64_t value) {
    WriteTag(buf, field, 0);  // varint
    WriteVarint(buf, static_cast<uint64_t>(value));
}

void ONNXExporter::WriteFloat(std::vector<uint8_t>& buf, int field, float value) {
    WriteTag(buf, field, 5);  // 32-bit
    uint8_t bytes[4];
    std::memcpy(bytes, &value, 4);
    buf.insert(buf.end(), bytes, bytes + 4);
}

void ONNXExporter::WriteSubMessage(std::vector<uint8_t>& buf, int field, const std::vector<uint8_t>& sub) {
    WriteTag(buf, field, 2);
    WriteVarint(buf, sub.size());
    buf.insert(buf.end(), sub.begin(), sub.end());
}

std::vector<uint8_t> ONNXExporter::BuildTensorProto(
    const std::string& name,
    const std::vector<float>& data,
    const std::vector<int64_t>& dims)
{
    std::vector<uint8_t> buf;

    // dims (field 1, repeated int64)
    for (auto d : dims) {
        WriteInt64(buf, 1, d);
    }

    // data_type = FLOAT (1) (field 2)
    WriteInt64(buf, 2, 1);

    // raw_data (field 13) — more compact than float_data
    std::vector<uint8_t> raw(data.size() * 4);
    std::memcpy(raw.data(), data.data(), raw.size());
    WriteBytes(buf, 13, raw);

    // name (field 8)
    WriteString(buf, 8, name);

    return buf;
}

std::vector<uint8_t> ONNXExporter::BuildValueInfoProto(
    const std::string& name,
    const std::vector<int64_t>& dims)
{
    std::vector<uint8_t> buf;

    // name (field 1)
    WriteString(buf, 1, name);

    // type (field 2) -> TypeProto
    {
        std::vector<uint8_t> type_proto;

        // tensor_type (field 1) -> TypeProto.Tensor
        {
            std::vector<uint8_t> tensor_type;

            // elem_type = FLOAT (1) (field 1)
            WriteInt64(tensor_type, 1, 1);

            // shape (field 2) -> TensorShapeProto
            {
                std::vector<uint8_t> shape;
                for (auto d : dims) {
                    // dim (field 1) -> TensorShapeProto.Dimension
                    std::vector<uint8_t> dim_proto;
                    WriteInt64(dim_proto, 1, d);  // dim_value (field 1)
                    WriteSubMessage(shape, 1, dim_proto);
                }
                WriteSubMessage(tensor_type, 2, shape);
            }

            WriteSubMessage(type_proto, 1, tensor_type);
        }

        WriteSubMessage(buf, 2, type_proto);
    }

    return buf;
}

std::vector<uint8_t> ONNXExporter::BuildNodeProto(
    const std::string& op_type,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const std::string& name)
{
    std::vector<uint8_t> buf;

    for (const auto& inp : inputs) WriteString(buf, 1, inp);
    for (const auto& out : outputs) WriteString(buf, 2, out);
    if (!name.empty()) WriteString(buf, 3, name);
    WriteString(buf, 4, op_type);

    return buf;
}

std::vector<uint8_t> ONNXExporter::BuildGraphProto(
    const PPOAgent& agent,
    const ExportConfig& config)
{
    std::vector<uint8_t> buf;

    const auto& layers = agent.GetPolicyLayers();
    int obs_dim = agent.GetObsDim();
    int act_dim = agent.GetActDim();
    int batch = config.batch_size;

    // Build computation nodes and initializers
    std::string prev_output = "observation";
    int layer_idx = 0;

    for (size_t i = 0; i < layers.size(); ++i) {
        const auto& layer = layers[i];
        std::string w_name = "policy_w" + std::to_string(i);
        std::string b_name = "policy_b" + std::to_string(i);
        std::string matmul_out = "matmul_" + std::to_string(i);
        std::string add_out = "add_" + std::to_string(i);
        std::string act_out = (i < layers.size() - 1) ? "tanh_" + std::to_string(i) : "action_mean";

        // MatMul node
        auto matmul_node = BuildNodeProto("MatMul", {prev_output, w_name}, {matmul_out}, "matmul_" + std::to_string(i));
        WriteSubMessage(buf, 1, matmul_node);

        // Add node (bias)
        auto add_node = BuildNodeProto("Add", {matmul_out, b_name}, {add_out}, "add_" + std::to_string(i));
        WriteSubMessage(buf, 1, add_node);

        // Activation: Tanh for hidden layers, none for output
        if (i < layers.size() - 1) {
            auto tanh_node = BuildNodeProto("Tanh", {add_out}, {act_out}, "tanh_" + std::to_string(i));
            WriteSubMessage(buf, 1, tanh_node);
            prev_output = act_out;
        } else {
            prev_output = add_out;
            // Rename add_out to action_mean
            auto identity_node = BuildNodeProto("Identity", {add_out}, {"action_mean"}, "output_identity");
            WriteSubMessage(buf, 1, identity_node);
        }

        // Weight initializer: shape [in_features, out_features] (transposed for MatMul)
        // PPO stores [out x in] row-major, need to transpose for ONNX MatMul (input @ weight)
        std::vector<float> w_transposed(layer.weights.size());
        for (int r = 0; r < layer.out_features; ++r) {
            for (int c = 0; c < layer.in_features; ++c) {
                w_transposed[c * layer.out_features + r] = layer.weights[r * layer.in_features + c];
            }
        }
        auto w_tensor = BuildTensorProto(w_name, w_transposed,
            {static_cast<int64_t>(layer.in_features), static_cast<int64_t>(layer.out_features)});
        WriteSubMessage(buf, 5, w_tensor);

        // Bias initializer
        auto b_tensor = BuildTensorProto(b_name, layer.bias,
            {static_cast<int64_t>(layer.out_features)});
        WriteSubMessage(buf, 5, b_tensor);

        layer_idx++;
    }

    // Graph name (field 2)
    WriteString(buf, 2, "policy_graph");

    // Input: observation (field 11)
    auto input_vi = BuildValueInfoProto("observation",
        {static_cast<int64_t>(batch), static_cast<int64_t>(obs_dim)});
    WriteSubMessage(buf, 11, input_vi);

    // Output: action_mean (field 12)
    auto output_vi = BuildValueInfoProto("action_mean",
        {static_cast<int64_t>(batch), static_cast<int64_t>(act_dim)});
    WriteSubMessage(buf, 12, output_vi);

    // Optionally add log_std as constant output
    if (config.include_log_std) {
        const auto& log_std = agent.GetLogStd();
        auto log_std_tensor = BuildTensorProto("log_std_init", log_std,
            {static_cast<int64_t>(act_dim)});
        WriteSubMessage(buf, 5, log_std_tensor);

        auto log_std_node = BuildNodeProto("Identity", {"log_std_init"}, {"log_std"}, "log_std_output");
        WriteSubMessage(buf, 1, log_std_node);

        auto log_std_vi = BuildValueInfoProto("log_std",
            {static_cast<int64_t>(act_dim)});
        WriteSubMessage(buf, 12, log_std_vi);
    }

    return buf;
}

std::vector<uint8_t> ONNXExporter::BuildModelProto(
    const std::vector<uint8_t>& graph,
    const ExportConfig& config)
{
    std::vector<uint8_t> buf;

    // ir_version = 8 (field 1)
    WriteInt64(buf, 1, 8);

    // producer_name (field 2)
    WriteString(buf, 2, config.producer);

    // producer_version (field 3)
    WriteString(buf, 3, "1.0.0");

    // model_version (field 5)
    WriteInt64(buf, 5, 1);

    // graph (field 7)
    WriteSubMessage(buf, 7, graph);

    // opset_import (field 8) -> OperatorSetIdProto
    {
        std::vector<uint8_t> opset;
        // version = 17 (field 2)
        WriteInt64(opset, 2, 17);
        WriteSubMessage(buf, 8, opset);
    }

    return buf;
}

ONNXExporter::ExportResult ONNXExporter::Export(
    const PPOAgent& agent,
    const std::string& output_path,
    const ExportConfig& config)
{
    ExportResult result;
    result.output_path = output_path;

    try {
        const auto& layers = agent.GetPolicyLayers();
        if (layers.empty()) {
            result.error_message = "Policy network has no layers";
            return result;
        }

        spdlog::info("ONNXExporter: Exporting policy (obs_dim={}, act_dim={}, {} layers)",
                      agent.GetObsDim(), agent.GetActDim(), layers.size());

        // Build ONNX model
        auto graph = BuildGraphProto(agent, config);
        auto model = BuildModelProto(graph, config);

        // Write to file
        std::ofstream out(output_path, std::ios::binary);
        if (!out.is_open()) {
            result.error_message = "Cannot open output file: " + output_path;
            return result;
        }

        out.write(reinterpret_cast<const char*>(model.data()), model.size());
        out.close();

        result.success = true;
        result.file_size = model.size();

        spdlog::info("ONNXExporter: Exported to {} ({} bytes)", output_path, result.file_size);

    } catch (const std::exception& e) {
        result.error_message = std::string("Export failed: ") + e.what();
        spdlog::error("ONNXExporter: {}", result.error_message);
    }

    return result;
}

} // namespace cyxwiz::plugin::rl
