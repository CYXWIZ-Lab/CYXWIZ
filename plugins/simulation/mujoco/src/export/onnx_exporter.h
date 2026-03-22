#pragma once

#include "../rl/ppo_agent.h"
#include <string>
#include <vector>

namespace cyxwiz::plugin::rl {

/**
 * ONNXExporter - Exports trained PPO policy network to ONNX format.
 *
 * Writes a minimal valid ONNX file containing the policy MLP
 * (linear layers + tanh activations). The exported model takes
 * observations as input and produces mean actions as output.
 *
 * No external ONNX library dependency — writes raw protobuf bytes
 * following the ONNX specification (IR version 8, opset 17).
 */
class ONNXExporter {
public:
    struct ExportConfig {
        std::string model_name = "ppo_policy";
        std::string producer = "CyxWiz Engine";
        int batch_size = 1;         // Fixed batch dimension (0 = dynamic)
        bool include_log_std = true; // Include log_std as additional output
    };

    struct ExportResult {
        bool success = false;
        std::string error_message;
        std::string output_path;
        size_t file_size = 0;
    };

    /**
     * Export PPO agent's policy network to ONNX file.
     *
     * @param agent Trained PPO agent
     * @param output_path Destination file path (.onnx)
     * @param config Export configuration
     * @return ExportResult with success status
     */
    static ExportResult Export(
        const PPOAgent& agent,
        const std::string& output_path,
        const ExportConfig& config = {}
    );

private:
    // ONNX protobuf helpers — write raw bytes
    static void WriteVarint(std::vector<uint8_t>& buf, uint64_t value);
    static void WriteTag(std::vector<uint8_t>& buf, int field, int wire_type);
    static void WriteString(std::vector<uint8_t>& buf, int field, const std::string& value);
    static void WriteBytes(std::vector<uint8_t>& buf, int field, const std::vector<uint8_t>& data);
    static void WriteInt64(std::vector<uint8_t>& buf, int field, int64_t value);
    static void WriteFloat(std::vector<uint8_t>& buf, int field, float value);
    static void WriteSubMessage(std::vector<uint8_t>& buf, int field, const std::vector<uint8_t>& sub);

    // Build ONNX graph components
    static std::vector<uint8_t> BuildTensorProto(
        const std::string& name,
        const std::vector<float>& data,
        const std::vector<int64_t>& dims
    );

    static std::vector<uint8_t> BuildValueInfoProto(
        const std::string& name,
        const std::vector<int64_t>& dims
    );

    static std::vector<uint8_t> BuildNodeProto(
        const std::string& op_type,
        const std::vector<std::string>& inputs,
        const std::vector<std::string>& outputs,
        const std::string& name = ""
    );

    static std::vector<uint8_t> BuildGraphProto(
        const PPOAgent& agent,
        const ExportConfig& config
    );

    static std::vector<uint8_t> BuildModelProto(
        const std::vector<uint8_t>& graph,
        const ExportConfig& config
    );
};

} // namespace cyxwiz::plugin::rl
