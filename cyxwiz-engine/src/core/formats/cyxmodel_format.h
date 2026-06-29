#pragma once

#include "../model_format.h"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <filesystem>

namespace cyxwiz {
namespace formats {

/**
 * CyxModel format handler
 *
 * The .cyxmodel format is a ZIP archive (or directory) containing:
 * - manifest.json: Model metadata and version info
 * - graph.cyxgraph: Node graph definition (JSON)
 * - config.json: Training configuration
 * - history.json: Training history (optional)
 * - weights/: Directory containing binary tensor files
 *   - manifest.json: Tensor metadata
 *   - layer0_weight.bin, layer0_bias.bin, etc.
 * - optimizer/: Directory containing optimizer state (optional)
 *   - state.json: Optimizer configuration
 *   - momentum.bin, etc.
 */
class CyxModelFormat {
public:
    CyxModelFormat() = default;
    ~CyxModelFormat() = default;

    /**
     * Create a .cyxmodel archive from components
     * @param output_path Path to output .cyxmodel file
     * @param manifest Model manifest
     * @param graph_json Node graph as JSON string
     * @param config Training configuration
     * @param history Training history (optional)
     * @param weights Map of parameter_name -> tensor data
     * @param optimizer_state Optimizer state tensors (optional)
     * @param options Export options
     * @return true on success
     */
    bool Create(
        const std::string& output_path,
        const ModelManifest& manifest,
        const std::string& graph_json,
        const TrainingConfig& config,
        const TrainingHistory* history,
        const std::map<std::string, std::vector<uint8_t>>& weights,
        const std::map<std::string, std::vector<int64_t>>& weight_shapes,
        const std::map<std::string, std::vector<uint8_t>>* optimizer_state,
        const ExportOptions& options
    );

    /**
     * Extract and read a .cyxmodel archive
     * @param input_path Path to .cyxmodel file
     * @param manifest Output manifest
     * @param graph_json Output graph JSON
     * @param config Output training config
     * @param history Output training history (optional)
     * @param weights Output weights map
     * @param weight_shapes Output weight shapes
     * @param optimizer_state Output optimizer state (optional)
     * @param options Import options
     * @return true on success
     */
    bool Extract(
        const std::string& input_path,
        ModelManifest& manifest,
        std::string& graph_json,
        TrainingConfig& config,
        TrainingHistory* history,
        std::map<std::string, std::vector<uint8_t>>& weights,
        std::map<std::string, std::vector<int64_t>>& weight_shapes,
        std::map<std::string, std::vector<uint8_t>>* optimizer_state,
        const ImportOptions& options
    );

    /**
     * Probe a .cyxmodel file for metadata without full extraction
     * @param input_path Path to .cyxmodel file
     * @return ProbeResult with metadata
     */
    ProbeResult Probe(const std::string& input_path);

    /**
     * Extract only the graph JSON from a .cyxmodel file
     * @param input_path Path to .cyxmodel file
     * @return Graph JSON string or empty on error
     */
    std::string ExtractGraphOnly(const std::string& input_path);

    /**
     * Extract tokenizer deployment assets from a .cyxmodel package.
     * @param input_path Path to .cyxmodel directory/archive
     * @param config_json Output tokenizer/config.json contents, if present
     * @param vocab_text Output tokenizer/vocab.txt contents, if present
     * @return true when at least one tokenizer asset is present
     */
    bool ExtractTextTokenizerAssets(
        const std::string& input_path,
        std::string& config_json,
        std::string& vocab_text
    );

    /**
     * Extract sequence vocabulary assets from a .cyxmodel package.
     * @param input_path Path to .cyxmodel directory/archive
     * @param token_vocab_text Output sequence/token vocabulary contents, if present
     * @param pos_vocab_text Output sequence/pos vocabulary contents, if present
     * @param tag_vocab_text Output sequence/tag vocabulary contents, if present
     * @return true when any sequence vocabulary asset is present
     */
    bool ExtractSequenceVocabularyAssets(
        const std::string& input_path,
        std::string& token_vocab_text,
        std::string& pos_vocab_text,
        std::string& tag_vocab_text
    );

    /**
     * Extract a native tree-family model artifact from a .cyxmodel package.
     * @param input_path Path to .cyxmodel directory/archive
     * @param artifact_json Output cyxwiz_tree_model JSON payload
     * @return true when tree/model.json is present
     */
    bool ExtractTreeModelArtifact(
        const std::string& input_path,
        std::string& artifact_json
    );

    /**
     * Get last error message
     */
    const std::string& GetLastError() const { return last_error_; }

    // JSON serialization helpers
    static nlohmann::json ManifestToJson(const ModelManifest& manifest);
    static ModelManifest JsonToManifest(const nlohmann::json& j);

    static nlohmann::json ConfigToJson(const TrainingConfig& config);
    static TrainingConfig JsonToConfig(const nlohmann::json& j);

    static nlohmann::json HistoryToJson(const TrainingHistory& history);
    static TrainingHistory JsonToHistory(const nlohmann::json& j);

    static nlohmann::json WeightsManifestToJson(const WeightsManifest& manifest);
    static WeightsManifest JsonToWeightsManifest(const nlohmann::json& j);

private:
    std::string last_error_;

    // Binary tensor serialization
    std::vector<uint8_t> SerializeTensorWithHeader(
        const std::vector<uint8_t>& data,
        const std::vector<int64_t>& shape,
        TensorDType dtype
    );

    bool DeserializeTensorWithHeader(
        const std::vector<uint8_t>& data,
        std::vector<uint8_t>& tensor_data,
        std::vector<int64_t>& shape,
        TensorDType& dtype
    );

    // ZIP operations (using directory fallback if minizip not available)
    bool CreateArchive(
        const std::string& output_path,
        const std::map<std::string, std::vector<uint8_t>>& files,
        bool compress
    );

    bool ExtractArchive(
        const std::string& input_path,
        std::map<std::string, std::vector<uint8_t>>& files
    );

    // Directory-based fallback operations
    bool CreateDirectory(
        const std::string& output_path,
        const std::map<std::string, std::vector<uint8_t>>& files
    );

    bool ReadDirectory(
        const std::string& input_path,
        std::map<std::string, std::vector<uint8_t>>& files
    );

    // Utility
    std::string GetTimestamp();
    bool IsZipFile(const std::string& path);
};

} // namespace formats
} // namespace cyxwiz
