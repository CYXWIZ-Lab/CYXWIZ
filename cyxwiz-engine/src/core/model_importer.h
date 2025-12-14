#pragma once

#include "model_format.h"
#include "formats/cyxmodel_format.h"
#include <cyxwiz/sequential.h>
#include <cyxwiz/optimizer.h>
#include <string>
#include <memory>
#include <functional>
#include <optional>

namespace cyxwiz {

// Forward declarations
struct TrainingMetrics;

/**
 * Model Importer - Load trained models from various formats
 *
 * Supported formats:
 * - .cyxmodel (Native): Complete model with graph, weights, config, history
 * - .onnx (ONNX): Industry standard interchange format
 * - .safetensors: Safe tensor serialization (HuggingFace)
 * - .gguf: GGML Universal Format for LLM inference
 */
class ModelImporter {
public:
    // Progress callback: (current_step, total_steps, status_message)
    using ProgressCallback = std::function<void(int, int, const std::string&)>;

    ModelImporter() = default;
    ~ModelImporter() = default;

    /**
     * Probe a model file for metadata without full loading
     * Useful for displaying file info before import
     *
     * @param input_path Path to model file
     * @return ProbeResult with format, metadata, and layer info
     */
    ProbeResult ProbeFile(const std::string& input_path);

    /**
     * Import a model from the specified file
     *
     * @param input_path Source file path
     * @param model Output: SequentialModel to populate
     * @param options Import options
     * @param progress_cb Progress callback (optional)
     * @return ImportResult with success status and details
     */
    ImportResult Import(
        const std::string& input_path,
        SequentialModel& model,
        const ImportOptions& options,
        ProgressCallback progress_cb = nullptr
    );

    /**
     * Import model from CyxModel format (.cyxmodel)
     * Supports both directory format and binary CYXW format
     */
    ImportResult ImportCyxModel(
        const std::string& input_path,
        SequentialModel& model,
        const ImportOptions& options,
        ProgressCallback progress_cb = nullptr
    );

    /**
     * Import model from binary CYXW format (.cyxmodel single file)
     * Magic: 0x43595857 ("CYXW"), Version: 2
     */
    ImportResult ImportCyxModelBinary(
        const std::string& input_path,
        SequentialModel& model,
        const ImportOptions& options,
        ProgressCallback progress_cb = nullptr
    );

    /**
     * Import model from ONNX format (.onnx)
     * Note: Requires CYXWIZ_HAS_ONNX compile flag
     */
    ImportResult ImportONNX(
        const std::string& input_path,
        SequentialModel& model,
        const ImportOptions& options,
        ProgressCallback progress_cb = nullptr
    );

    /**
     * Import model from Safetensors format (.safetensors)
     */
    ImportResult ImportSafetensors(
        const std::string& input_path,
        SequentialModel& model,
        const ImportOptions& options,
        ProgressCallback progress_cb = nullptr
    );

    /**
     * Import model from GGUF format (.gguf)
     */
    ImportResult ImportGGUF(
        const std::string& input_path,
        SequentialModel& model,
        const ImportOptions& options,
        ProgressCallback progress_cb = nullptr
    );

    /**
     * Extract only the graph JSON from a .cyxmodel file
     * Useful for loading graph into NodeEditor without weights
     *
     * @param input_path Path to .cyxmodel file
     * @return Graph JSON string, or empty on error
     */
    std::optional<std::string> ExtractGraph(const std::string& input_path);

    /**
     * Extract training history from a .cyxmodel file
     *
     * @param input_path Path to .cyxmodel file
     * @return TrainingHistory or empty if not available
     */
    std::optional<TrainingHistory> ExtractHistory(const std::string& input_path);

    /**
     * Get supported import formats
     */
    static std::vector<ModelFormat> GetSupportedFormats();

    /**
     * Check if a format is supported for import
     */
    static bool IsFormatSupported(ModelFormat format);

    /**
     * Detect model format from file path or magic bytes
     */
    static ModelFormat DetectFormat(const std::string& path);

    /**
     * Get last error message
     */
    const std::string& GetLastError() const { return last_error_; }

private:
    std::string last_error_;

    // Helper to create tensor from byte data
    Tensor BytesToTensor(
        const std::vector<uint8_t>& data,
        const std::vector<int64_t>& shape,
        TensorDType dtype
    );

    // Helper to populate model from weights map
    bool PopulateModelWeights(
        SequentialModel& model,
        const std::map<std::string, std::vector<uint8_t>>& weights,
        const std::map<std::string, std::vector<int64_t>>& shapes,
        const ImportOptions& options,
        std::vector<std::string>& warnings
    );

    // Validate that model architecture matches weights
    bool ValidateModelArchitecture(
        SequentialModel& model,
        const std::map<std::string, std::vector<int64_t>>& weight_shapes,
        std::string& error_message
    );
};

} // namespace cyxwiz
