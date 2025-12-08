#pragma once

#include "model_format.h"
#include "formats/cyxmodel_format.h"
#include <cyxwiz/sequential.h>
#include <cyxwiz/optimizer.h>
#include <string>
#include <memory>
#include <functional>

namespace cyxwiz {

// Forward declarations
struct TrainingMetrics;

/**
 * Model Exporter - Export trained models to various formats
 *
 * Supported formats:
 * - .cyxmodel (Native): Complete model with graph, weights, config, history
 * - .onnx (ONNX): Industry standard interchange format
 * - .safetensors: Safe tensor serialization (HuggingFace)
 * - .gguf: GGML Universal Format for LLM inference
 */
class ModelExporter {
public:
    // Progress callback: (current_step, total_steps, status_message)
    using ProgressCallback = std::function<void(int, int, const std::string&)>;

    ModelExporter() = default;
    ~ModelExporter() = default;

    /**
     * Export a trained model to the specified format
     *
     * @param model Trained SequentialModel
     * @param optimizer Optimizer state (optional, for resumable training)
     * @param training_metrics Training metrics (loss, accuracy history)
     * @param graph_json Node graph as JSON string (from NodeEditor)
     * @param output_path Destination file path
     * @param options Export options (format, quantization, etc.)
     * @param progress_cb Progress callback (optional)
     * @return ExportResult with success status and details
     */
    ExportResult Export(
        SequentialModel& model,
        const Optimizer* optimizer,
        const TrainingMetrics* training_metrics,
        const std::string& graph_json,
        const std::string& output_path,
        const ExportOptions& options,
        ProgressCallback progress_cb = nullptr
    );

    /**
     * Export model to CyxModel format (.cyxmodel)
     */
    ExportResult ExportCyxModel(
        SequentialModel& model,
        const Optimizer* optimizer,
        const TrainingMetrics* training_metrics,
        const std::string& graph_json,
        const std::string& output_path,
        const ExportOptions& options,
        ProgressCallback progress_cb = nullptr
    );

    /**
     * Export model to ONNX format (.onnx)
     * Note: Requires CYXWIZ_HAS_ONNX compile flag
     */
    ExportResult ExportONNX(
        SequentialModel& model,
        const std::string& output_path,
        const ExportOptions& options,
        ProgressCallback progress_cb = nullptr
    );

    /**
     * Export model to Safetensors format (.safetensors)
     */
    ExportResult ExportSafetensors(
        SequentialModel& model,
        const std::string& output_path,
        const ExportOptions& options,
        ProgressCallback progress_cb = nullptr
    );

    /**
     * Export model to GGUF format (.gguf)
     */
    ExportResult ExportGGUF(
        SequentialModel& model,
        const std::string& output_path,
        const ExportOptions& options,
        ProgressCallback progress_cb = nullptr
    );

    /**
     * Get supported export formats
     */
    static std::vector<ModelFormat> GetSupportedFormats();

    /**
     * Check if a format is supported
     */
    static bool IsFormatSupported(ModelFormat format);

    /**
     * Get file extension for a format
     */
    static std::string GetExtension(ModelFormat format);

    /**
     * Detect format from file path
     */
    static ModelFormat DetectFormat(const std::string& path);

    /**
     * Validate that a model can be exported to a format
     *
     * @param model Model to validate
     * @param format Target format
     * @param error_message Output: description of why validation failed
     * @return true if model can be exported to format
     */
    bool ValidateForExport(
        SequentialModel& model,
        ModelFormat format,
        std::string& error_message
    );

    /**
     * Get last error message
     */
    const std::string& GetLastError() const { return last_error_; }

private:
    std::string last_error_;

    // Helper to extract tensor data to byte vector
    std::vector<uint8_t> TensorToBytes(const Tensor& tensor);

    // Helper to get tensor shape as int64 vector
    std::vector<int64_t> GetTensorShape(const Tensor& tensor);

    // Helper to count total parameters
    int CountParameters(SequentialModel& model);

    // Create ModelManifest from model and options
    ModelManifest CreateManifest(
        SequentialModel& model,
        const TrainingMetrics* metrics,
        const ExportOptions& options
    );

    // Create TrainingConfig from optimizer and model
    TrainingConfig CreateTrainingConfig(
        const Optimizer* optimizer,
        const TrainingMetrics* metrics
    );

    // Create TrainingHistory from metrics
    TrainingHistory CreateTrainingHistory(const TrainingMetrics* metrics);
};

} // namespace cyxwiz
