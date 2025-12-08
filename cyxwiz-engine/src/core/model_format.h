#pragma once

#include <string>
#include <vector>
#include <map>
#include <optional>
#include <chrono>

namespace cyxwiz {

/**
 * Supported model file formats
 */
enum class ModelFormat {
    CyxModel,       // .cyxmodel - Native complete format (ZIP archive)
    ONNX,           // .onnx - ONNX interchange format
    Safetensors,    // .safetensors - Safe tensor serialization
    GGUF,           // .gguf - GGML Universal Format for LLMs
    Unknown
};

/**
 * Quantization options for model export
 */
enum class Quantization {
    None,       // Original precision (typically float32)
    FP16,       // 16-bit floating point
    BF16,       // Brain floating point 16
    INT8,       // 8-bit integer
    INT4,       // 4-bit integer
    Q4_0,       // GGUF-specific: 4-bit quantization, type 0
    Q4_1,       // GGUF-specific: 4-bit quantization, type 1
    Q5_0,       // GGUF-specific: 5-bit quantization, type 0
    Q5_1,       // GGUF-specific: 5-bit quantization, type 1
    Q8_0        // GGUF-specific: 8-bit quantization, type 0
};

/**
 * Data type for tensor storage
 */
enum class TensorDType {
    Float32 = 0,
    Float64 = 1,
    Int32 = 2,
    Int64 = 3,
    UInt8 = 4,
    Float16 = 5,
    BFloat16 = 6,
    Int8 = 7,
    Int4 = 8
};

/**
 * Export options for model serialization
 */
struct ExportOptions {
    ModelFormat format = ModelFormat::CyxModel;

    // Content options (for CyxModel format)
    bool include_optimizer_state = true;
    bool include_training_history = true;
    bool include_graph = true;

    // Quantization
    Quantization quantization = Quantization::None;

    // ONNX-specific options
    int opset_version = 17;
    bool optimize_for_inference = true;
    std::vector<std::string> dynamic_axes;  // e.g., {"input:0", "output:0"}

    // Metadata
    std::string model_name;
    std::string author;
    std::string description;
    std::map<std::string, std::string> custom_metadata;

    // Advanced options
    bool compress = true;                   // Use compression in ZIP
    int compression_level = 6;              // 0-9, where 9 is maximum compression
};

/**
 * Import options for model loading
 */
struct ImportOptions {
    bool load_optimizer_state = false;      // Load optimizer state for resuming training
    bool load_training_history = false;     // Load training history/metrics
    bool strict_mode = true;                // Fail if layer names don't match
    bool allow_shape_mismatch = false;      // Allow loading if shapes differ
    std::string device = "cpu";             // Target device (cpu, cuda:0, etc.)
};

/**
 * Result of a model export operation
 */
struct ExportResult {
    bool success = false;
    std::string output_path;
    std::string error_message;
    size_t file_size_bytes = 0;
    int64_t export_time_ms = 0;

    // Statistics
    int num_parameters = 0;
    int num_layers = 0;
    size_t total_tensor_bytes = 0;
};

/**
 * Result of a model import operation
 */
struct ImportResult {
    bool success = false;
    std::string error_message;

    // Loaded model info
    std::string model_name;
    std::string format_version;
    int num_parameters = 0;
    int num_layers = 0;
    std::vector<std::string> layer_names;

    // Warnings (non-fatal issues)
    std::vector<std::string> warnings;

    // Load statistics
    int64_t load_time_ms = 0;
};

/**
 * Result of probing a model file (metadata without full load)
 */
struct ProbeResult {
    bool valid = false;
    ModelFormat format = ModelFormat::Unknown;
    std::string format_version;
    std::string model_name;
    std::string author;
    std::string description;

    int num_parameters = 0;
    int num_layers = 0;
    size_t file_size = 0;

    // Layer information
    std::vector<std::string> layer_names;
    std::map<std::string, std::vector<int64_t>> layer_shapes;  // layer_name -> shape

    // Training info (if available)
    std::optional<int> epochs_trained;
    std::optional<float> final_accuracy;
    std::optional<float> final_loss;

    // CyxModel-specific
    bool has_optimizer_state = false;
    bool has_training_history = false;
    bool has_graph = false;

    std::string error_message;
};

/**
 * Tensor metadata for serialization
 */
struct TensorMeta {
    std::string name;
    std::vector<int64_t> shape;
    TensorDType dtype = TensorDType::Float32;
    size_t offset = 0;          // Offset in data file
    size_t size_bytes = 0;      // Size in bytes
};

/**
 * Manifest for weights directory in .cyxmodel
 */
struct WeightsManifest {
    std::string version = "1.0";
    std::vector<TensorMeta> tensors;

    // Total statistics
    int total_tensors = 0;
    size_t total_bytes = 0;
};

/**
 * Training configuration stored in .cyxmodel
 */
struct TrainingConfig {
    // Optimizer settings
    std::string optimizer_type;     // "SGD", "Adam", "AdamW", etc.
    float learning_rate = 0.001f;
    float momentum = 0.9f;
    float weight_decay = 0.0f;
    float beta1 = 0.9f;             // Adam
    float beta2 = 0.999f;           // Adam
    float epsilon = 1e-8f;          // Adam

    // Training settings
    int batch_size = 32;
    int epochs = 0;
    std::string loss_function;      // "CrossEntropy", "MSE", etc.

    // Data info
    std::string dataset_name;
    int num_classes = 0;
    std::vector<int64_t> input_shape;
};

/**
 * Training history stored in .cyxmodel
 */
struct TrainingHistory {
    std::vector<float> loss_history;
    std::vector<float> accuracy_history;
    std::vector<float> val_loss_history;
    std::vector<float> val_accuracy_history;
    std::vector<float> learning_rate_history;

    // Per-epoch timestamps (optional)
    std::vector<int64_t> epoch_timestamps;

    // Best metrics
    float best_accuracy = 0.0f;
    float best_loss = std::numeric_limits<float>::max();
    int best_epoch = 0;
};

/**
 * Model manifest for .cyxmodel format
 */
struct ModelManifest {
    std::string version = "1.0";
    std::string format = "cyxmodel";
    std::string created;            // ISO 8601 timestamp
    std::string cyxwiz_version;

    // Model info
    std::string model_name;
    std::string model_type;         // "SequentialModel", etc.
    int num_parameters = 0;
    int num_layers = 0;

    // Training summary
    int epochs_trained = 0;
    float final_accuracy = 0.0f;
    float final_loss = 0.0f;

    // Metadata
    std::string author;
    std::string description;
    std::map<std::string, std::string> custom_metadata;

    // Content flags
    bool has_optimizer_state = false;
    bool has_training_history = false;
    bool has_graph = false;
};

// Utility functions

/**
 * Get file extension for a model format
 */
inline std::string GetFormatExtension(ModelFormat format) {
    switch (format) {
        case ModelFormat::CyxModel:     return ".cyxmodel";
        case ModelFormat::ONNX:         return ".onnx";
        case ModelFormat::Safetensors:  return ".safetensors";
        case ModelFormat::GGUF:         return ".gguf";
        default:                        return "";
    }
}

/**
 * Get model format from file extension
 */
inline ModelFormat GetFormatFromExtension(const std::string& ext) {
    std::string lower_ext = ext;
    for (auto& c : lower_ext) c = std::tolower(c);

    if (lower_ext == ".cyxmodel") return ModelFormat::CyxModel;
    if (lower_ext == ".onnx") return ModelFormat::ONNX;
    if (lower_ext == ".safetensors") return ModelFormat::Safetensors;
    if (lower_ext == ".gguf") return ModelFormat::GGUF;
    return ModelFormat::Unknown;
}

/**
 * Get human-readable format name
 */
inline std::string GetFormatName(ModelFormat format) {
    switch (format) {
        case ModelFormat::CyxModel:     return "CyxWiz Model";
        case ModelFormat::ONNX:         return "ONNX";
        case ModelFormat::Safetensors:  return "Safetensors";
        case ModelFormat::GGUF:         return "GGUF";
        default:                        return "Unknown";
    }
}

/**
 * Get size of a data type in bytes
 */
inline size_t GetDTypeSize(TensorDType dtype) {
    switch (dtype) {
        case TensorDType::Float32:  return 4;
        case TensorDType::Float64:  return 8;
        case TensorDType::Int32:    return 4;
        case TensorDType::Int64:    return 8;
        case TensorDType::UInt8:    return 1;
        case TensorDType::Float16:  return 2;
        case TensorDType::BFloat16: return 2;
        case TensorDType::Int8:     return 1;
        case TensorDType::Int4:     return 1;  // packed, 2 values per byte
        default:                    return 0;
    }
}

/**
 * Get human-readable quantization name
 */
inline std::string GetQuantizationName(Quantization quant) {
    switch (quant) {
        case Quantization::None:  return "None (FP32)";
        case Quantization::FP16:  return "FP16";
        case Quantization::BF16:  return "BF16";
        case Quantization::INT8:  return "INT8";
        case Quantization::INT4:  return "INT4";
        case Quantization::Q4_0:  return "Q4_0";
        case Quantization::Q4_1:  return "Q4_1";
        case Quantization::Q5_0:  return "Q5_0";
        case Quantization::Q5_1:  return "Q5_1";
        case Quantization::Q8_0:  return "Q8_0";
        default:                  return "Unknown";
    }
}

} // namespace cyxwiz
