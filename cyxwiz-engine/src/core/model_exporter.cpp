#include "model_exporter.h"
#include "training_executor.h"
#include <spdlog/spdlog.h>
#include <chrono>
#include <filesystem>
#include <cstring>

namespace cyxwiz {

ExportResult ModelExporter::Export(
    SequentialModel& model,
    const Optimizer* optimizer,
    const TrainingMetrics* training_metrics,
    const std::string& graph_json,
    const std::string& output_path,
    const ExportOptions& options,
    ProgressCallback progress_cb
) {
    auto start_time = std::chrono::steady_clock::now();

    // Determine format from options or path
    ModelFormat format = options.format;
    if (format == ModelFormat::Unknown) {
        format = DetectFormat(output_path);
    }

    ExportResult result;

    switch (format) {
        case ModelFormat::CyxModel:
            result = ExportCyxModel(model, optimizer, training_metrics,
                                    graph_json, output_path, options, progress_cb);
            break;

        case ModelFormat::ONNX:
            result = ExportONNX(model, output_path, options, progress_cb);
            break;

        case ModelFormat::Safetensors:
            result = ExportSafetensors(model, output_path, options, progress_cb);
            break;

        case ModelFormat::GGUF:
            result = ExportGGUF(model, output_path, options, progress_cb);
            break;

        default:
            result.success = false;
            result.error_message = "Unknown or unsupported export format";
            return result;
    }

    auto end_time = std::chrono::steady_clock::now();
    result.export_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();

    return result;
}

ExportResult ModelExporter::ExportCyxModel(
    SequentialModel& model,
    const Optimizer* optimizer,
    const TrainingMetrics* training_metrics,
    const std::string& graph_json,
    const std::string& output_path,
    const ExportOptions& options,
    ProgressCallback progress_cb
) {
    ExportResult result;
    result.output_path = output_path;

    if (progress_cb) progress_cb(0, 6, "Preparing model data...");

    try {
        // 1. Create manifest
        ModelManifest manifest = CreateManifest(model, training_metrics, options);

        if (progress_cb) progress_cb(1, 6, "Extracting weights...");

        // 2. Extract weights from model
        std::map<std::string, std::vector<uint8_t>> weights;
        std::map<std::string, std::vector<int64_t>> weight_shapes;

        auto params = model.GetParameters();
        result.num_parameters = 0;
        result.total_tensor_bytes = 0;

        for (const auto& [name, tensor] : params) {
            weights[name] = TensorToBytes(tensor);
            weight_shapes[name] = GetTensorShape(tensor);
            result.num_parameters += static_cast<int>(tensor.NumElements());
            result.total_tensor_bytes += tensor.NumBytes();
        }

        result.num_layers = static_cast<int>(model.Size());
        manifest.num_parameters = result.num_parameters;
        manifest.num_layers = result.num_layers;

        if (progress_cb) progress_cb(2, 6, "Creating training config...");

        // 3. Create training config
        TrainingConfig config = CreateTrainingConfig(optimizer, training_metrics);

        if (progress_cb) progress_cb(3, 6, "Creating training history...");

        // 4. Create training history
        TrainingHistory* history_ptr = nullptr;
        TrainingHistory history;
        if (options.include_training_history && training_metrics) {
            history = CreateTrainingHistory(training_metrics);
            history_ptr = &history;
            manifest.has_training_history = true;
        }

        if (progress_cb) progress_cb(4, 6, "Extracting optimizer state...");

        // 5. Extract optimizer state (if requested)
        // Note: Optimizer state export requires adding GetState() to Optimizer interface
        // For now, we store basic optimizer info in the config
        std::map<std::string, std::vector<uint8_t>>* optimizer_state_ptr = nullptr;
        if (options.include_optimizer_state && optimizer) {
            // TODO: Implement optimizer state export when Optimizer::GetState() is added
            // For now, optimizer state is not exported but config contains optimizer type/lr
            manifest.has_optimizer_state = false;
        }

        // Update manifest flags
        manifest.has_graph = options.include_graph && !graph_json.empty();

        if (progress_cb) progress_cb(5, 6, "Writing .cyxmodel file...");

        // 6. Create the .cyxmodel archive
        formats::CyxModelFormat cyxmodel;
        bool success = cyxmodel.Create(
            output_path,
            manifest,
            options.include_graph ? graph_json : "",
            config,
            history_ptr,
            weights,
            weight_shapes,
            optimizer_state_ptr,
            options
        );

        if (!success) {
            result.success = false;
            result.error_message = cyxmodel.GetLastError();
            last_error_ = result.error_message;
            return result;
        }

        if (progress_cb) progress_cb(6, 6, "Export complete!");

        // Get file size
        if (std::filesystem::exists(output_path)) {
            if (std::filesystem::is_directory(output_path)) {
                // Directory-based format - sum up all files
                size_t total_size = 0;
                for (const auto& entry : std::filesystem::recursive_directory_iterator(output_path)) {
                    if (entry.is_regular_file()) {
                        total_size += entry.file_size();
                    }
                }
                result.file_size_bytes = total_size;
            } else {
                result.file_size_bytes = std::filesystem::file_size(output_path);
            }
        }

        result.success = true;
        spdlog::info("Exported model to {} ({} parameters, {} layers, {} bytes)",
                     output_path, result.num_parameters, result.num_layers, result.file_size_bytes);

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("Export failed: ") + e.what();
        last_error_ = result.error_message;
        spdlog::error("Model export failed: {}", e.what());
    }

    return result;
}

ExportResult ModelExporter::ExportONNX(
    SequentialModel& model,
    const std::string& output_path,
    const ExportOptions& options,
    ProgressCallback progress_cb
) {
    ExportResult result;
    result.output_path = output_path;

#ifdef CYXWIZ_HAS_ONNX
    // TODO: Implement ONNX export using onnxruntime
    // This requires building the ONNX graph from SequentialModel layers
    if (progress_cb) progress_cb(0, 1, "ONNX export not yet implemented");
    result.success = false;
    result.error_message = "ONNX export is planned but not yet implemented";
#else
    result.success = false;
    result.error_message = "ONNX support not compiled. Build with CYXWIZ_ENABLE_ONNX=ON";
#endif

    last_error_ = result.error_message;
    return result;
}

ExportResult ModelExporter::ExportSafetensors(
    SequentialModel& model,
    const std::string& output_path,
    const ExportOptions& options,
    ProgressCallback progress_cb
) {
    ExportResult result;
    result.output_path = output_path;

    if (progress_cb) progress_cb(0, 3, "Preparing tensors...");

    try {
        // Safetensors format:
        // [8 bytes] header_size (little-endian u64)
        // [header_size bytes] JSON header with tensor metadata
        // [remaining] tensor data (contiguous, aligned)

        auto params = model.GetParameters();

        // Build header JSON
        nlohmann::json header;
        size_t data_offset = 0;
        std::vector<std::pair<std::string, const Tensor*>> ordered_tensors;

        for (const auto& [name, tensor] : params) {
            nlohmann::json tensor_info;

            // Determine dtype string
            std::string dtype_str = "F32";
            switch (tensor.GetDataType()) {
                case DataType::Float32: dtype_str = "F32"; break;
                case DataType::Float64: dtype_str = "F64"; break;
                case DataType::Int32: dtype_str = "I32"; break;
                case DataType::Int64: dtype_str = "I64"; break;
                case DataType::UInt8: dtype_str = "U8"; break;
            }

            tensor_info["dtype"] = dtype_str;

            // Shape
            std::vector<int64_t> shape;
            for (size_t dim : tensor.Shape()) {
                shape.push_back(static_cast<int64_t>(dim));
            }
            tensor_info["shape"] = shape;

            // Offsets
            size_t tensor_size = tensor.NumBytes();
            tensor_info["data_offsets"] = std::vector<size_t>{data_offset, data_offset + tensor_size};

            header[name] = tensor_info;
            data_offset += tensor_size;

            ordered_tensors.push_back({name, &tensor});
            result.num_parameters += static_cast<int>(tensor.NumElements());
        }

        // Add metadata
        if (!options.model_name.empty() || !options.author.empty()) {
            nlohmann::json metadata;
            if (!options.model_name.empty()) metadata["model_name"] = options.model_name;
            if (!options.author.empty()) metadata["author"] = options.author;
            if (!options.description.empty()) metadata["description"] = options.description;
            header["__metadata__"] = metadata;
        }

        if (progress_cb) progress_cb(1, 3, "Writing header...");

        // Serialize header
        std::string header_str = header.dump();

        // Pad header to 8-byte alignment
        while ((header_str.size() + 8) % 8 != 0) {
            header_str += ' ';
        }

        uint64_t header_size = header_str.size();

        // Write file
        std::ofstream file(output_path, std::ios::binary);
        if (!file) {
            result.success = false;
            result.error_message = "Failed to open output file: " + output_path;
            last_error_ = result.error_message;
            return result;
        }

        // Write header size (little-endian)
        file.write(reinterpret_cast<const char*>(&header_size), 8);

        // Write header
        file.write(header_str.c_str(), header_str.size());

        if (progress_cb) progress_cb(2, 3, "Writing tensor data...");

        // Write tensor data in order
        for (const auto& [name, tensor] : ordered_tensors) {
            file.write(static_cast<const char*>(tensor->Data()), tensor->NumBytes());
            result.total_tensor_bytes += tensor->NumBytes();
        }

        file.close();

        if (progress_cb) progress_cb(3, 3, "Export complete!");

        result.success = true;
        result.num_layers = static_cast<int>(model.Size());
        result.file_size_bytes = std::filesystem::file_size(output_path);

        spdlog::info("Exported model to Safetensors: {} ({} tensors, {} bytes)",
                     output_path, ordered_tensors.size(), result.file_size_bytes);

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("Safetensors export failed: ") + e.what();
        last_error_ = result.error_message;
        spdlog::error("Safetensors export failed: {}", e.what());
    }

    return result;
}

ExportResult ModelExporter::ExportGGUF(
    SequentialModel& model,
    const std::string& output_path,
    const ExportOptions& options,
    ProgressCallback progress_cb
) {
    ExportResult result;
    result.output_path = output_path;

    // GGUF format is complex and primarily used for LLMs
    // Basic implementation here for simple models

    if (progress_cb) progress_cb(0, 1, "GGUF export...");

    result.success = false;
    result.error_message = "GGUF export is planned for future release. Use .cyxmodel or .safetensors for now.";
    last_error_ = result.error_message;

    return result;
}

std::vector<ModelFormat> ModelExporter::GetSupportedFormats() {
    std::vector<ModelFormat> formats = {
        ModelFormat::CyxModel,
        ModelFormat::Safetensors
    };

#ifdef CYXWIZ_HAS_ONNX
    formats.push_back(ModelFormat::ONNX);
#endif

    // GGUF support coming soon
    // formats.push_back(ModelFormat::GGUF);

    return formats;
}

bool ModelExporter::IsFormatSupported(ModelFormat format) {
    switch (format) {
        case ModelFormat::CyxModel:
            return true;
        case ModelFormat::Safetensors:
            return true;
        case ModelFormat::ONNX:
#ifdef CYXWIZ_HAS_ONNX
            return true;
#else
            return false;
#endif
        case ModelFormat::GGUF:
            return false;  // Not yet implemented
        default:
            return false;
    }
}

std::string ModelExporter::GetExtension(ModelFormat format) {
    return GetFormatExtension(format);
}

ModelFormat ModelExporter::DetectFormat(const std::string& path) {
    std::filesystem::path p(path);
    std::string ext = p.extension().string();
    return GetFormatFromExtension(ext);
}

bool ModelExporter::ValidateForExport(
    SequentialModel& model,
    ModelFormat format,
    std::string& error_message
) {
    // Check if model has any layers
    if (model.Size() == 0) {
        error_message = "Model has no layers";
        return false;
    }

    // Check if model has parameters
    auto params = model.GetParameters();
    if (params.empty()) {
        error_message = "Model has no trainable parameters";
        return false;
    }

    // Format-specific validation
    switch (format) {
        case ModelFormat::ONNX:
#ifndef CYXWIZ_HAS_ONNX
            error_message = "ONNX support not compiled";
            return false;
#endif
            // TODO: Validate that all layer types are supported in ONNX
            break;

        case ModelFormat::GGUF:
            error_message = "GGUF export not yet implemented";
            return false;

        default:
            break;
    }

    return true;
}

std::vector<uint8_t> ModelExporter::TensorToBytes(const Tensor& tensor) {
    size_t num_bytes = tensor.NumBytes();
    std::vector<uint8_t> bytes(num_bytes);

    const void* data = tensor.Data();
    if (data && num_bytes > 0) {
        std::memcpy(bytes.data(), data, num_bytes);
    }

    return bytes;
}

std::vector<int64_t> ModelExporter::GetTensorShape(const Tensor& tensor) {
    const auto& shape = tensor.Shape();
    std::vector<int64_t> result;
    result.reserve(shape.size());

    for (size_t dim : shape) {
        result.push_back(static_cast<int64_t>(dim));
    }

    return result;
}

int ModelExporter::CountParameters(SequentialModel& model) {
    int total = 0;
    auto params = model.GetParameters();
    for (const auto& [name, tensor] : params) {
        total += static_cast<int>(tensor.NumElements());
    }
    return total;
}

ModelManifest ModelExporter::CreateManifest(
    SequentialModel& model,
    const TrainingMetrics* metrics,
    const ExportOptions& options
) {
    ModelManifest manifest;

    manifest.version = "1.0";
    manifest.format = "cyxmodel";
    manifest.cyxwiz_version = "0.2.0";  // TODO: Get from version header

    // Timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&time_t));
    manifest.created = buf;

    // Model info
    manifest.model_name = options.model_name.empty() ? "Untitled Model" : options.model_name;
    manifest.model_type = "SequentialModel";
    manifest.num_parameters = CountParameters(model);
    manifest.num_layers = static_cast<int>(model.Size());

    // Training summary (if available)
    if (metrics) {
        manifest.epochs_trained = metrics->current_epoch;
        manifest.final_accuracy = metrics->train_accuracy;
        manifest.final_loss = metrics->train_loss;
    }

    // Metadata
    manifest.author = options.author;
    manifest.description = options.description;
    manifest.custom_metadata = options.custom_metadata;

    return manifest;
}

TrainingConfig ModelExporter::CreateTrainingConfig(
    const Optimizer* optimizer,
    const TrainingMetrics* metrics
) {
    TrainingConfig config;

    if (optimizer) {
        // Get optimizer type name
        // Note: This is a simplified implementation
        // In a full implementation, we'd have a proper type system
        config.optimizer_type = "Adam";  // Default, should be extracted from optimizer
        config.learning_rate = static_cast<float>(optimizer->GetLearningRate());

        // Get other optimizer parameters if available
        // This depends on the optimizer interface
    }

    if (metrics) {
        config.epochs = metrics->total_epochs;
        config.batch_size = metrics->total_batches > 0 ? 32 : 0;  // Approximation
    }

    return config;
}

TrainingHistory ModelExporter::CreateTrainingHistory(const TrainingMetrics* metrics) {
    TrainingHistory history;

    if (metrics) {
        history.loss_history = metrics->loss_history;
        history.accuracy_history = metrics->accuracy_history;
        history.val_loss_history = metrics->val_loss_history;
        history.val_accuracy_history = metrics->val_accuracy_history;

        // Compute best metrics
        if (!history.loss_history.empty()) {
            auto min_it = std::min_element(history.loss_history.begin(), history.loss_history.end());
            history.best_loss = *min_it;
            history.best_epoch = static_cast<int>(std::distance(history.loss_history.begin(), min_it));
        }

        if (!history.accuracy_history.empty()) {
            auto max_it = std::max_element(history.accuracy_history.begin(), history.accuracy_history.end());
            history.best_accuracy = *max_it;
        }
    }

    return history;
}

} // namespace cyxwiz
