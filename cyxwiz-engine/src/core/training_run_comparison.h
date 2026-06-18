#pragma once

#include "graph_compiler.h"
#include "training_run_comparison_record.h"
#include "training_executor.h"

#include <iomanip>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

namespace cyxwiz {

inline std::string TrainingRunComparisonCsvHeader() {
    return "run_id,run_status,dataset_name,model_family,primary_layer_type,"
           "architecture_summary,model_layer_count,epochs,batch_size,"
           "learning_rate,bidirectional,hidden_size,num_layers,"
           "save_best_checkpoint,early_stopping_patience,checkpoint_dir,"
           "checkpoint_used,has_validation_metrics,has_test_metrics,"
           "best_val_loss,best_val_accuracy,final_train_loss,"
           "final_train_accuracy,final_test_loss,final_test_accuracy,"
           "elapsed_seconds";
}

inline std::string EscapeTrainingRunComparisonCsvField(
    const std::string& value) {
    bool needs_quotes = false;
    for (char c : value) {
        if (c == ',' || c == '"' || c == '\n' || c == '\r') {
            needs_quotes = true;
            break;
        }
    }
    if (!needs_quotes) {
        return value;
    }

    std::string escaped = "\"";
    for (char c : value) {
        if (c == '"') {
            escaped += "\"\"";
        } else {
            escaped += c;
        }
    }
    escaped += '"';
    return escaped;
}

inline std::string TrainingRunComparisonToCsvRow(
    const TrainingRunComparisonRecord& record) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(6)
        << EscapeTrainingRunComparisonCsvField(record.run_id) << ','
        << EscapeTrainingRunComparisonCsvField(record.run_status) << ','
        << EscapeTrainingRunComparisonCsvField(record.dataset_name) << ','
        << EscapeTrainingRunComparisonCsvField(record.model_family) << ','
        << EscapeTrainingRunComparisonCsvField(record.primary_layer_type) << ','
        << EscapeTrainingRunComparisonCsvField(record.architecture_summary) << ','
        << record.model_layer_count << ','
        << record.epochs << ','
        << record.batch_size << ','
        << record.learning_rate << ','
        << (record.bidirectional ? "true" : "false") << ','
        << record.hidden_size << ','
        << record.num_layers << ','
        << (record.save_best_checkpoint ? "true" : "false") << ','
        << record.early_stopping_patience << ','
        << EscapeTrainingRunComparisonCsvField(record.checkpoint_dir) << ','
        << EscapeTrainingRunComparisonCsvField(record.checkpoint_used) << ','
        << (record.has_validation_metrics ? "true" : "false") << ','
        << (record.has_test_metrics ? "true" : "false") << ','
        << record.best_val_loss << ','
        << record.best_val_accuracy << ','
        << record.final_train_loss << ','
        << record.final_train_accuracy << ','
        << record.final_test_loss << ','
        << record.final_test_accuracy << ','
        << record.elapsed_seconds;
    return out.str();
}

inline std::string TrainingRunComparisonLayerTypeName(gui::NodeType type) {
    switch (type) {
        case gui::NodeType::Dense: return "Dense";
        case gui::NodeType::Embedding: return "Embedding";
        case gui::NodeType::GRU: return "GRU";
        case gui::NodeType::LSTM: return "LSTM";
        case gui::NodeType::Dropout: return "Dropout";
        case gui::NodeType::Flatten: return "Flatten";
        case gui::NodeType::ReLU: return "ReLU";
        case gui::NodeType::Sigmoid: return "Sigmoid";
        case gui::NodeType::Softmax: return "Softmax";
        default: return "Layer";
    }
}

inline std::string BuildTrainingRunArchitectureSummary(
    const TrainingConfiguration& config) {
    std::ostringstream out;
    for (size_t i = 0; i < config.layers.size(); ++i) {
        if (i > 0) {
            out << ">";
        }
        out << TrainingRunComparisonLayerTypeName(config.layers[i].type);
    }
    return out.str();
}

inline TrainingRunComparisonRecord MakeTrainingRunComparisonRecord(
    const std::string& run_id,
    const TrainingConfiguration& config,
    const TrainingMetrics& metrics,
    float elapsed_seconds,
    const std::string& checkpoint_used = "",
    const std::string& run_status = "complete") {

    TrainingRunComparisonRecord record;
    record.run_id = run_id;
    record.run_status = run_status;
    record.dataset_name = config.dataset_name;
    record.model_layer_count = static_cast<int>(config.layers.size());
    record.architecture_summary = BuildTrainingRunArchitectureSummary(config);
    record.primary_layer_type = config.layers.empty()
        ? "Unknown"
        : TrainingRunComparisonLayerTypeName(config.layers.front().type);
    record.epochs = config.epochs;
    record.batch_size = config.batch_size;
    record.learning_rate = config.learning_rate;
    record.save_best_checkpoint = config.save_best_checkpoint;
    record.early_stopping_patience = config.early_stopping_patience;
    record.checkpoint_dir = config.checkpoint_dir;
    record.checkpoint_used = checkpoint_used;
    record.final_train_loss = metrics.train_loss;
    record.final_train_accuracy = metrics.train_accuracy;
    record.final_test_loss = metrics.test_loss;
    record.final_test_accuracy = metrics.test_accuracy;
    record.elapsed_seconds = elapsed_seconds;
    record.has_validation_metrics =
        !metrics.val_loss_history.empty() ||
        !metrics.val_accuracy_history.empty() ||
        metrics.val_loss != 0.0f ||
        metrics.val_accuracy != 0.0f;
    record.has_test_metrics =
        metrics.test_loss != 0.0f || metrics.test_accuracy != 0.0f;

    if (!metrics.val_loss_history.empty()) {
        record.best_val_loss = metrics.val_loss_history.front();
        for (float value : metrics.val_loss_history) {
            if (value < record.best_val_loss) {
                record.best_val_loss = value;
            }
        }
    } else {
        record.best_val_loss = metrics.val_loss;
    }

    if (!metrics.val_accuracy_history.empty()) {
        record.best_val_accuracy = metrics.val_accuracy_history.front();
        for (float value : metrics.val_accuracy_history) {
            if (value > record.best_val_accuracy) {
                record.best_val_accuracy = value;
            }
        }
    } else {
        record.best_val_accuracy = metrics.val_accuracy;
    }

    record.model_family = record.primary_layer_type;
    for (const auto& layer : config.layers) {
        if (layer.type == gui::NodeType::GRU ||
            layer.type == gui::NodeType::LSTM) {
            record.model_family =
                layer.type == gui::NodeType::GRU ? "GRU" : "LSTM";

            auto hidden_it = layer.parameters.find("hidden_size");
            if (hidden_it != layer.parameters.end()) {
                try { record.hidden_size = std::stoi(hidden_it->second); }
                catch (...) { record.hidden_size = 0; }
            }

            auto layers_it = layer.parameters.find("num_layers");
            if (layers_it != layer.parameters.end()) {
                try { record.num_layers = std::stoi(layers_it->second); }
                catch (...) { record.num_layers = 0; }
            }

            auto bidir_it = layer.parameters.find("bidirectional");
            if (bidir_it != layer.parameters.end()) {
                record.bidirectional =
                    bidir_it->second == "true" || bidir_it->second == "1";
            }
            return record;
        }
    }

    if (!config.layers.empty()) {
        record.model_family = "Sequential";
        record.num_layers = static_cast<int>(config.layers.size());
    } else {
        record.model_family = "Unknown";
    }
    return record;
}

inline std::string TrainingRunComparisonTableSummary(
    const std::vector<TrainingRunComparisonRecord>& records) {
    std::ostringstream out;
    out << TrainingRunComparisonCsvHeader();
    for (const auto& record : records) {
        out << '\n' << TrainingRunComparisonToCsvRow(record);
    }
    return out.str();
}

inline bool WriteTrainingRunComparisonCsv(
    const std::filesystem::path& output_path,
    const std::vector<TrainingRunComparisonRecord>& records,
    std::string* error_message = nullptr) {

    std::error_code ec;
    const auto parent = output_path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            if (error_message) {
                *error_message = "Failed to create comparison output directory: " +
                    ec.message();
            }
            return false;
        }
    }

    std::ofstream file(output_path, std::ios::binary | std::ios::trunc);
    if (!file) {
        if (error_message) {
            *error_message = "Failed to open comparison output file: " +
                output_path.string();
        }
        return false;
    }

    file << TrainingRunComparisonTableSummary(records) << '\n';
    if (!file) {
        if (error_message) {
            *error_message = "Failed to write comparison output file: " +
                output_path.string();
        }
        return false;
    }
    return true;
}

inline std::vector<TrainingRunComparisonRecord>
SortTrainingRunComparisonsByBestMetric(
    std::vector<TrainingRunComparisonRecord> records) {

    std::stable_sort(
        records.begin(),
        records.end(),
        [](const TrainingRunComparisonRecord& lhs,
           const TrainingRunComparisonRecord& rhs) {
            if (lhs.has_test_metrics != rhs.has_test_metrics) {
                return lhs.has_test_metrics;
            }
            if (lhs.has_test_metrics &&
                lhs.final_test_accuracy != rhs.final_test_accuracy) {
                return lhs.final_test_accuracy > rhs.final_test_accuracy;
            }
            if (lhs.has_validation_metrics != rhs.has_validation_metrics) {
                return lhs.has_validation_metrics;
            }
            if (lhs.has_validation_metrics &&
                lhs.best_val_accuracy != rhs.best_val_accuracy) {
                return lhs.best_val_accuracy > rhs.best_val_accuracy;
            }
            if (lhs.has_validation_metrics &&
                lhs.best_val_loss != rhs.best_val_loss) {
                return lhs.best_val_loss < rhs.best_val_loss;
            }
            return lhs.elapsed_seconds < rhs.elapsed_seconds;
        });
    return records;
}

} // namespace cyxwiz
