#pragma once

#include "graph_compiler.h"
#include "training_run_comparison_record.h"
#include "training_executor.h"

#include <iomanip>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace cyxwiz {

enum class TrainingRunPartitionCompatibility {
    SameManifest,
    DifferentManifest,
    Unknown
};

inline TrainingRunPartitionCompatibility CompareTrainingRunPartitions(
    const TrainingRunComparisonRecord& reference,
    const TrainingRunComparisonRecord& candidate) {
    if (reference.partition_manifest_fingerprint.empty() ||
        candidate.partition_manifest_fingerprint.empty()) {
        return TrainingRunPartitionCompatibility::Unknown;
    }
    return reference.partition_manifest_fingerprint ==
               candidate.partition_manifest_fingerprint
        ? TrainingRunPartitionCompatibility::SameManifest
        : TrainingRunPartitionCompatibility::DifferentManifest;
}

inline const char* TrainingRunPartitionCompatibilityLabel(
    TrainingRunPartitionCompatibility compatibility) {
    switch (compatibility) {
        case TrainingRunPartitionCompatibility::SameManifest:
            return "same";
        case TrainingRunPartitionCompatibility::DifferentManifest:
            return "different";
        case TrainingRunPartitionCompatibility::Unknown:
            return "unknown";
    }
    return "unknown";
}

inline std::string TrainingRunComparisonDomainName(
    PreprocessingDomain domain) {
    switch (domain) {
        case PreprocessingDomain::Tabular: return "tabular";
        case PreprocessingDomain::Image: return "image";
        case PreprocessingDomain::Audio: return "audio";
        case PreprocessingDomain::Text: return "text";
        case PreprocessingDomain::TimeSeries: return "time-series";
        case PreprocessingDomain::General: return "general";
    }
    return "unknown";
}

inline int TrainingRunComparisonValidationHistoryEpoch(
    const TrainingConfiguration& config,
    size_t validation_history_index) {
    const int validation_freq = std::max(1, config.validation_freq);
    const int epoch = static_cast<int>(validation_history_index + 1) *
                      validation_freq;
    return std::min(std::max(1, config.epochs), epoch);
}

inline std::string TrainingRunComparisonCsvHeader() {
    return "run_id,run_status,dataset_name,preprocessing_domain,"
           "sequence_batch_enabled,model_family,primary_layer_type,"
           "architecture_summary,model_layer_count,epochs,batch_size,"
           "learning_rate,train_ratio,val_ratio,test_ratio,"
           "train_sample_count,val_sample_count,test_sample_count,"
           "train_source_name,dev_source_name,test_source_name,"
           "train_origin,dev_origin,test_origin,"
           "train_label_column,dev_label_column,test_label_column,"
           "partition_manifest_fingerprint,dev_schema_compatibility,"
           "test_schema_compatibility,dev_leakage_status,test_leakage_status,"
           "dev_partition_status_reason,test_partition_status_reason,"
           "bidirectional,hidden_size,num_layers,"
           "save_best_checkpoint,early_stopping_patience,checkpoint_dir,"
           "checkpoint_used,has_validation_metrics,has_test_metrics,"
           "best_val_loss,best_val_accuracy,best_val_loss_epoch,"
           "best_val_accuracy_epoch,final_train_loss,"
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
        << EscapeTrainingRunComparisonCsvField(record.preprocessing_domain) << ','
        << (record.sequence_batch_enabled ? "true" : "false") << ','
        << EscapeTrainingRunComparisonCsvField(record.model_family) << ','
        << EscapeTrainingRunComparisonCsvField(record.primary_layer_type) << ','
        << EscapeTrainingRunComparisonCsvField(record.architecture_summary) << ','
        << record.model_layer_count << ','
        << record.epochs << ','
        << record.batch_size << ','
        << record.learning_rate << ','
        << record.train_ratio << ','
        << record.val_ratio << ','
        << record.test_ratio << ','
        << record.train_sample_count << ','
        << record.val_sample_count << ','
        << record.test_sample_count << ','
        << EscapeTrainingRunComparisonCsvField(record.train_source_name) << ','
        << EscapeTrainingRunComparisonCsvField(record.dev_source_name) << ','
        << EscapeTrainingRunComparisonCsvField(record.test_source_name) << ','
        << EscapeTrainingRunComparisonCsvField(record.train_origin) << ','
        << EscapeTrainingRunComparisonCsvField(record.dev_origin) << ','
        << EscapeTrainingRunComparisonCsvField(record.test_origin) << ','
        << EscapeTrainingRunComparisonCsvField(record.train_label_column) << ','
        << EscapeTrainingRunComparisonCsvField(record.dev_label_column) << ','
        << EscapeTrainingRunComparisonCsvField(record.test_label_column) << ','
        << EscapeTrainingRunComparisonCsvField(record.partition_manifest_fingerprint) << ','
        << EscapeTrainingRunComparisonCsvField(record.dev_schema_compatibility) << ','
        << EscapeTrainingRunComparisonCsvField(record.test_schema_compatibility) << ','
        << EscapeTrainingRunComparisonCsvField(record.dev_leakage_status) << ','
        << EscapeTrainingRunComparisonCsvField(record.test_leakage_status) << ','
        << EscapeTrainingRunComparisonCsvField(record.dev_partition_status_reason) << ','
        << EscapeTrainingRunComparisonCsvField(record.test_partition_status_reason) << ','
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
        << record.best_val_loss_epoch << ','
        << record.best_val_accuracy_epoch << ','
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

inline std::string ResolveTrainingRunCheckpointDisplay(
    const TrainingConfiguration& config,
    const std::string& checkpoint_used) {
    if (!checkpoint_used.empty()) {
        return checkpoint_used;
    }
    if (!config.save_best_checkpoint) {
        return "final epoch model state";
    }
    if (!config.checkpoint_dir.empty()) {
        return config.checkpoint_dir;
    }
    return "default .cyxwiz/checkpoints run folder";
}

inline std::string EscapeTrainingRunPartitionPart(const std::string& value) {
    std::ostringstream out;
    for (char ch : value) {
        switch (ch) {
        case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '|': out << "\\|"; break;
        case '=': out << "\\="; break;
        default: out << ch; break;
        }
    }
    return out.str();
}

inline std::string TrainingRunRoleSourceName(
    const TrainingConfiguration& config,
    const ResolvedDatasetRole& role,
    const std::string& fallback_source) {
    if (!role.dataset_name.empty()) {
        return role.dataset_name;
    }
    if (!fallback_source.empty()) {
        return fallback_source;
    }
    return config.dataset_name;
}

inline std::string TrainingRunRoleLabelColumn(
    const ResolvedDatasetRole& role,
    const std::string& fallback_label) {
    return role.label_column.empty() ? fallback_label : role.label_column;
}

inline std::string TrainingRunRoleOrigin(
    const ResolvedDatasetRole& role,
    bool train_role) {
    if (train_role) {
        return "external";
    }
    return role.IsSupplied() ? "external" : "derived";
}

inline std::string BuildTrainingRunPartitionFingerprint(
    const TrainingConfiguration& config,
    const TrainingMetrics& metrics,
    const TrainingRunComparisonRecord& record) {
    auto manifest = config.dataset_roles.manifest;
    const auto fallback_source_fingerprint = [](const std::string& name) {
        return name.empty()
            ? std::string{}
            : StablePartitionFingerprint("dataset_reference.v1\n" + name);
    };
    if (manifest.training_source_fingerprint.empty()) {
        manifest.training_source_fingerprint =
            fallback_source_fingerprint(record.train_source_name);
    }
    if (manifest.validation_source_fingerprint.empty()) {
        manifest.validation_source_fingerprint =
            fallback_source_fingerprint(record.dev_source_name);
    }
    if (manifest.test_source_fingerprint.empty()) {
        manifest.test_source_fingerprint =
            fallback_source_fingerprint(record.test_source_name);
    }
    if (manifest.feature_schema_fingerprint.empty()) {
        manifest.feature_schema_fingerprint =
            config.dataset_roles.train.feature_schema_fingerprint;
    }
    manifest.train_origin = PartitionOrigin::External;
    manifest.dev_origin = record.dev_origin == "external"
        ? PartitionOrigin::External : PartitionOrigin::Derived;
    manifest.test_origin = record.test_origin == "external"
        ? PartitionOrigin::External : PartitionOrigin::Derived;
    manifest.label_column = record.train_label_column;
    manifest.split_method = config.dataset_roles.policy.method;
    manifest.train_ratio = config.train_ratio;
    manifest.dev_ratio = config.val_ratio;
    manifest.test_ratio = config.test_ratio;
    manifest.seed = config.split_seed;
    manifest.shuffle = config.dataset_roles.policy.shuffle;
    manifest.stratified = config.dataset_roles.policy.stratified;
    manifest.train_rows = static_cast<int64_t>(metrics.train_sample_count);
    manifest.dev_rows = static_cast<int64_t>(metrics.val_sample_count);
    manifest.test_rows = static_cast<int64_t>(metrics.test_sample_count);
    return BuildPartitionManifestFingerprint(manifest);
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
    record.preprocessing_domain =
        TrainingRunComparisonDomainName(config.preprocessing_domain);
    record.sequence_batch_enabled = config.sequence_batch.enabled;
    record.model_layer_count = static_cast<int>(config.layers.size());
    record.architecture_summary = BuildTrainingRunArchitectureSummary(config);
    record.primary_layer_type = config.layers.empty()
        ? "Unknown"
        : TrainingRunComparisonLayerTypeName(config.layers.front().type);
    record.epochs = config.epochs;
    record.batch_size = config.batch_size;
    record.learning_rate = config.learning_rate;
    record.train_ratio = config.train_ratio;
    record.val_ratio = config.val_ratio;
    record.test_ratio = config.test_ratio;
    record.train_sample_count = metrics.train_sample_count;
    record.val_sample_count = metrics.val_sample_count;
    record.test_sample_count = metrics.test_sample_count;

    const std::string train_source = TrainingRunRoleSourceName(
        config, config.dataset_roles.train, config.dataset_name);
    const std::string train_label = TrainingRunRoleLabelColumn(
        config.dataset_roles.train, "");
    record.train_source_name = train_source;
    record.dev_source_name = TrainingRunRoleSourceName(
        config, config.dataset_roles.dev, train_source);
    record.test_source_name = TrainingRunRoleSourceName(
        config, config.dataset_roles.test, train_source);
    record.train_origin = TrainingRunRoleOrigin(config.dataset_roles.train, true);
    record.dev_origin = TrainingRunRoleOrigin(config.dataset_roles.dev, false);
    record.test_origin = TrainingRunRoleOrigin(config.dataset_roles.test, false);
    record.train_label_column = train_label;
    record.dev_label_column = TrainingRunRoleLabelColumn(
        config.dataset_roles.dev, train_label);
    record.test_label_column = TrainingRunRoleLabelColumn(
        config.dataset_roles.test, train_label);
    record.partition_manifest_fingerprint =
        BuildTrainingRunPartitionFingerprint(config, metrics, record);
    const auto& manifest = config.dataset_roles.manifest;
    record.dev_schema_compatibility =
        PartitionCompatibilityName(manifest.dev_compatibility);
    record.test_schema_compatibility =
        PartitionCompatibilityName(manifest.test_compatibility);
    record.dev_leakage_status =
        PartitionLeakageStatusName(manifest.dev_leakage);
    record.test_leakage_status =
        PartitionLeakageStatusName(manifest.test_leakage);
    record.dev_partition_status_reason = manifest.dev_status_reason;
    record.test_partition_status_reason = manifest.test_status_reason;

    record.save_best_checkpoint = config.save_best_checkpoint;
    record.early_stopping_patience = config.early_stopping_patience;
    record.checkpoint_dir = config.checkpoint_dir;
    record.checkpoint_used =
        ResolveTrainingRunCheckpointDisplay(config, checkpoint_used);
    record.final_train_loss = metrics.train_loss;
    record.final_train_accuracy = metrics.train_accuracy;
    record.final_test_loss = metrics.test_loss;
    record.final_test_accuracy = metrics.test_accuracy;
    record.elapsed_seconds = elapsed_seconds;
    record.has_validation_metrics =
        metrics.has_validation_metrics ||
        !metrics.val_loss_history.empty() ||
        !metrics.val_accuracy_history.empty();
    record.has_test_metrics = metrics.has_test_metrics;

    if (!metrics.val_loss_history.empty()) {
        record.best_val_loss = metrics.val_loss_history.front();
        record.best_val_loss_epoch =
            TrainingRunComparisonValidationHistoryEpoch(config, 0);
        for (size_t i = 0; i < metrics.val_loss_history.size(); ++i) {
            const float value = metrics.val_loss_history[i];
            if (value < record.best_val_loss) {
                record.best_val_loss = value;
                record.best_val_loss_epoch =
                    TrainingRunComparisonValidationHistoryEpoch(config, i);
            }
        }
    } else {
        record.best_val_loss = metrics.val_loss;
        record.best_val_loss_epoch = record.has_validation_metrics ? 1 : 0;
    }

    if (!metrics.val_accuracy_history.empty()) {
        record.best_val_accuracy = metrics.val_accuracy_history.front();
        record.best_val_accuracy_epoch =
            TrainingRunComparisonValidationHistoryEpoch(config, 0);
        for (size_t i = 0; i < metrics.val_accuracy_history.size(); ++i) {
            const float value = metrics.val_accuracy_history[i];
            if (value > record.best_val_accuracy) {
                record.best_val_accuracy = value;
                record.best_val_accuracy_epoch =
                    TrainingRunComparisonValidationHistoryEpoch(config, i);
            }
        }
    } else {
        record.best_val_accuracy = metrics.val_accuracy;
        record.best_val_accuracy_epoch = record.has_validation_metrics ? 1 : 0;
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
            if (lhs.elapsed_seconds != rhs.elapsed_seconds) {
                return lhs.elapsed_seconds < rhs.elapsed_seconds;
            }
            return lhs.run_id < rhs.run_id;
        });
    return records;
}

} // namespace cyxwiz
