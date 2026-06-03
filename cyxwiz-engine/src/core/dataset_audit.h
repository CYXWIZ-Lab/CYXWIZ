#pragma once

#include "data_registry.h"

#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

class ArrowDataset;
class ParquetBackedDataset;

enum class DatasetAuditSeverity {
    Info,
    Warning,
    Error
};

struct DatasetAuditIssue {
    DatasetAuditSeverity severity = DatasetAuditSeverity::Info;
    std::string code;
    std::string message;
    std::vector<std::string> examples;
};

struct DatasetAuditResult {
    std::string dataset_name;
    std::string domain;
    int64_t sample_count = 0;
    int64_t feature_count = 0;
    int64_t class_count = 0;
    std::vector<DatasetAuditIssue> issues;

    bool HasErrors() const;
    bool HasWarnings() const;
    int ErrorCount() const;
    int WarningCount() const;
    void Add(DatasetAuditSeverity severity,
             std::string code,
             std::string message,
             std::vector<std::string> examples = {});
};

class DatasetAudit {
public:
    static DatasetAuditResult AuditTabular(
        const std::string& dataset_name,
        const std::shared_ptr<ArrowDataset>& dataset,
        const std::string& label_column);

    static DatasetAuditResult AuditParquet(
        const std::string& dataset_name,
        const std::shared_ptr<ParquetBackedDataset>& dataset,
        const std::string& label_column);

    static DatasetAuditResult AuditImage(
        const std::string& dataset_name,
        const DataRegistry::ImageDatasetEntry& entry);

    static DatasetAuditResult AuditAudio(
        const std::string& dataset_name,
        const DataRegistry::AudioDatasetEntry& entry);

    static DatasetAuditResult AuditText(
        const std::string& dataset_name,
        const DataRegistry::TextDatasetEntry& entry);
};

const char* ToString(DatasetAuditSeverity severity);
std::string FormatAuditSummary(const DatasetAuditResult& result);
std::vector<std::string> FormatAuditIssueLines(const DatasetAuditResult& result);

}  // namespace cyxwiz
