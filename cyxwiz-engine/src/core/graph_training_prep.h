#pragma once

// Headless training preparation shared by every host (TOFIX118 P2); see
// graph_training_prep.cpp. Datasets come from the GraphDatasetCatalog.

#include "dataset_partitions.h"
#include "graph_compiler.h"
#include "graph_model.h"

#include <arrow/api.h>

#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

// Why a launch was refused (same fields the Engine shows).
struct LaunchBlock {
    std::string title;
    std::string detail;
    std::string error_message;
};

// The first Data Input's dataset name, and the label column configured for a
// dataset (the data-source node wins, then the matching Data Input, then the
// first Data Input).
std::string FindGraphDatasetName(const std::vector<gui::MLNode>& nodes);
std::string FindGraphLabelColumn(const std::vector<gui::MLNode>& nodes, const std::string& dataset_name,
                                 int data_source_node_id);

// Schema of a catalog dataset (Arrow, Parquet or sparse CSR), or null.
std::shared_ptr<arrow::Schema> FindGraphDatasetSchema(const std::string& dataset_name);

// Runtime label column for a loaded dataset (requested if present, else the
// common-label fallback), the target column reconciled to it, and the tabular
// input width reconciled to the runtime schema.
std::string ResolveRuntimeArrowLabelColumn(const std::string& dataset_name, const std::string& requested_label);
void ReconcileRuntimeDatasetTarget(TrainingConfiguration& config, const std::string& resolved_label,
                                   const std::string& dataset_name);
void ReconcileRuntimeTabularFeatureWidth(const std::string& dataset_name, const std::string& label_column,
                                         TrainingConfiguration& config);

// Supplied Dev/Test roles: schema compatibility and bounded leakage checks
// (tabular), or required sequence columns (sequence graphs). False = refuse.
bool ValidateSuppliedRolePreflight(ResolvedDatasetRoles& roles, const TrainingConfiguration& config,
                                   LaunchBlock& launch_result);
bool ValidateSequenceLaunchColumns(const std::string& dataset_name, const TrainingConfiguration& config,
                                   std::string& error_message);

}  // namespace cyxwiz
