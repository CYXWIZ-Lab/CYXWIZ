#pragma once

#include "node_editor.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace gui::properties_truth {

enum class TruthStatus {
    OK,
    Missing,
    Defaulted,
    AliasUsed,
    Stale,
    Conflicting,
    RuntimeOnly,
    CompilerOnly,
    Unsupported,
    RequiresDialog
};

enum class TruthOwner {
    UI,
    Compiler,
    Loader,
    Materializer,
    Runtime,
    Exporter
};

struct AliasValue {
    std::string key;
    std::string value;
};

struct PropertyTruth {
    std::string label;
    std::string canonical_key;
    std::string source_key;
    std::string effective_value;
    std::string default_value;
    TruthOwner owner = TruthOwner::UI;
    bool quick_editable = false;
    bool requires_dialog = false;
    std::vector<TruthStatus> statuses;
    std::vector<AliasValue> aliases_present;
    std::string message;
};

struct RawParameterTruth {
    std::string key;
    std::string value;
    std::string maps_to;
    bool cleanup_allowed = false;
    std::string cleanup_reason;
    std::vector<TruthStatus> statuses;
};

struct DatasetTruthFact {
    std::string dataset_name;
    std::string backing_store;
    bool found = false;
    std::vector<std::string> columns;
    bool has_labels = false;
    bool has_label_column_metadata = false;
    std::string label_column;
    bool has_class_count = false;
    size_t class_count = 0;
    std::string message;
};

struct BackendPlacementTruthFact {
    int node_id = -1;
    std::string node_type;
    std::string expected_backend;
    std::string fallback_backend;
    std::string status;
    std::string reason_code;
    std::string explanation;
    std::string suggested_action;
};

struct NodeTruthContext {
    const std::vector<MLNode>* nodes = nullptr;
    const std::vector<NodeLink>* links = nullptr;
    const std::vector<DatasetTruthFact>* dataset_facts = nullptr;
    const std::vector<BackendPlacementTruthFact>* backend_placements = nullptr;
};

struct NodeTruthReport {
    std::vector<PropertyTruth> properties;
    std::vector<RawParameterTruth> raw_parameters;
    bool has_issue = false;
};

const char* TruthStatusName(TruthStatus status);
const char* TruthOwnerName(TruthOwner owner);

const std::vector<NodeType>& SpecializedTruthCoverageNodeTypes();
bool HasSpecializedTruthCoverage(NodeType type);

NodeTruthReport ResolveNodeTruth(const MLNode& node,
                                 const NodeTruthContext& context = {});

void WriteCanonicalAndAliases(MLNode& node,
                              const std::string& canonical_key,
                              const std::string& value);

} // namespace gui::properties_truth
