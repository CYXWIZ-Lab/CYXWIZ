#include "properties_truth.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <set>

namespace gui::properties_truth {
namespace {

bool HasStatus(const std::vector<TruthStatus>& statuses, TruthStatus status) {
    return std::find(statuses.begin(), statuses.end(), status) != statuses.end();
}

void AddStatus(PropertyTruth& truth, TruthStatus status) {
    if (!HasStatus(truth.statuses, status)) {
        truth.statuses.push_back(status);
    }
}

void AddStatus(RawParameterTruth& truth, TruthStatus status) {
    if (!HasStatus(truth.statuses, status)) {
        truth.statuses.push_back(status);
    }
}

const std::string* FindParameter(const MLNode& node, const std::string& key) {
    const auto it = node.parameters.find(key);
    return it == node.parameters.end() ? nullptr : &it->second;
}

const std::string* FindDataInputDatasetName(const MLNode& node) {
    const std::string* dataset_name = FindParameter(node, "dataset_name");
    if (dataset_name && !dataset_name->empty()) {
        return dataset_name;
    }
    const std::string* legacy_dataset = FindParameter(node, "dataset");
    return legacy_dataset && !legacy_dataset->empty() ? legacy_dataset : nullptr;
}

bool LooksLikeTextDataInput(const MLNode& node) {
    if (node.type != NodeType::DataInput) {
        return false;
    }
    const std::string* text_label = FindParameter(node, "text_label_column");
    if (text_label && !text_label->empty()) {
        return true;
    }
    const std::string* category = FindParameter(node, "file_category");
    if (category && *category == "text") {
        return true;
    }
    const std::string* type = FindParameter(node, "type");
    return type && (*type == "txt" || *type == "text" || *type == "text_corpus");
}

int ParsePositiveInt(const std::string* value) {
    if (!value || value->empty()) {
        return 0;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value->c_str(), &end, 10);
    if (end == value->c_str() || *end != '\0' || parsed <= 0) {
        return 0;
    }
    return static_cast<int>(parsed);
}

bool ParseBoolValue(const std::string* value, bool default_value) {
    if (!value || value->empty()) {
        return default_value;
    }
    return *value == "true" || *value == "1" || *value == "yes" ||
           *value == "on";
}

int ExtractTrailingPositiveInt(const std::string& text) {
    size_t end = text.size();
    while (end > 0 &&
           std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    size_t begin = end;
    while (begin > 0 &&
           std::isdigit(static_cast<unsigned char>(text[begin - 1]))) {
        --begin;
    }
    if (begin == end) {
        return 0;
    }
    const std::string digits = text.substr(begin, end - begin);
    return ParsePositiveInt(&digits);
}

const DatasetTruthFact* FindDatasetFact(const MLNode& node,
                                        const NodeTruthContext& context) {
    if (!context.dataset_facts) {
        return nullptr;
    }
    const std::string* dataset_name = FindDataInputDatasetName(node);
    if (!dataset_name || dataset_name->empty()) {
        return nullptr;
    }
    const auto it = std::find_if(
        context.dataset_facts->begin(),
        context.dataset_facts->end(),
        [dataset_name](const DatasetTruthFact& fact) {
            return fact.dataset_name == *dataset_name;
        });
    return it == context.dataset_facts->end() ? nullptr : &*it;
}

const BackendPlacementTruthFact* FindBackendPlacementFact(
    const MLNode& node,
    const NodeTruthContext& context) {
    if (!context.backend_placements) {
        return nullptr;
    }
    const auto it = std::find_if(
        context.backend_placements->begin(),
        context.backend_placements->end(),
        [&node](const BackendPlacementTruthFact& fact) {
            return fact.node_id == node.id;
        });
    return it == context.backend_placements->end() ? nullptr : &*it;
}

bool ContainsColumn(const std::vector<std::string>& columns,
                    const std::string& column) {
    return std::find(columns.begin(), columns.end(), column) != columns.end();
}

std::vector<std::string> LabelAliasesFor(const std::string& canonical_key) {
    if (canonical_key == "text_label_column") {
        return {"label_column", "label_col"};
    }
    return {"text_label_column", "label_col"};
}

bool IsLabelCanonicalKey(const std::string& key) {
    return key == "label_column" || key == "text_label_column";
}

bool HasNonEmptyParameter(const MLNode& node, const std::string& key) {
    const std::string* value = FindParameter(node, key);
    return value && !value->empty();
}

bool IsSafeLegacyDuplicate(const MLNode& node,
                           const std::string& key,
                           std::string& reason) {
    if (node.type == NodeType::DataInput && key == "dataset" &&
        HasNonEmptyParameter(node, "dataset_name")) {
        reason = "Legacy dataset key is ignored while dataset_name is set.";
        return true;
    }
    return false;
}

PropertyTruth ResolveAliasedStringProperty(const MLNode& node,
                                           std::string label,
                                           std::string canonical_key,
                                           std::vector<std::string> aliases,
                                           TruthOwner owner,
                                           bool quick_editable,
                                           bool requires_dialog) {
    PropertyTruth truth;
    truth.label = std::move(label);
    truth.canonical_key = std::move(canonical_key);
    truth.owner = owner;
    truth.quick_editable = quick_editable;
    truth.requires_dialog = requires_dialog;

    const std::string* canonical = FindParameter(node, truth.canonical_key);
    if (canonical && !canonical->empty()) {
        truth.source_key = truth.canonical_key;
        truth.effective_value = *canonical;
    }

    for (const auto& alias : aliases) {
        const std::string* value = FindParameter(node, alias);
        if (!value) {
            continue;
        }
        truth.aliases_present.push_back({alias, *value});
        if (truth.effective_value.empty() && !value->empty()) {
            truth.source_key = alias;
            truth.effective_value = *value;
            AddStatus(truth, TruthStatus::AliasUsed);
        } else if (!value->empty() && !truth.effective_value.empty() &&
                   *value != truth.effective_value) {
            AddStatus(truth, TruthStatus::Conflicting);
            truth.message = "Alias value does not match the effective value.";
        } else {
            AddStatus(truth, TruthStatus::AliasUsed);
        }
    }

    if (truth.effective_value.empty()) {
        AddStatus(truth, TruthStatus::Missing);
        truth.message = "No label column is configured.";
    } else if (truth.statuses.empty()) {
        AddStatus(truth, TruthStatus::OK);
    }

    if (truth.requires_dialog) {
        AddStatus(truth, TruthStatus::RequiresDialog);
    }
    return truth;
}

PropertyTruth ResolveIntProperty(const MLNode& node,
                                 std::string label,
                                 std::string canonical_key,
                                 std::string default_value,
                                 TruthOwner owner,
                                 bool quick_editable,
                                 bool requires_dialog,
                                 std::string message = {}) {
    PropertyTruth truth;
    truth.label = std::move(label);
    truth.canonical_key = std::move(canonical_key);
    truth.default_value = std::move(default_value);
    truth.owner = owner;
    truth.quick_editable = quick_editable;
    truth.requires_dialog = requires_dialog;
    truth.message = std::move(message);

    const std::string* value = FindParameter(node, truth.canonical_key);
    if (value && !value->empty()) {
        truth.source_key = truth.canonical_key;
        truth.effective_value = *value;
        AddStatus(truth, TruthStatus::OK);
    } else {
        truth.source_key = "default";
        truth.effective_value = truth.default_value;
        AddStatus(truth, TruthStatus::Defaulted);
    }
    if (truth.requires_dialog) {
        AddStatus(truth, TruthStatus::RequiresDialog);
    }
    return truth;
}

PropertyTruth ResolveAliasedIntProperty(const MLNode& node,
                                        std::string label,
                                        std::string canonical_key,
                                        std::vector<std::string> aliases,
                                        std::string default_value,
                                        TruthOwner owner,
                                        bool quick_editable,
                                        bool requires_dialog,
                                        std::string message = {}) {
    PropertyTruth truth;
    truth.label = std::move(label);
    truth.canonical_key = std::move(canonical_key);
    truth.default_value = std::move(default_value);
    truth.owner = owner;
    truth.quick_editable = quick_editable;
    truth.requires_dialog = requires_dialog;
    truth.message = std::move(message);

    const std::string* canonical = FindParameter(node, truth.canonical_key);
    if (canonical && !canonical->empty()) {
        truth.source_key = truth.canonical_key;
        truth.effective_value = *canonical;
    }

    for (const auto& alias : aliases) {
        const std::string* value = FindParameter(node, alias);
        if (!value) {
            continue;
        }
        truth.aliases_present.push_back({alias, *value});
        if (truth.effective_value.empty() && !value->empty()) {
            truth.source_key = alias;
            truth.effective_value = *value;
            AddStatus(truth, TruthStatus::AliasUsed);
        } else if (!value->empty() && !truth.effective_value.empty() &&
                   *value != truth.effective_value) {
            AddStatus(truth, TruthStatus::Conflicting);
            if (!truth.message.empty()) {
                truth.message += " ";
            }
            truth.message += "Alias value does not match the effective value.";
        } else {
            AddStatus(truth, TruthStatus::AliasUsed);
        }
    }

    if (truth.effective_value.empty()) {
        truth.source_key = "default";
        truth.effective_value = truth.default_value;
        AddStatus(truth, TruthStatus::Defaulted);
    } else if (truth.statuses.empty()) {
        AddStatus(truth, TruthStatus::OK);
    }
    if (truth.requires_dialog) {
        AddStatus(truth, TruthStatus::RequiresDialog);
    }
    return truth;
}

PropertyTruth ResolveBoolProperty(const MLNode& node,
                                  std::string label,
                                  std::string canonical_key,
                                  bool default_value,
                                  TruthOwner owner,
                                  bool quick_editable,
                                  bool requires_dialog,
                                  std::string message = {}) {
    PropertyTruth truth;
    truth.label = std::move(label);
    truth.canonical_key = std::move(canonical_key);
    truth.default_value = default_value ? "true" : "false";
    truth.owner = owner;
    truth.quick_editable = quick_editable;
    truth.requires_dialog = requires_dialog;
    truth.message = std::move(message);

    const std::string* value = FindParameter(node, truth.canonical_key);
    if (value && !value->empty()) {
        truth.source_key = truth.canonical_key;
        truth.effective_value =
            ParseBoolValue(value, default_value) ? "true" : "false";
        AddStatus(truth, TruthStatus::OK);
    } else {
        truth.source_key = "default";
        truth.effective_value = truth.default_value;
        AddStatus(truth, TruthStatus::Defaulted);
    }
    if (truth.requires_dialog) {
        AddStatus(truth, TruthStatus::RequiresDialog);
    }
    return truth;
}

const MLNode* FindNodeById(const std::vector<MLNode>& nodes, int id) {
    const auto it = std::find_if(nodes.begin(), nodes.end(),
                                 [id](const MLNode& node) {
                                     return node.id == id;
                                 });
    return it == nodes.end() ? nullptr : &*it;
}

const NodePin* FindPin(const MLNode& node, int pin_id) {
    for (const auto& pin : node.inputs) {
        if (pin.id == pin_id) {
            return &pin;
        }
    }
    for (const auto& pin : node.outputs) {
        if (pin.id == pin_id) {
            return &pin;
        }
    }
    return nullptr;
}

bool LinkTargetsPinNamed(const MLNode& to_node,
                         const NodeLink& link,
                         const char* expected) {
    const NodePin* pin = FindPin(to_node, link.to_pin);
    return !pin || pin->name == expected;
}

const MLNode* FindUpstreamModelNode(const MLNode& node,
                                    const NodeTruthContext& context) {
    if (!context.nodes || !context.links) {
        return nullptr;
    }

    const MLNode* current = &node;
    std::set<int> visited;
    while (current && visited.insert(current->id).second) {
        const NodeLink* incoming = nullptr;
        for (const auto& link : *context.links) {
            if (link.to_node == current->id &&
                LinkTargetsPinNamed(*current, link, "Predictions")) {
                incoming = &link;
                break;
            }
        }
        if (!incoming) {
            for (const auto& link : *context.links) {
                if (link.to_node == current->id) {
                    incoming = &link;
                    break;
                }
            }
        }
        if (!incoming) {
            return nullptr;
        }

        const MLNode* upstream = FindNodeById(*context.nodes, incoming->from_node);
        if (!upstream) {
            return nullptr;
        }
        if (upstream->type == NodeType::Dense ||
            upstream->type == NodeType::TimeDistributed) {
            return upstream;
        }
        current = upstream;
    }
    return nullptr;
}

int FindGraphCrossEntropyOutputWidth(const MLNode& node,
                                     const NodeTruthContext& context) {
    if (!context.nodes) {
        return 0;
    }

    const bool selected_is_output = node.type == NodeType::Output;
    const bool selected_is_loss = node.type == NodeType::CrossEntropyLoss;
    const MLNode* target = selected_is_output || selected_is_loss ? &node : nullptr;

    if (!target) {
        for (const auto& graph_node : *context.nodes) {
            if (graph_node.type == NodeType::Output) {
                target = &graph_node;
                break;
            }
        }
    }
    if (!target) {
        for (const auto& graph_node : *context.nodes) {
            if (graph_node.type == NodeType::CrossEntropyLoss) {
                target = &graph_node;
                break;
            }
        }
    }
    if (!target) {
        return 0;
    }

    const MLNode* model_node = FindUpstreamModelNode(*target, context);
    if (!model_node) {
        return 0;
    }
    return ParsePositiveInt(FindParameter(*model_node, "units"));
}

bool GraphHasCrossEntropy(const NodeTruthContext& context) {
    if (!context.nodes) {
        return false;
    }
    return std::any_of(context.nodes->begin(), context.nodes->end(),
                       [](const MLNode& node) {
                           return node.type == NodeType::CrossEntropyLoss;
                       });
}

void AppendMessage(PropertyTruth& truth, const std::string& message) {
    if (message.empty()) {
        return;
    }
    if (!truth.message.empty()) {
        truth.message += " ";
    }
    truth.message += message;
}

const DatasetTruthFact* FindUniqueDatasetClassFact(const NodeTruthContext& context,
                                                   bool& conflicting_counts) {
    conflicting_counts = false;
    if (!context.dataset_facts) {
        return nullptr;
    }

    const DatasetTruthFact* selected = nullptr;
    for (const auto& fact : *context.dataset_facts) {
        if (!fact.found || !fact.has_class_count) {
            continue;
        }
        if (!selected) {
            selected = &fact;
            continue;
        }
        if (selected->class_count != fact.class_count) {
            conflicting_counts = true;
            return nullptr;
        }
    }
    return selected;
}

void AddRawParameters(NodeTruthReport& report, const MLNode& node) {
    std::map<std::string, std::string> canonical_by_key;
    std::set<std::string> canonical_keys;
    for (const auto& property : report.properties) {
        canonical_keys.insert(property.canonical_key);
        canonical_by_key[property.canonical_key] = property.canonical_key;
        for (const auto& alias : property.aliases_present) {
            canonical_by_key[alias.key] = property.canonical_key;
        }
        if (IsLabelCanonicalKey(property.canonical_key)) {
            for (const auto& alias : LabelAliasesFor(property.canonical_key)) {
                canonical_by_key.emplace(alias, property.canonical_key);
            }
        }
    }

    for (const auto& [key, value] : node.parameters) {
        RawParameterTruth raw;
        raw.key = key;
        raw.value = value;
        const auto mapped = canonical_by_key.find(key);
        if (mapped != canonical_by_key.end()) {
            raw.maps_to = mapped->second;
            if (canonical_keys.count(key) == 0) {
                AddStatus(raw, TruthStatus::AliasUsed);
            } else {
                AddStatus(raw, TruthStatus::OK);
            }
        } else if (node.type == NodeType::DataInput &&
                   key == "dataset" &&
                   !HasNonEmptyParameter(node, "dataset_name")) {
            raw.maps_to = "dataset_name";
            AddStatus(raw, TruthStatus::AliasUsed);
        } else {
            AddStatus(raw, TruthStatus::Stale);
            raw.maps_to = "";
            raw.cleanup_allowed = IsSafeLegacyDuplicate(
                node, key, raw.cleanup_reason);
        }
        report.raw_parameters.push_back(std::move(raw));
    }
}

void MarkIssues(NodeTruthReport& report) {
    for (const auto& property : report.properties) {
        if (HasStatus(property.statuses, TruthStatus::Missing) ||
            HasStatus(property.statuses, TruthStatus::Conflicting) ||
            HasStatus(property.statuses, TruthStatus::Unsupported)) {
            report.has_issue = true;
            return;
        }
    }
}

PropertyTruth BuildDatasetClassesTruth(const DatasetTruthFact& fact) {
    PropertyTruth classes;
    classes.label = "Dataset classes";
    classes.canonical_key = "dataset_class_count";
    classes.source_key = fact.backing_store.empty() ? "registry"
                                                    : fact.backing_store;
    classes.owner = TruthOwner::Loader;
    classes.quick_editable = false;
    classes.requires_dialog = false;

    if (!fact.found) {
        AddStatus(classes, TruthStatus::Missing);
        classes.effective_value = "0";
        classes.message = fact.message.empty() ? "Dataset is not loaded."
                                               : fact.message;
        return classes;
    }

    if (!fact.has_class_count) {
        AddStatus(classes, TruthStatus::RuntimeOnly);
        classes.effective_value = "unknown";
        classes.message = "Loaded dataset does not expose class-count metadata.";
        return classes;
    }

    classes.effective_value = std::to_string(fact.class_count);
    if (fact.has_labels && fact.class_count == 0) {
        AddStatus(classes, TruthStatus::Missing);
        classes.message = "Dataset is labeled but registered 0 classes.";
    } else {
        AddStatus(classes, TruthStatus::OK);
    }
    return classes;
}

PropertyTruth BuildBackendPlacementTruth(const BackendPlacementTruthFact& fact) {
    PropertyTruth placement;
    placement.label = "Backend placement";
    placement.canonical_key = "backend_placement";
    placement.source_key = "compile report";
    placement.owner = TruthOwner::Compiler;
    placement.quick_editable = false;
    placement.requires_dialog = false;
    placement.effective_value = fact.expected_backend.empty()
        ? fact.status
        : fact.expected_backend;

    if (fact.status == "gpu") {
        AddStatus(placement, TruthStatus::OK);
    } else if (fact.status == "cpu") {
        AddStatus(placement, TruthStatus::RuntimeOnly);
    } else if (fact.status == "mixed" || fact.status == "risk") {
        AddStatus(placement, TruthStatus::RuntimeOnly);
        AddStatus(placement, TruthStatus::CompilerOnly);
    } else if (fact.status == "unsupported") {
        AddStatus(placement, TruthStatus::Unsupported);
    } else {
        AddStatus(placement, TruthStatus::CompilerOnly);
    }

    if (!fact.reason_code.empty()) {
        placement.message = "Reason: " + fact.reason_code;
    }
    if (!fact.explanation.empty()) {
        if (!placement.message.empty()) {
            placement.message += ". ";
        }
        placement.message += fact.explanation;
    }
    if (!fact.suggested_action.empty()) {
        if (!placement.message.empty()) {
            placement.message += " ";
        }
        placement.message += "Action: " + fact.suggested_action;
    }
    return placement;
}

} // namespace

const char* TruthStatusName(TruthStatus status) {
    switch (status) {
    case TruthStatus::OK: return "OK";
    case TruthStatus::Missing: return "Missing";
    case TruthStatus::Defaulted: return "Defaulted";
    case TruthStatus::AliasUsed: return "Alias used";
    case TruthStatus::Stale: return "Stale";
    case TruthStatus::Conflicting: return "Conflicting";
    case TruthStatus::RuntimeOnly: return "Runtime-only";
    case TruthStatus::CompilerOnly: return "Compiler-only";
    case TruthStatus::Unsupported: return "Unsupported";
    case TruthStatus::RequiresDialog: return "Requires dialog";
    }
    return "Unknown";
}

const char* TruthOwnerName(TruthOwner owner) {
    switch (owner) {
    case TruthOwner::UI: return "UI";
    case TruthOwner::Compiler: return "Compiler";
    case TruthOwner::Loader: return "Loader";
    case TruthOwner::Materializer: return "Materializer";
    case TruthOwner::Runtime: return "Runtime";
    case TruthOwner::Exporter: return "Exporter";
    }
    return "Unknown";
}

NodeTruthReport ResolveNodeTruth(const MLNode& node,
                                 const NodeTruthContext& context) {
    NodeTruthReport report;

    if (node.type == NodeType::DataInput) {
        const bool text_input = LooksLikeTextDataInput(node);
        const std::string canonical = text_input ? "text_label_column"
                                                 : "label_column";
        auto label_truth = ResolveAliasedStringProperty(
            node,
            "Label column",
            canonical,
            LabelAliasesFor(canonical),
            TruthOwner::Loader,
            true,
            true);

        const DatasetTruthFact* fact = FindDatasetFact(node, context);
        if (fact) {
            if (!fact->found) {
                AddStatus(label_truth, TruthStatus::Missing);
                label_truth.message = fact->message.empty()
                    ? "Configured dataset is not loaded."
                    : fact->message;
            } else if (!label_truth.effective_value.empty()) {
                if (!fact->columns.empty() &&
                    !ContainsColumn(fact->columns, label_truth.effective_value)) {
                    AddStatus(label_truth, TruthStatus::Missing);
                    label_truth.message =
                        "Label column was not found in loaded dataset columns.";
                }
                if (fact->has_label_column_metadata &&
                    !fact->label_column.empty() &&
                    fact->label_column != label_truth.effective_value) {
                    AddStatus(label_truth, TruthStatus::Conflicting);
                    label_truth.message =
                        "Loaded dataset label column is '" + fact->label_column + "'.";
                }
            }
            report.properties.push_back(BuildDatasetClassesTruth(*fact));
        }

        report.properties.push_back(std::move(label_truth));
    }

    if (node.type == NodeType::TFIDFVectorizer) {
        auto max_features = ResolveIntProperty(
            node,
            "Effective feature width",
            "max_features",
            "2000",
            TruthOwner::Materializer,
            true,
            false,
            "Dense output width is capped by max_features.");
        const int parsed = ParsePositiveInt(&max_features.effective_value);
        if (parsed <= 0) {
            max_features.statuses.clear();
            AddStatus(max_features, TruthStatus::Missing);
            max_features.message = "max_features must be >= 1.";
        }
        report.properties.push_back(std::move(max_features));
    }

    if (node.type == NodeType::TextTokenizer) {
        auto max_length = ResolveIntProperty(
            node,
            "Compiled token width",
            "max_length",
            "256",
            TruthOwner::Compiler,
            true,
            true,
            "GraphCompiler uses max_length as the text input shape width.");
        const int parsed = ParsePositiveInt(&max_length.effective_value);
        if (parsed <= 0) {
            max_length.statuses.clear();
            AddStatus(max_length, TruthStatus::Missing);
            max_length.message = "max_length must be >= 1.";
        }
        report.properties.push_back(std::move(max_length));
    }

    if (node.type == NodeType::DataLoader) {
        if (FindParameter(node, "pin_memory")) {
            auto pin_memory = ResolveBoolProperty(
                node,
                "Pinned host memory",
                "pin_memory",
                false,
                TruthOwner::Compiler,
                false,
                true,
                "GraphCompiler serializes pin_memory for compatibility.");
            if (pin_memory.effective_value == "true") {
                pin_memory.statuses.clear();
                AddStatus(pin_memory, TruthStatus::Unsupported);
                AddStatus(pin_memory, TruthStatus::RequiresDialog);
                pin_memory.message =
                    "pin_memory=true is ignored because current batchers do not "
                    "provide a pinned host-memory transfer backend.";
            }
            report.properties.push_back(std::move(pin_memory));
        }

        const bool balance_classes =
            ParseBoolValue(FindParameter(node, "balance_classes"), false) ||
            ParseBoolValue(FindParameter(node, "weighted_sampler"), false) ||
            ParseBoolValue(FindParameter(node, "oversample"), false) ||
            ParseBoolValue(FindParameter(node, "undersample"), false);
        if (balance_classes || FindParameter(node, "balance_classes")) {
            auto balance = ResolveBoolProperty(
                node,
                "Class balancing",
                "balance_classes",
                false,
                TruthOwner::Runtime,
                false,
                true,
                "DataLoader class balancing is a training-batcher policy.");
            if (balance_classes) {
                balance.effective_value = "true";
                AddStatus(balance, TruthStatus::RuntimeOnly);
                const std::string* mode = FindParameter(node, "balance_mode");
                const std::string* target = FindParameter(node, "balance_target");
                balance.message =
                    "Applied to training batchers only; validation and test "
                    "batchers keep natural evaluation distribution. Mode: " +
                    (mode && !mode->empty() ? *mode : std::string("oversample")) +
                    ", target: " +
                    (target && !target->empty() ? *target : std::string("max")) +
                    ".";
            }
            report.properties.push_back(std::move(balance));
        }
    }

    if (node.type == NodeType::LSTM || node.type == NodeType::GRU) {
        auto hidden_size = ResolveIntProperty(
            node,
            "Hidden size",
            "hidden_size",
            "128",
            TruthOwner::Compiler,
            true,
            false,
            "GraphCompiler and model builder use hidden_size for recurrent width.");
        const int parsed_hidden_size = ParsePositiveInt(&hidden_size.effective_value);
        if (parsed_hidden_size <= 0) {
            hidden_size.statuses.clear();
            AddStatus(hidden_size, TruthStatus::Missing);
            hidden_size.message = "hidden_size must be >= 1.";
        } else {
            const int name_width = ExtractTrailingPositiveInt(node.name);
            if (name_width > 0 && name_width != parsed_hidden_size) {
                AddStatus(hidden_size, TruthStatus::Conflicting);
                hidden_size.message =
                    "Node name ends with " + std::to_string(name_width) +
                    ", but effective hidden_size is " +
                    std::to_string(parsed_hidden_size) + ".";
            }
        }
        report.properties.push_back(std::move(hidden_size));

        auto return_sequences = ResolveBoolProperty(
            node,
            "Sequence output",
            "return_sequences",
            false,
            TruthOwner::Compiler,
            true,
            false,
            "When true, downstream nodes receive one vector per sequence step.");
        report.properties.push_back(std::move(return_sequences));

        if (const auto* placement_fact = FindBackendPlacementFact(node, context)) {
            report.properties.push_back(BuildBackendPlacementTruth(*placement_fact));
        } else {
            PropertyTruth missing_placement;
            missing_placement.label = "Backend placement";
            missing_placement.canonical_key = "backend_placement";
            missing_placement.source_key = "compile report";
            missing_placement.owner = TruthOwner::Compiler;
            missing_placement.quick_editable = false;
            missing_placement.effective_value = "Run Compile";
            AddStatus(missing_placement, TruthStatus::CompilerOnly);
            missing_placement.message =
                "No compile report is available for this recurrent node yet.";
            report.properties.push_back(std::move(missing_placement));
        }
    }

    if (node.type == NodeType::Output) {
        auto classes = ResolveAliasedIntProperty(
            node,
            "Output classes",
            "num_classes",
            {"classes"},
            "10",
            TruthOwner::Compiler,
            true,
            false);

        const int configured_classes = ParsePositiveInt(&classes.effective_value);
        const int model_width = FindGraphCrossEntropyOutputWidth(node, context);
        if (GraphHasCrossEntropy(context) && model_width > 0) {
            AppendMessage(classes, "CrossEntropy model output width: " +
                                       std::to_string(model_width) + ".");
            if (configured_classes > 0 && configured_classes != model_width) {
                AddStatus(classes, TruthStatus::Conflicting);
                AppendMessage(classes,
                    "CrossEntropy expects class count to match model output width " +
                    std::to_string(model_width) + ".");
            } else if (!HasStatus(classes.statuses, TruthStatus::Defaulted)) {
                classes.statuses.clear();
                AddStatus(classes, TruthStatus::OK);
            }
        }

        bool conflicting_dataset_counts = false;
        const DatasetTruthFact* dataset_fact =
            FindUniqueDatasetClassFact(context, conflicting_dataset_counts);
        if (conflicting_dataset_counts) {
            AddStatus(classes, TruthStatus::Conflicting);
            AppendMessage(classes,
                          "Loaded datasets expose different class counts.");
        } else if (dataset_fact) {
            AppendMessage(classes,
                          "Dataset class count from '" +
                              dataset_fact->dataset_name + "': " +
                              std::to_string(dataset_fact->class_count) + ".");
            if (configured_classes > 0 &&
                configured_classes != static_cast<int>(dataset_fact->class_count)) {
                AddStatus(classes, TruthStatus::Conflicting);
                AppendMessage(classes,
                              "Output classes should match the loaded dataset class count.");
            }
        }
        report.properties.push_back(std::move(classes));
    }

    if (node.type == NodeType::CrossEntropyLoss) {
        PropertyTruth width;
        width.label = "Model output width";
        width.canonical_key = "dense_units";
        width.source_key = "graph";
        width.owner = TruthOwner::Compiler;
        width.quick_editable = false;
        width.requires_dialog = false;
        const int model_width = FindGraphCrossEntropyOutputWidth(node, context);
        if (model_width > 0) {
            width.effective_value = std::to_string(model_width);
            AddStatus(width, TruthStatus::OK);
        } else {
            AddStatus(width, TruthStatus::Missing);
            width.message = "No upstream Dense/TimeDistributed output width was found.";
        }
        report.properties.push_back(std::move(width));
    }

    AddRawParameters(report, node);
    MarkIssues(report);
    return report;
}

void WriteCanonicalAndAliases(MLNode& node,
                              const std::string& canonical_key,
                              const std::string& value) {
    node.parameters[canonical_key] = value;
    if (canonical_key == "text_label_column") {
        node.parameters["label_column"] = value;
    } else if (canonical_key == "label_column") {
        node.parameters["text_label_column"] = value;
    } else if (canonical_key == "num_classes") {
        node.parameters["classes"] = value;
    } else if (canonical_key == "classes") {
        node.parameters["num_classes"] = value;
    }
}

} // namespace gui::properties_truth
