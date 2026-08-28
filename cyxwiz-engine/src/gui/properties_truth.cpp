#include "properties_truth.h"
#include "../core/dense_activation_configuration_policy.h"
#include "../core/normalization_regularization_configuration_policy.h"
#include "../core/transformer_configuration_policy.h"

#include <algorithm>
#include <cctype>
#include <cmath>
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

int ParseNonNegativeInt(const std::string* value, int invalid_value = -1) {
    if (!value || value->empty()) {
        return invalid_value;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value->c_str(), &end, 10);
    if (end == value->c_str() || *end != '\0' || parsed < 0) {
        return invalid_value;
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

bool ParseDoubleValue(const std::string& value, double& out) {
    char* end = nullptr;
    const double parsed = std::strtod(value.c_str(), &end);
    if (end == value.c_str() || *end != '\0' || !std::isfinite(parsed)) {
        return false;
    }
    out = parsed;
    return true;
}

std::string TrimAscii(std::string value) {
    const auto first = std::find_if_not(
        value.begin(), value.end(), [](unsigned char c) {
            return std::isspace(c) != 0;
        });
    const auto last = std::find_if_not(
        value.rbegin(), value.rend(), [](unsigned char c) {
            return std::isspace(c) != 0;
        }).base();
    if (first >= last) {
        return {};
    }
    return std::string(first, last);
}

std::string NormalizeAsciiToken(std::string value) {
    value = TrimAscii(std::move(value));
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return value;
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

PropertyTruth ResolveAliasedStringPropertyWithDefault(
    const MLNode& node,
    std::string label,
    std::string canonical_key,
    std::vector<std::string> aliases,
    std::string default_value,
    TruthOwner owner,
    bool quick_editable,
    bool requires_dialog,
    bool required,
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

    if (truth.effective_value.empty() && !truth.default_value.empty()) {
        truth.source_key = "default";
        truth.effective_value = truth.default_value;
        if (truth.statuses.empty()) {
            AddStatus(truth, TruthStatus::Defaulted);
        }
    } else if (truth.effective_value.empty() && required) {
        truth.source_key = truth.canonical_key;
        AddStatus(truth, TruthStatus::Missing);
        truth.message = truth.canonical_key + " is required.";
    } else if (truth.effective_value.empty()) {
        truth.source_key = "default";
        if (truth.statuses.empty()) {
            AddStatus(truth, TruthStatus::Defaulted);
        }
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

PropertyTruth ResolveStringProperty(const MLNode& node,
                                    std::string label,
                                    std::string canonical_key,
                                    std::string default_value,
                                    TruthOwner owner,
                                    bool quick_editable,
                                    bool requires_dialog,
                                    bool required,
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
    } else if (!truth.default_value.empty()) {
        truth.source_key = "default";
        truth.effective_value = truth.default_value;
        AddStatus(truth, TruthStatus::Defaulted);
    } else if (required) {
        truth.source_key = truth.canonical_key;
        AddStatus(truth, TruthStatus::Missing);
        truth.message = truth.canonical_key + " is required by the materializer.";
    } else {
        truth.source_key = "default";
        AddStatus(truth, TruthStatus::Defaulted);
    }
    if (truth.requires_dialog) {
        AddStatus(truth, TruthStatus::RequiresDialog);
    }
    return truth;
}

PropertyTruth ResolveFloatProperty(const MLNode& node,
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
    double parsed = 0.0;
    if (!ParseDoubleValue(truth.effective_value, parsed)) {
        truth.statuses.clear();
        AddStatus(truth, TruthStatus::Missing);
        truth.message = truth.canonical_key + " must be a finite number.";
    }
    if (truth.requires_dialog) {
        AddStatus(truth, TruthStatus::RequiresDialog);
    }
    return truth;
}

void RequirePositiveInt(PropertyTruth& truth, const std::string& key) {
    const int parsed = ParsePositiveInt(&truth.effective_value);
    if (parsed <= 0) {
        truth.statuses.clear();
        AddStatus(truth, TruthStatus::Missing);
        truth.message = key + " must be >= 1.";
    }
}

void RequireNonNegativeInt(PropertyTruth& truth, const std::string& key) {
    const int parsed = ParseNonNegativeInt(&truth.effective_value);
    if (parsed < 0) {
        truth.statuses.clear();
        AddStatus(truth, TruthStatus::Missing);
        truth.message = key + " must be >= 0.";
    }
}

void AddPreprocessingScalerTruth(NodeTruthReport& report,
                                 const MLNode& node) {
    report.properties.push_back(ResolveStringProperty(
        node,
        "Feature columns",
        "columns",
        "numeric auto-detect",
        TruthOwner::Materializer,
        true,
        false,
        false,
        "Empty columns means the materializer auto-detects numeric columns."));

    if (node.type == NodeType::StandardScaler) {
        auto role = ResolveStringProperty(
            node,
            "Transform role",
            "transform_role",
            "features",
            TruthOwner::Compiler,
            true,
            true,
            "Regression target makes Train/Test MAE and RMSE use the fitted state to report original target units.");
        if (role.effective_value != "features" &&
            role.effective_value != "regression_target") {
            role.statuses.clear();
            AddStatus(role, TruthStatus::Missing);
            role.message =
                "transform_role must be features or regression_target.";
        }
        report.properties.push_back(std::move(role));
        report.properties.push_back(ResolveBoolProperty(
            node,
            "Center data",
            "with_mean",
            true,
            TruthOwner::Materializer,
            true,
            false));
        report.properties.push_back(ResolveBoolProperty(
            node,
            "Scale to unit variance",
            "with_std",
            true,
            TruthOwner::Materializer,
            true,
            false));
        return;
    }

    if (node.type == NodeType::MinMaxScaler) {
        auto min_value = ResolveFloatProperty(
            node,
            "Target minimum",
            "min",
            "0.0",
            TruthOwner::Materializer,
            true,
            false);
        auto max_value = ResolveFloatProperty(
            node,
            "Target maximum",
            "max",
            "1.0",
            TruthOwner::Materializer,
            true,
            false);
        double min_parsed = 0.0;
        double max_parsed = 0.0;
        if (ParseDoubleValue(min_value.effective_value, min_parsed) &&
            ParseDoubleValue(max_value.effective_value, max_parsed) &&
            max_parsed <= min_parsed) {
            max_value.statuses.clear();
            AddStatus(max_value, TruthStatus::Missing);
            max_value.message = "max must be greater than min.";
        }
        report.properties.push_back(std::move(min_value));
        report.properties.push_back(std::move(max_value));
        return;
    }

    if (node.type == NodeType::RobustScaler) {
        report.properties.push_back(ResolveBoolProperty(
            node,
            "Subtract median",
            "with_centering",
            true,
            TruthOwner::Materializer,
            true,
            false));
        report.properties.push_back(ResolveBoolProperty(
            node,
            "Scale by IQR",
            "with_scaling",
            true,
            TruthOwner::Materializer,
            true,
            false));
        auto qmin = ResolveFloatProperty(
            node,
            "Lower quantile",
            "quantile_min",
            "25",
            TruthOwner::Materializer,
            true,
            false);
        auto qmax = ResolveFloatProperty(
            node,
            "Upper quantile",
            "quantile_max",
            "75",
            TruthOwner::Materializer,
            true,
            false);
        double min_parsed = 0.0;
        double max_parsed = 0.0;
        if (ParseDoubleValue(qmin.effective_value, min_parsed) &&
            (min_parsed < 0.0 || min_parsed > 100.0)) {
            qmin.statuses.clear();
            AddStatus(qmin, TruthStatus::Missing);
            qmin.message = "quantile_min must be between 0 and 100.";
        }
        if (ParseDoubleValue(qmax.effective_value, max_parsed) &&
            (max_parsed < 0.0 || max_parsed > 100.0)) {
            qmax.statuses.clear();
            AddStatus(qmax, TruthStatus::Missing);
            qmax.message = "quantile_max must be between 0 and 100.";
        }
        if (ParseDoubleValue(qmin.effective_value, min_parsed) &&
            ParseDoubleValue(qmax.effective_value, max_parsed) &&
            max_parsed <= min_parsed) {
            qmax.statuses.clear();
            AddStatus(qmax, TruthStatus::Missing);
            qmax.message = "quantile_max must be greater than quantile_min.";
        }
        report.properties.push_back(std::move(qmin));
        report.properties.push_back(std::move(qmax));
    }
}

void AddEncoderTruth(NodeTruthReport& report, const MLNode& node) {
    if (node.type == NodeType::LabelEncoder) {
        report.properties.push_back(ResolveStringProperty(
            node,
            "Encoded column",
            "column",
            "",
            TruthOwner::Materializer,
            true,
            false,
            true));
        return;
    }

    report.properties.push_back(ResolveStringProperty(
        node,
        "Encoded columns",
        "columns",
        "",
        TruthOwner::Materializer,
        true,
        false,
        true));

    if (node.type == NodeType::OrdinalEncoder) {
        auto categories = ResolveStringProperty(
            node,
            "Category ordering",
            "categories",
            "auto",
            TruthOwner::Materializer,
            true,
            false,
            false);
        if (categories.effective_value != "auto") {
            categories.statuses.clear();
            AddStatus(categories, TruthStatus::Unsupported);
            categories.message =
                "OrdinalEncoder v1 only supports categories=auto.";
        }
        report.properties.push_back(std::move(categories));
        return;
    }

    if (node.type == NodeType::TargetEncoder) {
        report.properties.push_back(ResolveStringProperty(
            node,
            "Target column",
            "target_col",
            "",
            TruthOwner::Materializer,
            true,
            false,
            true));
        auto smoothing = ResolveFloatProperty(
            node,
            "Smoothing",
            "smoothing",
            "1.0",
            TruthOwner::Materializer,
            true,
            false);
        double parsed = 0.0;
        if (ParseDoubleValue(smoothing.effective_value, parsed) &&
            parsed < 0.0) {
            smoothing.statuses.clear();
            AddStatus(smoothing, TruthStatus::Missing);
            smoothing.message = "smoothing must be >= 0.";
        }
        report.properties.push_back(std::move(smoothing));
    }
}

void AddOutlierTruth(NodeTruthReport& report, const MLNode& node) {
    report.properties.push_back(ResolveStringProperty(
        node,
        "Inspected columns",
        "columns",
        "all",
        TruthOwner::Materializer,
        true,
        false,
        false,
        "all means the materializer auto-detects numeric columns."));

    auto method = ResolveStringProperty(
        node,
        "Detection method",
        "method",
        "iqr",
        TruthOwner::Materializer,
        true,
        false,
        false);
    if (method.effective_value != "iqr" && method.effective_value != "zscore") {
        method.statuses.clear();
        AddStatus(method, TruthStatus::Unsupported);
        method.message = "OutlierDetector v1 supports only iqr and zscore.";
    }
    report.properties.push_back(std::move(method));

    auto threshold = ResolveFloatProperty(
        node,
        "Threshold",
        "threshold",
        "1.5",
        TruthOwner::Materializer,
        true,
        false);
    double parsed = 0.0;
    if (ParseDoubleValue(threshold.effective_value, parsed) && parsed <= 0.0) {
        threshold.statuses.clear();
        AddStatus(threshold, TruthStatus::Missing);
        threshold.message = "threshold must be > 0.";
    }
    report.properties.push_back(std::move(threshold));

    auto action = ResolveStringProperty(
        node,
        "Action",
        "action",
        "flag",
        TruthOwner::Materializer,
        true,
        false,
        false,
        "v1 appends an is_outlier flag column.");
    if (action.effective_value != "flag") {
        action.statuses.clear();
        AddStatus(action, TruthStatus::Unsupported);
        action.message = "OutlierDetector v1 only supports action=flag.";
    }
    report.properties.push_back(std::move(action));
}

void AddVectorizerTruth(NodeTruthReport& report,
                        const MLNode& node,
                        bool count_vectorizer) {
    report.properties.push_back(ResolveStringProperty(
        node,
        "Text column",
        "text_col",
        "",
        TruthOwner::Materializer,
        true,
        false,
        true));

    auto max_features = ResolveIntProperty(
        node,
        "Effective feature width",
        "max_features",
        "2000",
        TruthOwner::Materializer,
        true,
        false,
        "Dense output width is capped by max_features.");
    RequirePositiveInt(max_features, "max_features");
    report.properties.push_back(std::move(max_features));

    if (!count_vectorizer) {
        auto min_df = ResolveIntProperty(
            node,
            "Minimum document frequency",
            "min_df",
            "1",
            TruthOwner::Materializer,
            true,
            false);
        RequirePositiveInt(min_df, "min_df");
        report.properties.push_back(std::move(min_df));
    }

    auto ngram_min = ResolveIntProperty(
        node,
        "Minimum n-gram",
        "ngram_min",
        "1",
        TruthOwner::Materializer,
        true,
        false);
    auto ngram_max = ResolveIntProperty(
        node,
        "Maximum n-gram",
        "ngram_max",
        "1",
        TruthOwner::Materializer,
        true,
        false);
    if (const std::string* range = FindParameter(node, "ngram_range");
        range && !range->empty()) {
        const auto comma = range->find(',');
        if (comma == std::string::npos) {
            ngram_max.statuses.clear();
            AddStatus(ngram_max, TruthStatus::Missing);
            ngram_max.message = "ngram_range must be formatted as min,max.";
        } else {
            ngram_min.source_key = "ngram_range";
            ngram_min.effective_value = range->substr(0, comma);
            ngram_min.statuses.clear();
            AddStatus(ngram_min, TruthStatus::AliasUsed);
            ngram_min.message =
                "ngram_range is canonical and overrides legacy ngram_min/ngram_max values.";
            ngram_max.source_key = "ngram_range";
            ngram_max.effective_value = range->substr(comma + 1);
            ngram_max.statuses.clear();
            AddStatus(ngram_max, TruthStatus::AliasUsed);
            ngram_max.message = ngram_min.message;
        }
    }
    RequirePositiveInt(ngram_min, "ngram_min");
    RequirePositiveInt(ngram_max, "ngram_max");
    const int min_parsed = ParsePositiveInt(&ngram_min.effective_value);
    const int max_parsed = ParsePositiveInt(&ngram_max.effective_value);
    if (min_parsed > 0 && max_parsed > 0 && min_parsed > max_parsed) {
        ngram_max.statuses.clear();
        AddStatus(ngram_max, TruthStatus::Missing);
        ngram_max.message = "ngram_max must be >= ngram_min.";
    } else if (max_parsed > 3) {
        ngram_max.statuses.clear();
        AddStatus(ngram_max, TruthStatus::Unsupported);
        ngram_max.message = "ngram_max > 3 is not supported yet.";
    }
    report.properties.push_back(std::move(ngram_min));
    report.properties.push_back(std::move(ngram_max));

    if (count_vectorizer) {
        report.properties.push_back(ResolveBoolProperty(
            node,
            "Binary counts",
            "binary",
            false,
            TruthOwner::Materializer,
            true,
            false));
    }

    auto output_format = ResolveStringProperty(
        node,
        "Output format",
        "output_format",
        "dense",
        TruthOwner::Materializer,
        true,
        false,
        false,
        "Current materializer support is dense Arrow feature columns.");
    if (output_format.effective_value != "dense") {
        output_format.statuses.clear();
        AddStatus(output_format, TruthStatus::Unsupported);
        output_format.message = "Sparse vectorizer output is planned but not executable.";
    }
    report.properties.push_back(std::move(output_format));
}

void RequireFloatAtLeast(PropertyTruth& truth,
                         const std::string& key,
                         double minimum,
                         bool inclusive) {
    double parsed = 0.0;
    if (!ParseDoubleValue(truth.effective_value, parsed) ||
        (inclusive ? parsed < minimum : parsed <= minimum)) {
        truth.statuses.clear();
        AddStatus(truth, TruthStatus::Missing);
        truth.message = key + (inclusive ? " must be >= " : " must be > ") +
                        std::to_string(minimum) + ".";
    }
}

void RequireFloatInRange(PropertyTruth& truth,
                         const std::string& key,
                         double minimum,
                         double maximum,
                         bool include_minimum,
                         bool include_maximum) {
    double parsed = 0.0;
    if (!ParseDoubleValue(truth.effective_value, parsed)) {
        truth.statuses.clear();
        AddStatus(truth, TruthStatus::Missing);
        truth.message = key + " must be a finite number.";
        return;
    }
    const bool below = include_minimum ? parsed < minimum : parsed <= minimum;
    const bool above = include_maximum ? parsed > maximum : parsed >= maximum;
    if (below || above) {
        truth.statuses.clear();
        AddStatus(truth, TruthStatus::Missing);
        truth.message = key + " must be in the supported range.";
    }
}

PropertyTruth ResolveLearningRateTruth(const MLNode& node,
                                       std::string default_value) {
    PropertyTruth truth;
    truth.label = "Learning rate";
    truth.canonical_key = "learning_rate";
    truth.default_value = std::move(default_value);
    truth.owner = TruthOwner::Runtime;
    truth.quick_editable = true;

    const std::string* learning_rate = FindParameter(node, "learning_rate");
    const std::string* lr = FindParameter(node, "lr");
    if (learning_rate && !learning_rate->empty()) {
        truth.source_key = "learning_rate";
        truth.effective_value = *learning_rate;
        AddStatus(truth, TruthStatus::OK);
        if (lr && !lr->empty() && *lr != *learning_rate) {
            truth.aliases_present.push_back({"lr", *lr});
            AddStatus(truth, TruthStatus::Conflicting);
            truth.message = "Legacy lr alias differs from learning_rate.";
        }
    } else if (lr && !lr->empty()) {
        truth.source_key = "lr";
        truth.effective_value = *lr;
        truth.aliases_present.push_back({"lr", *lr});
        AddStatus(truth, TruthStatus::AliasUsed);
    } else {
        truth.source_key = "default";
        truth.effective_value = truth.default_value;
        AddStatus(truth, TruthStatus::Defaulted);
    }
    RequireFloatAtLeast(truth, "learning_rate", 0.0, false);
    return truth;
}

void AddUnsupportedOptimizerParameterTruth(NodeTruthReport& report,
                                           const MLNode& node,
                                           const std::string& key,
                                           const std::string& label,
                                           const std::string& message) {
    const std::string* value = FindParameter(node, key);
    if (!value || value->empty()) {
        return;
    }
    PropertyTruth truth;
    truth.label = label;
    truth.canonical_key = key;
    truth.source_key = key;
    truth.effective_value = *value;
    truth.owner = TruthOwner::UI;
    truth.quick_editable = false;
    AddStatus(truth, TruthStatus::Unsupported);
    truth.message = message;
    report.properties.push_back(std::move(truth));
}

void AddOptimizerTruth(NodeTruthReport& report, const MLNode& node) {
    std::string default_lr = "0.001";
    if (node.type == NodeType::SGD || node.type == NodeType::Adagrad) {
        default_lr = "0.01";
    } else if (node.type == NodeType::NAdam) {
        default_lr = "0.002";
    }
    report.properties.push_back(ResolveLearningRateTruth(node, default_lr));

    const auto add_bounded = [&](const std::string& key,
                                 const std::string& label,
                                 const std::string& default_value,
                                 double minimum,
                                 bool include_minimum,
                                 double maximum,
                                 bool include_maximum) {
        auto truth = ResolveFloatProperty(
            node, label, key, default_value, TruthOwner::Runtime, true, false);
        RequireFloatInRange(truth, key, minimum, maximum,
                            include_minimum, include_maximum);
        report.properties.push_back(std::move(truth));
    };
    const auto add_positive = [&](const std::string& key,
                                  const std::string& label,
                                  const std::string& default_value) {
        auto truth = ResolveFloatProperty(
            node, label, key, default_value, TruthOwner::Runtime, true, false);
        RequireFloatAtLeast(truth, key, 0.0, false);
        report.properties.push_back(std::move(truth));
    };

    switch (node.type) {
        case NodeType::SGD:
            add_bounded("momentum", "Momentum", "0.9", 0.0, true,
                        1.0, false);
            AddUnsupportedOptimizerParameterTruth(
                report, node, "weight_decay", "Weight decay",
                "The current SGD backend has no weight-decay term; use AdamW "
                "or an explicit supported regularization path.");
            break;
        case NodeType::Adam:
        case NodeType::NAdam:
            add_bounded("beta1", "Beta1", "0.9", 0.0, true, 1.0, false);
            add_bounded("beta2", "Beta2", "0.999", 0.0, true, 1.0, false);
            add_positive("epsilon", "Epsilon", "1e-8");
            break;
        case NodeType::AdamW:
            add_bounded("beta1", "Beta1", "0.9", 0.0, true, 1.0, false);
            add_bounded("beta2", "Beta2", "0.999", 0.0, true, 1.0, false);
            add_positive("epsilon", "Epsilon", "1e-8");
            {
                auto weight_decay = ResolveFloatProperty(
                    node, "Weight decay", "weight_decay", "0.01",
                    TruthOwner::Runtime, true, false);
                RequireFloatAtLeast(weight_decay, "weight_decay", 0.0, true);
                report.properties.push_back(std::move(weight_decay));
            }
            break;
        case NodeType::RMSprop:
            add_bounded("alpha", "RMSprop alpha", "0.99", 0.0, true,
                        1.0, false);
            add_positive("epsilon", "Epsilon", "1e-8");
            add_bounded("momentum", "Momentum", "0.0", 0.0, true,
                        1.0, false);
            break;
        case NodeType::Adagrad:
            add_positive("epsilon", "Epsilon", "1e-10");
            AddUnsupportedOptimizerParameterTruth(
                report, node, "lr_decay", "Learning-rate decay",
                "The current Adagrad backend does not implement lr_decay.");
            break;
        default:
            break;
    }
}

bool IsLossNode(NodeType type) {
    return type == NodeType::MSELoss ||
           type == NodeType::CrossEntropyLoss ||
           type == NodeType::FocalLoss ||
           type == NodeType::BCELoss ||
           type == NodeType::BCEWithLogits ||
           type == NodeType::L1Loss ||
           type == NodeType::SmoothL1Loss ||
           type == NodeType::HuberLoss ||
           type == NodeType::NLLLoss ||
           type == NodeType::SoftDiceLoss ||
           type == NodeType::TverskyLoss ||
           type == NodeType::JaccardLoss;
}

void AddLossTruth(NodeTruthReport& report, const MLNode& node) {
    auto reduction = ResolveStringProperty(
        node,
        "Reduction",
        "reduction",
        "mean",
        TruthOwner::Runtime,
        true,
        false,
        false);
    if (reduction.effective_value != "mean" &&
        reduction.effective_value != "sum" &&
        reduction.effective_value != "none") {
        reduction.statuses.clear();
        AddStatus(reduction, TruthStatus::Unsupported);
        reduction.message = "Loss reduction must be one of mean, sum, none.";
    }
    report.properties.push_back(std::move(reduction));

    if (node.type == NodeType::CrossEntropyLoss) {
        auto label_smoothing = ResolveFloatProperty(
            node,
            "Label smoothing",
            "label_smoothing",
            "0.0",
            TruthOwner::Runtime,
            true,
            false);
        RequireFloatInRange(label_smoothing, "label_smoothing",
                            0.0, 1.0, true, false);
        report.properties.push_back(std::move(label_smoothing));

        auto class_weight = ResolveStringProperty(
            node,
            "Class weight mode",
            "class_weight",
            "none",
            TruthOwner::Runtime,
            true,
            false,
            false);
        if (class_weight.effective_value == "balanced") {
            class_weight.statuses.clear();
            AddStatus(class_weight, TruthStatus::Unsupported);
            class_weight.message =
                "class_weight=balanced is not resolved before loss construction; "
                "training falls back to unweighted CrossEntropy.";
        } else if (class_weight.effective_value == "manual" &&
                   !HasNonEmptyParameter(node, "class_weights")) {
            class_weight.statuses.clear();
            AddStatus(class_weight, TruthStatus::Missing);
            class_weight.message =
                "class_weight=manual requires class_weights.";
        }
        report.properties.push_back(std::move(class_weight));
        return;
    }

    if (node.type == NodeType::BCEWithLogits) {
        auto pos_weight = ResolveFloatProperty(
            node,
            "Positive-class weight",
            "pos_weight",
            "1.0",
            TruthOwner::Runtime,
            true,
            false);
        RequireFloatAtLeast(pos_weight, "pos_weight", 0.0, false);
        report.properties.push_back(std::move(pos_weight));
        return;
    }

    if (node.type == NodeType::FocalLoss) {
        auto alpha = ResolveFloatProperty(
            node, "Alpha", "alpha", "0.25", TruthOwner::Runtime, true, false);
        RequireFloatAtLeast(alpha, "alpha", 0.0, true);
        report.properties.push_back(std::move(alpha));
        auto gamma = ResolveFloatProperty(
            node, "Gamma", "gamma", "2.0", TruthOwner::Runtime, true, false);
        RequireFloatAtLeast(gamma, "gamma", 0.0, true);
        report.properties.push_back(std::move(gamma));
        return;
    }

    if (node.type == NodeType::SmoothL1Loss ||
        node.type == NodeType::HuberLoss) {
        auto beta = ResolveFloatProperty(
            node, "Beta", "beta", "1.0", TruthOwner::Runtime, true, false);
        RequireFloatAtLeast(beta, "beta", 0.0, false);
        report.properties.push_back(std::move(beta));
        return;
    }

    if (node.type == NodeType::SoftDiceLoss ||
        node.type == NodeType::JaccardLoss) {
        auto smooth = ResolveFloatProperty(
            node, "Smooth", "smooth", "1.0", TruthOwner::Runtime, true, false);
        RequireFloatAtLeast(smooth, "smooth", 0.0, true);
        report.properties.push_back(std::move(smooth));
        return;
    }

    if (node.type == NodeType::TverskyLoss) {
        auto alpha = ResolveFloatProperty(
            node, "Alpha", "alpha", "0.5", TruthOwner::Runtime, true, false);
        RequireFloatAtLeast(alpha, "alpha", 0.0, true);
        report.properties.push_back(std::move(alpha));
        auto beta = ResolveFloatProperty(
            node, "Beta", "beta", "0.5", TruthOwner::Runtime, true, false);
        RequireFloatAtLeast(beta, "beta", 0.0, true);
        report.properties.push_back(std::move(beta));
        auto smooth = ResolveFloatProperty(
            node, "Smooth", "smooth", "1.0", TruthOwner::Runtime, true, false);
        RequireFloatAtLeast(smooth, "smooth", 0.0, true);
        report.properties.push_back(std::move(smooth));
    }
}

void AddMetricTruth(NodeTruthReport& report, const MLNode& node) {
    report.properties.push_back(ResolveStringProperty(
        node,
        "Actual column",
        "actual_col",
        "",
        TruthOwner::Materializer,
        true,
        false,
        true));

    const bool score_metric = node.type == NodeType::ROCCurveNode ||
                              node.type == NodeType::PRCurveNode;
    report.properties.push_back(ResolveStringProperty(
        node,
        score_metric ? "Score column" : "Predicted column",
        score_metric ? "score_col" : "predicted_col",
        "",
        TruthOwner::Materializer,
        true,
        false,
        true));

    if (node.type == NodeType::RegressionMetricsNode ||
        node.type == NodeType::ClassificationMetricsNode) {
        report.properties.push_back(ResolveStringProperty(
            node,
            "Metrics",
            "metrics",
            node.type == NodeType::RegressionMetricsNode
                ? "mse,rmse,mae,r2"
                : "accuracy,precision,recall,f1,weighted_f1,count",
            TruthOwner::Materializer,
            true,
            false,
            false));
    }
}

PropertyTruth ResolveAliasedFloatProperty(const MLNode& node,
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

    double parsed = 0.0;
    if (!ParseDoubleValue(truth.effective_value, parsed)) {
        truth.statuses.clear();
        AddStatus(truth, TruthStatus::Missing);
        truth.message = truth.canonical_key + " must be a finite number.";
    }
    if (truth.requires_dialog) {
        AddStatus(truth, TruthStatus::RequiresDialog);
    }
    return truth;
}

bool IsActivationNode(NodeType type) {
    return cyxwiz::IsExecutableActivationNode(type);
}

bool IsShapeOpNode(NodeType type) {
    return type == NodeType::Flatten ||
           type == NodeType::Reshape ||
           type == NodeType::View ||
           type == NodeType::Permute ||
           type == NodeType::Squeeze ||
           type == NodeType::Unsqueeze;
}

void AddCoreLayerTruth(NodeTruthReport& report, const MLNode& node) {
    const auto add_executable_configuration = [&]() {
        PropertyTruth configuration;
        configuration.label = "Executable configuration";
        configuration.canonical_key = "normalization_regularization_contract";
        configuration.source_key = "shared compiler/runtime policy";
        configuration.owner = TruthOwner::Compiler;
        configuration.quick_editable = false;
        if (const auto error =
                cyxwiz::ResolveInvalidNormalizationRegularizationConfigurationReason(
                    node.type, node.parameters)) {
            configuration.effective_value = "rejected";
            configuration.message = *error;
            AddStatus(configuration, TruthStatus::Unsupported);
        } else {
            configuration.effective_value = "validated exact construction";
            AddStatus(configuration, TruthStatus::OK);
        }
        report.properties.push_back(std::move(configuration));
    };

    if (node.type == NodeType::Dense ||
        node.type == NodeType::TimeDistributed) {
        auto units = node.type == NodeType::TimeDistributed
            ? ResolveAliasedIntProperty(
                  node,
                  "Per-timestep output units",
                  "units",
                  {"out_features"},
                  "128",
                  TruthOwner::Compiler,
                  true,
                  false,
                  "GraphCompiler and ModelBuilder use this as the shared Dense output width at every timestep.")
            : ResolveIntProperty(
                  node,
                  "Output units",
                  "units",
                  "64",
                  TruthOwner::Compiler,
                  true,
                  false,
                  "GraphCompiler and ModelBuilder use units as the linear output width.");
        RequirePositiveInt(units, "units");
        report.properties.push_back(std::move(units));

        const std::string* activation = FindParameter(node, "activation");
        if (activation && !activation->empty() && *activation != "none") {
            PropertyTruth activation_truth;
            activation_truth.label = "Inline activation";
            activation_truth.canonical_key = "activation";
            activation_truth.source_key = "activation";
            activation_truth.effective_value = *activation;
            activation_truth.owner = TruthOwner::Compiler;
            activation_truth.quick_editable = false;
            AddStatus(activation_truth, TruthStatus::Unsupported);
            activation_truth.message =
                "Dense activation parameters are not consumed by ModelBuilder; "
                "use an explicit activation node after Dense.";
            report.properties.push_back(std::move(activation_truth));
        }
        return;
    }

    if (node.type == NodeType::Dropout) {
        auto rate = ResolveFloatProperty(
            node,
            "Dropout rate",
            "rate",
            "0.5",
            TruthOwner::Compiler,
            true,
            false,
            "Dropout preserves tensor shape and randomly zeros activations during training.");
        RequireFloatInRange(rate, "rate", 0.0, 1.0, true, true);
        report.properties.push_back(std::move(rate));
        add_executable_configuration();
        return;
    }

    if (node.type == NodeType::BatchNorm) {
        auto eps = ResolveAliasedFloatProperty(
            node,
            "Epsilon",
            "eps",
            {"epsilon"},
            "1e-5",
            TruthOwner::Compiler,
            true,
            false,
            "GraphCompiler consumes eps; legacy epsilon maps to eps for truth.");
        RequireFloatAtLeast(eps, "eps", 0.0, false);
        report.properties.push_back(std::move(eps));

        auto momentum = ResolveFloatProperty(
            node,
            "Momentum",
            "momentum",
            "0.1",
            TruthOwner::Compiler,
            true,
            false);
        RequireFloatInRange(momentum, "momentum", 0.0, 1.0, true, true);
        report.properties.push_back(std::move(momentum));
        add_executable_configuration();
        return;
    }

    if (node.type == NodeType::LayerNorm) {
        auto normalized_shape = ResolveStringProperty(
            node,
            "Normalized shape",
            "normalized_shape",
            "",
            TruthOwner::Compiler,
            true,
            false,
            false,
            "Empty uses the incoming feature width; otherwise enter positive trailing dimensions separated by commas.");
        if (normalized_shape.effective_value.empty()) {
            normalized_shape.effective_value = "automatic current feature width";
        }
        report.properties.push_back(std::move(normalized_shape));

        auto eps = ResolveAliasedFloatProperty(
            node,
            "Epsilon",
            "eps",
            {"epsilon"},
            "1e-5",
            TruthOwner::Runtime,
            true,
            false,
            "ModelBuilder accepts legacy epsilon and canonical eps.");
        RequireFloatAtLeast(eps, "eps", 0.0, false);
        report.properties.push_back(std::move(eps));
        report.properties.push_back(ResolveBoolProperty(
            node,
            "Elementwise affine",
            "elementwise_affine",
            true,
            TruthOwner::Runtime,
            true,
            false,
            "When false, LayerNorm has no trainable scale or bias."));
        add_executable_configuration();
        return;
    }

    if (IsActivationNode(node.type)) {
        PropertyTruth shape;
        shape.label = "Shape effect";
        shape.canonical_key = "shape_effect";
        shape.source_key = "compiler";
        shape.effective_value = "preserves input shape";
        shape.owner = TruthOwner::Compiler;
        shape.quick_editable = false;
        AddStatus(shape, TruthStatus::OK);
        report.properties.push_back(std::move(shape));

        if (node.type == NodeType::LeakyReLU) {
            auto slope = ResolveFloatProperty(
                node,
                "Negative slope",
                "negative_slope",
                "0.01",
                TruthOwner::Compiler,
                true,
                false);
            RequireFloatAtLeast(slope, "negative_slope", 0.0, true);
            report.properties.push_back(std::move(slope));
        } else if (node.type == NodeType::ELU) {
            auto alpha = ResolveFloatProperty(
                node,
                "Negative saturation scale",
                "alpha",
                "1.0",
                TruthOwner::Compiler,
                true,
                false,
                "GraphCompiler and ModelBuilder use this exact positive ELU alpha.");
            RequireFloatAtLeast(alpha, "alpha", 0.0, false);
            report.properties.push_back(std::move(alpha));
        }

        PropertyTruth configuration;
        configuration.label = "Executable configuration";
        configuration.canonical_key = "dense_activation_contract";
        configuration.source_key = "shared compiler/runtime policy";
        configuration.owner = TruthOwner::Compiler;
        configuration.quick_editable = false;
        if (const auto error =
                cyxwiz::ResolveInvalidDenseActivationConfigurationReason(
                    node.type, node.parameters)) {
            configuration.effective_value = "rejected";
            configuration.message = *error;
            AddStatus(configuration, TruthStatus::Unsupported);
        } else {
            configuration.effective_value =
                "validated shape-preserving activation";
            AddStatus(configuration, TruthStatus::OK);
        }
        report.properties.push_back(std::move(configuration));
        return;
    }

    if (node.type == NodeType::Flatten) {
        PropertyTruth shape;
        shape.label = "Shape effect";
        shape.canonical_key = "shape_effect";
        shape.source_key = "compiler";
        shape.effective_value = "flattens all input dimensions";
        shape.owner = TruthOwner::Compiler;
        shape.quick_editable = false;
        AddStatus(shape, TruthStatus::OK);
        report.properties.push_back(std::move(shape));
        return;
    }

    if (node.type == NodeType::Reshape || node.type == NodeType::View) {
        report.properties.push_back(ResolveStringProperty(
            node,
            "Target shape",
            "shape",
            "-1,256",
            TruthOwner::Compiler,
            true,
            false,
            true,
            "One -1 entry can be inferred by the compiler."));
        return;
    }

    if (node.type == NodeType::Permute) {
        report.properties.push_back(ResolveStringProperty(
            node,
            "Dimension order",
            "dims",
            "0,2,1",
            TruthOwner::Compiler,
            true,
            false,
            true));
        return;
    }

    if (node.type == NodeType::Squeeze ||
        node.type == NodeType::Unsqueeze) {
        auto dim = ResolveIntProperty(
            node,
            "Dimension",
            "dim",
            "0",
            TruthOwner::Compiler,
            true,
            false);
        report.properties.push_back(std::move(dim));
    }
}

void AddEmbeddingTruth(NodeTruthReport& report, const MLNode& node) {
    auto vocabulary_size = ResolveIntProperty(
        node, "Vocabulary size", "num_embeddings", "10000",
        TruthOwner::Runtime, false, true,
        "ModelBuilder uses this as the trainable lookup-table row count.");
    RequirePositiveInt(vocabulary_size, "num_embeddings");
    if (ParsePositiveInt(&vocabulary_size.effective_value) == 1) {
        vocabulary_size.statuses.clear();
        AddStatus(vocabulary_size, TruthStatus::Missing);
        vocabulary_size.message = "num_embeddings must be >= 2.";
    }
    const int vocabulary_count =
        ParsePositiveInt(&vocabulary_size.effective_value);
    report.properties.push_back(std::move(vocabulary_size));

    auto embedding_dim = ResolveIntProperty(
        node, "Embedding dimension", "embedding_dim", "256",
        TruthOwner::Runtime, false, true,
        "ModelBuilder uses this as the Float32 feature width for each token.");
    RequirePositiveInt(embedding_dim, "embedding_dim");
    report.properties.push_back(std::move(embedding_dim));

    auto padding = ResolveIntProperty(
        node, "Padding index", "padding_idx", "-1",
        TruthOwner::Runtime, false, true,
        "-1 disables padding; a configured token row emits zeros and receives no gradient.");
    char* padding_end = nullptr;
    const long padding_value =
        std::strtol(padding.effective_value.c_str(), &padding_end, 10);
    if (padding_end == padding.effective_value.c_str() ||
        *padding_end != '\0' || padding_value < -1 ||
        (vocabulary_count > 0 && padding_value >= vocabulary_count)) {
        padding.statuses.clear();
        AddStatus(padding, TruthStatus::Missing);
        AddStatus(padding, TruthStatus::RequiresDialog);
        padding.message =
            "padding_idx must be -1 or smaller than num_embeddings.";
    }
    report.properties.push_back(std::move(padding));

    auto max_norm = ResolveFloatProperty(
        node, "Maximum vector norm", "max_norm", "0",
        TruthOwner::Runtime, false, true,
        "0 disables clipping; positive values cap each row's L2 norm before lookup.");
    double max_norm_value = 0.0;
    if (ParseDoubleValue(max_norm.effective_value, max_norm_value) &&
        max_norm_value < 0.0) {
        max_norm.statuses.clear();
        AddStatus(max_norm, TruthStatus::Missing);
        AddStatus(max_norm, TruthStatus::RequiresDialog);
        max_norm.message = "max_norm must be >= 0.";
    }
    report.properties.push_back(std::move(max_norm));

    report.properties.push_back(ResolveBoolProperty(
        node, "Freeze loaded weights", "freeze", false,
        TruthOwner::Runtime, false, true,
        "When true, a loaded pretrained table is excluded from optimizer updates."));

    report.properties.push_back(ResolveAliasedStringPropertyWithDefault(
        node,
        "Pretrained weights file",
        "weights_file",
        {"embedding_weights_file"},
        "",
        TruthOwner::Runtime,
        false,
        true,
        false,
        "Empty uses the backend's trainable random-normal initialization. The legacy alias remains readable."));

    auto init_mode = ResolveStringProperty(
        node, "Starter matrix initialization", "init_mode", "normal",
        TruthOwner::UI, false, true, false,
        "Used only by Build, Save, and Use in the Embedding dialog; it does not change runtime initialization without a weights file.");
    if (init_mode.effective_value != "normal" &&
        init_mode.effective_value != "uniform" &&
        init_mode.effective_value != "one_hot") {
        init_mode.statuses.clear();
        AddStatus(init_mode, TruthStatus::Missing);
        AddStatus(init_mode, TruthStatus::RequiresDialog);
        init_mode.message = "init_mode must be normal, uniform, or one_hot.";
    } else {
        AddStatus(init_mode, TruthStatus::CompilerOnly);
    }
    report.properties.push_back(std::move(init_mode));

}

void AddTransformerTruth(NodeTruthReport& report, const MLNode& node) {
    const bool is_attention = node.type == NodeType::MultiHeadAttention;
    const bool is_positional = node.type == NodeType::PositionalEncoding;

    auto model_width = ResolveAliasedIntProperty(
        node,
        "Model width",
        is_attention ? "embed_dim" : "d_model",
        is_attention ? std::vector<std::string>{"d_model"}
                     : std::vector<std::string>{"embed_dim"},
        "512",
        TruthOwner::Compiler,
        true,
        false,
        "The configured width must match the incoming feature dimension.");
    RequirePositiveInt(model_width, is_attention ? "embed_dim" : "d_model");
    report.properties.push_back(std::move(model_width));

    if (is_positional) {
        auto maximum_length = ResolveAliasedIntProperty(
            node,
            "Maximum sequence length",
            "max_sequence_length",
            {"max_len", "max_length", "max_seq_len"},
            "5000",
            TruthOwner::Runtime,
            true,
            false,
            "Sequences longer than this bound fail closed.");
        RequirePositiveInt(maximum_length, "max_sequence_length");
        report.properties.push_back(std::move(maximum_length));
    } else {
        auto heads = ResolveAliasedIntProperty(
            node,
            "Attention heads",
            "num_heads",
            is_attention ? std::vector<std::string>{"heads"}
                         : std::vector<std::string>{"nhead"},
            "8",
            TruthOwner::Compiler,
            true,
            false,
            "Model width must divide evenly by this value.");
        RequirePositiveInt(heads, "num_heads");
        report.properties.push_back(std::move(heads));

        auto dropout = ResolveAliasedFloatProperty(
            node,
            "Dropout",
            "dropout",
            {"dropout_rate"},
            is_attention ? "0.0" : "0.1",
            TruthOwner::Runtime,
            true,
            false);
        RequireFloatInRange(dropout, "dropout", 0.0, 1.0, true, false);
        report.properties.push_back(std::move(dropout));

        if (is_attention) {
            report.properties.push_back(ResolveBoolProperty(
                node, "Projection bias", "use_bias", true,
                TruthOwner::Runtime, true, false));
        } else {
            auto feedforward = ResolveAliasedIntProperty(
                node,
                "Feedforward width",
                "dim_feedforward",
                {"ff_dim", "d_ff"},
                "2048",
                TruthOwner::Runtime,
                true,
                false);
            RequirePositiveInt(feedforward, "dim_feedforward");
            report.properties.push_back(std::move(feedforward));
            report.properties.push_back(ResolveBoolProperty(
                node, "Normalize first", "norm_first", false,
                TruthOwner::Runtime, true, false));

            if (const std::string* layers = FindParameter(node, "num_layers")) {
                auto layer_count = ResolveIntProperty(
                    node,
                    "Legacy layer count",
                    "num_layers",
                    "1",
                    TruthOwner::Compiler,
                    false,
                    false,
                    "One node owns exactly one block; stack nodes for depth.");
                layer_count.statuses.clear();
                if (*layers == "1") {
                    AddStatus(layer_count, TruthStatus::CompilerOnly);
                } else {
                    AddStatus(layer_count, TruthStatus::Unsupported);
                }
                report.properties.push_back(std::move(layer_count));
            }
        }
    }

    PropertyTruth configuration;
    configuration.label = "Executable configuration";
    configuration.canonical_key = "transformer_contract";
    configuration.source_key = "shared compiler/runtime policy";
    configuration.owner = TruthOwner::Compiler;
    configuration.quick_editable = false;
    if (const auto error = cyxwiz::ResolveInvalidTransformerConfigurationReason(
            node.type, node.parameters)) {
        configuration.effective_value = "rejected";
        configuration.message = *error;
        AddStatus(configuration, TruthStatus::Unsupported);
    } else {
        configuration.effective_value = is_positional
            ? "one deterministic CPU-backed encoding"
            : "one CPU-backed attention block";
        AddStatus(configuration, TruthStatus::OK);
    }
    report.properties.push_back(std::move(configuration));
}

void AddSequenceVocabularyTruth(NodeTruthReport& report, const MLNode& node) {
    const bool is_token = node.type == NodeType::TokenVocabulary;
    const bool is_pos = node.type == NodeType::POSVocabulary;
    const bool is_tag = node.type == NodeType::NERTagVocabulary;
    const std::string default_column =
        is_token ? "tokens" : (is_pos ? "pos_tags" : "ner_tags");

    report.properties.push_back(ResolveStringProperty(
        node,
        "Source column",
        "column",
        default_column,
        TruthOwner::Materializer,
        true,
        false,
        true,
        "PipelineExecutor reads this sequence column from the input table."));

    auto min_frequency = ResolveAliasedIntProperty(
        node,
        "Minimum frequency",
        "min_frequency",
        {"min_freq"},
        "1",
        TruthOwner::Materializer,
        false,
        false,
        "Runtime accepts canonical min_frequency and editor alias min_freq.");
    RequirePositiveInt(min_frequency, "min_frequency");
    report.properties.push_back(std::move(min_frequency));

    auto max_size = ResolveAliasedIntProperty(
        node,
        "Max vocabulary size",
        "max_size",
        {"max_vocab_size"},
        "0",
        TruthOwner::Materializer,
        false,
        false,
        "0 means unlimited; runtime accepts editor alias max_vocab_size.");
    RequireNonNegativeInt(max_size, "max_size");
    report.properties.push_back(std::move(max_size));

    if (!is_tag) {
        report.properties.push_back(ResolveBoolProperty(
            node,
            "Lowercase values",
            "lowercase",
            is_token,
            TruthOwner::Materializer,
            true,
            false));
        report.properties.push_back(ResolveStringProperty(
            node,
            "Padding token",
            "pad_token",
            "[PAD]",
            TruthOwner::Materializer,
            true,
            false,
            false));
        report.properties.push_back(ResolveStringProperty(
            node,
            "Unknown token",
            "unk_token",
            "[UNK]",
            TruthOwner::Materializer,
            true,
            false,
            false));
        return;
    }

    auto outside_tag = ResolveStringProperty(
        node,
        "Outside tag",
        "outside_tag",
        "O",
        TruthOwner::Materializer,
        true,
        false,
        false,
        "The current tag vocabulary builder orders hard-coded O first.");
    if (outside_tag.effective_value != "O") {
        outside_tag.statuses.clear();
        AddStatus(outside_tag, TruthStatus::Unsupported);
        outside_tag.message =
            "Custom outside_tag values are not consumed by BuildSequenceVocabulary.";
    }
    report.properties.push_back(std::move(outside_tag));

    auto bio_scheme = ResolveStringProperty(
        node,
        "Tag scheme",
        "bio_scheme",
        "BIO",
        TruthOwner::Materializer,
        true,
        false,
        false);
    if (bio_scheme.effective_value != "BIO") {
        bio_scheme.statuses.clear();
        AddStatus(bio_scheme, TruthStatus::Unsupported);
        bio_scheme.message = "Only BIO tag vocabularies are supported.";
    }
    report.properties.push_back(std::move(bio_scheme));
}

void AddNERSequenceBuilderTruth(NodeTruthReport& report, const MLNode& node) {
    report.properties.push_back(ResolveAliasedStringPropertyWithDefault(
        node,
        "Token column",
        "token_column",
        {"tokens_column", "token_sequence_column"},
        "tokens",
        TruthOwner::Materializer,
        true,
        false,
        true,
        "Compiler and materializer use this as the token sequence source."));
    report.properties.push_back(ResolveAliasedStringPropertyWithDefault(
        node,
        "POS column",
        "pos_column",
        {"pos_sequence_column"},
        "",
        TruthOwner::Materializer,
        true,
        false,
        false,
        "Empty POS column disables POS ids."));
    report.properties.push_back(ResolveAliasedStringPropertyWithDefault(
        node,
        "Tag column",
        "tag_column",
        {"tags_column", "tag_sequence_column"},
        "ner_tags",
        TruthOwner::Materializer,
        true,
        false,
        true));
    report.properties.push_back(ResolveAliasedStringPropertyWithDefault(
        node,
        "Sentence id column",
        "sentence_id_column",
        {"sequence_id_column"},
        "",
        TruthOwner::Materializer,
        true,
        false,
        false));
    report.properties.push_back(ResolveAliasedStringPropertyWithDefault(
        node,
        "Target ids column",
        "target_column",
        {"target_ids_column", "decoder_target_column"},
        "",
        TruthOwner::Compiler,
        false,
        false,
        false,
        "GraphCompiler copies target aliases into the sequence batch contract."));

    auto max_length = ResolveIntProperty(
        node,
        "Max sequence length",
        "max_sequence_length",
        "0",
        TruthOwner::Compiler,
        true,
        false,
        "0 means infer from data; negative values are clamped by runtime.");
    RequireNonNegativeInt(max_length, "max_sequence_length");
    report.properties.push_back(std::move(max_length));

    report.properties.push_back(ResolveIntProperty(
        node,
        "Padding label ignore index",
        "ignore_index",
        "-100",
        TruthOwner::Compiler,
        true,
        false));
    report.properties.push_back(ResolveIntProperty(
        node,
        "Target ignore index",
        "target_ignore_index",
        "-100",
        TruthOwner::Compiler,
        false,
        false));
    report.properties.push_back(ResolveBoolProperty(
        node,
        "Create attention mask",
        "create_attention_mask",
        true,
        TruthOwner::Compiler,
        true,
        false));

    auto min_frequency = ResolveAliasedIntProperty(
        node,
        "Vocabulary minimum frequency",
        "min_frequency",
        {"min_freq"},
        "1",
        TruthOwner::Materializer,
        false,
        false,
        "NERSequenceBuilder applies one vocabulary threshold to token/POS/tag vocabularies.");
    RequirePositiveInt(min_frequency, "min_frequency");
    report.properties.push_back(std::move(min_frequency));

    auto max_size = ResolveAliasedIntProperty(
        node,
        "Vocabulary max size",
        "max_size",
        {"max_vocab_size"},
        "0",
        TruthOwner::Materializer,
        false,
        false,
        "0 means unlimited; applied to token/POS/tag vocabularies.");
    RequireNonNegativeInt(max_size, "max_size");
    report.properties.push_back(std::move(max_size));
}

void AddSequenceTagOutputTruth(NodeTruthReport& report, const MLNode& node) {
    auto num_tags = ResolveIntProperty(
        node,
        "Number of tags",
        "num_tags",
        "0",
        TruthOwner::Compiler,
        true,
        false,
        "0 means infer; positive values drive shape and CrossEntropy class-count validation.");
    RequireNonNegativeInt(num_tags, "num_tags");
    report.properties.push_back(std::move(num_tags));

    report.properties.push_back(ResolveStringProperty(
        node,
        "Tag vocabulary file",
        "tag_vocab_file",
        "",
        TruthOwner::Exporter,
        true,
        false,
        false,
        "Exporter uses this path as the sequence tag vocabulary asset."));

    auto decode_scheme = ResolveStringProperty(
        node,
        "Decode scheme",
        "decode_scheme",
        "BIO",
        TruthOwner::Exporter,
        true,
        false,
        false);
    if (decode_scheme.effective_value != "BIO") {
        decode_scheme.statuses.clear();
        AddStatus(decode_scheme, TruthStatus::Unsupported);
        decode_scheme.message = "Only BIO sequence-tag decode metadata is supported.";
    }
    report.properties.push_back(std::move(decode_scheme));
}

PropertyTruth BuildReadOnlyRuntimeTruth(std::string label,
                                        std::string canonical_key,
                                        std::string effective_value,
                                        std::string message = {}) {
    PropertyTruth truth;
    truth.label = std::move(label);
    truth.canonical_key = std::move(canonical_key);
    truth.source_key = "PipelineExecutor";
    truth.effective_value = std::move(effective_value);
    truth.owner = TruthOwner::Runtime;
    truth.quick_editable = false;
    truth.message = std::move(message);
    AddStatus(truth, TruthStatus::OK);
    return truth;
}

void AddDataOutputTruth(NodeTruthReport& report, const MLNode& node) {
    auto output_file = ResolveAliasedStringPropertyWithDefault(
        node,
        "Output file",
        "file_path",
        {"path"},
        "",
        TruthOwner::Exporter,
        true,
        true,
        true,
        "PipelineExecutor uses a working-directory default when no path is supplied; legacy path is accepted.");
    if (output_file.effective_value.empty() &&
        HasStatus(output_file.statuses, TruthStatus::Missing)) {
        output_file.statuses.clear();
        AddStatus(output_file, TruthStatus::Defaulted);
        if (output_file.requires_dialog) {
            AddStatus(output_file, TruthStatus::RequiresDialog);
        }
        output_file.message = "No output path set; runtime will create a working-directory export file.";
    }
    report.properties.push_back(std::move(output_file));

    auto format = ResolveAliasedStringPropertyWithDefault(
        node,
        "Output format",
        "file_type",
        {"format"},
        "csv",
        TruthOwner::Exporter,
        true,
        true,
        false,
        "DataOutput supports CSV and Parquet table export.");
    format.effective_value = NormalizeAsciiToken(format.effective_value);
    if (format.effective_value != "csv" && format.effective_value != "parquet") {
        format.statuses.clear();
        AddStatus(format, TruthStatus::Unsupported);
        format.message = "DataOutput runtime supports only csv and parquet.";
    }
    report.properties.push_back(std::move(format));

    report.properties.push_back(BuildReadOnlyRuntimeTruth(
        "Runtime result",
        "export_result",
        "passes through input dataset and sets output path",
        "On success, the input dataset remains available downstream and ctx.output_dataset is the output file path."));
}

const char* ExportNodeFormat(NodeType type) {
    switch (type) {
    case NodeType::ExportCSV: return "csv";
    case NodeType::ExportParquet: return "parquet";
    case NodeType::ExportJSON: return "json";
    default: break;
    }
    return "";
}

void AddFixedExportTruth(NodeTruthReport& report, const MLNode& node) {
    const std::string format = ExportNodeFormat(node.type);
    auto output_file = ResolveAliasedStringPropertyWithDefault(
        node,
        "Output file",
        "file_path",
        {"path"},
        "",
        TruthOwner::Exporter,
        true,
        false,
        true,
        "PipelineExecutor uses a working-directory default when no path is supplied; legacy path is accepted.");
    if (output_file.effective_value.empty() &&
        HasStatus(output_file.statuses, TruthStatus::Missing)) {
        output_file.statuses.clear();
        AddStatus(output_file, TruthStatus::Defaulted);
        output_file.message = "No output path set; runtime will create a working-directory export file.";
    }
    report.properties.push_back(std::move(output_file));

    report.properties.push_back(BuildReadOnlyRuntimeTruth(
        "Export format",
        "export_format",
        format,
        node.type == NodeType::ExportJSON
            ? "ExportJSON writes Arrow table rows as a JSON array."
            : "Exporter writes the input Arrow table through DataRegistry."));
}

bool HasIncomingLink(const MLNode& node, const NodeTruthContext& context) {
    if (!context.links) {
        return false;
    }
    return std::any_of(context.links->begin(), context.links->end(),
                       [&node](const NodeLink& link) {
                           return link.to_node == node.id;
                       });
}

void AddDataConvertTruth(NodeTruthReport& report,
                         const MLNode& node,
                         const NodeTruthContext& context) {
    const bool has_upstream_dataset = HasIncomingLink(node, context);
    auto input_path = ResolveStringProperty(
        node,
        "Input file",
        "input_path",
        "",
        TruthOwner::Loader,
        true,
        true,
        !has_upstream_dataset,
        "Required only when no upstream dataset is connected.");
    if (has_upstream_dataset && HasStatus(input_path.statuses,
                                          TruthStatus::Missing)) {
        input_path.statuses.clear();
        AddStatus(input_path, TruthStatus::RuntimeOnly);
        input_path.message =
            "Input is resolved from the connected upstream Arrow dataset.";
    }
    report.properties.push_back(std::move(input_path));

    report.properties.push_back(ResolveStringProperty(
        node,
        "Output file",
        "output_path",
        "",
        TruthOwner::Exporter,
        true,
        true,
        true,
        "DataConvert writes a file and registers the typed in-memory result without reparsing it."));
    report.properties.push_back(ResolveStringProperty(
        node,
        "Input format",
        "input_format",
        "auto",
        TruthOwner::Loader,
        true,
        true,
        false));
    report.properties.push_back(ResolveStringProperty(
        node,
        "Output format",
        "output_format",
        "auto",
        TruthOwner::Exporter,
        true,
        true,
        false));
    report.properties.push_back(ResolveStringProperty(
        node,
        "Input decimal separator",
        "decimal_point",
        ".",
        TruthOwner::Loader,
        true,
        true,
        false));
    report.properties.push_back(BuildReadOnlyRuntimeTruth(
        "Runtime result",
        "convert_result",
        "file plus registered dataset",
        "On success, DataConvert writes the output file and registers the typed table as ds_dataconvert_<node id>. A fresh cached output is reparsed only when no in-memory table exists."));
}

void AddDeployToNodeEditorTruth(NodeTruthReport& report, const MLNode& node) {
    report.properties.push_back(ResolveStringProperty(
        node,
        "Deployment dataset name",
        "name",
        "deployed_" + std::to_string(node.id),
        TruthOwner::Runtime,
        true,
        false,
        false,
        "When empty, PipelineExecutor uses deployed_<node id>."));
    report.properties.push_back(BuildReadOnlyRuntimeTruth(
        "Runtime result",
        "deployment_result",
        "deployment_ready=true",
        "On success, ctx.deployment_dataset and ctx.output_dataset are set to the deployment dataset name."));
}

void AddDataProfilerTruth(NodeTruthReport& report, const MLNode& node) {
    auto minimal = ResolveBoolProperty(
        node,
        "Minimal mode",
        "minimal",
        false,
        TruthOwner::Runtime,
        false,
        false,
        "Current DataProfiler executor always emits the same per-column profile schema.");
    if (minimal.effective_value == "true") {
        minimal.statuses.clear();
        AddStatus(minimal, TruthStatus::Unsupported);
        minimal.message =
            "Legacy minimal=true values are not consumed by ExecuteDataProfiler.";
    }
    report.properties.push_back(std::move(minimal));
    report.properties.push_back(BuildReadOnlyRuntimeTruth(
        "Report schema",
        "profile_report_schema",
        "column,type,nullable,row_count,null_count,non_null_count",
        "ExecuteDataProfiler registers ds_dataprofiler_<node id>."));
}

void AddTreeModelPredictorTruth(NodeTruthReport& report, const MLNode& node) {
    report.properties.push_back(ResolveStringProperty(
        node,
        "Model artifact",
        "model_path",
        "",
        TruthOwner::Runtime,
        true,
        false,
        true,
        "TreeModelPredictor requires a saved CyxWiz tree-family JSON artifact."));
    report.properties.push_back(ResolveStringProperty(
        node,
        "Feature columns",
        "feature_cols",
        "artifact feature order",
        TruthOwner::Runtime,
        true,
        false,
        false,
        "Empty feature_cols uses feature names stored in the model artifact."));
    report.properties.push_back(ResolveStringProperty(
        node,
        "Prediction column",
        "prediction_col",
        "prediction",
        TruthOwner::Runtime,
        true,
        false,
        false));
    report.properties.push_back(BuildReadOnlyRuntimeTruth(
        "Inference result",
        "inference_result",
        "input table plus prediction column",
        "PipelineOperatorFactory routes TreeModelPredictor to the native tree model predictor operator."));
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

const std::vector<NodeType>& SpecializedTruthCoverageNodeTypes() {
    static const std::vector<NodeType> node_types = {
        NodeType::DataInput,
        NodeType::DataOutput,
        NodeType::DataConvert,
        NodeType::DeployToNodeEditorNode,
        NodeType::DataLoader,
        NodeType::DataProfiler,
        NodeType::StandardScaler,
        NodeType::MinMaxScaler,
        NodeType::RobustScaler,
        NodeType::LabelEncoder,
        NodeType::OrdinalEncoder,
        NodeType::TargetEncoder,
        NodeType::OutlierDetector,
        NodeType::TFIDFVectorizer,
        NodeType::CountVectorizer,
        NodeType::TextTokenizer,
        NodeType::RegressionMetricsNode,
        NodeType::ClassificationMetricsNode,
        NodeType::ConfusionMatrixNode,
        NodeType::ROCCurveNode,
        NodeType::PRCurveNode,
        NodeType::Dense,
        NodeType::Embedding,
        NodeType::TimeDistributed,
        NodeType::MultiHeadAttention,
        NodeType::TransformerEncoder,
        NodeType::TransformerDecoder,
        NodeType::PositionalEncoding,
        NodeType::Dropout,
        NodeType::BatchNorm,
        NodeType::LayerNorm,
        NodeType::ReLU,
        NodeType::Sigmoid,
        NodeType::Softmax,
        NodeType::GELU,
        NodeType::Tanh,
        NodeType::LeakyReLU,
        NodeType::ELU,
        NodeType::Swish,
        NodeType::Mish,
        NodeType::Flatten,
        NodeType::Reshape,
        NodeType::View,
        NodeType::Permute,
        NodeType::Squeeze,
        NodeType::Unsqueeze,
        NodeType::LSTM,
        NodeType::GRU,
        NodeType::NERSequenceBuilder,
        NodeType::TokenVocabulary,
        NodeType::POSVocabulary,
        NodeType::NERTagVocabulary,
        NodeType::SequenceTagOutput,
        NodeType::MSELoss,
        NodeType::FocalLoss,
        NodeType::BCELoss,
        NodeType::BCEWithLogits,
        NodeType::L1Loss,
        NodeType::SmoothL1Loss,
        NodeType::HuberLoss,
        NodeType::NLLLoss,
        NodeType::SoftDiceLoss,
        NodeType::TverskyLoss,
        NodeType::JaccardLoss,
        NodeType::Adam,
        NodeType::SGD,
        NodeType::AdamW,
        NodeType::RMSprop,
        NodeType::Adagrad,
        NodeType::NAdam,
        NodeType::Output,
        NodeType::CrossEntropyLoss,
        NodeType::ExportCSV,
        NodeType::ExportParquet,
        NodeType::ExportJSON,
        NodeType::TreeModelPredictor,
    };
    return node_types;
}

bool HasSpecializedTruthCoverage(NodeType type) {
    const auto& node_types = SpecializedTruthCoverageNodeTypes();
    return std::find(node_types.begin(), node_types.end(), type) !=
           node_types.end();
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

    if (node.type == NodeType::DataOutput) {
        AddDataOutputTruth(report, node);
    }

    if (node.type == NodeType::DataConvert) {
        AddDataConvertTruth(report, node, context);
    }

    if (node.type == NodeType::DeployToNodeEditorNode) {
        AddDeployToNodeEditorTruth(report, node);
    }

    if (node.type == NodeType::StandardScaler ||
        node.type == NodeType::MinMaxScaler ||
        node.type == NodeType::RobustScaler) {
        AddPreprocessingScalerTruth(report, node);
    }

    if (node.type == NodeType::LabelEncoder ||
        node.type == NodeType::OrdinalEncoder ||
        node.type == NodeType::TargetEncoder) {
        AddEncoderTruth(report, node);
    }

    if (node.type == NodeType::OutlierDetector) {
        AddOutlierTruth(report, node);
    }

    if (node.type == NodeType::TFIDFVectorizer ||
        node.type == NodeType::CountVectorizer) {
        AddVectorizerTruth(
            report,
            node,
            node.type == NodeType::CountVectorizer);
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

    if (node.type == NodeType::RegressionMetricsNode ||
        node.type == NodeType::ClassificationMetricsNode ||
        node.type == NodeType::ConfusionMatrixNode ||
        node.type == NodeType::ROCCurveNode ||
        node.type == NodeType::PRCurveNode) {
        AddMetricTruth(report, node);
    }

    if (node.type == NodeType::Dense ||
        node.type == NodeType::TimeDistributed ||
        node.type == NodeType::Dropout ||
        node.type == NodeType::BatchNorm ||
        node.type == NodeType::LayerNorm ||
        IsActivationNode(node.type) ||
        IsShapeOpNode(node.type)) {
        AddCoreLayerTruth(report, node);
    }

    if (node.type == NodeType::Embedding) {
        AddEmbeddingTruth(report, node);
        if (const auto* placement_fact =
                FindBackendPlacementFact(node, context)) {
            report.properties.push_back(
                BuildBackendPlacementTruth(*placement_fact));
        }
    }

    if (node.type == NodeType::MultiHeadAttention ||
        node.type == NodeType::TransformerEncoder ||
        node.type == NodeType::TransformerDecoder ||
        node.type == NodeType::PositionalEncoding) {
        AddTransformerTruth(report, node);
        if (const auto* placement_fact =
                FindBackendPlacementFact(node, context)) {
            report.properties.push_back(
                BuildBackendPlacementTruth(*placement_fact));
        }
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
                    "pin_memory=true is a GPU host-to-device transfer "
                    "optimization request, not a data materialization "
                    "accelerator. Current batchers report the effective "
                    "runtime transfer mode after compile/training.";
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

    if (node.type == NodeType::Adam ||
        node.type == NodeType::SGD ||
        node.type == NodeType::AdamW ||
        node.type == NodeType::RMSprop ||
        node.type == NodeType::Adagrad ||
        node.type == NodeType::NAdam) {
        AddOptimizerTruth(report, node);
    }

    if (node.type == NodeType::LSTM || node.type == NodeType::GRU) {
        auto hidden_size = ResolveIntProperty(
            node,
            "Hidden size",
            "hidden_size",
            "256",
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

        auto num_layers = ResolveIntProperty(
            node,
            "Recurrent layers",
            "num_layers",
            "1",
            TruthOwner::Runtime,
            true,
            false,
            "The Engine constructs this many stacked recurrent layers.");
        if (ParsePositiveInt(&num_layers.effective_value) <= 0) {
            num_layers.statuses.clear();
            AddStatus(num_layers, TruthStatus::Missing);
            num_layers.message = "num_layers must be >= 1.";
        }
        report.properties.push_back(std::move(num_layers));

        auto bidirectional = ResolveBoolProperty(
            node,
            "Bidirectional",
            "bidirectional",
            false,
            TruthOwner::Runtime,
            true,
            false,
            node.type == NodeType::LSTM
                ? "Engine LSTM training currently supports only one direction."
                : "GRU uses explicit forward and reverse branches when enabled.");
        if (node.type == NodeType::LSTM &&
            bidirectional.effective_value == "true") {
            bidirectional.statuses.clear();
            AddStatus(bidirectional, TruthStatus::Unsupported);
            bidirectional.message =
                "Reverse-direction LSTM backward gradients are not implemented; "
                "Engine training fails closed for bidirectional=true.";
        } else if (node.type == NodeType::GRU &&
                   bidirectional.effective_value == "true") {
            AddStatus(bidirectional, TruthStatus::RuntimeOnly);
            bidirectional.message =
                "Engine training uses the split forward/reverse GRU "
                "path; current placement is native CPU.";
        }
        report.properties.push_back(std::move(bidirectional));

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

        auto dropout = ResolveFloatProperty(
            node,
            "Recurrent dropout",
            "dropout",
            "0.0",
            TruthOwner::Runtime,
            true,
            false,
            "Engine recurrent modules currently require dropout=0.0; use an "
            "explicit Dropout node for executable regularization.");
        double parsed_dropout = 0.0;
        if (ParseDoubleValue(dropout.effective_value, parsed_dropout) &&
            parsed_dropout != 0.0) {
            dropout.statuses.clear();
            AddStatus(dropout, TruthStatus::Unsupported);
            dropout.message =
                "This value is not wired into the Engine recurrent module. "
                "Set dropout=0.0 and add an explicit Dropout node.";
        }
        report.properties.push_back(std::move(dropout));

        PropertyTruth hidden_output;
        hidden_output.label = "Hidden state output";
        hidden_output.canonical_key = "hidden_output";
        hidden_output.source_key = "legacy pin";
        hidden_output.effective_value = "not routed";
        hidden_output.owner = TruthOwner::Runtime;
        hidden_output.quick_editable = false;
        bool hidden_connected = false;
        if (context.links) {
            for (const auto& pin : node.outputs) {
                if (pin.name != "Hidden") continue;
                hidden_connected = std::any_of(
                    context.links->begin(), context.links->end(),
                    [&](const NodeLink& link) {
                        return link.from_node == node.id &&
                               link.from_pin == pin.id;
                    });
                if (hidden_connected) break;
            }
        }
        AddStatus(hidden_output,
                  hidden_connected ? TruthStatus::Unsupported
                                   : TruthStatus::CompilerOnly);
        hidden_output.message = hidden_connected
            ? "The connected legacy Hidden pin has no separate Engine output. "
              "Disconnect it and use Output."
            : "Compatibility pin retained for saved graphs; Engine "
              "SequentialModel exposes only Output.";
        report.properties.push_back(std::move(hidden_output));

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

    if (node.type == NodeType::NERSequenceBuilder) {
        AddNERSequenceBuilderTruth(report, node);
    }

    if (node.type == NodeType::TokenVocabulary ||
        node.type == NodeType::POSVocabulary ||
        node.type == NodeType::NERTagVocabulary) {
        AddSequenceVocabularyTruth(report, node);
    }

    if (node.type == NodeType::SequenceTagOutput) {
        AddSequenceTagOutputTruth(report, node);
    }

    if (node.type == NodeType::ExportCSV ||
        node.type == NodeType::ExportParquet ||
        node.type == NodeType::ExportJSON) {
        AddFixedExportTruth(report, node);
    }

    if (node.type == NodeType::DataProfiler) {
        AddDataProfilerTruth(report, node);
    }

    if (node.type == NodeType::TreeModelPredictor) {
        AddTreeModelPredictorTruth(report, node);
    }

    if (IsLossNode(node.type)) {
        AddLossTruth(report, node);
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
    } else if (canonical_key == "file_path") {
        node.parameters.erase("path");
    } else if (canonical_key == "file_type") {
        node.parameters.erase("format");
    } else if (canonical_key == "token_column") {
        node.parameters.erase("tokens_column");
        node.parameters.erase("token_sequence_column");
    } else if (canonical_key == "pos_column") {
        node.parameters.erase("pos_sequence_column");
    } else if (canonical_key == "tag_column") {
        node.parameters.erase("tags_column");
        node.parameters.erase("tag_sequence_column");
    } else if (canonical_key == "sentence_id_column") {
        node.parameters.erase("sequence_id_column");
    } else if (canonical_key == "target_column") {
        node.parameters.erase("target_ids_column");
        node.parameters.erase("decoder_target_column");
    } else if (canonical_key == "min_frequency") {
        node.parameters.erase("min_freq");
    } else if (canonical_key == "max_size") {
        node.parameters.erase("max_vocab_size");
    }
}

} // namespace gui::properties_truth
