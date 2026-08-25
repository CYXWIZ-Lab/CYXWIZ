#pragma once

#include "node_editor.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <map>
#include <string>

namespace gui::detail {

inline constexpr int kLegacyDataBoundaryVersion = 1;
inline constexpr int kCurrentDataBoundaryVersion = 2;
inline constexpr int kLegacyDataValidatorContractVersion = 1;
inline constexpr int kCurrentDataValidatorContractVersion = 2;
inline constexpr int kCurrentEvaluationTableContractVersion = 2;
inline constexpr int kCurrentClassicalTreeTableContractVersion = 2;

inline int ReadSerializedDataBoundaryVersion(const nlohmann::json& graph_json) {
    if (!graph_json.contains("data_boundary_version")) {
        return kLegacyDataBoundaryVersion;
    }
    const auto& value = graph_json["data_boundary_version"];
    if (!value.is_number_integer()) {
        return kLegacyDataBoundaryVersion;
    }
    const int version = value.get<int>();
    return version == kCurrentDataBoundaryVersion
        ? kCurrentDataBoundaryVersion
        : kLegacyDataBoundaryVersion;
}

inline bool PreserveLegacyDataBoundaryPins(const nlohmann::json& graph_json) {
    return ReadSerializedDataBoundaryVersion(graph_json) <
           kCurrentDataBoundaryVersion;
}

inline bool PreserveLegacyDataValidatorOutputs(
    const nlohmann::json& graph_json) {
    if (!graph_json.contains("data_validator_contract_version")) {
        return true;
    }
    const auto& value = graph_json["data_validator_contract_version"];
    return !value.is_number_integer() ||
           value.get<int>() != kCurrentDataValidatorContractVersion;
}

inline bool ResolveLegacyDataValidatorOutputPinIndex(
    const nlohmann::json& link_json,
    int& resolved_index) {
    resolved_index = 0;
    if (!link_json.contains("from_pin_index") ||
        !link_json["from_pin_index"].is_number_integer()) {
        return false;
    }

    // Legacy output indices 0 and 1 were the unimplemented Valid/Invalid
    // promises. Only index 2 represented the real Issues dataset.
    return link_json["from_pin_index"].get<int>() == 2;
}

inline bool PreserveLegacyEvaluationTableInputs(
    const nlohmann::json& graph_json) {
    if (!graph_json.contains("evaluation_table_contract_version")) {
        return true;
    }
    const auto& value = graph_json["evaluation_table_contract_version"];
    return !value.is_number_integer() ||
           value.get<int>() != kCurrentEvaluationTableContractVersion;
}

inline bool IsLegacySplitInputEvaluationNode(NodeType type) {
    return type == NodeType::ConfusionMatrixNode ||
           type == NodeType::ROCCurveNode ||
           type == NodeType::PRCurveNode;
}

inline bool PreserveLegacyClassicalTreeTablePins(
    const nlohmann::json& graph_json) {
    if (!graph_json.contains("classical_tree_table_contract_version")) {
        return true;
    }
    const auto& value = graph_json["classical_tree_table_contract_version"];
    return !value.is_number_integer() ||
           value.get<int>() != kCurrentClassicalTreeTableContractVersion;
}

inline bool IsLegacySplitInputTreeTrainer(NodeType type) {
    return type == NodeType::DecisionTreeClassifier ||
           type == NodeType::RandomForestClassifier ||
           type == NodeType::GradientBoostingClassifier;
}

inline bool ResolveLegacyClassicalTreeOutputPinIndex(
    const nlohmann::json& link_json,
    int& resolved_index) {
    resolved_index = 0;
    if (!link_json.contains("from_pin_index") ||
        !link_json["from_pin_index"].is_number_integer()) {
        return false;
    }

    // Legacy index 0 was a fictional in-memory Model output. The real
    // materialized prediction table was exposed at index 1.
    return link_json["from_pin_index"].get<int>() == 1;
}

inline bool HasImportGuardParam(const std::map<std::string, std::string>& params,
                                const char* key) {
    return params.find(key) != params.end();
}

inline bool ResolveSerializedPinIndex(const nlohmann::json& link_json,
                                      const char* index_key,
                                      std::size_t pin_count,
                                      int& resolved_index) {
    resolved_index = 0;
    if (!link_json.contains(index_key)) {
        return pin_count > 0;
    }

    const auto& value = link_json[index_key];
    if (!value.is_number_integer()) {
        return false;
    }

    const int index = value.get<int>();
    if (index < 0 || static_cast<std::size_t>(index) >= pin_count) {
        return false;
    }

    resolved_index = index;
    return true;
}

inline bool IsDenseEncodedSequencePlaceholderName(
    NodeType node_type,
    const std::string& node_name,
    std::string& matched_marker) {
    if (node_type != NodeType::Dense) {
        return false;
    }

    const char* placeholder_names[] = {
        "NERSequenceBuilder",
        "Sentence Sequences",
        "TokenVocabulary",
        "Word Vocabulary",
        "POSVocabulary",
        "POS Vocabulary",
        "NERTagVocabulary",
        "NER Tag Vocabulary",
        "SequencePadding",
        "Pad Tokens + Tags",
        "FeatureConcat",
        "Concat Word + POS",
        "TimeDistributedDense",
        "Token Classifier",
        "TokenCrossEntropyLoss",
        "Token CrossEntropy",
        "SequenceTagMetrics",
        "NER Metrics",
        "SequenceTagOutput",
        "NER Output"
    };

    for (const char* name : placeholder_names) {
        if (node_name == name) {
            matched_marker = name;
            return true;
        }
    }
    return false;
}

inline bool HasDenseEncodedSequenceTargetDesignParameter(
    NodeType node_type,
    const std::map<std::string, std::string>& params,
    std::string& matched_marker) {
    if (node_type != NodeType::Dense) {
        return false;
    }

    const char* keys[] = {
        "bio_scheme",
        "token_column",
        "tag_column",
        "pos_column",
        "pad_token",
        "unk_token",
        "pad_tag",
        "outside_tag",
        "create_attention_mask",
        "token_pad_value",
        "pos_pad_value",
        "tag_pad_value",
        "decode_scheme",
        "tag_vocab_file",
        "ignore_index",
        "from_logits"
    };

    for (const char* key : keys) {
        if (HasImportGuardParam(params, key)) {
            matched_marker = key;
            return true;
        }
    }

    if (HasImportGuardParam(params, "vocab_file") &&
        (HasImportGuardParam(params, "max_vocab_size") ||
         HasImportGuardParam(params, "min_freq"))) {
        matched_marker = "vocab_file";
        return true;
    }

    if (HasImportGuardParam(params, "axis") && params.size() > 1) {
        matched_marker = "axis";
        return true;
    }

    return false;
}

inline bool IsDenseEncodedSequencePlaceholder(const MLNode& node,
                                              std::string& matched_marker) {
    return IsDenseEncodedSequencePlaceholderName(
               node.type,
               node.name,
               matched_marker) ||
           HasDenseEncodedSequenceTargetDesignParameter(
               node.type,
               node.parameters,
               matched_marker);
}

inline bool IsDenseEncodedSequencePlaceholder(NodeType node_type,
                                              const std::string& node_name) {
    std::string matched_marker;
    return IsDenseEncodedSequencePlaceholderName(node_type,
                                                 node_name,
                                                 matched_marker);
}

} // namespace gui::detail
