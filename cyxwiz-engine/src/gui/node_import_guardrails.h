#pragma once

#include "node_editor.h"

#include <map>
#include <string>

namespace gui::detail {

inline bool HasImportGuardParam(const std::map<std::string, std::string>& params,
                                const char* key) {
    return params.find(key) != params.end();
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
