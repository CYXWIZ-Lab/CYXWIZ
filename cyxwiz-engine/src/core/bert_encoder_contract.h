#pragma once

#include "compiled_graph_plan.h"

#include <cyxwiz/tensor.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace cyxwiz {

enum class BertEncoderTask {
    None,
    SequenceClassification,
    TokenClassification,
};

enum class BertEncoderInputKind {
    Unknown,
    TokenIds,
    EncodedTensor,
};

inline const char* BertEncoderTaskName(BertEncoderTask task) {
    switch (task) {
        case BertEncoderTask::SequenceClassification:
            return "sequence_classification";
        case BertEncoderTask::TokenClassification:
            return "token_classification";
        case BertEncoderTask::None:
        default:
            return "none";
    }
}

inline const char* BertEncoderInputKindName(BertEncoderInputKind kind) {
    switch (kind) {
        case BertEncoderInputKind::TokenIds:
            return "token_ids";
        case BertEncoderInputKind::EncodedTensor:
            return "encoded_tensor";
        case BertEncoderInputKind::Unknown:
        default:
            return "unknown";
    }
}

struct BertEncoderGraphContract {
    bool detected = false;
    bool supported = false;
    BertEncoderTask task = BertEncoderTask::None;
    BertEncoderInputKind input_kind = BertEncoderInputKind::Unknown;

    bool has_encoder_stack = false;
    bool has_token_embedding = false;
    bool has_positional_encoding = false;
    bool has_attention_mask = false;
    bool requires_token_type_ids = false;
    bool has_cls_extraction = false;
    bool has_sequence_pooling = false;
    bool has_flattened_sequence_head = false;

    std::vector<int> encoder_node_ids;
    std::vector<int> head_node_ids;
    std::vector<int> output_node_ids;
    std::vector<std::string> blockers;
    std::string output_contract;
};

struct BertEncoderTokenInputContract {
    bool compatible = false;
    std::string error;
    size_t batch_size = 0;
    size_t sequence_length = 0;
    bool has_attention_mask = false;
    bool has_token_type_ids = false;
};

struct BertEncoderRuntimeOutputContract {
    bool compatible = false;
    std::string error;
    std::vector<size_t> output_shape;
    BertEncoderTask task = BertEncoderTask::None;
    size_t batch_size = 0;
    size_t sequence_length = 0;
    size_t class_count = 0;
};

namespace bert_encoder_contract_detail {

inline std::string LowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                   });
    return value;
}

inline bool ContainsInsensitive(const std::string& value,
                                const std::string& needle) {
    return LowerAscii(value).find(LowerAscii(needle)) != std::string::npos;
}

inline bool HasNonEmptyParam(
    const std::map<std::string, std::string>& params,
    const char* key) {
    const auto it = params.find(key);
    return it != params.end() && !it->second.empty();
}

inline bool ParamIsTruthy(
    const std::map<std::string, std::string>& params,
    const char* key) {
    const auto it = params.find(key);
    if (it == params.end()) {
        return false;
    }
    const std::string value = LowerAscii(it->second);
    return value == "true" || value == "1" || value == "yes" ||
           value == "on" || value == "required";
}

inline bool DeclaresBertStyle(const CompiledGraphNode& node) {
    if (ContainsInsensitive(node.name, "bert")) {
        return true;
    }

    const char* family_keys[] = {
        "model_family",
        "encoder_family",
        "text_model_family",
    };
    for (const char* key : family_keys) {
        const auto it = node.parameters.find(key);
        if (it == node.parameters.end()) {
            continue;
        }
        const std::string value = LowerAscii(it->second);
        if (value == "bert" || value == "bert_encoder" ||
            value == "bert-style" || value == "bert_style") {
            return true;
        }
    }

    return ParamIsTruthy(node.parameters, "bert_style") ||
           ParamIsTruthy(node.parameters, "bert_encoder");
}

inline bool DeclaresAttentionMask(const CompiledGraphNode& node) {
    return ParamIsTruthy(node.parameters, "attention_mask") ||
           ParamIsTruthy(node.parameters, "create_attention_mask") ||
           HasNonEmptyParam(node.parameters, "attention_mask_column");
}

inline bool DeclaresTokenTypeIds(const CompiledGraphNode& node) {
    if (ParamIsTruthy(node.parameters, "token_type_ids") ||
        ParamIsTruthy(node.parameters, "segment_ids") ||
        ParamIsTruthy(node.parameters, "use_token_type_ids") ||
        ParamIsTruthy(node.parameters, "use_segment_ids")) {
        return true;
    }

    const char* required_keys[] = {
        "token_type_column",
        "token_type_ids_column",
        "segment_column",
        "segment_ids_column",
        "segment_vocab_file",
        "token_type_vocab_file",
        "type_vocab_size",
    };
    for (const char* key : required_keys) {
        if (HasNonEmptyParam(node.parameters, key)) {
            return true;
        }
    }
    return false;
}

inline std::vector<int> ParseIntList(std::string value) {
    value.erase(std::remove(value.begin(), value.end(), '['), value.end());
    value.erase(std::remove(value.begin(), value.end(), ']'), value.end());
    value.erase(std::remove(value.begin(), value.end(), ' '), value.end());

    std::vector<int> result;
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (token.empty()) {
            continue;
        }
        try {
            result.push_back(std::stoi(token));
        } catch (...) {
            return {};
        }
    }
    return result;
}

inline int ParseIntParam(
    const std::map<std::string, std::string>& params,
    const char* key,
    int fallback) {
    const auto it = params.find(key);
    if (it == params.end()) {
        return fallback;
    }
    try {
        return std::stoi(it->second);
    } catch (...) {
        return fallback;
    }
}

inline bool SelectsClsToken(const CompiledGraphNode& node) {
    if (node.type != gui::NodeType::TensorIndexSelect) {
        return false;
    }
    if (ParseIntParam(node.parameters, "dim", 0) != 0) {
        return false;
    }
    const auto it = node.parameters.find("indices");
    if (it == node.parameters.end()) {
        return false;
    }
    const std::vector<int> indices = ParseIntList(it->second);
    return indices.size() == 1 && indices[0] == 0;
}

inline bool PoolsSequence(const CompiledGraphNode& node) {
    if (node.type != gui::NodeType::TensorMean &&
        node.type != gui::NodeType::TensorMax) {
        return false;
    }
    return ParseIntParam(node.parameters, "dim", -1) == 0;
}

inline void AddBlocker(BertEncoderGraphContract& contract,
                       std::string blocker) {
    for (const auto& existing : contract.blockers) {
        if (existing == blocker) {
            return;
        }
    }
    contract.blockers.push_back(std::move(blocker));
}

inline bool IsSequenceIdDType(DataType dtype) {
    return dtype == DataType::Int64 || dtype == DataType::Int32 ||
           dtype == DataType::Float32;
}

inline bool IsNumericMaskDType(DataType dtype) {
    return dtype == DataType::Int64 || dtype == DataType::Int32 ||
           dtype == DataType::Float32 || dtype == DataType::Float64 ||
           dtype == DataType::UInt8;
}

inline bool TensorIsPresent(const Tensor* tensor) {
    return tensor != nullptr && !tensor->Shape().empty() &&
           tensor->NumElements() > 0;
}

inline std::string JoinIssues(const std::vector<std::string>& issues) {
    std::ostringstream stream;
    for (size_t i = 0; i < issues.size(); ++i) {
        if (i > 0) {
            stream << "; ";
        }
        stream << issues[i];
    }
    return stream.str();
}

} // namespace bert_encoder_contract_detail

inline BertEncoderGraphContract AnalyzeBertEncoderGraphContract(
    const CompiledGraphPlan& plan) {
    using namespace bert_encoder_contract_detail;

    BertEncoderGraphContract contract;
    if (!plan.available) {
        return contract;
    }

    bool declared_bert = false;
    bool has_dense_head = false;
    bool has_time_distributed_head = false;
    bool has_sequence_tag_output = false;

    for (const auto& node : plan.nodes) {
        declared_bert = declared_bert || DeclaresBertStyle(node);
        contract.has_attention_mask =
            contract.has_attention_mask || DeclaresAttentionMask(node);
        contract.requires_token_type_ids =
            contract.requires_token_type_ids || DeclaresTokenTypeIds(node);

        switch (node.type) {
            case gui::NodeType::Embedding:
                contract.has_token_embedding = true;
                break;
            case gui::NodeType::PositionalEncoding:
                contract.has_positional_encoding = true;
                break;
            case gui::NodeType::TransformerEncoder:
                contract.has_encoder_stack = true;
                contract.encoder_node_ids.push_back(node.node_id);
                break;
            case gui::NodeType::TensorIndexSelect:
                contract.has_cls_extraction =
                    contract.has_cls_extraction || SelectsClsToken(node);
                break;
            case gui::NodeType::TensorMean:
            case gui::NodeType::TensorMax:
                contract.has_sequence_pooling =
                    contract.has_sequence_pooling || PoolsSequence(node);
                break;
            case gui::NodeType::Flatten:
                contract.has_flattened_sequence_head = true;
                break;
            case gui::NodeType::Dense:
                has_dense_head = true;
                contract.head_node_ids.push_back(node.node_id);
                break;
            case gui::NodeType::TimeDistributed:
                has_time_distributed_head = true;
                contract.head_node_ids.push_back(node.node_id);
                break;
            case gui::NodeType::Output:
                contract.output_node_ids.push_back(node.node_id);
                break;
            case gui::NodeType::SequenceTagOutput:
                has_sequence_tag_output = true;
                contract.output_node_ids.push_back(node.node_id);
                break;
            default:
                break;
        }
    }

    contract.detected =
        declared_bert ||
        (contract.has_encoder_stack &&
         (has_time_distributed_head || has_sequence_tag_output ||
          contract.has_cls_extraction || contract.has_sequence_pooling));
    if (!contract.detected) {
        return contract;
    }

    contract.input_kind = contract.has_token_embedding
        ? BertEncoderInputKind::TokenIds
        : BertEncoderInputKind::EncodedTensor;

    if (!contract.has_encoder_stack) {
        AddBlocker(contract,
                   "BERT-style encoder graphs require a TransformerEncoder stack");
    }
    if (contract.requires_token_type_ids) {
        AddBlocker(contract,
                   "BERT token_type/segment IDs are not supported yet; remove token_type_ids or segment_ids from this graph");
    }
    if (contract.input_kind == BertEncoderInputKind::TokenIds &&
        !contract.has_positional_encoding) {
        AddBlocker(contract,
                   "BERT-style token-id encoder graphs require positional encoding before TransformerEncoder");
    }

    if (has_sequence_tag_output || has_time_distributed_head) {
        contract.task = BertEncoderTask::TokenClassification;
        contract.output_contract = "Float32[batch,seq,classes]";
        if (!has_time_distributed_head) {
            AddBlocker(contract,
                       "BERT-style token classification requires a TimeDistributed classifier head");
        }
    } else {
        contract.task = BertEncoderTask::SequenceClassification;
        contract.output_contract = "Float32[batch,classes]";
        if (!has_dense_head) {
            AddBlocker(contract,
                       "BERT-style sequence classification requires a Dense classifier head");
        }
        if (!contract.has_cls_extraction && !contract.has_sequence_pooling) {
            if (contract.has_flattened_sequence_head) {
                AddBlocker(contract,
                           "BERT-style sequence classification requires explicit CLS extraction or sequence pooling; a flattened full-sequence head is not a pooling contract");
            } else {
                AddBlocker(contract,
                           "BERT-style sequence classification requires explicit CLS extraction or sequence pooling before Dense");
            }
        }
    }

    contract.supported = contract.blockers.empty();
    return contract;
}

inline BertEncoderTokenInputContract ValidateBertEncoderTokenInput(
    const Tensor& token_ids,
    const Tensor* attention_mask = nullptr,
    const Tensor* token_type_ids = nullptr,
    size_t max_sequence_length = 0) {
    using namespace bert_encoder_contract_detail;

    BertEncoderTokenInputContract contract;
    const auto& token_shape = token_ids.Shape();
    if (token_shape.size() == 2) {
        contract.batch_size = token_shape[0];
        contract.sequence_length = token_shape[1];
    }
    contract.has_attention_mask = TensorIsPresent(attention_mask);
    contract.has_token_type_ids = TensorIsPresent(token_type_ids);

    std::vector<std::string> issues;
    if (token_shape.size() != 2) {
        issues.push_back("BERT token ids must have shape [batch, seq]");
    } else {
        if (contract.batch_size == 0 || contract.sequence_length == 0) {
            issues.push_back("BERT token ids must not be empty");
        }
        if (max_sequence_length > 0 &&
            contract.sequence_length > max_sequence_length) {
            issues.push_back(
                "BERT token ids exceed max_sequence_length");
        }
    }
    if (!IsSequenceIdDType(token_ids.GetDataType())) {
        issues.push_back(
            "BERT token ids must be Int64, Int32, or Float32");
    }

    if (contract.has_attention_mask) {
        if (attention_mask->Shape() != token_shape) {
            issues.push_back(
                "BERT attention_mask shape must match token ids");
        }
        if (!IsNumericMaskDType(attention_mask->GetDataType())) {
            issues.push_back("BERT attention_mask must be numeric");
        }
    }

    if (contract.has_token_type_ids) {
        issues.push_back(
            "BERT token_type/segment IDs are not supported yet");
    }

    contract.compatible = issues.empty();
    contract.error = contract.compatible ? std::string{} : JoinIssues(issues);
    return contract;
}

inline BertEncoderRuntimeOutputContract ValidateBertEncoderRuntimeOutput(
    const Tensor& logits,
    BertEncoderTask task,
    size_t expected_batch_size = 0,
    size_t expected_sequence_length = 0,
    size_t expected_class_count = 0) {
    using namespace bert_encoder_contract_detail;

    BertEncoderRuntimeOutputContract contract;
    contract.task = task;
    contract.output_shape = logits.Shape();

    if (task == BertEncoderTask::SequenceClassification &&
        contract.output_shape.size() == 2) {
        contract.batch_size = contract.output_shape[0];
        contract.class_count = contract.output_shape[1];
    } else if (task == BertEncoderTask::TokenClassification &&
               contract.output_shape.size() == 3) {
        contract.batch_size = contract.output_shape[0];
        contract.sequence_length = contract.output_shape[1];
        contract.class_count = contract.output_shape[2];
    }

    std::vector<std::string> issues;
    if (task == BertEncoderTask::None) {
        issues.push_back("BERT encoder task must be specified");
    }
    if (logits.GetDataType() != DataType::Float32) {
        issues.push_back("BERT encoder output must be Float32 logits");
    }

    if (task == BertEncoderTask::SequenceClassification) {
        if (contract.output_shape.size() != 2) {
            issues.push_back(
                "BERT sequence classifier output must have rank 2 [batch, classes]");
        }
    } else if (task == BertEncoderTask::TokenClassification) {
        if (contract.output_shape.size() != 3) {
            issues.push_back(
                "BERT token classifier output must have rank 3 [batch, seq, classes]");
        }
    }

    if ((task == BertEncoderTask::SequenceClassification &&
         contract.output_shape.size() == 2) ||
        (task == BertEncoderTask::TokenClassification &&
         contract.output_shape.size() == 3)) {
        if (contract.batch_size == 0) {
            issues.push_back("BERT encoder output batch size must be greater than 0");
        }
        if (task == BertEncoderTask::TokenClassification &&
            contract.sequence_length == 0) {
            issues.push_back(
                "BERT token classifier output sequence length must be greater than 0");
        }
        if (contract.class_count < 2) {
            issues.push_back(
                "BERT encoder output must contain at least two classes");
        }
        if (expected_batch_size > 0 &&
            contract.batch_size != expected_batch_size) {
            issues.push_back(
                "BERT encoder output batch size does not match input batch");
        }
        if (task == BertEncoderTask::TokenClassification &&
            expected_sequence_length > 0 &&
            contract.sequence_length != expected_sequence_length) {
            issues.push_back(
                "BERT token classifier output sequence length does not match input sequence");
        }
        if (expected_class_count > 0 &&
            contract.class_count != expected_class_count) {
            issues.push_back(
                "BERT encoder output class count does not match metadata");
        }
    }

    contract.compatible = issues.empty();
    contract.error = contract.compatible ? std::string{} : JoinIssues(issues);
    return contract;
}

} // namespace cyxwiz
