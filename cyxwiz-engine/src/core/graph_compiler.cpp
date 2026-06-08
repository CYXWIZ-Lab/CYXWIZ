#include "graph_compiler.h"
#include "data_registry.h"
#include "arrow_dataset.h"
#include "parquet_backed_dataset.h"
#include "graph_topology_utils.h"
#include "worker_defaults.h"
#include "node_metadata_registry.h"
#include "pipeline_runtime_capabilities.h"
#include "../gui/loaders/data_loader.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace cyxwiz {

namespace {
// Local helpers for issue collection. Kept in an anonymous namespace so
// they don't pollute the cyxwiz public surface.

void AddIssue(TrainingConfiguration& config, IssueLevel level,
              const std::string& message, int node_id = -1,
              const std::string& node_name = "") {
    ValidationIssue issue;
    issue.level = level;
    issue.message = message;
    issue.node_id = node_id;
    issue.node_name = node_name;
    config.issues.push_back(std::move(issue));
}

bool IsDatasetSourceType(gui::NodeType type) {
    return type == gui::NodeType::DataInput ||
           type == gui::NodeType::DatasetInput;
}

bool IsLossNodeType(gui::NodeType type) {
    return type == gui::NodeType::MSELoss ||
           type == gui::NodeType::CrossEntropyLoss ||
           type == gui::NodeType::BCELoss ||
           type == gui::NodeType::BCEWithLogits ||
           type == gui::NodeType::L1Loss ||
           type == gui::NodeType::SmoothL1Loss ||
           type == gui::NodeType::HuberLoss ||
           type == gui::NodeType::NLLLoss;
}

bool IsSupportedOptimizerNodeType(gui::NodeType type) {
    return type == gui::NodeType::SGD ||
           type == gui::NodeType::Adam ||
           type == gui::NodeType::AdamW ||
           type == gui::NodeType::RMSprop ||
           type == gui::NodeType::Adagrad ||
           type == gui::NodeType::NAdam;
}

const char* IssueLevelLabel(IssueLevel level) {
    switch (level) {
        case IssueLevel::Error:   return "ERROR";
        case IssueLevel::Warning: return "WARN";
        case IssueLevel::Info:    return "INFO";
    }
    return "?";
}

const char* ImplementationStatusLabel(NodeImplementationStatus status) {
    switch (status) {
        case NodeImplementationStatus::Implemented:
            return "implemented";
        case NodeImplementationStatus::Template:
            return "template/deferred";
        case NodeImplementationStatus::Deprecated:
            return "deprecated";
        case NodeImplementationStatus::External:
            return "external";
    }
    return "unknown";
}

const char* PreprocessingDomainLabel(PreprocessingDomain domain) {
    switch (domain) {
        case PreprocessingDomain::Tabular:    return "tabular";
        case PreprocessingDomain::Image:      return "image";
        case PreprocessingDomain::Audio:      return "audio";
        case PreprocessingDomain::Text:       return "text";
        case PreprocessingDomain::TimeSeries: return "time-series";
        case PreprocessingDomain::General:    return "general";
    }
    return "unknown";
}

bool IsGraphRuntimeMergeOp(gui::NodeType type) {
    return type == gui::NodeType::Add ||
           type == gui::NodeType::Multiply ||
           type == gui::NodeType::Average ||
           type == gui::NodeType::Concatenate;
}

bool IsGraphRuntimeBinaryMaskOp(gui::NodeType type) {
    return type == gui::NodeType::TensorCompare ||
           type == gui::NodeType::TensorLogicalMask;
}

bool IsGraphRuntimeLinalgOp(gui::NodeType type) {
    return type == gui::NodeType::TensorDot;
}

bool IsGraphRuntimeFanInOp(gui::NodeType type) {
    return IsGraphRuntimeMergeOp(type) ||
           IsGraphRuntimeBinaryMaskOp(type) ||
           IsGraphRuntimeLinalgOp(type);
}

bool HasParam(const std::map<std::string, std::string>& params,
              const char* key) {
    return params.find(key) != params.end();
}

bool LooksLikeDenseEncodedTargetDesignNode(const gui::MLNode& node,
                                           std::string& matched_key) {
    if (node.type != gui::NodeType::Dense) {
        return false;
    }

    const auto& params = node.parameters;
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
        if (HasParam(params, key)) {
            matched_key = key;
            return true;
        }
    }

    if (HasParam(params, "vocab_file") &&
        (HasParam(params, "max_vocab_size") ||
         HasParam(params, "min_freq"))) {
        matched_key = "vocab_file";
        return true;
    }

    if (HasParam(params, "axis") && params.size() > 1) {
        matched_key = "axis";
        return true;
    }

    return false;
}

bool ParamIsEnabled(const std::map<std::string, std::string>& params,
                    const char* key) {
    auto it = params.find(key);
    if (it == params.end()) {
        return false;
    }

    std::string value = it->second;
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return value == "true" || value == "1" || value == "yes" ||
           value == "on";
}

bool LooksLikeGenerativeTrainingSketch(const gui::MLNode& node,
                                       std::string& matched_key) {
    if (node.type == gui::NodeType::TransformerDecoder) {
        matched_key = "TransformerDecoder";
        return true;
    }

    const auto& params = node.parameters;
    const char* enabled_keys[] = {
        "causal",
        "causal_mask",
        "autoregressive",
        "decoder_only",
        "generation",
        "generate",
        "language_model",
        "causal_lm",
        "teacher_forcing"
    };

    for (const char* key : enabled_keys) {
        if (ParamIsEnabled(params, key)) {
            matched_key = key;
            return true;
        }
    }

    const char* design_keys[] = {
        "shifted_targets",
        "shift_targets",
        "target_shift",
        "next_token_target",
        "prompt_column",
        "completion_column",
        "decoder_target_column"
    };

    for (const char* key : design_keys) {
        if (HasParam(params, key)) {
            matched_key = key;
            return true;
        }
    }

    return false;
}

bool LooksLikeImportedFineTuningSketch(const gui::MLNode& node,
                                       std::string& matched_key) {
    if (node.type == gui::NodeType::DNNModelLoad ||
        node.type == gui::NodeType::PretrainedYOLO ||
        node.type == gui::NodeType::PretrainedMobileNet ||
        node.type == gui::NodeType::PretrainedOpenPose ||
        node.type == gui::NodeType::PretrainedFaceNet) {
        matched_key = "pretrained model node";
        return true;
    }

    const auto& params = node.parameters;
    const char* enabled_keys[] = {
        "pretrained",
        "fine_tune",
        "fine_tuning",
        "transfer_learning",
        "enable_transfer_learning",
        "load_optimizer_state",
        "allow_shape_mismatch",
        "freeze",
        "freeze_layers",
        "unfreeze_layers",
        "lora"
    };

    for (const char* key : enabled_keys) {
        if (ParamIsEnabled(params, key)) {
            matched_key = key;
            return true;
        }
    }

    const char* design_keys[] = {
        "pretrained_model",
        "pretrained_model_path",
        "pretrained_checkpoint",
        "checkpoint_path",
        "weights_path",
        "model_path",
        "base_model",
        "adapter_path",
        "import_format",
        "onnx_path",
        "safetensors_path",
        "gguf_path",
        "freeze_mode",
        "unfreeze_last_n",
        "frozen_layers",
        "trainable_layers"
    };

    for (const char* key : design_keys) {
        if (HasParam(params, key)) {
            matched_key = key;
            return true;
        }
    }

    return false;
}

bool IsTensorInputPin(const gui::MLNode& node, int pin_id) {
    for (const auto& pin : node.inputs) {
        if (pin.id == pin_id) {
            return pin.is_input && pin.type == gui::PinType::Tensor;
        }
    }
    return false;
}

int CountConnectedSelectedTensorInputs(
    const gui::MLNode& node,
    const std::vector<gui::NodeLink>& links,
    const std::unordered_set<int>& selected_node_ids) {
    int connected = 0;
    for (const auto& link : links) {
        if (link.to_node != node.id) {
            continue;
        }
        if (selected_node_ids.count(link.from_node) == 0 ||
            selected_node_ids.count(link.to_node) == 0) {
            continue;
        }
        if (IsTensorInputPin(node, link.to_pin)) {
            ++connected;
        }
    }
    return connected;
}

// Build the legacy single-string error_message from the issues list so
// existing callers that only look at error_message keep working.
std::string JoinErrorMessages(const std::vector<ValidationIssue>& issues) {
    std::ostringstream out;
    bool first = true;
    for (const auto& i : issues) {
        if (i.level != IssueLevel::Error) continue;
        if (!first) out << "; ";
        first = false;
        if (!i.node_name.empty()) {
            out << "[" << i.node_name << "] ";
        }
        out << i.message;
    }
    return out.str();
}

void ApplyTextInputShape(TrainingConfiguration& config) {
    if (config.preprocessing_domain != PreprocessingDomain::Text) {
        return;
    }

    int max_length = config.text_preprocessing.max_length;
    const bool graph_overrides_length =
        config.text_preprocessing.has_tokenizer_node ||
        config.text_preprocessing.has_padding_node;

    if (!graph_overrides_length && !config.dataset_name.empty()) {
        if (const auto* entry =
                DataRegistry::Instance().GetTextDatasetEntry(config.dataset_name)) {
            max_length = entry->max_length;
        }
    }

    if (max_length <= 0) {
        max_length = 1;
    }

    config.input_shape = {static_cast<size_t>(max_length)};
    config.input_size = static_cast<size_t>(max_length);
}

bool ParseBoolParam(const std::map<std::string, std::string>& params,
                    const std::string& key,
                    bool fallback = false) {
    auto it = params.find(key);
    if (it == params.end()) {
        return fallback;
    }
    return it->second == "true" || it->second == "1";
}

size_t ParseSizeParam(const std::map<std::string, std::string>& params,
                      const std::string& key,
                      size_t fallback) {
    auto it = params.find(key);
    if (it == params.end()) {
        return fallback;
    }
    try {
        const int parsed = std::stoi(it->second);
        return parsed > 0 ? static_cast<size_t>(parsed) : fallback;
    } catch (...) {
        return fallback;
    }
}

bool ParseFloatParam(const std::map<std::string, std::string>& params,
                     const std::string& key,
                     float fallback,
                     float& value,
                     std::string& error) {
    auto it = params.find(key);
    if (it == params.end() || it->second.empty()) {
        value = fallback;
        return true;
    }

    try {
        size_t parsed_chars = 0;
        value = std::stof(it->second, &parsed_chars);
        if (parsed_chars != it->second.size()) {
            error = key + " must be a float";
            return false;
        }
        if (!std::isfinite(value)) {
            error = key + " must be finite";
            return false;
        }
        return true;
    } catch (...) {
        error = key + " must be a float";
        return false;
    }
}

bool ValidateTensorScalarMathParams(gui::NodeType type,
                                    const std::map<std::string, std::string>& params,
                                    std::string& error) {
    if (type == gui::NodeType::TensorPow) {
        float exponent = 2.0f;
        return ParseFloatParam(params, "exponent", 2.0f, exponent, error);
    }

    if (type == gui::NodeType::TensorClip) {
        float min_val = 0.0f;
        float max_val = 1.0f;
        if (!ParseFloatParam(params, "min", 0.0f, min_val, error) ||
            !ParseFloatParam(params, "max", 1.0f, max_val, error)) {
            return false;
        }
        if (min_val > max_val) {
            error = "TensorClip min must be <= max";
            return false;
        }
    }

    return true;
}

const gui::MLNode* FindFirstLossNodeInSet(
    const std::vector<gui::MLNode>& nodes,
    const std::unordered_set<int>& node_ids) {

    for (const auto& node : nodes) {
        if (IsLossNodeType(node.type) &&
            node_ids.count(node.id) > 0) {
            return &node;
        }
    }
    return nullptr;
}

const gui::MLNode* FindFirstOptimizerNodeInSet(
    const std::vector<gui::MLNode>& nodes,
    const std::unordered_set<int>& node_ids) {

    for (const auto& node : nodes) {
        if (IsSupportedOptimizerNodeType(node.type) &&
            node_ids.count(node.id) > 0) {
            return &node;
        }
    }
    return nullptr;
}

void ValidateSingleDatasetReachableLossNode(
    const std::vector<gui::MLNode>& nodes,
    const std::unordered_set<int>& dataset_reachable,
    TrainingConfiguration& config) {

    if (dataset_reachable.empty()) {
        return;
    }

    std::vector<const gui::MLNode*> reachable_losses;
    for (const auto& node : nodes) {
        if (IsLossNodeType(node.type) &&
            dataset_reachable.count(node.id) > 0) {
            reachable_losses.push_back(&node);
        }
    }

    if (reachable_losses.size() <= 1) {
        return;
    }

    std::ostringstream names;
    for (size_t i = 0; i < reachable_losses.size(); ++i) {
        if (i > 0) {
            names << ", ";
        }
        names << "'" << reachable_losses[i]->name << "'";
    }

    AddIssue(
        config,
        IssueLevel::Error,
        "Current Studio training supports exactly one dataset-reachable "
        "loss node. Found " + std::to_string(reachable_losses.size()) +
        " losses (" + names.str() + "). Multi-head/multi-task or "
        "alternating-loss training needs a first-class loss aggregation "
        "contract before it can compile truthfully.");
}

void ValidateSingleDatasetSourceForSelectedLoss(
    const std::vector<gui::MLNode>& nodes,
    const gui::MLNode* selected_loss,
    const std::vector<gui::NodeLink>& links,
    TrainingConfiguration& config) {

    if (!selected_loss) {
        return;
    }

    const auto loss_ancestors = CollectAncestorNodeIds(selected_loss->id, links);
    std::vector<const gui::MLNode*> selected_sources;
    for (const auto& node : nodes) {
        if (IsDatasetSourceType(node.type) &&
            loss_ancestors.count(node.id) > 0) {
            selected_sources.push_back(&node);
        }
    }

    if (selected_sources.size() <= 1) {
        return;
    }

    std::ostringstream names;
    for (size_t i = 0; i < selected_sources.size(); ++i) {
        if (i > 0) {
            names << ", ";
        }
        names << "'" << selected_sources[i]->name << "'";
    }

    AddIssue(
        config,
        IssueLevel::Error,
        "Current Studio training supports exactly one dataset source on "
        "the selected loss path. Found " +
        std::to_string(selected_sources.size()) + " sources (" +
        names.str() + ") feeding loss '" + selected_loss->name +
        "'. Multi-input/shared-weight training needs a typed named-batch "
        "contract before it can compile truthfully.",
        selected_loss->id,
        selected_loss->name);
}

bool ContainsWhenFiltered(
    const std::unordered_set<int>& node_ids,
    int node_id) {

    return node_ids.empty() || node_ids.count(node_id) > 0;
}

bool HasConnectedInputAfterFirst(const gui::MLNode& node,
                                 const std::vector<gui::NodeLink>& links) {
    if (node.inputs.size() <= 1) {
        return false;
    }
    std::unordered_set<int> connected_input_pins;
    connected_input_pins.reserve(links.size());
    for (const auto& link : links) {
        connected_input_pins.insert(link.to_pin);
    }
    for (size_t i = 1; i < node.inputs.size(); ++i) {
        if (connected_input_pins.count(node.inputs[i].id) > 0) {
            return true;
        }
    }
    return false;
}

bool CheckedMultiplySize(size_t lhs, size_t rhs, size_t& out) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    out = lhs * rhs;
    return true;
}

size_t ShapeElementCount(const std::vector<size_t>& shape) {
    size_t count = 1;
    for (size_t dim : shape) {
        size_t next = 0;
        if (!CheckedMultiplySize(count, dim, next)) {
            throw std::overflow_error("shape element count overflow");
        }
        count = next;
    }
    return count;
}

bool ResolveReshapeTargetShape(const std::map<std::string, std::string>& params,
                               const std::vector<size_t>& input_shape,
                               std::vector<size_t>& target_shape,
                               std::string& error) {
    auto it = params.find("shape");
    if (it == params.end() || it->second.empty()) {
        error = "Reshape/View requires a non-empty shape parameter";
        return false;
    }
    if (input_shape.empty()) {
        error = "Reshape/View requires a known input shape";
        return false;
    }

    std::string shape_str = it->second;
    shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), '['), shape_str.end());
    shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ']'), shape_str.end());
    shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ' '), shape_str.end());

    std::vector<int64_t> dims;
    std::stringstream ss(shape_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            error = "Reshape/View shape contains an empty dimension";
            return false;
        }
        try {
            dims.push_back(std::stoll(token));
        } catch (...) {
            error = "Reshape/View shape contains a non-integer dimension";
            return false;
        }
    }

    if (dims.empty()) {
        error = "Reshape/View requires at least one target dimension";
        return false;
    }

    int infer_index = -1;
    size_t known_product = 1;
    for (size_t i = 0; i < dims.size(); ++i) {
        const int64_t dim = dims[i];
        if (dim == -1) {
            if (infer_index != -1) {
                error = "Reshape/View allows at most one -1 inferred dimension";
                return false;
            }
            infer_index = static_cast<int>(i);
            continue;
        }
        if (dim <= 0) {
            error = "Reshape/View dimensions must be positive, except one optional -1";
            return false;
        }
        size_t next = 0;
        if (!CheckedMultiplySize(known_product, static_cast<size_t>(dim), next)) {
            error = "Reshape/View target shape product overflows";
            return false;
        }
        known_product = next;
    }

    size_t input_elements = 0;
    try {
        input_elements = ShapeElementCount(input_shape);
    } catch (...) {
        error = "Reshape/View input shape product overflows";
        return false;
    }

    target_shape.clear();
    target_shape.reserve(dims.size());
    if (infer_index >= 0) {
        if (known_product == 0 || input_elements % known_product != 0) {
            error = "Reshape/View inferred dimension is not divisible by input elements";
            return false;
        }
        dims[static_cast<size_t>(infer_index)] =
            static_cast<int64_t>(input_elements / known_product);
    } else if (known_product != input_elements) {
        error = "Reshape/View target shape must preserve element count";
        return false;
    }

    for (int64_t dim : dims) {
        target_shape.push_back(static_cast<size_t>(dim));
    }
    return true;
}

bool ParseIntParam(const std::map<std::string, std::string>& params,
                   const std::string& key,
                   int fallback,
                   int& out,
                   std::string& error) {
    auto it = params.find(key);
    if (it == params.end() || it->second.empty()) {
        out = fallback;
        return true;
    }
    try {
        out = std::stoi(it->second);
        return true;
    } catch (...) {
        error = key + " must be an integer";
        return false;
    }
}

bool ParsePositiveShapeParam(const std::map<std::string, std::string>& params,
                             const std::string& key,
                             const std::string& op_name,
                             std::vector<size_t>& dims,
                             std::string& error) {
    auto it = params.find(key);
    if (it == params.end() || it->second.empty()) {
        error = op_name + " requires a non-empty " + key + " parameter";
        return false;
    }

    std::string shape_str = it->second;
    shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), '['), shape_str.end());
    shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ']'), shape_str.end());
    shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ' '), shape_str.end());

    dims.clear();
    std::stringstream ss(shape_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            error = op_name + " shape contains an empty dimension";
            return false;
        }
        try {
            const int64_t dim = std::stoll(token);
            if (dim <= 0) {
                error = op_name + " shape dimensions must be positive";
                return false;
            }
            dims.push_back(static_cast<size_t>(dim));
        } catch (...) {
            error = op_name + " shape contains a non-integer dimension";
            return false;
        }
    }

    if (dims.empty()) {
        error = op_name + " requires at least one target dimension";
        return false;
    }
    return true;
}

bool ParseIndexListParam(const std::map<std::string, std::string>& params,
                         const std::string& key,
                         const std::string& op_name,
                         std::vector<int>& indices,
                         std::string& error) {
    auto it = params.find(key);
    if (it == params.end() || it->second.empty()) {
        error = op_name + " requires a non-empty " + key + " parameter";
        return false;
    }

    std::string list_str = it->second;
    list_str.erase(std::remove(list_str.begin(), list_str.end(), '['), list_str.end());
    list_str.erase(std::remove(list_str.begin(), list_str.end(), ']'), list_str.end());
    list_str.erase(std::remove(list_str.begin(), list_str.end(), ' '), list_str.end());

    indices.clear();
    std::stringstream ss(list_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            error = op_name + " indices contains an empty value";
            return false;
        }
        try {
            indices.push_back(std::stoi(token));
        } catch (...) {
            error = op_name + " indices contains a non-integer value";
            return false;
        }
    }

    if (indices.empty()) {
        error = op_name + " requires at least one index";
        return false;
    }
    return true;
}

bool NormalizeCompilerDim(int dim,
                          int rank,
                          bool allow_end,
                          const std::string& op_name,
                          int& normalized,
                          std::string& error) {
    const int min_dim = allow_end ? -rank - 1 : -rank;
    const int max_dim = allow_end ? rank : rank - 1;
    if (dim < min_dim || dim > max_dim) {
        error = op_name + " dimension is out of range";
        return false;
    }
    normalized = dim < 0 ? dim + rank + (allow_end ? 1 : 0) : dim;
    return true;
}

bool ResolveSqueezeTargetShape(const std::map<std::string, std::string>& params,
                               const std::vector<size_t>& input_shape,
                               std::vector<size_t>& target_shape,
                               std::string& error) {
    if (input_shape.empty()) {
        error = "Squeeze requires a known input shape";
        return false;
    }

    int dim = -1;
    if (!ParseIntParam(params, "dim", -1, dim, error)) {
        error = "Squeeze " + error;
        return false;
    }

    target_shape.clear();
    if (dim == -1) {
        for (size_t size : input_shape) {
            if (size != 1) {
                target_shape.push_back(size);
            }
        }
    } else {
        int axis = 0;
        if (!NormalizeCompilerDim(dim,
                                  static_cast<int>(input_shape.size()),
                                  false,
                                  "Squeeze",
                                  axis,
                                  error)) {
            return false;
        }
        if (input_shape[static_cast<size_t>(axis)] != 1) {
            error = "Squeeze selected dimension must have size 1";
            return false;
        }
        for (size_t i = 0; i < input_shape.size(); ++i) {
            if (i != static_cast<size_t>(axis)) {
                target_shape.push_back(input_shape[i]);
            }
        }
    }

    if (target_shape.empty()) {
        target_shape.push_back(1);
    }
    return true;
}

bool ResolveUnsqueezeTargetShape(const std::map<std::string, std::string>& params,
                                 const std::vector<size_t>& input_shape,
                                 std::vector<size_t>& target_shape,
                                 std::string& error) {
    if (input_shape.empty()) {
        error = "Unsqueeze requires a known input shape";
        return false;
    }

    int dim = 0;
    if (!ParseIntParam(params, "dim", 0, dim, error)) {
        error = "Unsqueeze " + error;
        return false;
    }

    int axis = 0;
    if (!NormalizeCompilerDim(dim,
                              static_cast<int>(input_shape.size()),
                              true,
                              "Unsqueeze",
                              axis,
                              error)) {
        return false;
    }

    target_shape.clear();
    target_shape.reserve(input_shape.size() + 1);
    for (int i = 0; i < static_cast<int>(input_shape.size()); ++i) {
        if (i == axis) {
            target_shape.push_back(1);
        }
        target_shape.push_back(input_shape[static_cast<size_t>(i)]);
    }
    if (axis == static_cast<int>(input_shape.size())) {
        target_shape.push_back(1);
    }
    return true;
}

bool ResolvePermuteTargetShape(const std::map<std::string, std::string>& params,
                               const std::vector<size_t>& input_shape,
                               std::vector<size_t>& target_shape,
                               std::vector<int>& normalized_dims,
                               std::string& error) {
    if (input_shape.empty()) {
        error = "Permute requires a known input shape";
        return false;
    }

    auto it = params.find("dims");
    if (it == params.end() || it->second.empty()) {
        error = "Permute requires a non-empty dims parameter";
        return false;
    }

    std::string dims_str = it->second;
    dims_str.erase(std::remove(dims_str.begin(), dims_str.end(), '['), dims_str.end());
    dims_str.erase(std::remove(dims_str.begin(), dims_str.end(), ']'), dims_str.end());
    dims_str.erase(std::remove(dims_str.begin(), dims_str.end(), ' '), dims_str.end());

    const int rank = static_cast<int>(input_shape.size());
    std::stringstream ss(dims_str);
    std::string token;
    normalized_dims.clear();
    std::vector<bool> seen(input_shape.size(), false);

    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            error = "Permute dims contains an empty dimension";
            return false;
        }

        int raw_dim = 0;
        try {
            raw_dim = std::stoi(token);
        } catch (...) {
            error = "Permute dims contains a non-integer dimension";
            return false;
        }

        int axis = 0;
        if (!NormalizeCompilerDim(raw_dim, rank, false, "Permute", axis, error)) {
            return false;
        }
        if (seen[static_cast<size_t>(axis)]) {
            error = "Permute dims must not contain duplicates";
            return false;
        }
        seen[static_cast<size_t>(axis)] = true;
        normalized_dims.push_back(axis);
    }

    if (normalized_dims.size() != input_shape.size()) {
        error = "Permute dims must match input rank";
        return false;
    }

    target_shape.clear();
    target_shape.reserve(normalized_dims.size());
    for (int axis : normalized_dims) {
        target_shape.push_back(input_shape[static_cast<size_t>(axis)]);
    }
    return true;
}

bool ResolveTensorBroadcastTargetShape(gui::NodeType type,
                                       const std::map<std::string, std::string>& params,
                                       const std::vector<size_t>& input_shape,
                                       std::vector<size_t>& target_shape,
                                       std::string& error) {
    const std::string op_name = type == gui::NodeType::TensorBroadcastTo
        ? "TensorBroadcastTo"
        : "TensorExpand";
    if (input_shape.empty()) {
        error = op_name + " requires a known input shape";
        return false;
    }
    if (!ParsePositiveShapeParam(params, "shape", op_name, target_shape, error)) {
        return false;
    }
    if (target_shape.size() < input_shape.size()) {
        error = op_name + " target sample rank must be >= input sample rank";
        return false;
    }

    const size_t sample_pad = target_shape.size() - input_shape.size();
    for (size_t axis = 0; axis < target_shape.size(); ++axis) {
        const size_t input_dim = axis < sample_pad ? 1 : input_shape[axis - sample_pad];
        const size_t target_dim = target_shape[axis];
        if (input_dim != 1 && input_dim != target_dim) {
            error = op_name + " target shape is not broadcast-compatible";
            return false;
        }
    }
    return true;
}

bool ResolveTensorIndexSelectTargetShape(const std::map<std::string, std::string>& params,
                                         const std::vector<size_t>& input_shape,
                                         std::vector<size_t>& target_shape,
                                         std::string& error) {
    if (input_shape.empty()) {
        error = "TensorIndexSelect requires a known input shape";
        return false;
    }

    int dim = 0;
    if (!ParseIntParam(params, "dim", 0, dim, error)) {
        error = "TensorIndexSelect " + error;
        return false;
    }

    int normalized_dim = 0;
    if (!NormalizeCompilerDim(dim,
                              static_cast<int>(input_shape.size()),
                              false,
                              "TensorIndexSelect",
                              normalized_dim,
                              error)) {
        return false;
    }

    std::vector<int> indices;
    if (!ParseIndexListParam(params, "indices", "TensorIndexSelect", indices, error)) {
        return false;
    }

    const int dim_size = static_cast<int>(input_shape[static_cast<size_t>(normalized_dim)]);
    for (int index : indices) {
        int normalized = index;
        if (normalized < 0) {
            normalized += dim_size;
        }
        if (normalized < 0 || normalized >= dim_size) {
            error = "TensorIndexSelect index is out of range";
            return false;
        }
    }

    target_shape = input_shape;
    target_shape[static_cast<size_t>(normalized_dim)] = indices.size();
    return true;
}

bool ResolveShapeOpTargetShape(gui::NodeType type,
                               const std::map<std::string, std::string>& params,
                               const std::vector<size_t>& input_shape,
                               std::vector<size_t>& target_shape,
                               std::string& error) {
    switch (type) {
        case gui::NodeType::Reshape:
        case gui::NodeType::View:
            return ResolveReshapeTargetShape(params, input_shape, target_shape, error);
        case gui::NodeType::Squeeze:
            return ResolveSqueezeTargetShape(params, input_shape, target_shape, error);
        case gui::NodeType::Unsqueeze:
            return ResolveUnsqueezeTargetShape(params, input_shape, target_shape, error);
        case gui::NodeType::Permute: {
            std::vector<int> ignored_dims;
            return ResolvePermuteTargetShape(params,
                                             input_shape,
                                             target_shape,
                                             ignored_dims,
                                             error);
        }
        case gui::NodeType::TensorBroadcastTo:
        case gui::NodeType::TensorExpand:
            return ResolveTensorBroadcastTargetShape(type,
                                                     params,
                                                     input_shape,
                                                     target_shape,
                                                     error);
        case gui::NodeType::TensorIndexSelect:
            return ResolveTensorIndexSelectTargetShape(params,
                                                       input_shape,
                                                       target_shape,
                                                       error);
        default:
            error = "Unsupported shape operation";
            return false;
    }
}

bool ResolveReductionTargetShape(gui::NodeType type,
                                 const std::map<std::string, std::string>& params,
                                 const std::vector<size_t>& input_shape,
                                 std::vector<size_t>& target_shape,
                                 std::string& error) {
    if (type != gui::NodeType::TensorSum &&
        type != gui::NodeType::TensorMean &&
        type != gui::NodeType::TensorMax &&
        type != gui::NodeType::TensorMin &&
        type != gui::NodeType::TensorProd &&
        type != gui::NodeType::TensorVar &&
        type != gui::NodeType::TensorStd) {
        error = "Unsupported tensor reduction";
        return false;
    }
    if (input_shape.empty()) {
        error = "Tensor reduction requires a known input shape";
        return false;
    }

    int dim = -1;
    if (!ParseIntParam(params, "dim", -1, dim, error)) {
        error = "Tensor reduction " + error;
        return false;
    }
    const bool keepdim = ParseBoolParam(params, "keepdim", false);

    target_shape.clear();
    if (dim == -1) {
        if (keepdim) {
            target_shape.assign(input_shape.size(), 1);
        } else {
            target_shape = {1};
        }
        return true;
    }

    if (dim < 0 || dim >= static_cast<int>(input_shape.size())) {
        error = "Tensor reduction dim is out of range";
        return false;
    }

    target_shape = input_shape;
    if (keepdim) {
        target_shape[static_cast<size_t>(dim)] = 1;
    } else {
        target_shape.erase(target_shape.begin() + dim);
        if (target_shape.empty()) {
            target_shape.push_back(1);
        }
    }
    return true;
}

bool ValidateTensorMaskParams(gui::NodeType type,
                              const std::map<std::string, std::string>& params,
                              std::string& error) {
    if (type == gui::NodeType::TensorCompare) {
        auto op_it = params.find("op");
        const std::string op = op_it != params.end() ? op_it->second : ">";
        if (op != ">" && op != ">=" && op != "<" &&
            op != "<=" && op != "==" && op != "!=") {
            error = "TensorCompare op must be one of >, >=, <, <=, ==, !=";
            return false;
        }

        float scalar = 0.0f;
        if (!ParseFloatParam(params, "scalar", 0.0f, scalar, error)) {
            error = "TensorCompare " + error;
            return false;
        }
        return true;
    }

    if (type == gui::NodeType::TensorLogicalMask) {
        auto op_it = params.find("op");
        const std::string op = op_it != params.end() ? op_it->second : "not";
        if (op != "not") {
            error = "TensorLogicalMask currently supports only op=not in the sequential runtime";
            return false;
        }
        return true;
    }

    error = "Unsupported tensor mask operation";
    return false;
}

void ValidateTrainingPathImplementationStatus(
    const std::vector<gui::MLNode>& nodes,
    const std::unordered_set<int>& training_path_ids,
    TrainingConfiguration& config) {

    if (training_path_ids.empty()) {
        return;
    }

    auto& registry = NodeMetadataRegistry::Instance();
    if (!registry.IsInitialized()) {
        registry.Initialize();
    }

    for (const auto& node : nodes) {
        if (training_path_ids.count(node.id) == 0) {
            continue;
        }

        std::string target_design_key;
        if (LooksLikeDenseEncodedTargetDesignNode(node, target_design_key)) {
            std::ostringstream msg;
            msg << "Node '" << node.name
                << "' is encoded as Dense but contains target-design parameter '"
                << target_design_key
                << "'. This graph needs first-class sequence/NER nodes and "
                   "cannot be compiled as a Dense layer.";
            AddIssue(config, IssueLevel::Error, msg.str(), node.id, node.name);
            continue;
        }

        std::string generative_key;
        if (LooksLikeGenerativeTrainingSketch(node, generative_key)) {
            std::ostringstream msg;
            msg << "Node '" << node.name
                << "' sketches decoder/generative training via '"
                << generative_key
                << "', but Studio training does not have a trainable "
                   "TransformerDecoder or causal language-model objective. "
                   "This path needs shifted-token targets, causal masks, "
                   "token-level loss, and generation packaging before it can "
                   "compile truthfully.";
            AddIssue(config, IssueLevel::Error, msg.str(), node.id, node.name);
            continue;
        }

        std::string fine_tuning_key;
        if (LooksLikeImportedFineTuningSketch(node, fine_tuning_key)) {
            std::ostringstream msg;
            msg << "Node '" << node.name
                << "' sketches imported/pretrained fine-tuning via '"
                << fine_tuning_key
                << "', but Studio training does not have a model-import to "
                   "training-graph contract. This path needs parameter "
                   "mapping, shape validation, freeze/unfreeze ownership, "
                   "optimizer-state compatibility, and tokenizer/preprocessor "
                   "packaging before it can compile truthfully.";
            AddIssue(config, IssueLevel::Error, msg.str(), node.id, node.name);
            continue;
        }

        const auto training_support =
            ResolvePipelineTrainingBackendSupport(node.type);
        if (!training_support.compile_supported &&
            training_support.mode ==
                PipelineTrainingBackendSupportMode::UnsupportedSequentialModelLayer) {
            std::ostringstream msg;
            msg << "Node '" << node.name << "' is "
                << training_support.reason;
            AddIssue(config, IssueLevel::Error, msg.str(), node.id, node.name);
            continue;
        }

        const NodeMetadata* metadata = registry.GetMetadata(node.type);
        if (!metadata) {
            continue;
        }
        if (metadata->status == NodeImplementationStatus::Implemented) {
            continue;
        }
        if (IsGraphRuntimeFanInOp(node.type)) {
            continue;
        }

        std::ostringstream msg;
        msg << "Node '" << node.name << "' is "
            << ImplementationStatusLabel(metadata->status)
            << " and cannot run in the training path";
        if (!metadata->brief_description.empty()) {
            msg << " (" << metadata->brief_description << ")";
        }
        AddIssue(config, IssueLevel::Error, msg.str(), node.id, node.name);
    }
}

void ValidateUnsupportedTrainingControlNodes(
    const std::vector<gui::MLNode>& nodes,
    TrainingConfiguration& config) {

    for (const auto& node : nodes) {
        const auto training_support =
            ResolvePipelineTrainingBackendSupport(node.type);
        if (training_support.compile_supported ||
            training_support.mode !=
                PipelineTrainingBackendSupportMode::UnsupportedTrainingControl) {
            continue;
        }

        std::ostringstream msg;
        msg << "Node '" << node.name << "' is " << training_support.reason;
        AddIssue(config, IssueLevel::Error, msg.str(), node.id, node.name);
    }
}

void CollectGraphRuntimeOpNodeIds(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    const std::vector<int>& sorted_node_ids,
    const std::unordered_set<int>& training_path_ids,
    TrainingConfiguration& config) {
    if (training_path_ids.empty()) {
        return;
    }

    std::unordered_map<int, const gui::MLNode*> by_id;
    by_id.reserve(nodes.size());
    for (const auto& node : nodes) {
        by_id[node.id] = &node;
    }

    for (int node_id : sorted_node_ids) {
        if (training_path_ids.count(node_id) == 0) {
            continue;
        }

        auto it = by_id.find(node_id);
        if (it == by_id.end()) {
            continue;
        }
        const gui::MLNode& node = *it->second;
        if (!IsGraphRuntimeFanInOp(node.type)) {
            continue;
        }

        const int connected_inputs =
            CountConnectedSelectedTensorInputs(node, links, training_path_ids);
        if (connected_inputs < 2) {
            std::ostringstream msg;
            msg << "Graph runtime node '" << node.name
                << "' requires at least two connected tensor inputs";
            AddIssue(config, IssueLevel::Error, msg.str(), node.id, node.name);
            continue;
        }
        if ((IsGraphRuntimeBinaryMaskOp(node.type) ||
             IsGraphRuntimeLinalgOp(node.type)) &&
            connected_inputs != 2) {
            std::ostringstream msg;
            msg << "Graph runtime node '" << node.name
                << "' requires exactly two connected tensor inputs";
            AddIssue(config, IssueLevel::Error, msg.str(), node.id, node.name);
            continue;
        }
        if (IsGraphRuntimeBinaryMaskOp(node.type)) {
            auto op_it = node.parameters.find("op");
            const std::string op = op_it != node.parameters.end()
                ? op_it->second
                : (node.type == gui::NodeType::TensorCompare ? ">" : "and");
            if (node.type == gui::NodeType::TensorCompare &&
                op != ">" && op != ">=" && op != "<" &&
                op != "<=" && op != "==" && op != "!=") {
                AddIssue(config, IssueLevel::Error,
                         "TensorCompare graph op must use one of >, >=, <, <=, ==, !=",
                         node.id, node.name);
                continue;
            }
            if (node.type == gui::NodeType::TensorLogicalMask &&
                op != "and" && op != "or") {
                AddIssue(config, IssueLevel::Error,
                         "TensorLogicalMask graph op supports only op=and or op=or with two inputs",
                         node.id, node.name);
                continue;
            }
        }

        // Runtime tensor-shape checks remain in GraphExecutableModel, where
        // concrete shapes and broadcast/concat dimensions are known. The
        // compiler only records deliberately-enabled graph ops here.
        config.graph_op_node_ids.push_back(node.id);
    }
}
} // anonymous namespace

TrainingConfiguration GraphCompiler::Compile(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    bool allow_unloaded_data)
{
    TrainingConfiguration config;

    // === Structural validation ===
    // Collect all structural errors at once (the old ValidateGraph stopped
    // at the first one). is_valid is determined at the end of Compile by
    // checking whether config.issues contains any Error-level entries.
    const gui::MLNode* dataset_node = FindDatasetInputNode(nodes, links);
    const std::unordered_set<int> dataset_reachable =
        dataset_node
            ? CollectReachableNodeIds(dataset_node->id, links)
            : std::unordered_set<int>{};
    const gui::MLNode* loss_node = FindFirstLossNodeInSet(nodes, dataset_reachable);
    if (!loss_node) {
        loss_node = FindLossNode(nodes);
    }
    const gui::MLNode* optimizer_node = FindOptimizerNode(nodes);

    if (nodes.empty()) {
        AddIssue(config, IssueLevel::Error, "Graph is empty - add nodes to create a model");
    } else {
        if (!dataset_node) {
            AddIssue(config, IssueLevel::Error,
                     "Graph must have a DataInput or DatasetInput node");
        }
        if (!loss_node) {
            AddIssue(config, IssueLevel::Error,
                     "Graph must have a loss function (MSELoss, CrossEntropyLoss, etc.)");
        }
        if (!optimizer_node) {
            AddIssue(config, IssueLevel::Error,
                     "Graph must have an optimizer (SGD, Adam, AdamW, RMSprop, Adagrad, or NAdam)");
        }
        bool has_model_layer = false;
        for (const auto& node : nodes) {
            if (IsModelLayer(node.type)) { has_model_layer = true; break; }
        }
        if (!has_model_layer) {
            AddIssue(config, IssueLevel::Error,
                     "Graph must have at least one model layer (Dense, Conv2D, etc.)");
        }
        if (HasCycle(nodes, links)) {
            AddIssue(config, IssueLevel::Error,
                     "Graph contains a cycle - remove circular connections");
        }

        // Pin-connectivity checks. The runtime currently ignores pin
        // topology (reads dataset_name from registry instead), so a
        // graph with a disconnected Loss.Targets pin will train
        // anyway. These checks make the canvas the source of truth at
        // compile time so users can't ship a visually-broken graph.
        ValidateRequiredInputsConnected(nodes, links, config);
        ValidateRequiredOutputsConnected(nodes, links, config);
        ValidateLossTargetsReachLabels(nodes, links, config);
        ValidateLossPredictionsReachModel(nodes, links, config);
        ValidateOptimizerReachesLoss(nodes, links, config);
        ValidateSingleDatasetReachableLossNode(nodes, dataset_reachable, config);
        ValidateSingleDatasetSourceForSelectedLoss(nodes, loss_node, links, config);
        ValidateUnsupportedTrainingControlNodes(nodes, config);
    }

    // Extract dataset configuration
    if (dataset_node) {
        // dataset_name parameter is what DataInputDialog::Apply writes when
        // a dataset is loaded. The legacy "dataset" key was used by the
        // older DatasetInput dialog and is now obsolete — fall back to it
        // only if dataset_name is empty, so older project files still load.
        if (dataset_node->parameters.count("dataset_name") &&
            !dataset_node->parameters.at("dataset_name").empty()) {
            config.dataset_name = dataset_node->parameters.at("dataset_name");
        } else if (dataset_node->parameters.count("dataset")) {
            config.dataset_name = dataset_node->parameters.at("dataset");
        }

        // === New error checks tied to the dataset node ===
        //
        // Registry is the source of truth for "is data loaded", NOT the
        // `data_loaded` param hint on the node. The hint is cached dialog
        // state that can go stale when an async Apply completes after the
        // dialog has already closed — PollAsyncLoadResult only runs while
        // the dialog is visible, so the provisional "false" that Apply
        // sets before launching the worker gets stuck. This mirrors the
        // DataInputDialog constructor's design, which probes the registry
        // directly instead of trusting the param hint (see CLAUDE.md under
        // "Async load + UX").
        //
        // The decision matrix is:
        //   in_registry=yes → data is loaded, proceed (ignore stale hint)
        //   in_registry=no, dataset_name empty → never-applied, block
        //   in_registry=no, dataset_name set, hint=true → registry wiped,
        //     "re-apply" error
        //   in_registry=no, dataset_name set, hint=false → never-applied
        //     (or async still running at compile time), block
        const std::string loaded_param = dataset_node->parameters.count("data_loaded")
            ? dataset_node->parameters.at("data_loaded")
            : std::string("false");

        // Registry probe via the loader factory. Each loader checks its
        // own registry map (tabular: Arrow+Parquet, image/audio/text:
        // their entries). GetByRegisteredDataset returns non-null iff
        // ANY loader claims the name — same semantics as the 5-way OR
        // this replaces.
        const bool in_registry = !config.dataset_name.empty() &&
            loaders::GetByRegisteredDataset(config.dataset_name) != nullptr;

        if (in_registry) {
            // Data IS loaded, regardless of the hint. Log when we had to
            // override a stale hint so it's traceable at training time.
            if (loaded_param != "true") {
                spdlog::info("GraphCompiler: dataset '{}' found in registry "
                             "despite data_loaded='{}' hint (stale from "
                             "async Apply after dialog close) - proceeding",
                             config.dataset_name, loaded_param);
            }
        } else if (allow_unloaded_data) {
            spdlog::info("GraphCompiler: dataset '{}' not in registry, but "
                         "compile is running in deployment mode - skipping "
                         "data_loaded validation",
                         config.dataset_name);
        } else if (config.dataset_name.empty()) {
            if (loaded_param == "true") {
                AddIssue(config, IssueLevel::Error,
                         "Data marked loaded but no dataset_name parameter set",
                         dataset_node->id, dataset_node->name);
            } else {
                AddIssue(config, IssueLevel::Error,
                         "Data is not loaded - open the node and click Apply",
                         dataset_node->id, dataset_node->name);
            }
        } else {
            // dataset_name is set but registry has nothing under it.
            if (loaded_param == "true") {
                AddIssue(config, IssueLevel::Error,
                         "Dataset '" + config.dataset_name +
                         "' is marked loaded but missing from registry - "
                         "re-apply the DataInput node",
                         dataset_node->id, dataset_node->name);
            } else {
                AddIssue(config, IssueLevel::Error,
                         "Data is not loaded - open the node and click Apply",
                         dataset_node->id, dataset_node->name);
            }
        }

        // Check 3: label column. Required for tabular supervised training,
        // but image / audio / text datasets get labels from folder
        // structure or an explicit dialog column, not from a flag on
        // the DataInput node — don't emit a misleading warning for them.
        // Each loader reports LabelsFromStructure() — Tabular returns
        // false; Image/Audio/Text return true.
        auto cat_it_lbl = dataset_node->parameters.find("file_category");
        const std::string cat_str = (cat_it_lbl != dataset_node->parameters.end())
            ? cat_it_lbl->second : std::string();
        bool labels_from_structure = false;
        if (auto* cat_loader = loaders::GetByCategory(
                loaders::FileCategoryFromString(cat_str))) {
            labels_from_structure = cat_loader->LabelsFromStructure();
        }

        const std::string label_col = dataset_node->parameters.count("label_column")
            ? dataset_node->parameters.at("label_column")
            : std::string();
        if (label_col.empty() && !labels_from_structure) {
            AddIssue(config, IssueLevel::Warning,
                     "No label column selected - training will use the last "
                     "column as label by default",
                     dataset_node->id, dataset_node->name);
        }

        // Extract input shape from dataset node
        if (dataset_node->parameters.count("shape")) {
            // Parse shape string like "[28, 28, 1]"
            std::string shape_str = dataset_node->parameters.at("shape");
            // Simple parsing - remove brackets and split by comma
            shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), '['), shape_str.end());
            shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ']'), shape_str.end());
            shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ' '), shape_str.end());

            size_t pos = 0;
            while ((pos = shape_str.find(',')) != std::string::npos) {
                config.input_shape.push_back(std::stoul(shape_str.substr(0, pos)));
                shape_str.erase(0, pos + 1);
            }
            if (!shape_str.empty()) {
                config.input_shape.push_back(std::stoul(shape_str));
            }
        }

        // Calculate flattened input size
        config.input_size = 1;
        for (size_t dim : config.input_shape) {
            config.input_size *= dim;
        }
    }

    // Extract split ratios from the DataSplit node on the selected data path.
    if (const gui::MLNode* split_node = FindFirstReachableNodeOfType(
            nodes, dataset_reachable, gui::NodeType::DataSplit)) {
        config.has_data_split = true;
        try {
            if (split_node->parameters.count("train_ratio"))
                config.train_ratio = std::stof(split_node->parameters.at("train_ratio"));
            if (split_node->parameters.count("val_ratio"))
                config.val_ratio = std::stof(split_node->parameters.at("val_ratio"));
            if (split_node->parameters.count("test_ratio"))
                config.test_ratio = std::stof(split_node->parameters.at("test_ratio"));
            if (split_node->parameters.count("seed"))
                config.split_seed = std::stoi(split_node->parameters.at("seed"));
        } catch (const std::exception& e) {
            spdlog::warn("GraphCompiler: DataSplit param parse error ({}) - using defaults", e.what());
        }
    }
    if (config.has_data_split) {
        spdlog::info("GraphCompiler: DataSplit node found - train={:.2f}, val={:.2f}, test={:.2f}, seed={}",
                     config.train_ratio, config.val_ratio, config.test_ratio, config.split_seed);
    } else {
        spdlog::info("GraphCompiler: No DataSplit node - using defaults (train=0.80, val=0.10, test=0.10)");
    }

    // Extract training-loop config from DataLoader node if present.
    // DataLoader owns batch_size / epochs / shuffle / drop_last / num_workers —
    // all the "how do I iterate training" hyperparameters.
    if (const gui::MLNode* loader_node = FindFirstReachableNodeOfType(
            nodes, dataset_reachable, gui::NodeType::DataLoader)) {
        config.has_data_loader = true;
        try {
            if (loader_node->parameters.count("batch_size"))
                config.batch_size = std::stoi(loader_node->parameters.at("batch_size"));
            if (loader_node->parameters.count("epochs"))
                config.epochs = std::stoi(loader_node->parameters.at("epochs"));
            if (loader_node->parameters.count("shuffle"))
                config.shuffle = (loader_node->parameters.at("shuffle") == "true");
            if (loader_node->parameters.count("drop_last"))
                config.drop_last = (loader_node->parameters.at("drop_last") == "true");
            if (loader_node->parameters.count("num_workers"))
                config.num_workers = std::stoi(loader_node->parameters.at("num_workers"));
            const int requested_workers = config.num_workers;
            config.num_workers = ClampNumWorkersToPlatform(config.num_workers);
            if (config.num_workers != requested_workers) {
                spdlog::warn("GraphCompiler: clamping DataLoader num_workers from {} to {} based on platform",
                             requested_workers, config.num_workers);
            }
            if (loader_node->parameters.count("prefetch_factor")) {
                spdlog::warn("GraphCompiler: DataLoader prefetch_factor is currently ignored; "
                             "num_workers uses synchronous per-batch workers only");
            }
            if (loader_node->parameters.count("pin_memory") &&
                loader_node->parameters.at("pin_memory") == "true") {
                spdlog::warn("GraphCompiler: DataLoader pin_memory=true is currently ignored");
            }
            if (loader_node->parameters.count("save_best_checkpoint"))
                config.save_best_checkpoint = (loader_node->parameters.at("save_best_checkpoint") == "true");
            if (loader_node->parameters.count("early_stopping_patience"))
                config.early_stopping_patience = std::stoi(loader_node->parameters.at("early_stopping_patience"));
            if (loader_node->parameters.count("checkpoint_dir"))
                config.checkpoint_dir = loader_node->parameters.at("checkpoint_dir");
        } catch (const std::exception& e) {
            spdlog::warn("GraphCompiler: DataLoader param parse error ({}) - using defaults", e.what());
        }
    }
    if (config.has_data_loader) {
        spdlog::info("GraphCompiler: DataLoader node found - batch_size={}, epochs={}, shuffle={}, drop_last={}, num_workers={}, save_best_checkpoint={}, early_stopping_patience={}, checkpoint_dir='{}'",
                     config.batch_size, config.epochs, config.shuffle, config.drop_last, config.num_workers,
                     config.save_best_checkpoint, config.early_stopping_patience, config.checkpoint_dir);
        if (config.num_workers > 0) {
            spdlog::info("GraphCompiler: num_workers={} will be forwarded to supported batchers",
                         config.num_workers);
        }
    } else {
        spdlog::info("GraphCompiler: No DataLoader node - using defaults (batch_size=32, epochs=10, shuffle=true, drop_last=false)");
    }

    std::unordered_set<int> training_path_ids;
    if (dataset_node && loss_node) {
        const auto loss_ancestors = CollectAncestorNodeIds(loss_node->id, links);
        for (int node_id : dataset_reachable) {
            if (loss_ancestors.count(node_id) > 0) {
                training_path_ids.insert(node_id);
            }
        }
    }
    if (loss_node) {
        const auto loss_reachable = CollectReachableNodeIds(loss_node->id, links);
        if (const gui::MLNode* path_optimizer_node =
                FindFirstOptimizerNodeInSet(nodes, loss_reachable)) {
            optimizer_node = path_optimizer_node;
        }
    }
    ValidateTrainingPathImplementationStatus(nodes, training_path_ids, config);

    if (dataset_node) {
        config.data_source_node_id = dataset_node->id;
    }
    if (loss_node) {
        config.loss_node_id = loss_node->id;
    }
    if (optimizer_node) {
        config.optimizer_node_id = optimizer_node->id;
    }

    // Phase 4 Time-Series detection. If the graph contains a
    // TimeSeriesWindow node, mark the config so the training dispatch
    // routes the Arrow batcher in regression mode (float labels, no
    // one-hot) and drives index selection from the __partition__ column
    // emitted by TimeSeriesSplitOperator. The downstream batcher is the
    // same ArrowDatasetBatcher used by tabular classification — only
    // the constructor args and SetRegressionMode switch differ.
    //
    // Also override config.input_size to match the operator's
    // input_width. The raw CSV column count is meaningless once
    // TimeSeriesWindow materializes the table — the first Dense layer
    // receives `input_width` features (x_0..x_{input_width-1}), not
    // the original per-row scalar. Without this override
    // BuildSequentialFromConfig would construct Linear(1 -> hidden_units)
    // and crash at the first forward pass with a dimension mismatch.
    for (const auto& node : nodes) {
        if (node.type == gui::NodeType::TimeSeriesWindow &&
            ContainsWhenFiltered(training_path_ids, node.id)) {
            config.is_time_series = true;
            int input_width = 12;
            auto iw_it = node.parameters.find("input_width");
            if (iw_it != node.parameters.end() && !iw_it->second.empty()) {
                try { input_width = std::stoi(iw_it->second); }
                catch (...) { /* fall back to default */ }
            }
            // Multivariate: input_size = input_width * (1 + num_feature_cols).
            // feature_cols is comma-separated; count non-empty tokens.
            // Single-variate (empty feature_cols) keeps input_size = input_width.
            int num_features = 1;
            auto fc_it = node.parameters.find("feature_cols");
            if (fc_it != node.parameters.end() && !fc_it->second.empty()) {
                int extras = 0;
                std::stringstream ss(fc_it->second);
                std::string token;
                while (std::getline(ss, token, ',')) {
                    size_t start = token.find_first_not_of(" \t");
                    if (start != std::string::npos) ++extras;
                }
                num_features = 1 + extras;
            }
            const int total_input = input_width * num_features;
            if (total_input > 0) {
                config.input_size = static_cast<size_t>(total_input);
                config.input_shape = {static_cast<size_t>(total_input)};
            }
            spdlog::info("GraphCompiler: TimeSeriesWindow found - regression mode, "
                         "input_size={} (input_width={} x num_features={})",
                         config.input_size, input_width, num_features);
            break;
        }
    }

    // === Early domain detection ===
    // Shape inference below needs the dataset domain before it sees the
    // first model layer. Text graphs in particular use max_length as the
    // synthetic/debug input width; if this is left empty, Local Debug falls
    // back to [1] and Studio Debugger reports cosmetic shape mismatches.
    if (dataset_node) {
        auto cat_it = dataset_node->parameters.find("file_category");
        const std::string cat = (cat_it != dataset_node->parameters.end())
            ? cat_it->second : std::string();

        if (auto* cat_loader = loaders::GetByCategory(
                loaders::FileCategoryFromString(cat))) {
            config.preprocessing_domain = cat_loader->Domain(cat);
        }
    }

    // Get topologically sorted node IDs
    std::vector<int> sorted_ids = TopologicalSort(nodes, links);
    CollectGraphRuntimeOpNodeIds(
        nodes,
        links,
        sorted_ids,
        training_path_ids,
        config);
    config.graph_plan = BuildCompiledGraphPlan(
        nodes,
        links,
        sorted_ids,
        training_path_ids,
        config.data_source_node_id,
        config.loss_node_id,
        config.optimizer_node_id);

    // Check if we're using CrossEntropyLoss (which applies softmax internally)
    bool using_cross_entropy = (loss_node && loss_node->type == gui::NodeType::CrossEntropyLoss);

    // Log all nodes in the graph for debugging
    spdlog::info("GraphCompiler: Processing {} nodes (using_cross_entropy={})", sorted_ids.size(), using_cross_entropy);
    for (int node_id : sorted_ids) {
        const gui::MLNode* n = FindNodeById(node_id, nodes);
        if (n) {
            spdlog::debug("GraphCompiler: Node[{}] '{}' type={}", n->id, n->name, static_cast<int>(n->type));
        }
    }

    // Extract model layers and preprocessing in execution order
    std::vector<size_t> current_shape = config.input_shape;

    for (int node_id : sorted_ids) {
        const gui::MLNode* node = FindNodeById(node_id, nodes);
        if (!node) continue;
        const bool is_execution_node =
            IsPreprocessing(node->type) ||
            IsModelLayer(node->type) ||
            IsActivation(node->type);
        if (is_execution_node &&
            !ContainsWhenFiltered(training_path_ids, node->id)) {
            spdlog::debug("GraphCompiler: Skipping node '{}' outside selected training path",
                          node->name);
            continue;
        }

        // Handle preprocessing nodes
        if (IsPreprocessing(node->type)) {
            spdlog::info("GraphCompiler: Found preprocessing node '{}' (type={})", node->name, static_cast<int>(node->type));
            ExtractPreprocessing(*node, config);
            ApplyTextInputShape(config);
            current_shape = config.input_shape;
            if (node->type == gui::NodeType::Normalize) {
                spdlog::info("GraphCompiler: Normalization enabled - mean={}, std={}",
                             config.preprocessing.norm_mean, config.preprocessing.norm_std);
            }
            continue;
        }

        // Skip Softmax when using CrossEntropyLoss (it applies softmax internally)
        // This prevents double-softmax which kills gradients
        if (node->type == gui::NodeType::Softmax && using_cross_entropy) {
            spdlog::warn("GraphCompiler: Softmax layer skipped - CrossEntropyLoss applies softmax internally");
            continue;
        }

        if (IsGraphRuntimeFanInOp(node->type) &&
            HasConnectedInputAfterFirst(*node, links)) {
            continue;
        }

        // Handle model layers and activations
        if (IsModelLayer(node->type) || IsActivation(node->type)) {
            CompiledLayer layer = ExtractLayerConfig(*node);
            layer.input_shape = current_shape;

            // Infer output shape
            if (node->type == gui::NodeType::Reshape ||
                node->type == gui::NodeType::View ||
                node->type == gui::NodeType::Squeeze ||
                node->type == gui::NodeType::Unsqueeze ||
                node->type == gui::NodeType::Permute ||
                node->type == gui::NodeType::TensorBroadcastTo ||
                node->type == gui::NodeType::TensorExpand ||
                node->type == gui::NodeType::TensorIndexSelect) {
                std::string error;
                bool ok = false;
                if (node->type == gui::NodeType::Permute) {
                    ok = ResolvePermuteTargetShape(layer.parameters,
                                                   current_shape,
                                                   layer.output_shape,
                                                   layer.dims,
                                                   error);
                } else {
                    ok = ResolveShapeOpTargetShape(node->type,
                                                   layer.parameters,
                                                   current_shape,
                                                   layer.output_shape,
                                                   error);
                }
                if (!ok) {
                    AddIssue(config, IssueLevel::Error, error, node->id, node->name);
                    layer.output_shape = current_shape;
                }
            } else if (node->type == gui::NodeType::TensorPow ||
                       node->type == gui::NodeType::TensorClip) {
                std::string error;
                if (!ValidateTensorScalarMathParams(node->type,
                                                    layer.parameters,
                                                    error)) {
                    AddIssue(config, IssueLevel::Error, error, node->id, node->name);
                }
                layer.output_shape = current_shape;
            } else if (node->type == gui::NodeType::TensorCompare ||
                       node->type == gui::NodeType::TensorLogicalMask) {
                std::string error;
                if (HasConnectedInputAfterFirst(*node, links)) {
                    error = node->type == gui::NodeType::TensorCompare
                        ? "TensorCompare currently supports only scalar comparison; disconnect input B"
                        : "TensorLogicalMask currently supports only unary op=not; disconnect input B";
                    AddIssue(config, IssueLevel::Error, error, node->id, node->name);
                } else if (!ValidateTensorMaskParams(node->type,
                                                     layer.parameters,
                                                     error)) {
                    AddIssue(config, IssueLevel::Error, error, node->id, node->name);
                }
                layer.output_shape = current_shape;
            } else if (node->type == gui::NodeType::TensorSum ||
                       node->type == gui::NodeType::TensorMean ||
                       node->type == gui::NodeType::TensorMax ||
                       node->type == gui::NodeType::TensorMin ||
                       node->type == gui::NodeType::TensorProd ||
                       node->type == gui::NodeType::TensorVar ||
                       node->type == gui::NodeType::TensorStd) {
                std::string error;
                if (!ResolveReductionTargetShape(node->type,
                                                 layer.parameters,
                                                 current_shape,
                                                 layer.output_shape,
                                                 error)) {
                    AddIssue(config, IssueLevel::Error, error, node->id, node->name);
                    layer.output_shape = current_shape;
                }
            } else {
                layer.output_shape = InferOutputShape(layer, current_shape);
            }
            current_shape = layer.output_shape;

            config.layers.push_back(layer);

            // Track output size from last Dense layer
            if (node->type == gui::NodeType::Dense) {
                config.output_size = layer.units;
            }
        }
    }

    // Extract loss configuration
    if (loss_node) {
        config.loss_type = loss_node->type;
        config.loss_params = loss_node->parameters;
    }

    // Extract optimizer configuration
    if (optimizer_node) {
        config.optimizer_type = optimizer_node->type;

        if (optimizer_node->parameters.count("learning_rate"))
            config.learning_rate = std::stof(optimizer_node->parameters.at("learning_rate"));
        if (optimizer_node->parameters.count("lr"))
            config.learning_rate = std::stof(optimizer_node->parameters.at("lr"));
        if (optimizer_node->parameters.count("momentum"))
            config.momentum = std::stof(optimizer_node->parameters.at("momentum"));
        if (optimizer_node->parameters.count("beta1"))
            config.beta1 = std::stof(optimizer_node->parameters.at("beta1"));
        if (optimizer_node->parameters.count("beta2"))
            config.beta2 = std::stof(optimizer_node->parameters.at("beta2"));
        if (optimizer_node->parameters.count("weight_decay"))
            config.weight_decay = std::stof(optimizer_node->parameters.at("weight_decay"));
    }

    // Set one-hot encoding if we have classification (CrossEntropy loss)
    if (config.loss_type == gui::NodeType::CrossEntropyLoss) {
        config.preprocessing.has_onehot = true;

        // Try to get num_classes from Output node's "classes" parameter
        const gui::MLNode* output_node = FindFirstReachableNodeOfType(
            nodes, training_path_ids, gui::NodeType::Output);
        if (!output_node) {
            output_node = FindOutputNode(nodes);
        }
        if (output_node) {
            auto it = output_node->parameters.find("classes");
            if (it != output_node->parameters.end() && !it->second.empty()) {
                try {
                    config.preprocessing.num_classes = std::stoul(it->second);
                    spdlog::info("GraphCompiler: num_classes={} from Output node",
                                 config.preprocessing.num_classes);
                } catch (...) {
                    config.preprocessing.num_classes = 0;  // Will fall back below
                }
            }
        }

        // Fallback to output_size if Output node doesn't specify classes
        if (config.preprocessing.num_classes == 0) {
            config.preprocessing.num_classes = config.output_size;
            spdlog::warn("GraphCompiler: num_classes not specified in Output node, using output_size={}",
                         config.output_size);
        }

        if (config.output_size > 0 && config.output_size < 2) {
            AddIssue(config, IssueLevel::Error,
                     "CrossEntropyLoss requires at least two prediction logits; "
                     "the selected model path outputs " +
                     std::to_string(config.output_size));
        }

        if (config.preprocessing.num_classes > 0 &&
            config.output_size > 0 &&
            config.preprocessing.num_classes != config.output_size) {
            AddIssue(config, IssueLevel::Error,
                     "CrossEntropyLoss class count (" +
                     std::to_string(config.preprocessing.num_classes) +
                     ") does not match the model output size (" +
                     std::to_string(config.output_size) + ")");
        }
    }

    if (config.loss_type == gui::NodeType::BCELoss ||
        config.loss_type == gui::NodeType::BCEWithLogits) {
        if (config.output_size > 0 && config.output_size != 1) {
            const char* loss_name = config.loss_type == gui::NodeType::BCELoss
                ? "BCELoss"
                : "BCEWithLogits";
            AddIssue(config, IssueLevel::Error,
                     std::string(loss_name) +
                     " requires a single prediction output for binary "
                     "classification; the selected model path outputs " +
                     std::to_string(config.output_size));
        }
    }

    // === Post-compile sanity checks ===
    // These need the values populated by the layer-extraction passes
    // above, so they live at the end of Compile().

    // DataSplit ratios should sum to ~1.0. Drift > 0.05 is almost
    // certainly a typo or stale state from the user adjusting one
    // ratio without rebalancing the others.
    if (config.has_data_split) {
        float sum = config.train_ratio + config.val_ratio + config.test_ratio;
        if (std::abs(sum - 1.0f) > 0.05f) {
            std::ostringstream msg;
            msg << "DataSplit ratios sum to " << sum
                << " (expected 1.0) - train=" << config.train_ratio
                << ", val=" << config.val_ratio
                << ", test=" << config.test_ratio;
            AddIssue(config, IssueLevel::Warning, msg.str());
        }
        if (config.val_ratio < 1e-6f) {
            AddIssue(config, IssueLevel::Warning,
                     "Validation split is 0 - training will run without validation metrics");
        }
    }

    // batch_size must fit in the train set. We can only check the upper
    // bound when the dataset is actually loaded — otherwise the row count
    // is unknown and the existing data_loaded check above already fired.
    if (config.batch_size <= 0) {
        AddIssue(config, IssueLevel::Error,
                 "batch_size must be positive (got " +
                 std::to_string(config.batch_size) + ")");
    } else if (dataset_node && !config.dataset_name.empty()) {
        auto& reg = DataRegistry::Instance();
        int64_t total_rows = 0;
        if (auto arrow_ds = reg.GetArrowDataset(config.dataset_name)) {
            total_rows = arrow_ds->GetNumRows();
        } else if (auto pq_ds = reg.GetParquetBackedDataset(config.dataset_name)) {
            total_rows = pq_ds->GetNumRows();
        }
        if (total_rows > 0) {
            int64_t train_rows = static_cast<int64_t>(total_rows * config.train_ratio);
            if (config.batch_size > train_rows) {
                AddIssue(config, IssueLevel::Error,
                         "batch_size (" + std::to_string(config.batch_size) +
                         ") is larger than the train split (" +
                         std::to_string(train_rows) + " rows)");
            } else if (config.batch_size > train_rows / 2) {
                AddIssue(config, IssueLevel::Warning,
                         "batch_size (" + std::to_string(config.batch_size) +
                         ") is more than half the train split (" +
                         std::to_string(train_rows) + " rows) - few iterations per epoch");
            }
        }
    }

    // === Domain detection ===
    // The DataInput node carries file_category (tabular / image / audio /
    // text / timeseries). The owning loader reports its PreprocessingDomain
    // via Domain(file_category) — TabularLoader splits Tabular vs
    // TimeSeries based on the category string; other loaders return a
    // fixed value.
    //
    // Per-domain notes preserved from the pre-refactor code:
    //   - TimeSeries: TimeSeriesWindow runs as a real Cat-1
    //     IPipelineOperator in the materializer pass, NOT as a config
    //     extractor. The compile gate's checks below (label column,
    //     batch size vs row count) run on the pre-materialized CSV —
    //     validating the user's raw column pick, not the post-window
    //     x_0..x_n schema.
    //   - Audio / Text: either the graph has extractor nodes (Spectrogram
    //     / MelSpec / MFCC for audio, TextTokenizer / TextVocabulary /
    //     TextPadding for text) that populate config.audio_preprocessing
    //     / text_preprocessing later, or we fall back to the dialog-
    //     baked defaults on the registry entry. The end-of-Compile()
    //     log reports which mode won.
    if (dataset_node) {
        auto cat_it = dataset_node->parameters.find("file_category");
        const std::string cat = (cat_it != dataset_node->parameters.end())
            ? cat_it->second : std::string();

        if (auto* cat_loader = loaders::GetByCategory(
                loaders::FileCategoryFromString(cat))) {
            config.preprocessing_domain = cat_loader->Domain(cat);
        }

        for (const auto& node : nodes) {
            if (!ContainsWhenFiltered(training_path_ids, node.id)) {
                continue;
            }

            auto node_domain = GetPreprocessingNodeDomain(node.type);
            if (!node_domain ||
                *node_domain == PreprocessingDomain::General ||
                *node_domain == config.preprocessing_domain) {
                continue;
            }

            std::ostringstream msg;
            msg << "Preprocessing node '" << node.name << "' is for "
                << PreprocessingDomainLabel(*node_domain)
                << " data, but the selected DataInput is "
                << PreprocessingDomainLabel(config.preprocessing_domain)
                << ". Use a matching DataInput category or replace this "
                   "preprocessing node.";
            AddIssue(config, IssueLevel::Error, msg.str(), node.id, node.name);
        }

        // is_image kept as a local for the image-specific checks below
        // (Resize, Augmentation, etc.) — those stay domain-scoped and
        // don't need to move into the loader yet.
        const bool is_image = (cat == "image");

        if (is_image) {

            // Check 1: Resize node required for image datasets.
            // Without it, images load at their native (variable) size,
            // which can't be batched AND can cause massive tensors that
            // OOM the GPU. This is the most critical image check.
            bool has_resize = false;
            for (const auto& node : nodes) {
                if (node.type == gui::NodeType::Resize) {
                    has_resize = true;
                    break;
                }
            }
            if (!has_resize) {
                AddIssue(config, IssueLevel::Error,
                         "Image datasets require a Resize node to set target "
                         "dimensions. Without it, images load at their native "
                         "size which can exceed GPU memory. Add a Resize node "
                         "between DataInput and the first layer (e.g. "
                         "Resize 64x64 for fast training, 224x224 for accuracy).",
                         dataset_node->id, dataset_node->name);
            }

            // Check 2: Flatten required before Dense layers for images.
            // Images are 3D (H×W×C); Dense expects a 1D flat vector.
            if (!config.layers.empty()) {
                auto first_layer = config.layers[0];
                if (first_layer.type == gui::NodeType::Dense) {
                    bool has_flatten_before_dense = false;
                    for (const auto& node : nodes) {
                        if (node.type == gui::NodeType::Flatten) {
                            has_flatten_before_dense = true;
                            break;
                        }
                    }
                    if (!has_flatten_before_dense) {
                        AddIssue(config, IssueLevel::Error,
                                 "First model layer is Dense but no Flatten node "
                                 "found. Image data is 3D (H x W x C) and needs "
                                 "Flatten before Dense layers.",
                                 first_layer.node_id, first_layer.name);
                    }
                }
            }

            // Check 3: Memory estimation for image models.
            // Estimate total model memory (weights + gradients + optimizer)
            // and warn if it's likely to exceed reasonable GPU memory.
            if (has_resize && config.image_preprocessing.target_width > 0) {
                size_t img_features = static_cast<size_t>(
                    config.image_preprocessing.target_width) *
                    config.image_preprocessing.target_height * 3;
                size_t total_params = 0;
                size_t prev_size = img_features;
                for (const auto& layer : config.layers) {
                    if (layer.type == gui::NodeType::Dense) {
                        total_params += prev_size * layer.units + layer.units;
                        prev_size = layer.units;
                    } else if (layer.type == gui::NodeType::Flatten) {
                        // prev_size stays the same (just reshaped)
                    }
                }
                // Memory = weights × 4 (weights + grads + adam_m + adam_v)
                // + batch activations (batch_size × largest_layer)
                size_t model_bytes = total_params * 4 * sizeof(float);
                size_t batch_bytes = static_cast<size_t>(config.batch_size) *
                    img_features * sizeof(float);
                size_t total_est = model_bytes + batch_bytes;

                double est_mb = total_est / (1024.0 * 1024.0);

                std::ostringstream mem_msg;
                mem_msg << "Estimated GPU memory: " << std::fixed
                        << std::setprecision(0) << est_mb << " MB ("
                        << total_params << " parameters, batch_size="
                        << config.batch_size << ", input="
                        << config.image_preprocessing.target_width << "x"
                        << config.image_preprocessing.target_height << "x3)";

                // Real GPU usage is ~3-5x the raw parameter estimate due to
                // CUDA context (~300 MB), forward/backward activations,
                // matmul temporaries, and optimizer state. Use a 4x
                // multiplier for a realistic bound.
                double realistic_mb = est_mb * 4.0;

                if (realistic_mb > 3000) {
                    // Suggest a smaller size that would fit
                    int safe_features = 64 * 64 * 3;  // 12288
                    mem_msg << ". WARNING: real GPU usage is ~"
                            << std::setprecision(0) << realistic_mb
                            << " MB (incl. activations + optimizer + CUDA "
                            << "overhead). This will likely OOM on your GPU. "
                            << "Set the Resize node to a smaller size "
                            << "(e.g. 64x64 for " << safe_features
                            << " features)";
                    AddIssue(config, IssueLevel::Error, mem_msg.str());
                } else if (realistic_mb > 2000) {
                    mem_msg << ". This is close to typical GPU memory limits "
                            << "— consider reducing Resize dimensions or "
                            << "batch_size if training crashes";
                    AddIssue(config, IssueLevel::Warning, mem_msg.str());
                } else {
                    AddIssue(config, IssueLevel::Info, mem_msg.str());
                }
            }

            // Check 4: Pipeline ordering — Normalize before Resize is
            // almost certainly wrong (normalization stats are scale-
            // dependent). Walk the TOPOLOGICAL order (from links), not
            // the raw node vector (creation order), so the check reflects
            // the actual data flow.
            {
                std::vector<int> topo = TopologicalSort(nodes, links);
                bool seen_normalize = false;
                for (int nid : topo) {
                    if (!ContainsWhenFiltered(training_path_ids, nid)) {
                        continue;
                    }
                    const gui::MLNode* n = FindNodeById(nid, nodes);
                    if (!n) continue;
                    if (n->type == gui::NodeType::Normalize) {
                        seen_normalize = true;
                    }
                    if (n->type == gui::NodeType::Resize && seen_normalize) {
                        AddIssue(config, IssueLevel::Warning,
                                 "Normalize appears before Resize in the pipeline. "
                                 "Normalization is scale-dependent — resizing after "
                                 "normalization gives wrong per-pixel values. Move "
                                 "Resize before Normalize.");
                        break;
                    }
                }
            }
        }
    }

    // Audio preprocessing mode summary — surface whether the graph's
    // feature node overrode the dialog defaults, or whether we fell
    // back. Useful for debugging "why isn't my MelSpectrogram n_mels=64
    // taking effect" without having to grep through the logs for
    // individual extractor calls.
    if (config.preprocessing_domain == PreprocessingDomain::Audio) {
        if (config.audio_preprocessing.has_feature_node) {
            spdlog::info("GraphCompiler: audio feature extraction driven by graph "
                         "node (type={})",
                         static_cast<int>(config.audio_preprocessing.feature_type));
        } else {
            spdlog::info("GraphCompiler: audio feature extraction uses dialog defaults "
                         "(no Spectrogram/MelSpectrogram/MFCC node in graph)");
        }
    }

    // Text preprocessing is no longer extracted into TrainingConfiguration.
    // TextTokenizer is a real Arrow operator; TextVocabulary/TextPadding
    // fold into it during PipelineMaterializer. If no materialized Arrow
    // table is selected, TextDatasetBatcher uses the dialog-baked defaults
    // from DataRegistry::TextDatasetEntry.
    if (config.preprocessing_domain == PreprocessingDomain::Text) {
        spdlog::info("GraphCompiler: text preprocessing nodes are materialized "
                     "through Arrow operators; legacy text fallback uses dialog "
                     "defaults from the registered dataset");
    }

    // Final verdict: is_valid is the absence of any Error-level issue.
    // Warnings and Info don't block training.
    config.is_valid = !config.HasErrors();
    config.error_message = JoinErrorMessages(config.issues);

    spdlog::info("GraphCompiler: Compiled {} layers, input_size={}, output_size={}, "
                 "issues: {} errors / {} warnings / {} info, valid={}",
                 config.layers.size(), config.input_size, config.output_size,
                 config.CountIssues(IssueLevel::Error),
                 config.CountIssues(IssueLevel::Warning),
                 config.CountIssues(IssueLevel::Info),
                 config.is_valid);
    for (const auto& issue : config.issues) {
        spdlog::info("  [{}] {}{}",
                     IssueLevelLabel(issue.level),
                     issue.node_name.empty() ? "" : ("[" + issue.node_name + "] "),
                     issue.message);
    }

    return config;
}

bool GraphCompiler::ValidateGraph(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    std::string& error)
{
    // Check for empty graph
    if (nodes.empty()) {
        error = "Graph is empty - add nodes to create a model";
        return false;
    }

    // Check for dataset input
    if (!FindDatasetInputNode(nodes, links)) {
        error = "Graph must have a DatasetInput node";
        return false;
    }

    // Check for at least one model layer
    bool has_model_layer = false;
    for (const auto& node : nodes) {
        if (IsModelLayer(node.type)) {
            has_model_layer = true;
            break;
        }
    }
    if (!has_model_layer) {
        error = "Graph must have at least one model layer (Dense, Conv2D, etc.)";
        return false;
    }

    // Check for loss function
    if (!FindLossNode(nodes)) {
        error = "Graph must have a loss function (MSELoss or CrossEntropyLoss)";
        return false;
    }

    // Check for optimizer
    if (!FindOptimizerNode(nodes)) {
        error = "Graph must have an optimizer (SGD, Adam, AdamW, RMSprop, Adagrad, or NAdam)";
        return false;
    }

    // Check for cycles
    if (HasCycle(nodes, links)) {
        error = "Graph contains a cycle - remove circular connections";
        return false;
    }

    return true;
}

std::vector<int> GraphCompiler::TopologicalSort(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links)
{
    // Build adjacency list and in-degree count
    std::map<int, std::vector<int>> adj;
    std::map<int, int> in_degree;

    for (const auto& node : nodes) {
        adj[node.id] = {};
        in_degree[node.id] = 0;
    }

    for (const auto& link : links) {
        adj[link.from_node].push_back(link.to_node);
        in_degree[link.to_node]++;
    }

    // Kahn's algorithm
    std::queue<int> queue;
    for (const auto& node : nodes) {
        if (in_degree[node.id] == 0) {
            queue.push(node.id);
        }
    }

    std::vector<int> sorted;
    while (!queue.empty()) {
        int node_id = queue.front();
        queue.pop();
        sorted.push_back(node_id);

        for (int neighbor : adj[node_id]) {
            in_degree[neighbor]--;
            if (in_degree[neighbor] == 0) {
                queue.push(neighbor);
            }
        }
    }

    return sorted;
}

const gui::MLNode* GraphCompiler::FindNodeById(int id, const std::vector<gui::MLNode>& nodes) const {
    for (const auto& node : nodes) {
        if (node.id == id) return &node;
    }
    return nullptr;
}

std::vector<int> GraphCompiler::GetConnectedNodes(
    int from_node_id,
    const std::vector<gui::NodeLink>& links) const
{
    std::vector<int> connected;
    for (const auto& link : links) {
        if (link.from_node == from_node_id) {
            connected.push_back(link.to_node);
        }
    }
    return connected;
}

std::vector<int> GraphCompiler::GetInputNodes(
    int to_node_id,
    const std::vector<gui::NodeLink>& links) const
{
    std::vector<int> inputs;
    for (const auto& link : links) {
        if (link.to_node == to_node_id) {
            inputs.push_back(link.from_node);
        }
    }
    return inputs;
}

const gui::MLNode* GraphCompiler::FindDatasetInputNode(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links) const {
    const gui::MLNode* first_source = nullptr;
    const gui::MLNode* first_connected_source = nullptr;
    for (const auto& node : nodes) {
        if (!IsDatasetSourceType(node.type)) {
            continue;
        }

        if (!first_source) {
            first_source = &node;
        }

        if (!HasOutgoingLink(node.id, links)) {
            continue;
        }

        if (!first_connected_source) {
            first_connected_source = &node;
        }

        const auto reachable = CollectReachableNodeIds(node.id, links);
        for (const auto& candidate : nodes) {
            if (IsLossNodeType(candidate.type) &&
                reachable.count(candidate.id) > 0) {
                return &node;
            }
        }
    }
    return first_connected_source ? first_connected_source : first_source;
}

const gui::MLNode* GraphCompiler::FindLossNode(const std::vector<gui::MLNode>& nodes) const {
    for (const auto& node : nodes) {
        if (IsLossNodeType(node.type)) {
            return &node;
        }
    }
    return nullptr;
}

const gui::MLNode* GraphCompiler::FindOptimizerNode(const std::vector<gui::MLNode>& nodes) const {
    for (const auto& node : nodes) {
        if (IsSupportedOptimizerNodeType(node.type)) {
            return &node;
        }
    }
    return nullptr;
}

const gui::MLNode* GraphCompiler::FindOutputNode(const std::vector<gui::MLNode>& nodes) const {
    for (const auto& node : nodes) {
        if (node.type == gui::NodeType::Output) {
            return &node;
        }
    }
    return nullptr;
}

bool GraphCompiler::IsModelLayer(gui::NodeType type) const {
    switch (type) {
        case gui::NodeType::Dense:
        case gui::NodeType::Conv2D:
        case gui::NodeType::MaxPool2D:
        case gui::NodeType::AvgPool2D:
        case gui::NodeType::GlobalMaxPool:
        case gui::NodeType::GlobalAvgPool:
        case gui::NodeType::Flatten:
        case gui::NodeType::Reshape:
        case gui::NodeType::View:
        case gui::NodeType::Permute:
        case gui::NodeType::Squeeze:
        case gui::NodeType::Unsqueeze:
        case gui::NodeType::TensorBroadcastTo:
        case gui::NodeType::TensorExpand:
        case gui::NodeType::TensorIndexSelect:
        case gui::NodeType::TensorAbs:
        case gui::NodeType::TensorExp:
        case gui::NodeType::TensorLog:
        case gui::NodeType::TensorSqrt:
        case gui::NodeType::TensorSign:
        case gui::NodeType::TensorPow:
        case gui::NodeType::TensorClip:
        case gui::NodeType::TensorCompare:
        case gui::NodeType::TensorLogicalMask:
        case gui::NodeType::TensorSum:
        case gui::NodeType::TensorMean:
        case gui::NodeType::TensorMax:
        case gui::NodeType::TensorMin:
        case gui::NodeType::TensorProd:
        case gui::NodeType::TensorVar:
        case gui::NodeType::TensorStd:
        case gui::NodeType::Dropout:
        case gui::NodeType::BatchNorm:
        case gui::NodeType::ConvTranspose2D:
        case gui::NodeType::Upsample:
        case gui::NodeType::PixelShuffle:
        case gui::NodeType::PolicyNetwork:
        case gui::NodeType::ValueNetwork:
        case gui::NodeType::Embedding:
        // Recurrent layers — required for text/time-series models.
        // Omitting them here caused the LSTM smoke test to silently
        // drop the LSTM node from `config_.layers`, leaving
        // training_executor to feed [batch, seq, embed] straight into
        // a Dense layer sized for [batch, seq*embed] — runtime shape
        // mismatch and crash.
        case gui::NodeType::LSTM:
        case gui::NodeType::GRU:
        case gui::NodeType::RNN:
        case gui::NodeType::Bidirectional:
            return true;
        default:
            return false;
    }
}

bool GraphCompiler::IsActivation(gui::NodeType type) const {
    switch (type) {
        case gui::NodeType::ReLU:
        case gui::NodeType::LeakyReLU:
        case gui::NodeType::ELU:
        case gui::NodeType::GELU:
        case gui::NodeType::Swish:
        case gui::NodeType::Mish:
        case gui::NodeType::Sigmoid:
        case gui::NodeType::Tanh:
        case gui::NodeType::Softmax:
            return true;
        default:
            return false;
    }
}

// ---------------------------------------------------------------------------
// Table-driven preprocessing registry
//
// Each entry maps a NodeType to a preprocessing domain and an optional
// extraction function. IsPreprocessing and ExtractPreprocessing both
// consult this table instead of maintaining separate switch statements.
//
// Adding a new preprocessing node for any domain is a single line here
// plus a static extraction function. No switch cases to update in
// multiple places.
// ---------------------------------------------------------------------------

namespace {

using ExtractorFn = void(*)(const gui::MLNode&, TrainingConfiguration&);

struct PreprocessingNodeSpec {
    gui::NodeType type;
    PreprocessingDomain domain;
    ExtractorFn extractor;  // nullptr = recognized but not yet wired
};

// --- Tabular extractors (migrated from the old switch) ---

static void ExtractReshape(const gui::MLNode& node, TrainingConfiguration& config) {
    config.preprocessing.has_reshape = true;
    if (node.parameters.count("shape")) {
        std::string shape_str = node.parameters.at("shape");
        shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), '['), shape_str.end());
        shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ']'), shape_str.end());
        shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ' '), shape_str.end());

        size_t pos = 0;
        while ((pos = shape_str.find(',')) != std::string::npos) {
            config.preprocessing.reshape_dims.push_back(std::stoi(shape_str.substr(0, pos)));
            shape_str.erase(0, pos + 1);
        }
        if (!shape_str.empty()) {
            config.preprocessing.reshape_dims.push_back(std::stoi(shape_str));
        }
    }
}

static void ExtractOneHot(const gui::MLNode& node, TrainingConfiguration& config) {
    config.preprocessing.has_onehot = true;
    if (node.parameters.count("num_classes"))
        config.preprocessing.num_classes = std::stoul(node.parameters.at("num_classes"));
}

// --- Image extractors (Phase 1) ---

static void ExtractImageResize(const gui::MLNode& node, TrainingConfiguration& config) {
    config.image_preprocessing.resize_mode = ResizeMode::Exact;
    if (node.parameters.count("width"))
        config.image_preprocessing.target_width = std::stoi(node.parameters.at("width"));
    if (node.parameters.count("height"))
        config.image_preprocessing.target_height = std::stoi(node.parameters.at("height"));
    if (node.parameters.count("mode")) {
        std::string mode = node.parameters.at("mode");
        if (mode == "exact")       config.image_preprocessing.resize_mode = ResizeMode::Exact;
        else if (mode == "fit")    config.image_preprocessing.resize_mode = ResizeMode::AspectFit;
        else if (mode == "fill")   config.image_preprocessing.resize_mode = ResizeMode::AspectFill;
        else if (mode == "center") config.image_preprocessing.resize_mode = ResizeMode::Center;
    }
    spdlog::info("GraphCompiler: Resize {}x{} mode={}",
                 config.image_preprocessing.target_width,
                 config.image_preprocessing.target_height,
                 static_cast<int>(config.image_preprocessing.resize_mode));
}

static void ExtractImageNormalize(const gui::MLNode& node, TrainingConfiguration& config) {
    // Domain-aware Normalize: when upstream is image data, populate
    // both the legacy GraphPreprocessingConfig (for backward-compat
    // with the tabular training path) and the image-specific config.
    // The image batcher reads image_preprocessing; the tabular batcher
    // reads config.preprocessing. Both work.
    if (config.preprocessing_domain == PreprocessingDomain::Image) {
        config.image_preprocessing.enable_denoise = false;  // not related but clear the field
        // Image normalize is handled by the ImageTransformPipeline;
        // the legacy fields are also set so existing tabular code paths
        // don't break if they run accidentally.
    }
    // Always populate the legacy tabular fields (backward-compat).
    config.preprocessing.has_normalization = true;
    if (node.parameters.count("mean"))
        config.preprocessing.norm_mean = std::stof(node.parameters.at("mean"));
    if (node.parameters.count("std"))
        config.preprocessing.norm_std = std::stof(node.parameters.at("std"));
}

static void ExtractGrayscale(const gui::MLNode& /*node*/, TrainingConfiguration& config) {
    config.image_preprocessing.convert_to_grayscale = true;
}

static void ExtractImageGaussianBlur(const gui::MLNode& node, TrainingConfiguration& config) {
    config.image_preprocessing.blur_config.enabled = true;
    config.image_preprocessing.blur_config.type = BlurType::Gaussian;
    if (node.parameters.count("kernel_size"))
        config.image_preprocessing.blur_config.kernel_size = std::stoi(node.parameters.at("kernel_size"));
    if (node.parameters.count("sigma"))
        config.image_preprocessing.blur_config.sigma = std::stof(node.parameters.at("sigma"));
}

// --- Audio extractors (Phase 2.1) ---
//
// Each extractor sets has_feature_node=true (except AudioAugmentation,
// which sets has_augmentation=true) and copies the GUI node's parameters
// into config.audio_preprocessing. AudioDatasetBatcher later merges this
// into its AudioDatasetConfig, overriding the dialog-baked defaults on
// AudioDatasetEntry. If no audio feature node is found in the graph,
// has_feature_node stays false and the batcher uses the dialog defaults
// — "fallback mode", the standard UX.

static void ExtractSpectrogram(const gui::MLNode& node, TrainingConfiguration& config) {
    config.audio_preprocessing.has_feature_node = true;
    config.audio_preprocessing.feature_type = AudioFeatureType::Spectrogram;
    if (node.parameters.count("n_fft"))
        config.audio_preprocessing.n_fft = std::stoi(node.parameters.at("n_fft"));
    if (node.parameters.count("hop_length"))
        config.audio_preprocessing.hop_length = std::stoi(node.parameters.at("hop_length"));
    if (node.parameters.count("log_scale"))
        config.audio_preprocessing.log_scale = (node.parameters.at("log_scale") == "true");
    spdlog::info("GraphCompiler: Spectrogram n_fft={}, hop={}, log={}",
                 config.audio_preprocessing.n_fft,
                 config.audio_preprocessing.hop_length,
                 config.audio_preprocessing.log_scale);
}

static void ExtractMelSpectrogram(const gui::MLNode& node, TrainingConfiguration& config) {
    config.audio_preprocessing.has_feature_node = true;
    config.audio_preprocessing.feature_type = AudioFeatureType::MelSpectrogram;
    if (node.parameters.count("n_fft"))
        config.audio_preprocessing.n_fft = std::stoi(node.parameters.at("n_fft"));
    if (node.parameters.count("hop_length"))
        config.audio_preprocessing.hop_length = std::stoi(node.parameters.at("hop_length"));
    if (node.parameters.count("n_mels"))
        config.audio_preprocessing.n_mels = std::stoi(node.parameters.at("n_mels"));
    if (node.parameters.count("fmin"))
        config.audio_preprocessing.fmin = std::stof(node.parameters.at("fmin"));
    if (node.parameters.count("fmax"))
        config.audio_preprocessing.fmax = std::stof(node.parameters.at("fmax"));
    if (node.parameters.count("log_scale"))
        config.audio_preprocessing.log_scale = (node.parameters.at("log_scale") == "true");
    spdlog::info("GraphCompiler: MelSpectrogram n_mels={}, n_fft={}, hop={}",
                 config.audio_preprocessing.n_mels,
                 config.audio_preprocessing.n_fft,
                 config.audio_preprocessing.hop_length);
}

static void ExtractMFCC(const gui::MLNode& node, TrainingConfiguration& config) {
    config.audio_preprocessing.has_feature_node = true;
    config.audio_preprocessing.feature_type = AudioFeatureType::MFCC;
    if (node.parameters.count("n_mfcc"))
        config.audio_preprocessing.n_mfcc = std::stoi(node.parameters.at("n_mfcc"));
    if (node.parameters.count("n_fft"))
        config.audio_preprocessing.n_fft = std::stoi(node.parameters.at("n_fft"));
    if (node.parameters.count("hop_length"))
        config.audio_preprocessing.hop_length = std::stoi(node.parameters.at("hop_length"));
    if (node.parameters.count("n_mels"))
        config.audio_preprocessing.n_mels = std::stoi(node.parameters.at("n_mels"));
    spdlog::info("GraphCompiler: MFCC n_mfcc={}, n_mels={}, n_fft={}",
                 config.audio_preprocessing.n_mfcc,
                 config.audio_preprocessing.n_mels,
                 config.audio_preprocessing.n_fft);
}

static void ExtractAudioAugmentation(const gui::MLNode& node, TrainingConfiguration& config) {
    config.audio_preprocessing.has_augmentation = true;
    if (node.parameters.count("noise_level"))
        config.audio_preprocessing.noise_level = std::stof(node.parameters.at("noise_level"));
    if (node.parameters.count("time_stretch"))
        config.audio_preprocessing.time_stretch = (node.parameters.at("time_stretch") == "true");
    if (node.parameters.count("pitch_shift"))
        config.audio_preprocessing.pitch_shift = (node.parameters.at("pitch_shift") == "true");
    spdlog::info("GraphCompiler: AudioAugmentation noise={}, time_stretch={}, pitch_shift={}",
                 config.audio_preprocessing.noise_level,
                 config.audio_preprocessing.time_stretch,
                 config.audio_preprocessing.pitch_shift);
}

// --- Text preprocessing (Arrow materializer path) ---
// TextTokenizer is applied by PipelineMaterializer. TextVocabulary and
// TextPadding fold into the reachable tokenizer there instead of populating
// TrainingConfiguration::text_preprocessing.

// --- TimeSeries extractors (Phase 4 — deferred) ---

// --- The table ---

static const PreprocessingNodeSpec kPreprocessingSpecs[] = {
    // Tabular (existing, migrated from switch)
    {gui::NodeType::Normalize,          PreprocessingDomain::Tabular,     ExtractImageNormalize},
    {gui::NodeType::TensorReshape,      PreprocessingDomain::Tabular,     ExtractReshape},
    {gui::NodeType::OneHotEncode,       PreprocessingDomain::Tabular,     ExtractOneHot},
    // General (domain-agnostic data pipeline nodes — no extraction needed)
    {gui::NodeType::DataSplit,          PreprocessingDomain::General,     nullptr},
    {gui::NodeType::DataLoader,         PreprocessingDomain::General,     nullptr},
    // Image (Phase 1)
    {gui::NodeType::Resize,             PreprocessingDomain::Image,       ExtractImageResize},
    {gui::NodeType::CenterCrop,         PreprocessingDomain::Image,       nullptr},
    {gui::NodeType::RandomCrop,         PreprocessingDomain::Image,       nullptr},
    {gui::NodeType::HorizontalFlip,     PreprocessingDomain::Image,       nullptr},
    {gui::NodeType::VerticalFlip,       PreprocessingDomain::Image,       nullptr},
    {gui::NodeType::ImageRotate,        PreprocessingDomain::Image,       nullptr},
    {gui::NodeType::ColorJitter,        PreprocessingDomain::Image,       nullptr},
    {gui::NodeType::ImageGaussianBlur,  PreprocessingDomain::Image,       ExtractImageGaussianBlur},
    {gui::NodeType::Grayscale,          PreprocessingDomain::Image,       ExtractGrayscale},
    {gui::NodeType::Augmentation,       PreprocessingDomain::Image,       nullptr},
    // Audio (Phase 2.1)
    {gui::NodeType::AudioInput,         PreprocessingDomain::Audio,       nullptr},
    {gui::NodeType::Spectrogram,        PreprocessingDomain::Audio,       ExtractSpectrogram},
    {gui::NodeType::MelSpectrogram,     PreprocessingDomain::Audio,       ExtractMelSpectrogram},
    {gui::NodeType::MFCC,               PreprocessingDomain::Audio,       ExtractMFCC},
    {gui::NodeType::AudioAugmentation,  PreprocessingDomain::Audio,       ExtractAudioAugmentation},
    // Text (Arrow materializer path; no TrainingConfiguration extraction)
    {gui::NodeType::TextTokenizer,      PreprocessingDomain::Text,        nullptr},
    {gui::NodeType::TextVocabulary,     PreprocessingDomain::Text,        nullptr},
    {gui::NodeType::TextPadding,        PreprocessingDomain::Text,        nullptr},
    // TimeSeries (Phase 4)
    {gui::NodeType::TimeSeriesWindow,   PreprocessingDomain::TimeSeries,  nullptr},
    {gui::NodeType::TimeSeriesFeatures, PreprocessingDomain::TimeSeries,  nullptr},
    {gui::NodeType::TimeSeriesSplit,    PreprocessingDomain::TimeSeries,  nullptr},
    {gui::NodeType::LogTransform,       PreprocessingDomain::TimeSeries,  nullptr},
    {gui::NodeType::Differencing,       PreprocessingDomain::TimeSeries,  nullptr},
};

} // anonymous namespace

bool GraphCompiler::IsPreprocessing(gui::NodeType type) const {
    for (const auto& spec : kPreprocessingSpecs) {
        if (spec.type == type) return true;
    }
    return false;
}

std::optional<PreprocessingDomain> GraphCompiler::GetPreprocessingNodeDomain(gui::NodeType type) const {
    for (const auto& spec : kPreprocessingSpecs) {
        if (spec.type == type) {
            return spec.domain;
        }
    }
    return std::nullopt;
}

CompiledLayer GraphCompiler::ExtractLayerConfig(const gui::MLNode& node) const {
    CompiledLayer layer;
    layer.type = node.type;
    layer.node_id = node.id;
    layer.name = node.name;
    layer.parameters = node.parameters;

    // Extract specific parameters
    switch (node.type) {
        case gui::NodeType::Dense:
            if (node.parameters.count("units"))
                layer.units = std::stoi(node.parameters.at("units"));
            break;

        case gui::NodeType::Embedding:
            // Embedding params (num_embeddings, embedding_dim) stay in
            // the generic `parameters` map and are read by the training
            // executor. No dedicated fields on CompiledLayer to keep
            // the struct slim; the raw parameter passthrough is enough.
            break;

        case gui::NodeType::Conv2D:
            if (node.parameters.count("filters"))
                layer.filters = std::stoi(node.parameters.at("filters"));
            if (node.parameters.count("kernel_size"))
                layer.kernel_size = std::stoi(node.parameters.at("kernel_size"));
            if (node.parameters.count("stride"))
                layer.stride = std::stoi(node.parameters.at("stride"));
            if (node.parameters.count("padding"))
                layer.padding = std::stoi(node.parameters.at("padding"));
            break;

        case gui::NodeType::MaxPool2D:
        case gui::NodeType::AvgPool2D:
            if (node.parameters.count("pool_size"))
                layer.pool_size = std::stoi(node.parameters.at("pool_size"));
            if (node.parameters.count("stride"))
                layer.stride = std::stoi(node.parameters.at("stride"));
            break;

        case gui::NodeType::BatchNorm:
            if (node.parameters.count("eps"))
                layer.eps = std::stof(node.parameters.at("eps"));
            if (node.parameters.count("momentum"))
                layer.momentum = std::stof(node.parameters.at("momentum"));
            break;

        case gui::NodeType::Dropout:
            if (node.parameters.count("rate"))
                layer.dropout_rate = std::stof(node.parameters.at("rate"));
            break;

        case gui::NodeType::LeakyReLU:
            if (node.parameters.count("negative_slope"))
                layer.negative_slope = std::stof(node.parameters.at("negative_slope"));
            break;

        case gui::NodeType::ELU:
            if (node.parameters.count("alpha"))
                layer.alpha = std::stof(node.parameters.at("alpha"));
            break;

        case gui::NodeType::ConvTranspose2D:
            if (node.parameters.count("in_channels"))
                layer.in_channels = std::stoi(node.parameters.at("in_channels"));
            if (node.parameters.count("out_channels"))
                layer.filters = std::stoi(node.parameters.at("out_channels"));
            if (node.parameters.count("kernel_size"))
                layer.kernel_size = std::stoi(node.parameters.at("kernel_size"));
            if (node.parameters.count("stride"))
                layer.stride = std::stoi(node.parameters.at("stride"));
            if (node.parameters.count("padding"))
                layer.padding = std::stoi(node.parameters.at("padding"));
            if (node.parameters.count("output_padding"))
                layer.output_padding = std::stoi(node.parameters.at("output_padding"));
            break;

        case gui::NodeType::Upsample:
            if (node.parameters.count("scale_factor"))
                layer.scale_factor = std::stoi(node.parameters.at("scale_factor"));
            if (node.parameters.count("mode"))
                layer.upsample_mode = std::stoi(node.parameters.at("mode"));
            break;

        case gui::NodeType::PixelShuffle:
            if (node.parameters.count("upscale_factor"))
                layer.scale_factor = std::stoi(node.parameters.at("upscale_factor"));
            break;

        default:
            break;
    }

    return layer;
}

void GraphCompiler::ExtractPreprocessing(
    const gui::MLNode& node,
    TrainingConfiguration& config) const
{
    for (const auto& spec : kPreprocessingSpecs) {
        if (spec.type == node.type) {
            if (spec.extractor) {
                spec.extractor(node, config);
            }
            return;
        }
    }
}

std::vector<size_t> GraphCompiler::InferOutputShape(
    const CompiledLayer& layer,
    const std::vector<size_t>& input_shape) const
{
    std::vector<size_t> output_shape;

    switch (layer.type) {
        case gui::NodeType::Dense:
            // Dense: [...] -> [units]
            output_shape = {static_cast<size_t>(layer.units)};
            break;

        case gui::NodeType::Embedding: {
            // Embedding: [seq_len] -> [seq_len, embedding_dim]
            // Input shape is the sequence length (from text dataset
            // max_length). Output adds the embedding_dim channel so a
            // downstream Flatten produces seq_len * embedding_dim.
            int embed_dim = 64;
            auto it = layer.parameters.find("embedding_dim");
            if (it != layer.parameters.end()) {
                try { embed_dim = std::stoi(it->second); } catch (...) {}
            }
            if (!input_shape.empty()) {
                output_shape = {input_shape[0],
                                static_cast<size_t>(embed_dim)};
            } else {
                output_shape = {static_cast<size_t>(embed_dim)};
            }
            break;
        }

        case gui::NodeType::LSTM:
        case gui::NodeType::GRU:
        case gui::NodeType::RNN: {
            const size_t hidden_size =
                ParseSizeParam(layer.parameters, "hidden_size", 128);
            const bool bidirectional =
                ParseBoolParam(layer.parameters, "bidirectional", false);
            const bool return_sequences =
                ParseBoolParam(layer.parameters, "return_sequences", false);
            const size_t output_features =
                hidden_size * (bidirectional ? 2 : 1);

            if (return_sequences) {
                const size_t seq_len = input_shape.empty() ? 1 : input_shape[0];
                output_shape = {seq_len, output_features};
            } else {
                output_shape = {output_features};
            }
            break;
        }

        case gui::NodeType::Conv2D:
            // Conv2D: [H, W, C] -> [(H + 2*padding - kernel_size) / stride + 1, W', filters]
            if (input_shape.size() >= 2) {
                size_t out_h = (input_shape[0] + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;
                size_t out_w = (input_shape[1] + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;
                output_shape = {out_h, out_w, static_cast<size_t>(layer.filters)};
            }
            break;

        case gui::NodeType::MaxPool2D:
        case gui::NodeType::AvgPool2D:
            // Pool2D: [H, W, C] -> [H/pool_size, W/pool_size, C]
            if (input_shape.size() >= 3) {
                int stride = layer.stride > 0 ? layer.stride : layer.pool_size;
                size_t out_h = (input_shape[0] - layer.pool_size) / stride + 1;
                size_t out_w = (input_shape[1] - layer.pool_size) / stride + 1;
                output_shape = {out_h, out_w, input_shape[2]};
            }
            break;

        case gui::NodeType::GlobalMaxPool:
        case gui::NodeType::GlobalAvgPool:
            // Global pooling: [H, W, C] -> [C]
            if (input_shape.size() >= 3) {
                output_shape = {input_shape[2]};  // Just channels remain
            } else if (!input_shape.empty()) {
                output_shape = {input_shape.back()};
            }
            break;

        case gui::NodeType::Flatten:
            // Flatten: [H, W, C] -> [H*W*C]
            {
                size_t flat_size = 1;
                for (size_t dim : input_shape) flat_size *= dim;
                output_shape = {flat_size};
            }
            break;

        case gui::NodeType::Reshape:
        case gui::NodeType::View:
        case gui::NodeType::Permute:
        case gui::NodeType::Squeeze:
        case gui::NodeType::Unsqueeze:
        case gui::NodeType::TensorBroadcastTo:
        case gui::NodeType::TensorExpand:
        case gui::NodeType::TensorIndexSelect: {
            std::string error;
            std::vector<int> ignored_dims;
            const bool ok = layer.type == gui::NodeType::Permute
                ? ResolvePermuteTargetShape(layer.parameters,
                                            input_shape,
                                            output_shape,
                                            ignored_dims,
                                            error)
                : ResolveShapeOpTargetShape(layer.type,
                                            layer.parameters,
                                            input_shape,
                                            output_shape,
                                            error);
            if (!ok) {
                spdlog::warn("GraphCompiler: {}", error);
                output_shape = input_shape;
            }
            break;
        }

        case gui::NodeType::TensorSum:
        case gui::NodeType::TensorMean:
        case gui::NodeType::TensorMax:
        case gui::NodeType::TensorMin:
        case gui::NodeType::TensorProd:
        case gui::NodeType::TensorVar:
        case gui::NodeType::TensorStd: {
            std::string error;
            if (!ResolveReductionTargetShape(layer.type,
                                             layer.parameters,
                                             input_shape,
                                             output_shape,
                                             error)) {
                spdlog::warn("GraphCompiler: {}", error);
                output_shape = input_shape;
            }
            break;
        }

        // Layers/activations that don't change shape
        case gui::NodeType::Dropout:
        case gui::NodeType::BatchNorm:
        case gui::NodeType::ReLU:
        case gui::NodeType::LeakyReLU:
        case gui::NodeType::ELU:
        case gui::NodeType::GELU:
        case gui::NodeType::Swish:
        case gui::NodeType::Mish:
        case gui::NodeType::Sigmoid:
        case gui::NodeType::Tanh:
        case gui::NodeType::Softmax:
        case gui::NodeType::TensorAbs:
        case gui::NodeType::TensorExp:
        case gui::NodeType::TensorLog:
        case gui::NodeType::TensorSqrt:
        case gui::NodeType::TensorSign:
        case gui::NodeType::TensorPow:
        case gui::NodeType::TensorClip:
        case gui::NodeType::TensorCompare:
        case gui::NodeType::TensorLogicalMask:
            output_shape = input_shape;
            break;

        default:
            output_shape = input_shape;
            break;
    }

    return output_shape;
}

bool GraphCompiler::HasCycle(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links) const
{
    // Build adjacency list
    std::map<int, std::vector<int>> adj;
    for (const auto& node : nodes) {
        adj[node.id] = {};
    }
    for (const auto& link : links) {
        adj[link.from_node].push_back(link.to_node);
    }

    // DFS-based cycle detection
    std::set<int> white;  // Not visited
    std::set<int> gray;   // Currently visiting
    std::set<int> black;  // Fully processed

    for (const auto& node : nodes) {
        white.insert(node.id);
    }

    std::function<bool(int)> dfs = [&](int node_id) -> bool {
        white.erase(node_id);
        gray.insert(node_id);

        for (int neighbor : adj[node_id]) {
            if (gray.count(neighbor)) {
                // Back edge found - cycle exists
                return true;
            }
            if (white.count(neighbor) && dfs(neighbor)) {
                return true;
            }
        }

        gray.erase(node_id);
        black.insert(node_id);
        return false;
    };

    while (!white.empty()) {
        int node_id = *white.begin();
        if (dfs(node_id)) {
            return true;
        }
    }

    return false;
}

bool GraphCompiler::IsFullyConnected(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links) const
{
    if (nodes.empty()) return true;

    // BFS from first node
    std::set<int> visited;
    std::queue<int> queue;

    // Build undirected adjacency
    std::map<int, std::set<int>> adj;
    for (const auto& node : nodes) {
        adj[node.id] = {};
    }
    for (const auto& link : links) {
        adj[link.from_node].insert(link.to_node);
        adj[link.to_node].insert(link.from_node);
    }

    queue.push(nodes[0].id);
    visited.insert(nodes[0].id);

    while (!queue.empty()) {
        int node_id = queue.front();
        queue.pop();

        for (int neighbor : adj[node_id]) {
            if (!visited.count(neighbor)) {
                visited.insert(neighbor);
                queue.push(neighbor);
            }
        }
    }

    return visited.size() == nodes.size();
}

// ---------------------------------------------------------------------------
// Pin-connectivity validation
//
// The runtime path (TrainingExecutor) currently bypasses graph topology
// and reads the dataset by name from DataRegistry, so a graph with a
// disconnected Loss.Targets pin will train anyway and silently produce
// nonsense gradients. The architectural fix is the multi-day "walk
// pins, not registry" rewrite tracked in tofix.md. Until that lands,
// these compile-time checks make the canvas the source of truth: the
// user can't ship a graph whose required pins are unwired or whose
// label stream never reaches the loss.
//
// The checks here are deliberately conservative — they only flag
// situations that are unambiguously wrong. Anything that depends on
// runtime semantics (e.g. "this Tensor pin produces a label-shaped
// tensor in practice") is left alone to avoid false positives.
// ---------------------------------------------------------------------------

void GraphCompiler::ValidateRequiredInputsConnected(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    TrainingConfiguration& config) const
{
    // Build the set of input pin IDs that have at least one incoming link.
    std::unordered_set<int> connected_input_pins;
    connected_input_pins.reserve(links.size());
    for (const auto& link : links) {
        connected_input_pins.insert(link.to_pin);
    }

    // Walk every node's input pins and flag required ones with no incoming
    // link. Output pins are skipped intentionally — the existing pin
    // metadata defaults `is_required=true` on every output, which would
    // generate false positives for legitimately-optional outputs (LSTM's
    // Hidden, MultiHeadAttention's Attn Weights, DataSplit's Val/Test
    // streams when the user only trains). When those get explicit
    // `is_required=false` markers, output validation can join.
    for (const auto& node : nodes) {
        for (const auto& pin : node.inputs) {
            if (!pin.is_required) continue;
            if (connected_input_pins.count(pin.id) > 0) continue;

            std::ostringstream msg;
            msg << "Required input pin '" << pin.name
                << "' on node '" << node.name
                << "' has no incoming connection";
            AddIssue(config, IssueLevel::Error, msg.str(),
                     node.id, node.name);
        }
    }
}

void GraphCompiler::ValidateRequiredOutputsConnected(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    TrainingConfiguration& config) const
{
    // Output-side symmetry of ValidateRequiredInputsConnected. Catches
    // "dangling" producers: a DataLoader whose Labels output goes
    // nowhere, a Normalize whose Output isn't consumed, etc. Only
    // outputs that were explicitly marked is_required=false (LSTM's
    // Hidden, attention's Attn Weights, DataSplit's Val/Test,
    // Output.Predictions, Optimizer.State) are skipped — everything
    // else must have at least one downstream consumer.
    std::unordered_set<int> connected_output_pins;
    connected_output_pins.reserve(links.size());
    for (const auto& link : links) {
        connected_output_pins.insert(link.from_pin);
    }

    for (const auto& node : nodes) {
        for (const auto& pin : node.outputs) {
            if (!pin.is_required) continue;
            if (connected_output_pins.count(pin.id) > 0) continue;

            std::ostringstream msg;
            msg << "Required output pin '" << pin.name
                << "' on node '" << node.name
                << "' has no outgoing connection - the produced data is "
                   "never consumed";
            AddIssue(config, IssueLevel::Error, msg.str(),
                     node.id, node.name);
        }
    }
}

void GraphCompiler::ValidateLossTargetsReachLabels(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    TrainingConfiguration& config) const
{
    // Build a quick (node_id, pin_id) → pin lookup so we can recover pin
    // type during the BFS. NodePin IDs are unique across the graph, so a
    // flat pin_id → pointer map is enough.
    std::unordered_map<int, const gui::NodePin*> pin_by_id;
    std::unordered_map<int, int> pin_to_node;  // pin_id → owning node_id
    for (const auto& node : nodes) {
        for (const auto& pin : node.inputs) {
            pin_by_id[pin.id] = &pin;
            pin_to_node[pin.id] = node.id;
        }
        for (const auto& pin : node.outputs) {
            pin_by_id[pin.id] = &pin;
            pin_to_node[pin.id] = node.id;
        }
    }

    // Reverse adjacency: input pin ID → list of upstream output pin IDs
    // feeding it. Lets us walk backwards from any input pin.
    std::unordered_map<int, std::vector<int>> upstream_outputs;
    upstream_outputs.reserve(links.size());
    for (const auto& link : links) {
        upstream_outputs[link.to_pin].push_back(link.from_pin);
    }

    // Node-id → vector<input pin id> so that when BFS arrives at a node
    // (via an output pin), we can step into its inputs and continue the
    // walk upstream through the node's own inputs.
    std::unordered_map<int, std::vector<int>> node_input_pins;
    node_input_pins.reserve(nodes.size());
    for (const auto& node : nodes) {
        std::vector<int> ids;
        ids.reserve(node.inputs.size());
        for (const auto& pin : node.inputs) ids.push_back(pin.id);
        node_input_pins[node.id] = std::move(ids);
    }

    auto is_loss_node = [](gui::NodeType t) {
        return t == gui::NodeType::MSELoss ||
               t == gui::NodeType::CrossEntropyLoss ||
               t == gui::NodeType::BCELoss ||
               t == gui::NodeType::BCEWithLogits ||
               t == gui::NodeType::L1Loss ||
               t == gui::NodeType::SmoothL1Loss ||
               t == gui::NodeType::HuberLoss ||
               t == gui::NodeType::NLLLoss;
    };

    // For each loss node, find its Targets input pin (the one tagged
    // PinType::Labels) and BFS backwards. If no ancestor output pin is
    // PinType::Labels, the targets stream isn't real labels — the user
    // wired Targets to something else (a model layer's Tensor output,
    // a random preprocessing tensor, etc.). That's the canonical "pin
    // is fooling user" case.
    for (const auto& node : nodes) {
        if (!is_loss_node(node.type)) continue;

        // Identify Targets pin. Every loss node above creates the
        // Targets input as PinType::Labels (see node_editor_nodes.cpp);
        // older graphs may have it as PinType::Tensor — fall back to
        // matching by name.
        const gui::NodePin* targets_pin = nullptr;
        for (const auto& pin : node.inputs) {
            if (pin.type == gui::PinType::Labels) {
                targets_pin = &pin;
                break;
            }
        }
        if (!targets_pin) {
            for (const auto& pin : node.inputs) {
                if (pin.name == "Targets") {
                    targets_pin = &pin;
                    break;
                }
            }
        }
        if (!targets_pin) continue;  // Loss with no Targets input — can't check.

        // If the Targets pin is unconnected, the required-input check
        // will have already flagged it; skip to avoid duplicate errors.
        if (upstream_outputs.find(targets_pin->id) == upstream_outputs.end()) {
            continue;
        }

        // BFS backwards from targets_pin. State is the queue of input
        // pin IDs to walk upstream from. visited tracks visited input
        // pin IDs to bound the walk.
        std::queue<int> queue;
        std::unordered_set<int> visited_inputs;
        queue.push(targets_pin->id);
        visited_inputs.insert(targets_pin->id);

        bool found_labels_source = false;
        while (!queue.empty() && !found_labels_source) {
            int input_pin_id = queue.front();
            queue.pop();

            auto it = upstream_outputs.find(input_pin_id);
            if (it == upstream_outputs.end()) continue;

            for (int upstream_out_pin_id : it->second) {
                auto pin_it = pin_by_id.find(upstream_out_pin_id);
                if (pin_it == pin_by_id.end()) continue;
                if (pin_it->second->type == gui::PinType::Labels) {
                    found_labels_source = true;
                    break;
                }
                // Not a Labels source itself — keep walking upstream
                // through this output pin's owning node's inputs.
                auto owner_it = pin_to_node.find(upstream_out_pin_id);
                if (owner_it == pin_to_node.end()) continue;
                auto inputs_it = node_input_pins.find(owner_it->second);
                if (inputs_it == node_input_pins.end()) continue;
                for (int upstream_in_pin_id : inputs_it->second) {
                    if (visited_inputs.insert(upstream_in_pin_id).second) {
                        queue.push(upstream_in_pin_id);
                    }
                }
            }
        }

        if (!found_labels_source) {
            std::ostringstream msg;
            msg << "Loss node '" << node.name
                << "' has its Targets pin wired, but the upstream chain "
                   "never passes through a Labels-typed pin "
                   "(DataInput.Labels, DataSplit.*Labels, or "
                   "DataLoader.Labels). The model is being trained "
                   "against the wrong stream.";
            AddIssue(config, IssueLevel::Error, msg.str(),
                     node.id, node.name);
        }
    }
}

void GraphCompiler::ValidateLossPredictionsReachModel(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    TrainingConfiguration& config) const
{
    // Symmetric to ValidateLossTargetsReachLabels: BFS upstream from
    // each Loss node's Predictions pin, but the success criterion is
    // "ancestor node is a model layer or Output node" instead of
    // "ancestor pin is PinType::Labels". Catches the case where a user
    // wires Predictions to DataInput.Data, a Normalize output, or
    // (worst) the Labels stream — all of which would silently train
    // garbage before the runtime fix lands.
    std::unordered_map<int, int> pin_to_node;  // pin_id → owning node_id
    for (const auto& node : nodes) {
        for (const auto& pin : node.inputs)  pin_to_node[pin.id] = node.id;
        for (const auto& pin : node.outputs) pin_to_node[pin.id] = node.id;
    }

    std::unordered_map<int, std::vector<int>> upstream_outputs;
    upstream_outputs.reserve(links.size());
    for (const auto& link : links) {
        upstream_outputs[link.to_pin].push_back(link.from_pin);
    }

    std::unordered_map<int, std::vector<int>> node_input_pins;
    node_input_pins.reserve(nodes.size());
    for (const auto& node : nodes) {
        std::vector<int> ids;
        ids.reserve(node.inputs.size());
        for (const auto& pin : node.inputs) ids.push_back(pin.id);
        node_input_pins[node.id] = std::move(ids);
    }

    auto is_loss_node = [](gui::NodeType t) {
        return t == gui::NodeType::MSELoss ||
               t == gui::NodeType::CrossEntropyLoss ||
               t == gui::NodeType::BCELoss ||
               t == gui::NodeType::BCEWithLogits ||
               t == gui::NodeType::L1Loss ||
               t == gui::NodeType::SmoothL1Loss ||
               t == gui::NodeType::HuberLoss ||
               t == gui::NodeType::NLLLoss;
    };

    // Build node_id → node* lookup so the BFS can check owning node
    // type without a linear search per visited pin.
    std::unordered_map<int, const gui::MLNode*> node_by_id;
    node_by_id.reserve(nodes.size());
    for (const auto& node : nodes) node_by_id[node.id] = &node;

    for (const auto& node : nodes) {
        if (!is_loss_node(node.type)) continue;

        // Predictions is the FIRST input on every loss node above —
        // it's PinType::Tensor. Match by name "Predictions" for older
        // graphs that may not preserve ordering.
        const gui::NodePin* predictions_pin = nullptr;
        for (const auto& pin : node.inputs) {
            if (pin.name == "Predictions") {
                predictions_pin = &pin;
                break;
            }
        }
        if (!predictions_pin && !node.inputs.empty()) {
            // Fall back to the first input pin — older loss nodes may
            // not have set the name explicitly.
            predictions_pin = &node.inputs[0];
        }
        if (!predictions_pin) continue;

        // Already-flagged disconnect — skip to avoid duplicate errors.
        if (upstream_outputs.find(predictions_pin->id) == upstream_outputs.end()) {
            continue;
        }

        std::queue<int> queue;
        std::unordered_set<int> visited_inputs;
        queue.push(predictions_pin->id);
        visited_inputs.insert(predictions_pin->id);

        bool found_model_source = false;
        while (!queue.empty() && !found_model_source) {
            int input_pin_id = queue.front();
            queue.pop();

            auto it = upstream_outputs.find(input_pin_id);
            if (it == upstream_outputs.end()) continue;

            for (int upstream_out_pin_id : it->second) {
                auto owner_it = pin_to_node.find(upstream_out_pin_id);
                if (owner_it == pin_to_node.end()) continue;
                auto node_it = node_by_id.find(owner_it->second);
                if (node_it == node_by_id.end()) continue;

                const gui::MLNode* upstream_node = node_it->second;
                if (upstream_node->type == gui::NodeType::Output ||
                    IsModelLayer(upstream_node->type)) {
                    found_model_source = true;
                    break;
                }

                // Activation nodes (ReLU/Sigmoid/etc.) are NOT in
                // IsModelLayer but a chain ending Activation → Loss
                // is legitimate (the Activation's own input traces
                // back to a model layer). Keep walking upstream
                // through this node's inputs.
                auto inputs_it = node_input_pins.find(owner_it->second);
                if (inputs_it == node_input_pins.end()) continue;
                for (int upstream_in_pin_id : inputs_it->second) {
                    if (visited_inputs.insert(upstream_in_pin_id).second) {
                        queue.push(upstream_in_pin_id);
                    }
                }
            }
        }

        if (!found_model_source) {
            std::ostringstream msg;
            msg << "Loss node '" << node.name
                << "' has its Predictions pin wired, but the upstream "
                   "chain never passes through a model layer or Output "
                   "node. The loss is being computed against a "
                   "non-prediction tensor (often the Labels stream "
                   "wired by mistake, or a raw DataInput tensor with "
                   "no model in between).";
            AddIssue(config, IssueLevel::Error, msg.str(),
                     node.id, node.name);
        }
    }
}

void GraphCompiler::ValidateOptimizerReachesLoss(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    TrainingConfiguration& config) const
{
    // Closes the last visible pin vector in the compile → loss → optimizer
    // chain. The pin-type system's "Tensor is universal" rule means the
    // user CAN wire a Dense.Output (Tensor) directly into Optimizer.Loss
    // (Loss) — the ImNodes link validation won't block it. The required-
    // input check then passes because the pin has SOME incoming wire.
    // Without this BFS, a graph with Optimizer.Loss fed by a random
    // tensor would train with nonsense gradients.
    std::unordered_map<int, int> pin_to_node;
    for (const auto& node : nodes) {
        for (const auto& pin : node.inputs)  pin_to_node[pin.id] = node.id;
        for (const auto& pin : node.outputs) pin_to_node[pin.id] = node.id;
    }

    std::unordered_map<int, std::vector<int>> upstream_outputs;
    upstream_outputs.reserve(links.size());
    for (const auto& link : links) {
        upstream_outputs[link.to_pin].push_back(link.from_pin);
    }

    std::unordered_map<int, std::vector<int>> node_input_pins;
    node_input_pins.reserve(nodes.size());
    for (const auto& node : nodes) {
        std::vector<int> ids;
        ids.reserve(node.inputs.size());
        for (const auto& pin : node.inputs) ids.push_back(pin.id);
        node_input_pins[node.id] = std::move(ids);
    }

    std::unordered_map<int, const gui::MLNode*> node_by_id;
    node_by_id.reserve(nodes.size());
    for (const auto& node : nodes) node_by_id[node.id] = &node;

    auto is_optimizer = [](gui::NodeType t) {
        return t == gui::NodeType::SGD ||
               t == gui::NodeType::Adam ||
               t == gui::NodeType::AdamW ||
               t == gui::NodeType::RMSprop ||
               t == gui::NodeType::Adagrad ||
               t == gui::NodeType::NAdam;
    };

    auto is_loss_node = [](gui::NodeType t) {
        return t == gui::NodeType::MSELoss ||
               t == gui::NodeType::CrossEntropyLoss ||
               t == gui::NodeType::BCELoss ||
               t == gui::NodeType::BCEWithLogits ||
               t == gui::NodeType::L1Loss ||
               t == gui::NodeType::SmoothL1Loss ||
               t == gui::NodeType::HuberLoss ||
               t == gui::NodeType::NLLLoss;
    };

    for (const auto& node : nodes) {
        if (!is_optimizer(node.type)) continue;

        // Optimizer nodes have a single Loss-typed input pin. Match by
        // PinType first (cleanest) with a name fallback for older graphs.
        const gui::NodePin* loss_pin = nullptr;
        for (const auto& pin : node.inputs) {
            if (pin.type == gui::PinType::Loss) { loss_pin = &pin; break; }
        }
        if (!loss_pin) {
            for (const auto& pin : node.inputs) {
                if (pin.name == "Loss") { loss_pin = &pin; break; }
            }
        }
        if (!loss_pin) continue;

        // Disconnect is handled by the required-input check.
        if (upstream_outputs.find(loss_pin->id) == upstream_outputs.end()) {
            continue;
        }

        std::queue<int> queue;
        std::unordered_set<int> visited_inputs;
        queue.push(loss_pin->id);
        visited_inputs.insert(loss_pin->id);

        bool found_loss_node = false;
        while (!queue.empty() && !found_loss_node) {
            int input_pin_id = queue.front();
            queue.pop();

            auto it = upstream_outputs.find(input_pin_id);
            if (it == upstream_outputs.end()) continue;

            for (int upstream_out_pin_id : it->second) {
                auto owner_it = pin_to_node.find(upstream_out_pin_id);
                if (owner_it == pin_to_node.end()) continue;
                auto node_it = node_by_id.find(owner_it->second);
                if (node_it == node_by_id.end()) continue;

                if (is_loss_node(node_it->second->type)) {
                    found_loss_node = true;
                    break;
                }

                // Not a loss node — keep walking. Unlikely to hit in
                // practice (optimizer is normally one hop from a loss)
                // but the walk stays general for the edge case where
                // the user inserted something between loss and
                // optimizer.
                auto inputs_it = node_input_pins.find(owner_it->second);
                if (inputs_it == node_input_pins.end()) continue;
                for (int upstream_in_pin_id : inputs_it->second) {
                    if (visited_inputs.insert(upstream_in_pin_id).second) {
                        queue.push(upstream_in_pin_id);
                    }
                }
            }
        }

        if (!found_loss_node) {
            std::ostringstream msg;
            msg << "Optimizer node '" << node.name
                << "' has its Loss pin wired, but the upstream chain "
                   "never passes through a loss function (MSELoss, "
                   "CrossEntropyLoss, etc.). The optimizer will "
                   "backprop from a non-loss tensor — training is "
                   "meaningless.";
            AddIssue(config, IssueLevel::Error, msg.str(),
                     node.id, node.name);
        }
    }
}

} // namespace cyxwiz
