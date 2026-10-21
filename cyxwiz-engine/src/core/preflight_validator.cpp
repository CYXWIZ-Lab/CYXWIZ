#include "preflight_validator.h"
#include "error_codes.h"

#include <algorithm>
#include <filesystem>
#include <sstream>

namespace cyxwiz {

namespace {

void AddIssue(DebugPreflightResult& result, IssueLevel level,
              const std::string& message, int node_id = -1,
              const std::string& node_name = "",
              const std::string& error_code = "") {
    result.issues.push_back({
        level,
        node_id,
        node_name,
        message,
        error_code.empty() ? errors::Training::InvalidTrainingSetup
                           : error_code
    });
}

const char* DomainName(PreprocessingDomain domain) {
    switch (domain) {
        case PreprocessingDomain::Tabular: return "Tabular";
        case PreprocessingDomain::Image: return "Image";
        case PreprocessingDomain::Audio: return "Audio";
        case PreprocessingDomain::Text: return "Text";
        case PreprocessingDomain::TimeSeries: return "TimeSeries";
        case PreprocessingDomain::General: return "General";
    }
    return "Unknown";
}

bool TryReadIntParam(const gui::MLNode& node,
                     const std::string& key,
                     int& out) {
    auto it = node.parameters.find(key);
    if (it == node.parameters.end() || it->second.empty()) {
        return false;
    }
    try {
        out = std::stoi(it->second);
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace

DebugPreflightResult PreflightValidator::Validate(
    const TrainingConfiguration& config,
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    uint64_t graph_hash) const {
    DebugPreflightResult result;

    for (const auto& issue : config.issues) {
        result.issues.push_back(issue);
    }

    if (nodes.empty()) {
        AddIssue(result, IssueLevel::Error, "Graph is empty.",
                 -1, "", errors::Compiler::MissingTrainingPathNode);
    }

    if (config.dataset_name.empty()) {
        AddIssue(result, IssueLevel::Error, "No dataset is selected or loaded.",
                 -1, "", errors::Runtime::InputDatasetMissing);
    }

    if (config.layers.empty()) {
        AddIssue(result, IssueLevel::Error, "No executable model layers were compiled.",
                 -1, "", errors::Compiler::MissingTrainingPathNode);
    }

    if (config.input_size == 0 && config.input_shape.empty()) {
        AddIssue(result, IssueLevel::Error, "Model input shape is unknown.",
                 -1, "", errors::Compiler::TensorShapeMismatch);
    }

    if (config.output_size == 0) {
        AddIssue(result, IssueLevel::Error, "Model output size is unknown.",
                 -1, "", errors::Compiler::LabelOutputShapeMismatch);
    }

    if (config.loss_type == gui::NodeType::CrossEntropyLoss &&
        config.preprocessing.num_classes == 0 &&
        config.output_size == 0) {
        AddIssue(result, IssueLevel::Error,
                 "CrossEntropy requires a known class count.",
                 -1, "", errors::Compiler::LabelOutputShapeMismatch);
    }

    if (config.preprocessing_domain == PreprocessingDomain::Text) {
        for (const auto& node : nodes) {
            if (node.type == gui::NodeType::TextVocabulary) {
                auto vocab_file = node.parameters.find("vocab_file");
                if (vocab_file != node.parameters.end() &&
                    !vocab_file->second.empty() &&
                    !std::filesystem::exists(vocab_file->second)) {
                    AddIssue(result, IssueLevel::Error,
                             "TextVocabulary vocab_file does not exist: " +
                             vocab_file->second,
                             node.id, node.name,
                             errors::File::NotFound);
                }

                int max_vocab_size = -1;
                if (TryReadIntParam(node, "max_vocab_size", max_vocab_size) &&
                    max_vocab_size == 0) {
                    AddIssue(result, IssueLevel::Warning,
                             "TextVocabulary max_vocab_size is zero; vocabulary may be unusable.",
                             node.id, node.name,
                             errors::Compiler::InvalidParameter);
                }
            } else if (node.type == gui::NodeType::TextPadding) {
                int max_length = 0;
                if (TryReadIntParam(node, "max_length", max_length) &&
                    max_length <= 0) {
                    AddIssue(result, IssueLevel::Error,
                             "TextPadding max_length must be greater than zero.",
                             node.id, node.name,
                             errors::Compiler::InvalidParameter);
                }
            }
        }
    }

    const bool has_error = std::any_of(
        result.issues.begin(), result.issues.end(),
        [](const ValidationIssue& issue) {
            return issue.level == IssueLevel::Error;
        });
    result.ready = !has_error;

    std::ostringstream out;
    out << "Preflight " << (result.ready ? "ready" : "blocked") << "\n";
    out << "Graph hash: 0x" << std::hex << graph_hash << std::dec << "\n";
    out << "Nodes: " << nodes.size() << "  Links: " << links.size() << "\n";
    out << "Dataset: " << (config.dataset_name.empty() ? "<none>" : config.dataset_name) << "\n";
    out << "Domain: " << DomainName(config.preprocessing_domain) << "\n";
    out << "Layers: " << config.layers.size() << "\n";
    out << "Input size: " << config.input_size << "\n";
    out << "Output size: " << config.output_size << "\n";
    out << "Issues: " << result.issues.size() << "\n";
    result.summary = out.str();

    return result;
}

} // namespace cyxwiz
