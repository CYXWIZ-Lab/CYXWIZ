#include "preflight_validator.h"

#include <algorithm>
#include <filesystem>
#include <sstream>

namespace cyxwiz {

namespace {

void AddIssue(DebugPreflightResult& result, IssueLevel level,
              const std::string& message, int node_id = -1,
              const std::string& node_name = "") {
    result.issues.push_back({level, node_id, node_name, message});
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
        AddIssue(result, IssueLevel::Error, "Graph is empty.");
    }

    if (config.dataset_name.empty()) {
        AddIssue(result, IssueLevel::Error, "No dataset is selected or loaded.");
    }

    if (config.layers.empty()) {
        AddIssue(result, IssueLevel::Error, "No executable model layers were compiled.");
    }

    if (config.input_size == 0 && config.input_shape.empty()) {
        AddIssue(result, IssueLevel::Error, "Model input shape is unknown.");
    }

    if (config.output_size == 0) {
        AddIssue(result, IssueLevel::Error, "Model output size is unknown.");
    }

    if (config.loss_type == gui::NodeType::CrossEntropyLoss &&
        config.preprocessing.num_classes == 0 &&
        config.output_size == 0) {
        AddIssue(result, IssueLevel::Error,
                 "CrossEntropy requires a known class count.");
    }

    if (config.preprocessing_domain == PreprocessingDomain::Text) {
        const auto& text = config.text_preprocessing;
        if (text.has_vocabulary_node && !text.vocab_file.empty() &&
            !std::filesystem::exists(text.vocab_file)) {
            AddIssue(result, IssueLevel::Error,
                     "TextVocabulary vocab_file does not exist: " + text.vocab_file);
        }

        if (text.has_padding_node && text.max_length <= 0) {
            AddIssue(result, IssueLevel::Error,
                     "TextPadding max_length must be greater than zero.");
        }

        if (text.has_vocabulary_node && text.max_vocab_size == 0) {
            AddIssue(result, IssueLevel::Warning,
                     "TextVocabulary max_vocab_size is zero; vocabulary may be unusable.");
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
