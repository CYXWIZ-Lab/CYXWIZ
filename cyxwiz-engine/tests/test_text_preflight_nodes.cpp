#include "../src/core/preflight_validator.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

gui::MLNode MakeNode(int id, gui::NodeType type, const std::string& name) {
    gui::MLNode node;
    node.id = id;
    node.type = type;
    node.name = name;
    return node;
}

bool HasIssue(const cyxwiz::DebugPreflightResult& result,
              cyxwiz::IssueLevel level,
              const std::string& text) {
    for (const auto& issue : result.issues) {
        if (issue.level == level &&
            issue.message.find(text) != std::string::npos) {
            return true;
        }
    }
    return false;
}

} // namespace

int main() {
    cyxwiz::TrainingConfiguration config;
    config.dataset_name = "text_data";
    config.input_size = 4;
    config.input_shape = {4};
    config.output_size = 2;
    config.preprocessing_domain = cyxwiz::PreprocessingDomain::Text;

    auto data = MakeNode(1, gui::NodeType::DataInput, "Data Input");
    auto tokenizer = MakeNode(2, gui::NodeType::TextTokenizer, "Text Tokenizer");
    auto vocab = MakeNode(3, gui::NodeType::TextVocabulary, "Text Vocabulary");
    const auto missing_vocab =
        std::filesystem::temp_directory_path() /
        "cyxwiz_missing_preflight_vocab.txt";
    std::filesystem::remove(missing_vocab);
    vocab.parameters = {
        {"vocab_file", missing_vocab.string()},
        {"max_vocab_size", "0"},
    };
    auto padding = MakeNode(4, gui::NodeType::TextPadding, "Text Padding");
    padding.parameters = {
        {"max_length", "0"},
    };
    auto dense = MakeNode(5, gui::NodeType::Dense, "Dense");

    std::vector<gui::MLNode> nodes = {
        data, tokenizer, vocab, padding, dense,
    };
    std::vector<gui::NodeLink> links;

    cyxwiz::PreflightValidator validator;
    auto result = validator.Validate(config, nodes, links, 0x1234);

    Check(HasIssue(result, cyxwiz::IssueLevel::Error,
                   "TextVocabulary vocab_file does not exist"),
          "missing vocab_file should be a preflight error");
    Check(HasIssue(result, cyxwiz::IssueLevel::Warning,
                   "TextVocabulary max_vocab_size is zero"),
          "zero max_vocab_size should be a preflight warning");
    Check(HasIssue(result, cyxwiz::IssueLevel::Error,
                   "TextPadding max_length must be greater than zero"),
          "bad TextPadding max_length should be a preflight error");

    Check(!config.text_preprocessing.has_vocabulary_node,
          "preflight should not depend on text_preprocessing flags");
    Check(!config.text_preprocessing.has_padding_node,
          "preflight should not depend on text_preprocessing padding flag");

    std::cout << "Text preflight node validation passed\n";
    return 0;
}
