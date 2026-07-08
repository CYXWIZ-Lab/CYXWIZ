#include "../src/gui/properties_truth.h"

#include <cstdlib>
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

bool HasStatus(const gui::properties_truth::PropertyTruth& truth,
               gui::properties_truth::TruthStatus status) {
    for (const auto actual : truth.statuses) {
        if (actual == status) {
            return true;
        }
    }
    return false;
}

bool HasStatus(const gui::properties_truth::RawParameterTruth& truth,
               gui::properties_truth::TruthStatus status) {
    for (const auto actual : truth.statuses) {
        if (actual == status) {
            return true;
        }
    }
    return false;
}

const gui::properties_truth::RawParameterTruth* FindRaw(
    const gui::properties_truth::NodeTruthReport& report,
    const std::string& key) {
    for (const auto& raw : report.raw_parameters) {
        if (raw.key == key) {
            return &raw;
        }
    }
    return nullptr;
}

const gui::properties_truth::PropertyTruth* FindProperty(
    const gui::properties_truth::NodeTruthReport& report,
    const std::string& canonical_key) {
    for (const auto& property : report.properties) {
        if (property.canonical_key == canonical_key) {
            return &property;
        }
    }
    return nullptr;
}

gui::MLNode MakeNode(int id, gui::NodeType type, std::string name) {
    gui::MLNode node;
    node.id = id;
    node.type = type;
    node.name = std::move(name);
    return node;
}

void AddInput(gui::MLNode& node, int pin_id, std::string name) {
    gui::NodePin pin;
    pin.id = pin_id;
    pin.name = std::move(name);
    pin.is_input = true;
    node.inputs.push_back(pin);
}

void AddOutput(gui::MLNode& node, int pin_id, std::string name) {
    gui::NodePin pin;
    pin.id = pin_id;
    pin.name = std::move(name);
    pin.is_input = false;
    node.outputs.push_back(pin);
}

} // namespace

int main() {
    {
        auto input = MakeNode(1, gui::NodeType::DataInput, "Text input");
        input.parameters["file_category"] = "text";
        input.parameters["label_column"] = "status";

        const auto report = gui::properties_truth::ResolveNodeTruth(input);
        Check(report.properties.size() == 1,
              "DataInput should expose one initial truth row");
        const auto& label = report.properties.front();
        Check(label.canonical_key == "text_label_column",
              "text DataInput should canonicalize label to text_label_column");
        Check(label.source_key == "label_column",
              "legacy label_column should be the source when canonical is missing");
        Check(label.effective_value == "status",
              "legacy label_column should resolve as effective text label");
        Check(HasStatus(label, gui::properties_truth::TruthStatus::AliasUsed),
              "legacy label_column should be tagged as alias-used");
    }

    {
        auto input = MakeNode(1, gui::NodeType::DataInput, "Text input");
        input.parameters["file_category"] = "text";
        input.parameters["dataset_name"] = "sentiment";
        input.parameters["text_label_column"] = "status";

        gui::properties_truth::DatasetTruthFact fact;
        fact.dataset_name = "sentiment";
        fact.backing_store = "TextDataset";
        fact.found = true;
        fact.has_labels = true;
        fact.has_label_column_metadata = true;
        fact.label_column = "status";
        fact.has_class_count = true;
        fact.class_count = 0;
        std::vector<gui::properties_truth::DatasetTruthFact> facts = {fact};

        const auto report = gui::properties_truth::ResolveNodeTruth(
            input,
            gui::properties_truth::NodeTruthContext{nullptr, nullptr, &facts});
        Check(report.properties.size() == 2,
              "DataInput with dataset fact should expose class and label truth");
        Check(report.properties[0].canonical_key == "dataset_class_count",
              "dataset classes should be surfaced before label truth");
        Check(report.properties[0].effective_value == "0",
              "zero registered class count should be visible");
        Check(HasStatus(report.properties[0],
                        gui::properties_truth::TruthStatus::Missing),
              "labeled dataset with zero classes should be a truth issue");
        Check(report.has_issue,
              "zero-class labeled dataset should mark report as issue");
    }

    {
        auto input = MakeNode(1, gui::NodeType::DataInput, "Tabular input");
        input.parameters["dataset_name"] = "table";
        input.parameters["label_column"] = "missing";

        gui::properties_truth::DatasetTruthFact fact;
        fact.dataset_name = "table";
        fact.backing_store = "Arrow";
        fact.found = true;
        fact.columns = {"text", "status"};
        std::vector<gui::properties_truth::DatasetTruthFact> facts = {fact};

        const auto report = gui::properties_truth::ResolveNodeTruth(
            input,
            gui::properties_truth::NodeTruthContext{nullptr, nullptr, &facts});
        const auto& label = report.properties.back();
        Check(label.canonical_key == "label_column",
              "tabular input should use label_column as canonical key");
        Check(HasStatus(label, gui::properties_truth::TruthStatus::Missing),
              "missing label column should be visible");
    }

    {
        auto input = MakeNode(1, gui::NodeType::DataInput, "Text input");
        input.parameters["file_category"] = "text";
        input.parameters["text_label_column"] = "status";
        input.parameters["label_column"] = "sentiment";

        const auto report = gui::properties_truth::ResolveNodeTruth(input);
        const auto& label = report.properties.front();
        Check(label.effective_value == "status",
              "canonical text label should win over alias");
        Check(HasStatus(label, gui::properties_truth::TruthStatus::Conflicting),
              "conflicting label aliases should be visible");
        Check(report.has_issue,
              "conflicting aliases should mark report as an issue");
    }

    {
        auto input = MakeNode(1, gui::NodeType::DataInput, "Input");
        input.parameters["dataset_name"] = "current";
        input.parameters["dataset"] = "legacy";

        const auto report = gui::properties_truth::ResolveNodeTruth(input);
        const auto* raw_dataset = FindRaw(report, "dataset");
        Check(raw_dataset != nullptr,
              "legacy dataset parameter should appear in raw mapping");
        Check(HasStatus(*raw_dataset, gui::properties_truth::TruthStatus::Stale),
              "redundant legacy dataset parameter should be marked stale");
        Check(raw_dataset->cleanup_allowed,
              "legacy dataset should be cleanup-safe when dataset_name is set");
        Check(raw_dataset->cleanup_reason.find("dataset_name") != std::string::npos,
              "cleanup reason should explain dataset_name precedence");
    }

    {
        auto input = MakeNode(1, gui::NodeType::DataInput, "Legacy input");
        input.parameters["dataset"] = "legacy";

        const auto report = gui::properties_truth::ResolveNodeTruth(input);
        const auto* raw_dataset = FindRaw(report, "dataset");
        Check(raw_dataset != nullptr,
              "legacy-only dataset parameter should appear in raw mapping");
        Check(raw_dataset->maps_to == "dataset_name",
              "legacy-only dataset parameter should map to canonical dataset_name");
        Check(HasStatus(*raw_dataset, gui::properties_truth::TruthStatus::AliasUsed),
              "legacy-only dataset parameter should be marked alias-used");
        Check(!raw_dataset->cleanup_allowed,
              "legacy-only dataset parameter should not be cleanup-safe");
    }

    {
        auto input = MakeNode(1, gui::NodeType::DataInput, "Legacy input");
        input.parameters["dataset"] = "legacy_text";
        input.parameters["label_column"] = "status";

        gui::properties_truth::DatasetTruthFact fact;
        fact.dataset_name = "legacy_text";
        fact.backing_store = "TextDataset";
        fact.found = true;
        fact.has_labels = true;
        fact.has_class_count = true;
        fact.class_count = 2;
        std::vector<gui::properties_truth::DatasetTruthFact> facts = {fact};

        const auto report = gui::properties_truth::ResolveNodeTruth(
            input,
            gui::properties_truth::NodeTruthContext{nullptr, nullptr, &facts});
        const auto* classes = FindProperty(report, "dataset_class_count");
        Check(classes != nullptr,
              "legacy dataset key should still resolve dataset facts");
        Check(classes->effective_value == "2",
              "legacy dataset key should expose loaded class count");
    }

    {
        auto tfidf = MakeNode(2, gui::NodeType::TFIDFVectorizer, "TF-IDF");
        const auto report = gui::properties_truth::ResolveNodeTruth(tfidf);
        const auto& width = report.properties.front();
        Check(width.canonical_key == "max_features",
              "TF-IDF width should use max_features");
        Check(width.effective_value == "2000",
              "missing TF-IDF max_features should default to 2000");
        Check(HasStatus(width, gui::properties_truth::TruthStatus::Defaulted),
              "missing TF-IDF max_features should be marked defaulted");
    }

    {
        auto tfidf = MakeNode(2, gui::NodeType::TFIDFVectorizer, "TF-IDF");
        tfidf.parameters["max_features"] = "0";

        const auto report = gui::properties_truth::ResolveNodeTruth(tfidf);
        const auto* width = FindProperty(report, "max_features");
        Check(width != nullptr,
              "TF-IDF should surface max_features truth");
        Check(HasStatus(*width,
                        gui::properties_truth::TruthStatus::Missing),
              "invalid TF-IDF max_features should be visible");
        Check(width->message.find(">= 1") != std::string::npos,
              "invalid TF-IDF max_features should explain the lower bound");
        Check(report.has_issue,
              "invalid TF-IDF max_features should mark report as issue");
    }

    {
        auto tokenizer = MakeNode(2, gui::NodeType::TextTokenizer, "Tokenizer");
        tokenizer.parameters["max_length"] = "128";
        const auto report = gui::properties_truth::ResolveNodeTruth(tokenizer);
        const auto& width = report.properties.front();
        Check(width.canonical_key == "max_length",
              "TextTokenizer compiled width should use max_length");
        Check(width.effective_value == "128",
              "TextTokenizer compiled width should resolve from max_length");
        Check(width.owner == gui::properties_truth::TruthOwner::Compiler,
              "TextTokenizer max_length should be compiler-owned truth");
        Check(HasStatus(width, gui::properties_truth::TruthStatus::RequiresDialog),
              "TextTokenizer remains dialog-backed for full configuration");
    }

    {
        auto loader = MakeNode(3, gui::NodeType::DataLoader, "Loader");
        loader.parameters["pin_memory"] = "true";

        const auto report = gui::properties_truth::ResolveNodeTruth(loader);
        const auto* pin_memory = FindProperty(report, "pin_memory");
        Check(pin_memory != nullptr,
              "DataLoader should surface pin_memory truth when configured");
        Check(pin_memory->effective_value == "true",
              "DataLoader pin_memory truth should preserve configured value");
        Check(HasStatus(*pin_memory,
                        gui::properties_truth::TruthStatus::Unsupported),
              "pin_memory=true should be marked unsupported");
        Check(HasStatus(*pin_memory,
                        gui::properties_truth::TruthStatus::RequiresDialog),
              "pin_memory remains configured through the DataLoader dialog");
        Check(pin_memory->message.find("GPU host-to-device transfer") !=
                  std::string::npos,
              "pin_memory truth should describe the actual transfer scope");
        Check(pin_memory->message.find("materialization") != std::string::npos,
              "pin_memory truth should not imply materialization acceleration");
        Check(report.has_issue,
              "unsupported pin_memory should mark the truth report as issue");
    }

    {
        auto loader = MakeNode(3, gui::NodeType::DataLoader, "Balanced loader");
        loader.parameters["balance_classes"] = "true";
        loader.parameters["balance_mode"] = "weighted_sampler";
        loader.parameters["balance_target"] = "max";

        const auto report = gui::properties_truth::ResolveNodeTruth(loader);
        const auto* balance = FindProperty(report, "balance_classes");
        Check(balance != nullptr,
              "DataLoader should surface class balancing truth");
        Check(balance->effective_value == "true",
              "DataLoader class balancing should preserve configured value");
        Check(HasStatus(*balance,
                        gui::properties_truth::TruthStatus::RuntimeOnly),
              "class balancing should be marked runtime policy truth");
        Check(balance->message.find("training batchers only") !=
                  std::string::npos,
              "class balancing truth should explain train-only sampler behavior");
        Check(balance->message.find("weighted_sampler") != std::string::npos,
              "class balancing truth should include mode");
    }

    {
        auto gru = MakeNode(4, gui::NodeType::GRU, "GRU 32");
        gui::properties_truth::BackendPlacementTruthFact fact;
        fact.node_id = 4;
        fact.node_type = "GRU";
        fact.expected_backend = "CPU";
        fact.status = "cpu";
        fact.reason_code = "gru_arrayfire_cuda_probe_required";
        fact.explanation = "GRU recurrent step is expected to run on CPU.";
        std::vector<gui::properties_truth::BackendPlacementTruthFact> facts = {fact};

        const auto report = gui::properties_truth::ResolveNodeTruth(
            gru,
            gui::properties_truth::NodeTruthContext{
                nullptr, nullptr, nullptr, &facts});
        const auto* placement = FindProperty(report, "backend_placement");
        Check(placement != nullptr,
              "GRU should surface backend placement truth");
        Check(placement->effective_value == "CPU",
              "GRU placement should use compile-report expected backend");
        Check(HasStatus(*placement,
                        gui::properties_truth::TruthStatus::RuntimeOnly),
              "CPU recurrent placement should be marked runtime truth");
        Check(placement->message.find("gru_arrayfire_cuda_probe_required") !=
                  std::string::npos,
              "recurrent placement reason should be visible");
    }

    {
        auto gru = MakeNode(4, gui::NodeType::GRU, "GRU 32");
        gru.parameters["hidden_size"] = "8";
        gru.parameters["return_sequences"] = "true";

        const auto report = gui::properties_truth::ResolveNodeTruth(gru);
        const auto* hidden_size = FindProperty(report, "hidden_size");
        Check(hidden_size != nullptr,
              "GRU should surface effective hidden_size truth");
        Check(hidden_size->effective_value == "8",
              "GRU hidden_size should resolve from parameters");
        Check(HasStatus(*hidden_size,
                        gui::properties_truth::TruthStatus::Conflicting),
              "stale numeric recurrent node names should be visible");
        Check(hidden_size->message.find("32") != std::string::npos,
              "hidden_size mismatch should mention the stale name width");

        const auto* return_sequences = FindProperty(report, "return_sequences");
        Check(return_sequences != nullptr,
              "GRU should surface return_sequences truth");
        Check(return_sequences->effective_value == "true",
              "return_sequences should normalize true values");
    }

    {
        auto dense = MakeNode(1, gui::NodeType::Dense, "Dense 7");
        dense.parameters["units"] = "7";
        AddOutput(dense, 10, "Output");

        auto loss = MakeNode(2, gui::NodeType::CrossEntropyLoss, "CrossEntropy");
        AddInput(loss, 20, "Predictions");
        AddOutput(loss, 21, "Loss");

        auto output = MakeNode(3, gui::NodeType::Output, "Output");
        output.parameters["num_classes"] = "10";
        AddInput(output, 30, "Predictions");

        std::vector<gui::MLNode> nodes = {dense, loss, output};
        std::vector<gui::NodeLink> links = {
            {1, 1, 10, 2, 20},
            {2, 2, 21, 3, 30},
        };

        const auto report = gui::properties_truth::ResolveNodeTruth(
            nodes.back(),
            gui::properties_truth::NodeTruthContext{&nodes, &links});
        const auto& classes = report.properties.front();
        Check(classes.effective_value == "10",
              "Output classes should resolve from num_classes");
        Check(HasStatus(classes, gui::properties_truth::TruthStatus::Conflicting),
              "Output/Dense class mismatch should be visible");
        Check(classes.message.find("7") != std::string::npos,
              "mismatch message should include model width");
    }

    {
        auto output = MakeNode(3, gui::NodeType::Output, "Output");
        output.parameters["classes"] = "4";

        const auto report = gui::properties_truth::ResolveNodeTruth(output);
        const auto* classes = FindProperty(report, "num_classes");
        Check(classes != nullptr,
              "Output should surface canonical num_classes truth");
        Check(classes->source_key == "classes",
              "legacy classes key should feed Output class truth");
        Check(classes->effective_value == "4",
              "legacy classes key should resolve as effective Output classes");
        Check(HasStatus(*classes,
                        gui::properties_truth::TruthStatus::AliasUsed),
              "legacy classes key should be marked alias-used");
    }

    {
        auto output = MakeNode(3, gui::NodeType::Output, "Output");
        output.parameters["num_classes"] = "5";
        output.parameters["classes"] = "4";

        const auto report = gui::properties_truth::ResolveNodeTruth(output);
        const auto* classes = FindProperty(report, "num_classes");
        Check(classes != nullptr,
              "Output should surface class truth when both keys exist");
        Check(classes->effective_value == "5",
              "canonical num_classes should win over classes alias");
        Check(HasStatus(*classes,
                        gui::properties_truth::TruthStatus::Conflicting),
              "conflicting Output class aliases should be visible");
    }

    {
        auto output = MakeNode(3, gui::NodeType::Output, "Output");
        output.parameters["num_classes"] = "10";

        gui::properties_truth::DatasetTruthFact fact;
        fact.dataset_name = "sentiment";
        fact.backing_store = "TextDataset";
        fact.found = true;
        fact.has_class_count = true;
        fact.class_count = 7;
        std::vector<gui::properties_truth::DatasetTruthFact> facts = {fact};

        const auto report = gui::properties_truth::ResolveNodeTruth(
            output,
            gui::properties_truth::NodeTruthContext{nullptr, nullptr, &facts});
        const auto* classes = FindProperty(report, "num_classes");
        Check(classes != nullptr,
              "Output should surface num_classes truth");
        Check(HasStatus(*classes,
                        gui::properties_truth::TruthStatus::Conflicting),
              "Output/dataset class mismatch should be visible");
        Check(classes->message.find("sentiment") != std::string::npos &&
                  classes->message.find("7") != std::string::npos,
              "dataset mismatch should include dataset name and class count");
    }

    {
        auto output = MakeNode(3, gui::NodeType::Output, "Output");
        output.parameters["num_classes"] = "10";

        gui::properties_truth::DatasetTruthFact first;
        first.dataset_name = "train";
        first.found = true;
        first.has_class_count = true;
        first.class_count = 7;

        gui::properties_truth::DatasetTruthFact second;
        second.dataset_name = "eval";
        second.found = true;
        second.has_class_count = true;
        second.class_count = 10;
        std::vector<gui::properties_truth::DatasetTruthFact> facts = {
            first, second};

        const auto report = gui::properties_truth::ResolveNodeTruth(
            output,
            gui::properties_truth::NodeTruthContext{nullptr, nullptr, &facts});
        const auto* classes = FindProperty(report, "num_classes");
        Check(classes != nullptr,
              "Output should surface num_classes truth with dataset facts");
        Check(HasStatus(*classes,
                        gui::properties_truth::TruthStatus::Conflicting),
              "ambiguous dataset class counts should be visible");
        Check(classes->message.find("different class counts") !=
                  std::string::npos,
              "ambiguous dataset class-count message should be clear");
    }

    {
        auto input = MakeNode(1, gui::NodeType::DataInput, "Text input");
        gui::properties_truth::WriteCanonicalAndAliases(
            input, "text_label_column", "status");
        Check(input.parameters["text_label_column"] == "status",
              "canonical text label should be written");
        Check(input.parameters["label_column"] == "status",
              "compat label alias should be written");

        auto output = MakeNode(2, gui::NodeType::Output, "Output");
        gui::properties_truth::WriteCanonicalAndAliases(
            output, "num_classes", "7");
        Check(output.parameters["num_classes"] == "7",
              "canonical Output num_classes should be written");
        Check(output.parameters["classes"] == "7",
              "compat Output classes alias should be written");
    }

    std::cout << "Properties truth resolver tests passed\n";
    return 0;
}
