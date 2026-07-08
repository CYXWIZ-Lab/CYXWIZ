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
        const auto& covered = gui::properties_truth::SpecializedTruthCoverageNodeTypes();
        Check(covered.size() == 68,
              "tofix48 baseline should record each specialized truth-covered node");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::DataInput),
              "DataInput should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::DataOutput),
              "DataOutput should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::DataConvert),
              "DataConvert should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::DeployToNodeEditorNode),
              "DeployToNodeEditorNode should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::DataLoader),
              "DataLoader should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::DataProfiler),
              "DataProfiler should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::StandardScaler),
              "StandardScaler should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::MinMaxScaler),
              "MinMaxScaler should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::RobustScaler),
              "RobustScaler should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::LabelEncoder),
              "LabelEncoder should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::OrdinalEncoder),
              "OrdinalEncoder should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::TargetEncoder),
              "TargetEncoder should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::OutlierDetector),
              "OutlierDetector should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::TFIDFVectorizer),
              "TFIDFVectorizer should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::CountVectorizer),
              "CountVectorizer should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::TextTokenizer),
              "TextTokenizer should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::RegressionMetricsNode),
              "RegressionMetricsNode should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::Adam),
              "Adam should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::Dense),
              "Dense should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::BatchNorm),
              "BatchNorm should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::Reshape),
              "Reshape should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::LSTM),
              "LSTM should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::GRU),
              "GRU should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::NERSequenceBuilder),
              "NERSequenceBuilder should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::TokenVocabulary),
              "TokenVocabulary should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::POSVocabulary),
              "POSVocabulary should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::NERTagVocabulary),
              "NERTagVocabulary should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::SequenceTagOutput),
              "SequenceTagOutput should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::MSELoss),
              "MSELoss should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::Output),
              "Output should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::CrossEntropyLoss),
              "CrossEntropyLoss should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::ExportCSV),
              "ExportCSV should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::ExportParquet),
              "ExportParquet should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::ExportJSON),
              "ExportJSON should be in specialized truth coverage");
        Check(gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::TreeModelPredictor),
              "TreeModelPredictor should be in specialized truth coverage");
        Check(!gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::ExportSQL),
              "ExportSQL should stay blocked metadata, not specialized truth coverage");
        Check(!gui::properties_truth::HasSpecializedTruthCoverage(gui::NodeType::Embedding),
              "Embedding should stay dialog-only until its focused contract is audited");
    }

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
        auto output = MakeNode(2, gui::NodeType::DataOutput, "Data Output");
        output.parameters["path"] = "out.parquet";
        output.parameters["format"] = " Parquet ";

        const auto report = gui::properties_truth::ResolveNodeTruth(output);
        const auto* file_path = FindProperty(report, "file_path");
        Check(file_path != nullptr,
              "DataOutput should surface output file truth");
        Check(file_path->source_key == "path",
              "DataOutput should map legacy path alias");
        Check(file_path->effective_value == "out.parquet",
              "DataOutput path alias should become effective output file");
        Check(HasStatus(*file_path,
                        gui::properties_truth::TruthStatus::AliasUsed),
              "DataOutput path alias should be marked alias-used");

        const auto* file_type = FindProperty(report, "file_type");
        Check(file_type != nullptr,
              "DataOutput should surface output format truth");
        Check(file_type->source_key == "format",
              "DataOutput should map legacy format alias");
        Check(file_type->effective_value == "parquet",
              "DataOutput format alias should normalize like runtime");
        Check(!HasStatus(*file_type,
                         gui::properties_truth::TruthStatus::Unsupported),
              "DataOutput normalized Parquet format should be supported");

        const auto* export_result = FindProperty(report, "export_result");
        Check(export_result != nullptr,
              "DataOutput should surface runtime export result truth");
        Check(export_result->message.find("ctx.output_dataset") !=
                  std::string::npos,
              "DataOutput runtime truth should explain output path result");
    }

    {
        auto output = MakeNode(2, gui::NodeType::DataOutput, "Data Output");
        output.parameters["file_type"] = "json";

        const auto report = gui::properties_truth::ResolveNodeTruth(output);
        const auto* file_path = FindProperty(report, "file_path");
        Check(file_path != nullptr,
              "DataOutput should surface required file_path truth");
        Check(HasStatus(*file_path,
                        gui::properties_truth::TruthStatus::Missing),
              "DataOutput missing file_path should be visible");

        const auto* file_type = FindProperty(report, "file_type");
        Check(file_type != nullptr,
              "DataOutput should surface file_type truth");
        Check(HasStatus(*file_type,
                        gui::properties_truth::TruthStatus::Unsupported),
              "DataOutput unsupported file_type should be visible");
    }

    {
        auto csv = MakeNode(2, gui::NodeType::ExportCSV, "Export CSV");
        csv.parameters["path"] = "rows.csv";

        const auto report = gui::properties_truth::ResolveNodeTruth(csv);
        const auto* file_path = FindProperty(report, "file_path");
        Check(file_path != nullptr,
              "ExportCSV should surface output file truth");
        Check(file_path->source_key == "path",
              "ExportCSV should map legacy path alias");
        const auto* format = FindProperty(report, "export_format");
        Check(format != nullptr && format->effective_value == "csv",
              "ExportCSV should surface fixed csv runtime format");
    }

    {
        auto json = MakeNode(2, gui::NodeType::ExportJSON, "Export JSON");

        const auto report = gui::properties_truth::ResolveNodeTruth(json);
        const auto* file_path = FindProperty(report, "file_path");
        Check(file_path != nullptr,
              "ExportJSON should surface output file truth");
        Check(HasStatus(*file_path,
                        gui::properties_truth::TruthStatus::Missing),
              "ExportJSON missing file_path should be visible");
        const auto* format = FindProperty(report, "export_format");
        Check(format != nullptr && format->effective_value == "json",
              "ExportJSON should surface fixed json runtime format");
        Check(format->message.find("JSON array") != std::string::npos,
              "ExportJSON runtime truth should describe JSON array output");
    }

    {
        auto convert = MakeNode(2, gui::NodeType::DataConvert, "Convert");
        convert.parameters["output_format"] = "parquet";

        const auto report = gui::properties_truth::ResolveNodeTruth(convert);
        const auto* input_path = FindProperty(report, "input_path");
        Check(input_path != nullptr,
              "DataConvert should surface input_path truth");
        Check(HasStatus(*input_path,
                        gui::properties_truth::TruthStatus::Missing),
              "DataConvert without input_path or upstream input should be visible");
        const auto* output_path = FindProperty(report, "output_path");
        Check(output_path != nullptr,
              "DataConvert should surface output_path truth");
        Check(HasStatus(*output_path,
                        gui::properties_truth::TruthStatus::Missing),
              "DataConvert missing output_path should be visible");
    }

    {
        auto source = MakeNode(1, gui::NodeType::DataInput, "Input");
        AddOutput(source, 10, "Data");
        auto convert = MakeNode(2, gui::NodeType::DataConvert, "Convert");
        AddInput(convert, 20, "Input");
        convert.parameters["output_path"] = "converted.parquet";

        std::vector<gui::MLNode> nodes = {source, convert};
        std::vector<gui::NodeLink> links = {
            {1, 1, 10, 2, 20},
        };

        const auto report = gui::properties_truth::ResolveNodeTruth(
            nodes.back(),
            gui::properties_truth::NodeTruthContext{&nodes, &links});
        const auto* input_path = FindProperty(report, "input_path");
        Check(input_path != nullptr,
              "connected DataConvert should still surface input source truth");
        Check(!HasStatus(*input_path,
                         gui::properties_truth::TruthStatus::Missing),
              "connected DataConvert should not require input_path");
        const auto* result = FindProperty(report, "convert_result");
        Check(result != nullptr,
              "DataConvert should surface runtime conversion result truth");
        Check(result->message.find("ds_dataconvert_") != std::string::npos,
              "DataConvert runtime result should name registered dataset pattern");
    }

    {
        auto deploy = MakeNode(2, gui::NodeType::DeployToNodeEditorNode,
                               "Deploy");

        const auto report = gui::properties_truth::ResolveNodeTruth(deploy);
        const auto* name = FindProperty(report, "name");
        Check(name != nullptr,
              "DeployToNodeEditor should surface deployment name truth");
        Check(name->effective_value == "deployed_2",
              "DeployToNodeEditor should default to deployed_<node id>");
        const auto* result = FindProperty(report, "deployment_result");
        Check(result != nullptr,
              "DeployToNodeEditor should surface runtime result truth");
        Check(result->message.find("deployment_dataset") != std::string::npos,
              "DeployToNodeEditor truth should describe deployment context fields");
    }

    {
        auto profiler = MakeNode(2, gui::NodeType::DataProfiler,
                                 "Profiler");
        profiler.parameters["minimal"] = "true";

        const auto report = gui::properties_truth::ResolveNodeTruth(profiler);
        const auto* minimal = FindProperty(report, "minimal");
        Check(minimal != nullptr,
              "DataProfiler should surface minimal mode truth");
        Check(HasStatus(*minimal,
                        gui::properties_truth::TruthStatus::Unsupported),
              "DataProfiler minimal=true should be marked unsupported");
        const auto* schema = FindProperty(report, "profile_report_schema");
        Check(schema != nullptr,
              "DataProfiler should surface report schema truth");
        Check(schema->effective_value.find("null_count") != std::string::npos,
              "DataProfiler schema truth should name emitted columns");
    }

    {
        auto predictor = MakeNode(2, gui::NodeType::TreeModelPredictor,
                                  "Tree Predict");

        const auto report = gui::properties_truth::ResolveNodeTruth(predictor);
        const auto* model_path = FindProperty(report, "model_path");
        Check(model_path != nullptr,
              "TreeModelPredictor should surface model_path truth");
        Check(HasStatus(*model_path,
                        gui::properties_truth::TruthStatus::Missing),
              "TreeModelPredictor missing model_path should be visible");
        const auto* feature_cols = FindProperty(report, "feature_cols");
        Check(feature_cols != nullptr &&
                  feature_cols->effective_value == "artifact feature order",
              "TreeModelPredictor empty feature_cols should use artifact order truth");
        const auto* inference = FindProperty(report, "inference_result");
        Check(inference != nullptr,
              "TreeModelPredictor should surface inference result truth");
        Check(inference->message.find("PipelineOperatorFactory") !=
                  std::string::npos,
              "TreeModelPredictor truth should describe operator-factory routing");
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
        const auto* text_col = FindProperty(report, "text_col");
        Check(text_col != nullptr,
              "TF-IDF should surface required text_col truth");
        Check(HasStatus(*text_col, gui::properties_truth::TruthStatus::Missing),
              "missing TF-IDF text_col should be visible");
        const auto* width = FindProperty(report, "max_features");
        Check(width != nullptr,
              "TF-IDF width should use max_features");
        Check(width->effective_value == "2000",
              "missing TF-IDF max_features should default to 2000");
        Check(HasStatus(*width, gui::properties_truth::TruthStatus::Defaulted),
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
        auto scaler = MakeNode(2, gui::NodeType::MinMaxScaler, "MinMax");
        scaler.parameters["min"] = "2";
        scaler.parameters["max"] = "1";

        const auto report = gui::properties_truth::ResolveNodeTruth(scaler);
        const auto* max_value = FindProperty(report, "max");
        Check(max_value != nullptr,
              "MinMaxScaler should surface target max truth");
        Check(HasStatus(*max_value, gui::properties_truth::TruthStatus::Missing),
              "MinMaxScaler max <= min should be visible");
        Check(max_value->message.find("greater than min") != std::string::npos,
              "MinMaxScaler range issue should explain ordering");
        Check(report.has_issue,
              "invalid MinMaxScaler range should mark report as issue");
    }

    {
        auto robust = MakeNode(2, gui::NodeType::RobustScaler, "Robust");
        robust.parameters["quantile_min"] = "80";
        robust.parameters["quantile_max"] = "50";

        const auto report = gui::properties_truth::ResolveNodeTruth(robust);
        const auto* qmax = FindProperty(report, "quantile_max");
        Check(qmax != nullptr,
              "RobustScaler should surface upper quantile truth");
        Check(HasStatus(*qmax, gui::properties_truth::TruthStatus::Missing),
              "RobustScaler invalid quantile ordering should be visible");
        Check(qmax->message.find("greater than quantile_min") != std::string::npos,
              "RobustScaler range issue should explain ordering");
    }

    {
        auto label = MakeNode(2, gui::NodeType::LabelEncoder, "Label");

        const auto report = gui::properties_truth::ResolveNodeTruth(label);
        const auto* column = FindProperty(report, "column");
        Check(column != nullptr,
              "LabelEncoder should surface required column truth");
        Check(HasStatus(*column, gui::properties_truth::TruthStatus::Missing),
              "missing LabelEncoder column should be visible");
    }

    {
        auto ordinal = MakeNode(2, gui::NodeType::OrdinalEncoder, "Ordinal");
        ordinal.parameters["columns"] = "city";
        ordinal.parameters["categories"] = "manual";

        const auto report = gui::properties_truth::ResolveNodeTruth(ordinal);
        const auto* categories = FindProperty(report, "categories");
        Check(categories != nullptr,
              "OrdinalEncoder should surface category ordering truth");
        Check(HasStatus(*categories,
                        gui::properties_truth::TruthStatus::Unsupported),
              "unsupported OrdinalEncoder category ordering should be visible");
        Check(report.has_issue,
              "unsupported OrdinalEncoder category ordering should mark report as issue");
    }

    {
        auto target = MakeNode(2, gui::NodeType::TargetEncoder, "Target");
        target.parameters["columns"] = "city";
        target.parameters["smoothing"] = "-1";

        const auto report = gui::properties_truth::ResolveNodeTruth(target);
        const auto* target_col = FindProperty(report, "target_col");
        Check(target_col != nullptr,
              "TargetEncoder should surface required target_col truth");
        Check(HasStatus(*target_col,
                        gui::properties_truth::TruthStatus::Missing),
              "missing TargetEncoder target_col should be visible");
        const auto* smoothing = FindProperty(report, "smoothing");
        Check(smoothing != nullptr,
              "TargetEncoder should surface smoothing truth");
        Check(HasStatus(*smoothing,
                        gui::properties_truth::TruthStatus::Missing),
              "negative TargetEncoder smoothing should be visible");
    }

    {
        auto outlier = MakeNode(2, gui::NodeType::OutlierDetector, "Outlier");
        outlier.parameters["method"] = "lof";
        outlier.parameters["action"] = "remove";

        const auto report = gui::properties_truth::ResolveNodeTruth(outlier);
        const auto* method = FindProperty(report, "method");
        Check(method != nullptr,
              "OutlierDetector should surface method truth");
        Check(HasStatus(*method,
                        gui::properties_truth::TruthStatus::Unsupported),
              "unsupported OutlierDetector method should be visible");
        const auto* action = FindProperty(report, "action");
        Check(action != nullptr,
              "OutlierDetector should surface action truth");
        Check(HasStatus(*action,
                        gui::properties_truth::TruthStatus::Unsupported),
              "unsupported OutlierDetector action should be visible");
    }

    {
        auto count = MakeNode(2, gui::NodeType::CountVectorizer, "Count");
        count.parameters["text_col"] = "text";
        count.parameters["ngram_max"] = "4";

        const auto report = gui::properties_truth::ResolveNodeTruth(count);
        const auto* ngram_max = FindProperty(report, "ngram_max");
        Check(ngram_max != nullptr,
              "CountVectorizer should surface ngram_max truth");
        Check(HasStatus(*ngram_max,
                        gui::properties_truth::TruthStatus::Unsupported),
              "CountVectorizer ngram_max > 3 should be visible");
        const auto* binary = FindProperty(report, "binary");
        Check(binary != nullptr,
              "CountVectorizer should surface binary-count truth");
        Check(binary->effective_value == "false",
              "CountVectorizer binary should default to false");
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
        auto metrics = MakeNode(2, gui::NodeType::RegressionMetricsNode, "Metrics");
        metrics.parameters["actual_col"] = "y";

        const auto report = gui::properties_truth::ResolveNodeTruth(metrics);
        const auto* predicted = FindProperty(report, "predicted_col");
        Check(predicted != nullptr,
              "RegressionMetrics should surface predicted_col truth");
        Check(HasStatus(*predicted,
                        gui::properties_truth::TruthStatus::Missing),
              "missing RegressionMetrics predicted_col should be visible");
        const auto* metric_list = FindProperty(report, "metrics");
        Check(metric_list != nullptr,
              "RegressionMetrics should surface metrics truth");
        Check(metric_list->effective_value == "mse,rmse,mae,r2",
              "RegressionMetrics should default to runtime metric list");
    }

    {
        auto roc = MakeNode(2, gui::NodeType::ROCCurveNode, "ROC");
        roc.parameters["actual_col"] = "label";

        const auto report = gui::properties_truth::ResolveNodeTruth(roc);
        const auto* score = FindProperty(report, "score_col");
        Check(score != nullptr,
              "ROCCurve should surface score_col truth");
        Check(HasStatus(*score, gui::properties_truth::TruthStatus::Missing),
              "missing ROCCurve score_col should be visible");
    }

    {
        auto adam = MakeNode(2, gui::NodeType::Adam, "Adam");
        adam.parameters["lr"] = "0.005";
        adam.parameters["beta1"] = "0.8";

        const auto report = gui::properties_truth::ResolveNodeTruth(adam);
        const auto* lr = FindProperty(report, "learning_rate");
        Check(lr != nullptr,
              "Adam should surface canonical learning_rate truth");
        Check(lr->source_key == "lr",
              "Adam should map legacy lr alias to learning_rate");
        Check(HasStatus(*lr, gui::properties_truth::TruthStatus::AliasUsed),
              "Adam legacy lr should be marked alias-used");
        const auto* beta1 = FindProperty(report, "beta1");
        Check(beta1 != nullptr,
              "Adam should surface unsupported beta1 truth when present");
        Check(HasStatus(*beta1,
                        gui::properties_truth::TruthStatus::Unsupported),
              "Adam beta1 should be marked unsupported until optimizer construction applies it");
        Check(report.has_issue,
              "unsupported Adam beta1 should mark report as issue");
    }

    {
        auto sgd = MakeNode(2, gui::NodeType::SGD, "SGD");
        sgd.parameters["learning_rate"] = "0";
        sgd.parameters["momentum"] = "0.9";

        const auto report = gui::properties_truth::ResolveNodeTruth(sgd);
        const auto* lr = FindProperty(report, "learning_rate");
        Check(lr != nullptr,
              "SGD should surface learning_rate truth");
        Check(HasStatus(*lr, gui::properties_truth::TruthStatus::Missing),
              "SGD learning_rate <= 0 should be visible");
        const auto* momentum = FindProperty(report, "momentum");
        Check(momentum != nullptr,
              "SGD should surface unsupported momentum truth when present");
        Check(HasStatus(*momentum,
                        gui::properties_truth::TruthStatus::Unsupported),
              "SGD momentum should be marked unsupported until optimizer construction applies it");
    }

    {
        auto loss = MakeNode(2, gui::NodeType::CrossEntropyLoss, "CE");
        loss.parameters["label_smoothing"] = "1.0";
        loss.parameters["class_weight"] = "balanced";

        const auto report = gui::properties_truth::ResolveNodeTruth(loss);
        const auto* smoothing = FindProperty(report, "label_smoothing");
        Check(smoothing != nullptr,
              "CrossEntropy should surface label_smoothing truth");
        Check(HasStatus(*smoothing,
                        gui::properties_truth::TruthStatus::Missing),
              "invalid CrossEntropy label_smoothing should be visible");
        const auto* class_weight = FindProperty(report, "class_weight");
        Check(class_weight != nullptr,
              "CrossEntropy should surface class_weight truth");
        Check(HasStatus(*class_weight,
                        gui::properties_truth::TruthStatus::Unsupported),
              "unresolved CrossEntropy balanced class_weight should be visible");
    }

    {
        auto loss = MakeNode(2, gui::NodeType::BCEWithLogits, "BCE logits");
        loss.parameters["pos_weight"] = "0";

        const auto report = gui::properties_truth::ResolveNodeTruth(loss);
        const auto* pos_weight = FindProperty(report, "pos_weight");
        Check(pos_weight != nullptr,
              "BCEWithLogits should surface pos_weight truth");
        Check(HasStatus(*pos_weight,
                        gui::properties_truth::TruthStatus::Missing),
              "BCEWithLogits pos_weight <= 0 should be visible");
    }

    {
        auto dense = MakeNode(2, gui::NodeType::Dense, "Dense");
        dense.parameters["units"] = "0";
        dense.parameters["activation"] = "relu";

        const auto report = gui::properties_truth::ResolveNodeTruth(dense);
        const auto* units = FindProperty(report, "units");
        Check(units != nullptr,
              "Dense should surface units truth");
        Check(HasStatus(*units, gui::properties_truth::TruthStatus::Missing),
              "Dense units <= 0 should be visible");
        const auto* activation = FindProperty(report, "activation");
        Check(activation != nullptr,
              "Dense should surface unsupported inline activation truth");
        Check(HasStatus(*activation,
                        gui::properties_truth::TruthStatus::Unsupported),
              "Dense inline activation should be marked unsupported");
    }

    {
        auto norm = MakeNode(2, gui::NodeType::BatchNorm, "BatchNorm");
        norm.parameters["epsilon"] = "0.001";
        norm.parameters["momentum"] = "0";

        const auto report = gui::properties_truth::ResolveNodeTruth(norm);
        const auto* eps = FindProperty(report, "eps");
        Check(eps != nullptr,
              "BatchNorm should surface canonical eps truth");
        Check(eps->source_key == "epsilon",
              "BatchNorm should map legacy epsilon alias to eps");
        Check(HasStatus(*eps, gui::properties_truth::TruthStatus::AliasUsed),
              "BatchNorm legacy epsilon should be marked alias-used");
        const auto* momentum = FindProperty(report, "momentum");
        Check(momentum != nullptr,
              "BatchNorm should surface momentum truth");
        Check(HasStatus(*momentum,
                        gui::properties_truth::TruthStatus::Missing),
              "BatchNorm momentum <= 0 should be visible because ModelBuilder falls back");
    }

    {
        auto dropout = MakeNode(2, gui::NodeType::Dropout, "Dropout");
        dropout.parameters["rate"] = "1.0";

        const auto report = gui::properties_truth::ResolveNodeTruth(dropout);
        const auto* rate = FindProperty(report, "rate");
        Check(rate != nullptr,
              "Dropout should surface rate truth");
        Check(HasStatus(*rate, gui::properties_truth::TruthStatus::Missing),
              "Dropout rate >= 1 should be visible");
    }

    {
        auto reshape = MakeNode(2, gui::NodeType::Reshape, "Reshape");
        const auto report = gui::properties_truth::ResolveNodeTruth(reshape);
        const auto* shape = FindProperty(report, "shape");
        Check(shape != nullptr,
              "Reshape should surface target shape truth");
        Check(shape->effective_value == "-1,256",
              "Reshape shape should default to metadata/compiler example");

        auto relu = MakeNode(3, gui::NodeType::ReLU, "ReLU");
        const auto activation_report =
            gui::properties_truth::ResolveNodeTruth(relu);
        const auto* shape_effect = FindProperty(activation_report,
                                                "shape_effect");
        Check(shape_effect != nullptr,
              "Activation nodes should surface shape-preserving truth");
        Check(shape_effect->effective_value.find("preserves") !=
                  std::string::npos,
              "Activation shape truth should explain shape preservation");
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
        auto builder = MakeNode(5, gui::NodeType::NERSequenceBuilder,
                                "NER Sequences");
        builder.parameters["tokens_column"] = "words";
        builder.parameters["tag_column"] = "bio";
        builder.parameters["sequence_id_column"] = "sentence";
        builder.parameters["max_sequence_length"] = "-2";
        builder.parameters["min_freq"] = "2";
        builder.parameters["max_vocab_size"] = "4096";

        const auto report = gui::properties_truth::ResolveNodeTruth(builder);
        const auto* token_column = FindProperty(report, "token_column");
        Check(token_column != nullptr,
              "NERSequenceBuilder should surface token column truth");
        Check(token_column->source_key == "tokens_column",
              "NERSequenceBuilder should map tokens_column alias");
        Check(token_column->effective_value == "words",
              "NERSequenceBuilder token alias should become effective value");
        Check(HasStatus(*token_column,
                        gui::properties_truth::TruthStatus::AliasUsed),
              "NERSequenceBuilder token alias should be marked alias-used");

        const auto* sentence_id = FindProperty(report, "sentence_id_column");
        Check(sentence_id != nullptr &&
                  sentence_id->source_key == "sequence_id_column",
              "NERSequenceBuilder should map sequence_id_column alias");

        const auto* max_length = FindProperty(report, "max_sequence_length");
        Check(max_length != nullptr,
              "NERSequenceBuilder should surface max_sequence_length truth");
        Check(HasStatus(*max_length,
                        gui::properties_truth::TruthStatus::Missing),
              "negative max_sequence_length should be reported");

        const auto* min_frequency = FindProperty(report, "min_frequency");
        Check(min_frequency != nullptr &&
                  min_frequency->source_key == "min_freq",
              "NERSequenceBuilder should map min_freq alias");
        const auto* max_size = FindProperty(report, "max_size");
        Check(max_size != nullptr &&
                  max_size->source_key == "max_vocab_size",
              "NERSequenceBuilder should map max_vocab_size alias");

        const auto* raw_tokens = FindRaw(report, "tokens_column");
        Check(raw_tokens != nullptr && raw_tokens->maps_to == "token_column",
              "raw sequence token alias should map to canonical token_column");
    }

    {
        auto vocab = MakeNode(6, gui::NodeType::TokenVocabulary,
                              "Token Vocabulary");
        vocab.parameters["column"] = "words";
        vocab.parameters["min_freq"] = "0";
        vocab.parameters["max_vocab_size"] = "0";

        const auto report = gui::properties_truth::ResolveNodeTruth(vocab);
        const auto* min_frequency = FindProperty(report, "min_frequency");
        Check(min_frequency != nullptr,
              "TokenVocabulary should surface min_frequency truth");
        Check(min_frequency->source_key == "min_freq",
              "TokenVocabulary should map editor min_freq alias");
        Check(HasStatus(*min_frequency,
                        gui::properties_truth::TruthStatus::Missing),
              "TokenVocabulary min_frequency must be positive");

        const auto* max_size = FindProperty(report, "max_size");
        Check(max_size != nullptr,
              "TokenVocabulary should surface max_size truth");
        Check(max_size->effective_value == "0",
              "TokenVocabulary max_size should allow unlimited 0");
        Check(!HasStatus(*max_size,
                         gui::properties_truth::TruthStatus::Missing),
              "TokenVocabulary max_size=0 should be valid");
    }

    {
        auto tag_vocab = MakeNode(7, gui::NodeType::NERTagVocabulary,
                                  "NER Tag Vocabulary");
        tag_vocab.parameters["outside_tag"] = "NONE";
        tag_vocab.parameters["bio_scheme"] = "BILOU";

        const auto report = gui::properties_truth::ResolveNodeTruth(tag_vocab);
        const auto* outside_tag = FindProperty(report, "outside_tag");
        Check(outside_tag != nullptr,
              "NERTagVocabulary should surface outside_tag truth");
        Check(HasStatus(*outside_tag,
                        gui::properties_truth::TruthStatus::Unsupported),
              "custom outside_tag should be marked unsupported");
        Check(report.has_issue,
              "unsupported tag vocabulary values should mark report issue");

        const auto* bio_scheme = FindProperty(report, "bio_scheme");
        Check(bio_scheme != nullptr,
              "NERTagVocabulary should surface bio_scheme truth");
        Check(HasStatus(*bio_scheme,
                        gui::properties_truth::TruthStatus::Unsupported),
              "non-BIO tag schemes should be marked unsupported");
    }

    {
        auto output = MakeNode(8, gui::NodeType::SequenceTagOutput,
                               "Sequence Tags");
        output.parameters["num_tags"] = "-1";
        output.parameters["decode_scheme"] = "BILOU";

        const auto report = gui::properties_truth::ResolveNodeTruth(output);
        const auto* num_tags = FindProperty(report, "num_tags");
        Check(num_tags != nullptr,
              "SequenceTagOutput should surface num_tags truth");
        Check(HasStatus(*num_tags,
                        gui::properties_truth::TruthStatus::Missing),
              "SequenceTagOutput num_tags must be non-negative");

        const auto* decode_scheme = FindProperty(report, "decode_scheme");
        Check(decode_scheme != nullptr,
              "SequenceTagOutput should surface decode_scheme truth");
        Check(HasStatus(*decode_scheme,
                        gui::properties_truth::TruthStatus::Unsupported),
              "SequenceTagOutput should reject unsupported decode schemes");
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

        auto export_node = MakeNode(3, gui::NodeType::DataOutput, "Data Output");
        export_node.parameters["path"] = "old.csv";
        export_node.parameters["format"] = "csv";
        gui::properties_truth::WriteCanonicalAndAliases(
            export_node, "file_path", "new.parquet");
        gui::properties_truth::WriteCanonicalAndAliases(
            export_node, "file_type", "parquet");
        Check(export_node.parameters["file_path"] == "new.parquet",
              "DataOutput canonical file_path should be written");
        Check(export_node.parameters.find("path") == export_node.parameters.end(),
              "DataOutput legacy path alias should be cleared after canonical edit");
        Check(export_node.parameters["file_type"] == "parquet",
              "DataOutput canonical file_type should be written");
        Check(export_node.parameters.find("format") == export_node.parameters.end(),
              "DataOutput legacy format alias should be cleared after canonical edit");

        auto sequence = MakeNode(4, gui::NodeType::NERSequenceBuilder,
                                 "NER Sequences");
        sequence.parameters["tokens_column"] = "words";
        sequence.parameters["min_freq"] = "2";
        gui::properties_truth::WriteCanonicalAndAliases(
            sequence, "token_column", "tokens");
        gui::properties_truth::WriteCanonicalAndAliases(
            sequence, "min_frequency", "3");
        Check(sequence.parameters["token_column"] == "tokens",
              "NERSequenceBuilder canonical token column should be written");
        Check(sequence.parameters.find("tokens_column") ==
                  sequence.parameters.end(),
              "NERSequenceBuilder token alias should be cleared");
        Check(sequence.parameters["min_frequency"] == "3",
              "canonical min_frequency should be written");
        Check(sequence.parameters.find("min_freq") == sequence.parameters.end(),
              "legacy min_freq alias should be cleared after canonical edit");
    }

    std::cout << "Properties truth resolver tests passed\n";
    return 0;
}
