#include "../src/core/model_exporter.h"
#include "../src/core/model_importer.h"
#include "../src/core/graph_compiler.h"
#include "../src/gui/loaders/data_loader.h"

#include <cyxwiz/sequential.h>
#include <cyxwiz/tensor.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace cyxwiz {

std::string ReadTreeModelArtifactType(const std::string&, std::string*) {
    return {};
}

} // namespace cyxwiz

namespace cyxwiz::loaders {

DataLoader* GetByCategory(FileCategory) {
    return nullptr;
}

DataLoader* GetByRegisteredDataset(const std::string&) {
    return nullptr;
}

DataLoader* GetByBackendTag(int) {
    return nullptr;
}

const std::vector<DataLoader*>& All() {
    static const std::vector<DataLoader*> loaders;
    return loaders;
}

FileCategory FileCategoryFromString(const std::string&) {
    return FileCategory::Tabular;
}

} // namespace cyxwiz::loaders

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

void CheckNear(float actual,
               float expected,
               float tolerance,
               const std::string& message) {
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << message << ": expected=" << expected
                  << " actual=" << actual << "\n";
        std::exit(1);
    }
}

std::string BuildTransformerEncoderGraphJson() {
    using json = nlohmann::json;
    json graph;
    graph["nodes"] = json::array();
    graph["links"] = json::array();

    graph["nodes"].push_back({
        {"id", 1},
        {"type", static_cast<int>(gui::NodeType::DatasetInput)},
        {"name", "Input"},
        {"parameters", {
            {"dataset_name", "transformer_encoder_roundtrip_dataset"},
            {"dataset", "transformer_encoder_roundtrip_dataset"},
            {"shape", "[4]"}
        }}
    });
    graph["nodes"].push_back({
        {"id", 2},
        {"type", static_cast<int>(gui::NodeType::TransformerEncoder)},
        {"name", "Encoder"},
        {"parameters", {
            {"d_model", "4"},
            {"num_heads", "2"},
            {"dim_feedforward", "8"},
            {"dropout", "0"},
            {"norm_first", "false"}
        }}
    });
    graph["nodes"].push_back({
        {"id", 3},
        {"type", static_cast<int>(gui::NodeType::Dense)},
        {"name", "Classifier"},
        {"parameters", {{"units", "2"}}}
    });
    graph["nodes"].push_back({
        {"id", 4},
        {"type", static_cast<int>(gui::NodeType::Output)},
        {"name", "Output"},
        {"parameters", {{"num_classes", "2"}}}
    });
    graph["nodes"].push_back({
        {"id", 5},
        {"type", static_cast<int>(gui::NodeType::MSELoss)},
        {"name", "MSE"},
        {"parameters", json::object()}
    });
    graph["nodes"].push_back({
        {"id", 6},
        {"type", static_cast<int>(gui::NodeType::Adam)},
        {"name", "Adam"},
        {"parameters", {{"learning_rate", "0.001"}}}
    });

    graph["links"].push_back({{"id", 101}, {"from_node", 1}, {"to_node", 2}});
    graph["links"].push_back({{"id", 102}, {"from_node", 2}, {"to_node", 3}});
    graph["links"].push_back({{"id", 103}, {"from_node", 3}, {"to_node", 4}});
    graph["links"].push_back({{"id", 104}, {"from_node", 3}, {"to_node", 5}});
    graph["links"].push_back({{"id", 105}, {"from_node", 6}, {"to_node", 5}});

    return graph.dump();
}

void CheckTensorValues(const cyxwiz::Tensor& tensor,
                       const cyxwiz::Tensor& expected,
                       const std::string& name) {
    Check(tensor.Shape() == expected.Shape(), name + " shape mismatch");
    const float* actual_data = tensor.Data<float>();
    const float* expected_data = expected.Data<float>();
    for (size_t i = 0; i < tensor.NumElements(); ++i) {
        CheckNear(actual_data[i], expected_data[i], 1e-6f, name);
    }
}

std::string BuildBertSequenceClassifierGraphJson() {
    using json = nlohmann::json;
    json graph;
    graph["nodes"] = json::array();
    graph["links"] = json::array();

    graph["nodes"].push_back({
        {"id", 1},
        {"type", static_cast<int>(gui::NodeType::DatasetInput)},
        {"name", "BERT Tokens"},
        {"parameters", {
            {"dataset_name", "bert_sequence_roundtrip_dataset"},
            {"dataset", "bert_sequence_roundtrip_dataset"},
            {"shape", "[4]"},
            {"model_family", "bert_encoder"},
            {"token_column", "token_ids"},
            {"create_attention_mask", "true"}
        }}
    });
    graph["nodes"].push_back({
        {"id", 2},
        {"type", static_cast<int>(gui::NodeType::Embedding)},
        {"name", "Token Embedding"},
        {"parameters", {
            {"num_embeddings", "16"},
            {"embedding_dim", "4"},
            {"padding_idx", "0"}
        }}
    });
    graph["nodes"].push_back({
        {"id", 3},
        {"type", static_cast<int>(gui::NodeType::PositionalEncoding)},
        {"name", "Position"},
        {"parameters", {
            {"d_model", "4"},
            {"max_sequence_length", "4"}
        }}
    });
    graph["nodes"].push_back({
        {"id", 4},
        {"type", static_cast<int>(gui::NodeType::TransformerEncoder)},
        {"name", "BERT Encoder"},
        {"parameters", {
            {"d_model", "4"},
            {"num_heads", "2"},
            {"dim_feedforward", "8"},
            {"dropout", "0"},
            {"norm_first", "false"}
        }}
    });
    graph["nodes"].push_back({
        {"id", 5},
        {"type", static_cast<int>(gui::NodeType::TensorIndexSelect)},
        {"name", "CLS Select"},
        {"parameters", {{"dim", "0"}, {"indices", "0"}}}
    });
    graph["nodes"].push_back({
        {"id", 6},
        {"type", static_cast<int>(gui::NodeType::Dense)},
        {"name", "Classifier"},
        {"parameters", {{"units", "2"}}}
    });
    graph["nodes"].push_back({
        {"id", 7},
        {"type", static_cast<int>(gui::NodeType::Output)},
        {"name", "Class Output"},
        {"parameters", {{"classes", "2"}}}
    });
    graph["nodes"].push_back({
        {"id", 8},
        {"type", static_cast<int>(gui::NodeType::CrossEntropyLoss)},
        {"name", "Cross Entropy"},
        {"parameters", json::object()}
    });
    graph["nodes"].push_back({
        {"id", 9},
        {"type", static_cast<int>(gui::NodeType::Adam)},
        {"name", "Adam"},
        {"parameters", {{"learning_rate", "0.001"}}}
    });

    graph["links"].push_back({{"id", 201}, {"from_node", 1}, {"to_node", 2}});
    graph["links"].push_back({{"id", 202}, {"from_node", 2}, {"to_node", 3}});
    graph["links"].push_back({{"id", 203}, {"from_node", 3}, {"to_node", 4}});
    graph["links"].push_back({{"id", 204}, {"from_node", 4}, {"to_node", 5}});
    graph["links"].push_back({{"id", 205}, {"from_node", 5}, {"to_node", 6}});
    graph["links"].push_back({{"id", 206}, {"from_node", 6}, {"to_node", 7}});
    graph["links"].push_back({{"id", 207}, {"from_node", 6}, {"to_node", 8}});
    graph["links"].push_back({{"id", 208}, {"from_node", 1}, {"to_node", 8}});
    graph["links"].push_back({{"id", 209}, {"from_node", 9}, {"to_node", 8}});

    return graph.dump();
}

std::string BuildBertTokenClassifierGraphJson() {
    using json = nlohmann::json;
    json graph;
    graph["nodes"] = json::array();
    graph["links"] = json::array();

    graph["nodes"].push_back({
        {"id", 1},
        {"type", static_cast<int>(gui::NodeType::DatasetInput)},
        {"name", "BERT Tokens"},
        {"parameters", {
            {"dataset_name", "bert_token_roundtrip_dataset"},
            {"dataset", "bert_token_roundtrip_dataset"},
            {"shape", "[4]"},
            {"model_family", "bert_encoder"},
            {"token_column", "token_ids"},
            {"create_attention_mask", "true"}
        }}
    });
    graph["nodes"].push_back({
        {"id", 2},
        {"type", static_cast<int>(gui::NodeType::Embedding)},
        {"name", "Token Embedding"},
        {"parameters", {
            {"num_embeddings", "16"},
            {"embedding_dim", "4"},
            {"padding_idx", "0"}
        }}
    });
    graph["nodes"].push_back({
        {"id", 3},
        {"type", static_cast<int>(gui::NodeType::PositionalEncoding)},
        {"name", "Position"},
        {"parameters", {
            {"d_model", "4"},
            {"max_sequence_length", "4"}
        }}
    });
    graph["nodes"].push_back({
        {"id", 4},
        {"type", static_cast<int>(gui::NodeType::TransformerEncoder)},
        {"name", "BERT Encoder"},
        {"parameters", {
            {"d_model", "4"},
            {"num_heads", "2"},
            {"dim_feedforward", "8"},
            {"dropout", "0"},
            {"norm_first", "false"}
        }}
    });
    graph["nodes"].push_back({
        {"id", 5},
        {"type", static_cast<int>(gui::NodeType::TimeDistributed)},
        {"name", "Token Classifier"},
        {"parameters", {{"units", "3"}}}
    });
    graph["nodes"].push_back({
        {"id", 6},
        {"type", static_cast<int>(gui::NodeType::SequenceTagOutput)},
        {"name", "Tag Output"},
        {"parameters", {{"num_tags", "3"}}}
    });
    graph["nodes"].push_back({
        {"id", 7},
        {"type", static_cast<int>(gui::NodeType::CrossEntropyLoss)},
        {"name", "Cross Entropy"},
        {"parameters", json::object()}
    });
    graph["nodes"].push_back({
        {"id", 8},
        {"type", static_cast<int>(gui::NodeType::Adam)},
        {"name", "Adam"},
        {"parameters", {{"learning_rate", "0.001"}}}
    });

    graph["links"].push_back({{"id", 301}, {"from_node", 1}, {"to_node", 2}});
    graph["links"].push_back({{"id", 302}, {"from_node", 2}, {"to_node", 3}});
    graph["links"].push_back({{"id", 303}, {"from_node", 3}, {"to_node", 4}});
    graph["links"].push_back({{"id", 304}, {"from_node", 4}, {"to_node", 5}});
    graph["links"].push_back({{"id", 305}, {"from_node", 5}, {"to_node", 6}});
    graph["links"].push_back({{"id", 306}, {"from_node", 5}, {"to_node", 7}});
    graph["links"].push_back({{"id", 307}, {"from_node", 1}, {"to_node", 7}});
    graph["links"].push_back({{"id", 308}, {"from_node", 8}, {"to_node", 7}});

    return graph.dump();
}

void CheckParameterRoundTrip(
    const std::map<std::string, cyxwiz::Tensor>& source_params,
    const std::map<std::string, cyxwiz::Tensor>& imported_params,
    const std::string& label) {
    Check(imported_params.size() == source_params.size(),
          label + " parameter count should round-trip");
    for (const auto& [name, tensor] : source_params) {
        Check(imported_params.count(name) == 1,
              label + " imported parameter missing: " + name);
        CheckTensorValues(imported_params.at(name), tensor, label + " " + name);
    }
}
} // namespace

int main() {
    namespace fs = std::filesystem;

    const fs::path root =
        fs::temp_directory_path() / "cyxwiz_cyxmodel_transformer_encoder_roundtrip";
    const fs::path package_path = root / "transformer_encoder.cyxmodel";
    const fs::path bert_sequence_path =
        root / "bert_sequence_classifier.cyxmodel";
    const fs::path bert_token_path =
        root / "bert_token_classifier.cyxmodel";
    fs::remove_all(root);
    fs::create_directories(root);

    cyxwiz::SequentialModel source;
    source.Add<cyxwiz::TransformerEncoderModule>(4, 2, 8, 0.0f, false);
    source.Add<cyxwiz::LinearModule>(16, 2, true);

    const auto source_params = source.GetParameters();
    Check(!source_params.empty(),
          "source TransformerEncoder model should expose parameters");

    cyxwiz::ExportOptions export_options;
    export_options.format = cyxwiz::ModelFormat::CyxModel;
    export_options.include_graph = true;
    export_options.include_training_history = false;
    export_options.include_optimizer_state = false;

    cyxwiz::ModelExporter exporter;
    const cyxwiz::ExportResult exported = exporter.ExportCyxModel(
        source,
        nullptr,
        nullptr,
        BuildTransformerEncoderGraphJson(),
        package_path.string(),
        export_options);
    Check(exported.success,
          "TransformerEncoder .cyxmodel export failed: " +
              exported.error_message);

    cyxwiz::SequentialModel imported;
    cyxwiz::ImportOptions import_options;
    import_options.strict_mode = true;

    cyxwiz::ModelImporter importer;
    const cyxwiz::ImportResult imported_result = importer.ImportCyxModel(
        package_path.string(),
        imported,
        import_options);
    Check(imported_result.success,
          "TransformerEncoder .cyxmodel import failed: " +
              imported_result.error_message);
    Check(imported.Size() == 2,
          "TransformerEncoder .cyxmodel import should rebuild two modules");

    const auto imported_params = imported.GetParameters();
    CheckParameterRoundTrip(source_params,
                            imported_params,
                            "TransformerEncoder");

    cyxwiz::SequentialModel bert_sequence_source;
    bert_sequence_source.Add<cyxwiz::EmbeddingModule>(16, 4, 0);
    bert_sequence_source.Add<cyxwiz::PositionalEncodingModule>(4, 4);
    bert_sequence_source.Add<cyxwiz::TransformerEncoderModule>(4, 2, 8, 0.0f, false);
    bert_sequence_source.Add<cyxwiz::TensorShapeModule>(
        cyxwiz::TensorShapeOp::IndexSelect,
        std::vector<size_t>{},
        0,
        std::vector<int>{0});
    bert_sequence_source.Add<cyxwiz::LinearModule>(16, 2, true);
    const auto bert_sequence_params = bert_sequence_source.GetParameters();

    const cyxwiz::ExportResult bert_sequence_exported = exporter.ExportCyxModel(
        bert_sequence_source,
        nullptr,
        nullptr,
        BuildBertSequenceClassifierGraphJson(),
        bert_sequence_path.string(),
        export_options);
    Check(bert_sequence_exported.success,
          "BERT sequence classifier .cyxmodel export failed: " +
              bert_sequence_exported.error_message);

    const cyxwiz::ProbeResult bert_sequence_probe =
        importer.ProbeFile(bert_sequence_path.string());
    Check(bert_sequence_probe.valid,
          "BERT sequence classifier probe should be valid: " +
              bert_sequence_probe.error_message);
    Check(bert_sequence_probe.model_family == "bert_encoder",
          "BERT sequence classifier probe should expose model family");
    Check(bert_sequence_probe.supports_bert_encoder,
          "BERT sequence classifier probe should expose supported encoder flag");
    Check(bert_sequence_probe.bert_encoder_task == "sequence_classification",
          "BERT sequence classifier probe should expose task");
    Check(bert_sequence_probe.bert_encoder_input_kind == "token_ids",
          "BERT sequence classifier probe should expose token-id input kind");
    Check(bert_sequence_probe.bert_encoder_output_contract ==
              "Float32[batch,classes]",
          "BERT sequence classifier probe should expose output contract");
    Check(bert_sequence_probe.bert_encoder_has_attention_mask,
          "BERT sequence classifier probe should expose attention-mask support");
    Check(!bert_sequence_probe.bert_encoder_requires_token_type_ids,
          "BERT sequence classifier probe should fail closed for segment ids");

    cyxwiz::SequentialModel bert_sequence_imported;
    const cyxwiz::ImportResult bert_sequence_import_result =
        importer.ImportCyxModel(bert_sequence_path.string(),
                                bert_sequence_imported,
                                import_options);
    Check(bert_sequence_import_result.success,
          "BERT sequence classifier .cyxmodel import failed: " +
              bert_sequence_import_result.error_message);
    Check(bert_sequence_import_result.model_family == "bert_encoder",
          "BERT sequence classifier import should preserve model family");
    Check(bert_sequence_import_result.bert_encoder_task ==
              "sequence_classification",
          "BERT sequence classifier import should preserve task metadata");
    Check(bert_sequence_import_result.bert_encoder_output_contract ==
              "Float32[batch,classes]",
          "BERT sequence classifier import should preserve output contract");
    CheckParameterRoundTrip(bert_sequence_params,
                            bert_sequence_imported.GetParameters(),
                            "BERT sequence classifier");

    cyxwiz::SequentialModel bert_token_source;
    bert_token_source.Add<cyxwiz::EmbeddingModule>(16, 4, 0);
    bert_token_source.Add<cyxwiz::PositionalEncodingModule>(4, 4);
    bert_token_source.Add<cyxwiz::TransformerEncoderModule>(4, 2, 8, 0.0f, false);
    bert_token_source.Add<cyxwiz::TimeDistributedDenseModule>(4, 3, true);
    const auto bert_token_params = bert_token_source.GetParameters();

    const cyxwiz::ExportResult bert_token_exported = exporter.ExportCyxModel(
        bert_token_source,
        nullptr,
        nullptr,
        BuildBertTokenClassifierGraphJson(),
        bert_token_path.string(),
        export_options);
    Check(bert_token_exported.success,
          "BERT token classifier .cyxmodel export failed: " +
              bert_token_exported.error_message);

    const cyxwiz::ProbeResult bert_token_probe =
        importer.ProbeFile(bert_token_path.string());
    Check(bert_token_probe.valid,
          "BERT token classifier probe should be valid: " +
              bert_token_probe.error_message);
    Check(bert_token_probe.model_family == "bert_encoder",
          "BERT token classifier probe should expose model family");
    Check(bert_token_probe.supports_bert_encoder,
          "BERT token classifier probe should expose supported encoder flag");
    Check(bert_token_probe.bert_encoder_task == "token_classification",
          "BERT token classifier probe should expose task");
    Check(bert_token_probe.bert_encoder_output_contract ==
              "Float32[batch,seq,classes]",
          "BERT token classifier probe should expose output contract");

    cyxwiz::SequentialModel bert_token_imported;
    const cyxwiz::ImportResult bert_token_import_result =
        importer.ImportCyxModel(bert_token_path.string(),
                                bert_token_imported,
                                import_options);
    Check(bert_token_import_result.success,
          "BERT token classifier .cyxmodel import failed: " +
              bert_token_import_result.error_message);
    Check(bert_token_import_result.bert_encoder_task == "token_classification",
          "BERT token classifier import should preserve task metadata");
    Check(bert_token_import_result.bert_encoder_output_contract ==
              "Float32[batch,seq,classes]",
          "BERT token classifier import should preserve output contract");
    CheckParameterRoundTrip(bert_token_params,
                            bert_token_imported.GetParameters(),
                            "BERT token classifier");

    fs::remove_all(root);
    std::cout << "CyxModel TransformerEncoder round-trip test passed\n";
    return 0;
}
