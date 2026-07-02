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

std::string BuildTransformerDecoderGraphJson() {
    using json = nlohmann::json;
    json graph;
    graph["nodes"] = json::array();
    graph["links"] = json::array();

    graph["nodes"].push_back({
        {"id", 1},
        {"type", static_cast<int>(gui::NodeType::DatasetInput)},
        {"name", "Input"},
        {"parameters", {
            {"dataset_name", "transformer_decoder_roundtrip_dataset"},
            {"dataset", "transformer_decoder_roundtrip_dataset"},
            {"shape", "[4]"}
        }}
    });
    graph["nodes"].push_back({
        {"id", 2},
        {"type", static_cast<int>(gui::NodeType::TransformerDecoder)},
        {"name", "Decoder"},
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
        {"type", static_cast<int>(gui::NodeType::Flatten)},
        {"name", "Flatten"},
        {"parameters", nlohmann::json::object()}
    });
    graph["nodes"].push_back({
        {"id", 4},
        {"type", static_cast<int>(gui::NodeType::Dense)},
        {"name", "Head"},
        {"parameters", {{"units", "2"}}}
    });
    graph["nodes"].push_back({
        {"id", 5},
        {"type", static_cast<int>(gui::NodeType::Output)},
        {"name", "Output"},
        {"parameters", {{"num_classes", "2"}}}
    });
    graph["nodes"].push_back({
        {"id", 6},
        {"type", static_cast<int>(gui::NodeType::MSELoss)},
        {"name", "MSE"},
        {"parameters", json::object()}
    });
    graph["nodes"].push_back({
        {"id", 7},
        {"type", static_cast<int>(gui::NodeType::Adam)},
        {"name", "Adam"},
        {"parameters", {{"learning_rate", "0.001"}}}
    });

    graph["links"].push_back({{"id", 101}, {"from_node", 1}, {"to_node", 2}});
    graph["links"].push_back({{"id", 102}, {"from_node", 2}, {"to_node", 3}});
    graph["links"].push_back({{"id", 103}, {"from_node", 3}, {"to_node", 4}});
    graph["links"].push_back({{"id", 104}, {"from_node", 4}, {"to_node", 5}});
    graph["links"].push_back({{"id", 105}, {"from_node", 4}, {"to_node", 6}});
    graph["links"].push_back({{"id", 106}, {"from_node", 7}, {"to_node", 6}});

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

} // namespace

int main() {
    namespace fs = std::filesystem;

    const fs::path root =
        fs::temp_directory_path() / "cyxwiz_cyxmodel_transformer_decoder_roundtrip";
    const fs::path package_path = root / "transformer_decoder.cyxmodel";
    fs::remove_all(root);
    fs::create_directories(root);

    cyxwiz::SequentialModel source;
    source.Add<cyxwiz::TransformerDecoderModule>(4, 2, 8, 0.0f, false);
    source.Add<cyxwiz::FlattenModule>();
    source.Add<cyxwiz::LinearModule>(16, 2, true);

    const auto source_params = source.GetParameters();
    Check(!source_params.empty(),
          "source TransformerDecoder model should expose parameters");

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
        BuildTransformerDecoderGraphJson(),
        package_path.string(),
        export_options);
    Check(exported.success,
          "TransformerDecoder .cyxmodel export failed: " +
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
          "TransformerDecoder .cyxmodel import failed: " +
              imported_result.error_message);
    Check(imported.Size() == 3,
          "TransformerDecoder .cyxmodel import should rebuild three modules");

    const auto imported_params = imported.GetParameters();
    Check(imported_params.size() == source_params.size(),
          "TransformerDecoder parameter count should round-trip");
    for (const auto& [name, tensor] : source_params) {
        Check(imported_params.count(name) == 1,
              "imported TransformerDecoder parameter missing: " + name);
        CheckTensorValues(imported_params.at(name), tensor, name);
    }

    const float input_values[] = {
        0.2f, -0.1f, 0.4f, 0.7f,
        -0.3f, 0.5f, 0.1f, -0.2f,
        0.6f, 0.2f, -0.4f, 0.3f,
        -0.5f, 0.8f, 0.2f, -0.1f,
    };
    const cyxwiz::Tensor input({1, 4, 4}, input_values,
                               cyxwiz::DataType::Float32);
    const cyxwiz::Tensor source_output = source.Forward(input);
    const cyxwiz::Tensor imported_output = imported.Forward(input);
    CheckTensorValues(imported_output,
                      source_output,
                      "TransformerDecoder imported inference output");

    fs::remove_all(root);
    std::cout << "CyxModel TransformerDecoder round-trip test passed\n";
    return 0;
}
