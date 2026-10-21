#include "../src/core/model_exporter.h"
#include "../src/core/model_importer.h"
#include "../src/core/graph_compiler.h"
#include "../src/gui/loaders/data_loader.h"

#include <cyxwiz/sequential.h>
#include <cyxwiz/tensor.h>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
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

std::string BuildLayerNormGraphJson() {
    using json = nlohmann::json;
    json graph;
    graph["nodes"] = json::array();
    graph["links"] = json::array();

    graph["nodes"].push_back({
        {"id", 1},
        {"type", static_cast<int>(gui::NodeType::DatasetInput)},
        {"name", "Input"},
        {"parameters", {
            {"dataset_name", "layernorm_roundtrip_dataset"},
            {"dataset", "layernorm_roundtrip_dataset"},
            {"shape", "[4]"}
        }}
    });
    graph["nodes"].push_back({
        {"id", 2},
        {"type", static_cast<int>(gui::NodeType::LayerNorm)},
        {"name", "LayerNorm"},
        {"parameters", {
            {"normalized_shape", "4"},
            {"epsilon", "1e-5"},
            {"elementwise_affine", "true"}
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

void WriteBinaryProbeFixture(const std::filesystem::path& path) {
    using json = nlohmann::json;
    const json metadata = {
        {"metadata", {
            {"name", "Binary probe fixture"},
            {"author", "CyxWiz"},
            {"description", "CYXW probe regression fixture"}
        }},
        {"modules", json::array({{
            {"name", "Linear(4 -> 2)"},
            {"has_parameters", true},
            {"parameters", json::array({
                {{"name", "weight"}, {"shape", json::array({2, 4})}},
                {{"name", "bias"}, {"shape", json::array({2})}}
            })}
        }})}
    };
    const std::string metadata_text = metadata.dump();
    const uint32_t magic = 0x43595857;
    const uint32_t version = 2;
    const uint64_t metadata_length = metadata_text.size();
    const size_t module_count = 1;

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    output.write(reinterpret_cast<const char*>(&version), sizeof(version));
    output.write(reinterpret_cast<const char*>(&metadata_length),
                 sizeof(metadata_length));
    output.write(metadata_text.data(),
                 static_cast<std::streamsize>(metadata_text.size()));
    output.write(reinterpret_cast<const char*>(&module_count),
                 sizeof(module_count));
}

} // namespace

int main() {
    namespace fs = std::filesystem;

    const fs::path root =
        fs::temp_directory_path() / "cyxwiz_cyxmodel_layernorm_roundtrip";
    const fs::path package_path = root / "layernorm.cyxmodel";
    fs::remove_all(root);
    fs::create_directories(root);

    cyxwiz::SequentialModel source;
    source.Add<cyxwiz::LayerNormModule>(std::vector<int>{4}, 1e-5f, true);
    source.Add<cyxwiz::LinearModule>(4, 2, true);

    const float gamma_values[] = {0.5f, 1.5f, -0.25f, 2.0f};
    const float beta_values[] = {0.1f, -0.2f, 0.3f, -0.4f};
    std::map<std::string, cyxwiz::Tensor> params = source.GetParameters();
    params["layer0.gamma"] =
        cyxwiz::Tensor({4}, gamma_values, cyxwiz::DataType::Float32);
    params["layer0.beta"] =
        cyxwiz::Tensor({4}, beta_values, cyxwiz::DataType::Float32);
    source.SetParameters(params);

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
        BuildLayerNormGraphJson(),
        package_path.string(),
        export_options);
    Check(exported.success,
          "LayerNorm .cyxmodel export failed: " + exported.error_message);

    cyxwiz::SequentialModel imported;
    cyxwiz::ImportOptions import_options;
    import_options.strict_mode = true;

    cyxwiz::ModelImporter importer;
    const fs::path binary_path = root / "probe.cyxmodel";
    WriteBinaryProbeFixture(binary_path);
    const cyxwiz::ProbeResult binary_probe =
        importer.ProbeFile(binary_path.string());
    Check(binary_probe.valid,
          "binary CYXW probe should not require a directory manifest: " +
              binary_probe.error_message);
    Check(binary_probe.model_name == "Binary probe fixture",
          "binary CYXW probe should expose its metadata name");
    Check(binary_probe.format_version == "CYXW v2",
          "binary CYXW probe should expose its format version");
    Check(binary_probe.num_layers == 1,
          "binary CYXW probe should expose its module count");
    Check(binary_probe.num_parameters == 10,
          "binary CYXW probe should count metadata parameters");

    const cyxwiz::ImportResult imported_result = importer.ImportCyxModel(
        package_path.string(),
        imported,
        import_options);
    Check(imported_result.success,
          "LayerNorm .cyxmodel import failed: " +
              imported_result.error_message);
    Check(imported.Size() == 2,
          "LayerNorm .cyxmodel import should rebuild two modules from graph");

    const auto imported_params = imported.GetParameters();
    Check(imported_params.count("layer0.gamma") == 1,
          "imported LayerNorm gamma should exist");
    Check(imported_params.count("layer0.beta") == 1,
          "imported LayerNorm beta should exist");
    const float* imported_gamma =
        imported_params.at("layer0.gamma").Data<float>();
    const float* imported_beta =
        imported_params.at("layer0.beta").Data<float>();
    for (size_t i = 0; i < 4; ++i) {
        CheckNear(imported_gamma[i], gamma_values[i], 1e-6f,
                  "LayerNorm gamma .cyxmodel round-trip");
        CheckNear(imported_beta[i], beta_values[i], 1e-6f,
                  "LayerNorm beta .cyxmodel round-trip");
    }

    fs::remove_all(root);
    std::cout << "CyxModel LayerNorm round-trip test passed\n";
    return 0;
}
