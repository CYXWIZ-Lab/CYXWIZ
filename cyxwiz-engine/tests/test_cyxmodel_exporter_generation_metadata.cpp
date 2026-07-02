#include "../src/core/formats/cyxmodel_format.h"
#include "../src/core/model_exporter.h"

#include <cyxwiz/sequential.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace cyxwiz {

std::string ReadTreeModelArtifactType(const std::string&, std::string* error) {
    if (error) {
        error->clear();
    }
    return {};
}

} // namespace cyxwiz

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

} // namespace

int main() {
    namespace fs = std::filesystem;

    const fs::path root =
        fs::temp_directory_path() /
        "cyxwiz_cyxmodel_exporter_generation_metadata_test";
    const fs::path package_path = root / "exported_causal_lm.cyxmodel";

    fs::remove_all(root);
    fs::create_directories(root);

    cyxwiz::SequentialModel model;
    model.Add<cyxwiz::LinearModule>(4, 8, true);

    cyxwiz::ExportOptions options;
    options.format = cyxwiz::ModelFormat::CyxModel;
    options.model_name = "Exporter causal LM";
    options.include_graph = true;
    options.include_training_history = false;
    options.include_optimizer_state = false;
    options.include_sequence_assets = true;
    options.sequence_create_causal_lm_targets = true;
    options.sequence_max_sequence_length = 8;

    cyxwiz::ModelExporter exporter;
    const cyxwiz::ExportResult exported = exporter.ExportCyxModel(
        model,
        nullptr,
        nullptr,
        "{\"nodes\":[]}",
        package_path.string(),
        options);
    Check(exported.success,
          "causal LM export failed: " + exported.error_message);

    cyxwiz::formats::CyxModelFormat format;
    const cyxwiz::ProbeResult probe = format.Probe(package_path.string());
    Check(probe.valid, "probe should be valid: " + probe.error_message);
    Check(probe.model_name == "Exporter causal LM",
          "probe should preserve exported model name");
    Check(probe.model_family == "causal_lm",
          "exporter should mark causal LM model family");
    Check(probe.supports_generation,
          "exporter should mark causal LM generation support");
    Check(probe.generation_output_contract == "Float32[1,seq,vocab]",
          "exporter should mark causal LM generation output contract");
    Check(probe.sequence_create_causal_lm_targets,
          "probe should preserve causal LM target creation flag");

    fs::remove_all(root);
    std::cout << "CyxModel exporter generation metadata test passed\n";
    return 0;
}
