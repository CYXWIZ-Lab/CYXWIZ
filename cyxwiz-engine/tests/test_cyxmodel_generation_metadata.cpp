#include "../src/core/formats/cyxmodel_format.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

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
        fs::temp_directory_path() / "cyxwiz_cyxmodel_generation_metadata_test";
    const fs::path package_path = root / "causal_lm.cyxmodel";
    const fs::path default_package_path = root / "default_model.cyxmodel";

    fs::remove_all(root);
    fs::create_directories(root);

    cyxwiz::formats::CyxModelFormat format;
    cyxwiz::TrainingConfig config;
    config.dataset_name = "tiny_language_model";
    config.input_shape = {8};
    config.num_classes = 32;

    cyxwiz::ModelManifest manifest;
    manifest.model_name = "Tiny causal LM";
    manifest.model_type = "SequentialModel";
    manifest.model_family = "causal_lm";
    manifest.supports_generation = true;
    manifest.generation_output_contract = "Float32[1,seq,vocab]";
    manifest.has_graph = true;

    cyxwiz::ExportOptions options;
    options.include_graph = true;

    const bool created = format.Create(
        package_path.string(),
        manifest,
        "{\"nodes\":[]}",
        config,
        nullptr,
        {},
        {},
        nullptr,
        options);
    Check(created, "failed to create causal LM package: " +
                       format.GetLastError());

    const cyxwiz::ProbeResult probe = format.Probe(package_path.string());
    Check(probe.valid, "probe should be valid: " + probe.error_message);
    Check(probe.model_name == "Tiny causal LM",
          "probe should preserve model name");
    Check(probe.model_family == "causal_lm",
          "probe should preserve model family");
    Check(probe.supports_generation,
          "probe should preserve generation support flag");
    Check(probe.generation_output_contract == "Float32[1,seq,vocab]",
          "probe should preserve generation output contract");

    cyxwiz::ModelManifest default_manifest;
    default_manifest.model_name = "Default model";
    default_manifest.model_type = "SequentialModel";

    const bool default_created = format.Create(
        default_package_path.string(),
        default_manifest,
        "{\"nodes\":[]}",
        config,
        nullptr,
        {},
        {},
        nullptr,
        options);
    Check(default_created, "failed to create default package: " +
                               format.GetLastError());

    const cyxwiz::ProbeResult default_probe =
        format.Probe(default_package_path.string());
    Check(default_probe.valid,
          "default probe should be valid: " + default_probe.error_message);
    Check(default_probe.model_family.empty(),
          "default model family should remain empty");
    Check(!default_probe.supports_generation,
          "default generation support should remain false");
    Check(default_probe.generation_output_contract.empty(),
          "default generation contract should remain empty");

    fs::remove_all(root);
    std::cout << "CyxModel generation metadata test passed\n";
    return 0;
}
