#pragma once

// What the graph compiler needs to know about the host's dataset loaders,
// without depending on them (TOFIX118 P2: the compiler is part of the
// GUI-free training core). The Engine installs hooks backed by its loader
// registry (gui/loaders/data_loader.cpp); hosts without loaders keep the
// defaults: no dataset is registered, categories carry no loader facts.

#include <functional>
#include <optional>
#include <string>

namespace cyxwiz {

enum class PreprocessingDomain;

struct GraphCompilerDatasetHooks {
    // True when a loaded dataset of this name is held by some loader.
    std::function<bool(const std::string& dataset_name)> is_dataset_registered;
    // For a Data Input file_category: labels come from the file structure
    // (image folders, audio/text layouts) rather than a label column.
    std::function<std::optional<bool>(const std::string& file_category)> labels_from_structure;
    // For a Data Input file_category: the preprocessing domain its loader uses.
    std::function<std::optional<PreprocessingDomain>(const std::string& file_category)> preprocessing_domain;
};

void SetGraphCompilerDatasetHooks(GraphCompilerDatasetHooks hooks);
const GraphCompilerDatasetHooks& GetGraphCompilerDatasetHooks();

// Convenience wrappers with the defaults applied.
bool GraphDatasetIsRegistered(const std::string& dataset_name);
bool GraphCategoryLabelsFromStructure(const std::string& file_category);
std::optional<PreprocessingDomain> GraphCategoryPreprocessingDomain(const std::string& file_category);

}  // namespace cyxwiz
