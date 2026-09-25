#include "graph_compiler_dataset_hooks.h"

#include "graph_compiler.h"

namespace cyxwiz {

namespace {
GraphCompilerDatasetHooks& Hooks() {
    static GraphCompilerDatasetHooks hooks;
    return hooks;
}
}  // namespace

void SetGraphCompilerDatasetHooks(GraphCompilerDatasetHooks hooks) { Hooks() = std::move(hooks); }

const GraphCompilerDatasetHooks& GetGraphCompilerDatasetHooks() { return Hooks(); }

bool GraphDatasetIsRegistered(const std::string& dataset_name) {
    const auto& hooks = Hooks();
    return !dataset_name.empty() && hooks.is_dataset_registered && hooks.is_dataset_registered(dataset_name);
}

bool GraphCategoryLabelsFromStructure(const std::string& file_category) {
    const auto& hooks = Hooks();
    if (!hooks.labels_from_structure) return false;
    return hooks.labels_from_structure(file_category).value_or(false);
}

std::optional<PreprocessingDomain> GraphCategoryPreprocessingDomain(const std::string& file_category) {
    const auto& hooks = Hooks();
    if (!hooks.preprocessing_domain) return std::nullopt;
    return hooks.preprocessing_domain(file_category);
}

}  // namespace cyxwiz
