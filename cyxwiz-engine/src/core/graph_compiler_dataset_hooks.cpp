#include "graph_compiler_dataset_hooks.h"

#include "graph_compiler.h"
#include "arrow_dataset.h"
#include "parquet_backed_dataset.h"

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

namespace {
GraphDatasetCatalog& Catalog() {
    static GraphDatasetCatalog catalog;
    return catalog;
}
}  // namespace

void SetGraphDatasetCatalog(GraphDatasetCatalog catalog) { Catalog() = std::move(catalog); }

GraphDatasetCatalog GetGraphDatasetCatalog() { return Catalog(); }

std::shared_ptr<ArrowDataset> GraphArrowDataset(const std::string& name) {
    const auto& catalog = Catalog();
    return (!name.empty() && catalog.arrow_dataset) ? catalog.arrow_dataset(name) : nullptr;
}

std::shared_ptr<ParquetBackedDataset> GraphParquetDataset(const std::string& name) {
    const auto& catalog = Catalog();
    return (!name.empty() && catalog.parquet_dataset) ? catalog.parquet_dataset(name) : nullptr;
}

bool GraphDatasetIsKind(const std::string& name, GraphDatasetKind kind) {
    const auto& catalog = Catalog();
    return !name.empty() && catalog.is_kind && catalog.is_kind(name, kind);
}

std::optional<GraphTextDatasetInfo> GraphTextDatasetInfoFor(const std::string& name) {
    const auto& catalog = Catalog();
    if (name.empty() || !catalog.text_info) return std::nullopt;
    return catalog.text_info(name);
}

std::optional<std::string> GraphDatasetSourcePath(const std::string& name) {
    const auto& catalog = Catalog();
    if (name.empty() || !catalog.source_path) return std::nullopt;
    return catalog.source_path(name);
}

}  // namespace cyxwiz
