#pragma once

// What the graph compiler needs to know about the host's dataset loaders,
// without depending on them (TOFIX118 P2: the compiler is part of the
// GUI-free training core). The Engine installs hooks backed by its loader
// registry (gui/loaders/data_loader.cpp); hosts without loaders keep the
// defaults: no dataset is registered, categories carry no loader facts.

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace cyxwiz {

enum class PreprocessingDomain;
class ArrowDataset;
class ParquetBackedDataset;

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


// The datasets a graph can name, as the compiler sees them (resolving label
// columns, schemas, row counts, token-window metadata). The Engine installs a
// catalog over its DataRegistry (data_registry_core.cpp); a Server Node
// installs one over the files a job brought. Empty by default.
enum class GraphDatasetKind { Sparse, Image, Audio, Text };

struct GraphTextDatasetInfo {
    size_t num_samples = 0;
    size_t vocab_size = 0;
    int max_length = 0;
};

struct GraphDatasetCatalog {
    std::function<std::shared_ptr<ArrowDataset>(const std::string& name)> arrow_dataset;
    std::function<std::shared_ptr<ParquetBackedDataset>(const std::string& name)> parquet_dataset;
    std::function<bool(const std::string& name, GraphDatasetKind kind)> is_kind;
    std::function<std::optional<GraphTextDatasetInfo>(const std::string& name)> text_info;
    // The file a tabular dataset was loaded from (source fingerprinting).
    std::function<std::optional<std::string>(const std::string& name)> source_path;
};

void SetGraphDatasetCatalog(GraphDatasetCatalog catalog);
GraphDatasetCatalog GetGraphDatasetCatalog();

std::shared_ptr<ArrowDataset> GraphArrowDataset(const std::string& name);
std::shared_ptr<ParquetBackedDataset> GraphParquetDataset(const std::string& name);
bool GraphDatasetIsKind(const std::string& name, GraphDatasetKind kind);
std::optional<GraphTextDatasetInfo> GraphTextDatasetInfoFor(const std::string& name);
std::optional<std::string> GraphDatasetSourcePath(const std::string& name);

}  // namespace cyxwiz
