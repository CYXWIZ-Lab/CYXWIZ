// The Engine's dataset catalog: DataRegistry as its source (dataset_catalog.h).

#include "dataset_catalog.h"

#include "arrow_dataset.h"
#include "data_registry.h"
#include "parquet_backed_dataset.h"
#include "sparse_feature_dataset.h"

#include <cstdint>
#include <functional>

namespace cyxwiz {

namespace {

uint64_t AddressOf(const void* p) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p));
}

uint64_t HashOf(const std::string& text, size_t a, size_t b) {
    uint64_t h = std::hash<std::string>{}(text);
    h ^= static_cast<uint64_t>(a) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    h ^= static_cast<uint64_t>(b) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
}

}  // namespace

std::vector<DatasetCatalogItem> DataRegistry::CatalogItems() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<DatasetCatalogItem> out;
    const auto path_of = [&](const std::string& name) {
        auto it = tabular_source_paths_by_name_.find(name);
        return it == tabular_source_paths_by_name_.end() ? std::string() : it->second;
    };
    for (const auto& [name, ds] : arrow_datasets_) {
        if (!ds) continue;
        out.push_back({name, kBackingArrow, static_cast<size_t>(ds->GetNumRows()), static_cast<size_t>(ds->GetNumColumns()), 0,
                       path_of(name), AddressOf(ds.get())});
    }
    for (const auto& [name, ds] : parquet_backed_datasets_) {
        if (!ds) continue;
        out.push_back({name, kBackingParquet, static_cast<size_t>(ds->GetNumRows()), static_cast<size_t>(ds->GetNumColumns()), 0,
                       path_of(name), AddressOf(ds.get())});
    }
    for (const auto& [name, ds] : sparse_feature_datasets_) {
        if (!ds) continue;
        out.push_back({name, kBackingSparse, static_cast<size_t>(ds->GetNumRows()), static_cast<size_t>(ds->GetNumFeatures()), 0,
                       std::string(), AddressOf(ds.get())});
    }
    for (const auto& [name, e] : image_dataset_entries_)
        out.push_back({name, kBackingImage, e.num_images, 0, e.num_classes, e.folder_path, HashOf(e.folder_path + "|" + e.labels_csv, e.num_images, e.num_classes)});
    for (const auto& [name, e] : audio_dataset_entries_)
        out.push_back({name, kBackingAudio, e.num_samples, 0, e.num_classes, e.folder_path, HashOf(e.folder_path + "|" + e.csv_path, e.num_samples, e.num_classes)});
    for (const auto& [name, e] : text_dataset_entries_)
        out.push_back({name, kBackingText, e.num_samples, 0, e.num_classes, e.source_path, HashOf(e.source_path + "|" + e.text_column, e.num_samples, e.vocab_size)});
    for (const auto& [name, ds] : datasets_) {
        if (!ds) continue;
        const DatasetInfo info = ds->GetInfo();
        out.push_back({name, kBackingLegacy, info.num_samples, 0, info.num_classes, info.path, AddressOf(ds.get())});
    }
    return out;
}

DatasetCatalog& DatasetCatalog::Instance() {
    static DatasetCatalog catalog([] { return DataRegistry::Instance().CatalogItems(); });
    return catalog;
}

}  // namespace cyxwiz
