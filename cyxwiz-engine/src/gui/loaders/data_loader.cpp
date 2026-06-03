#include "data_loader.h"
#include "audio_loader.h"
#include "image_loader.h"
#include "tabular_loader.h"
#include "text_loader.h"

#include "../../core/graph_compiler.h"
#include "../../core/synthetic_batch.h"

#include <memory>
#include <string>
#include <utility>

namespace cyxwiz::loaders {

namespace {

std::string ShapeString(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        return "";
    }
    std::string out;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) out += "x";
        out += std::to_string(shape[i]);
    }
    return out;
}

// Heap-allocated loader instances owned by `owners`; `list` holds raw
// pointers into that vector so callers never deal with unique_ptrs.
// Initialized once at first access via the static-local IIFE below.
struct Registry {
    std::vector<std::unique_ptr<DataLoader>> owners;
    std::vector<DataLoader*> list;
};

Registry& GetRegistry() {
    static Registry r = [] {
        Registry tmp;
        auto push = [&tmp](std::unique_ptr<DataLoader> l) {
            tmp.list.push_back(l.get());
            tmp.owners.push_back(std::move(l));
        };
        // Order of registration is observable via All(); keep it stable.
        // Commit 1: TabularLoader. Commit 2: TextLoader.
        // Commit 3: ImageLoader. Commit 4: AudioLoader.
        // Video remains deliberately unregistered — the guard at the
        // top of DataInputDialog::Apply fails the load loudly long
        // before reaching the dispatch.
        push(std::make_unique<TabularLoader>());
        push(std::make_unique<TextLoader>());
        push(std::make_unique<ImageLoader>());
        push(std::make_unique<AudioLoader>());
        return tmp;
    }();
    return r;
}

}  // namespace

SyntheticBatch MakeSyntheticForDomain(
    const cyxwiz::TrainingConfiguration& config,
    cyxwiz::PreprocessingDomain domain,
    uint32_t seed,
    const char* domain_name) {
    auto local_config = config;
    local_config.preprocessing_domain = domain;

    auto generated = cyxwiz::MakeSyntheticBatch(local_config, seed);

    SyntheticBatch out;
    out.is_empty = false;
    out.features = std::move(generated.features);
    out.labels = std::move(generated.labels);
    const auto& feature_shape = out.features.Shape();
    out.sample_count = feature_shape.empty() ? 0 : feature_shape.front();
    out.summary = std::string(domain_name ? domain_name : "Dataset") +
        " synthetic batch: features=[" + ShapeString(feature_shape) +
        "], labels=[" + ShapeString(out.labels.Shape()) + "]";
    return out;
}

DataLoader* GetByCategory(FileCategory cat) {
    auto& r = GetRegistry();
    for (auto* l : r.list) {
        if (l->Category() == cat) return l;
    }
    // TimeSeries shares the Tabular load path; a dedicated
    // TimeSeriesLoader subclass may land in a later commit for
    // timeseries-specific node params, but the load itself is the
    // same as tabular.
    if (cat == FileCategory::TimeSeries) {
        for (auto* l : r.list) {
            if (l->Category() == FileCategory::Tabular) return l;
        }
    }
    return nullptr;
}

DataLoader* GetByRegisteredDataset(const std::string& name) {
    // Linear scan across 4 loaders, each checking a hashmap via
    // IsRegistered — effectively O(1) for practical loader counts. A
    // name → loader sidecar inside DataRegistry would shave one hash
    // lookup but requires DataRegistry to know the FileCategory enum
    // (currently owned by the loaders module); deferred as future work
    // since the performance delta is negligible.
    // Text CSVs also register their raw Arrow table for Cat-1 text
    // operators. Prefer explicit domain loaders over the generic
    // TabularLoader when both maps contain the same original dataset
    // name. Materialized outputs have only Arrow backing, so they still
    // route through TabularLoader.
    for (auto* l : GetRegistry().list) {
        if (l->Category() != FileCategory::Tabular && l->IsRegistered(name)) {
            return l;
        }
    }
    for (auto* l : GetRegistry().list) {
        if (l->Category() == FileCategory::Tabular && l->IsRegistered(name)) {
            return l;
        }
    }
    return nullptr;
}

DataLoader* GetByBackendTag(int backend) {
    // Tabular covers backend 1 (Arrow in-mem) + backend 2 (Parquet
    // disk-backed). Map 2 → Tabular explicitly since TabularLoader's
    // BackendTag() is 1. Other loaders have unique tags.
    if (backend == 2) {
        for (auto* l : GetRegistry().list) {
            if (l->Category() == FileCategory::Tabular) return l;
        }
        return nullptr;
    }
    for (auto* l : GetRegistry().list) {
        if (l->BackendTag() == backend) return l;
    }
    return nullptr;
}

const std::vector<DataLoader*>& All() {
    return GetRegistry().list;
}

FileCategory FileCategoryFromString(const std::string& s) {
    if (s == "image")      return FileCategory::Image;
    if (s == "audio")      return FileCategory::Audio;
    if (s == "video")      return FileCategory::Video;
    if (s == "text")       return FileCategory::Text;
    if (s == "timeseries") return FileCategory::TimeSeries;
    return FileCategory::Tabular;  // "tabular" + unknown default
}

}  // namespace cyxwiz::loaders
