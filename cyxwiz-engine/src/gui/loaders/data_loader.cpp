#include "data_loader.h"
#include "tabular_loader.h"
#include "text_loader.h"

#include <memory>

namespace cyxwiz::loaders {

namespace {

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
        // Commits 3-4 will add Image / Audio.
        push(std::make_unique<TabularLoader>());
        push(std::make_unique<TextLoader>());
        return tmp;
    }();
    return r;
}

}  // namespace

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
    for (auto* l : GetRegistry().list) {
        if (l->IsRegistered(name)) return l;
    }
    return nullptr;
}

const std::vector<DataLoader*>& All() {
    return GetRegistry().list;
}

}  // namespace cyxwiz::loaders
