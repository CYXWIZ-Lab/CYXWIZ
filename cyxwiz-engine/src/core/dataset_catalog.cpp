#include "dataset_catalog.h"

#include <algorithm>

namespace cyxwiz {

namespace {

constexpr const char* kMaterializedSuffix = "__materialized";

bool EndsWith(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

uint64_t Combine(uint64_t seed, uint64_t v) {
    return seed ^ (v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

}  // namespace

const char* StorageText(DatasetStorageKind storage) {
    switch (storage) {
        case DatasetStorageKind::InMemoryArrow: return "table (in memory)";
        case DatasetStorageKind::DiskBackedParquet: return "table (on disk, Parquet)";
        case DatasetStorageKind::SparseFeatureCSR: return "sparse features";
        case DatasetStorageKind::ImageCached: return "images";
        case DatasetStorageKind::AudioCached: return "audio";
        case DatasetStorageKind::TextCached: return "text";
        case DatasetStorageKind::AdapterDefined: return "adapter";
        case DatasetStorageKind::Unknown: break;
    }
    return "dataset";
}

DatasetCatalog::DatasetCatalog(Source source) : source_(std::move(source)) {}

std::vector<DatasetEntry> DatasetCatalog::Merge(const std::vector<DatasetCatalogItem>& items) const {
    std::map<std::string, DatasetEntry> by_name;
    std::map<std::string, uint64_t> identity;
    for (const auto& it : items) {
        DatasetEntry& e = by_name[it.name];
        e.name = it.name;
        e.backings |= it.backing;
        // Rows and columns from the table when there is one, else from the entry.
        const bool table = it.backing == kBackingArrow || it.backing == kBackingParquet || it.backing == kBackingSparse;
        if (table || e.rows == 0) {
            if (it.rows) e.rows = it.rows;
            if (it.columns) e.columns = it.columns;
        }
        if (it.classes) e.classes = it.classes;
        if (e.source_path.empty()) e.source_path = it.source_path;
        identity[it.name] = Combine(Combine(identity[it.name], it.backing), it.identity);
    }
    std::vector<DatasetEntry> out;
    out.reserve(by_name.size());
    for (auto& [name, e] : by_name) {
        if (e.Has(kBackingArrow)) e.storage = DatasetStorageKind::InMemoryArrow;
        else if (e.Has(kBackingParquet)) e.storage = DatasetStorageKind::DiskBackedParquet;
        else if (e.Has(kBackingSparse)) e.storage = DatasetStorageKind::SparseFeatureCSR;
        else if (e.Has(kBackingImage)) e.storage = DatasetStorageKind::ImageCached;
        else if (e.Has(kBackingAudio)) e.storage = DatasetStorageKind::AudioCached;
        else if (e.Has(kBackingText)) e.storage = DatasetStorageKind::TextCached;
        if (e.Has(kBackingImage)) e.modality = DatasetModality::Image;
        else if (e.Has(kBackingAudio)) e.modality = DatasetModality::Audio;
        else if (e.Has(kBackingText)) e.modality = DatasetModality::Text;
        else if (e.backings & (kBackingArrow | kBackingParquet | kBackingSparse)) e.modality = DatasetModality::Tabular;
        e.materialized = EndsWith(name, kMaterializedSuffix);
        e.generation = identity[name];  // the identity for now; List() puts the generation in
        out.push_back(std::move(e));
    }
    return out;
}

std::vector<DatasetEntry> DatasetCatalog::List() const {
    std::vector<DatasetEntry> out = Merge(source_ ? source_() : std::vector<DatasetCatalogItem>{});
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& e : out) {
        auto it = generation_.find(e.name);
        e.generation = it == generation_.end() ? 0 : it->second;
        auto l = labels_.find(e.name);
        if (l != labels_.end()) e.label = l->second;
    }
    return out;
}

void DatasetCatalog::SetLabels(std::map<std::string, std::string> labels) {
    std::lock_guard<std::mutex> lock(mutex_);
    labels_ = std::move(labels);
}

std::string DatasetCatalog::NameFor(const std::string& label_or_name) const {
    if (label_or_name.empty()) return {};
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& [name, label] : labels_)
            if (label == label_or_name) return name;
    }
    return Resolve(label_or_name) ? label_or_name : std::string();
}

std::optional<DatasetEntry> DatasetCatalog::Resolve(const std::string& name) const {
    if (name.empty()) return std::nullopt;
    for (auto& e : List())
        if (e.name == name) return e;
    return std::nullopt;
}

uint64_t DatasetCatalog::GenerationOf(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = generation_.find(name);
    return it == generation_.end() ? 0 : it->second;
}

size_t DatasetCatalog::Pump(std::chrono::milliseconds interval) {
    const auto now = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (interval.count() > 0 && last_pump_ != std::chrono::steady_clock::time_point{} && now - last_pump_ < interval) return 0;
        last_pump_ = now;
    }
    // The registry is read without the catalog lock (the source takes its own).
    const std::vector<DatasetEntry> current = Merge(source_ ? source_() : std::vector<DatasetCatalogItem>{});
    std::vector<DatasetChange> changes;
    std::vector<Subscriber> listeners;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::map<std::string, std::pair<uint64_t, DatasetEntry>> next;
        for (const auto& e : current) {
            const uint64_t identity = e.generation;  // Merge() put the identity here
            DatasetEntry entry = e;
            auto old = seen_.find(e.name);
            if (old == seen_.end() || old->second.first != identity) {
                const uint64_t g = ++next_generation_;
                generation_[e.name] = g;
                entry.generation = g;
                changes.push_back({old == seen_.end() ? DatasetChange::Kind::Added : DatasetChange::Kind::Replaced, e.name, g, entry});
            } else {
                entry.generation = generation_[e.name];
            }
            next.emplace(e.name, std::make_pair(identity, entry));
        }
        for (const auto& [name, old] : seen_) {
            if (next.count(name)) continue;
            const uint64_t g = ++next_generation_;
            generation_[name] = g;  // kept: a dataset that comes back gets a newer one
            changes.push_back({DatasetChange::Kind::Removed, name, g, old.second});
        }
        seen_ = std::move(next);
        // Drop listeners whose owner is gone; copy the rest to call outside the lock.
        subscribers_.erase(std::remove_if(subscribers_.begin(), subscribers_.end(),
                                          [](const Subscriber& s) { return s.owned && s.owner.expired(); }),
                           subscribers_.end());
        if (!changes.empty()) listeners = subscribers_;
    }
    for (const auto& change : changes)
        for (const auto& s : listeners) {
            std::shared_ptr<const void> alive;
            if (s.owned) {
                alive = s.owner.lock();
                if (!alive) continue;
            }
            {
                // Unsubscribed by an earlier listener in this round: skip.
                std::lock_guard<std::mutex> lock(mutex_);
                if (std::none_of(subscribers_.begin(), subscribers_.end(), [&](const Subscriber& x) { return x.id == s.id; })) continue;
            }
            s.listener(change);
        }
    return changes.size();
}

int DatasetCatalog::Subscribe(std::weak_ptr<const void> owner, Listener listener) {
    std::lock_guard<std::mutex> lock(mutex_);
    const bool owned = !owner.expired();
    const int id = next_id_++;
    subscribers_.push_back({id, std::move(owner), owned, std::move(listener)});
    return id;
}

void DatasetCatalog::Unsubscribe(int id) {
    std::lock_guard<std::mutex> lock(mutex_);
    subscribers_.erase(std::remove_if(subscribers_.begin(), subscribers_.end(), [&](const Subscriber& s) { return s.id == id; }),
                       subscribers_.end());
}

}  // namespace cyxwiz
