#pragma once

// The dataset catalog (TOFIX134 P3 foundation, dashboard_architecture.md L1):
// the one place that lists every dataset the Engine holds (in-memory Arrow,
// disk-backed Parquet, sparse features, image / audio / text entries, the
// old map) and says what each is. Consumers (Data Studio picker, query
// service, dashboards, the graph compiler's hook, the loader lookup) ask it
// instead of probing the registry maps one by one.
//
// Change events: Pump() runs on the UI thread, compares the registry with
// what it saw last, bumps a per-dataset generation for each dataset added,
// replaced or removed, and calls the subscribers there, never under a
// registry lock. Every registry path is covered because the catalog looks
// at the result, not at each call that changed it.

#include "dataset_partitions.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

// Which registry maps hold a name (bit flags).
enum DatasetBacking : unsigned {
    kBackingArrow = 1u << 0,
    kBackingParquet = 1u << 1,
    kBackingSparse = 1u << 2,
    kBackingImage = 1u << 3,
    kBackingAudio = 1u << 4,
    kBackingText = 1u << 5,
    kBackingLegacy = 1u << 6,  // the old DatasetHandle map (being retired)
};

// One registry map entry as the registry reports it.
struct DatasetCatalogItem {
    std::string name;
    DatasetBacking backing = kBackingArrow;
    size_t rows = 0;        // rows, files, clips or samples
    size_t columns = 0;     // tabular columns (0 for files)
    size_t classes = 0;     // image / audio / text classes
    std::string source_path;
    uint64_t identity = 0;  // changes when the entry is replaced (object address or content hash)
};

struct DatasetEntry {
    std::string name;
    unsigned backings = 0;
    // What backs the table (the compiler's question): Arrow > Parquet >
    // sparse > image > audio > text.
    DatasetStorageKind storage = DatasetStorageKind::Unknown;
    // What kind of data it is (the loaders' question): an image, audio or
    // text entry wins over the raw table a text CSV also registers.
    DatasetModality modality = DatasetModality::Unknown;
    size_t rows = 0, columns = 0, classes = 0;
    std::string source_path;
    bool materialized = false;  // a pipeline result ("..__materialized")
    // What people call it: the Data Input node's name ("Spotify"); empty
    // when no graph names it. Queries may name a table by either.
    std::string label;
    const std::string& Shown() const { return label.empty() ? name : label; }
    // The graph's target for it: the Data Input's label column (empty: none).
    std::string target_column;
    uint64_t generation = 0;    // bumped by Pump() on each change; 0 = not seen yet
    bool Has(DatasetBacking b) const { return (backings & b) != 0; }
};

struct DatasetChange {
    enum class Kind { Added, Replaced, Removed };
    Kind kind = Kind::Added;
    std::string name;
    uint64_t generation = 0;
    DatasetEntry entry;  // Removed: the last entry seen
};

// "in memory", "on disk (Parquet)", "images", ...
const char* StorageText(DatasetStorageKind storage);

class DatasetCatalog {
public:
    using Source = std::function<std::vector<DatasetCatalogItem>()>;
    using Listener = std::function<void(const DatasetChange&)>;

    // The Engine's catalog over DataRegistry.
    static DatasetCatalog& Instance();
    // A catalog over any source (tests, a Server Node job).
    explicit DatasetCatalog(Source source);

    // Every dataset now (one entry per name, sorted by name). Thread-safe.
    std::vector<DatasetEntry> List() const;
    std::optional<DatasetEntry> Resolve(const std::string& name) const;

    // UI thread: at most every `interval` (0: now), compare with the last
    // look and deliver the changes. Returns how many changed.
    size_t Pump(std::chrono::milliseconds interval = std::chrono::milliseconds(200));
    uint64_t GenerationOf(const std::string& name) const;

    // The listener runs on the UI thread inside Pump(); it is dropped when
    // `owner` expires or on Unsubscribe. A listener may call the catalog,
    // subscribe or unsubscribe.
    int Subscribe(std::weak_ptr<const void> owner, Listener listener);
    void Unsubscribe(int id);

    // UI thread: dataset name -> label from the graph (Data Input node names).
    // Replaces the previous labels; List() and Resolve() then carry them.
    void SetLabels(std::map<std::string, std::string> labels);
    // UI thread: dataset name -> the Data Input's label column (the contract's target).
    void SetTargets(std::map<std::string, std::string> targets);
    // UI thread: dataset name -> the file its Data Input reads (when the registry does not know it).
    void SetSources(std::map<std::string, std::string> sources);
    // The dataset a label (or a name) stands for; empty when none.
    std::string NameFor(const std::string& label_or_name) const;

private:
    std::vector<DatasetEntry> Merge(const std::vector<DatasetCatalogItem>& items) const;

    Source source_;
    mutable std::mutex mutex_;  // guards the members below; never held while calling out
    std::map<std::string, std::pair<uint64_t, DatasetEntry>> seen_;  // name -> (identity, entry)
    std::map<std::string, uint64_t> generation_;
    std::map<std::string, std::string> labels_;
    std::map<std::string, std::string> targets_;
    std::map<std::string, std::string> sources_;
    uint64_t next_generation_ = 0;
    std::chrono::steady_clock::time_point last_pump_{};
    struct Subscriber {
        int id;
        std::weak_ptr<const void> owner;
        bool owned;
        Listener listener;
    };
    std::vector<Subscriber> subscribers_;
    int next_id_ = 1;
};

}  // namespace cyxwiz
