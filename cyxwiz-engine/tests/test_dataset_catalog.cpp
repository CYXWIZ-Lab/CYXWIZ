// Dataset catalog (TOFIX134 P3 foundation, commit 1): one list over every
// registry map, one rule for what a dataset is, change events with
// generations delivered outside any lock.

#include "../src/core/dataset_catalog.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace cyxwiz;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

// A fake registry: its own lock is held while the catalog reads it, like DataRegistry.
struct FakeRegistry {
    std::mutex mutex;
    std::vector<DatasetCatalogItem> items;
    std::vector<DatasetCatalogItem> Snapshot() {
        std::lock_guard<std::mutex> lock(mutex);
        return items;
    }
};
}  // namespace

int main() {
    FakeRegistry reg;
    reg.items = {
        {"Spotify", kBackingArrow, 8582, 15, 0, "D:/data/spotify.csv", 101},
        {"MNIST", kBackingParquet, 70000, 785, 0, "D:/data/mnist.csv", 102},
        // A text CSV registers its text entry and its raw table under one name.
        {"Reviews", kBackingText, 50000, 0, 2, "D:/data/reviews.csv", 103},
        {"Reviews", kBackingArrow, 50000, 2, 0, "D:/data/reviews.csv", 104},
        {"Cats", kBackingImage, 2000, 0, 2, "D:/data/cats", 105},
        {"tfidf__materialized", kBackingSparse, 50000, 20000, 0, "", 106},
        {"old", kBackingLegacy, 10, 0, 0, "", 107},
    };
    DatasetCatalog catalog([&] { return reg.Snapshot(); });

    // List: one entry per name, sorted, storage and modality by one rule.
    auto list = catalog.List();
    Check(list.size() == 6, "six datasets (Reviews merged)");
    Check(list[0].name == "Cats" && list[1].name == "MNIST" && list.back().name == "tfidf__materialized", "sorted by name");
    auto reviews = catalog.Resolve("Reviews");
    Check(reviews && reviews->storage == DatasetStorageKind::InMemoryArrow && reviews->modality == DatasetModality::Text &&
              reviews->Has(kBackingText) && reviews->Has(kBackingArrow) && reviews->rows == 50000 && reviews->columns == 2 && reviews->classes == 2,
          "text CSV: stored as an Arrow table, modality text (the loaders' rule), both backings");
    Check(catalog.Resolve("MNIST")->storage == DatasetStorageKind::DiskBackedParquet && catalog.Resolve("MNIST")->modality == DatasetModality::Tabular,
          "Parquet: tabular");
    Check(catalog.Resolve("Cats")->storage == DatasetStorageKind::ImageCached && catalog.Resolve("Cats")->modality == DatasetModality::Image, "images");
    auto tfidf = catalog.Resolve("tfidf__materialized");
    Check(tfidf->storage == DatasetStorageKind::SparseFeatureCSR && tfidf->materialized, "sparse pipeline result");
    Check(catalog.Resolve("old")->modality == DatasetModality::Unknown, "old map: no modality");
    Check(!catalog.Resolve("nope") && !catalog.Resolve(""), "unknown names");
    Check(catalog.Resolve("Spotify")->generation == 0, "not pumped yet: generation 0");
    Check(std::string(StorageText(DatasetStorageKind::DiskBackedParquet)) == "table (on disk, Parquet)", "storage words");

    // Pump: everything is new once; generations rise.
    std::vector<DatasetChange> seen;
    auto owner = std::make_shared<int>(1);
    const int id = catalog.Subscribe(owner, [&](const DatasetChange& c) {
        seen.push_back(c);
        // A listener may call the catalog: it is not called under any lock
        // (a held lock would deadlock here).
        (void)catalog.List();
        (void)catalog.GenerationOf(c.name);
    });
    Check(catalog.Pump(std::chrono::milliseconds(0)) == 6 && seen.size() == 6, "first pump: six added");
    for (const auto& c : seen) Check(c.kind == DatasetChange::Kind::Added && c.generation > 0, "added with a generation");
    const uint64_t g_spotify = catalog.GenerationOf("Spotify");
    Check(g_spotify > 0 && catalog.Resolve("Spotify")->generation == g_spotify, "generation in the entry");
    seen.clear();
    Check(catalog.Pump(std::chrono::milliseconds(0)) == 0 && seen.empty(), "nothing changed: no events");

    // Replaced (a re-load gives a new object), added, removed.
    {
        std::lock_guard<std::mutex> lock(reg.mutex);
        reg.items[0].identity = 201;  // Spotify re-loaded
        reg.items[0].rows = 8000;
        reg.items.erase(reg.items.begin() + 1);  // MNIST removed
        reg.items.push_back({"World", kBackingArrow, 175, 6, 0, "D:/data/world.csv", 202});
    }
    Check(catalog.Pump(std::chrono::milliseconds(0)) == 3, "three changes");
    bool replaced = false, removed = false, added = false;
    for (const auto& c : seen) {
        if (c.name == "Spotify") replaced = c.kind == DatasetChange::Kind::Replaced && c.entry.rows == 8000 && c.generation > g_spotify;
        if (c.name == "MNIST") removed = c.kind == DatasetChange::Kind::Removed && c.entry.rows == 70000;
        if (c.name == "World") added = c.kind == DatasetChange::Kind::Added;
    }
    Check(replaced && removed && added, "replaced (newer generation), removed (last entry kept), added");
    Check(catalog.GenerationOf("MNIST") > 0 && !catalog.Resolve("MNIST"), "a removed dataset keeps its last generation");

    // A text entry added to an existing table changes the entry (identity combines backings).
    {
        std::lock_guard<std::mutex> lock(reg.mutex);
        reg.items.push_back({"World", kBackingText, 175, 0, 0, "D:/data/world.csv", 203});
    }
    seen.clear();
    Check(catalog.Pump(std::chrono::milliseconds(0)) == 1 && seen[0].kind == DatasetChange::Kind::Replaced &&
              seen[0].entry.modality == DatasetModality::Text, "a new backing is a change");

    // The interval: a second pump right away does nothing.
    {
        std::lock_guard<std::mutex> lock(reg.mutex);
        reg.items.pop_back();
    }
    Check(catalog.Pump(std::chrono::milliseconds(60000)) == 0, "within the interval: not looked at");
    Check(catalog.Pump(std::chrono::milliseconds(0)) == 1, "then delivered");

    // Labels from the graph: shown names, and a label leads to its dataset.
    catalog.SetLabels({{"World", "World countries"}});
    Check(catalog.Resolve("World")->label == "World countries" && catalog.Resolve("World")->Shown() == "World countries" &&
              catalog.Resolve("Cats")->Shown() == "Cats", "labels and shown names");
    Check(catalog.NameFor("World countries") == "World" && catalog.NameFor("Cats") == "Cats" && catalog.NameFor("nothing").empty(),
          "a label or a name leads to the dataset");
    catalog.SetTargets({{"World", "gdp_per_person"}});
    Check(catalog.Resolve("World")->target_column == "gdp_per_person" && catalog.Resolve("Cats")->target_column.empty(), "graph targets");
    catalog.SetSources({{"World", "D:/other/world.csv"}, {"tfidf__materialized", "D:/data/reviews.csv"}});
    Check(catalog.Resolve("World")->source_path == "D:/data/world.csv", "the registry's source path wins");
    Check(catalog.Resolve("tfidf__materialized")->source_path == "D:/data/reviews.csv", "the graph's file fills a missing source");

    // A listener that unsubscribes another during delivery; owner expiry.
    int calls_a = 0, calls_b = 0;
    int id_b = 0;
    const int id_a = catalog.Subscribe({}, [&](const DatasetChange&) {
        ++calls_a;
        catalog.Unsubscribe(id_b);
    });
    id_b = catalog.Subscribe({}, [&](const DatasetChange&) { ++calls_b; });
    {
        std::lock_guard<std::mutex> lock(reg.mutex);
        reg.items.push_back({"New", kBackingArrow, 1, 1, 0, "", 301});
    }
    seen.clear();
    owner.reset();  // the first subscriber's owner is gone
    catalog.Pump(std::chrono::milliseconds(0));
    Check(calls_a == 1 && calls_b == 0, "a listener unsubscribed during delivery is not called");
    Check(seen.empty(), "an expired owner's listener is dropped");
    catalog.Unsubscribe(id_a);
    catalog.Unsubscribe(id);

    std::cout << "dataset catalog: one list over every map, storage and modality rules, change events with generations, "
                 "listeners outside the lock, unsubscribe in delivery, owner expiry, interval. OK\n";
    return 0;
}
