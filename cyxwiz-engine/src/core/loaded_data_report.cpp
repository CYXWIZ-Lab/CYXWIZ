#include "loaded_data_report.h"

#include <arrow/api.h>

#include <algorithm>

namespace cyxwiz {

const char* LoadedDataKindName(LoadedDataKind kind) {
    switch (kind) {
        case LoadedDataKind::TableInMemory: return "Table in memory";
        case LoadedDataKind::TableOnDisk: return "Table on disk";
        case LoadedDataKind::SparseFeatures: return "Sparse features";
        case LoadedDataKind::Images: return "Images";
        case LoadedDataKind::Audio: return "Audio";
        case LoadedDataKind::Text: return "Text";
        case LoadedDataKind::Other: return "Other";
    }
    return "Other";
}

namespace {

// Every buffer of an array, its dictionary and its children (a list's values,
// a struct's fields).
template <typename Out>
void CollectBuffers(const std::shared_ptr<arrow::ArrayData>& data, Out& out) {
    if (!data) return;
    for (const auto& buffer : data->buffers) {
        if (buffer && buffer->size() > 0) out(buffer->address(), buffer->size());
    }
    for (const auto& child : data->child_data) CollectBuffers(child, out);
    CollectBuffers(data->dictionary, out);
}

}  // namespace

void LoadedDataReportBuilder::AddTable(LoadedDataEntry entry, const std::shared_ptr<arrow::Table>& table) {
    Pending pending;
    pending.entry = std::move(entry);
    if (table) {
        if (pending.entry.rows < 0) pending.entry.rows = table->num_rows();
        auto add = [&pending](uint64_t address, int64_t size) {
            pending.ranges.push_back({static_cast<uintptr_t>(address), static_cast<uintptr_t>(address + size)});
        };
        for (const auto& column : table->columns()) {
            for (const auto& chunk : column->chunks()) CollectBuffers(chunk->data(), add);
        }
    }
    pending_.push_back(std::move(pending));
}

void LoadedDataReportBuilder::Add(LoadedDataEntry entry) {
    Pending pending;
    pending.entry = std::move(entry);
    pending_.push_back(std::move(pending));
}

LoadedDataReport LoadedDataReportBuilder::Finish() {
    LoadedDataReport report;

    // A sweep over every buffer range: a stretch of memory counts once in the
    // total, once in each entry that holds it, and as shared in each entry
    // when two or more entries hold it.
    struct Event {
        uintptr_t at;
        int delta;
        size_t entry;
    };
    std::vector<Event> events;
    for (size_t i = 0; i < pending_.size(); ++i) {
        for (const auto& r : pending_[i].ranges) {
            events.push_back({r.begin, +1, i});
            events.push_back({r.end, -1, i});
        }
    }
    std::sort(events.begin(), events.end(), [](const Event& a, const Event& b) { return a.at < b.at; });

    std::vector<int> depth(pending_.size(), 0);
    std::vector<uint64_t> held(pending_.size(), 0), shared(pending_.size(), 0);
    size_t holders = 0;  // entries with depth > 0
    uintptr_t last = 0;
    for (size_t k = 0; k < events.size();) {
        const uintptr_t at = events[k].at;
        if (holders > 0 && at > last) {
            const uint64_t length = at - last;
            report.memory_bytes += length;
            for (size_t i = 0; i < pending_.size(); ++i) {
                if (depth[i] == 0) continue;
                held[i] += length;
                if (holders > 1) shared[i] += length;
            }
        }
        for (; k < events.size() && events[k].at == at; ++k) {
            int& d = depth[events[k].entry];
            if (d == 0 && events[k].delta > 0) ++holders;
            d += events[k].delta;
            if (d == 0 && events[k].delta < 0) --holders;
        }
        last = at;
    }

    for (size_t i = 0; i < pending_.size(); ++i) {
        LoadedDataEntry entry = std::move(pending_[i].entry);
        if (!pending_[i].ranges.empty()) {
            entry.memory_bytes = held[i];
            entry.shared_bytes = shared[i];
        } else {
            report.memory_bytes += entry.memory_bytes;  // a size given, nothing to compare
        }
        report.disk_bytes += entry.disk_bytes;
        report.entries.push_back(std::move(entry));
    }
    pending_.clear();

    arrow::MemoryPool* pool = arrow::default_memory_pool();
    report.arrow_pool_bytes = pool->bytes_allocated();
    report.arrow_pool_peak_bytes = pool->max_memory();
    return report;
}

}  // namespace cyxwiz
