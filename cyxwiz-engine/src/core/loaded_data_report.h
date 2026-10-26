#pragma once

// What loaded data holds (TOFIX160 P1): every dataset the registry keeps, its
// memory and its disk. Arrow buffers held by several names (a pipeline run's
// copy of its input, a slice, the same table registered twice) are counted
// once, and each entry says how much of its memory is shared, so a copy that
// should have been a share shows up. No ImGui; the registry builds it under
// its lock (DataRegistry::LoadedData).

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace arrow {
class Table;
}

namespace cyxwiz {

enum class LoadedDataKind { TableInMemory, TableOnDisk, SparseFeatures, Images, Audio, Text, Other };
const char* LoadedDataKindName(LoadedDataKind kind);

struct LoadedDataEntry {
    std::string name;
    LoadedDataKind kind = LoadedDataKind::Other;
    uint64_t memory_bytes = 0;  // its buffers in memory, each byte once
    uint64_t shared_bytes = 0;  // the part of memory_bytes another entry also holds
    uint64_t disk_bytes = 0;    // a file it is kept in (Parquet on disk)
    std::string path;           // the file or folder it reads, "" when none
    int64_t rows = -1;          // -1: not known
};

struct LoadedDataReport {
    std::vector<LoadedDataEntry> entries;  // in the order added
    uint64_t memory_bytes = 0;             // every entry's buffers, each byte once
    uint64_t disk_bytes = 0;
    // The Arrow memory pool: everything Arrow holds now (loaded tables, query
    // results, views' copies) and its peak since start.
    int64_t arrow_pool_bytes = 0;
    int64_t arrow_pool_peak_bytes = 0;
};

class LoadedDataReportBuilder {
public:
    // A table in memory: its size is its buffers (values, offsets, validity).
    void AddTable(LoadedDataEntry entry, const std::shared_ptr<arrow::Table>& table);
    // An entry whose size is already known (no Arrow buffers to compare).
    void Add(LoadedDataEntry entry);
    LoadedDataReport Finish();

private:
    struct Range {
        uintptr_t begin = 0, end = 0;
    };
    struct Pending {
        LoadedDataEntry entry;
        std::vector<Range> ranges;
    };
    std::vector<Pending> pending_;
};

}  // namespace cyxwiz
