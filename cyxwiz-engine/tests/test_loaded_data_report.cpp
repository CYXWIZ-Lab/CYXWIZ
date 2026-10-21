// What loaded data holds (TOFIX160 P1): each buffer counted once, the part
// of an entry another entry also holds, sizes given without buffers, disk
// bytes and the Arrow pool.
#include "../src/core/loaded_data_report.h"

#include <arrow/api.h>

#include <cstdio>
#include <string>

using namespace cyxwiz;

namespace {

int failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        ++failures;
    }
}

// `rows` int64 values and a text column (no nulls: no validity buffers).
std::shared_ptr<arrow::Table> Table(int64_t rows) {
    arrow::Int64Builder ids;
    arrow::StringBuilder names;
    for (int64_t i = 0; i < rows; ++i) {
        (void)ids.Append(i);
        (void)names.Append("n" + std::to_string(i % 10));
    }
    std::shared_ptr<arrow::Array> a, b;
    (void)ids.Finish(&a);
    (void)names.Finish(&b);
    return arrow::Table::Make(arrow::schema({arrow::field("id", arrow::int64()), arrow::field("name", arrow::utf8())}),
                              {a, b});
}

// The table's own buffer bytes (as ArrowDataset::GetMemoryUsage counts them).
uint64_t BufferBytes(const std::shared_ptr<arrow::Table>& t) {
    uint64_t total = 0;
    for (const auto& column : t->columns())
        for (const auto& chunk : column->chunks())
            for (const auto& buffer : chunk->data()->buffers)
                if (buffer) total += static_cast<uint64_t>(buffer->size());
    return total;
}

LoadedDataEntry Entry(const std::string& name, LoadedDataKind kind = LoadedDataKind::TableInMemory) {
    LoadedDataEntry e;
    e.name = name;
    e.kind = kind;
    return e;
}

}  // namespace

int main() {
    const auto a = Table(1000);
    const auto b = Table(500);
    const uint64_t a_bytes = BufferBytes(a), b_bytes = BufferBytes(b);
    Check(a_bytes > 8000 && b_bytes > 4000, "tables have buffers");

    // Two separate tables: nothing shared.
    {
        LoadedDataReportBuilder builder;
        builder.AddTable(Entry("a"), a);
        builder.AddTable(Entry("b"), b);
        const auto r = builder.Finish();
        Check(r.entries.size() == 2, "two entries");
        Check(r.entries[0].memory_bytes == a_bytes && r.entries[0].shared_bytes == 0, "a alone");
        Check(r.entries[1].memory_bytes == b_bytes && r.entries[1].shared_bytes == 0, "b alone");
        Check(r.memory_bytes == a_bytes + b_bytes, "total is the sum");
        Check(r.entries[0].rows == 1000, "rows from the table");
    }

    // The same table under two names (a pipeline run's copy that shares): counted once, both shared.
    {
        LoadedDataReportBuilder builder;
        builder.AddTable(Entry("input"), a);
        builder.AddTable(Entry("ds_datainput_1"), a);
        const auto r = builder.Finish();
        Check(r.memory_bytes == a_bytes, "a shared table counts once");
        Check(r.entries[0].shared_bytes == a_bytes && r.entries[1].shared_bytes == a_bytes, "both are shared");
        Check(r.entries[1].memory_bytes == a_bytes, "each entry still holds all of it");
    }

    // A slice holds its parent's buffers: no new memory.
    {
        LoadedDataReportBuilder builder;
        builder.AddTable(Entry("table"), a);
        builder.AddTable(Entry("first 100 rows"), a->Slice(0, 100));
        const auto r = builder.Finish();
        Check(r.memory_bytes == a_bytes, "a slice adds nothing");
        Check(r.entries[1].shared_bytes == r.entries[1].memory_bytes && r.entries[1].memory_bytes > 0,
              "the slice is all shared");
        Check(r.entries[1].rows == 100, "the slice's rows");
    }

    // A real copy is not shared (what a copy that should have been a share looks like).
    {
        auto copy = arrow::Table::Make(a->schema(), a->columns());  // same arrays: still shared
        arrow::Int64Builder ids;
        for (int64_t i = 0; i < 1000; ++i) (void)ids.Append(i);
        std::shared_ptr<arrow::Array> fresh;
        (void)ids.Finish(&fresh);
        auto other = arrow::Table::Make(arrow::schema({arrow::field("id", arrow::int64())}), {fresh});
        LoadedDataReportBuilder builder;
        builder.AddTable(Entry("table"), a);
        builder.AddTable(Entry("same arrays"), copy);
        builder.AddTable(Entry("fresh ids"), other);
        const auto r = builder.Finish();
        Check(r.entries[1].shared_bytes == r.entries[1].memory_bytes, "a table of the same arrays shares");
        Check(r.entries[2].shared_bytes == 0 && r.entries[2].memory_bytes == 8000, "new values are its own");
        Check(r.memory_bytes == a_bytes + 8000, "total: the table once plus the new values");
    }

    // Sizes given, disk, and entries without buffers.
    {
        LoadedDataReportBuilder builder;
        LoadedDataEntry disk = Entry("big", LoadedDataKind::TableOnDisk);
        disk.disk_bytes = 5000000;
        disk.path = "D:/cache/big.parquet";
        builder.Add(disk);
        LoadedDataEntry sparse = Entry("tfidf", LoadedDataKind::SparseFeatures);
        sparse.memory_bytes = 1234;
        builder.Add(sparse);
        builder.Add(Entry("photos", LoadedDataKind::Images));
        builder.AddTable(Entry("a"), a);
        const auto r = builder.Finish();
        Check(r.disk_bytes == 5000000, "disk bytes summed");
        Check(r.memory_bytes == a_bytes + 1234, "a given size is added");
        Check(r.entries[1].memory_bytes == 1234 && r.entries[1].shared_bytes == 0, "the given size stays");
        Check(r.entries[2].memory_bytes == 0 && r.entries[2].rows == -1, "images hold nothing in memory");
        Check(std::string(LoadedDataKindName(r.entries[0].kind)) == "Table on disk", "kind name");
        Check(r.arrow_pool_bytes > 0 && r.arrow_pool_peak_bytes >= r.arrow_pool_bytes, "the Arrow pool is read");
    }

    // Empty.
    {
        LoadedDataReportBuilder builder;
        const auto r = builder.Finish();
        Check(r.entries.empty() && r.memory_bytes == 0, "an empty report");
    }

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("test_loaded_data_report: all checks passed\n");
    return 0;
}
