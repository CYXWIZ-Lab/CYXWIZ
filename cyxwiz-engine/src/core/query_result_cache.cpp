#include "query_result_cache.h"

#include <arrow/io/file.h>
#include <arrow/util/byte_size.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/properties.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <system_error>
#include <thread>

namespace cyxwiz {

namespace fs = std::filesystem;

namespace {

uint64_t Fnv(const void* data, size_t size, uint64_t h) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < size; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

uint64_t Fnv(const std::string& text, uint64_t seed) { return Fnv(text.data(), text.size(), seed); }

// The bytes an array is made of (its buffers, then its children), with its
// window into them: a slice of a buffer differs from the whole.
uint64_t HashArray(const arrow::ArrayData& a, uint64_t h) {
    h = Fnv(&a.offset, sizeof(a.offset), h);
    h = Fnv(&a.length, sizeof(a.length), h);
    for (const auto& buf : a.buffers) {
        if (!buf) continue;
        h = Fnv(buf->data(), static_cast<size_t>(buf->size()), h);
    }
    for (const auto& child : a.child_data)
        if (child) h = HashArray(*child, h);
    if (a.dictionary) h = HashArray(*a.dictionary, h);
    return h;
}

std::string Hex(uint64_t v) {
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(v));
    return buf;
}

// Rides along in the saved schema: whether the result was cut at its row limit.
constexpr const char* kTruncatedKey = "cyxwiz.truncated";

}  // namespace

void QueryResultCache::SetFolder(const fs::path& folder) {
    std::lock_guard<std::mutex> lock(mutex_);
    folder_ = folder;
}

fs::path QueryResultCache::Folder() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return folder_;
}

std::string QueryResultCache::Key(const QueryRequest& request, std::vector<std::string> identities) {
    std::sort(identities.begin(), identities.end());
    // q1: the layout of this text; a change to it must change the version.
    std::string text = "q1\n" + request.sql + "\n" + std::to_string(request.row_limit) + "\n";
    for (const auto& p : request.params) {
        text += std::to_string(static_cast<int>(p.type)) + ":";
        switch (p.type) {
            case QueryParam::Type::Bool: text += p.b ? "1" : "0"; break;
            case QueryParam::Type::Int: text += std::to_string(p.i); break;
            case QueryParam::Type::Double: {
                char buf[40];
                std::snprintf(buf, sizeof(buf), "%.17g", p.d);
                text += buf;
                break;
            }
            case QueryParam::Type::Text: text += std::to_string(p.s.size()) + ":" + p.s; break;
            case QueryParam::Type::Null: break;
        }
        text += "\n";
    }
    for (const auto& id : identities) text += "t:" + id + "\n";
    return Hex(Fnv(text, 14695981039346656037ull)) + Hex(Fnv(text, 0x9e3779b97f4a7c15ull));
}

std::string QueryResultCache::ContentIdentity(const arrow::Table& table) {
    uint64_t h = Fnv(table.schema()->ToString(), 14695981039346656037ull);
    uint64_t h2 = Fnv(table.schema()->ToString(), 0x9e3779b97f4a7c15ull);
    for (int c = 0; c < table.num_columns(); ++c)
        for (const auto& chunk : table.column(c)->chunks()) {
            h = HashArray(*chunk->data(), h);
            h2 = HashArray(*chunk->data(), h2);
        }
    return "arrow:" + std::to_string(table.num_rows()) + ":" + Hex(h) + Hex(h2);
}

std::string QueryResultCache::FileIdentity(const std::string& path) {
    std::error_code ec;
    const auto size = fs::file_size(path, ec);
    const auto time = fs::last_write_time(path, ec).time_since_epoch().count();
    return path + ":" + std::to_string(ec ? 0 : size) + ":" + std::to_string(ec ? 0 : time);
}

fs::path QueryResultCache::PathOf(const std::string& key) const {
    return folder_ / (key + ".parquet");
}

std::shared_ptr<arrow::Table> QueryResultCache::Load(const std::string& key, bool* truncated) const {
    fs::path path;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (folder_.empty()) return nullptr;
        path = PathOf(key);
    }
    std::error_code ec;
    if (!fs::is_regular_file(path, ec)) return nullptr;
    auto file = arrow::io::ReadableFile::Open(path.string());
    if (!file.ok()) return nullptr;
    auto reader = parquet::arrow::OpenFile(*file, arrow::default_memory_pool());
    if (!reader.ok()) return nullptr;
    std::shared_ptr<arrow::Table> table;
    if (!(*reader)->ReadTable(&table).ok() || !table) return nullptr;
    if (truncated) {
        const auto meta = table->schema()->metadata();
        *truncated = meta && meta->Contains(kTruncatedKey) && meta->Get(kTruncatedKey).ValueOr("0") == "1";
    }
    fs::last_write_time(path, fs::file_time_type::clock::now(), ec);  // used now
    return table;
}

bool QueryResultCache::Save(const std::string& key, const std::shared_ptr<arrow::Table>& in, bool truncated) {
    if (!in) return false;
    std::shared_ptr<arrow::Table> table = in;
    if (truncated) {
        auto meta = table->schema()->metadata() ? table->schema()->metadata()->Copy() : std::make_shared<arrow::KeyValueMetadata>();
        meta->Append(kTruncatedKey, "1");
        table = table->ReplaceSchemaMetadata(meta);
    }
    fs::path path;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (folder_.empty()) return false;
        path = PathOf(key);
    }
    auto bytes = arrow::util::TotalBufferSize(*table);
    if (static_cast<uint64_t>(bytes) > kMaxResultBytes) return false;
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    // A name per thread: two savers of one key never share a temporary file.
    const fs::path tmp = path.string() + "." + Hex(std::hash<std::thread::id>{}(std::this_thread::get_id())) + ".tmp";
    {
        auto file = arrow::io::FileOutputStream::Open(tmp.string());
        if (!file.ok()) return false;
        auto props = parquet::WriterProperties::Builder().compression(parquet::Compression::SNAPPY)->build();
        // The Arrow schema rides along, so the result reads back with its exact types.
        auto arrow_props = parquet::ArrowWriterProperties::Builder().store_schema()->build();
        const auto status = parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *file, 64 * 1024, props, arrow_props);
        const auto closed = (*file)->Close();
        if (!status.ok() || !closed.ok()) {
            fs::remove(tmp, ec);
            return false;
        }
    }
    fs::rename(tmp, path, ec);
    if (ec) {
        fs::remove(tmp, ec);
        return false;
    }
    bool prune = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        prune = ++saves_ % 32 == 0;
    }
    if (prune) Prune();
    return true;
}

void QueryResultCache::Prune(uint64_t max_bytes) {
    const fs::path folder = Folder();
    if (folder.empty()) return;
    struct Item {
        fs::path path;
        uint64_t size;
        fs::file_time_type time;
    };
    std::vector<Item> items;
    uint64_t total = 0;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(folder, ec)) {
        if (!e.is_regular_file(ec) || e.path().extension() != ".parquet") continue;
        const uint64_t size = e.file_size(ec);
        items.push_back({e.path(), size, e.last_write_time(ec)});
        total += size;
    }
    if (total <= max_bytes) return;
    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) { return a.time < b.time; });
    for (const auto& it : items) {
        if (total <= max_bytes) break;
        if (fs::remove(it.path, ec)) total -= it.size;
    }
}

std::shared_ptr<QueryResultCache::Flight> QueryResultCache::Join(const std::string& key, bool* leader) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = flights_.find(key);
    if (it != flights_.end()) {
        *leader = false;
        return it->second;
    }
    auto flight = std::make_shared<Flight>();
    flights_[key] = flight;
    *leader = true;
    return flight;
}

void QueryResultCache::Finish(const std::string& key, const std::shared_ptr<Flight>& flight, const QueryResult& result) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = flights_.find(key);
        if (it != flights_.end() && it->second == flight) flights_.erase(it);
    }
    {
        std::lock_guard<std::mutex> fl(flight->mutex);
        flight->result = result;
        flight->done = true;
    }
    flight->cv.notify_all();
}

bool QueryResultCache::Wait(Flight& flight, const std::function<bool()>& stopped, QueryResult* out) {
    std::unique_lock<std::mutex> fl(flight.mutex);
    while (!flight.done) {
        if (stopped && stopped()) return false;
        flight.cv.wait_for(fl, std::chrono::milliseconds(50));
    }
    if (!flight.result.ok) return false;
    *out = flight.result;
    return true;
}

}  // namespace cyxwiz
