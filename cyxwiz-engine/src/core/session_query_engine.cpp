#include "session_query_engine.h"

#include "duckdb_connector.h"

#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>

namespace cyxwiz {

namespace fs = std::filesystem;

namespace {

// DuckDB paths with forward slashes, folders ending in '/'.
std::string Slashes(std::string p) {
    std::replace(p.begin(), p.end(), '\\', '/');
    return p;
}

std::string FolderOf(const std::string& path) {
    std::string f = Slashes(fs::path(path).parent_path().string());
    if (!f.empty() && f.back() != '/') f += '/';
    return f;
}

std::string SqlString(const std::string& text) {
    std::string out = "'";
    for (char c : text) out += c == '\'' ? std::string("''") : std::string(1, c);
    return out + "'";
}

std::string Lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// The text without trailing semicolons and spaces (so it can be wrapped).
std::string Trimmed(const std::string& sql) {
    size_t end = sql.size();
    while (end > 0 && (std::isspace(static_cast<unsigned char>(sql[end - 1])) || sql[end - 1] == ';')) --end;
    return sql.substr(0, end);
}

duckdb::Value ToDuck(const QueryParam& p) {
    switch (p.type) {
        case QueryParam::Type::Bool: return duckdb::Value::BOOLEAN(p.b);
        case QueryParam::Type::Int: return duckdb::Value::BIGINT(p.i);
        case QueryParam::Type::Double: return duckdb::Value::DOUBLE(p.d);
        case QueryParam::Type::Text: return duckdb::Value(p.s);
        case QueryParam::Type::Null: break;
    }
    return duckdb::Value();
}

}  // namespace

std::string SessionQueryEngine::QuoteIdentifier(const std::string& name) {
    std::string q = "\"";
    for (char c : name) q += c == '"' ? std::string("\"\"") : std::string(1, c);
    return q + "\"";
}

SessionQueryEngine::SessionQueryEngine(std::string attach_dir) : dir_(Slashes(std::move(attach_dir))) {
    std::error_code ec;
    fs::create_directories(dir_, ec);
    if (!dir_.empty() && dir_.back() != '/') dir_ += '/';
    allowed_.push_back(dir_);
    std::string error;
    Open(&error);
}

SessionQueryEngine::~SessionQueryEngine() {
    db_.reset();
    std::error_code ec;
    for (const auto& [name, a] : attached_)
        if (a.owned_file) fs::remove(a.path, ec);
}

bool SessionQueryEngine::Ready() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return db_ && db_->IsReady();
}

bool SessionQueryEngine::Open(std::string* error) {
    DuckDBConnectorPolicy policy;
    policy.allow_external_access = false;
    policy.allowed_directories = allowed_;
    db_ = std::make_unique<DuckDBConnector>(policy);
    if (!db_->IsReady()) {
        if (error) *error = "the query engine could not start: " + db_->GetLastError();
        return false;
    }
    for (const auto& [name, a] : attached_)
        if (!CreateView(name, a.path, error)) return false;
    return true;
}

bool SessionQueryEngine::CreateView(const std::string& name, const std::string& path, std::string* error) {
    if (!db_->Execute("CREATE OR REPLACE VIEW " + QuoteIdentifier(name) + " AS SELECT * FROM read_parquet(" + SqlString(Slashes(path)) + ")")) {
        if (error) *error = "could not attach '" + name + "': " + db_->GetLastError();
        return false;
    }
    return true;
}

bool SessionQueryEngine::IsAttached(const std::string& name, uint64_t version) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = attached_.find(name);
    return it != attached_.end() && it->second.version == version;
}

std::vector<std::string> SessionQueryEngine::Attached() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> out;
    for (const auto& [name, a] : attached_) out.push_back(name);
    return out;
}

bool SessionQueryEngine::AttachArrow(const std::string& name, uint64_t version, const std::shared_ptr<arrow::Table>& table,
                                     std::string* error) {
    if (!table) {
        if (error) *error = "'" + name + "' has no table";
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!db_ || !db_->IsReady()) {
        if (error) *error = "the query engine is not running";
        return false;
    }
    auto it = attached_.find(name);
    if (it != attached_.end() && it->second.version == version) return true;
    // A file name from the version (names may hold any characters).
    const std::string path = dir_ + "t" + std::to_string(std::hash<std::string>{}(name)) + "_" + std::to_string(version) + ".parquet";
    auto out = arrow::io::FileOutputStream::Open(path);
    if (!out.ok()) {
        if (error) *error = "could not write the copy of '" + name + "': " + out.status().ToString();
        return false;
    }
    const arrow::Status st = parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *out, 128 * 1024);
    const arrow::Status closed = (*out)->Close();
    if (!st.ok() || !closed.ok()) {
        if (error) *error = "could not write the copy of '" + name + "': " + (st.ok() ? closed : st).ToString();
        return false;
    }
    if (!CreateView(name, path, error)) return false;
    std::error_code ec;
    if (it != attached_.end() && it->second.owned_file) fs::remove(it->second.path, ec);
    attached_[name] = Attachment{version, path, static_cast<size_t>(table->num_rows()), true};
    return true;
}

bool SessionQueryEngine::AttachParquet(const std::string& name, uint64_t version, const std::string& path, size_t rows,
                                       std::string* error) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = attached_.find(name);
    if (it != attached_.end() && it->second.version == version) return true;
    const std::string folder = FolderOf(path);
    if (std::find(allowed_.begin(), allowed_.end(), folder) == allowed_.end()) {
        // A new cache folder: DuckDB's allowed folders are fixed when it opens
        // (the configuration is locked), so reopen with it.
        allowed_.push_back(folder);
        if (!Open(error)) return false;
    }
    if (!CreateView(name, path, error)) return false;
    std::error_code ec;
    if (it != attached_.end() && it->second.owned_file) fs::remove(it->second.path, ec);
    attached_[name] = Attachment{version, Slashes(path), rows, false};
    return true;
}

void SessionQueryEngine::Detach(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = attached_.find(name);
    if (it == attached_.end()) return;
    if (db_) db_->Execute("DROP VIEW IF EXISTS " + QuoteIdentifier(name));
    std::error_code ec;
    if (it->second.owned_file) fs::remove(it->second.path, ec);
    attached_.erase(it);
}

std::vector<std::string> SessionQueryEngine::NamesIn(const std::string& sql) const {
    std::vector<std::string> out;
    const std::string text = Lower(sql);
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& [name, a] : attached_) {
        const std::string n = Lower(name);
        for (size_t at = text.find(n); at != std::string::npos; at = text.find(n, at + 1)) {
            const auto word = [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; };
            const bool start = at == 0 || !word(text[at - 1]);
            const bool end = at + n.size() >= text.size() || !word(text[at + n.size()]);
            if (start && end) {
                out.push_back(name);
                break;
            }
        }
    }
    return out;
}

void SessionQueryEngine::Interrupt() {
    {
        std::lock_guard<std::mutex> lock(interrupt_mutex_);
        interrupted_ = true;
    }
    // The connection's Interrupt is thread-safe; db_ is not replaced while a query runs (both under mutex_).
    if (db_) db_->Interrupt();
}

QueryResult SessionQueryEngine::Run(const QueryRequest& request) {
    QueryResult r;
    const auto start = std::chrono::steady_clock::now();
    r.inputs = NamesIn(request.sql);
    std::lock_guard<std::mutex> lock(mutex_);
    {
        std::lock_guard<std::mutex> il(interrupt_mutex_);
        interrupted_ = false;
    }
    if (!db_ || !db_->IsReady()) {
        r.error = "the query engine is not running";
        return r;
    }
    for (const auto& name : r.inputs) r.rows_in_inputs += attached_[name].rows;
    const std::string rejection = db_->ReadOnlySelectRejection(request.sql);
    if (!rejection.empty()) {
        // The connector words it for the SQL step; say it for a query.
        r.error = rejection.find("exactly one statement") != std::string::npos ? "Run one statement at a time."
                  : rejection.find("only a read-only SELECT") != std::string::npos
                      ? "Only reading is allowed here: a SELECT (or WITH ... SELECT). Tables are changed in the graph."
                      : rejection;
        return r;
    }
    std::string sql = Trimmed(request.sql);
    if (request.row_limit > 0) sql = "SELECT * FROM (" + sql + ") AS cyxwiz_q LIMIT " + std::to_string(request.row_limit + 1);
    std::vector<duckdb::Value> params;
    for (const auto& p : request.params) params.push_back(ToDuck(p));
    r.table = db_->QueryWithParams(sql, std::move(params));
    r.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    bool interrupted;
    {
        std::lock_guard<std::mutex> il(interrupt_mutex_);
        interrupted = interrupted_;
    }
    if (!r.table) {
        r.cancelled = interrupted || db_->GetLastError().find("INTERRUPT") != std::string::npos ||
                      db_->GetLastError().find("nterrupt") != std::string::npos;
        r.error = r.cancelled ? "Cancelled." : db_->GetLastError();
        return r;
    }
    if (request.row_limit > 0 && static_cast<size_t>(r.table->num_rows()) > request.row_limit) {
        r.table = r.table->Slice(0, static_cast<int64_t>(request.row_limit));
        r.truncated = true;
    }
    r.ok = true;
    return r;
}

}  // namespace cyxwiz
