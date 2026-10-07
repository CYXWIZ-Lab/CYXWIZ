// Session query engine (TOFIX134 P3 foundation, commit 2): restricted
// DuckDB, read-only SQL, bound parameters, Arrow and Parquet attachment,
// row limit, cancellation.

#include "../src/core/session_query_engine.h"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

using namespace cyxwiz;
namespace fs = std::filesystem;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::shared_ptr<arrow::Table> Tracks() {
    arrow::StringBuilder type;
    arrow::Int64Builder pop;
    const char* types[] = {"album", "album", "single", "compilation", "single", "album"};
    const int64_t pops[] = {60, 50, 40, 30, 50, 70};
    for (int i = 0; i < 6; ++i) {
        (void)type.Append(types[i]);
        (void)pop.Append(pops[i]);
    }
    std::shared_ptr<arrow::Array> a, b;
    (void)type.Finish(&a);
    (void)pop.Finish(&b);
    return arrow::Table::Make(arrow::schema({arrow::field("album_type", arrow::utf8()), arrow::field("popularity", arrow::int64())}), {a, b});
}

int64_t IntAt(const std::shared_ptr<arrow::Table>& t, int col, int64_t row) {
    auto s = t->column(col)->GetScalar(row);
    return std::static_pointer_cast<arrow::Int64Scalar>(*s)->value;
}
}  // namespace

int main() {
    const fs::path root = fs::temp_directory_path() / "cyxwiz_test_session_query";
    std::error_code ec;
    fs::remove_all(root, ec);
    {
        SessionQueryEngine engine((root / "attach").string());
        Check(engine.Ready(), "engine starts");

        // Attach an in-memory table and query it by its real name (spaces allowed).
        std::string error;
        auto t = Tracks();
        Check(engine.AttachArrow("Spotify tracks", 1, t, &error), "attach Arrow: " + error);
        QueryRequest q;
        q.sql = "SELECT album_type, count(*) AS n, avg(popularity) AS pop FROM \"Spotify tracks\" GROUP BY album_type ORDER BY n DESC, album_type;";
        QueryResult r = engine.Run(q);
        Check(r.ok && r.table && r.table->num_rows() == 3 && IntAt(r.table, 1, 0) == 3, "group by: " + r.error);
        Check(r.inputs.size() == 1 && r.inputs[0] == "Spotify tracks" && r.rows_in_inputs == 6, "the tables the SQL reads, with their rows");
        Check(engine.AttachArrow("Spotify tracks", 1, t, &error), "same version: no work");

        // Parameters are bound, never pasted into the text.
        q.sql = "SELECT count(*) AS n FROM \"Spotify tracks\" WHERE album_type = ? AND popularity >= ?";
        q.params = {QueryParam::Of(std::string("single")), QueryParam::Of(int64_t{45})};
        r = engine.Run(q);
        Check(r.ok && IntAt(r.table, 0, 0) == 1, "parameters: " + r.error);
        q.params = {QueryParam::Of(std::string("x' OR '1'='1")), QueryParam::Of(int64_t{0})};
        r = engine.Run(q);
        Check(r.ok && IntAt(r.table, 0, 0) == 0, "a parameter is a value, not SQL");
        q.params.clear();

        // Read-only and restricted.
        q.sql = "DELETE FROM \"Spotify tracks\"";
        r = engine.Run(q);
        Check(!r.ok && r.error.find("Only reading is allowed") != std::string::npos, "no changes: " + r.error);
        q.sql = "SELECT 1; SELECT 2";
        Check(engine.Run(q).error == "Run one statement at a time.", "one statement");
        const fs::path outside = root / "outside.csv";
        std::ofstream(outside) << "a\n1\n";
        q.sql = "SELECT * FROM read_csv('" + outside.generic_string() + "')";
        r = engine.Run(q);
        Check(!r.ok, "files outside the engine's folders cannot be read: " + r.error);
        q.sql = "SELECT * FROM read_parquet('" + (root / "attach" / "*.parquet").generic_string() + "') LIMIT 1";
        r = engine.Run(q);
        Check(r.ok, "the engine's own folder is readable (its copies): " + r.error);
        q.sql = "SET enable_external_access = true";
        Check(!engine.Run(q).ok, "the configuration cannot be changed");

        // Row limit: cut and said so.
        q.sql = "SELECT * FROM \"Spotify tracks\"";
        q.row_limit = 4;
        r = engine.Run(q);
        Check(r.ok && r.table->num_rows() == 4 && r.truncated, "row limit cut and flagged");
        q.row_limit = 10;
        r = engine.Run(q);
        Check(r.ok && r.table->num_rows() == 6 && !r.truncated, "under the limit: not truncated");
        q.row_limit = 0;

        // A new version replaces the view (and its copy).
        auto t2 = t->Slice(0, 2);
        Check(engine.AttachArrow("Spotify tracks", 2, t2, &error) && engine.IsAttached("Spotify tracks", 2), "re-attach a new version");
        r = engine.Run(q);
        Check(r.ok && r.table->num_rows() == 2, "the new version is read");

        // A disk-backed Parquet dataset in another folder: viewed from its file.
        const fs::path cache = root / "cache";
        fs::create_directories(cache);
        const fs::path pq = cache / "mnist_small.parquet";
        {
            auto out = arrow::io::FileOutputStream::Open(pq.string());
            Check(out.ok() && parquet::arrow::WriteTable(*t, arrow::default_memory_pool(), *out, 1024).ok() && (*out)->Close().ok(), "write a cache file");
        }
        Check(engine.AttachParquet("MNIST", 7, pq.string(), 6, &error), "attach Parquet: " + error);
        q.sql = "SELECT count(*) FROM MNIST m JOIN \"Spotify tracks\" s USING (album_type)";
        r = engine.Run(q);
        Check(r.ok && r.inputs.size() == 2, "join a Parquet dataset with an attached table (reopened with its folder): " + r.error);
        Check(engine.IsAttached("Spotify tracks", 2), "views survive the reopen");

        // Cancellation from another thread.
        q.sql = "SELECT count(*) FROM range(100000000000) a";
        std::thread stopper([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            engine.Interrupt();
        });
        r = engine.Run(q);
        stopper.join();
        Check(!r.ok && r.cancelled && r.error == "Cancelled.", "cancelled: " + r.error);
        q.sql = "SELECT 42 AS answer";
        r = engine.Run(q);
        Check(r.ok && r.table->num_rows() == 1, "runs again after a cancel: " + r.error);

        // Queries side by side: a long one does not hold up short ones, and
        // its token stops only it.
        {
            QueryToken long_token;
            QueryRequest slow;
            slow.sql = "SELECT count(*) FROM range(100000000000) a";
            QueryResult slow_r;
            std::thread runner([&] { slow_r = engine.Run(slow, &long_token); });
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<std::thread> quick;
            std::vector<QueryResult> quick_r(4);
            for (size_t i = 0; i < quick_r.size(); ++i)
                quick.emplace_back([&, i] {
                    QueryRequest qq;
                    qq.sql = "SELECT count(*) AS n FROM \"Spotify tracks\"";
                    quick_r[i] = engine.Run(qq);
                });
            for (auto& th : quick) th.join();
            const double quick_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            bool all_ok = true;
            for (const auto& x : quick_r) all_ok = all_ok && x.ok && x.table->num_rows() == 1;
            Check(all_ok, "short queries run while a long one runs");
            Check(quick_ms < 5000, "short queries do not wait for the long one (" + std::to_string(quick_ms) + " ms)");
            engine.Interrupt(long_token);
            runner.join();
            Check(!slow_r.ok && slow_r.cancelled, "its token stops the long query: " + slow_r.error);
            // A token stopped before its query starts: the query never runs.
            QueryToken early;
            engine.Interrupt(early);
            const auto t1 = std::chrono::steady_clock::now();
            QueryResult e = engine.Run(slow, &early);
            const double early_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
            Check(!e.ok && e.cancelled && early_ms < 1000, "a stopped token's query does not start");
            q.sql = "SELECT 42 AS answer";
            Check(engine.Run(q).ok, "runs after the side-by-side queries");
        }

        engine.Detach("Spotify tracks");
        q.sql = "SELECT * FROM \"Spotify tracks\"";
        Check(!engine.Run(q).ok, "detached");
        Check(SessionQueryEngine::QuoteIdentifier("a\"b") == "\"a\"\"b\"", "identifier quoting");
    }
    fs::remove_all(root, ec);
    std::cout << "session query engine: read-only, restricted files and settings, bound parameters, Arrow and Parquet "
                 "attachment, versions, joins, row limit, cancellation, queries side by side with per-query stop. OK\n";
    return 0;
}
