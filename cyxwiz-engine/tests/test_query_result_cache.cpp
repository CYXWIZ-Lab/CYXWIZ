// Saved query results (TOFIX134 Dashboard loading): the key, saving and
// reading back (with the row-limit flag), the size limits, pruning, and
// one run for the same key asked twice at once.

#include "../src/core/query_result_cache.h"

#include <arrow/api.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
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

std::shared_ptr<arrow::Table> Small(int64_t n) {
    arrow::StringBuilder name;
    arrow::DoubleBuilder value;
    arrow::BooleanBuilder flag;
    for (int64_t i = 0; i < n; ++i) {
        (void)name.Append("row " + std::to_string(i));
        (void)value.Append(static_cast<double>(i) * 0.5);
        (void)flag.Append(i % 2 == 0);
    }
    std::shared_ptr<arrow::Array> a, b, c;
    (void)name.Finish(&a);
    (void)value.Finish(&b);
    (void)flag.Finish(&c);
    return arrow::Table::Make(
        arrow::schema({arrow::field("name", arrow::utf8()), arrow::field("value", arrow::float64()), arrow::field("flag", arrow::boolean())}),
        {a, b, c});
}

std::shared_ptr<arrow::Table> Big(int64_t n) {
    arrow::Int64Builder b;
    (void)b.Reserve(n);
    for (int64_t i = 0; i < n; ++i) b.UnsafeAppend(i);
    std::shared_ptr<arrow::Array> a;
    (void)b.Finish(&a);
    return arrow::Table::Make(arrow::schema({arrow::field("n", arrow::int64())}), {a});
}

size_t Files(const fs::path& folder) {
    size_t n = 0;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(folder, ec))
        if (e.path().extension() == ".parquet") ++n;
    return n;
}
}  // namespace

int main() {
    const fs::path root = fs::temp_directory_path() / "cyxwiz_test_query_result_cache";
    std::error_code ec;
    fs::remove_all(root, ec);

    // The key: the SQL, its values and the tables read, in any order.
    QueryRequest q;
    q.sql = "SELECT count(*) FROM t";
    q.params.push_back(QueryParam::Of(int64_t{3}));
    q.params.push_back(QueryParam::Of(std::string("x")));
    const std::string key = QueryResultCache::Key(q, {"t=arrow:1", "w=side:a"});
    Check(key == QueryResultCache::Key(q, {"w=side:a", "t=arrow:1"}), "the key ignores the order of the tables");
    Check(key.size() == 32, "the key is a file name");
    QueryRequest other = q;
    other.sql += " ";
    Check(QueryResultCache::Key(other, {"t=arrow:1", "w=side:a"}) != key, "another SQL, another key");
    other = q;
    other.params[0] = QueryParam::Of(int64_t{4});
    Check(QueryResultCache::Key(other, {"t=arrow:1", "w=side:a"}) != key, "another value, another key");
    other = q;
    other.row_limit = 10;
    Check(QueryResultCache::Key(other, {"t=arrow:1", "w=side:a"}) != key, "another row limit, another key");
    Check(QueryResultCache::Key(q, {"t=arrow:2", "w=side:a"}) != key, "other data, another key");

    // A table's identity is its content: the same values give the same
    // identity, a changed value, a slice or a new column another one.
    const std::string id = QueryResultCache::ContentIdentity(*Small(4));
    Check(id == QueryResultCache::ContentIdentity(*Small(4)), "the same values in a new table: the same identity");
    Check(id != QueryResultCache::ContentIdentity(*Small(5)), "one more row: another identity");
    Check(id != QueryResultCache::ContentIdentity(*Small(5)->Slice(0, 4)), "a slice of a bigger table: another identity");
    auto changed = Small(4);
    arrow::DoubleBuilder vb;
    for (int i = 0; i < 4; ++i) (void)vb.Append(i == 2 ? 99.0 : i * 0.5);
    std::shared_ptr<arrow::Array> va;
    (void)vb.Finish(&va);
    changed = changed->SetColumn(1, arrow::field("value", arrow::float64()), std::make_shared<arrow::ChunkedArray>(va)).ValueOrDie();
    Check(id != QueryResultCache::ContentIdentity(*changed), "one changed value: another identity");
    Check(id != QueryResultCache::ContentIdentity(*Small(4)->RenameColumns({"a", "b", "c"}).ValueOrDie()), "renamed columns: another identity");

    QueryResultCache cache;
    Check(!cache.Save(key, Small(3)), "nothing is saved without a folder");
    Check(!cache.Load(key), "nothing is read without a folder");

    cache.SetFolder(root / "query_results");
    Check(!cache.Load(key), "a result not saved yet is not found");
    Check(cache.Save(key, Small(5), true), "a result is saved");
    bool truncated = false;
    auto back = cache.Load(key, &truncated);
    Check(back && back->num_rows() == 5 && back->num_columns() == 3, "the result reads back");
    Check(back->schema()->field(0)->type()->Equals(arrow::utf8()) && back->schema()->field(2)->type()->Equals(arrow::boolean()),
          "with its types");
    Check(truncated, "with its row-limit flag");
    Check(back->column(0)->GetScalar(4).ValueOrDie()->ToString() == "row 4", "with its values");
    Check(cache.Save("k2", Small(2), false), "another result");
    Check(cache.Load("k2", &truncated) && !truncated, "a whole result is not flagged");

    // Too big to save: it runs again next time.
    Check(!cache.Save("big", Big(5'000'000)), "a result over the limit is not saved");
    Check(!cache.Load("big"), "and is not found");

    // Pruning removes the least recently used results first.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    Check(cache.Save("k3", Small(2)), "a third result");
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    Check(cache.Load(key) != nullptr, "reading the first result counts as a use");
    Check(Files(root / "query_results") == 3, "three saved results");
    cache.Prune(1);
    Check(Files(root / "query_results") == 0, "pruning to nothing removes them all");
    Check(cache.Save(key, Small(5)) && cache.Save("k2", Small(2)), "saved again");
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    Check(cache.Load(key) != nullptr, "the first is used again");
    const uint64_t one = fs::file_size(cache.Folder() / "k2.parquet", ec);
    cache.Prune(one + one / 2);
    Check(Files(root / "query_results") == 1 && cache.Load(key) != nullptr && !cache.Load("k2"), "the result not used lately goes first");

    // The same key asked twice at once: one runs, the other waits for it.
    {
        bool leader = false, second = false;
        auto flight = cache.Join("flight", &leader);
        auto again = cache.Join("flight", &second);
        Check(leader && !second && flight == again, "the first caller leads, the second joins");
        QueryResult got;
        std::thread waiter([&] { QueryResultCache::Wait(*again, nullptr, &got); });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        QueryResult r;
        r.ok = true;
        r.table = Small(1);
        cache.Finish("flight", flight, r);
        waiter.join();
        Check(got.ok && got.table && got.table->num_rows() == 1, "the second caller gets the leader's result");
        bool fresh = false;
        cache.Join("flight", &fresh);
        Check(fresh, "after it finished the key is free again");

        // Waiting stops when the waiter is stopped.
        bool l2 = false, f2 = false;
        auto f = cache.Join("stopped", &l2);
        auto g = cache.Join("stopped", &f2);
        bool stop = false;
        QueryResult none;
        std::thread w2([&] { Check(!QueryResultCache::Wait(*g, [&] { return stop; }, &none), "a stopped waiter leaves"); });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        stop = true;
        w2.join();
        QueryResult failed;
        failed.error = "boom";
        cache.Finish("stopped", f, failed);
        Check(!QueryResultCache::Wait(*g, nullptr, &none), "a leader's failure gives the waiter nothing");
    }

    fs::remove_all(root, ec);
    std::cout << "query result cache: key, content identity, save and read back with types and the row-limit flag, size limit, "
                 "pruning, one run per key. OK\n";
    return 0;
}
