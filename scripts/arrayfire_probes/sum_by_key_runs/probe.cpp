// sumByKey on sorted int32 keys with repeated runs, against a host reference.
// Usage: sum_by_key_runs_probe <cpu|cuda|opencl|oneapi>
// Prints, per case, whether the unique keys and sums match the host's, and the
// first difference.

#include <arrayfire.h>

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

bool RunCase(const char* name, const std::vector<int>& keys, int columns) {
    // Host reference: runs of equal consecutive keys.
    std::vector<int> want_keys;
    std::vector<long long> want_sums;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (want_keys.empty() || keys[i] != want_keys.back()) {
            want_keys.push_back(keys[i]);
            want_sums.push_back(0);
        }
        want_sums.back() += 1;
    }
    const af::array k(static_cast<dim_t>(keys.size()), keys.data());
    af::array got_keys, got_sums;
    if (columns == 1) {
        af::sumByKey(got_keys, got_sums, k, af::constant(1, static_cast<dim_t>(keys.size()), s32), 0);
    } else {
        af::sumByKey(got_keys, got_sums, k, af::constant(1.0f, static_cast<dim_t>(keys.size()), columns, f32), 0);
        got_sums = got_sums(af::span, 0).as(s32);
    }
    std::vector<int> host_keys(static_cast<size_t>(got_keys.elements()));
    std::vector<int> host_sums(static_cast<size_t>(got_sums.elements()));
    got_keys.as(s32).host(host_keys.data());
    got_sums.as(s32).host(host_sums.data());
    bool ok = host_keys.size() == want_keys.size();
    size_t first = 0;
    for (; ok && first < want_keys.size(); ++first) {
        if (host_keys[first] != want_keys[first] || host_sums[first] != want_sums[first]) {
            ok = false;
            break;
        }
    }
    std::printf("%-40s %zu keys, %zu runs: %s", name, keys.size(), want_keys.size(), ok ? "ok" : "WRONG");
    if (!ok) {
        std::printf(" (got %zu runs", host_keys.size());
        if (first < want_keys.size() && first < host_keys.size())
            std::printf("; first difference at run %zu: key %d sum %d, expected key %d sum %lld", first,
                        host_keys[first], host_sums[first], want_keys[first], want_sums[first]);
        std::printf(")");
    }
    std::printf("\n");
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    const char* backend = argc > 1 ? argv[1] : "cpu";
    af::Backend b = AF_BACKEND_CPU;
    if (!std::strcmp(backend, "cuda")) b = AF_BACKEND_CUDA;
    if (!std::strcmp(backend, "opencl")) b = AF_BACKEND_OPENCL;
    if (!std::strcmp(backend, "oneapi")) b = AF_BACKEND_ONEAPI;
    af::setBackend(b);
    af::info();
    std::mt19937 rng(7);
    bool all = true;
    for (const int n : {64, 912, 1000, 4096, 100000}) {
        for (const int max_run : {1, 2, 4, 30}) {
            // Sorted keys with runs of 1..max_run, as an inclusive scan of
            // counts with zeros gives.
            std::vector<int> keys;
            int key = 0;
            std::uniform_int_distribution<int> run(1, max_run);
            while (static_cast<int>(keys.size()) < n) {
                const int r = run(rng);
                for (int i = 0; i < r && static_cast<int>(keys.size()) < n; ++i) keys.push_back(key);
                key += 1 + static_cast<int>(rng() % 20);
            }
            char name[64];
            std::snprintf(name, sizeof(name), "runs up to %d, 1 column", max_run);
            all = RunCase(name, keys, 1) && all;
            std::snprintf(name, sizeof(name), "runs up to %d, 8 columns", max_run);
            all = RunCase(name, keys, 8) && all;
        }
    }
    // The sequence fastText's first row mapping used: counts with zeros ->
    // inclusive scan (ends) -> sumByKey(ends, 1) -> marks(unique ends) = runs
    // -> inclusive scan = the item of each row.
    for (const int items : {912, 3072}) {
        std::vector<int> count(static_cast<size_t>(items));
        for (auto& c : count) c = (rng() % 10 < 3) ? 0 : 5 + static_cast<int>(rng() % 36);
        std::vector<int> want;
        for (int i = 0; i < items; ++i)
            for (int r = 0; r < count[static_cast<size_t>(i)]; ++r) want.push_back(i);
        const dim_t total = static_cast<dim_t>(want.size());
        // As fastText built them: looked up per word and zeroed for empty
        // slots, lazily (a JIT expression) before the scan.
        std::vector<int> words(static_cast<size_t>(items)), per_word(static_cast<size_t>(items));
        for (int i = 0; i < items; ++i) {
            words[static_cast<size_t>(i)] = count[static_cast<size_t>(i)] == 0 ? -1 : i;
            per_word[static_cast<size_t>(i)] = count[static_cast<size_t>(i)] == 0 ? 7 : count[static_cast<size_t>(i)];
        }
        const af::array word_ids(items, words.data()), table(items, per_word.data());
        const af::array valid = word_ids >= 0;
        const af::array safe = af::select(valid, word_ids, 0.0).as(s32);
        const af::array counts = af::select(valid, af::lookup(table, safe), 0.0).as(s32);
        const af::array ends = af::scan(counts, 0);
        af::array unique_ends, runs;
        af::sumByKey(unique_ends, runs, ends, af::constant(1, items, s32), 0);
        af::array marks = af::constant(0, total + 1, s32);
        marks(unique_ends) = runs;
        const af::array item = af::scan(marks, 0)(af::seq(0, static_cast<double>(total) - 1));
        std::vector<int> got(static_cast<size_t>(total));
        item.as(s32).host(got.data());
        size_t first = 0;
        while (first < got.size() && got[first] == want[first]) ++first;
        const bool ok = first == got.size();
        std::printf("%-40s %d items, %lld rows: %s", "row -> item via sumByKey + scatter", items,
                    static_cast<long long>(total), ok ? "ok" : "WRONG");
        if (!ok) std::printf(" (first difference at row %zu: item %d, expected %d)", first, got[first], want[first]);
        std::printf("\n");
        all = ok && all;
    }
    std::printf("%s\n", all ? "all cases match" : "SOME CASES WRONG");
    return all ? 0 : 1;
}
