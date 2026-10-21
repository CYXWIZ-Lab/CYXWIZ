#pragma once

// Word-vector tables between the host (row-major [rows, cols]) and the device
// ([rows, cols]), shared by the word2vec, GloVe and fastText code (TOFIX152).

#include "arrayfire_host_materialization.h"

#include <arrayfire.h>

#include <string>
#include <vector>

namespace cyxwiz {

inline af::array RowsToDevice(const std::vector<float>& values, dim_t rows, dim_t cols) {
    return af::array(cols, rows, values.data()).T();
}

inline std::vector<float> RowsToHost(const af::array& matrix, const char* operation) {
    std::vector<float> values(static_cast<size_t>(matrix.elements()));
    af::array transposed = matrix.T();
    transposed.eval();
    MaterializeArrayFireToHost(transposed, values.data(), ArrayFireHostSyncCategory::OutputMaterialization, operation,
                               "row_major_words_by_dim");
    return values;
}

// ASCII letters in lowercase (the vocabulary's lowercase fallback).
inline std::string AsciiLowercase(const std::string& word) {
    std::string lower = word;
    for (char& c : lower)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return lower;
}

// The mean of each group of rows of `table` (host, row-major [table_rows,
// dim]): `rows` index it, `group` (sorted) says which of `groups` each row
// belongs to; a group without rows is zeros. On the device; row-major
// [groups, dim] back on the host.
inline std::vector<float> GroupMeans(const std::vector<float>& table, dim_t table_rows, dim_t dim,
                                     const std::vector<int>& rows, const std::vector<int>& group, dim_t groups,
                                     const char* operation) {
    if (groups == 0) return {};
    af::array means = af::constant(0.0f, groups, dim, f32);
    if (!rows.empty()) {
        const af::array keys_in(static_cast<dim_t>(group.size()), group.data());
        const af::array picked = af::lookup(RowsToDevice(table, table_rows, dim),
                                            af::array(static_cast<dim_t>(rows.size()), rows.data()), 0);
        af::array keys, sums, count_keys, counts;
        af::sumByKey(keys, sums, keys_in, picked, 0);
        af::sumByKey(count_keys, counts, keys_in, af::constant(1.0f, static_cast<dim_t>(rows.size()), f32), 0);
        means(keys, af::span) = sums / af::tile(counts, 1, static_cast<unsigned>(dim));
    }
    return RowsToHost(means, operation);
}

}  // namespace cyxwiz
