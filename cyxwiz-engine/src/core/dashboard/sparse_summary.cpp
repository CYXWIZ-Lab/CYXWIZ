#include "sparse_summary.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <numeric>

namespace cyxwiz::dashboard {

namespace {

std::string FeatureName(const SparseInput& in, size_t f) {
    return f < in.feature_names.size() && !in.feature_names[f].empty() ? in.feature_names[f] : "f" + std::to_string(f);
}

// The indices of the `n` largest values, largest first (ties: lower index first).
std::vector<size_t> TopIndices(const std::vector<double>& v, size_t n) {
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), size_t{0});
    n = std::min(n, idx.size());
    std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(n), idx.end(),
                      [&](size_t a, size_t b) { return v[a] != v[b] ? v[a] > v[b] : a < b; });
    idx.resize(n);
    while (!idx.empty() && v[idx.back()] <= 0) idx.pop_back();
    return idx;
}

}  // namespace

SparseSummary SummarizeSparse(const SparseInput& in, const std::set<std::string>& keep, size_t top, size_t by_class_top, size_t sample_rows) {
    SparseSummary s;
    s.rows_all = in.rows;
    s.features = in.features;
    s.memory_mb = static_cast<double>(in.bytes) / 1e6;
    if (!in.offsets || in.rows == 0) return s;
    const bool labelled = in.labels.size() == in.rows;

    // Classes, most rows first.
    std::map<std::string, size_t> all_counts;
    if (labelled)
        for (const auto& l : in.labels) ++all_counts[l];
    for (const auto& [name, n] : all_counts) s.classes.push_back(name);
    std::stable_sort(s.classes.begin(), s.classes.end(), [&](const std::string& a, const std::string& b) { return all_counts[a] > all_counts[b]; });
    std::map<std::string, size_t> class_index;
    for (size_t c = 0; c < s.classes.size(); ++c) {
        class_index[s.classes[c]] = c;
        s.class_rows_all.push_back(all_counts[s.classes[c]]);
    }
    s.class_rows.assign(s.classes.size(), 0);

    std::vector<double> weight(in.features, 0.0), used_by(in.features, 0.0);
    std::vector<double> class_sum(in.features * s.classes.size(), 0.0);
    for (size_t r = 0; r < in.rows; ++r) {
        if (labelled && !keep.empty() && !keep.count(in.labels[r])) continue;
        ++s.rows;
        const size_t c = labelled ? class_index[in.labels[r]] : 0;
        if (labelled) ++s.class_rows[c];
        const int32_t begin = in.offsets[r], end = in.offsets[r + 1];
        s.nnz += static_cast<size_t>(std::max(0, end - begin));
        s.features_per_row.push_back(static_cast<double>(std::max(0, end - begin)));
        for (int32_t k = begin; k < end; ++k) {
            const int32_t f = in.indices[k];
            if (f < 0 || static_cast<size_t>(f) >= in.features) continue;
            const double v = in.values ? in.values[k] : 1.0;
            weight[static_cast<size_t>(f)] += v;
            used_by[static_cast<size_t>(f)] += 1.0;
            if (labelled) class_sum[static_cast<size_t>(f) * s.classes.size() + c] += v;
        }
        if (s.sample.size() < sample_rows) {
            SparseSummary::Row row;
            row.row = r + 1;
            row.label = labelled ? in.labels[r] : std::string();
            row.used = static_cast<size_t>(std::max(0, end - begin));
            std::vector<std::pair<double, int32_t>> parts;
            for (int32_t k = begin; k < end; ++k) parts.push_back({in.values ? in.values[k] : 1.0, in.indices[k]});
            std::sort(parts.begin(), parts.end(), [](const auto& a, const auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
            for (size_t i = 0; i < parts.size() && i < 4; ++i) {
                char buf[24];
                std::snprintf(buf, sizeof(buf), " %.2f", parts[i].first);
                row.strongest += (i ? ", " : "") + FeatureName(in, static_cast<size_t>(parts[i].second)) + buf;
            }
            s.sample.push_back(std::move(row));
        }
    }
    if (s.rows > 0 && in.features > 0) s.density = static_cast<double>(s.nnz) / (static_cast<double>(s.rows) * static_cast<double>(in.features));
    for (size_t f : TopIndices(weight, top)) s.top_features.push_back({FeatureName(in, f), weight[f]});
    for (size_t f = 0; f < in.features; ++f)
        if (used_by[f] > 0) s.rows_per_feature.push_back(used_by[f]);
    if (labelled) {
        // Top features by class: the overall top features, the mean weight per row of each class kept.
        for (size_t f : TopIndices(weight, by_class_top)) {
            s.by_class_features.push_back(FeatureName(in, f));
            for (size_t c = 0; c < s.classes.size(); ++c)
                s.by_class_weight.push_back(s.class_rows[c] ? class_sum[f * s.classes.size() + c] / static_cast<double>(s.class_rows[c]) : 0.0);
        }
    }
    return s;
}

}  // namespace cyxwiz::dashboard
