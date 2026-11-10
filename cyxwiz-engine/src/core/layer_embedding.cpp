#include "layer_embedding.h"

#include <cyxwiz/dimensionality_reduction.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>

namespace cyxwiz::layer_embedding {

const char* MethodName(Method method) {
    switch (method) {
        case Method::PCA: return "PCA";
        case Method::TSNE: return "t-SNE";
        case Method::UMAP: return "UMAP";
    }
    return "";
}

Projection Project(const std::vector<float>& rows, size_t items, size_t features, Method method) {
    Projection p;
    if (items < 3 || features == 0 || rows.size() != items * features) {
        p.error = "too few items to project (" + std::to_string(items) + ")";
        return p;
    }
    std::vector<std::vector<double>> data(items, std::vector<double>(features));
    for (size_t i = 0; i < items; ++i)
        for (size_t f = 0; f < features; ++f) data[i][f] = rows[i * features + f];
    std::vector<std::vector<double>> points;
    char about[96];
    if (method == Method::PCA) {
        const auto r = DimensionalityReduction::ComputePCA(data, 2);
        if (!r.success) {
            p.error = r.error_message;
            return p;
        }
        points = r.transformed;
        std::snprintf(about, sizeof(about), "PCA, 2 components, %.0f %% of the variance",
                      r.total_variance_explained * 100.0);
    } else if (method == Method::TSNE) {
        const int perplexity = static_cast<int>(std::clamp<size_t>((items - 1) / 3, 1, 30));
        const auto r = DimensionalityReduction::ComputetSNE(data, 2, perplexity);
        if (!r.success) {
            p.error = r.error_message;
            return p;
        }
        points = r.embeddings;
        std::snprintf(about, sizeof(about), "t-SNE, perplexity %d", perplexity);
    } else {
        const int neighbours = static_cast<int>(std::clamp<size_t>(items - 1, 2, 15));
        const auto r = DimensionalityReduction::ComputeUMAP(data, 2, neighbours);
        if (!r.success) {
            p.error = r.error_message;
            return p;
        }
        points = r.embeddings;
        std::snprintf(about, sizeof(about), "UMAP, %d neighbours", neighbours);
    }
    p.about = about;
    for (const auto& pt : points) {
        p.x.push_back(pt.size() > 0 ? static_cast<float>(pt[0]) : 0.0f);
        p.y.push_back(pt.size() > 1 ? static_cast<float>(pt[1]) : 0.0f);
    }
    return p;
}

Scaled ToUnit(const Projection& projection) {
    const auto scale = [](const std::vector<float>& v) {
        std::vector<float> out(v.size(), 0.5f);
        if (v.empty()) return out;
        const auto [lo, hi] = std::minmax_element(v.begin(), v.end());
        const float span = *hi - *lo;
        if (!(span > 0.0f) || !std::isfinite(span)) return out;
        for (size_t i = 0; i < v.size(); ++i) out[i] = (v[i] - *lo) / span;
        return out;
    };
    return {scale(projection.x), scale(projection.y)};
}

std::optional<size_t> Nearest(const Scaled& points, float px, float py, float radius, float aspect) {
    std::optional<size_t> best;
    float best_d = radius * radius;
    for (size_t i = 0; i < points.x.size() && i < points.y.size(); ++i) {
        const float dx = points.x[i] - px;
        const float dy = (points.y[i] - py) * aspect;
        const float d = dx * dx + dy * dy;
        if (d <= best_d) {
            best_d = d;
            best = i;
        }
    }
    return best;
}

std::vector<ClassCount> ClassCounts(const std::vector<int>& labels, const std::vector<std::string>& names) {
    std::map<int, size_t> counts;
    for (int l : labels)
        if (l >= 0) ++counts[l];
    std::vector<ClassCount> out;
    for (const auto& [label, count] : counts)
        out.push_back({label,
                       label < static_cast<int>(names.size()) && !names[static_cast<size_t>(label)].empty()
                           ? names[static_cast<size_t>(label)]
                           : "class " + std::to_string(label),
                       count});
    return out;
}

}  // namespace cyxwiz::layer_embedding
