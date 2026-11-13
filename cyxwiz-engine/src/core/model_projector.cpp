#include "model_projector.h"

#include <cyxwiz/dimensionality_reduction.h>

#include <arrow/api.h>

#include <algorithm>
#include <cstdio>

namespace cyxwiz {

std::vector<ProjectorLayer> ProjectorLayers(const TestingMetrics& test) {
    std::vector<ProjectorLayer> layers;
    for (const auto& features : test.layer_features) {
        // Only a layer recorded for every sample (a quick test or a dropped
        // batch leaves it short).
        if (features.width == 0 || features.values.size() != features.width * test.predictions.size() ||
            test.predictions.size() < 3)
            continue;
        layers.push_back({features.layer, features.width});
    }
    return layers;
}

std::shared_ptr<arrow::Table> ProjectorFeaturesTable(const TestingMetrics& test, size_t layer) {
    const auto features = std::find_if(test.layer_features.begin(), test.layer_features.end(),
                                       [&](const TestingMetrics::LayerFeatures& f) { return f.layer == layer; });
    if (features == test.layer_features.end() || features->width == 0) return nullptr;
    const size_t width = features->width, rows = features->values.size() / width;
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> columns;
    for (size_t c = 0; c < width; ++c) {
        arrow::FloatBuilder builder;
        if (!builder.Reserve(static_cast<int64_t>(rows)).ok()) return nullptr;
        for (size_t r = 0; r < rows; ++r) builder.UnsafeAppend(features->values[r * width + c]);
        std::shared_ptr<arrow::Array> column;
        if (!builder.Finish(&column).ok()) return nullptr;
        fields.push_back(arrow::field("feature_" + std::to_string(c + 1), arrow::float32()));
        columns.push_back(column);
    }
    const auto ints = [&](const char* name, const auto& value_of) {
        arrow::Int64Builder builder;
        for (size_t r = 0; r < rows; ++r) (void)builder.Append(value_of(r));
        std::shared_ptr<arrow::Array> column;
        (void)builder.Finish(&column);
        fields.push_back(arrow::field(name, arrow::int64()));
        columns.push_back(column);
    };
    ints("true_class", [&](size_t r) { return r < test.ground_truth.size() ? static_cast<int64_t>(test.ground_truth[r]) : -1; });
    ints("predicted_class", [&](size_t r) { return r < test.predictions.size() ? static_cast<int64_t>(test.predictions[r]) : -1; });
    ints("row", [&](size_t r) { return r < test.row_ids.size() ? test.row_ids[r] : static_cast<int64_t>(r); });
    return arrow::Table::Make(arrow::schema(fields), columns, static_cast<int64_t>(rows));
}

ProjectorOutcome ProjectTestRows(const TestingMetrics& test, const ProjectorRequest& request,
                                 const std::function<bool()>& should_stop) {
    ProjectorOutcome out;
    const auto features = std::find_if(test.layer_features.begin(), test.layer_features.end(),
                                       [&](const TestingMetrics::LayerFeatures& f) { return f.layer == request.layer; });
    if (features == test.layer_features.end() || features->width == 0) {
        out.error = "The last test has no features for this layer.";
        return out;
    }
    const size_t width = features->width;
    out.rows_total = features->values.size() / width;
    if (out.rows_total < 3) {
        out.error = "Too few test rows to project.";
        return out;
    }
    const size_t step = std::max<size_t>(1, (out.rows_total + request.max_rows - 1) / std::max<size_t>(1, request.max_rows));
    std::vector<std::vector<double>> data;
    std::vector<int> samples;
    for (size_t r = 0; r < out.rows_total; r += step) {
        const float* row = features->values.data() + r * width;
        data.emplace_back(row, row + width);
        samples.push_back(static_cast<int>(r));
    }
    if (should_stop && should_stop()) {
        out.error = "Cancelled.";
        return out;
    }

    std::vector<std::vector<double>> xy;
    char label[96];
    switch (request.method) {
        case ProjectorMethod::PCA: {
            const auto pca = DimensionalityReduction::ComputePCA(data, 2, true, false);
            if (!pca.success) {
                out.error = "PCA: " + pca.error_message;
                return out;
            }
            xy = pca.transformed;
            const double first = pca.explained_variance_ratio.size() > 0 ? pca.explained_variance_ratio[0] : 0.0;
            const double second = pca.explained_variance_ratio.size() > 1 ? pca.explained_variance_ratio[1] : 0.0;
            std::snprintf(label, sizeof(label), "PC 1 \xC2\xB7 %.1f %% of the variance", first * 100.0);
            out.x_label = label;
            std::snprintf(label, sizeof(label), "PC 2 \xC2\xB7 %.1f %%", second * 100.0);
            out.y_label = label;
            break;
        }
        case ProjectorMethod::TSNE: {
            const int perplexity = static_cast<int>(std::min<size_t>(30, (data.size() - 1) / 3));
            const auto tsne = DimensionalityReduction::ComputetSNE(data, 2, std::max(2, perplexity), 200.0, 1000);
            if (!tsne.success) {
                out.error = "t-SNE: " + tsne.error_message;
                return out;
            }
            xy = tsne.embeddings;
            out.x_label = "t-SNE 1";
            out.y_label = "t-SNE 2";
            break;
        }
        case ProjectorMethod::UMAP: {
            const int neighbors = static_cast<int>(std::min<size_t>(15, data.size() - 1));
            const auto umap = DimensionalityReduction::ComputeUMAP(data, 2, neighbors, 0.1, 0);
            if (!umap.success) {
                out.error = "UMAP: " + umap.error_message;
                return out;
            }
            xy = umap.embeddings;
            out.x_label = "UMAP 1";
            out.y_label = "UMAP 2";
            break;
        }
    }
    for (size_t i = 0; i < xy.size() && i < samples.size(); ++i) {
        if (xy[i].size() < 2) continue;
        ProjectorPoint p;
        p.x = xy[i][0];
        p.y = xy[i][1];
        p.sample = samples[i];
        const auto s = static_cast<size_t>(p.sample);
        p.row = s < test.row_ids.size() ? test.row_ids[s] : p.sample;
        p.truth = s < test.ground_truth.size() ? test.ground_truth[s] : -1;
        p.predicted = s < test.predictions.size() ? test.predictions[s] : -1;
        p.confidence = s < test.confidences.size() ? test.confidences[s] : 0.0f;
        p.wrong = p.truth >= 0 && p.truth != p.predicted;
        out.points.push_back(p);
    }
    out.note = std::to_string(out.points.size()) + " of " + std::to_string(out.rows_total) + " test rows drawn" +
               (step > 1 ? " (every " + std::to_string(step) + "th)" : std::string()) +
               "; ringed points are mistakes. Click a kept row to open it in Explain.";
    out.ok = true;
    return out;
}

}  // namespace cyxwiz
