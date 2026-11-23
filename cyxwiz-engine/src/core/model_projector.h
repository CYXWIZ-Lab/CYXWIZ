#pragma once

// The Model window's Projector tab (TOFIX129 D11): what a layer of the
// trained model makes of the test rows - each row's features at that layer
// (TestingMetrics::layer_features, recorded by Test) brought down to two
// dimensions with the SDK's PCA, t-SNE or UMAP, as Data Studio's Projection
// tab does for a table.

#include "test_executor.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <memory>
#include <vector>

namespace arrow {
class Table;
}

namespace cyxwiz {

enum class ProjectorMethod { PCA = 0, TSNE = 1, UMAP = 2 };

struct ProjectorRequest {
    size_t layer = 0;                 // a TestingMetrics::layer_features entry's layer
    ProjectorMethod method = ProjectorMethod::PCA;
    size_t max_rows = 2000;           // more rows are drawn evenly (every k-th)
};

struct ProjectorPoint {
    double x = 0.0, y = 0.0;
    int sample = 0;                   // index into the test's predictions
    long long row = -1;               // the dataset row, when known
    int truth = -1, predicted = -1;
    float confidence = 0.0f;
    bool wrong = false;
};

struct ProjectorOutcome {
    bool ok = false;
    std::string error;
    std::vector<ProjectorPoint> points;
    size_t rows_total = 0;            // test rows with features
    std::string x_label, y_label;     // "PC 1 · 22.4 % of the variance", "t-SNE 1"
    std::string note;                 // what was drawn
};

// Blocking (t-SNE and UMAP take seconds on thousands of rows): run it on a task.
ProjectorOutcome ProjectTestRows(const TestingMetrics& test, const ProjectorRequest& request,
                                 const std::function<bool()>& should_stop = {});

// The layers Projector can show: the layer index and its features' width.
struct ProjectorLayer {
    size_t layer = 0;
    size_t width = 0;
};
std::vector<ProjectorLayer> ProjectorLayers(const TestingMetrics& test);

// The layer's features as a table for Data Studio's Projection tab: one row
// per test sample - feature_1 .. feature_W, true_class, predicted_class,
// row. nullptr when the test has no features for the layer.
std::shared_ptr<arrow::Table> ProjectorFeaturesTable(const TestingMetrics& test, size_t layer);

}  // namespace cyxwiz
