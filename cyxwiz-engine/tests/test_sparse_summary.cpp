// The sparse features summary (TOFIX134 P3, board 19): counts, density, top
// features, per-row and per-feature use, by class, sample rows, label filter.

#include "../src/core/dashboard/sparse_summary.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace cyxwiz::dashboard;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}
bool Near(double a, double b) { return std::fabs(a - b) < 1e-6; }
}  // namespace

int main() {
    // 4 rows x 5 features (feel, like, sleep, work, sun):
    //   row 0 (sad):   feel 0.6, sleep 0.8
    //   row 1 (sad):   feel 0.5, like 0.5, sleep 0.7
    //   row 2 (happy): like 0.4, sun 0.9
    //   row 3 (happy): (empty)
    const int32_t offsets[] = {0, 2, 5, 7, 7};
    const int32_t indices[] = {0, 2, 0, 1, 2, 1, 4};
    const float values[] = {0.6f, 0.8f, 0.5f, 0.5f, 0.7f, 0.4f, 0.9f};
    SparseInput in;
    in.rows = 4;
    in.features = 5;
    in.offsets = offsets;
    in.indices = indices;
    in.values = values;
    in.feature_names = {"feel", "like", "sleep", "work", "sun"};
    in.labels = {"sad", "sad", "happy", "happy"};
    in.bytes = 2'000'000;

    SparseSummary s = SummarizeSparse(in);
    Check(s.rows_all == 4 && s.rows == 4 && s.features == 5 && s.nnz == 7, "counts");
    Check(Near(s.density, 7.0 / 20.0) && Near(s.memory_mb, 2.0), "density and memory");
    Check(s.classes.size() == 2 && s.class_rows_all[0] == 2 && s.class_rows_all[1] == 2, "two classes of two rows");
    Check(s.top_features.size() == 4 && s.top_features[0].first == "sleep" && Near(s.top_features[0].second, 1.5),
          "top feature: sleep 1.5 (work is never used): " + s.top_features[0].first);
    Check(s.top_features[1].first == "feel" && Near(s.top_features[1].second, 1.1), "then feel 1.1");
    Check(s.features_per_row == std::vector<double>({2, 3, 2, 0}), "features per row");
    Check(s.rows_per_feature.size() == 4, "rows per used feature (work left out)");
    Check(s.sample.size() == 4 && s.sample[0].row == 1 && s.sample[0].label == "sad" && s.sample[0].used == 2 &&
              s.sample[0].strongest == "sleep 0.80, feel 0.60",
          "first row, strongest first: " + s.sample[0].strongest);
    // By class: sleep (sad rows mean 0.75, happy 0), feel, like, sun.
    Check(s.by_class_features.size() == 4 && s.by_class_features[0] == "sleep", "by class: the top features");
    const size_t sad = s.classes[0] == "sad" ? 0 : 1;
    Check(Near(s.by_class_weight[0 * 2 + sad], 0.75) && Near(s.by_class_weight[0 * 2 + (1 - sad)], 0.0), "sleep by class");

    // The label filter: only the happy rows.
    s = SummarizeSparse(in, {"happy"});
    Check(s.rows == 2 && s.rows_all == 4 && s.nnz == 2, "kept rows: happy");
    Check(s.top_features[0].first == "sun" && s.top_features.size() == 2, "happy rows: sun, like");
    Check(s.class_rows_all[0] == 2 && s.class_rows[s.classes[0] == "happy" ? 0 : 1] == 2, "kept rows per class");
    Check(Near(s.density, 2.0 / 10.0), "density over the kept rows");

    // No labels, no names.
    in.labels.clear();
    in.feature_names.clear();
    s = SummarizeSparse(in);
    Check(s.classes.empty() && s.by_class_features.empty() && s.top_features[0].first == "f2", "no labels; features named f<n>");

    std::cout << "test_sparse_summary: all checks passed\n";
    return 0;
}
