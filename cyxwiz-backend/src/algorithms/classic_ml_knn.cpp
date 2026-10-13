// k-nearest neighbours on ArrayFire (TOFIX145 Group C), following scikit-learn's
// brute-force KNeighborsClassifier: distances from each query row to every
// training row, the n_neighbors nearest, votes per class (uniform, or 1/d with
// rows at distance 0 taking all the weight), probabilities as the vote shares
// and the first class with the most votes as the prediction.

#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/classic_ml.h"
#include "analysis_device.h"
#include "classic_ml_detail.h"

#include <algorithm>
#include <string>
#include <vector>

namespace cyxwiz {

using namespace analysis_detail;
using classic_detail::LabelRefusal;

KNNModel ClassicML::FitKNN(const ClassicMatrix& data, const std::vector<int>& labels, int n_classes, int n_neighbors,
                           const std::string& weights, const std::string& metric) {
    KNNModel model;
    if (const std::string refusal = MatrixRefusal(data); !refusal.empty()) {
        model.error_message = refusal;
        return model;
    }
    std::vector<double> counts;
    if (const std::string refusal = LabelRefusal(labels, data.size(), n_classes, counts); !refusal.empty()) {
        model.error_message = refusal;
        return model;
    }
    if (n_neighbors < 1 || static_cast<size_t>(n_neighbors) > data.size()) {
        model.error_message = "n_neighbors must be between 1 and the " + std::to_string(data.size()) + " training rows";
        return model;
    }
    if (weights != "uniform" && weights != "distance") {
        model.error_message = "weights must be uniform or distance";
        return model;
    }
    if (!IsKnownMetric(metric)) {
        model.error_message = "metric must be euclidean, manhattan or cosine";
        return model;
    }
    model.x = data;
    model.labels = labels;
    model.n_classes = n_classes;
    model.n_neighbors = n_neighbors;
    model.weights = weights;
    model.metric = metric;
    model.success = true;
    return model;
}

ClassicPrediction ClassicML::PredictKNN(const KNNModel& model, const ClassicMatrix& data) {
    ClassicPrediction result;
    if (!model.success || model.x.empty() || model.labels.size() != model.x.size() || model.n_classes < 1) {
        result.error_message = "The k-nearest neighbours model is not fitted";
        return result;
    }
    if (const std::string refusal = MatrixRefusal(data); !refusal.empty()) {
        result.error_message = refusal;
        return result;
    }
    if (data.front().size() != model.x.front().size()) {
        result.error_message = "Rows have " + std::to_string(data.front().size()) + " features; the model was fitted on " +
                               std::to_string(model.x.front().size());
        return result;
    }
    constexpr const char* kOperation = "ClassicML::PredictKNN";
    try {
        const af::array train = MatrixToDevice(model.x);
        const af::array query = MatrixToDevice(data).as(train.type());
        const af::array train_labels =
            af::array(static_cast<dim_t>(model.labels.size()), model.labels.data()).as(train.type());
        const dim_t n_query = query.dims(0), n_train = train.dims(0);
        const dim_t k = model.n_neighbors;
        const int n_classes = model.n_classes;
        const bool by_distance = model.weights == "distance";
        const dim_t block = RowBlock(n_train);

        af::array votes = af::constant(0.0, n_query, n_classes, train.type());
        for (dim_t start = 0; start < n_query; start += block) {
            const dim_t end = std::min(n_query, start + block);
            const af::seq rows(static_cast<double>(start), static_cast<double>(end - 1));
            const af::array distances = PairwiseDistances(query(rows, af::span), train, model.metric);
            af::array sorted, order;
            af::sort(sorted, order, distances, 1, true);
            const af::array nearest = sorted(af::span, af::seq(static_cast<double>(k)));
            const af::array nearest_index = order(af::span, af::seq(static_cast<double>(k)));
            const dim_t b = end - start;
            const af::array nearest_label = af::moddims(train_labels(af::flat(nearest_index)), b, k);

            af::array weight = af::constant(1.0, b, k, train.type());
            if (by_distance) {
                const af::array at_zero = nearest == 0.0;
                const af::array row_has_zero = af::tile(af::anyTrue(at_zero, 1), 1, static_cast<unsigned>(k));
                weight = af::select(row_has_zero, at_zero.as(train.type()), 1.0 / nearest);
            }
            af::array block_votes = af::constant(0.0, b, n_classes, train.type());
            for (int c = 0; c < n_classes; ++c)
                block_votes(af::span, c) = af::sum((nearest_label == static_cast<double>(c)).as(train.type()) * weight, 1);
            votes(rows, af::span) = block_votes;
        }
        af::array peak, labels;
        af::max(peak, labels, votes, 1);
        const af::array proba = votes / af::tile(af::sum(votes, 1), 1, static_cast<unsigned>(n_classes));
        result.labels = LabelsFromDevice(labels, kOperation);
        result.probabilities = MatrixFromDevice(proba, kOperation);
        result.success = true;
    } catch (const Refusal& refusal) {
        result = ClassicPrediction{};
        result.error_message = refusal.what();
    } catch (const af::exception& e) {
        result = ClassicPrediction{};
        result.error_message = DeviceError(kOperation, e);
    }
    return result;
}

}  // namespace cyxwiz
