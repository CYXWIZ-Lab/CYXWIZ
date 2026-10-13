#pragma once

// Classical supervised models on ArrayFire (TOFIX145 Group C): fit on host
// rows, compute on the selected device (float64 where it has it), results back
// as host values. scikit-learn is the oracle: same formulas, same class order
// (classes are 0..k-1 in the caller's sorted label order), same tie breaking.

#include "cyxwiz/api_export.h"

#include <string>
#include <vector>

namespace cyxwiz {

using ClassicMatrix = std::vector<std::vector<double>>;

// Class predictions with per-class probabilities [n, k] (empty when the model
// has none).
struct CYXWIZ_API ClassicPrediction {
    std::vector<int> labels;
    ClassicMatrix probabilities;

    bool success = false;
    std::string error_message;
};

// sklearn.naive_bayes.GaussianNB: per-class mean and variance (ddof 0), every
// variance + var_smoothing * the largest feature variance, priors from counts.
struct CYXWIZ_API GaussianNBModel {
    ClassicMatrix theta;               // [k, d] class means
    ClassicMatrix var;                 // [k, d] class variances incl. epsilon
    std::vector<double> class_prior;   // [k]
    std::vector<double> class_count;   // [k]
    double epsilon = 0.0;

    bool success = false;
    std::string error_message;
};

// sklearn.neighbors.KNeighborsClassifier (brute force): the training rows are
// the model. weights "uniform" or "distance" (1/d; rows at distance 0 take all
// the weight), metric "euclidean", "manhattan" or "cosine".
struct CYXWIZ_API KNNModel {
    ClassicMatrix x;            // [n, d] training rows
    std::vector<int> labels;    // [n]
    int n_classes = 0;
    int n_neighbors = 5;
    std::string weights = "uniform";
    std::string metric = "euclidean";

    bool success = false;
    std::string error_message;
};

// sklearn.linear_model.LogisticRegression (lbfgs, L2): two classes fit one
// coefficient row (sigmoid), more a multinomial softmax. Minimises the mean log
// loss + ||coef||^2 / (2 C n) (intercept not penalised); converged when the
// largest gradient entry is at most tol or a step lowers the loss by no more
// than 64 eps of it (sklearn's ftol); otherwise stops after max_iter.
struct CYXWIZ_API LogisticModel {
    ClassicMatrix coef;              // [1 or k, d]
    std::vector<double> intercept;   // [1 or k]
    int n_classes = 0;
    int n_iter = 0;
    double loss = 0.0;
    bool converged = false;

    bool success = false;
    std::string error_message;
};

class CYXWIZ_API ClassicML {
public:
    // labels in 0..n_classes-1; every class must have at least one row.
    static GaussianNBModel FitGaussianNB(const ClassicMatrix& x, const std::vector<int>& labels, int n_classes,
                                         double var_smoothing = 1e-9);
    static ClassicPrediction PredictGaussianNB(const GaussianNBModel& model, const ClassicMatrix& x);

    // Checks and keeps the training rows; n_neighbors <= rows.
    static KNNModel FitKNN(const ClassicMatrix& x, const std::vector<int>& labels, int n_classes, int n_neighbors = 5,
                           const std::string& weights = "uniform", const std::string& metric = "euclidean");
    static ClassicPrediction PredictKNN(const KNNModel& model, const ClassicMatrix& x);

    static LogisticModel FitLogistic(const ClassicMatrix& x, const std::vector<int>& labels, int n_classes,
                                     double C = 1.0, int max_iter = 100, double tol = 1e-4);
    static ClassicPrediction PredictLogistic(const LogisticModel& model, const ClassicMatrix& x);
};

}  // namespace cyxwiz
