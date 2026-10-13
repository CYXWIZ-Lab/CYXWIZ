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

// sklearn.svm.SVC / SVR (libsvm): the dual solved by SMO with libsvm's
// second-order working set selection, stopping when the maximal violating pair
// is within tol. SVC fits one binary problem per class pair (a < b, a is +1);
// SVR one epsilon-SVR problem. Kernels: linear u.v, poly (gamma u.v + coef0)^
// degree, rbf exp(-gamma |u - v|^2), sigmoid tanh(gamma u.v + coef0).
struct CYXWIZ_API SVMModel {
    std::string kernel = "rbf";
    double gamma = 1.0;
    int degree = 3;
    double coef0 = 0.0;
    ClassicMatrix support_vectors;                 // [n_sv, d], shared by the problems
    std::vector<int> pair_first, pair_second;      // SVC: the classes of each problem
    std::vector<std::vector<int>> pair_support;    // per problem: rows of support_vectors
    std::vector<std::vector<double>> pair_coef;    // per problem: y alpha of those rows
    std::vector<double> rho;                       // per problem: decision = sum coef K - rho
    int n_classes = 0;                             // 0 for SVR
    int iterations = 0;
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

    // gamma "scale" (1 / (d * X.var())), "auto" (1 / d) or a number; 0 when unreadable.
    static double ResolveSVMGamma(const ClassicMatrix& x, const std::string& gamma);
    static SVMModel FitSVC(const ClassicMatrix& x, const std::vector<int>& labels, int n_classes, double C = 1.0,
                           const std::string& kernel = "rbf", double gamma = 1.0, int degree = 3,
                           double coef0 = 0.0, double tol = 1e-3);
    static SVMModel FitSVR(const ClassicMatrix& x, const std::vector<double>& target, double C = 1.0,
                           double epsilon = 0.1, const std::string& kernel = "rbf", double gamma = 1.0,
                           int degree = 3, double coef0 = 0.0, double tol = 1e-3);
    // Decision values [n, problems] (SVC pair order: (0,1), (0,2), ..., (1,2), ...).
    static ClassicMatrix SVMDecision(const SVMModel& model, const ClassicMatrix& x, std::string* error = nullptr);
    // One-vs-one votes; no probabilities.
    static ClassicPrediction PredictSVC(const SVMModel& model, const ClassicMatrix& x);
    static std::vector<double> PredictSVR(const SVMModel& model, const ClassicMatrix& x, std::string* error = nullptr);
};

}  // namespace cyxwiz
