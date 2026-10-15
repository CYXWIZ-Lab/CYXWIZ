#pragma once

// Reads a saved cyxwiz_classic_model artifact (written by the classic ML nodes,
// classic_model_common.h) back into the backend model it was fitted as, for the
// Classic Model Predictor node (TOFIX145).

#include <cyxwiz/classic_ml.h>

#include <string>
#include <vector>

namespace cyxwiz {

struct ClassicModelArtifact {
    // NaiveBayesClassifier, KNNClassifier, LogisticRegressionNode, SVMClassifier
    // or SVMRegressor: the node that fitted it.
    std::string model_type;
    std::vector<std::string> feature_names;  // the training feature columns, in order
    std::vector<std::string> class_labels;   // classifiers: class text in class order
    bool numeric_labels = false;             // classifiers: labels were numbers
    std::string target;                      // SVMRegressor: the target column

    // The fitted model; only the one matching model_type is set.
    GaussianNBModel naive_bayes;
    KNNModel knn;
    LogisticModel logistic;
    SVMModel svm;

    bool IsRegressor() const { return model_type == "SVMRegressor"; }
};

// Loads and checks the artifact at `path`; false with `error` set when it is
// not a readable version 1 cyxwiz_classic_model of a known type.
bool LoadClassicModelArtifact(const std::string& path, ClassicModelArtifact& artifact, std::string* error);

}  // namespace cyxwiz
