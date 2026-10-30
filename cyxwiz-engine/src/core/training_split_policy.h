#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>

namespace cyxwiz {

// How a training graph's Data Split divides the rows (TOFIX178 S0-g2): the
// pipeline run marks each source row's partition ("__partition__") before
// the first data step, so the steps that learn from the data (scalers,
// vectorizers, encoders) learn from the Train rows only, and training reads
// the same partition. The shared split rule (split_partitioning.h) with the
// compiler's values: the Split's shares, seed, stratified and Label Column
// (else the Data Input's), and the Data Loader's shuffle. Supplied
// Validation / Test inputs (Data Inputs on the Split's pins 1 and 2,
// TOFIX178 S0-g4) are run too: their rows join the training input's rows
// with partition 1 / 2, so the Training branch's steps (fitted on the Train
// rows) transform them as well; a supplied role's share is 0. None when the
// graph has no Data Split, is a time series (its own split marks rows),
// neither shuffles nor stratifies and supplies no input (the rows keep their
// order), or supplies an input while the Training branch has more than one
// Data Input.
struct TrainingSplitPolicy {
    std::string label_column;
    float train = 0.8f;
    float val = 0.1f;
    float test = 0.1f;
    uint32_t seed = 42;
    bool stratified = false;
    bool shuffle = true;
    int training_input = -1;    // the Data Input the supplied rows join (-1: none supplied)
    int validation_input = -1;  // the Data Input on the Split's Validation pin
    int test_input = -1;        // the Data Input on the Split's Test pin
    std::string validation_label_column;  // its label column, renamed to label_column
    std::string test_label_column;
    bool Supplies() const { return validation_input >= 0 || test_input >= 0; }
};

// As carried in the pipeline JSON ("training_split").
inline nlohmann::json TrainingSplitPolicyJson(const TrainingSplitPolicy& policy) {
    nlohmann::json j = {{"label_column", policy.label_column}, {"train", policy.train}, {"val", policy.val},
                        {"test", policy.test}, {"seed", policy.seed}, {"stratified", policy.stratified},
                        {"shuffle", policy.shuffle}};
    if (policy.Supplies()) {
        j["training_input"] = policy.training_input;
        j["validation_input"] = policy.validation_input;
        j["test_input"] = policy.test_input;
        j["validation_label_column"] = policy.validation_label_column;
        j["test_label_column"] = policy.test_label_column;
    }
    return j;
}

inline std::optional<TrainingSplitPolicy> TrainingSplitPolicyFromJson(const nlohmann::json& pipeline_json) {
    const auto it = pipeline_json.find("training_split");
    if (it == pipeline_json.end() || !it->is_object()) return std::nullopt;
    TrainingSplitPolicy policy;
    policy.label_column = it->value("label_column", std::string());
    policy.train = it->value("train", policy.train);
    policy.val = it->value("val", policy.val);
    policy.test = it->value("test", policy.test);
    policy.seed = it->value("seed", policy.seed);
    policy.stratified = it->value("stratified", policy.stratified);
    policy.shuffle = it->value("shuffle", policy.shuffle);
    policy.training_input = it->value("training_input", policy.training_input);
    policy.validation_input = it->value("validation_input", policy.validation_input);
    policy.test_input = it->value("test_input", policy.test_input);
    policy.validation_label_column = it->value("validation_label_column", std::string());
    policy.test_label_column = it->value("test_label_column", std::string());
    return policy;
}

}  // namespace cyxwiz
