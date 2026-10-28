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
// (else the Data Input's), and the Data Loader's shuffle. None when the
// graph has no Data Split, is a time series (its own split marks rows), has
// supplied Validation / Test inputs (not yet), or neither shuffles nor
// stratifies (the rows keep their order).
struct TrainingSplitPolicy {
    std::string label_column;
    float train = 0.8f;
    float val = 0.1f;
    float test = 0.1f;
    uint32_t seed = 42;
    bool stratified = false;
    bool shuffle = true;
};

// As carried in the pipeline JSON ("training_split").
inline nlohmann::json TrainingSplitPolicyJson(const TrainingSplitPolicy& policy) {
    return {{"label_column", policy.label_column}, {"train", policy.train}, {"val", policy.val},
            {"test", policy.test}, {"seed", policy.seed}, {"stratified", policy.stratified},
            {"shuffle", policy.shuffle}};
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
    return policy;
}

}  // namespace cyxwiz
