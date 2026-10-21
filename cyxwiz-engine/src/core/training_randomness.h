#pragma once

#include <charconv>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <nlohmann/json.hpp>

namespace cyxwiz {

// -1 is deliberately unset: old graphs do not reset the model RNG.
inline int ParseModelRandomSeed(const std::string& value) {
    int seed = -1;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), seed);
    if (value.empty() || parsed.ec != std::errc{} ||
        parsed.ptr != value.data() + value.size() || seed < -1) {
        throw std::invalid_argument("Model RNG seed must be -1 (unset) or an integer from 0 to 2147483647");
    }
    return seed;
}

struct TrainingRandomness {
    int model_seed = -1;
    int dataloader_seed = -1; // Unknown for historical checkpoints.
    std::string generator;
    std::string execution_device;
};

inline void to_json(nlohmann::json& j, const TrainingRandomness& value) {
    j = {{"model_seed", value.model_seed}, {"dataloader_seed", value.dataloader_seed},
         {"generator", value.generator}, {"execution_device", value.execution_device},
         {"scope", value.model_seed < 0 ? "model_rng_unset" :
             "fresh_run_same_runtime_device_and_random_operation_order_arrayfire_only"},
         {"rng_continuation_state_present", false}};
}

inline void from_json(const nlohmann::json& j, TrainingRandomness& value) {
    for (const auto* key : {"model_seed", "dataloader_seed"}) {
        if (j.contains(key) && !j.at(key).is_number_integer())
            throw std::invalid_argument("Checkpoint seed provenance must use integers");
        if (j.contains(key) && j.at(key).is_number_unsigned() &&
            j.at(key).get<uint64_t>() > static_cast<uint64_t>((std::numeric_limits<int>::max)()))
            throw std::invalid_argument("Checkpoint seed provenance is out of range");
    }
    const auto seed = j.value("model_seed", int64_t{-1});
    const auto data_seed = j.value("dataloader_seed", int64_t{-1});
    if (seed < -1 || seed > (std::numeric_limits<int>::max)() ||
        data_seed < -1 || data_seed > (std::numeric_limits<int>::max)())
        throw std::invalid_argument("Checkpoint seed provenance is out of range");
    value.model_seed = static_cast<int>(seed);
    value.dataloader_seed = static_cast<int>(data_seed);
    value.generator = j.value("generator", std::string{});
    value.execution_device = j.value("execution_device", std::string{});
    // This record is provenance only; it never restores a random stream.
}

} // namespace cyxwiz
