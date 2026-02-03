#pragma once

#include "rl_training_executor.h"
#include <string>
#include <map>

namespace cyxwiz {

class RLScriptGenerator {
public:
    static std::string Generate(
        const RLTrainingConfig& config,
        const std::map<std::string, std::string>& reward_params,
        const std::map<std::string, std::string>& obs_filter_params,
        const std::string& save_path);
};

} // namespace cyxwiz
