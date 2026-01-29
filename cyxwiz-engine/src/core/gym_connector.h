#pragma once

#include "cyxwiz/rl_interface.h"
#include <string>
#include <vector>
#include <memory>

// Forward declare scripting engine
namespace scripting {
    class ScriptingEngine;
}

namespace cyxwiz {

/// Bridge to OpenAI Gym / Gymnasium environments via Python scripting engine
class GymConnector {
public:
    explicit GymConnector(scripting::ScriptingEngine* engine);
    ~GymConnector();

    /// Create a Gym environment by name (e.g. "CartPole-v1", "LunarLander-v2")
    bool CreateEnv(const std::string& env_name);

    /// Reset the environment, returns initial observation
    std::vector<float> Reset(int seed = -1);

    /// Take a step with a discrete action
    StepResult Step(int action);

    /// Take a step with a continuous action vector
    StepResult StepContinuous(const std::vector<float>& action);

    /// Get environment info (observation/action dimensions, etc.)
    EnvInfo GetEnvInfo() const { return env_info_; }

    /// Check if environment is ready
    bool IsReady() const { return env_ready_; }

    /// Close the environment
    void Close();

    /// Get the environment name
    const std::string& GetEnvName() const { return env_name_; }

private:
    /// Execute Python code and return output
    std::string ExecutePython(const std::string& code);

    /// Parse a Python list/tuple string to float vector
    std::vector<float> ParseFloatList(const std::string& output);

    scripting::ScriptingEngine* engine_ = nullptr;
    std::string env_name_;
    EnvInfo env_info_;
    bool env_ready_ = false;
};

} // namespace cyxwiz
