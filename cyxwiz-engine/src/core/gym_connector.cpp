#include "gym_connector.h"
#include "../scripting/scripting_engine.h"
#include <spdlog/spdlog.h>
#include <sstream>
#include <algorithm>

namespace cyxwiz {

GymConnector::GymConnector(scripting::ScriptingEngine* engine)
    : engine_(engine) {
}

GymConnector::~GymConnector() {
    Close();
}

bool GymConnector::CreateEnv(const std::string& env_name) {
    if (!engine_) {
        spdlog::error("GymConnector: no scripting engine available");
        return false;
    }

    // Sanitize env_name to prevent code injection
    for (char c : env_name) {
        if (!std::isalnum(c) && c != '-' && c != '_' && c != '/' && c != '.') {
            spdlog::error("GymConnector: invalid character '{}' in env name '{}'", c, env_name);
            return false;
        }
    }

    env_name_ = env_name;

    // Import gymnasium (or gym as fallback) and create environment
    std::string setup = R"(
try:
    import gymnasium as gym
    _cyx_gym_api = 'gymnasium'
except ImportError:
    import gym
    _cyx_gym_api = 'gym'

_cyx_env = gym.make(')" + env_name + R"(')

# Extract environment info
import json
_obs_space = _cyx_env.observation_space
_act_space = _cyx_env.action_space

_env_info = {}
_env_info['obs_dim'] = int(_obs_space.shape[0]) if hasattr(_obs_space, 'shape') and len(_obs_space.shape) > 0 else 1

if hasattr(_act_space, 'n'):
    _env_info['discrete'] = True
    _env_info['num_actions'] = int(_act_space.n)
    _env_info['act_dim'] = 1
    _env_info['act_low'] = []
    _env_info['act_high'] = []
else:
    _env_info['discrete'] = False
    _env_info['num_actions'] = 0
    _env_info['act_dim'] = int(_act_space.shape[0])
    _env_info['act_low'] = _act_space.low.tolist()
    _env_info['act_high'] = _act_space.high.tolist()

print(json.dumps(_env_info))
)";

    std::string output = ExecutePython(setup);
    if (output.empty()) {
        spdlog::error("GymConnector: failed to create env '{}'", env_name);
        env_ready_ = false;
        return false;
    }

    // Parse JSON env info
    try {
        // Simple JSON parsing for our known structure
        env_info_.name = env_name;
        env_info_.valid = true;

        auto find_int = [&](const std::string& key) -> int {
            auto pos = output.find("\"" + key + "\"");
            if (pos == std::string::npos) return 0;
            pos = output.find(':', pos);
            if (pos == std::string::npos) return 0;
            try {
                return std::stoi(output.substr(pos + 1));
            } catch (...) { return 0; }
        };

        auto find_bool = [&](const std::string& key) -> bool {
            auto pos = output.find("\"" + key + "\"");
            if (pos == std::string::npos) return false;
            pos = output.find(':', pos);
            if (pos == std::string::npos) return false;
            std::string rest = output.substr(pos + 1);
            auto true_pos = rest.find("true");
            auto false_pos = rest.find("false");
            if (true_pos == std::string::npos) return false;
            if (false_pos == std::string::npos) return true;
            return true_pos < false_pos;
        };

        env_info_.observation_dim = find_int("obs_dim");
        env_info_.action_dim = find_int("act_dim");
        env_info_.discrete_actions = find_bool("discrete");
        env_info_.num_actions = find_int("num_actions");

        // Parse action bounds for continuous spaces
        if (!env_info_.discrete_actions) {
            auto parse_list = [&](const std::string& key) -> std::vector<float> {
                auto pos = output.find("\"" + key + "\"");
                if (pos == std::string::npos) return {};
                auto start = output.find('[', pos);
                auto end = output.find(']', start);
                if (start == std::string::npos || end == std::string::npos) return {};
                return ParseFloatList(output.substr(start, end - start + 1));
            };
            env_info_.action_low = parse_list("act_low");
            env_info_.action_high = parse_list("act_high");
        }

    } catch (const std::exception& e) {
        spdlog::error("GymConnector: failed to parse env info: {}", e.what());
        env_info_.valid = false;
        env_info_.error_message = e.what();
        env_ready_ = false;
        return false;
    }

    env_ready_ = true;
    spdlog::info("GymConnector: created env '{}' (obs_dim={}, act_dim={}, discrete={})",
                 env_name, env_info_.observation_dim, env_info_.action_dim,
                 env_info_.discrete_actions);
    return true;
}

std::vector<float> GymConnector::Reset(int seed) {
    if (!env_ready_) return {};

    std::string code;
    if (seed >= 0) {
        code = "import json\n"
               "_obs, _info = _cyx_env.reset(seed=" + std::to_string(seed) + ")\n"
               "print(json.dumps(_obs.tolist() if hasattr(_obs, 'tolist') else list(_obs)))";
    } else {
        code = "import json\n"
               "_obs, _info = _cyx_env.reset()\n"
               "print(json.dumps(_obs.tolist() if hasattr(_obs, 'tolist') else list(_obs)))";
    }

    std::string output = ExecutePython(code);
    return ParseFloatList(output);
}

StepResult GymConnector::Step(int action) {
    if (!env_ready_) return {};

    std::string code =
        "import json\n"
        "_obs, _reward, _terminated, _truncated, _info = _cyx_env.step(" +
        std::to_string(action) + ")\n"
        "_step_result = {"
        "'obs': _obs.tolist() if hasattr(_obs, 'tolist') else list(_obs),"
        "'reward': float(_reward),"
        "'done': bool(_terminated),"
        "'truncated': bool(_truncated)"
        "}\n"
        "print(json.dumps(_step_result))";

    std::string output = ExecutePython(code);
    StepResult result;
    if (output.empty()) return result;

    // Parse JSON result
    try {
        auto find_float = [&](const std::string& key) -> float {
            auto pos = output.find("\"" + key + "\"");
            if (pos == std::string::npos) return 0.0f;
            pos = output.find(':', pos);
            if (pos == std::string::npos) return 0.0f;
            try { return std::stof(output.substr(pos + 1)); }
            catch (...) { return 0.0f; }
        };
        auto find_bool = [&](const std::string& key) -> bool {
            auto pos = output.find("\"" + key + "\"");
            if (pos == std::string::npos) return false;
            pos = output.find(':', pos);
            if (pos == std::string::npos) return false;
            std::string rest = output.substr(pos + 1);
            auto true_pos = rest.find("true");
            auto false_pos = rest.find("false");
            if (true_pos == std::string::npos) return false;
            if (false_pos == std::string::npos) return true;
            return true_pos < false_pos;
        };

        // Parse observation
        auto obs_start = output.find("[");
        auto obs_end = output.find("]");
        if (obs_start != std::string::npos && obs_end != std::string::npos) {
            result.observation = ParseFloatList(output.substr(obs_start, obs_end - obs_start + 1));
        }

        result.reward = find_float("reward");
        result.done = find_bool("done");
        result.truncated = find_bool("truncated");
    } catch (const std::exception& e) {
        spdlog::warn("GymConnector: failed to parse step result: {}", e.what());
    }

    return result;
}

StepResult GymConnector::StepContinuous(const std::vector<float>& action) {
    if (!env_ready_) return {};

    // Build action list string
    std::ostringstream action_str;
    action_str << "[";
    for (size_t i = 0; i < action.size(); i++) {
        if (i > 0) action_str << ",";
        action_str << action[i];
    }
    action_str << "]";

    std::string code =
        "import json, numpy as np\n"
        "_obs, _reward, _terminated, _truncated, _info = _cyx_env.step(np.array(" +
        action_str.str() + ", dtype=np.float32))\n"
        "_step_result = {"
        "'obs': _obs.tolist() if hasattr(_obs, 'tolist') else list(_obs),"
        "'reward': float(_reward),"
        "'done': bool(_terminated),"
        "'truncated': bool(_truncated)"
        "}\n"
        "print(json.dumps(_step_result))";

    std::string output = ExecutePython(code);
    StepResult result;
    if (output.empty()) return result;

    try {
        auto find_float = [&](const std::string& key) -> float {
            auto pos = output.find("\"" + key + "\"");
            if (pos == std::string::npos) return 0.0f;
            pos = output.find(':', pos);
            if (pos == std::string::npos) return 0.0f;
            try { return std::stof(output.substr(pos + 1)); }
            catch (...) { return 0.0f; }
        };
        auto find_bool = [&](const std::string& key) -> bool {
            auto pos = output.find("\"" + key + "\"");
            if (pos == std::string::npos) return false;
            pos = output.find(':', pos);
            if (pos == std::string::npos) return false;
            std::string rest = output.substr(pos + 1);
            auto true_pos = rest.find("true");
            auto false_pos = rest.find("false");
            if (true_pos == std::string::npos) return false;
            if (false_pos == std::string::npos) return true;
            return true_pos < false_pos;
        };

        auto obs_start = output.find("[");
        auto obs_end = output.find("]");
        if (obs_start != std::string::npos && obs_end != std::string::npos) {
            result.observation = ParseFloatList(output.substr(obs_start, obs_end - obs_start + 1));
        }

        result.reward = find_float("reward");
        result.done = find_bool("done");
        result.truncated = find_bool("truncated");
    } catch (const std::exception& e) {
        spdlog::warn("GymConnector: failed to parse step result: {}", e.what());
    }

    return result;
}

void GymConnector::Close() {
    if (env_ready_ && engine_) {
        ExecutePython("_cyx_env.close()");
        env_ready_ = false;
    }
}

std::string GymConnector::ExecutePython(const std::string& code) {
    if (!engine_) return "";

    auto result = engine_->ExecuteCommand(code);
    if (!result.success) {
        spdlog::warn("GymConnector Python error: {}", result.error_message);
        return "";
    }

    // Trim whitespace
    std::string output = result.output;
    while (!output.empty() && (output.back() == '\n' || output.back() == '\r' || output.back() == ' ')) {
        output.pop_back();
    }
    return output;
}

std::vector<float> GymConnector::ParseFloatList(const std::string& output) {
    std::vector<float> result;
    if (output.empty()) return result;

    // Remove brackets
    std::string clean = output;
    clean.erase(std::remove(clean.begin(), clean.end(), '['), clean.end());
    clean.erase(std::remove(clean.begin(), clean.end(), ']'), clean.end());

    std::istringstream ss(clean);
    std::string token;
    while (std::getline(ss, token, ',')) {
        // Trim
        auto start = token.find_first_not_of(" \t\n\r");
        if (start == std::string::npos) continue;
        token = token.substr(start);
        try {
            result.push_back(std::stof(token));
        } catch (...) {
            // Skip invalid tokens
        }
    }
    return result;
}

} // namespace cyxwiz
