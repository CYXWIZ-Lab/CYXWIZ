#include "policy_stress_test.h"

#include "../../plugin/registries/plugin_environment_registry.h"
#include "../execution_device_context.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <numeric>
#include <sstream>

namespace cyxwiz::rl {

const char* const* StressKindNames() {
    static const char* const kNames[kStressKindCount] = {"Rough", "Slopes", "Steps", "Obstacles", "Kicks"};
    return kNames;
}

const char* StressKindName(StressKind kind) {
    const int i = static_cast<int>(kind);
    return i >= 0 && i < kStressKindCount ? StressKindNames()[i] : StressKindNames()[0];
}

bool SupportsStress(const plugin::PluginEnvironmentInfo& info) {
    bool grounds = false, pushes = false;
    for (const auto& s : info.settings) {
        grounds = grounds || s.key == "grounds";
        pushes = pushes || s.key == "push_every";
    }
    return grounds && pushes;
}

double StressDifficulty(const StressTestSpec& spec, int level) {
    if (spec.levels < 2) return 1.0;
    return static_cast<double>(level) / static_cast<double>(spec.levels - 1);
}

uint64_t StressSeed(const StressTestSpec& spec, int level) { return spec.seed + 1000ull * static_cast<uint64_t>(level); }

std::map<std::string, std::string> StressSettings(const std::map<std::string, std::string>& own, StressKind kind,
                                                  double difficulty) {
    std::map<std::string, std::string> settings = own;
    char level[32];
    std::snprintf(level, sizeof(level), "%.6g", difficulty);
    // The test's conditions only: no curriculum, no random physics, no training pushes
    settings["curriculum"] = "0";
    settings["friction_percent"] = "0";
    settings["mass_percent"] = "0";
    settings["motor_percent"] = "0";
    settings["push_every"] = "0";
    settings["grounds"] = "";
    settings["difficulty_from"] = level;
    settings["difficulty_to"] = level;
    switch (kind) {
        case StressKind::Rough: settings["grounds"] = "rough"; break;
        case StressKind::Slopes: settings["grounds"] = "slopes"; break;
        case StressKind::Steps: settings["grounds"] = "steps"; break;
        case StressKind::Obstacles: settings["grounds"] = "obstacles"; break;
        case StressKind::Kicks: {
            // A push every 1.5 s of up to 4 m/s at the hardest; none at 0
            char speed[32];
            std::snprintf(speed, sizeof(speed), "%.6g", 4.0 * difficulty);
            settings["push_every"] = difficulty > 0.0 ? "1.5" : "0";
            settings["push_speed"] = speed;
            break;
        }
    }
    return settings;
}

double StressLevelResult::MeanReturn() const {
    if (returns.empty()) return 0.0;
    return std::accumulate(returns.begin(), returns.end(), 0.0) / static_cast<double>(returns.size());
}

PolicyStressTest::~PolicyStressTest() { Stop(); }

bool PolicyStressTest::Start(std::vector<std::pair<std::string, std::shared_ptr<TrainedPolicy>>> policies,
                             StressTestSpec spec, std::string& error) {
    Stop();
    if (policies.empty()) {
        error = "There is no policy to test.";
        return false;
    }
    for (const auto& [name, policy] : policies) {
        if (!policy || !policy->policy) {
            error = name + " has no trained networks.";
            return false;
        }
        plugin::PluginEnvironmentRegistry::Environment env;
        if (!plugin::PluginEnvironmentRegistry::Instance().Find(policy->config.environment_key, env)) {
            error = policy->config.environment_key + " is not available: load its plugin.";
            return false;
        }
        if (!SupportsStress(env.info)) {
            error = env.info.name + " has no grounds or pushes to test with.";
            return false;
        }
    }
    if (spec.levels < 2 || spec.levels > 21 || spec.episodes < 1 || spec.episodes > 100) {
        error = "Use 2 to 21 levels and 1 to 100 episodes a level.";
        return false;
    }
    policies_ = std::move(policies);
    spec_ = spec;
    stop_ = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        snapshot_ = Snapshot{};
        snapshot_.state = State::Running;
        snapshot_.total = spec.levels * static_cast<int>(policies_.size());
        for (const auto& [name, policy] : policies_) snapshot_.results.push_back({name, {}});
    }
    worker_ = std::thread(&PolicyStressTest::Run, this);
    return true;
}

void PolicyStressTest::Stop() {
    stop_ = true;
    if (worker_.joinable()) worker_.join();
}

PolicyStressTest::Snapshot PolicyStressTest::Get() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return snapshot_;
}

namespace {

// The batch's own count of falls ("Falls", "NN %" of the episodes it saw)
int FallsFromStats(const plugin::EnvironmentBatch& batch, int episodes) {
    for (const auto& stat : batch.Stats()) {
        if (stat.label != "Falls") continue;
        const double percent = std::atof(stat.value.c_str());
        return static_cast<int>(std::lround(percent / 100.0 * episodes));
    }
    return 0;
}

}  // namespace

void PolicyStressTest::Run() {
    const auto fail = [this](const std::string& why) {
        spdlog::error("Stress test: {}", why);
        std::lock_guard<std::mutex> lock(mutex_);
        snapshot_.state = State::Failed;
        snapshot_.message = why;
    };
    try {
        for (size_t p = 0; p < policies_.size(); ++p) {
            auto& policy = *policies_[p].second;
            // Act on the device the policy's networks live on
            const auto activation = Device(policy.device_type, policy.device_id).ActivateExact(false);
            if (!activation.success) return fail("the policy's device did not start: " + activation.message);
            const ScopedActiveExecutionDeviceContext in_use;
            for (int level = 0; level < spec_.levels; ++level) {
                if (stop_) break;
                const double difficulty = StressDifficulty(spec_, level);
                const int n = spec_.episodes;
                std::string error;
                auto batch = plugin::PluginEnvironmentRegistry::Instance().Create(
                    policy.config.environment_key,
                    StressSettings(policy.config.environment_settings, spec_.kind, difficulty), n, error);
                if (!batch) return fail(policies_[p].first + ": " + error);
                const size_t obs_size = static_cast<size_t>(batch->ObservationSize());
                const size_t act_size = static_cast<size_t>(batch->ActionSize());
                if (obs_size != policy.config.observation_size ||
                    act_size != static_cast<size_t>(policy.config.environment.action_size))
                    return fail(policies_[p].first + " was trained on " + std::to_string(policy.config.observation_size) +
                                " observed values; the environment now gives " + std::to_string(obs_size) +
                                ": its plugin or settings changed since.");
                std::vector<float> obs(n * obs_size), final_obs(n * obs_size), rewards(n);
                std::vector<uint8_t> terminated(n), truncated(n);
                const plugin::PluginEnvironmentStepBuffers out{obs.data(), rewards.data(), terminated.data(),
                                                               truncated.data(), final_obs.data()};
                if (!batch->Reset(StressSeed(spec_, level), obs.data(), error)) return fail(error);
                StressLevelResult result;
                result.difficulty = difficulty;
                std::vector<double> returns(n, 0.0);
                std::vector<uint8_t> done(n, 0);
                int ended = 0, terminated_count = 0;
                uint64_t sample_seed = spec_.seed;
                while (ended < n && !stop_) {
                    // PolicyActions holds the policy's own lock while the networks run
                    std::vector<float> actions = PolicyActions(policy, obs, static_cast<size_t>(n), true, ++sample_seed);
                    actions.resize(n * act_size);
                    if (!batch->Step(actions.data(), out, error)) return fail(error);
                    for (int i = 0; i < n; ++i) {
                        if (done[i]) continue;
                        returns[i] += rewards[i];
                        if (terminated[i] || truncated[i]) {
                            done[i] = 1;
                            ++ended;
                            if (terminated[i]) ++terminated_count;
                        }
                    }
                }
                if (stop_) break;
                result.returns = returns;
                // A copy that ended early starts again and may end again before the others finish,
                // so the batch's own count covers more than the test's episodes: then the early
                // endings are the falls. When none ended early, every copy ended once, together,
                // and the batch's count (ended early or upside down) is exactly theirs.
                result.falls = terminated_count > 0 ? terminated_count : std::min(n, FallsFromStats(*batch, n));
                std::lock_guard<std::mutex> lock(mutex_);
                snapshot_.results[p].levels.push_back(std::move(result));
                ++snapshot_.done;
            }
        }
        std::lock_guard<std::mutex> lock(mutex_);
        snapshot_.state = stop_ ? State::Stopped : State::Finished;
    } catch (const std::exception& e) {
        fail(std::string("the stress test failed: ") + e.what());
    }
}

std::string StressResultsCsv(const std::vector<StressPolicyResult>& results, int episodes) {
    std::ostringstream csv;
    csv << "difficulty";
    const auto quoted = [](std::string text) {  // CSV: a quote inside a field is doubled
        for (size_t at = text.find('"'); at != std::string::npos; at = text.find('"', at + 2)) text.insert(at, 1, '"');
        return text;
    };
    for (const auto& r : results)
        csv << ",\"" << quoted(r.name) << " mean return\",\"" << quoted(r.name) << " falls\"";
    csv << ",episodes\n";
    size_t levels = 0;
    for (const auto& r : results) levels = std::max(levels, r.levels.size());
    for (size_t l = 0; l < levels; ++l) {
        double difficulty = 0.0;
        for (const auto& r : results)
            if (l < r.levels.size()) difficulty = r.levels[l].difficulty;
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.4g", difficulty);
        csv << buf;
        for (const auto& r : results) {
            if (l < r.levels.size()) {
                std::snprintf(buf, sizeof(buf), ",%.2f,%d", r.levels[l].MeanReturn(), r.levels[l].falls);
                csv << buf;
            } else {
                csv << ",,";
            }
        }
        csv << "," << episodes << "\n";
    }
    return csv.str();
}

}  // namespace cyxwiz::rl
