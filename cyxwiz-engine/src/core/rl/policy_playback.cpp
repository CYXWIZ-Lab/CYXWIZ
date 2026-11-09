#include "policy_playback.h"

#include "../../plugin/registries/plugin_environment_registry.h"

#include <spdlog/spdlog.h>

#include <chrono>

namespace cyxwiz::rl {

namespace {

constexpr size_t kKeptReturns = 50;

}  // namespace

PolicyPlayback::~PolicyPlayback() { Stop(); }

bool PolicyPlayback::Start(std::shared_ptr<TrainedPolicy> policy, PlaybackSpeed speed, bool best, std::string& error) {
    Stop();
    if (!policy || !policy->policy) {
        error = "There is no trained policy to play.";
        return false;
    }
    policy_ = std::move(policy);
    stop_ = false;
    paused_ = false;
    new_episode_ = false;
    best_ = best;
    speed_ = static_cast<int>(speed);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        snapshot_ = Snapshot{};
        snapshot_.state = State::Playing;
    }
    worker_ = std::thread(&PolicyPlayback::Run, this);
    return true;
}

void PolicyPlayback::Pause() {
    paused_ = true;
    std::lock_guard<std::mutex> lock(mutex_);
    if (snapshot_.state == State::Playing) snapshot_.state = State::Paused;
}

void PolicyPlayback::Resume() {
    paused_ = false;
    std::lock_guard<std::mutex> lock(mutex_);
    if (snapshot_.state == State::Paused) snapshot_.state = State::Playing;
}

void PolicyPlayback::NewEpisode() { new_episode_ = true; }

void PolicyPlayback::Stop() {
    stop_ = true;
    if (worker_.joinable()) worker_.join();
}

PolicyPlayback::Snapshot PolicyPlayback::Get() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return snapshot_;
}

std::shared_ptr<TrainedPolicy> PolicyPlayback::Policy() const { return policy_; }

void PolicyPlayback::Run() {
    const auto fail = [this](const std::string& why) {
        spdlog::error("Policy playback: {}", why);
        std::lock_guard<std::mutex> lock(mutex_);
        snapshot_.state = State::Failed;
        snapshot_.message = why;
    };
    try {
        auto& p = *policy_;
        std::string error;
        auto batch = plugin::PluginEnvironmentRegistry::Instance().Create(p.config.environment_key,
                                                                          p.config.environment_settings, 1, error);
        if (!batch) return fail(error);
        const auto& info = batch->Info();
        std::string show_error;
        const bool shown = batch->ShowCopy(0, show_error);
        if (!show_error.empty()) spdlog::warn("Policy playback: {}", show_error);

        const size_t obs_size = static_cast<size_t>(info.observation_size);
        const size_t act_size = static_cast<size_t>(info.action_size);
        std::vector<float> obs(obs_size), final_obs(obs_size);
        float reward = 0.0f;
        uint8_t terminated = 0, truncated = 0;
        const plugin::PluginEnvironmentStepBuffers out{obs.data(), &reward, &terminated, &truncated,
                                                       final_obs.data()};
        uint64_t episode_seed = p.config.ppo.seed + 1000;
        if (!batch->Reset(episode_seed, obs.data(), error)) return fail(error);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            snapshot_.episode = 1;
            snapshot_.observation = obs;
            snapshot_.shown_in_view = shown;
        }
        uint64_t sample_seed = p.config.ppo.seed ^ 0xA5A5A5A5ull;
        auto next_step = std::chrono::steady_clock::now();
        while (!stop_) {
            if (paused_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                next_step = std::chrono::steady_clock::now();
                continue;
            }
            if (new_episode_.exchange(false)) {
                if (!batch->Reset(++episode_seed, obs.data(), error)) return fail(error);
                std::lock_guard<std::mutex> lock(mutex_);
                ++snapshot_.episode;
                snapshot_.step = 0;
                snapshot_.episode_return = 0.0;
                snapshot_.observation = obs;
            }
            const auto action = PolicyActions(p, obs, 1, best_, ++sample_seed);
            if (!batch->Step(action.data(), out, error)) return fail(error);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                ++snapshot_.step;
                snapshot_.episode_return += reward;
                snapshot_.action.assign(action.begin(), action.begin() + static_cast<std::ptrdiff_t>(act_size));
                // The state the episode ended on stays on screen until the next step
                snapshot_.observation = terminated || truncated ? final_obs : obs;
                if (terminated || truncated) {
                    snapshot_.returns.push_back(snapshot_.episode_return);
                    if (snapshot_.returns.size() > kKeptReturns) snapshot_.returns.erase(snapshot_.returns.begin());
                    ++snapshot_.episode;
                    snapshot_.step = 0;
                    snapshot_.episode_return = 0.0;
                }
            }
            // Pace: the environment's step time, or four times faster, or none
            const auto speed = static_cast<PlaybackSpeed>(speed_.load());
            if (speed == PlaybackSpeed::Fastest || info.step_seconds <= 0.0) {
                next_step = std::chrono::steady_clock::now();
                continue;
            }
            const double seconds = info.step_seconds / (speed == PlaybackSpeed::Times4 ? 4.0 : 1.0);
            next_step += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(seconds));
            const auto now = std::chrono::steady_clock::now();
            if (next_step > now) std::this_thread::sleep_until(next_step);
            else next_step = now;  // behind: do not race to catch up
        }
        batch->ShowCopy(-1, show_error);
        std::lock_guard<std::mutex> lock(mutex_);
        snapshot_.state = State::Stopped;
        snapshot_.shown_in_view = false;
    } catch (const std::exception& e) {
        fail(std::string("the policy could not play: ") + e.what());
    }
}

}  // namespace cyxwiz::rl
