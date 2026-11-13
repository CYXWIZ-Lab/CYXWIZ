#include "policy_playback.h"

#include "../../plugin/registries/plugin_environment_registry.h"
#include "../execution_device_context.h"

#include <spdlog/spdlog.h>

#include <chrono>

namespace cyxwiz::rl {

namespace {

constexpr size_t kKeptReturns = 50;

void KeepReturn(std::vector<double>& returns, double r) {
    returns.push_back(r);
    if (returns.size() > kKeptReturns) returns.erase(returns.begin());
}

}  // namespace

std::optional<std::map<std::string, std::string>> GhostSettings(
    const std::map<std::string, std::string>& played, const std::map<std::string, std::string>& ghost_own,
    int ghost_observation_size, const std::function<int(const std::map<std::string, std::string>&)>& observation_size,
    std::string& error) {
    std::map<std::string, std::string> mixed = played;
    int size = observation_size(mixed);
    // Each of the ghost's own values that changes the observation is taken
    for (const auto& [key, value] : ghost_own) {
        if (size == ghost_observation_size) break;
        const auto it = mixed.find(key);
        if (it != mixed.end() && it->second == value) continue;
        auto trial = mixed;
        trial[key] = value;
        const int trial_size = observation_size(trial);
        if (trial_size >= 0 && trial_size != size) {
            mixed = std::move(trial);
            size = trial_size;
        }
    }
    if (size != ghost_observation_size) {
        error = "The ghost observes " + std::to_string(ghost_observation_size) + " values; this environment gives " +
                std::to_string(size) + ".";
        return std::nullopt;
    }
    return mixed;
}

PolicyPlayback::~PolicyPlayback() { Stop(); }

bool PolicyPlayback::Start(std::shared_ptr<TrainedPolicy> policy, PlaybackSpeed speed, bool best, std::string& error) {
    Stop();
    // SetConditions is for this Start only, whether it starts or not
    has_conditions_ = pending_conditions_;
    settings_ = std::move(pending_settings_);
    first_seed_ = pending_seed_;
    pending_conditions_ = false;
    pending_settings_.clear();
    episode_start_ = std::move(pending_episode_start_);
    pending_episode_start_ = nullptr;
    ghost_ = std::move(pending_ghost_);
    ghost_name_ = std::move(pending_ghost_name_);
    played_name_ = std::move(pending_played_name_);
    pending_ghost_.reset();
    pending_ghost_name_.clear();
    pending_played_name_.clear();
    remove_ghost_ = false;
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

void PolicyPlayback::SetConditions(std::map<std::string, std::string> settings, uint64_t first_seed) {
    pending_conditions_ = true;
    pending_settings_ = std::move(settings);
    pending_seed_ = first_seed;
}

void PolicyPlayback::SetEpisodeStart(EpisodeStart hook) { pending_episode_start_ = std::move(hook); }

void PolicyPlayback::SetGhost(std::shared_ptr<TrainedPolicy> ghost, std::string ghost_name, std::string played_name) {
    pending_ghost_ = std::move(ghost);
    pending_ghost_name_ = std::move(ghost_name);
    pending_played_name_ = std::move(played_name);
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
        // Act on the device the policy's networks live on (its training run's route)
        const auto activation = Device(p.device_type, p.device_id).ActivateExact(false);
        if (!activation.success) return fail("the policy's device did not start: " + activation.message);
        // The device is in use while it plays: Preferences > Devices waits, as for a training
        const ScopedActiveExecutionDeviceContext in_use;
        std::string error;
        auto batch = plugin::PluginEnvironmentRegistry::Instance().Create(
            p.config.environment_key, has_conditions_ ? settings_ : p.config.environment_settings, 1, error);
        if (!batch) return fail(error);
        const auto& info = batch->Info();
        std::string show_error;
        const bool shown = batch->ShowCopy(0, show_error);
        if (!show_error.empty()) spdlog::warn("Policy playback: {}", show_error);

        // The ghost: the same environment, with the played copy's conditions
        // where it observes the same
        std::unique_ptr<plugin::EnvironmentBatch> ghost_batch;
        bool ghost_in_view = false;
        std::string ghost_note;
        if (ghost_ && ghost_->policy) {
            auto& registry = plugin::PluginEnvironmentRegistry::Instance();
            const auto& played_settings = has_conditions_ ? settings_ : p.config.environment_settings;
            const std::string key = ghost_->config.environment_key;
            std::string ghost_error;
            const auto ghost_settings = GhostSettings(
                played_settings, ghost_->config.environment_settings,
                static_cast<int>(ghost_->config.observation_size),
                [&registry, &key](const std::map<std::string, std::string>& settings) {
                    plugin::PluginEnvironmentRegistry::Environment described;
                    std::string describe_error;
                    if (!registry.Describe(key, settings, described, describe_error)) return -1;
                    return described.info.observation_size;
                },
                ghost_error);
            if (key != p.config.environment_key) ghost_note = "The ghost plays another environment.";
            else if (!ghost_settings) ghost_note = ghost_error;
            else if (!(ghost_batch = registry.Create(key, *ghost_settings, 1, ghost_error))) ghost_note = ghost_error;
            if (ghost_batch) {
                std::string ghost_show_error;
                ghost_in_view = shown && ghost_batch->ShowCopyAsGhost(0, ghost_name_, ghost_show_error);
                if (!ghost_show_error.empty()) spdlog::warn("Policy playback: {}", ghost_show_error);
                if (shown && !ghost_in_view) ghost_note = "The view does not draw a ghost; its numbers are here.";
                // Both names in the view's live numbers (the training watch labels its own)
                if (shown && !episode_start_) {
                    plugin::PluginShownCopyLabel label;
                    label.rows.push_back({"Policy", played_name_});
                    batch->LabelShownCopy(label);
                }
            } else {
                spdlog::warn("Policy playback: the ghost does not play: {}", ghost_note);
            }
        }
        std::vector<float> ghost_obs(ghost_batch ? static_cast<size_t>(ghost_batch->ObservationSize()) : 0);
        std::vector<float> ghost_final(ghost_obs.size());
        float ghost_reward = 0.0f;
        uint8_t ghost_terminated = 0, ghost_truncated = 0;
        const plugin::PluginEnvironmentStepBuffers ghost_out{ghost_obs.data(), &ghost_reward, &ghost_terminated,
                                                             &ghost_truncated, ghost_final.data()};
        // The ghost stops (Remove the ghost, or a step that failed): the policy plays on
        const auto drop_ghost = [this, &ghost_batch, &batch, shown](const std::string& why) {
            if (!ghost_batch) return;
            std::string ignored;
            ghost_batch->ShowCopyAsGhost(-1, "", ignored);
            ghost_batch.reset();
            if (shown && !episode_start_) batch->LabelShownCopy({});  // the view's own words again
            std::lock_guard<std::mutex> lock(mutex_);
            snapshot_.ghost = false;
            snapshot_.ghost_in_view = false;
            snapshot_.ghost_note = why;
        };

        const size_t obs_size = static_cast<size_t>(info.observation_size);
        const size_t act_size = static_cast<size_t>(info.action_size);
        std::vector<float> obs(obs_size), final_obs(obs_size);
        float reward = 0.0f;
        uint8_t terminated = 0, truncated = 0;
        const plugin::PluginEnvironmentStepBuffers out{obs.data(), &reward, &terminated, &truncated,
                                                       final_obs.data()};
        // Before every episode: the hook may change the networks and the view's label
        const auto episode_starts = [this, &p, &batch, shown]() {
            if (!episode_start_) return;
            const auto label = episode_start_(p);
            if (label && shown) batch->LabelShownCopy(*label);
        };
        episode_starts();
        uint64_t episode_seed = has_conditions_ ? first_seed_ : p.config.ppo.seed + 1000;
        if (!batch->Reset(episode_seed, obs.data(), error)) return fail(error);
        if (ghost_batch && !ghost_batch->Reset(episode_seed, ghost_obs.data(), error)) drop_ghost(error);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            snapshot_.episode = 1;
            snapshot_.observation = obs;
            snapshot_.shown_in_view = shown;
            snapshot_.ghost = ghost_batch != nullptr;
            snapshot_.ghost_in_view = ghost_in_view && ghost_batch;
            if (snapshot_.ghost_note.empty()) snapshot_.ghost_note = ghost_note;  // a failed Reset said its own
            snapshot_.ghost_episode = ghost_batch ? 1 : 0;
        }
        uint64_t ghost_sample_seed = (ghost_ ? ghost_->config.ppo.seed : 0) ^ 0x5A5A5A5Aull;
        uint64_t sample_seed = p.config.ppo.seed ^ 0xA5A5A5A5ull;
        auto next_step = std::chrono::steady_clock::now();
        while (!stop_) {
            if (remove_ghost_.exchange(false)) drop_ghost("");  // also while paused
            if (paused_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                next_step = std::chrono::steady_clock::now();
                continue;
            }
            if (new_episode_.exchange(false)) {
                episode_starts();
                if (!batch->Reset(++episode_seed, obs.data(), error)) return fail(error);
                // Both start again together, on one seed
                if (ghost_batch && !ghost_batch->Reset(episode_seed, ghost_obs.data(), error)) drop_ghost(error);
                std::lock_guard<std::mutex> lock(mutex_);
                ++snapshot_.episode;
                snapshot_.step = 0;
                snapshot_.episode_return = 0.0;
                snapshot_.observation = obs;
                if (ghost_batch) {
                    ++snapshot_.ghost_episode;
                    snapshot_.ghost_step = 0;
                    snapshot_.ghost_return = 0.0;
                }
            }
            const auto action = PolicyActions(p, obs, 1, best_, ++sample_seed);
            if (!batch->Step(action.data(), out, error)) return fail(error);
            if (ghost_batch) {  // the ghost's step, with its own actions
                const auto ghost_action = PolicyActions(*ghost_, ghost_obs, 1, best_, ++ghost_sample_seed);
                if (!ghost_batch->Step(ghost_action.data(), ghost_out, error)) {
                    drop_ghost(error);
                } else {
                    std::lock_guard<std::mutex> lock(mutex_);
                    ++snapshot_.ghost_step;
                    snapshot_.ghost_return += ghost_reward;
                    if (ghost_terminated || ghost_truncated) {
                        KeepReturn(snapshot_.ghost_returns, snapshot_.ghost_return);
                        ++snapshot_.ghost_episode;
                        snapshot_.ghost_step = 0;
                        snapshot_.ghost_return = 0.0;
                    }
                }
            }
            if (terminated || truncated) episode_starts();  // the copy started its next episode in Step
            {
                std::lock_guard<std::mutex> lock(mutex_);
                ++snapshot_.step;
                snapshot_.episode_return += reward;
                snapshot_.action.assign(action.begin(), action.begin() + static_cast<std::ptrdiff_t>(act_size));
                // The state the episode ended on stays on screen until the next step
                snapshot_.observation = terminated || truncated ? final_obs : obs;
                if (terminated || truncated) {
                    KeepReturn(snapshot_.returns, snapshot_.episode_return);
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
        drop_ghost("");
        batch->ShowCopy(-1, show_error);
        std::lock_guard<std::mutex> lock(mutex_);
        snapshot_.state = State::Stopped;
        snapshot_.shown_in_view = false;
    } catch (const std::exception& e) {
        fail(std::string("the policy could not play: ") + e.what());
    }
}

}  // namespace cyxwiz::rl
