#include "training_watch.h"

#include "../rl_playback_presentation.h"

#include <algorithm>

namespace cyxwiz::rl {

void TrainingWatch::Update(RLTrainingRun* run, RLTrainingRun::Watch watch, int every_updates) {
    if (run != run_) {
        Stop();
        run_ = run;
    }
    if (!run) return;
    const bool active = run->Active();
    run->SetWatch(active ? watch : RLTrainingRun::Watch::Off, every_updates);
    const auto policy = active && watch == RLTrainingRun::Watch::Preview ? run->PreviewPolicy() : nullptr;
    if (previewed_ && policy != previewed_) Stop();  // switched off, the run ended, or a new run's networks
    if (!policy || previewed_) return;

    // A preview started again (after Off) plays the networks the policy holds
    int holds_update = 0;
    long long holds_steps = 0;
    run->PreviewHolds(holds_update, holds_steps);
    update_ = holds_update;
    steps_ = holds_steps;
    started_ = 0;
    finished_update_ = 0;
    // On the preview's thread, before each episode: the newest networks, and
    // the view says which update they come from
    playback_.SetEpisodeStart([this, run](TrainedPolicy& played) -> std::optional<plugin::PluginShownCopyLabel> {
        int update = update_.load();
        long long steps = steps_.load();
        const bool first = started_.load() == 0;
        if (!first) finished_update_ = update;  // the episode that just ended played these
        ++started_;
        if (run->RefreshPreview(played, update, steps)) {
            update_ = update;
            steps_ = steps;
            return rl_playback::PreviewLabel(update);
        }
        if (first && update > 0) return rl_playback::PreviewLabel(update);  // the networks it holds
        return std::nullopt;
    });
    std::string error;
    if (playback_.Start(policy, PlaybackSpeed::RealTime, true, error)) previewed_ = policy;
}

void TrainingWatch::TakeNow() {
    if (run_) run_->TakeNetworksNow();
}

void TrainingWatch::Stop() {
    playback_.Stop();
    previewed_.reset();
    update_ = 0;
    steps_ = 0;
    started_ = 0;
    finished_update_ = 0;
}

std::string TrainingWatch::EnvironmentKey() const {
    return previewed_ ? previewed_->config.environment_key : std::string();
}

TrainingWatch::Status TrainingWatch::Get() const {
    Status s;
    const auto snap = playback_.Get();
    s.previewing = previewed_ && (snap.state == PolicyPlayback::State::Playing ||
                                  snap.state == PolicyPlayback::State::Paused);
    s.networks_update = update_.load();
    s.networks_steps = steps_.load();
    s.episodes = std::max(0, snap.episode - 1);  // the playback's own count: it records the return with it
    if (s.episodes > 0 && !snap.returns.empty()) {
        s.has_last_return = true;
        s.last_return = snap.returns.back();
        s.last_return_update = finished_update_.load();
    }
    if (snap.state == PolicyPlayback::State::Failed) s.message = snap.message;
    return s;
}

}  // namespace cyxwiz::rl
