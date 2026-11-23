#include "mj_history.h"

#include <algorithm>

namespace cyxwiz::plugin::mujoco {

void MjHistory::SetLimit(double seconds, size_t max_entries) {
    seconds_ = std::max(0.0, seconds);
    max_entries_ = std::max<size_t>(1, max_entries);
}

namespace {

// Simulated time between two steps kept one after the other (a new episode
// starts its clock again: no time between them)
double Between(const MjHistoryEntry& earlier, const MjHistoryEntry& later) {
    return later.episode == earlier.episode && later.time > earlier.time ? later.time - earlier.time : 0.0;
}

}  // namespace

void MjHistory::Add(MjHistoryEntry entry) {
    if (!entries_.empty()) span_ += Between(entries_.back(), entry);
    entries_.push_back(std::move(entry));
    while (entries_.size() > 1 && (entries_.size() > max_entries_ || (seconds_ > 0.0 && span_ > seconds_))) {
        span_ -= Between(entries_[0], entries_[1]);
        entries_.pop_front();
    }
}

void MjHistory::TruncateAfter(size_t index) {
    while (entries_.size() > index + 1) {
        span_ -= Between(entries_[entries_.size() - 2], entries_.back());
        entries_.pop_back();
    }
}

MjHistoryEntry MjHistory::Capture(const mjModel* m, const mjData* d, int episode, int step, double reward,
                                  int root_body) {
    MjHistoryEntry e;
    e.episode = episode;
    e.step = step;
    e.time = d->time;
    e.reward = reward;
    if (root_body > 0 && root_body < m->nbody) {
        mjtNum velocity[6];
        mj_objectVelocity(m, d, mjOBJ_BODY, root_body, velocity, 0);
        e.root_speed = mju_norm3(velocity + 3);
    }
    e.state.resize(static_cast<size_t>(mj_stateSize(m, kHistoryState)));
    mj_getState(m, d, e.state.data(), kHistoryState);
    e.controls.assign(d->ctrl, d->ctrl + m->nu);
    const size_t sensors = std::min(kHistorySensors, static_cast<size_t>(m->nsensordata));
    e.sensors.assign(d->sensordata, d->sensordata + sensors);
    return e;
}

bool MjHistory::Apply(const mjModel* m, mjData* d, const MjHistoryEntry& entry) {
    if (entry.state.size() != static_cast<size_t>(mj_stateSize(m, kHistoryState))) return false;
    mj_setState(m, d, entry.state.data(), kHistoryState);
    for (int i = 0; i < m->nu && static_cast<size_t>(i) < entry.controls.size(); ++i) d->ctrl[i] = entry.controls[i];
    mj_forward(m, d);
    return true;
}

size_t MjHistory::CountWithin(double seconds) const {
    if (entries_.empty()) return 0;
    const auto& newest = entries_.back();
    size_t count = 1;
    for (size_t i = entries_.size() - 1; i-- > 0;) {
        const auto& e = entries_[i];
        if (e.episode != newest.episode || newest.time - e.time > seconds) break;
        ++count;
    }
    return count;
}

}  // namespace cyxwiz::plugin::mujoco
