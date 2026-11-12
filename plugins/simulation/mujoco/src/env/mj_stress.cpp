#include "mj_stress.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace cyxwiz::plugin::mujoco {

MjStressRequest MjStressRequest::MakeKick(const std::array<double, 3>& dv, double seconds) {
    MjStressRequest r;
    r.kind = Kind::Kick;
    r.dv = dv;
    r.seconds = seconds;
    return r;
}

MjStressRequest MjStressRequest::MakeThrow(const MjThrownObject& object, const MjThrow& how) {
    MjStressRequest r;
    r.kind = Kind::Throw;
    r.object = object;
    r.how = how;
    return r;
}

MjStressRequest MjStressRequest::MakePush(int body, const mjtNum force[6]) {
    MjStressRequest r;
    r.kind = Kind::Push;
    r.body = body;
    std::copy(force, force + 6, r.force);
    return r;
}

MjStressRequest MjStressRequest::MakeEndPush() {
    MjStressRequest r;
    r.kind = Kind::EndPush;
    return r;
}

MjStressRequest MjStressRequest::MakeGround(const MjGround& ground, bool now) {
    MjStressRequest r;
    r.kind = Kind::Ground;
    r.ground = ground;
    r.now = now;
    return r;
}

void ApplyStressRequests(std::vector<MjStressRequest>& requests, MjStressRun& run, const MjStressHost& host) {
    for (const MjStressRequest& request : requests) {
        switch (request.kind) {
            case MjStressRequest::Kind::Kick: {
                const mjModel* m = host.model();
                const int root = RobotRootBody(m, run.robot);
                if (request.seconds <= 0.0 || host.step_seconds <= 0.0 || root < 0) {
                    KickRobot(m, host.data(), run.robot, request.dv);
                    break;
                }
                // The same change of velocity as a force held for whole steps: F = M dv / time
                const double steps = std::max(1.0, std::round(request.seconds / host.step_seconds));
                const double time = steps * host.step_seconds;
                const double mass = RobotMass(m, run.robot);
                EndTimedPush(run);
                run.timed_body = root;
                for (int i = 0; i < 3; ++i) run.timed_force[i] = mass * request.dv[i] / time;
                run.timed_until = host.data()->time + time - 0.5 * host.step_seconds;
                break;
            }
            case MjStressRequest::Kind::Throw: {
                MjScene next = run.scene;
                const int slot = PickObjectSlot(next, request.object, run.last_slot);
                if (next != run.scene) {
                    std::string error;
                    if (!host.rebuild(next, error)) {
                        run.message = error;
                        break;
                    }
                    run.scene = next;
                }
                if (ThrowObject(host.model(), host.data(), run.robot, slot, request.how))
                    run.last_slot = slot;
                break;
            }
            case MjStressRequest::Kind::Push:
                run.push_body = request.body;
                std::copy(request.force, request.force + 6, run.push_force);
                break;
            case MjStressRequest::Kind::EndPush:
                if (run.push_body >= 0 && run.push_body < host.model()->nbody)
                    std::fill(host.data()->xfrc_applied + 6 * run.push_body,
                              host.data()->xfrc_applied + 6 * run.push_body + 6, 0.0);
                run.push_body = -1;
                break;
            case MjStressRequest::Kind::Ground:
                // Built when the next episode starts (now: the caller starts it after
                // this step), so the robot never stands in a ground that appears
                run.next_ground = request.ground;
                if (request.now) run.cut_episode = true;
                break;
        }
    }
    requests.clear();
}

bool StartNextGround(MjStressRun& run, const MjStressHost& host) {
    if (!run.next_ground) return true;
    MjScene next = run.scene;
    next.ground = *run.next_ground;
    run.next_ground.reset();
    if (next == run.scene) return true;
    std::string error;
    if (!host.rebuild(next, error)) {
        run.message = error;
        return false;
    }
    run.scene = next;
    return true;
}

void EndTimedPush(MjStressRun& run) {
    run.timed_body = -1;
    run.timed_force = {0.0, 0.0, 0.0};
    run.timed_until = 0.0;
}

void ApplyPushes(MjStressRun& run, const mjModel* m, mjData* d) {
    if (run.timed_body >= 0 && run.timed_body < m->nbody) {
        mjtNum* f = d->xfrc_applied + 6 * run.timed_body;
        if (d->time < run.timed_until) {
            for (int i = 0; i < 3; ++i) f[i] = run.timed_force[i];
        } else {
            for (int i = 0; i < 3; ++i) f[i] = 0.0;
            EndTimedPush(run);
        }
    }
    if (run.push_body < 0 || run.push_body >= m->nbody) return;
    std::copy(run.push_force, run.push_force + 6, d->xfrc_applied + 6 * run.push_body);
}

namespace {

// "40°, up 25°", or the named direction it matches
std::string DirectionText(double around, double up) {
    char text[48];
    around = std::fmod(std::fmod(around, 360.0) + 360.0, 360.0);
    if (up >= 89.5) return "above";
    if (std::abs(up) < 0.5) {
        if (around < 0.5 || around > 359.5) return "ahead";
        if (std::abs(around - 180.0) < 0.5) return "behind";
        if (std::abs(around - 90.0) < 0.5) return "the left";
        if (std::abs(around - 270.0) < 0.5) return "the right";
    }
    std::snprintf(text, sizeof(text), "%.0f\xC2\xB0, up %.0f\xC2\xB0", around, up);
    return text;
}

}  // namespace

std::string DescribeKick(const std::array<double, 3>& dv, double seconds) {
    const double speed = std::sqrt(dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]);
    std::string way;
    if (speed <= 0.0) way = "nowhere";
    else if (dv[0] < 0.0 && dv[1] == 0.0 && dv[2] == 0.0) way = "back";
    else if (dv[0] > 0.0 && dv[1] == 0.0 && dv[2] == 0.0) way = "forward";
    else if (dv[2] > 0.0 && dv[0] == 0.0 && dv[1] == 0.0) way = "up";
    else {
        const double around = std::atan2(dv[1], dv[0]) * 180.0 / 3.14159265358979323846;
        const double up = std::asin(std::clamp(dv[2] / speed, -1.0, 1.0)) * 180.0 / 3.14159265358979323846;
        char text[48];
        std::snprintf(text, sizeof(text), "at %.0f\xC2\xB0, up %.0f\xC2\xB0", std::fmod(around + 360.0, 360.0), up);
        way = text;
    }
    char text[96];
    if (seconds > 0.0) std::snprintf(text, sizeof(text), "Push %.1f m/s %s over %.2f s", speed, way.c_str(), seconds);
    else std::snprintf(text, sizeof(text), "Kick %.1f m/s %s", speed, way.c_str());
    return text;
}

std::string DescribeThrow(const MjThrownObject& object, const MjThrow& how) {
    char size[64];
    const auto& s = object.size;
    switch (ShapeSizes(object.shape)) {
        case 1: std::snprintf(size, sizeof(size), "%.2f m", s[0]); break;
        case 2: std::snprintf(size, sizeof(size), "%.2f\xC3\x97%.2f m", s[0], s[1]); break;
        default: std::snprintf(size, sizeof(size), "%.2f\xC3\x97%.2f\xC3\x97%.2f m", s[0], s[1], s[2]); break;
    }
    char text[160];
    std::snprintf(text, sizeof(text), "%s %s, %.1f kg at %.1f m/s from %s", ShapeName(object.shape), size, object.mass,
                  how.speed, DirectionText(how.around, how.up).c_str());
    return text;
}

std::string DescribeGround(const MjGround& ground) {
    if (ground.kind == MjGroundKind::Flat) return "Flat";
    char text[64];
    std::snprintf(text, sizeof(text), "%s %.2f, seed %llu", GroundName(ground.kind), ground.difficulty,
                  static_cast<unsigned long long>(ground.seed));
    return text;
}

void MjStressLog::Add(double time, int episode, std::string what) {
    events_.push_front({time, episode, std::move(what), Outcome::Watching, 0.0});
    if (events_.size() > kKept) events_.pop_back();
}

void MjStressLog::Note(double time, int episode, std::string what) {
    events_.push_front({time, episode, std::move(what), Outcome::Note, 0.0});
    if (events_.size() > kKept) events_.pop_back();
}

void MjStressLog::EndEpisodeByUser() {
    for (Event& event : events_) {
        if (event.outcome == Outcome::Watching) event.outcome = Outcome::NewEpisode;
    }
}

void MjStressLog::Observe(double time, int episode, bool flipped, bool ended_by_task) {
    const bool episode_changed = last_episode_ >= 0 && episode != last_episode_;
    for (Event& event : events_) {
        if (event.outcome != Outcome::Watching) continue;
        if (event.episode != episode) {
            // Its episode is over: by the task's end (a fall), or cut at the step limit
            if (episode_changed && ended_by_task) {
                event.outcome = Outcome::Fell;
                event.outcome_time = last_time_;
            } else {
                event.outcome = Outcome::StayedUp;
            }
        } else if (flipped) {
            event.outcome = Outcome::Flipped;
            event.outcome_time = time;
        } else if (time - event.time >= kWatchSeconds) {
            event.outcome = Outcome::StayedUp;
        }
    }
    last_episode_ = episode;
    last_time_ = time;
}

void MjStressLog::Clear() {
    events_.clear();
    last_episode_ = -1;
    last_time_ = 0.0;
}

std::string MjStressLog::OutcomeText(const Event& event) {
    char text[64];
    switch (event.outcome) {
        case Outcome::Watching: return "watching";
        case Outcome::Fell:
            std::snprintf(text, sizeof(text), "fell at %.2f s, episode ended", event.outcome_time);
            return text;
        case Outcome::Flipped:
            std::snprintf(text, sizeof(text), "flipped over at %.2f s", event.outcome_time);
            return text;
        case Outcome::StayedUp: return "stayed up";
        case Outcome::NewEpisode: return "new episode started";
        case Outcome::Note: return {};
    }
    return {};
}

}  // namespace cyxwiz::plugin::mujoco
