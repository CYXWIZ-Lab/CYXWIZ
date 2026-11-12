#include "mj_stress.h"

#include <algorithm>
#include <cstdio>

namespace cyxwiz::plugin::mujoco {

MjStressRequest MjStressRequest::MakeKick(MjKick kick, double speed) {
    MjStressRequest r;
    r.kind = Kind::Kick;
    r.kick = kick;
    r.speed = speed;
    return r;
}

MjStressRequest MjStressRequest::MakeThrow(const MjThrownObject& object, MjThrowFrom from, double speed) {
    MjStressRequest r;
    r.kind = Kind::Throw;
    r.object = object;
    r.from = from;
    r.speed = speed;
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
            case MjStressRequest::Kind::Kick:
                KickRobot(host.model(), host.data(), run.robot, request.kick, request.speed);
                break;
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
                if (ThrowObject(host.model(), host.data(), run.robot, slot, request.from, request.speed))
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

void ApplyHeldPush(const MjStressRun& run, const mjModel* m, mjData* d) {
    if (run.push_body < 0 || run.push_body >= m->nbody) return;
    std::copy(run.push_force, run.push_force + 6, d->xfrc_applied + 6 * run.push_body);
}

std::string DescribeKick(MjKick kick, double speed) {
    char text[64];
    std::snprintf(text, sizeof(text), "Kick %.1f m/s %s", speed,
                  kick == MjKick::Back ? "back" : kick == MjKick::Forward ? "forward" : "up");
    return text;
}

std::string DescribeThrow(const MjThrownObject& object, MjThrowFrom from, double speed) {
    char text[96];
    std::snprintf(text, sizeof(text), "%s %.2f m, %.1f kg at %.1f m/s from %s", object.ball ? "Ball" : "Box",
                  object.size, object.mass, speed,
                  from == MjThrowFrom::Ahead ? "ahead" : from == MjThrowFrom::Behind ? "behind" : "above");
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
