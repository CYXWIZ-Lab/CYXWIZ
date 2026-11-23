#pragma once

// The Viewport's Stress panel at work (TOFIX212 P4): what it asks of a running
// copy (MjStressRequest), applied the same way to the played copy of a policy
// playback (MjEnvBatch, on the playback thread) and to the Viewport's own run
// (the plugin, under the physics lock); and what it shows (MjStressLog: each
// disturbance and whether the robot fell, flipped over or stayed up). No ImGui.

#include "mj_scene.h"

#include <mujoco/mujoco.h>

#include <deque>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz::plugin::mujoco {

struct MjStressRequest {
    enum class Kind { Kick, Throw, Push, EndPush, Ground };
    Kind kind = Kind::Kick;
    std::array<double, 3> dv{0.0, 0.0, 0.0};  // Kick: the velocity change (world, m/s)
    double seconds = 0.0;                // Kick: spread over this long as a push; 0: at once
    MjThrownObject object;               // Throw
    MjThrow how;
    int body = -1;                       // Push: the body and its force and torque (xfrc_applied)
    mjtNum force[6] = {0, 0, 0, 0, 0, 0};
    MjGround ground;                     // Ground
    bool now = false;                    // Ground: start a new episode on it now, not at the next one

    static MjStressRequest MakeKick(const std::array<double, 3>& dv, double seconds);
    static MjStressRequest MakeThrow(const MjThrownObject& object, const MjThrow& how);
    static MjStressRequest MakePush(int body, const mjtNum force[6]);
    static MjStressRequest MakeEndPush();
    static MjStressRequest MakeGround(const MjGround& ground, bool now);
};

// One stressed run: its scene and what its requests left to do.
struct MjStressRun {
    MjScene scene;
    MjRobotSize robot;
    int last_slot = -1;                  // the object slot thrown last
    int push_body = -1;                  // a body held by Ctrl+drag, and its force
    mjtNum push_force[6] = {0, 0, 0, 0, 0, 0};
    int timed_body = -1;                 // a kick spread over time: the root, its force, until when
    std::array<double, 3> timed_force{0.0, 0.0, 0.0};
    double timed_until = 0.0;            // simulated time
    std::optional<MjGround> next_ground; // for the next episode
    bool cut_episode = false;            // start a new episode now (Ground "now")
    std::string message;                 // why the last request could not be done; empty: done
};

// What the requests act on. model / data: the copy being stressed, current
// after a rebuild. rebuild: the same model file with another scene, every
// copy's state carried over; false with the reason when it does not build.
struct MjStressHost {
    std::function<const mjModel*()> model;
    std::function<mjData*()> data;
    std::function<bool(const MjScene& scene, std::string& error)> rebuild;
    double step_seconds = 0.0;  // one environment step (a timed kick lasts whole steps)
};

// Applies and empties `requests`, in order. Kicks, throws and pushes act on
// the copy now; a ground waits in next_ground for the next episode (now: also
// sets cut_episode, for the caller to end this one after its step).
// StartNextGround builds it when an episode starts.
void ApplyStressRequests(std::vector<MjStressRequest>& requests, MjStressRun& run, const MjStressHost& host);

// At an episode's start: the ground waiting in next_ground, if any. False
// with run.message set when it does not build (the ground stays as it was).
bool StartNextGround(MjStressRun& run, const MjStressHost& host);

// The pushes on the copy, set before every step: the held Ctrl+drag push, and
// a timed kick until its time is up (then its force is taken off)
void ApplyPushes(MjStressRun& run, const mjModel* m, mjData* d);
// A new episode: a timed kick does not carry over
void EndTimedPush(MjStressRun& run);

// "Kick 1.5 m/s back", "Push 1.5 m/s back over 0.2 s", "Push 1.5 m/s at 40°, up
// 10°"; "Box 0.60×0.15×0.10 m, 3.0 kg at 6.0 m/s from 40°, up 25°"; "Rough 0.40, seed 7"
std::string DescribeKick(const std::array<double, 3>& dv, double seconds);
std::string DescribeThrow(const MjThrownObject& object, const MjThrow& how);
std::string DescribeGround(const MjGround& ground);

// What happened after each disturbance. Fed every frame with the shown copy's
// time and episode; an event is settled 3 s after it: "fell" when its episode
// ended by the task's own end (terminated), "flipped over" when the robot's
// root turned upside down, else "stayed up". An episode cut at its step limit
// leaves its open events as "stayed up"; one the user starts anew, as "new episode".
class MjStressLog {
public:
    enum class Outcome { Watching, Fell, Flipped, StayedUp, NewEpisode, Note };
    struct Event {
        double time = 0.0;  // simulated time in its episode
        int episode = 0;
        std::string what;
        Outcome outcome = Outcome::Watching;
        double outcome_time = 0.0;
    };
    static constexpr double kWatchSeconds = 3.0;
    static constexpr size_t kKept = 50;

    // A disturbance to watch
    void Add(double time, int episode, std::string what);
    // A line with nothing to watch (a ground chosen for the next episode)
    void Note(double time, int episode, std::string what);
    // The user started a new episode: open events end as NewEpisode
    void EndEpisodeByUser();
    // Each frame. ended_by_task: the episode before `episode` ended by the
    // task's own end (read when the episode changes)
    void Observe(double time, int episode, bool flipped, bool ended_by_task);
    void Clear();

    const std::deque<Event>& Events() const { return events_; }  // newest first
    static std::string OutcomeText(const Event& event);         // "fell at 16.40 s, episode ended"

private:
    std::deque<Event> events_;
    int last_episode_ = -1;
    double last_time_ = 0.0;
};

}  // namespace cyxwiz::plugin::mujoco
