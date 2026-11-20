#pragma once

// What the RL Training Dashboard says (TOFIX129 B4-8, approved mockup
// RL-1..RL-6): the header in plain words, the figure cards and the updates
// CSV, from a PPO run's updates or a script's reported values. Pure data in
// and out (no ImGui, no ArrayFire); the panel only draws it.

#include "rl_device_presentation.h"

#include <cmath>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz::rl_dashboard {

// One PPO update as the dashboard keeps it (rl::PpoUpdateMetrics' figures).
// NaN where the update has no value (no episode finished yet, a discrete
// action space's action std).
struct Update {
    int update = 0;
    double steps = 0;  // environment steps so far
    double mean_return = NAN;
    double mean_length = NAN;
    double policy_loss = 0;
    double value_loss = 0;
    double entropy = 0;  // the policy's entropy (minus the entropy loss)
    double approx_kl = 0;
    double clip_fraction = 0;
    double explained_variance = NAN;
    double action_std = NAN;
    double steps_per_second = 0;
    bool reached_target = false;
};

enum class State {
    NoRun,             // nothing reported yet
    FromScript,        // a script's values (pycyxwiz.rl_update_metric)
    Starting,          // a PPO run before its device is known
    WaitingForGpu,
    Training,
    Paused,
    ReachedTarget,
    RanAllSteps,
    Stopped,
    Failed,
};

enum class Tone { Good, Caution, Error, Info, Plain, Faint };

// How a PPO run ended, as the run says it
enum class End { None, Finished, Stopped, Failed };

struct RunFacts {
    // The Engine's PPO run (absent: a script's values, or nothing)
    bool ppo = false;
    std::string environment;  // "CartPole-v1"
    int copies = 1;
    long long total_steps = 0;
    double target = NAN;  // mean return of the last 100 episodes it stops at; NaN: none
    bool continuous = false;
    bool running = false;
    bool paused = false;
    std::string waiting;  // why it waits for the GPU; empty: it does not
    End end = End::None;
    std::string end_message;  // the run's own words ("Stopped", the failure)
    bool has_policy = false;  // a trained policy to play or save
    rl_device::RunDevice device;
    std::string compute_path;  // "ArrayFire operations (...)"
    std::string device_note;   // rl_device::DashboardNote
    std::string reference_label;  // "SB3 2.7.1, 3 seeds"; empty: none for these settings
    std::optional<double> reference_now;  // the reference's return at the last update's step
    std::vector<Update> updates;

    // A script's values (no PPO run)
    std::vector<double> script_rewards, script_lengths, script_policy_losses;
    bool script_recent = false;  // a value arrived in the last few seconds
};

struct Row {
    std::string key, value;
};

struct Header {
    State state = State::NoRun;
    std::string pill;  // "Reached its target"
    Tone pill_tone = Tone::Faint;
    std::string beside;  // "Update 121 · 30,976 of 200,000 steps"
    bool show_progress = false;
    double progress = 0;  // 0..1
    Tone progress_tone = Tone::Info;  // Info while it trains, the end's tone after
    std::string run_line;    // "PPO on CartPole-v1 × 8 · stops early at ..."
    std::string where_line;  // "Runs on the CPU (Intel, ArrayFire CPU) · compatibility policy"
    Tone where_tone = Tone::Faint;
    std::string note;  // the device note
    Tone note_tone = Tone::Faint;
    std::string result;  // how it ended, in a sentence
    Tone result_tone = Tone::Plain;
    std::string policy_line;  // what to do with the trained policy
    std::string hint;         // no run: how to start one
    std::vector<Row> details;  // Run details: route, compute path, updates, reference, note
};
Header BuildHeader(const RunFacts& facts);

// One figure card. series: the dot's colour (0 return, 1 length, 2 policy
// loss; -1 the info tone).
struct Card {
    std::string id, label, value, tooltip;
    int series = -1;
    std::string change;  // "17.5 since update 111" (the panel adds the arrow)
    int direction = 0;   // +1 up, -1 down, 0 none
    Tone change_tone = Tone::Good;
    std::string line2, line3;
};
std::vector<Card> BuildCards(const RunFacts& facts);

// The time the updates took (each update's steps at its own speed), which
// leaves out the time a run was paused or waited
double TrainingSeconds(const std::vector<Update>& updates);
// "46.5 s", "23 s" (tenths under a minute, without ".0"), "1 m 55 s", "2 h 5 m"
std::string Duration(double seconds);

// Every update, one row each: steps, the returns, the losses, speed. Empty
// cells where an update has no value.
std::string UpdatesCsv(const std::vector<Update>& updates);

}  // namespace cyxwiz::rl_dashboard
