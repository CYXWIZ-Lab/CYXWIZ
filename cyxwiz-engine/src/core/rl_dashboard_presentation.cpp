#include "rl_dashboard_presentation.h"

#include <algorithm>
#include <cstdio>

namespace cyxwiz::rl_dashboard {

namespace {

constexpr const char* kDot = " \xC2\xB7 ";
constexpr const char* kTimes = " \xC3\x97 ";
constexpr const char* kDash = "\xE2\x80\x94";

std::string Thousands(double v) {
    const long long n = std::llround(v);
    std::string digits = std::to_string(n < 0 ? -n : n);
    std::string out;
    for (size_t i = 0; i < digits.size(); ++i) {
        if (i > 0 && (digits.size() - i) % 3 == 0) out += ',';
        out += digits[i];
    }
    return n < 0 ? "-" + out : out;
}

std::string Fixed(double v, int decimals) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", decimals, v);
    return buf;
}

// "475", "0.5"
std::string Short(double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%g", v);
    return buf;
}

// A loss: four decimals, or "4.3e-06" when that would read 0.0000
std::string Small(double v) {
    if (v != 0.0 && std::abs(v) < 1e-3) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.1e", v);
        return buf;
    }
    return Fixed(v, 4);
}

bool HasTarget(const RunFacts& f) { return std::isfinite(f.target); }

const Update* Last(const RunFacts& f) { return f.updates.empty() ? nullptr : &f.updates.back(); }

State StateOf(const RunFacts& f) {
    if (!f.ppo) {
        const bool any = !f.script_rewards.empty() || !f.script_lengths.empty() || !f.script_policy_losses.empty();
        return any ? State::FromScript : State::NoRun;
    }
    if (f.running) {
        if (!f.waiting.empty()) return State::WaitingForGpu;
        if (f.paused) return State::Paused;
        if (!f.device.valid && f.updates.empty()) return State::Starting;
        return State::Training;
    }
    switch (f.end) {
    case End::Failed: return State::Failed;
    case End::Stopped: return State::Stopped;
    case End::Finished: {
        const Update* last = Last(f);
        return last && last->reached_target ? State::ReachedTarget : State::RanAllSteps;
    }
    case End::None: break;
    }
    return State::Starting;
}

std::string WhereLine(const rl_device::RunDevice& d, bool ended) {
    const std::string word = rl_device::BackendWord(d.effective_backend);
    std::string where = word == "CPU" ? "the CPU" : word;
    std::string inside = d.device_name;
    inside += (inside.empty() ? "" : ", ") + std::string("ArrayFire ") + word;
    where += " (" + inside + ")";
    if (d.fallback)
        where += kDot + std::string("fallback: the chosen ") + rl_device::BackendWord(d.requested_backend) + " device " +
                 std::to_string(d.requested_device_id) + " failed its check";
    const std::string policy = d.policy.empty() ? "" : kDot + d.policy + " policy";
    return std::string(ended ? "Ran on " : "Runs on ") + where + policy;
}

// The best finite value
std::optional<double> Best(std::span<const double> v) {
    std::optional<double> best;
    for (double x : v)
        if (std::isfinite(x) && (!best || x > *best)) best = x;
    return best;
}

double Mean(std::span<const double> v) {
    double s = 0;
    size_t n = 0;
    for (double x : v)
        if (std::isfinite(x)) {
            s += x;
            ++n;
        }
    return n ? s / static_cast<double>(n) : NAN;
}

}  // namespace

double TrainingSeconds(std::span<const Update> updates) {
    double seconds = 0;
    double before = 0;
    for (const Update& u : updates) {
        const double steps = u.steps - before;
        before = u.steps;
        if (u.steps_per_second > 0 && steps > 0) seconds += steps / u.steps_per_second;
    }
    return seconds;
}

std::string Duration(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0) return kDash;
    if (seconds < 59.95) {
        std::string s = Fixed(seconds, 1);
        if (s.size() > 2 && s.compare(s.size() - 2, 2, ".0") == 0) s.resize(s.size() - 2);
        return s + " s";
    }
    const long long total = std::llround(seconds);
    if (total < 3600) return std::to_string(total / 60) + " m " + std::to_string(total % 60) + " s";
    return std::to_string(total / 3600) + " h " + std::to_string((total % 3600) / 60) + " m";
}

Header BuildHeader(const RunFacts& f) {
    Header h;
    h.state = StateOf(f);
    const Update* last = Last(f);
    const double steps = last ? last->steps : 0.0;
    const std::string of_total = Thousands(steps) + " of " + Thousands(static_cast<double>(f.total_steps)) + " steps";
    const bool ended = h.state == State::ReachedTarget || h.state == State::RanAllSteps ||
                       h.state == State::Stopped || h.state == State::Failed;

    switch (h.state) {
    case State::NoRun:
        h.pill = "No run yet";
        h.hint = "Train an RL graph (Environment, Policy Head, Value Head, PPO) on the canvas, or report metrics from "
                 "a script with pycyxwiz.rl_update_metric. Examples: File > Open, examples/cyxgraph/rl (CartPole needs "
                 "nothing else).";
        return h;
    case State::FromScript: {
        h.pill = f.script_recent ? "Reporting" : "From a script";
        h.pill_tone = f.script_recent ? Tone::Good : Tone::Info;
        const size_t n = f.script_rewards.size();
        h.beside = n == 0 ? std::string("No episodes reported yet")
                          : Thousands(static_cast<double>(n)) + (n == 1 ? " episode reported" : " episodes reported");
        return h;
    }
    case State::Starting: h.pill = "Starting"; h.pill_tone = Tone::Info; break;
    case State::WaitingForGpu: h.pill = "Waiting for the GPU"; h.pill_tone = Tone::Caution; break;
    case State::Training: h.pill = "Training"; h.pill_tone = Tone::Good; break;
    case State::Paused: h.pill = "Paused"; h.pill_tone = Tone::Caution; break;
    case State::ReachedTarget: h.pill = "Reached its target"; h.pill_tone = Tone::Info; break;
    case State::RanAllSteps: h.pill = "Ran all its steps"; h.pill_tone = Tone::Info; break;
    case State::Stopped: h.pill = "Stopped"; h.pill_tone = Tone::Caution; break;
    case State::Failed: h.pill = "Failed"; h.pill_tone = Tone::Error; break;
    }

    // Beside the pill: the update and steps; after the run, its time
    if (ended) {
        h.beside = of_total;
        if (last) h.beside += " in " + Duration(TrainingSeconds(f.updates));
    } else if (last) {
        h.beside = "Update " + Thousands(last->update) + kDot + of_total;
    } else {
        h.beside = of_total;
    }

    h.show_progress = true;
    const double fraction = f.total_steps > 0 ? std::clamp(steps / static_cast<double>(f.total_steps), 0.0, 1.0) : 0.0;
    h.progress = h.state == State::ReachedTarget || h.state == State::RanAllSteps ? 1.0 : fraction;
    h.progress_tone = ended ? h.pill_tone : Tone::Info;

    h.run_line = "PPO on " + f.environment + kTimes + std::to_string(f.copies);
    if (HasTarget(f)) h.run_line += kDot + std::string("stops early at a mean return of ") + Short(f.target) +
                                    " over the last 100 episodes";

    if (h.state == State::WaitingForGpu) {
        h.where_line = f.waiting;
        h.where_tone = Tone::Caution;
    } else if (f.device.valid) {
        h.where_line = WhereLine(f.device, ended);
        h.where_tone = f.device.fallback ? Tone::Caution : Tone::Faint;
    }
    h.note = f.device_note;
    h.note_tone = f.device.fallback ? Tone::Caution : Tone::Faint;

    // How it ended
    const auto ret = [&]() { return last && std::isfinite(last->mean_return) ? Fixed(last->mean_return, 1) : std::string(); };
    switch (h.state) {
    case State::ReachedTarget:
        h.result = "Reached its target: mean return " + ret() + " over the last 100 episodes (target " + Short(f.target) +
                   ") at update " + Thousands(last->update) + "; it stopped there.";
        h.result_tone = Tone::Good;
        break;
    case State::RanAllSteps:
        h.result = "Ran all its " + Thousands(static_cast<double>(f.total_steps)) + " steps";
        if (!ret().empty()) h.result += ": mean return " + ret() + " over the last 100 episodes";
        if (HasTarget(f)) h.result += " (target " + Short(f.target) + " not reached)";
        h.result += ".";
        h.result_tone = Tone::Plain;
        break;
    case State::Stopped:
        h.result = last ? "Stopped at update " + Thousands(last->update) + ", " + of_total + "."
                        : (f.end_message.empty() ? std::string("Stopped") : f.end_message) + ", before its first update.";
        h.result_tone = Tone::Caution;
        break;
    case State::Failed:
        h.result = f.end_message.empty() ? std::string("The run failed.") : f.end_message;
        h.result_tone = Tone::Error;
        break;
    default: break;
    }
    if (ended && f.has_policy) h.policy_line = "Its policy: Play policy shows it, Save policy... keeps it as a .cyxmodel.";

    // Run details: the codes behind the lines above
    if (f.device.valid) h.details.push_back({"Device", rl_device::DeviceRoute(f.device)});
    if (!f.compute_path.empty()) h.details.push_back({"Computes as", f.compute_path});
    if (last && last->steps > 0) {
        const long long per_update = std::llround(f.updates.front().steps);
        std::string each = Thousands(static_cast<double>(per_update)) + " steps each";
        if (f.copies > 1 && per_update % f.copies == 0)
            each += " (" + std::to_string(f.copies) + kTimes + Thousands(static_cast<double>(per_update / f.copies)) + " steps)";
        h.details.push_back({"Updates", each});
    }
    h.details.push_back({"Reference", f.reference_label.empty()
                                          ? std::string("none recorded for these settings")
                                          : f.reference_label + ", the same settings: under the Mean return chart"});
    if (!f.device_note.empty()) h.details.push_back({"Device note", f.device_note});
    return h;
}

std::vector<Card> BuildCards(const RunFacts& f) {
    std::vector<Card> cards;
    const State state = StateOf(f);
    if (state == State::NoRun) return cards;

    if (!f.ppo) {  // a script's values, as reported
        Card reward{"##kpi_reward", "Episode reward", kDash};
        reward.series = 0;
        if (!f.script_rewards.empty() && std::isfinite(f.script_rewards.back()))
            reward.value = Fixed(f.script_rewards.back(), 2);
        if (const auto best = Best(f.script_rewards))
            reward.line2 = "best " + Fixed(*best, 2) + kDot + "mean " + Fixed(Mean(f.script_rewards), 2);
        Card length{"##kpi_length", "Episode length", kDash};
        length.series = 1;
        if (!f.script_lengths.empty() && std::isfinite(f.script_lengths.back()))
            length.value = Fixed(f.script_lengths.back(), 0);
        if (std::isfinite(Mean(f.script_lengths))) length.line2 = "mean " + Fixed(Mean(f.script_lengths), 0);
        Card policy{"##kpi_policy", "Policy loss", kDash};
        policy.series = 2;
        if (!f.script_policy_losses.empty()) {
            if (std::isfinite(f.script_policy_losses.back())) policy.value = Fixed(f.script_policy_losses.back(), 4);
            policy.line2 = Thousands(static_cast<double>(f.script_policy_losses.size())) + " reported";
        }
        cards = {reward, length, policy};
        return cards;
    }

    const Update* last = Last(f);
    const bool ended = state == State::ReachedTarget || state == State::RanAllSteps || state == State::Stopped ||
                       state == State::Failed;

    // Mean return: the change over the last 10 updates, best, target, the reference here
    Card ret{"##kpi_return", "Mean return", kDash};
    ret.series = 0;
    ret.tooltip = "The mean return of the last 100 episodes, at the last update.";
    std::optional<double> best;
    for (const Update& u : f.updates)
        if (std::isfinite(u.mean_return) && (!best || u.mean_return > *best)) best = u.mean_return;
    if (last && std::isfinite(last->mean_return)) {
        ret.value = Fixed(last->mean_return, 1);
        const size_t i = f.updates.size() - 1;
        const size_t back = i >= 10 ? i - 10 : 0;
        if (back < i && std::isfinite(f.updates[back].mean_return)) {
            const double change = last->mean_return - f.updates[back].mean_return;
            ret.change = Fixed(std::abs(change), 1) + " since update " + Thousands(f.updates[back].update);
            ret.direction = change > 0 ? 1 : change < 0 ? -1 : 0;
            ret.change_tone = change >= 0 ? Tone::Good : Tone::Caution;
        }
        ret.line2 = "best " + Fixed(best.value_or(last->mean_return), 1);
        if (HasTarget(f))
            ret.line2 += kDot + std::string("target ") + Short(f.target) + (state == State::ReachedTarget ? " reached" : "");
        if (f.reference_now && std::isfinite(*f.reference_now))
            ret.line3 = "SB3 reference here: " + Fixed(*f.reference_now, 1);
    } else {
        ret.line2 = "no episode finished yet";
    }

    Card length{"##kpi_length", "Episode length", kDash};
    length.series = 1;
    length.line2 = "mean of the last 100 episodes";
    if (last && std::isfinite(last->mean_length)) length.value = Fixed(last->mean_length, 1);

    // Time: the updates' own time (not the time paused or waiting)
    Card time{"##kpi_time", ended ? "Total time" : "Time", kDash};
    time.tooltip = "The time the updates took; time paused or waiting for the GPU is left out.";
    if (last) {
        const double seconds = TrainingSeconds(f.updates);
        time.value = Duration(seconds);
        const double average = seconds > 0 ? last->steps / seconds : 0.0;
        if (ended) {
            time.line2 = Thousands(average) + " steps/s on average";
            time.line3 = Thousands(static_cast<double>(f.updates.size())) + " updates of " +
                         Thousands(f.updates.front().steps) + " steps";
        } else {
            time.line2 = Thousands(last->steps_per_second) + " steps/s now, " + Thousands(average) + " on average";
            const double left = static_cast<double>(f.total_steps) - last->steps;
            if (left > 0 && last->steps_per_second > 0) {
                time.line3 = "at most " + Duration(left / last->steps_per_second) + " for the other " + Thousands(left) +
                             " steps";
                if (HasTarget(f)) time.line3 += "; less if it reaches " + Short(f.target);
            }
        }
    } else {
        time.line2 = "no update yet";
    }

    Card updates{"##kpi_updates", "Updates", kDash};
    updates.series = 2;
    if (last) {
        updates.value = Thousands(static_cast<double>(f.updates.size()));
        updates.line2 = "policy loss " + Small(last->policy_loss) + kDot + "value loss " + Small(last->value_loss);
        updates.line3 = "approx KL " + Fixed(last->approx_kl, 4) + kDot + "clip fraction " + Fixed(last->clip_fraction, 3);
        updates.tooltip = "Entropy " + Fixed(last->entropy, 4);
        if (std::isfinite(last->explained_variance))
            updates.tooltip += kDot + std::string("explained variance ") + Fixed(last->explained_variance, 3);
        if (std::isfinite(last->action_std)) updates.tooltip += kDot + std::string("action std ") + Fixed(last->action_std, 3);
    } else {
        updates.line2 = "no update yet";
    }
    cards = {ret, length, time, updates};
    return cards;
}

std::string UpdatesCsv(std::span<const Update> updates) {
    std::string out =
        "update,steps,mean_return,mean_episode_length,policy_loss,value_loss,entropy,approx_kl,clip_fraction,"
        "explained_variance,action_std,steps_per_second,reached_target\n";
    const auto cell = [&out](double v) {
        out += ',';
        if (!std::isfinite(v)) return;
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.8g", v);
        out += buf;
    };
    for (const Update& u : updates) {
        out += std::to_string(u.update);
        cell(u.steps);
        cell(u.mean_return);
        cell(u.mean_length);
        cell(u.policy_loss);
        cell(u.value_loss);
        cell(u.entropy);
        cell(u.approx_kl);
        cell(u.clip_fraction);
        cell(u.explained_variance);
        cell(u.action_std);
        cell(u.steps_per_second);
        out += u.reached_target ? ",1\n" : ",0\n";
    }
    return out;
}

}  // namespace cyxwiz::rl_dashboard
