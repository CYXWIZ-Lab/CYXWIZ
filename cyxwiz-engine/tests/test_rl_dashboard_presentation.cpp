// What the RL Training Dashboard says (TOFIX129 B4-8): the header in plain
// words, the figure cards, the updates CSV. The run: CartPole-v1, PPO x 8 on
// the CPU, with the figures of a real run's updates (policy loss, value loss,
// KL and clip fraction of its update 121) on a shorter, rounder schedule.
#include "../src/core/rl_dashboard_presentation.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int failures = 0;

void Check(bool ok, const std::string& what) {
    if (ok) return;
    std::printf("FAIL: %s\n", what.c_str());
    ++failures;
}

void Same(const std::string& got, const std::string& want, const std::string& what) {
    if (got == want) return;
    std::printf("FAIL: %s\n  got:  %s\n  want: %s\n", what.c_str(), got.c_str(), want.c_str());
    ++failures;
}

using namespace cyxwiz::rl_dashboard;

const std::string kDot = " \xC2\xB7 ";

cyxwiz::rl_device::RunDevice Cpu() {
    cyxwiz::rl_device::RunDevice d;
    d.valid = true;
    d.requested_backend = d.effective_backend = "arrayfire_cpu";
    d.device_name = "Intel";
    d.policy = "compatibility";
    return d;
}

// Twelve updates of 256 steps (8 copies x 32 steps) at 1,280 steps/s
std::vector<Update> Updates() {
    std::vector<Update> updates;
    const double returns[] = {15.25, 23.83, 30, 40, 50, 60, 70, 80, 90, 95, 99, 120.5};
    for (int i = 0; i < 12; ++i) {
        Update u;
        u.update = i + 1;
        u.steps = 256.0 * (i + 1);
        u.mean_return = u.mean_length = returns[i];
        u.policy_loss = -0.0079103484749794;
        u.value_loss = 4.308330062485766e-06;
        u.entropy = 0.6312;
        u.approx_kl = 0.0039694723673164845;
        u.clip_fraction = 0.06015624850988388;
        u.explained_variance = -0.6523266434669495;
        u.steps_per_second = 1280;
        updates.push_back(u);
    }
    return updates;
}

// The facts read the updates where they are (the panel's own list)
RunFacts Training(const std::vector<Update>& updates) {
    RunFacts f;
    f.ppo = true;
    f.environment = "CartPole-v1";
    f.copies = 8;
    f.total_steps = 30720;
    f.target = 100;
    f.running = true;
    f.device = Cpu();
    f.compute_path = "ArrayFire operations (policy: no neural provider runs this network on cpu), optimizers per tensor";
    f.reference_label = "SB3 2.7.1, 3 seeds";
    f.reference_now = 118.31;
    f.updates = updates;
    return f;
}

RunFacts Ended(End end, const std::vector<Update>& updates) {
    RunFacts f = Training(updates);
    f.running = false;
    f.end = end;
    f.has_policy = true;
    return f;
}

const Card* FindCard(const std::vector<Card>& cards, const std::string& id) {
    for (const Card& c : cards)
        if (c.id == id) return &c;
    return nullptr;
}

std::string Detail(const Header& h, const std::string& key) {
    for (const Row& r : h.details)
        if (r.key == key) return r.value;
    return "(none)";
}

void CheckTraining() {
    const std::vector<Update> u = Updates();
    const RunFacts f = Training(u);
    const Header h = BuildHeader(f);
    Check(h.state == State::Training, "training: state");
    Same(h.pill, "Training", "training: pill");
    Check(h.pill_tone == Tone::Good, "training: pill tone");
    Same(h.beside, "Update 12" + kDot + "3,072 of 30,720 steps", "training: beside the pill");
    Check(h.show_progress && h.progress > 0.0999 && h.progress < 0.1001, "training: a tenth of the steps");
    Check(h.progress_tone == Tone::Info, "training: progress tone");
    Same(h.run_line, "PPO on CartPole-v1 \xC3\x97 8" + kDot + "stops early at a mean return of 100 over the last 100 episodes",
         "training: run line");
    Same(h.where_line, "Runs on the CPU (Intel, ArrayFire CPU)" + kDot + "compatibility policy", "training: where");
    Check(h.result.empty() && h.policy_line.empty(), "training: no result yet");
    Same(Detail(h, "Device"), "arrayfire_cpu device 0" + kDot + "Intel" + kDot + "policy Compatibility", "details: device");
    Same(Detail(h, "Computes as"), f.compute_path, "details: compute path");
    Same(Detail(h, "Updates"), "256 steps each (8 \xC3\x97 32 steps)", "details: updates");
    Same(Detail(h, "Reference"), "SB3 2.7.1, 3 seeds, the same settings: under the Mean return chart", "details: reference");
    Same(Detail(h, "Device note"), "(none)", "details: no device note row when there is none");

    const auto cards = BuildCards(f);
    Check(cards.size() == 4, "training: four cards");
    const Card* ret = FindCard(cards, "##kpi_return");
    Check(ret != nullptr, "training: return card");
    if (ret) {
        Same(ret->value, "120.5", "return: value");
        Same(ret->change, "96.7 since update 2", "return: change over 10 updates");
        Check(ret->direction == 1 && ret->change_tone == Tone::Good, "return: up and good");
        Same(ret->line2, "best 120.5" + kDot + "target 100", "return: best and target");
        Same(ret->line3, "SB3 reference here: 118.3", "return: the reference here");
    }
    const Card* time = FindCard(cards, "##kpi_time");
    Check(time != nullptr, "training: time card");
    if (time) {
        Same(time->label, "Time", "time: label while it trains");
        Same(time->value, "2.4 s", "time: the updates' own time");
        Same(time->line2, "1,280 steps/s now, 1,280 on average", "time: speed");
        Same(time->line3, "at most 21.6 s for the other 27,648 steps; less if it reaches 100", "time: what is left");
    }
    const Card* updates = FindCard(cards, "##kpi_updates");
    Check(updates != nullptr, "training: updates card");
    if (updates) {
        Same(updates->value, "12", "updates: count");
        Same(updates->line2, "policy loss -0.0079" + kDot + "value loss 4.3e-06", "updates: losses");
        Same(updates->line3, "approx KL 0.0040" + kDot + "clip fraction 0.060", "updates: KL and clip");
        Same(updates->tooltip, "Entropy 0.6312" + kDot + "explained variance -0.652", "updates: tooltip");
    }
    const Card* length = FindCard(cards, "##kpi_length");
    Check(length && length->value == "120.5" && length->line2 == "mean of the last 100 episodes", "length card");
}

void CheckEnds() {
    {  // reached its target
        std::vector<Update> u = Updates();
        RunFacts f = Ended(End::Finished, u);
        u.back().reached_target = true;
        const Header h = BuildHeader(f);
        Check(h.state == State::ReachedTarget, "target: state");
        Same(h.pill, "Reached its target", "target: pill");
        Same(h.beside, "3,072 of 30,720 steps in 2.4 s", "target: beside");
        Check(h.progress == 1.0 && h.progress_tone == Tone::Info, "target: full bar in the done tone");
        Same(h.where_line, "Ran on the CPU (Intel, ArrayFire CPU)" + kDot + "compatibility policy", "target: where, past");
        Same(h.result,
             "Reached its target: mean return 120.5 over the last 100 episodes (target 100) at update 12; it stopped there.",
             "target: result");
        Check(h.result_tone == Tone::Good, "target: result tone");
        Same(h.policy_line, "Its policy: Play policy shows it, Save policy... keeps it as a .cyxmodel.", "target: policy");
        const auto cards = BuildCards(f);
        const Card* ret = FindCard(cards, "##kpi_return");
        Check(ret && ret->line2 == "best 120.5" + kDot + "target 100 reached", "target: card says reached");
        const Card* time = FindCard(cards, "##kpi_time");
        Check(time && time->label == "Total time", "target: total time");
        if (time) {
            Same(time->line2, "1,280 steps/s on average", "target: average speed");
            Same(time->line3, "12 updates of 256 steps", "target: updates");
        }
    }
    {  // ran all its steps, no target
        std::vector<Update> u = Updates();
        RunFacts f = Ended(End::Finished, u);
        f.total_steps = 3072;
        f.target = NAN;
        const Header h = BuildHeader(f);
        Check(h.state == State::RanAllSteps, "all steps: state");
        Same(h.pill, "Ran all its steps", "all steps: pill");
        Same(h.run_line, "PPO on CartPole-v1 \xC3\x97 8", "all steps: no target in the run line");
        Same(h.result, "Ran all its 3,072 steps: mean return 120.5 over the last 100 episodes.", "all steps: result");
    }
    {  // stopped part way
        std::vector<Update> u = Updates();
        RunFacts f = Ended(End::Stopped, u);
        u.resize(5);
        f.updates = u;
        const Header h = BuildHeader(f);
        Same(h.pill, "Stopped", "stopped: pill");
        Check(h.pill_tone == Tone::Caution && h.progress_tone == Tone::Caution, "stopped: caution");
        Same(h.result, "Stopped at update 5, 1,280 of 30,720 steps.", "stopped: result");
        Check(h.progress > 0.0416 && h.progress < 0.0417, "stopped: the bar stays where it stopped");
    }
    {  // stopped while it waited for the GPU: no update, no device
        std::vector<Update> u = Updates();
        RunFacts f = Ended(End::Stopped, u);
        u.clear();
        f.updates = u;
        f.device = {};
        f.has_policy = false;
        f.end_message = "Stopped while waiting for the GPU";
        const Header h = BuildHeader(f);
        Same(h.result, "Stopped while waiting for the GPU, before its first update.", "stopped waiting: result");
        Check(h.where_line.empty() && h.policy_line.empty(), "stopped waiting: no device, no policy");
        Same(h.beside, "0 of 30,720 steps", "stopped waiting: beside");
        const auto cards = BuildCards(f);
        const Card* time = FindCard(cards, "##kpi_time");
        Check(time && time->line2 == "no update yet", "stopped waiting: no time yet");
    }
    {  // failed
        std::vector<Update> u = Updates();
        RunFacts f = Ended(End::Failed, u);
        f.end_message = "PPO training failed: out of memory";
        const Header h = BuildHeader(f);
        Same(h.pill, "Failed", "failed: pill");
        Same(h.result, "PPO training failed: out of memory", "failed: the run's words");
        Check(h.result_tone == Tone::Error && h.pill_tone == Tone::Error, "failed: error tone");
    }
}

void CheckWhileRunning() {
    {
        std::vector<Update> u = Updates();
        RunFacts f = Training(u);
        f.waiting = "Waiting for the GPU: another training run is using NVIDIA GeForce GTX 1050 Ti";
        const Header h = BuildHeader(f);
        Same(h.pill, "Waiting for the GPU", "waiting: pill");
        Same(h.where_line, f.waiting, "waiting: says why");
        Check(h.where_tone == Tone::Caution, "waiting: caution");
    }
    {
        std::vector<Update> u = Updates();
        RunFacts f = Training(u);
        f.paused = true;
        Same(BuildHeader(f).pill, "Paused", "paused: pill");
    }
    {
        std::vector<Update> u = Updates();
        RunFacts f = Training(u);
        u.clear();
        f.updates = u;
        f.device = {};
        const Header h = BuildHeader(f);
        Same(h.pill, "Starting", "starting: pill");
        Same(h.beside, "0 of 30,720 steps", "starting: beside");
        const auto cards = BuildCards(f);
        const Card* ret = FindCard(cards, "##kpi_return");
        Check(ret && ret->value == "\xE2\x80\x94" && ret->line2 == "no episode finished yet", "starting: no return yet");
    }
    {  // the chosen GPU failed its check and the run fell back to the CPU
        std::vector<Update> u = Updates();
        RunFacts f = Training(u);
        f.device.requested_backend = "arrayfire_cuda";
        f.device.fallback = true;
        f.device.error = "kernel launch failed";
        f.device_note = "Trained on the CPU: the chosen CUDA device failed its check.";
        const Header h = BuildHeader(f);
        Same(h.where_line,
             "Runs on the CPU (Intel, ArrayFire CPU)" + kDot + "fallback: the chosen CUDA device 0 failed its check" + kDot +
                 "compatibility policy",
             "fallback: where");
        Check(h.where_tone == Tone::Caution && h.note_tone == Tone::Caution, "fallback: caution");
        Same(Detail(h, "Device note"), f.device_note, "fallback: note in the details");
    }
    {  // a falling return
        std::vector<Update> u = Updates();
        RunFacts f = Training(u);
        u.back().mean_return = 10;
        const auto cards = BuildCards(f);
        const Card* ret = FindCard(cards, "##kpi_return");
        Check(ret && ret->direction == -1 && ret->change_tone == Tone::Caution && ret->change == "13.8 since update 2",
              "falling return: down and caution");
    }
    {  // no reference for these settings
        std::vector<Update> u = Updates();
        RunFacts f = Training(u);
        f.reference_label.clear();
        f.reference_now.reset();
        Same(Detail(BuildHeader(f), "Reference"), "none recorded for these settings", "no reference: details");
        const auto cards = BuildCards(f);
        const Card* ret = FindCard(cards, "##kpi_return");
        Check(ret && ret->line3.empty(), "no reference: no line");
    }
}

void CheckScriptAndEmpty() {
    RunFacts none;
    const Header h = BuildHeader(none);
    Same(h.pill, "No run yet", "no run: pill");
    Check(!h.hint.empty() && BuildCards(none).empty(), "no run: a hint, no cards");

    const std::vector<double> rewards{10, 20, 15}, lengths{100, 200, 150}, losses{0.5, 0.25};
    RunFacts s;
    s.script_rewards = rewards;
    s.script_lengths = lengths;
    s.script_policy_losses = losses;
    const Header sh = BuildHeader(s);
    Same(sh.pill, "From a script", "script: pill");
    Same(sh.beside, "3 episodes reported", "script: beside");
    Check(!sh.show_progress && sh.details.empty(), "script: no progress, no details");
    s.script_recent = true;
    Same(BuildHeader(s).pill, "Reporting", "script: reporting while values arrive");
    const auto cards = BuildCards(s);
    Check(cards.size() == 3, "script: three cards");
    if (cards.size() == 3) {
        Same(cards[0].value, "15.00", "script: last reward");
        Same(cards[0].line2, "best 20.00" + kDot + "mean 15.00", "script: best and mean");
        Same(cards[1].line2, "mean 150", "script: mean length");
        Same(cards[2].line2, "2 reported", "script: policy losses");
    }

    // A script that reported only NaN: no figure, nothing made up
    const std::vector<double> nan_rewards{NAN, NAN};
    RunFacts n;
    n.script_rewards = nan_rewards;
    const auto nan_cards = BuildCards(n);
    Check(nan_cards.size() == 3 && nan_cards[0].value == "\xE2\x80\x94" && nan_cards[0].line2.empty(),
          "script: NaN rewards show a dash, no best");
}

void CheckTimeAndCsv() {
    Same(Duration(46.5), "46.5 s", "duration: tenths");
    Same(Duration(23.0), "23 s", "duration: no .0");
    Same(Duration(115.06), "1 m 55 s", "duration: minutes");
    Same(Duration(59.96), "1 m 0 s", "duration: rounds into a minute");
    Same(Duration(7500), "2 h 5 m", "duration: hours");

    std::vector<Update> u(2);
    u[0].update = 1;
    u[0].steps = 256;
    u[0].policy_loss = -0.0136;
    u[0].value_loss = 13.5;
    u[0].entropy = 0.69;
    u[0].approx_kl = 0.0071;
    u[0].clip_fraction = 0.1;
    u[0].steps_per_second = 1280;
    u[1] = u[0];
    u[1].update = 2;
    u[1].steps = 512;
    u[1].mean_return = u[1].mean_length = 23.5;
    u[1].explained_variance = 0.25;
    u[1].reached_target = true;
    Same(UpdatesCsv(u),
         "update,steps,mean_return,mean_episode_length,policy_loss,value_loss,entropy,approx_kl,clip_fraction,"
         "explained_variance,action_std,steps_per_second,reached_target\n"
         "1,256,,,-0.0136,13.5,0.69,0.0071,0.1,,,1280,0\n"
         "2,512,23.5,23.5,-0.0136,13.5,0.69,0.0071,0.1,0.25,,1280,1\n",
         "updates CSV");
    Check(TrainingSeconds(u) > 0.3999 && TrainingSeconds(u) < 0.4001, "training seconds: each update's steps at its speed");
}

}  // namespace

int main() {
    CheckTraining();
    CheckEnds();
    CheckWhileRunning();
    CheckScriptAndEmpty();
    CheckTimeAndCsv();
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("RL dashboard presentation passed\n");
    return EXIT_SUCCESS;
}
