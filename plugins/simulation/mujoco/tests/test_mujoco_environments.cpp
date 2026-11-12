// The MuJoCo plugin's environments for reinforcement learning (src/env):
// what they describe, how settings are read, same-step autoreset, the same
// runs for one seed on any number of threads, steps a second, and showing one
// copy in the viewport (ShowCopy publishes its physics state).
//
//   test_mujoco_environments [--trajectory <file.json>]
//
// With --trajectory it also writes, for each task, a start state, actions
// and what the task gave back, for tests/mujoco_gymnasium_parity.py to
// replay in Gymnasium.

#include "env/mj_env_batch.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

int failures = 0;

void Check(bool ok, const std::string& what) {
    if (ok) return;
    std::printf("FAIL: %s\n", what.c_str());
    ++failures;
}

using namespace cyxwiz::plugin;
using namespace cyxwiz::plugin::mujoco;

std::string ModelPath(const MjTaskSpec& spec) { return std::string(CYXWIZ_MUJOCO_ASSETS_DIR) + "/" + spec.model_file; }

struct Rollout {
    std::vector<float> obs, rewards, final_obs;
    std::vector<uint8_t> terminated, truncated;
};

// Actions in the middle half of each range, from a fixed stream.
std::vector<float> Actions(const PluginEnvironmentInfo& info, int count, MjRandom& random) {
    std::vector<float> actions(static_cast<size_t>(count) * info.action_size);
    for (size_t i = 0; i < actions.size(); ++i) {
        const size_t a = i % info.action_size;
        actions[i] = static_cast<float>(0.5 * random.Uniform(info.action_low[a], info.action_high[a]));
    }
    return actions;
}

// Every value a batch gives over `steps` steps.
Rollout Run(const MjTaskSpec& spec, const PluginEnvironmentInfo& info, std::map<std::string, std::string> settings,
            int count, uint64_t seed, int steps) {
    Rollout all;
    std::string error;
    IEnvironmentBatch* batch = MjEnvBatch::Create(spec, ModelPath(spec), settings, count, nullptr, error);
    Check(batch != nullptr, spec.id + " batch: " + error);
    if (!batch) return all;
    const size_t obs_values = static_cast<size_t>(count) * info.observation_size;
    std::vector<float> obs(obs_values), final_obs(obs_values), rewards(count);
    std::vector<uint8_t> terminated(count), truncated(count);
    batch->Reset(seed, obs.data());
    all.obs.insert(all.obs.end(), obs.begin(), obs.end());
    MjRandom random(99);
    const PluginEnvironmentStepBuffers out{obs.data(), rewards.data(), terminated.data(), truncated.data(),
                                           final_obs.data()};
    for (int s = 0; s < steps; ++s) {
        const auto actions = Actions(info, count, random);
        std::fill(final_obs.begin(), final_obs.end(), 0.0f);
        Check(batch->Step(actions.data(), out), spec.id + " step: " + batch->LastError());
        all.obs.insert(all.obs.end(), obs.begin(), obs.end());
        all.final_obs.insert(all.final_obs.end(), final_obs.begin(), final_obs.end());
        all.rewards.insert(all.rewards.end(), rewards.begin(), rewards.end());
        all.terminated.insert(all.terminated.end(), terminated.begin(), terminated.end());
        all.truncated.insert(all.truncated.end(), truncated.begin(), truncated.end());
    }
    batch->Release();
    return all;
}

bool Same(const Rollout& a, const Rollout& b) {
    return a.obs == b.obs && a.rewards == b.rewards && a.final_obs == b.final_obs && a.terminated == b.terminated &&
           a.truncated == b.truncated;
}

PluginEnvironmentInfo Describe(const MjTaskSpec& spec) {
    PluginEnvironmentInfo info;
    std::string error;
    Check(DescribeMjTask(spec, ModelPath(spec), info, error), spec.id + " describes itself: " + error);
    return info;
}

void CheckDescriptions() {
    const auto pendulum = Describe(*FindMjTask("InvertedPendulum-v5"));
    Check(pendulum.observation_size == 4 && pendulum.action_size == 1 && !pendulum.discrete_actions,
          "InvertedPendulum-v5 observes 4 values and takes 1 action");
    Check(pendulum.action_low == std::vector<float>{-3.0f} && pendulum.action_high == std::vector<float>{3.0f},
          "the pendulum's force is in [-3, 3]");
    Check(std::abs(pendulum.step_seconds - 0.04) < 1e-9 && pendulum.max_episode_steps == 1000,
          "a pendulum step is 0.04 s (2 x 0.02 s), 1000 steps an episode");
    const auto cheetah = Describe(*FindMjTask("HalfCheetah-v5"));
    Check(cheetah.observation_size == 17 && cheetah.action_size == 6, "HalfCheetah-v5 observes 17 and takes 6");
    Check(std::abs(cheetah.step_seconds - 0.05) < 1e-9, "a cheetah step is 0.05 s (5 x 0.01 s)");
    bool has_threads = false;
    for (const auto& setting : cheetah.settings) has_threads = has_threads || setting.key == "threads";
    Check(cheetah.settings.size() == 6 && has_threads, "the cheetah's four settings and the two of every batch");
}

void CheckSettings() {
    const auto& spec = *FindMjTask("InvertedPendulum-v5");
    std::string error;
    const auto refused = [&](std::map<std::string, std::string> settings, const std::string& expected) {
        error.clear();
        IEnvironmentBatch* batch = MjEnvBatch::Create(spec, ModelPath(spec), settings, 1, nullptr, error);
        if (batch) batch->Release();
        return !batch && error == expected;
    };
    Check(refused({{"speed", "2"}}, "Inverted Pendulum has no setting 'speed'."), "an unknown setting is refused");
    Check(refused({{"frame_skip", "fast"}}, "Frame skip must be a number (got 'fast')."), "a word is not a number");
    Check(refused({{"frame_skip", "0"}}, "Frame skip must be at least 1."), "below the minimum");
    Check(refused({{"frame_skip", "1.5"}}, "Frame skip must be a whole number (got '1.5')."), "not whole");
    Check(refused({}, "") == false, "the defaults are fine");
    error.clear();
    Check(!MjEnvBatch::Create(spec, "missing.xml", {}, 1, nullptr, error) && error.find("did not load") != std::string::npos,
          "a model that does not load is an error");
}

void CheckAutoreset() {
    const auto& spec = *FindMjTask("InvertedPendulum-v5");
    const auto info = Describe(spec);
    std::string error;
    IEnvironmentBatch* batch = MjEnvBatch::Create(spec, ModelPath(spec), {{"max_episode_steps", "5"}}, 2, nullptr, error);
    Check(batch != nullptr, "pendulum batch: " + error);
    if (!batch) return;
    std::vector<float> obs(8), final_obs(8, 0.0f), rewards(2);
    std::vector<uint8_t> terminated(2), truncated(2);
    const PluginEnvironmentStepBuffers out{obs.data(), rewards.data(), terminated.data(), truncated.data(),
                                           final_obs.data()};
    Check(!batch->Step(obs.data(), out) && batch->LastError() == "Reset the environments before stepping them.",
          "a step before a reset is an error");
    batch->Reset(3, obs.data());
    Check(std::abs(obs[1]) <= 0.01f && std::abs(obs[5]) <= 0.01f, "each copy starts near upright");
    // Copy 0 is pushed hard until its pole falls; copy 1 is left alone
    bool fell = false;
    int cut = 0;
    for (int s = 0; s < 5; ++s) {
        const float actions[2] = {fell ? 0.0f : 3.0f, 0.0f};
        Check(batch->Step(actions, out), "step: " + batch->LastError());
        if (terminated[0] && !fell) {
            fell = true;
            Check(std::abs(final_obs[1]) > 0.2f && rewards[0] == 0.0f,
                  "the fallen pole's last angle is past 0.2 rad, no reward");
            Check(std::abs(obs[1]) <= 0.01f, "the fallen copy starts again in the same step");
        }
        cut += truncated[1];
        Check(terminated[1] == 0 && (s == 4) == (truncated[1] == 1), "copy 1 runs until the step limit");
    }
    Check(fell, "a hard push topples the pole within five steps");
    Check(cut == 1, "the copy left alone is cut once, at the five-step limit");
    const float bad[2] = {0.0f, std::nanf("")};
    Check(!batch->Step(bad, out) && batch->LastError() == "Copy 1 was given an action that is not a number.",
          "an action that is not a number is refused");
    batch->Release();
    (void)info;
}

void CheckDeterminism() {
    for (const char* id : {"InvertedPendulum-v5", "HalfCheetah-v5"}) {
        const auto& spec = *FindMjTask(id);
        const auto info = Describe(spec);
        const auto one = Run(spec, info, {{"threads", "1"}}, 8, 42, 300);
        const auto four = Run(spec, info, {{"threads", "4"}}, 8, 42, 300);
        Check(Same(one, four), std::string(id) + ": seed 42 gives the same runs on one thread and on four");
        const auto again = Run(spec, info, {{"threads", "3"}}, 8, 42, 300);
        Check(Same(one, again), std::string(id) + ": and on three, run again");
        const auto other = Run(spec, info, {{"threads", "1"}}, 8, 43, 1);
        Check(std::vector<float>(other.obs.begin(), other.obs.begin() + info.observation_size) !=
                  std::vector<float>(one.obs.begin(), one.obs.begin() + info.observation_size),
              std::string(id) + ": another seed starts elsewhere");
        Check(std::vector<float>(one.obs.begin(), one.obs.begin() + info.observation_size) !=
                  std::vector<float>(one.obs.begin() + info.observation_size,
                                     one.obs.begin() + 2 * info.observation_size),
              std::string(id) + ": two copies of one seed start apart");
    }
}

void ReportSpeed() {
    for (const char* id : {"InvertedPendulum-v5", "HalfCheetah-v5"}) {
        const auto& spec = *FindMjTask(id);
        const auto info = Describe(spec);
        for (const int count : {1, 8, 64}) {
            std::string error;
            IEnvironmentBatch* batch = MjEnvBatch::Create(spec, ModelPath(spec), {}, count, nullptr, error);
            if (!batch) continue;
            const size_t obs_values = static_cast<size_t>(count) * info.observation_size;
            std::vector<float> obs(obs_values), final_obs(obs_values), rewards(count);
            std::vector<uint8_t> terminated(count), truncated(count);
            const PluginEnvironmentStepBuffers out{obs.data(), rewards.data(), terminated.data(), truncated.data(),
                                                   final_obs.data()};
            batch->Reset(1, obs.data());
            MjRandom random(5);
            const auto actions = Actions(info, count, random);
            const int steps = count == 1 ? 4000 : (count == 8 ? 1000 : 200);
            const auto start = std::chrono::steady_clock::now();
            for (int s = 0; s < steps; ++s) batch->Step(actions.data(), out);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::printf("speed %-20s copies %3d threads %2d: %9.0f env steps/s\n", id, count,
                        static_cast<MjEnvBatch*>(batch)->Threads(), steps * count / seconds);
            batch->Release();
        }
    }
}

// A start state, the actions and what came back, for each task (one copy,
// no autoreset inside the window).
void WriteTrajectories(const std::string& path) {
    std::ofstream file(path);
    file.precision(17);
    file << "{\n";
    bool first_task = true;
    for (const char* id : {"InvertedPendulum-v5", "HalfCheetah-v5"}) {
        const auto& spec = *FindMjTask(id);
        const auto info = Describe(spec);
        std::string error;
        mjModel* m = mj_loadXML(ModelPath(spec).c_str(), nullptr, nullptr, 0);
        mjData* d = mj_makeData(m);
        MjTaskValues values;
        ReadMjTaskValues(spec, {}, values, error);
        const auto task = spec.create(values);
        MjRandom start(11);
        mj_resetData(m, d);
        task->ResetState(m, d, start);
        file << (first_task ? "" : ",\n") << "  \"" << id << "\": {\n    \"qpos\": [";
        first_task = false;
        for (int i = 0; i < m->nq; ++i) file << (i ? ", " : "") << d->qpos[i];
        file << "],\n    \"qvel\": [";
        for (int i = 0; i < m->nv; ++i) file << (i ? ", " : "") << d->qvel[i];
        file << "],\n    \"steps\": [";
        MjRandom random(17);
        std::vector<float> obs(info.observation_size);
        const int steps = std::string(id) == "InvertedPendulum-v5" ? 20 : 200;
        for (int s = 0; s < steps; ++s) {
            const auto action = Actions(info, 1, random);
            for (int a = 0; a < m->nu; ++a) d->ctrl[a] = action[a];
            const MjTaskStep step = task->Step(m, d, action.data());
            task->Observe(m, d, obs.data());
            file << (s ? ",\n      " : "\n      ") << "{\"action\": [";
            for (size_t a = 0; a < action.size(); ++a) file << (a ? ", " : "") << action[a];
            file << "], \"obs\": [";
            for (size_t o = 0; o < obs.size(); ++o) file << (o ? ", " : "") << obs[o];
            file << "], \"reward\": " << step.reward << ", \"terminated\": " << (step.terminated ? "true" : "false")
                 << "}";
            if (step.terminated) break;
        }
        file << "]\n  }";
        mj_deleteData(d);
        mj_deleteModel(m);
    }
    file << "\n}\n";
}

}  // namespace

// ShowCopy: the shown copy's physics state reaches the view slot after Reset
// and every Step; out-of-range copies are refused; another batch can take the
// view; hiding or releasing frees it; a batch without a view has no picture.
void CheckShowCopy() {
    const MjTaskSpec& spec = *FindMjTask("InvertedPendulum-v5");
    std::string error;
    auto view = std::make_shared<MjViewSlot>();
    IEnvironmentBatch* batch = MjEnvBatch::Create(spec, ModelPath(spec), {}, 2, view, error);
    Check(batch != nullptr, "a batch with a view: " + error);
    if (!batch) return;
    Check(!batch->ShowCopy(2), "copy 2 of 2 copies is refused");
    Check(batch->ShowCopy(1), "copy 1 is shown");
    Check(view->owner == batch && view->copy == 1 && view->model_path == ModelPath(spec) && !view->title.empty(),
          "the view names the batch, the copy, the model file and the task");
    std::vector<float> obs(2 * 4), final_obs(2 * 4), rewards(2), actions(2, 0.5f);
    std::vector<uint8_t> terminated(2), truncated(2);
    batch->Reset(3, obs.data());
    mjModel* model = mj_loadXML(ModelPath(spec).c_str(), nullptr, nullptr, 0);
    const size_t size = static_cast<size_t>(mj_stateSize(model, mjSTATE_FULLPHYSICS));
    mj_deleteModel(model);
    Check(view->state.size() == size, "Reset publishes the copy's full physics state");
    const uint64_t after_reset = view->version;
    const auto before = view->state;
    const PluginEnvironmentStepBuffers out{obs.data(), rewards.data(), terminated.data(), truncated.data(),
                                           final_obs.data()};
    for (int s = 0; s < 5; ++s) batch->Step(actions.data(), out);
    Check(view->version == after_reset + 5 && view->state != before, "every Step publishes the moved state");

    IEnvironmentBatch* other = MjEnvBatch::Create(spec, ModelPath(spec), {}, 1, view, error);
    Check(other && other->ShowCopy(0) && view->owner == other, "another batch takes the view");
    const uint64_t taken = view->version;
    batch->Step(actions.data(), out);
    Check(view->version == taken && view->owner == other, "the first batch no longer publishes");
    if (other) other->Release();
    Check(view->owner == nullptr, "releasing the shown batch frees the view");
    Check(batch->ShowCopy(0) && batch->ShowCopy(-1) && view->owner == nullptr, "ShowCopy(-1) hides it");
    batch->Release();

    IEnvironmentBatch* plain = MjEnvBatch::Create(spec, ModelPath(spec), {}, 1, nullptr, error);
    Check(plain && !plain->ShowCopy(0), "a batch without a view has no picture");
    if (plain) plain->Release();
}

// The shown copy's live numbers (the Viewport's overlay during playback):
// episode, step, the episode's return so far, the last action, and the
// return of the episode before once one ends.
void CheckLiveNumbers() {
    const MjTaskSpec& spec = *FindMjTask("InvertedPendulum-v5");
    std::string error;
    auto view = std::make_shared<MjViewSlot>();
    IEnvironmentBatch* batch = MjEnvBatch::Create(spec, ModelPath(spec), {{"max_episode_steps", "5"}}, 2, view, error);
    Check(batch != nullptr, "a batch with a view: " + error);
    if (!batch) return;
    Check(batch->ShowCopy(1), "copy 1 is shown");
    Check(std::abs(view->step_seconds - 0.04) < 1e-9, "the view knows a step is 0.04 s (2 x 0.02 s)");
    std::vector<float> obs(2 * 4), final_obs(2 * 4), rewards(2);
    std::vector<uint8_t> terminated(2), truncated(2);
    const PluginEnvironmentStepBuffers out{obs.data(), rewards.data(), terminated.data(), truncated.data(),
                                           final_obs.data()};
    batch->Reset(3, obs.data());
    Check(view->episode == 1 && view->step == 0 && view->max_steps == 5 && view->action.empty() &&
              view->episode_return == 0.0 && !view->has_last_return,
          "after Reset: episode 1, step 0 of 5, no action yet");
    double sum = 0.0;
    for (int s = 0; s < 4; ++s) {
        const float actions[2] = {0.0f, 0.1f * static_cast<float>(s + 1)};
        batch->Step(actions, out);
        sum += rewards[1];
    }
    Check(view->episode == 1 && view->step == 4 && std::abs(view->episode_return - sum) < 1e-9,
          "four steps in: step 4 and the four rewards of copy 1 summed");
    Check(view->action.size() == 1 && std::abs(view->action[0] - 0.4f) < 1e-6f, "the last action of copy 1");
    const float last[2] = {0.0f, 0.0f};
    batch->Step(last, out);  // the fifth step ends the episode (the step limit)
    sum += rewards[1];
    Check(truncated[1] == 1, "the fifth step is cut at the limit");
    Check(view->episode == 2 && view->step == 0 && view->episode_return == 0.0 && view->has_last_return &&
              std::abs(view->last_return - sum) < 1e-9,
          "the next episode starts at step 0; the ended one's return is kept");
    batch->Release();
}

int main(int argc, char** argv) {
    std::string trajectory;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--trajectory") trajectory = argv[i + 1];
    }
    CheckDescriptions();
    CheckSettings();
    CheckAutoreset();
    CheckDeterminism();
    CheckShowCopy();
    CheckLiveNumbers();
    ReportSpeed();
    if (!trajectory.empty()) WriteTrajectories(trajectory);
    if (failures == 0) std::printf("test_mujoco_environments: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
