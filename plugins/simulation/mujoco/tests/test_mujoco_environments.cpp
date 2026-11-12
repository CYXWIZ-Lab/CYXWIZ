// The MuJoCo plugin's environments for reinforcement learning (src/env):
// what they describe, how settings are read, same-step autoreset, the same
// runs for one seed on any number of threads, steps a second, and showing one
// copy in the viewport (ShowCopy publishes its physics state), and the
// Viewport's Stress panel: its scenes and its requests to the shown copy.
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
#include <map>
#include <memory>
#include <mutex>
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

// The Stress panel's scene (env/mj_scene.h): an empty scene is the model file
// as it is; a scene's objects come after the robot, which keeps its masses.
void CheckScenes() {
    for (const auto& spec : MjTasks()) {
        const std::string path = ModelPath(spec);
        char load_error[1000] = "";
        mjModel* file = mj_loadXML(path.c_str(), nullptr, load_error, sizeof(load_error));
        MjRobotSize robot;
        std::string error;
        mjModel* built = CompileMjScene(path, MjScene{}, robot, error);
        Check(file && built, spec.id + ": the model loads both ways: " + error);
        if (!file || !built) continue;
        Check(robot.nq == file->nq && robot.nv == file->nv && robot.nbody == file->nbody,
              spec.id + ": with no scene the robot is the whole file");
        Check(built->nq == file->nq && built->nv == file->nv && built->ngeom == file->ngeom &&
                  built->nbody == file->nbody,
              spec.id + ": an empty scene adds nothing");
        mjData* a = mj_makeData(file);
        mjData* b = mj_makeData(built);
        MjRandom random(5);
        bool same = true;
        for (int s = 0; s < 200; ++s) {
            for (int u = 0; u < file->nu; ++u) a->ctrl[u] = b->ctrl[u] = random.Uniform(-1.0, 1.0);
            mj_step(file, a);
            mj_step(built, b);
            for (int i = 0; i < file->nq; ++i) same = same && a->qpos[i] == b->qpos[i];
        }
        Check(same, spec.id + ": an empty scene steps exactly as the model file, value for value");
        mj_deleteData(a);
        mj_deleteData(b);
        mj_deleteModel(file);
        mj_deleteModel(built);
    }

    const MjTaskSpec& spec = *FindMjTask("HalfCheetah-v5");
    const std::string path = ModelPath(spec);
    char load_error[1000] = "";
    mjModel* plain = mj_loadXML(path.c_str(), nullptr, load_error, sizeof(load_error));
    MjScene scene;
    scene.ground = MjGround{MjGroundKind::Rough, 0.5, 3};
    scene.objects = {MjThrownObject{true, 0.2, 2.0}, MjThrownObject{false, 0.3, 5.0}};
    MjRobotSize robot;
    std::string error;
    mjModel* m = CompileMjScene(path, scene, robot, error);
    Check(plain && m, "HalfCheetah with Rough ground and two objects builds: " + error);
    if (plain && m) {
        Check(robot.nq == plain->nq && robot.nv == plain->nv && robot.nbody == plain->nbody,
              "the robot's slice is the model file's");
        Check(m->nq == plain->nq + 14 && m->nv == plain->nv + 12 && m->nbody == plain->nbody + 2,
              "two free objects come after the robot");
        bool masses = true;
        for (int body = 1; body < plain->nbody; ++body)
            masses = masses && std::abs(m->body_mass[body] - plain->body_mass[body]) < 1e-12;
        Check(masses, "the robot keeps the masses of its file (the file rescales to 14 kg)");
        Check(std::abs(m->body_mass[plain->nbody] - 2.0) < 1e-12 && std::abs(m->body_mass[plain->nbody + 1] - 5.0) < 1e-12,
              "each object has the mass asked for");
        Check(m->nhfield == 1 && mj_name2id(m, mjOBJ_GEOM, "cyx_stress_ground") >= 0, "Rough is one heightfield");
        Check(m->stat.extent == plain->stat.extent && m->stat.meaninertia == plain->stat.meaninertia &&
                  m->stat.center[0] == plain->stat.center[0],
              "the statistics stay the robot's (the view's near plane, the solver's scale)");
        mj_deleteModel(m);
    }
    MjScene stairs;
    stairs.ground = MjGround{MjGroundKind::Steps, 1.0, 5};
    mjModel* s = CompileMjScene(path, stairs, robot, error);
    Check(s && plain && s->ngeom == plain->ngeom + static_cast<int>(GroundBoxes(stairs.ground).size()) && s->nq == plain->nq,
          "Steps: one box a step, no state added");
    if (s) mj_deleteModel(s);
    if (plain) mj_deleteModel(plain);
}

// A HalfCheetah playback batch of one copy, shown, reset and stepped once
struct StressedCheetah {
    std::shared_ptr<MjViewSlot> view = std::make_shared<MjViewSlot>();
    IEnvironmentBatch* batch = nullptr;
    std::vector<float> obs = std::vector<float>(17), final_obs = std::vector<float>(17);
    float reward = 0.0f;
    uint8_t terminated = 0, truncated = 0;

    explicit StressedCheetah(std::map<std::string, std::string> settings = {}) {
        std::string error;
        batch = MjEnvBatch::Create(*FindMjTask("HalfCheetah-v5"), ModelPath(*FindMjTask("HalfCheetah-v5")), settings, 1,
                                   view, error);
        Check(batch != nullptr, "a HalfCheetah batch: " + error);
        if (!batch) return;
        batch->ShowCopy(0);
        batch->Reset(11, obs.data());
        Step();
    }
    ~StressedCheetah() {
        if (batch) batch->Release();
    }
    void Ask(const MjStressRequest& request) {
        std::lock_guard<std::mutex> lock(view->mutex);
        view->requests.push_back(request);
    }
    void Step() {
        const std::vector<float> still(6, 0.0f);
        const PluginEnvironmentStepBuffers out{obs.data(), &reward, &terminated, &truncated, final_obs.data()};
        Check(batch->Step(still.data(), out), "step: " + batch->LastError());
    }
    // The shown copy's state in a model built from the view's scene
    double BodyX(int body_from_robot_end) {
        MjRobotSize robot;
        std::string error;
        mjModel* m = CompileMjScene(view->model_path, view->scene, robot, error);
        if (!m) return 0.0;
        mjData* d = mj_makeData(m);
        double x = 0.0;
        if (static_cast<size_t>(mj_stateSize(m, mjSTATE_FULLPHYSICS)) == view->state.size()) {
            mj_setState(m, d, view->state.data(), mjSTATE_FULLPHYSICS);
            mj_forward(m, d);
            const int body = body_from_robot_end < 0 ? 1 : robot.nbody + body_from_robot_end;
            x = d->xpos[3 * body];
        }
        mj_deleteData(d);
        mj_deleteModel(m);
        return x;
    }
};

// The Stress panel's requests reach the copy a playback shows
void CheckStressRequests() {
    constexpr int kRootxSpeed = 8;  // HalfCheetah observes qpos[1:] (8 values), then qvel: rootx first
    {
        StressedCheetah left, kicked;
        kicked.Ask(MjStressRequest::MakeKick(MjKick::Forward, 2.0));
        left.Step();
        kicked.Step();
        Check(kicked.obs[kRootxSpeed] - left.obs[kRootxSpeed] > 1.5, "a forward kick of 2 m/s speeds the root up");
        StressedCheetah up;
        up.Ask(MjStressRequest::MakeKick(MjKick::Up, 2.0));
        up.Step();
        Check(up.obs[kRootxSpeed + 1] - left.obs[kRootxSpeed + 1] > 1.0, "an up kick lifts it");
    }
    {
        StressedCheetah pushed, left;
        const mjtNum force[6] = {300.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        pushed.Ask(MjStressRequest::MakePush(1, force));
        for (int i = 0; i < 3; ++i) {
            pushed.Step();
            left.Step();
        }
        Check(pushed.obs[kRootxSpeed] - left.obs[kRootxSpeed] > 0.5, "a held push keeps pulling the torso");
        pushed.Ask(MjStressRequest::MakeEndPush());
        pushed.Step();
        Check(std::isfinite(pushed.obs[kRootxSpeed]), "letting go");
    }
    {
        StressedCheetah c;
        c.Ask(MjStressRequest::MakeThrow(MjThrownObject{true, 0.2, 2.0}, MjThrowFrom::Ahead, 6.0));
        c.Step();
        Check(c.view->scene.objects.size() == 1, "a throw adds an object slot to the scene the view is told of");
        const double start_gap = c.BodyX(0) - c.BodyX(-1);
        Check(start_gap > 2.0 && start_gap < 3.5, "the ball starts about 3 m ahead of the torso");
        bool obs_finite = true;
        for (float v : c.obs) obs_finite = obs_finite && std::isfinite(v);
        Check(obs_finite, "the robot still observes 17 numbers, all finite");
        double closest = start_gap;
        for (int i = 0; i < 15; ++i) {
            c.Step();
            closest = std::min(closest, std::abs(c.BodyX(0) - c.BodyX(-1)));
        }
        Check(closest < 0.8, "the ball reaches the cheetah");
        c.Ask(MjStressRequest::MakeThrow(MjThrownObject{true, 0.2, 2.0}, MjThrowFrom::Above, 6.0));
        c.Step();
        Check(c.view->scene.objects.size() == 2, "a second throw while the first is out takes a new slot");
    }
    {
        StressedCheetah c({{"max_episode_steps", "5"}});
        const MjGround rough{MjGroundKind::Rough, 0.5, 3};
        c.Ask(MjStressRequest::MakeGround(rough, false));
        c.Step();
        Check(c.view->scene.ground.kind == MjGroundKind::Flat, "a ground for the next episode waits");
        for (int i = 0; i < 3; ++i) c.Step();
        Check(c.truncated == 1 && c.view->scene.ground == rough && c.view->episode == 2,
              "it comes in when the episode ends at its step limit");
        const MjGround steps{MjGroundKind::Steps, 0.3, 2};
        c.Ask(MjStressRequest::MakeGround(steps, true));
        c.Step();
        Check(c.truncated == 1 && c.terminated == 0 && c.view->scene.ground == steps && c.view->episode == 3 &&
                  !c.view->last_ended_by_task,
              "a ground now: the episode is cut (truncated) and the next starts on it");
        Check(c.view->step == 0, "the new episode starts at step 0");
    }
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
    CheckScenes();
    CheckStressRequests();
    ReportSpeed();
    if (!trajectory.empty()) WriteTrajectories(trajectory);
    if (failures == 0) std::printf("test_mujoco_environments: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
