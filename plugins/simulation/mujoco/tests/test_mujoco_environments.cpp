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

#include <array>
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

// The model file with the scene added, and its own sizes (null with the reason)
mjModel* CompileMjScene(const std::string& model_path, const MjScene& scene, MjRobotSize& robot, std::string& error) {
    MjSceneBuilder builder;
    if (!builder.Load(model_path, error)) return nullptr;
    robot = builder.Robot();
    return builder.Build(scene, error);
}

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
    Check(cheetah.settings.size() == 6 + RobustnessSettings().size() && has_threads,
          "the cheetah's four settings, the two of every batch and the robustness section");
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
    scene.objects = {MjThrownObject{MjShape::Ball, {0.2, 0.0, 0.0}, 2.0},
                     MjThrownObject{MjShape::Box, {0.3, 0.3, 0.3}, 5.0}};
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
    // Every shape, its geom and its mass and inertia
    MjScene shapes;
    shapes.objects = {MjThrownObject{MjShape::Box, {0.6, 0.15, 0.1}, 3.0},
                      MjThrownObject{MjShape::Capsule, {0.5, 0.1, 0.0}, 1.0},
                      MjThrownObject{MjShape::Cylinder, {0.4, 0.2, 0.0}, 4.0}};
    mjModel* sm = CompileMjScene(path, shapes, robot, error);
    Check(sm != nullptr, "a box, a capsule and a cylinder build: " + error);
    if (sm && plain) {
        const int box = static_cast<int>(plain->nbody), capsule = box + 1, cylinder = box + 2;
        const auto geom = [&](int body) { return sm->body_geomadr[body]; };
        Check(sm->geom_type[geom(box)] == mjGEOM_BOX && std::abs(sm->geom_size[3 * geom(box)] - 0.3) < 1e-12 &&
                  std::abs(sm->geom_size[3 * geom(box) + 1] - 0.075) < 1e-12 &&
                  std::abs(sm->geom_size[3 * geom(box) + 2] - 0.05) < 1e-12,
              "a box 0.60 x 0.15 x 0.10 m: half sizes 0.3, 0.075, 0.05");
        Check(sm->geom_type[geom(capsule)] == mjGEOM_CAPSULE && std::abs(sm->geom_size[3 * geom(capsule)] - 0.05) < 1e-12 &&
                  std::abs(sm->geom_size[3 * geom(capsule) + 1] - 0.2) < 1e-12,
              "a capsule 0.5 m end to end, 0.1 m across: radius 0.05, a 0.4 m middle");
        Check(sm->geom_type[geom(cylinder)] == mjGEOM_CYLINDER &&
                  std::abs(sm->geom_size[3 * geom(cylinder)] - 0.1) < 1e-12 &&
                  std::abs(sm->geom_size[3 * geom(cylinder) + 1] - 0.2) < 1e-12,
              "a cylinder 0.4 m long, 0.2 m across");
        const double* inertia = sm->body_inertia + 3 * box;
        Check(std::abs(sm->body_mass[box] - 3.0) < 1e-12 &&
                  std::abs(inertia[0] - 3.0 * (0.15 * 0.15 + 0.1 * 0.1) / 12.0) < 1e-12 &&
                  std::abs(inertia[2] - 3.0 * (0.6 * 0.6 + 0.15 * 0.15) / 12.0) < 1e-12,
              "the box's mass and inertia are exact");
        Check(std::abs(sm->body_inertia[3 * cylinder + 2] - 0.5 * 4.0 * 0.1 * 0.1) < 1e-12,
              "the cylinder's inertia about its axis is m r^2 / 2");
        mj_deleteModel(sm);
    }
    MjScene stubby;  // as long as it is wide: random throws make these from the default 0.2 x 0.2
    stubby.objects = {MjThrownObject{MjShape::Capsule, {0.2, 0.2, 0.0}, 1.0}};
    mjModel* stub = CompileMjScene(path, stubby, robot, error);
    Check(stub != nullptr, "a capsule no longer than it is wide still builds: " + error);
    if (stub) mj_deleteModel(stub);
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
    double BodyX(int body_from_robot_end) { return BodyPos(body_from_robot_end)[0]; }
    std::array<double, 3> BodyPos(int body_from_robot_end, double* spin = nullptr) {
        MjRobotSize robot;
        std::string error;
        mjModel* m = CompileMjScene(view->model_path, view->scene, robot, error);
        if (!m) return {0.0, 0.0, 0.0};
        mjData* d = mj_makeData(m);
        std::array<double, 3> pos{0.0, 0.0, 0.0};
        if (static_cast<size_t>(mj_stateSize(m, mjSTATE_FULLPHYSICS)) == view->state.size()) {
            mj_setState(m, d, view->state.data(), mjSTATE_FULLPHYSICS);
            mj_forward(m, d);
            const int body = body_from_robot_end < 0 ? 1 : robot.nbody + body_from_robot_end;
            for (int i = 0; i < 3; ++i) pos[i] = d->xpos[3 * body + i];
            if (spin && body_from_robot_end >= 0) {
                const int dof = m->jnt_dofadr[m->body_jntadr[body]];
                *spin = std::sqrt(d->qvel[dof + 3] * d->qvel[dof + 3] + d->qvel[dof + 4] * d->qvel[dof + 4] +
                                  d->qvel[dof + 5] * d->qvel[dof + 5]);
            }
        }
        mj_deleteData(d);
        mj_deleteModel(m);
        return pos;
    }
};

// The Stress panel's requests reach the copy a playback shows
void CheckStressRequests() {
    constexpr int kRootxSpeed = 8;  // HalfCheetah observes qpos[1:] (8 values), then qvel: rootx first
    {
        StressedCheetah left, kicked;
        kicked.Ask(MjStressRequest::MakeKick({2.0, 0.0, 0.0}, 0.0));
        left.Step();
        kicked.Step();
        Check(kicked.obs[kRootxSpeed] - left.obs[kRootxSpeed] > 1.5, "a forward kick of 2 m/s speeds the root up");
        StressedCheetah up;
        up.Ask(MjStressRequest::MakeKick({0.0, 0.0, 2.0}, 0.0));
        up.Step();
        Check(up.obs[kRootxSpeed + 1] - left.obs[kRootxSpeed + 1] > 1.0, "an up kick lifts it");
        // The same 2 m/s as a push held 0.2 s (4 steps of 0.05 s): slower to build, then about the same
        StressedCheetah timed, still;
        timed.Ask(MjStressRequest::MakeKick({2.0, 0.0, 0.0}, 0.2));
        timed.Step();
        still.Step();
        const double after_one = timed.obs[kRootxSpeed] - still.obs[kRootxSpeed];
        for (int i = 0; i < 3; ++i) {
            timed.Step();
            still.Step();
        }
        const double after_four = timed.obs[kRootxSpeed] - still.obs[kRootxSpeed];
        Check(after_one > 0.2 && after_one < 1.2 && after_four > 1.2,
              "a push over 0.2 s builds the speed over its four steps");
        timed.Step();
        still.Step();
        Check(std::abs((timed.obs[kRootxSpeed] - still.obs[kRootxSpeed]) - after_four) < 1.0,
              "and stops pushing when its time is up");
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
        c.Ask(MjStressRequest::MakeThrow(MjThrownObject{MjShape::Ball, {0.2, 0.0, 0.0}, 2.0}, MjThrow{}));
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
        MjThrow above;
        above.up = 90.0;
        c.Ask(MjStressRequest::MakeThrow(MjThrownObject{MjShape::Ball, {0.2, 0.0, 0.0}, 2.0}, above));
        c.Step();
        Check(c.view->scene.objects.size() == 2, "a second throw while the first is out takes a new slot");
    }
    {
        // From the left (around 90) and turned and spinning: it starts beside the robot, spinning
        StressedCheetah c;
        MjThrow left;
        left.around = 90.0;
        left.distance = 2.0;
        left.quat = {0.7071067811865476, 0.0, 0.7071067811865476, 0.0};
        left.spin = {0.0, 0.0, 5.0};
        c.Ask(MjStressRequest::MakeThrow(MjThrownObject{MjShape::Capsule, {0.5, 0.1, 0.0}, 1.0}, left));
        c.Step();
        double spin = 0.0;
        const std::array<double, 3> object = c.BodyPos(0, &spin);
        const std::array<double, 3> torso = c.BodyPos(-1);
        Check(object[1] - torso[1] > 1.0 && std::abs(object[0] - torso[0]) < 1.0,
              "a throw from the left starts beside the robot, not ahead of it");
        Check(spin > 3.0, "a tumbling object leaves spinning");
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

// Training for robustness (board 5): settings, a ground per copy and episode,
// factors, pushes, the curriculum, Look ahead and the stats
MjEnvBatch* CheetahBatch(const std::map<std::string, std::string>& settings, int count, std::string& error) {
    const MjTaskSpec& spec = *FindMjTask("HalfCheetah-v5");
    return MjEnvBatch::Create(spec, ModelPath(spec), settings, count, nullptr, error);
}

void CheckRobustnessSettings() {
    MjRobustness r;
    std::string error;
    Check(ReadMjRobustness({}, r, error) && !r.Any(), "nothing set: no robustness (the plain task)");
    Check(ReadMjRobustness({{"grounds", "rough,steps"}, {"difficulty_from", "0.2"}, {"difficulty_to", "0.6"},
                            {"curriculum", "3000"}, {"push_every", "2"}, {"push_speed", "1"}, {"friction_percent", "20"},
                            {"mass_percent", "10"}, {"motor_percent", "5"}, {"look_ahead", "1"}},
                           r, error) &&
              r.grounds.size() == 2 && r.difficulty_from == 0.2 && r.difficulty_to == 0.6 && r.curriculum_mark == 3000.0 &&
              r.push_every == 2.0 && r.friction_percent == 20.0 && r.look_ahead && r.PerCopyModels(),
          "every robustness setting reads: " + error);
    Check(!ReadMjRobustness({{"grounds", "lava"}}, r, error) && error.find("lava") != std::string::npos,
          "an unknown ground is refused");
    Check(!ReadMjRobustness({{"difficulty_from", "0.8"}, {"difficulty_to", "0.2"}}, r, error), "from above to is refused");
    Check(!ReadMjRobustness({{"mass_percent", "95"}}, r, error), "a factor beyond 90 % is refused");
    Check(!ReadMjRobustness({{"look_ahead", "yes"}}, r, error), "a flag is 1 or 0");

    const MjTaskSpec& spec = *FindMjTask("HalfCheetah-v5");
    PluginEnvironmentInfo listed, described;
    Check(DescribeMjTask(spec, ModelPath(spec), listed, error), "HalfCheetah describes itself");
    bool has_section = false;
    for (const auto& s : listed.settings) has_section = has_section || s.section == "Robustness (training)";
    Check(has_section, "the robustness settings are listed under their section");
    Check(DescribeMjTaskWith(spec, listed, {{"look_ahead", "1"}}, described, error) &&
              described.observation_size == listed.observation_size + kLookAheadCount,
          "Look ahead adds its 10 heights to the observation");
    Check(DescribeMjTaskWith(spec, listed, {}, described, error) && described.observation_size == listed.observation_size,
          "without it the observation is the task's");
    Check(!DescribeMjTaskWith(spec, listed, {{"grounds", "lava"}}, described, error), "bad settings are refused");
}

void CheckRobustBatches() {
    std::string error;
    {
        std::unique_ptr<MjEnvBatch, void (*)(MjEnvBatch*)> plain(CheetahBatch({}, 3, error),
                                                                [](MjEnvBatch* b) { b->Release(); });
        std::vector<float> obs(3 * 17);
        Check(plain && plain->Reset(1, obs.data()), "a plain batch: " + error);
        if (plain)
            Check(plain->ModelOf(0) == plain->ModelOf(1) && plain->ModelOf(1) == plain->ModelOf(2) &&
                      plain->Stats().empty(),
                  "without robustness all copies share one model, and no Difficulty or Falls before an episode ends");
    }
    const std::map<std::string, std::string> grounds{{"grounds", "rough,steps"}, {"difficulty_from", "0.3"},
                                                     {"difficulty_to", "0.7"}, {"max_episode_steps", "5"}};
    std::unique_ptr<MjEnvBatch, void (*)(MjEnvBatch*)> a(CheetahBatch(grounds, 4, error), [](MjEnvBatch* b) { b->Release(); });
    std::unique_ptr<MjEnvBatch, void (*)(MjEnvBatch*)> b(CheetahBatch(grounds, 4, error), [](MjEnvBatch* b) { b->Release(); });
    Check(a && b, "batches with grounds: " + error);
    if (!a || !b) return;
    std::vector<float> obs(4 * 17), final_obs(4 * 17), rewards(4);
    std::vector<uint8_t> terminated(4), truncated(4);
    a->Reset(9, obs.data());
    b->Reset(9, obs.data());
    bool kinds = true, same = true, differ = false;
    for (int i = 0; i < 4; ++i) {
        const MjGround& g = a->SceneOf(i).ground;
        kinds = kinds && (g.kind == MjGroundKind::Rough || g.kind == MjGroundKind::Steps) && g.difficulty >= 0.3 &&
                g.difficulty <= 0.7;
        same = same && g == b->SceneOf(i).ground;
        if (i) differ = differ || g != a->SceneOf(0).ground;
    }
    Check(kinds, "each copy plays a chosen ground with a difficulty from 0.3 to 0.7");
    Check(same, "the same seed gives each copy the same ground");
    Check(differ && a->ModelOf(0) != a->ModelOf(1), "copies get grounds of their own");
    const MjGround first = a->SceneOf(0).ground;
    const PluginEnvironmentStepBuffers out{obs.data(), rewards.data(), terminated.data(), truncated.data(),
                                           final_obs.data()};
    const std::vector<float> still(4 * 6, 0.0f);
    for (int s = 0; s < 5; ++s) a->Step(still.data(), out);
    Check(truncated[0] == 1 && a->SceneOf(0).ground != first, "the next episode draws a new ground");
    bool reported = false;
    for (const auto& stat : a->Stats()) reported = reported || (stat.label == "Difficulty" || stat.label == "Falls");
    Check(reported, "the batch reports its difficulty and falls");

    // Factors: each episode's model scaled within the ranges
    std::unique_ptr<MjEnvBatch, void (*)(MjEnvBatch*)> f(
        CheetahBatch({{"friction_percent", "50"}, {"mass_percent", "30"}, {"motor_percent", "20"}}, 3, error),
        [](MjEnvBatch* b) { b->Release(); });
    Check(f != nullptr, "a batch with factors: " + error);
    if (f) {
        std::vector<float> fobs(3 * 17);
        f->Reset(4, fobs.data());
        bool in_range = true, varied = false;
        for (int i = 0; i < 3; ++i) {
            // bthigh: gear 120 and gain 1 in the file; the gear stays, the gain scales
            const mjModel* fm = f->ModelOf(i);
            const double gain = fm->actuator_gainprm[0];
            in_range = in_range && gain >= 0.8 - 1e-9 && gain <= 1.2 + 1e-9 && fm->actuator_gear[0] == 120.0;
            varied = varied || std::abs(gain - 1.0) > 1e-6;
        }
        Check(in_range && varied, "motor strength varies within +/- 20 %");
    }

    // Pushes: the same seed with and without them parts the root's speed
    std::unique_ptr<MjEnvBatch, void (*)(MjEnvBatch*)> pushed(
        CheetahBatch({{"push_every", "0.1"}, {"push_speed", "3"}}, 1, error), [](MjEnvBatch* b) { b->Release(); });
    std::unique_ptr<MjEnvBatch, void (*)(MjEnvBatch*)> left(CheetahBatch({}, 1, error), [](MjEnvBatch* b) { b->Release(); });
    Check(pushed && left, "batches for pushes: " + error);
    if (pushed && left) {
        std::vector<float> po(17), lo(17), pf(17), lf(17), pr(1), lr(1);
        std::vector<uint8_t> pt(1), ptr(1), lt(1), ltr(1);
        pushed->Reset(2, po.data());
        left->Reset(2, lo.data());
        const PluginEnvironmentStepBuffers pout{po.data(), pr.data(), pt.data(), ptr.data(), pf.data()};
        const PluginEnvironmentStepBuffers lout{lo.data(), lr.data(), lt.data(), ltr.data(), lf.data()};
        const std::vector<float> zero(6, 0.0f);
        double apart = 0.0;
        for (int s = 0; s < 20; ++s) {
            pushed->Step(zero.data(), pout);
            left->Step(zero.data(), lout);
            for (int k = 8; k < 11; ++k) apart = std::max(apart, static_cast<double>(std::abs(po[k] - lo[k])));
        }
        Check(apart > 0.3, "pushes change the robot's motion");
    }

    // Look ahead: 10 more values, the ground's height ahead of the root
    std::unique_ptr<MjEnvBatch, void (*)(MjEnvBatch*)> look(
        CheetahBatch({{"look_ahead", "1"}, {"grounds", "steps"}, {"difficulty_from", "1"}, {"difficulty_to", "1"}}, 2,
                     error),
        [](MjEnvBatch* b) { b->Release(); });
    Check(look && look->ObservationSize() == 17 + kLookAheadCount, "Look ahead observes 27 values: " + error);
    if (look) {
        std::vector<float> lobs(2 * 27);
        look->Reset(6, lobs.data());
        bool match = true;
        for (int i = 0; i < 2; ++i)
            for (int k = 0; k < kLookAheadCount; ++k)
                match = match && std::abs(lobs[static_cast<size_t>(i) * 27 + 17 + k] -
                                          static_cast<float>(look->ProfileOf(i).At(look->RootX(i) + 0.2 * (k + 1)))) < 1e-6f;
        Check(match, "the heights are the copy's ground ahead of its root");
        Check(look->ProfileOf(0).At(look->ProfileOf(0).x0 + look->ProfileOf(0).boxes.front().x0 + 0.01) ==
                  look->ProfileOf(0).boxes.front().height,
              "the profile reads a step's height on the step");
    }

    // The curriculum: the level rises by 0.1 when the mean of recent returns passes the mark
    MjRobustness r;
    r.grounds = {MjGroundKind::Rough};
    r.difficulty_from = 0.0;
    r.difficulty_to = 0.25;
    r.curriculum_mark = 100.0;
    MjCurriculum curriculum;
    curriculum.Start(r, 1);  // judged on 20 episodes
    for (int e = 0; e < 19; ++e) curriculum.Finished(150.0);
    Check(curriculum.Level() == 0.0 && curriculum.Rising(), "19 good episodes are not yet enough");
    Check(curriculum.Finished(150.0) && std::abs(curriculum.Level() - 0.1) < 1e-12, "the 20th raises it by 0.1");
    for (int e = 0; e < 20; ++e) curriculum.Finished(50.0);
    Check(std::abs(curriculum.Level() - 0.1) < 1e-12, "poor episodes do not raise it");
    for (int e = 0; e < 40; ++e) curriculum.Finished(500.0);
    Check(std::abs(curriculum.Level() - 0.25) < 1e-12 && !curriculum.Rising(), "it stops at 'to'");
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

    // The Engine's label for the shown copy (a training preview)
    PluginShownCopyLabel label{"Preview of update 12", "Training preview", {{"Networks of", "update 12"}}};
    batch->LabelShownCopy(label);
    Check(view->label.subtitle == "Preview of update 12" && view->label.status == "Training preview" &&
              view->label.rows.size() == 1 && view->label.rows[0].first == "Networks of",
          "the label reaches the view");
    Check(batch->ShowCopy(1) && view->label.subtitle.empty() && view->label.rows.empty(),
          "showing a copy again starts with the view's own words");
    // A training's own copy is watch only: a waiting disturbance is dropped
    view->requests.push_back(MjStressRequest{});
    PluginShownCopyLabel training{"Training copy 1", "Training", {}, true};
    batch->LabelShownCopy(training);
    Check(view->label.watch_only && view->requests.empty(), "a watch-only copy takes no disturbance");
    batch->ShowCopy(1);
    batch->LabelShownCopy(label);

    IEnvironmentBatch* other = MjEnvBatch::Create(spec, ModelPath(spec), {}, 1, view, error);
    Check(other && other->ShowCopy(0) && view->owner == other, "another batch takes the view");
    batch->LabelShownCopy(label);
    Check(view->label.subtitle.empty(), "a batch that does not show the copy cannot label it");
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

// ShowCopyAsGhost: a second batch publishes its copy into the view slot's
// ghost part beside the shown copy, with its own episode numbers; Release or
// ShowCopyAsGhost(-1) frees it; no view, no ghost.
void CheckGhostCopy() {
    const MjTaskSpec& spec = *FindMjTask("InvertedPendulum-v5");
    std::string error;
    auto view = std::make_shared<MjViewSlot>();
    IEnvironmentBatch* played = MjEnvBatch::Create(spec, ModelPath(spec), {}, 1, view, error);
    IEnvironmentBatch* ghost = MjEnvBatch::Create(spec, ModelPath(spec), {}, 1, view, error);
    Check(played && ghost, "two batches with one view: " + error);
    if (!played || !ghost) return;
    Check(played->ShowCopy(0) && !ghost->ShowCopyAsGhost(1, "b") && ghost->ShowCopyAsGhost(0, "robust"),
          "copy 1 of 1 is refused; copy 0 is the ghost");
    Check(view->owner == played && view->ghost.owner == ghost && view->ghost.name == "robust" &&
              view->ghost.model_path == ModelPath(spec),
          "the view shows one batch's copy and the other's as the ghost, named");
    std::vector<float> obs(4), ghost_obs(4), final_obs(4), rewards(1), ghost_rewards(1);
    std::vector<uint8_t> terminated(1), truncated(1);
    played->Reset(5, obs.data());
    ghost->Reset(5, ghost_obs.data());
    Check(!view->ghost.state.empty() && view->ghost.state == view->state,
          "on the same seed the ghost starts where the played copy starts");
    const PluginEnvironmentStepBuffers out{obs.data(), rewards.data(), terminated.data(), truncated.data(),
                                           final_obs.data()};
    const PluginEnvironmentStepBuffers ghost_out{ghost_obs.data(), ghost_rewards.data(), terminated.data(),
                                                 truncated.data(), final_obs.data()};
    const std::vector<float> push(1, 0.8f), still(1, 0.0f);
    const uint64_t shown_version = view->version;
    for (int s = 0; s < 4; ++s) {
        played->Step(push.data(), out);
        ghost->Step(still.data(), ghost_out);
    }
    Check(view->ghost.step == 4 && view->step == 4 && view->version == shown_version + 4,
          "both publish every step, each into its own part");
    Check(view->ghost.state != view->state, "other actions, another state");
    Check(view->ghost.episode_return > 3.0, "the ghost counts its own return: " +
                                                std::to_string(view->ghost.episode_return));
    Check(ghost->ShowCopyAsGhost(-1, {}) && view->ghost.owner == nullptr && view->owner == played,
          "ShowCopyAsGhost(-1) removes the ghost, the shown copy stays");
    ghost->ShowCopyAsGhost(0, "robust");
    ghost->Release();
    Check(view->ghost.owner == nullptr, "releasing the ghost's batch frees it");
    played->Release();
    IEnvironmentBatch* plain = MjEnvBatch::Create(spec, ModelPath(spec), {}, 1, nullptr, error);
    Check(plain && !plain->ShowCopyAsGhost(0, "x"), "a batch without a view draws no ghost");
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
    CheckGhostCopy();
    CheckLiveNumbers();
    CheckScenes();
    CheckStressRequests();
    CheckRobustnessSettings();
    CheckRobustBatches();
    ReportSpeed();
    if (!trajectory.empty()) WriteTrajectories(trajectory);
    if (failures == 0) std::printf("test_mujoco_environments: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
