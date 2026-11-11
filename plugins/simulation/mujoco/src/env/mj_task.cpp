#include "mj_task.h"

#include <charconv>
#include <cmath>

namespace cyxwiz::plugin::mujoco {

double MjRandom::Normal() {
    if (has_spare_) {
        has_spare_ = false;
        return spare_;
    }
    double u1 = Uniform();
    while (u1 <= 0.0) u1 = Uniform();
    const double u2 = Uniform();
    const double radius = std::sqrt(-2.0 * std::log(u1));
    const double angle = 2.0 * 3.14159265358979323846 * u2;
    spare_ = radius * std::sin(angle);
    has_spare_ = true;
    return radius * std::cos(angle);
}

uint64_t MjRandom::CopySeed(uint64_t seed, int copy) {
    MjRandom mix(seed ^ (0xD1B54A32D192ED03ull * (static_cast<uint64_t>(copy) + 1)));
    return mix.Next();
}

namespace {

double Value(const MjTaskValues& values, const char* key) { return values.at(key); }

void SetState(const mjModel* m, mjData* d, const double* qpos, const double* qvel) {
    for (int i = 0; i < m->nq; ++i) d->qpos[i] = qpos[i];
    for (int i = 0; i < m->nv; ++i) d->qvel[i] = qvel[i];
    mj_forward(m, d);
}

void StepFrames(const mjModel* m, mjData* d, int frames) {
    for (int i = 0; i < frames; ++i) mj_step(m, d);
}

// InvertedPendulum-v5: keep the pole within 0.2 rad of upright. +1 a step
// while it is; the episode ends when it is not or a value is not finite.
class InvertedPendulum final : public MjTask {
public:
    explicit InvertedPendulum(const MjTaskValues& values)
        : frame_skip_(static_cast<int>(Value(values, "frame_skip"))), noise_(Value(values, "reset_noise_scale")) {}
    int ObservationSize(const mjModel* m) const override { return static_cast<int>(m->nq + m->nv); }
    void Observe(const mjModel* m, const mjData* d, float* out) const override {
        for (int i = 0; i < m->nq; ++i) *out++ = static_cast<float>(d->qpos[i]);
        for (int i = 0; i < m->nv; ++i) *out++ = static_cast<float>(d->qvel[i]);
    }
    void ResetState(const mjModel* m, mjData* d, MjRandom& random) const override {
        std::vector<double> qpos(m->nq), qvel(m->nv);
        for (int i = 0; i < m->nq; ++i) qpos[i] = m->qpos0[i] + random.Uniform(-noise_, noise_);
        for (int i = 0; i < m->nv; ++i) qvel[i] = random.Uniform(-noise_, noise_);
        SetState(m, d, qpos.data(), qvel.data());
    }
    MjTaskStep Step(const mjModel* m, mjData* d, const float*) const override {
        StepFrames(m, d, frame_skip_);
        bool finite = true;
        for (int i = 0; i < m->nq; ++i) finite = finite && std::isfinite(d->qpos[i]);
        for (int i = 0; i < m->nv; ++i) finite = finite && std::isfinite(d->qvel[i]);
        MjTaskStep step;
        step.terminated = !finite || std::abs(d->qpos[1]) > 0.2;
        step.reward = step.terminated ? 0.0 : 1.0;
        return step;
    }
    int FrameSkip() const override { return frame_skip_; }

private:
    int frame_skip_;
    double noise_;
};

std::unique_ptr<MjTask> MakeInvertedPendulum(const MjTaskValues& values) {
    return std::make_unique<InvertedPendulum>(values);
}

// HalfCheetah-v5: run forward. Reward = forward weight x forward speed -
// control weight x sum of squared actions; never terminates.
class HalfCheetah final : public MjTask {
public:
    explicit HalfCheetah(const MjTaskValues& values)
        : frame_skip_(static_cast<int>(Value(values, "frame_skip"))),
          noise_(Value(values, "reset_noise_scale")),
          forward_weight_(Value(values, "forward_reward_weight")),
          ctrl_weight_(Value(values, "ctrl_cost_weight")) {}
    // The x position is left out of the observation (Gymnasium's default)
    int ObservationSize(const mjModel* m) const override { return static_cast<int>(m->nq - 1 + m->nv); }
    void Observe(const mjModel* m, const mjData* d, float* out) const override {
        for (int i = 1; i < m->nq; ++i) *out++ = static_cast<float>(d->qpos[i]);
        for (int i = 0; i < m->nv; ++i) *out++ = static_cast<float>(d->qvel[i]);
    }
    void ResetState(const mjModel* m, mjData* d, MjRandom& random) const override {
        std::vector<double> qpos(m->nq), qvel(m->nv);
        for (int i = 0; i < m->nq; ++i) qpos[i] = m->qpos0[i] + random.Uniform(-noise_, noise_);
        for (int i = 0; i < m->nv; ++i) qvel[i] = noise_ * random.Normal();
        SetState(m, d, qpos.data(), qvel.data());
    }
    MjTaskStep Step(const mjModel* m, mjData* d, const float* action) const override {
        const double x_before = d->qpos[0];
        StepFrames(m, d, frame_skip_);
        const double x_velocity = (d->qpos[0] - x_before) / (m->opt.timestep * frame_skip_);
        double ctrl_cost = 0.0;
        for (int i = 0; i < m->nu; ++i) ctrl_cost += static_cast<double>(action[i]) * action[i];
        MjTaskStep step;
        step.reward = forward_weight_ * x_velocity - ctrl_weight_ * ctrl_cost;
        return step;
    }
    int FrameSkip() const override { return frame_skip_; }

private:
    int frame_skip_;
    double noise_;
    double forward_weight_;
    double ctrl_weight_;
};

std::unique_ptr<MjTask> MakeHalfCheetah(const MjTaskValues& values) { return std::make_unique<HalfCheetah>(values); }

MjTaskSetting FrameSkip(double value) {
    return {"frame_skip", "Frame skip", value, 1.0, true, "Model steps in one environment step"};
}

MjTaskSetting ResetNoise(double value) {
    return {"reset_noise_scale", "Start noise", value, 0.0, false,
            "How far the start state is set at random from the model's rest state"};
}

bool ParseNumber(const std::string& text, double& out) {
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto result = std::from_chars(begin, end, out);
    return result.ec == std::errc() && result.ptr == end && std::isfinite(out);
}

}  // namespace

const std::vector<MjTaskSpec>& MjTasks() {
    static const std::vector<MjTaskSpec> tasks = [] {
        std::vector<MjTaskSpec> list;
        list.push_back({"InvertedPendulum-v5", "Inverted Pendulum",
                        "Balance a pole on a cart. +1 for each step the pole stays within 0.2 rad of upright.",
                        "Classic Control", "inverted_pendulum.xml", 1000,
                        {FrameSkip(2), ResetNoise(0.01)}, &MakeInvertedPendulum});
        list.push_back({"HalfCheetah-v5", "Half Cheetah",
                        "Make a 2D cheetah run forward: forward speed minus a control cost.", "Locomotion",
                        "half_cheetah.xml", 1000,
                        {FrameSkip(5), ResetNoise(0.1),
                         {"forward_reward_weight", "Forward reward weight", 1.0, 0.0, false,
                          "Reward per metre a second of forward speed"},
                         {"ctrl_cost_weight", "Control cost weight", 0.1, 0.0, false,
                          "Cost per unit of squared action"}},
                        &MakeHalfCheetah});
        return list;
    }();
    return tasks;
}

const MjTaskSpec* FindMjTask(const std::string& id) {
    for (const auto& task : MjTasks()) {
        if (task.id == id) return &task;
    }
    return nullptr;
}

const std::vector<MjTaskSetting>& MjBatchSettings() {
    static const std::vector<MjTaskSetting> settings = {
        {"max_episode_steps", "Episode steps", 0.0, 0.0, true,
         "Steps before an episode is cut; 0 keeps the task's own limit"},
        {"threads", "Threads", 0.0, 0.0, true, "Threads that step the copies; 0 picks one per core, up to the copies"},
    };
    return settings;
}

bool ReadMjTaskValues(const MjTaskSpec& spec, const std::map<std::string, std::string>& given, MjTaskValues& values,
                      std::string& error) {
    values.clear();
    std::vector<const MjTaskSetting*> known;
    for (const auto& setting : spec.settings) known.push_back(&setting);
    for (const auto& setting : MjBatchSettings()) known.push_back(&setting);
    for (const auto* setting : known) values[setting->key] = setting->default_value;
    for (const auto& [key, text] : given) {
        const MjTaskSetting* setting = nullptr;
        for (const auto* candidate : known) {
            if (candidate->key == key) setting = candidate;
        }
        if (!setting) {
            error = spec.name + " has no setting '" + key + "'.";
            return false;
        }
        double value = 0.0;
        if (!ParseNumber(text, value)) {
            error = setting->label + " must be a number (got '" + text + "').";
            return false;
        }
        if (value < setting->min) {
            error = setting->label + " must be at least " + std::to_string(static_cast<long long>(setting->min)) + ".";
            return false;
        }
        if (setting->integer && value != std::floor(value)) {
            error = setting->label + " must be a whole number (got '" + text + "').";
            return false;
        }
        values[key] = value;
    }
    return true;
}

}  // namespace cyxwiz::plugin::mujoco
