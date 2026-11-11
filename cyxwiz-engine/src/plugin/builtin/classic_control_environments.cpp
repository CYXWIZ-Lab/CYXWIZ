#include "classic_control_environments.h"

#include <array>
#include <charconv>
#include <map>
#include <string>
#include <cmath>
#include <vector>

namespace cyxwiz::plugin {

namespace {

// SplitMix64, the same on every compiler (the MuJoCo plugin uses the same
// generator for its copies).
class Random {
public:
    explicit Random(uint64_t seed = 0) : state_(seed) {}
    uint64_t Next() {
        uint64_t z = (state_ += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double Uniform(double low, double high) {
        return low + (high - low) * (static_cast<double>(Next() >> 11) * 0x1.0p-53);
    }
    static uint64_t CopySeed(uint64_t seed, int copy) {
        Random mix(seed ^ (0xD1B54A32D192ED03ull * (static_cast<uint64_t>(copy) + 1)));
        return mix.Next();
    }

private:
    uint64_t state_;
};

constexpr const char* kCartPoleId = "CartPole-v1";
constexpr int kCartPoleSteps = 500;

// Gymnasium 1.2.3 classic_control/cartpole.py: Euler steps of 0.02 s; +1
// every step, the last included; ends when the cart leaves [-2.4, 2.4] or
// the pole leans past 12 degrees.
class CartPoleBatch final : public IEnvironmentBatch {
public:
    CartPoleBatch(int count, int max_steps) : count_(count), max_steps_(max_steps), state_(count), random_(count), steps_(count) {}

    int Count() const override { return count_; }
    int ObservationSize() const override { return 4; }
    int ActionSize() const override { return 1; }

    bool Reset(uint64_t seed, float* observations) override {
        for (int i = 0; i < count_; ++i) {
            random_[i] = Random(Random::CopySeed(seed, i));
            Start(i, observations + 4 * i);
        }
        started_ = true;
        return true;
    }

    bool Step(const float* actions, const PluginEnvironmentStepBuffers& out) override {
        if (!started_) {
            error_ = "Reset the environments before stepping them.";
            return false;
        }
        for (int i = 0; i < count_; ++i) {
            if (actions[i] != 0.0f && actions[i] != 1.0f) {
                error_ = "Copy " + std::to_string(i) + " was given the action " + std::to_string(actions[i]) +
                         "; CartPole takes 0 (push left) or 1 (push right).";
                return false;
            }
        }
        constexpr double gravity = 9.8, masscart = 1.0, masspole = 0.1, total_mass = masspole + masscart;
        constexpr double length = 0.5, polemass_length = masspole * length, force_mag = 10.0, tau = 0.02;
        const double theta_threshold = 12 * 2 * 3.14159265358979323846 / 360;
        for (int i = 0; i < count_; ++i) {
            auto& s = state_[i];
            const double x = s[0], x_dot = s[1], theta = s[2], theta_dot = s[3];
            const double force = actions[i] == 1.0f ? force_mag : -force_mag;
            const double costheta = std::cos(theta), sintheta = std::sin(theta);
            const double temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass;
            const double thetaacc = (gravity * sintheta - costheta * temp) /
                                    (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass));
            const double xacc = temp - polemass_length * thetaacc * costheta / total_mass;
            s = {x + tau * x_dot, x_dot + tau * xacc, theta + tau * theta_dot, theta_dot + tau * thetaacc};
            ++steps_[i];
            const bool terminated = s[0] < -2.4 || s[0] > 2.4 || s[2] < -theta_threshold || s[2] > theta_threshold;
            const bool truncated = !terminated && steps_[i] >= max_steps_;
            float* observation = out.observations + 4 * i;
            Observe(i, observation);
            out.rewards[i] = 1.0f;
            out.terminated[i] = terminated ? 1 : 0;
            out.truncated[i] = truncated ? 1 : 0;
            if (terminated || truncated) {
                for (int k = 0; k < 4; ++k) out.final_observations[4 * i + k] = observation[k];
                Start(i, observation);
            }
        }
        error_.clear();
        return true;
    }

    std::string LastError() const override { return error_; }
    void Release() override { delete this; }

private:
    void Start(int i, float* observation) {
        for (auto& value : state_[i]) value = random_[i].Uniform(-0.05, 0.05);
        steps_[i] = 0;
        Observe(i, observation);
    }
    void Observe(int i, float* observation) const {
        for (int k = 0; k < 4; ++k) observation[k] = static_cast<float>(state_[i][k]);
    }

    int count_;
    int max_steps_;
    std::vector<std::array<double, 4>> state_;
    std::vector<Random> random_;
    std::vector<int> steps_;
    bool started_ = false;
    std::string error_;
};

class ClassicControl final : public IEnvironmentProvider {
public:
    std::vector<PluginEnvironmentInfo> GetEnvironments() override {
        PluginEnvironmentInfo cartpole;
        cartpole.id = kCartPoleId;
        cartpole.name = "Cart Pole";
        cartpole.description =
            "Balance a pole on a cart by pushing it left or right. +1 for each step until the pole leans past "
            "12 degrees or the cart leaves the track.";
        cartpole.category = "Classic Control";
        cartpole.observation_size = 4;
        cartpole.discrete_actions = true;
        cartpole.action_size = 1;
        cartpole.action_count = 2;
        cartpole.max_episode_steps = kCartPoleSteps;
        cartpole.step_seconds = 0.02;
        cartpole.settings = {{"max_episode_steps", "Episode steps", "0",
                              "Steps before an episode is cut; 0 keeps the task's own limit (500)"}};
        return {cartpole};
    }

    PluginEnvironmentCreateResult CreateEnvironments(const std::string& id,
                                                     const std::map<std::string, std::string>& settings,
                                                     int count) override {
        if (id != kCartPoleId) return {nullptr, "The Engine has no built-in environment '" + id + "'."};
        if (count < 1) return {nullptr, "Ask for at least one copy."};
        int max_steps = kCartPoleSteps;
        for (const auto& [key, text] : settings) {
            if (key != "max_episode_steps") return {nullptr, "Cart Pole has no setting '" + key + "'."};
            int value = 0;
            const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
            if (result.ec != std::errc() || result.ptr != text.data() + text.size() || value < 0)
                return {nullptr, "Episode steps must be a whole number, 0 or more (got '" + text + "')."};
            if (value > 0) max_steps = value;
        }
        return {new CartPoleBatch(count, max_steps), {}};
    }
};

}  // namespace

IEnvironmentProvider& ClassicControlEnvironments() {
    static ClassicControl provider;
    return provider;
}

}  // namespace cyxwiz::plugin
