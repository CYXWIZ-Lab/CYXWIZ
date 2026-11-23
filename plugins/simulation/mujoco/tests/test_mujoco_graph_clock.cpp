// The MuJoCo Plant's graph clock (src/mj_graph_clock.h): over a graph run the
// physics time stays within half a model step of the graph's time, whatever
// the model step and the tick rate.

#include "mj_graph_clock.h"

#include <cmath>
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

using cyxwiz::plugin::mujoco::MjGraphClock;

// Runs `ticks` graph ticks at `rate_hz` on a model of step `timestep` that
// starts at `physics_start`; returns the worst gap between physics time and
// graph time at the end of a tick, and the physics time at the end.
double WorstGap(double timestep, double rate_hz, int ticks, double physics_start, double& physics_end) {
    MjGraphClock clock;
    clock.Begin(physics_start);
    const float dt = static_cast<float>(1.0 / rate_hz);
    double physics = physics_start;
    float sim_time = 0.0f;  // as the graph executor keeps it
    double worst = 0.0;
    for (int i = 0; i < ticks; ++i) {
        const int steps = clock.StepsForTick(physics, timestep, sim_time, dt);
        for (int s = 0; s < steps; ++s) physics += timestep;
        sim_time += dt;
        worst = std::max(worst, std::abs((physics - physics_start) - static_cast<double>(sim_time)));
    }
    physics_end = physics;
    return worst;
}

}  // namespace

int main() {
    double end = 0.0;
    // Inverted pendulum: 0.02 s steps on a 60 Hz run (the on-screen case:
    // one step a tick ran the physics 1.2x ahead)
    double gap = WorstGap(0.02, 60.0, 249, 0.0, end);
    Check(gap <= 0.01 + 1e-4, "a model step longer than the tick stays on the graph's time");
    Check(std::abs(end - 4.15) <= 0.011, "after 4.15 s of graph time the physics is at 4.15 s, not 4.98 s");

    gap = WorstGap(0.002, 60.0, 600, 0.0, end);
    Check(gap <= 0.001 + 1e-4, "a short model step takes several steps a tick");
    Check(std::abs(end - 10.0) <= 0.0011, "10 s of graph time");

    gap = WorstGap(0.01, 240.0, 1000, 3.5, end);
    Check(gap <= 0.005 + 1e-4, "a run that begins with the physics at 3.5 s counts from there");

    {
        MjGraphClock clock;
        clock.Begin(0.0);
        Check(clock.StepsForTick(0.0, 0.01, 0.0f, 0.1f) == 10, "0.1 s tick, 0.01 s steps: ten steps");
        Check(clock.StepsForTick(0.0, 0.0, 0.0f, 0.1f) == 0, "no timestep, no steps");
        Check(clock.StepsForTick(0.1, 0.01, 0.1f, 0.0f) == 0, "a tick of no time steps nothing");
    }
    {
        // MuJoCo reset the unstable model to time 0 at graph time 5 s: the
        // next tick takes one tick's steps, not five seconds of them
        MjGraphClock clock;
        clock.Begin(0.0);
        const int steps = clock.StepsForTick(0.0, 0.01, 5.0f, 0.1f);
        Check(steps == 10, "after a physics reset the run goes on from there");
        Check(std::abs(clock.Origin() + 5.0) < 1e-9, "the origin moves with the reset");
    }

    if (failures == 0) std::printf("test_mujoco_graph_clock: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
