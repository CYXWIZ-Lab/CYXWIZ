#pragma once

// The MuJoCo Plant keeps the physics on the graph run's clock. A tick covers
// [sim_time, sim_time + dt) of graph time; the model advances in steps of its
// own timestep. StepsForTick says how many model steps bring the physics to
// within half a model step of the tick's end, so a model step longer than the
// tick steps on some ticks only and the physics never runs ahead.

namespace cyxwiz::plugin::mujoco {

class MjGraphClock {
public:
    // The run begins: graph time 0 is this physics time.
    void Begin(double physics_time) { origin_ = physics_time; }

    // Model steps for the tick at graph time sim_time lasting dt. A physics
    // time behind the tick's start (MuJoCo resets an unstable model to time
    // 0) moves the origin, so the run goes on from there instead of catching
    // up all at once.
    int StepsForTick(double physics_time, double timestep, double sim_time, double dt) {
        if (!(timestep > 0.0)) return 0;
        const double half_step = 0.5 * timestep;
        if (physics_time + half_step < origin_ + sim_time) origin_ = physics_time - sim_time;
        const double target = origin_ + sim_time + dt;
        int steps = 0;
        // MuJoCo adds the timestep to the time each step: count the same way
        for (double t = physics_time; target - t >= half_step; t += timestep) ++steps;
        return steps;
    }

    double Origin() const { return origin_; }

private:
    double origin_ = 0.0;
};

}  // namespace cyxwiz::plugin::mujoco
