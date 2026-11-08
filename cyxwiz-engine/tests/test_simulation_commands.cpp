// The graph run's commands (core/simulation_commands): what the canvas
// toolbar, the Simulation menu and the Simulation Monitor offer in each
// state, with the same words and reasons.

#include "core/simulation_commands.h"

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

using cyxwiz::SimulationState;
using cyxwiz::simulation::BuildCommands;
using cyxwiz::simulation::Command;
using cyxwiz::simulation::CommandInputs;

CommandInputs Graph(SimulationState state, bool scopes = true) {
    CommandInputs in;
    in.simulation_graph = true;
    in.has_scopes = scopes;
    in.state = state;
    return in;
}

}  // namespace

int main() {
    {
        const auto c = BuildCommands(Graph(SimulationState::Stopped));
        Check(c.run.enabled && c.run.label == "Run Sim" && c.run.short_label == "Run", "stopped: Run Sim");
        Check(!c.pause.enabled && c.pause.disabled_reason == "Nothing is running.", "stopped: nothing to pause");
        Check(!c.step.enabled && c.step.disabled_reason == "Run the graph first.", "stopped: Step needs a run");
        Check(!c.stop.enabled, "stopped: nothing to stop");
        Check(c.clear.enabled, "a graph with Scopes can clear their traces");
    }
    {
        const auto c = BuildCommands(Graph(SimulationState::Running));
        Check(!c.run.enabled && c.run.disabled_reason == "The graph is running.", "running: no second run");
        Check(c.pause.enabled && c.step.enabled && c.stop.enabled, "running: pause, step and stop");
    }
    {
        const auto c = BuildCommands(Graph(SimulationState::Paused));
        Check(c.run.enabled && c.run.label == "Resume" && c.run.short_label == "Resume", "paused: Run resumes");
        Check(!c.pause.enabled && c.step.enabled && c.stop.enabled, "paused: step and stop, no pause");
    }
    {
        const auto c = BuildCommands(Graph(SimulationState::Failed));
        Check(c.run.enabled && c.run.label == "Run Sim" && !c.stop.enabled, "failed: run again");
    }
    {
        auto in = Graph(SimulationState::Stopped, false);
        in.rl_running = true;
        const auto c = BuildCommands(in);
        Check(!c.run.enabled && c.run.disabled_reason == "Stop RL training first.", "Train RL holds the graph");
        Check(!c.clear.enabled && c.clear.disabled_reason == "The graph has no Signal Scope.", "no Scope: nothing to clear");
    }
    {
        const auto c = BuildCommands(CommandInputs{});
        Check(!c.run.enabled && c.run.disabled_reason == "The open graph has nothing to simulate.",
              "no simulation graph: nothing to run");
        Check(!c.step.enabled && c.step.disabled_reason == "Nothing is running.", "no simulation graph: no step");
    }
    for (const Command command : {Command::Run, Command::Pause, Command::Step, Command::Stop, Command::Clear}) {
        Command back = Command::Run;
        const std::string id = cyxwiz::simulation::ActionId(command);
        Check(id.rfind("simulation.", 0) == 0 && cyxwiz::simulation::CommandFromActionId(id, back) && back == command,
              "the action id goes there and back: " + id);
    }
    Command unknown = Command::Clear;
    Check(!cyxwiz::simulation::CommandFromActionId("simulation.monitor", unknown) && unknown == Command::Clear,
          "the Monitor window is no run command");
    Check(std::string(cyxwiz::simulation::StateLabel(SimulationState::Failed)) == "Failed", "state words");

    if (failures == 0) std::printf("test_simulation_commands: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
