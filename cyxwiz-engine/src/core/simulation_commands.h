// The graph run's commands (TOFIX195 S2.4): one model for the canvas toolbar,
// the Simulation menu and the Simulation Monitor, so every surface offers the
// same commands with the same words and the same reasons. Pure data: no
// ImGui, no session; NodeEditor fills the inputs and runs the commands.
#pragma once

#include "graph_simulation_session.h"

#include <string>

namespace cyxwiz::simulation {

enum class Command { Run, Pause, Step, Stop, Clear };

struct CommandInputs {
    bool simulation_graph = false;  // the open graph has nodes a graph run steps
    bool has_scopes = false;        // ... and a Signal Scope to plot
    SimulationState state = SimulationState::Stopped;
    bool rl_running = false;        // Train RL owns the graph meanwhile
};

struct CommandView {
    Command command = Command::Run;
    std::string label;        // the toolbar's and the menu's words
    std::string short_label;  // the Monitor's button
    std::string hint;         // one sentence: the status bar, a tooltip
    bool enabled = false;
    std::string disabled_reason;  // shown on hover while disabled
};

struct Commands {
    CommandView run;  // "Run Sim", or "Resume" while paused
    CommandView pause;
    CommandView step;
    CommandView stop;
    CommandView clear;
};

Commands BuildCommands(const CommandInputs& inputs);

// "Running", "Paused", "Stopped", "Failed"
const char* StateLabel(SimulationState state);

// The menu's action id of a command ("simulation.run") and back.
const char* ActionId(Command command);
bool CommandFromActionId(const std::string& id, Command& command);

}  // namespace cyxwiz::simulation
