#include "simulation_commands.h"

namespace cyxwiz::simulation {

namespace {

CommandView View(Command command, const char* label, const char* short_label, const char* hint,
                 const std::string& why) {
    CommandView view;
    view.command = command;
    view.label = label;
    view.short_label = short_label;
    view.hint = hint;
    view.enabled = why.empty();
    view.disabled_reason = why;
    return view;
}

}  // namespace

Commands BuildCommands(const CommandInputs& in) {
    const bool running = in.state == SimulationState::Running;
    const bool paused = in.state == SimulationState::Paused;
    const bool active = running || paused;

    std::string run_why;
    if (!in.simulation_graph) run_why = "The open graph has nothing to simulate.";
    else if (running) run_why = "The graph is running.";
    else if (in.rl_running) run_why = "Stop RL training first.";
    const std::string idle_why = active ? "" : (in.simulation_graph ? "Run the graph first." : "Nothing is running.");

    Commands c;
    c.run = paused ? View(Command::Run, "Resume", "Resume", "Resumes the graph run.", run_why)
                   : View(Command::Run, "Run Sim", "Run", "Runs the open graph's simulation.", run_why);
    c.pause = View(Command::Pause, "Pause", "Pause", "Pauses the graph run.", running ? "" : "Nothing is running.");
    c.step = View(Command::Step, "Step Once", "Step", "Pauses the graph run and advances it one tick.", idle_why);
    c.stop = View(Command::Stop, "Stop", "Stop", "Stops the graph run.", active ? "" : "Nothing is running.");
    c.clear = View(Command::Clear, "Clear Traces", "Clear", "Empties every Signal Scope's trace; a run goes on.",
                   in.has_scopes ? "" : "The graph has no Signal Scope.");
    return c;
}

const char* StateLabel(SimulationState state) {
    switch (state) {
        case SimulationState::Running: return "Running";
        case SimulationState::Paused: return "Paused";
        case SimulationState::Failed: return "Failed";
        case SimulationState::Stopped: break;
    }
    return "Stopped";
}

namespace {
constexpr struct {
    Command command;
    const char* id;
} kActions[] = {{Command::Run, "simulation.run"},
                {Command::Pause, "simulation.pause"},
                {Command::Step, "simulation.step"},
                {Command::Stop, "simulation.stop"},
                {Command::Clear, "simulation.clear"}};
}  // namespace

const char* ActionId(Command command) {
    for (const auto& a : kActions)
        if (a.command == command) return a.id;
    return "";
}

bool CommandFromActionId(const std::string& id, Command& command) {
    for (const auto& a : kActions) {
        if (id != a.id) continue;
        command = a.command;
        return true;
    }
    return false;
}

}  // namespace cyxwiz::simulation
