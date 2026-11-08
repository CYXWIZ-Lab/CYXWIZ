// GraphSimulationSession: Scope history without any UI, bounded windows,
// failure state and the worker's run / pause / step / stop.

#include "../src/core/graph_simulation_session.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

bool Near(float actual, float expected, float tolerance = 1e-5f) {
    return std::abs(actual - expected) <= tolerance;
}

gui::NodePin Pin(int id, const char* name, bool is_input) {
    gui::NodePin pin{};
    pin.id = id;
    pin.type = gui::PinType::Tensor;
    pin.name = name;
    pin.is_input = is_input;
    return pin;
}

gui::MLNode Source(int id, gui::NodeType type, int output_pin) {
    gui::MLNode node{};
    node.id = id;
    node.type = type;
    node.category = type == gui::NodeType::Constant ? gui::NodeCategory::Utility : gui::NodeCategory::Signal;
    node.name = "Source";
    node.outputs.push_back(Pin(output_pin, type == gui::NodeType::Constant ? "Value" : "Signal", false));
    return node;
}

gui::MLNode Scope(int id, int input_pin) {
    gui::MLNode node{};
    node.id = id;
    node.type = gui::NodeType::SignalScope;
    node.category = gui::NodeCategory::Signal;
    node.name = "Signal Scope";
    node.inputs.push_back(Pin(input_pin, "Signal", true));
    return node;
}

std::size_t Samples(const cyxwiz::GraphSimulationSession& session, int scope_id) {
    std::vector<float> times;
    std::vector<float> values;
    session.CopyScope(scope_id, times, values);
    return times.size();
}

bool WaitFor(const std::function<bool()>& condition) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
        if (condition()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return false;
}

// Ramp -> Scope: every sample pairs a value with the time it was evaluated
// at, and nothing draws.
void CheckHistoryWithoutUi() {
    auto ramp = Source(1, gui::NodeType::RampSignal, 101);
    ramp.parameters["start_value"] = "0";
    ramp.parameters["end_value"] = "1";
    ramp.parameters["duration"] = "1";
    cyxwiz::GraphSimulationSession session;
    Check(session.BuildForTest({ramp, Scope(2, 102)}, {{1, 1, 101, 2, 102}}, nullptr), session.GetError());
    for (int i = 0; i < 5; ++i) Check(session.TickForTest(0.1f), session.GetError());

    std::vector<float> times;
    std::vector<float> values;
    Check(session.CopyScope(2, times, values), "the Scope should have a history");
    Check(times.size() == 5 && values.size() == 5, "one sample a tick");
    for (std::size_t i = 0; i < times.size(); ++i) {
        Check(Near(times[i], 0.1f * static_cast<float>(i)), "sample time is the tick's time");
        Check(Near(values[i], times[i]), "the ramp's value at the sample's time");
    }
    Check(!session.CopyScope(1, times, values), "a source node is no Scope");
}

void CheckBoundedWindows() {
    auto constant = Source(1, gui::NodeType::Constant, 101);
    constant.parameters["value"] = "2.5";
    auto small = Scope(2, 102);
    small.parameters["window_size"] = "10";
    auto other = Scope(3, 103);  // default window
    cyxwiz::GraphSimulationSession session;
    Check(session.BuildForTest({constant, small, other}, {{1, 1, 101, 2, 102}, {2, 1, 101, 3, 103}}, nullptr),
          session.GetError());
    for (int i = 0; i < 50; ++i) Check(session.TickForTest(0.01f), session.GetError());
    Check(Samples(session, 2) == 10, "window_size bounds the history");
    Check(Samples(session, 3) == 50, "each Scope keeps its own history");

    session.SetNodeParameter(2, "window_size", "12");
    for (int i = 0; i < 5; ++i) Check(session.TickForTest(0.01f), session.GetError());
    Check(Samples(session, 2) == 12, "a live window_size edit resizes the history");
    session.SetNodeParameter(2, "window_size", "3");
    Check(Samples(session, 2) == 10, "the window is at least 10 samples");
    session.SetNodeParameter(3, "window_size", "not a number");
    Check(Samples(session, 3) == 55, "an unreadable window keeps the default (500)");

    session.ClearScope(2);
    Check(Samples(session, 2) == 0 && Samples(session, 3) == 55, "Clear empties one Scope only");
    session.ClearHistory();
    std::vector<float> times;
    std::vector<float> values;
    Check(!session.CopyScope(3, times, values), "ClearHistory forgets the Scopes");
}

void CheckFailureState() {
    cyxwiz::GraphSimulationSession session;
    Check(session.BuildForTest({Scope(2, 102)}, {}, nullptr), session.GetError());
    Check(!session.TickForTest(0.1f), "an unconnected Scope fails the tick");
    Check(session.GetState() == cyxwiz::SimulationState::Failed, "a failed tick leaves the Failed state");
    Check(session.GetError().find("not connected") != std::string::npos, "the reason says what is wrong");

    gui::NodeLink dangling{1, 9, 900, 2, 102};
    Check(!session.Start({Scope(2, 102)}, {dangling}, nullptr), "a graph that does not build does not start");
    Check(session.GetState() == cyxwiz::SimulationState::Failed && !session.IsActive(), "no worker after a failed build");
    Check(!session.Start({Scope(2, 102)}, {}, nullptr, 0.0f), "a zero rate does not start");
}

void CheckWorkerCommands() {
    auto constant = Source(1, gui::NodeType::Constant, 101);
    cyxwiz::GraphSimulationSession session;
    Check(session.Start({constant, Scope(2, 102)}, {{1, 1, 101, 2, 102}}, nullptr, 1000.0f), session.GetError());
    const auto first_run = session.GetRunId();
    Check(session.GetState() == cyxwiz::SimulationState::Running, "Start runs");
    Check(WaitFor([&]() { return Samples(session, 2) >= 20; }), "the worker fills the Scope by itself");

    session.Pause();
    Check(session.GetState() == cyxwiz::SimulationState::Paused, "Pause pauses");
    std::this_thread::sleep_for(std::chrono::milliseconds(20));  // a tick in flight lands
    const auto paused = Samples(session, 2);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    Check(Samples(session, 2) == paused, "no ticks while paused");
    session.Step();
    Check(WaitFor([&]() { return Samples(session, 2) == paused + 1; }), "Step runs one tick");
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    Check(Samples(session, 2) == paused + 1, "and only one");

    session.Resume();
    Check(WaitFor([&]() { return Samples(session, 2) > paused + 10; }), "Resume ticks again");
    session.Stop();
    Check(session.GetState() == cyxwiz::SimulationState::Stopped && !session.IsActive(), "Stop stops");
    const auto kept = Samples(session, 2);
    Check(kept > 0, "history stays after Stop");
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    Check(Samples(session, 2) == kept, "no ticks after Stop");
    session.Stop();  // a second Stop is harmless

    Check(session.Start({constant, Scope(2, 102)}, {{1, 1, 101, 2, 102}}, nullptr, 1000.0f), session.GetError());
    Check(session.GetRunId() == first_run + 1, "each run has its own id");
    Check(WaitFor([&]() { return Samples(session, 2) > 0; }), "the new run ticks");
    std::vector<float> times;
    std::vector<float> values;
    session.CopyScope(2, times, values);
    Check(Near(times.front(), 0.0f), "a new run starts a new history");
    // The destructor stops and joins a running worker.
}

} // namespace

int main() {
    CheckHistoryWithoutUi();
    CheckBoundedWindows();
    CheckFailureState();
    CheckWorkerCommands();
    std::cout << "test_graph_simulation_session: all checks passed\n";
    return 0;
}
