// Unloading a plugin removes everything it registered (TOFIX125, found in the
// step 1.5 check). UnloadPlugin freed the plugin library but left its nodes and
// panels registered, pointing into the freed library; only Shutdown and Disable
// removed them. Uses the image example plugin, built with the engine:
// plugins/examples/image_nodes (three nodes, one panel, no dangerous
// permissions, so it initialises without the approval dialog).
#include "../src/core/extension_node_registry.h"
#include "../src/plugin/plugin_manager.h"
#include "../src/plugin/registries/plugin_panel_registry.h"
#include "../src/plugin/registries/plugin_command_registry.h"
#include "../src/plugin/registries/plugin_preferences_registry.h"
#include "../src/plugin/engine_plugin_context.h"
#include "../src/plugin/registries/plugin_simulation_registry.h"
#include "../src/plugin/registries/plugin_environment_registry.h"
#include "../src/plugin/builtin/classic_control_environments.h"

#include <cmath>
#include <cstdlib>
#include <map>
#include <memory>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char* kPluginId = "com.cyxwiz.examples.image-nodes";

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::filesystem::path PluginDir() {
    auto dir = std::filesystem::current_path();
    while (!dir.empty()) {
        const auto candidate = dir / "plugins" / "examples" / "image_nodes";
        if (std::filesystem::exists(candidate / "plugin.json")) return candidate;
        const auto parent = dir.parent_path();
        if (parent == dir) break;
        dir = parent;
    }
    return {};
}

size_t NodesOfPlugin() {
    size_t count = 0;
    for (const auto& descriptor : cyxwiz::ExtensionNodeRegistry::Instance().All()) {
        if (descriptor.provider_id == kPluginId) ++count;
    }
    return count;
}

void LoadAndInitialize(cyxwiz::plugin::PluginManager& manager, const std::filesystem::path& dir) {
    Check(manager.LoadPlugin(dir), "the image example plugin loads from " + dir.string());
    Check(manager.InitializePlugin(kPluginId), "the image example plugin initialises");
    Check(NodesOfPlugin() == 3, "its three nodes are registered, got " + std::to_string(NodesOfPlugin()));
    Check(cyxwiz::plugin::PluginPanelRegistry::Instance().HasPanel(
              "com.cyxwiz.examples.image-nodes.settings"),
          "its settings panel is registered");
    auto& panels = cyxwiz::plugin::PluginPanelRegistry::Instance();
    Check(!panels.ShowPanel("com.example.other", "com.cyxwiz.examples.image-nodes.settings"),
          "a plugin cannot open another plugin's panel");
    Check(!panels.ShowPanel(kPluginId, "com.cyxwiz.examples.image-nodes.missing"),
          "showing a panel the plugin does not have fails");
    panels.SetPanelVisible("com.cyxwiz.examples.image-nodes.settings", false);
    Check(panels.ShowPanel(kPluginId, "com.cyxwiz.examples.image-nodes.settings") &&
              panels.IsPanelVisible("com.cyxwiz.examples.image-nodes.settings"),
          "a plugin opens its own panel");
}

// A plugin's menu commands (plugin API 1.4): one enabled, one disabled with a reason
class FakeCommands final : public cyxwiz::plugin::ICommandProvider {
public:
    int runs = 0;
    std::vector<cyxwiz::plugin::PluginCommandInfo> GetCommands() override {
        return {{"go", "Simulation", "run", "Go", "Goes.", false, true},
                {"halt", "Simulation", "run", "Halt", "Halts.", false}};
    }
    cyxwiz::plugin::PluginCommandState GetCommandState(const std::string& id) override {
        if (id == "halt") return {false, "Nothing is running.", false};
        return {};
    }
    void RunCommand(const std::string& id) override {
        if (id == "go") ++runs;
    }
};

void CheckCommandRegistry() {
    auto& commands = cyxwiz::plugin::PluginCommandRegistry::Instance();
    FakeCommands fake;
    for (const auto& info : fake.GetCommands()) commands.Register("com.example.sim", info, &fake);
    commands.Register("com.example.sim", fake.GetCommands()[0], &fake);  // a duplicate is skipped
    const auto snapshot = commands.Snapshot();
    Check(snapshot.size() == 2 && snapshot[0].key == "com.example.sim:go" && snapshot[0].state.enabled,
          "commands are listed in order with their state");
    Check(!snapshot[1].state.enabled && snapshot[1].state.disabled_reason == "Nothing is running.",
          "a disabled command keeps its reason");
    Check(snapshot[0].info.replaced_by_graph_run && !snapshot[1].info.replaced_by_graph_run,
          "a command keeps whether a graph run replaces it");
    Check(commands.Run("com.example.sim:go") && fake.runs == 1, "a command runs");
    Check(!commands.Run("com.example.sim:missing"), "an unknown command does not run");
    commands.RemoveByPlugin("com.example.sim");
    Check(commands.GetCount() == 0 && !commands.Run("com.example.sim:go"), "removing the plugin removes its commands");
}

// A plugin's Preferences sections (plugin API 1.4)
class FakePreferences final : public cyxwiz::plugin::IPreferencesProvider {
public:
    std::vector<std::string> drawn;
    std::vector<cyxwiz::plugin::PluginPreferencesSection> GetPreferenceSections() override {
        return {{"viewer", "Simulation", "My Viewer", "How it starts."}, {"odd", "Nowhere", "Odd one", ""}};
    }
    void RenderPreferences(const std::string& id) override {
        if (id == "odd") throw std::runtime_error("broken");
        drawn.push_back(id);
    }
};

void CheckPreferencesRegistry() {
    auto& preferences = cyxwiz::plugin::PluginPreferencesRegistry::Instance();
    FakePreferences fake;
    for (const auto& section : fake.GetPreferenceSections()) preferences.Register("com.example.sim", section, &fake);
    preferences.Register("com.example.sim", fake.GetPreferenceSections()[0], &fake);  // a duplicate is skipped
    const std::vector<std::string> tabs = {"General", "Simulation"};
    const auto simulation = preferences.SectionsFor("Simulation", tabs, "General");
    const auto general = preferences.SectionsFor("General", tabs, "General");
    Check(preferences.GetCount() == 2 && simulation.size() == 1 && simulation[0].key == "com.example.sim:viewer" &&
              simulation[0].section.title == "My Viewer" && simulation[0].section.hint == "How it starts.",
          "a section is listed in the tab it names");
    Check(general.size() == 1 && general[0].section.title == "Odd one", "a tab the Engine does not have puts it in General");
    Check(preferences.Render("com.example.sim:viewer") && fake.drawn == std::vector<std::string>{"viewer"},
          "the section draws");
    Check(!preferences.Render("com.example.sim:odd"), "a section that throws is reported, the dialog goes on");
    Check(!preferences.Render("com.example.sim:missing"), "an unknown section does not draw");
    preferences.RemoveByPlugin("com.example.sim");
    Check(preferences.GetCount() == 0 && preferences.SectionsFor("Simulation", tabs, "General").empty(),
          "removing the plugin removes its sections");
}

// A plugin's settings and the project folder (plugin API 1.4): kept for the
// session without a host's store, else read and written through it
void CheckPluginSettings() {
    using cyxwiz::plugin::EnginePluginContext;
    EnginePluginContext::SetHostServices({});
    EnginePluginContext a("com.example.a", "1.0.0", {1, 4, 0}, nullptr);
    EnginePluginContext b("com.example.b", "1.0.0", {1, 4, 0}, nullptr);
    Check(a.GetProjectDir().empty(), "no project without the host's services");
    Check(a.GetSetting("speed", "1") == "1", "a key never set gives the fallback");
    a.SetSetting("speed", "2");
    Check(a.GetSetting("speed", "1") == "2" && b.GetSetting("speed", "1") == "1",
          "a setting is kept per plugin for the session");

    std::map<std::string, std::string> store;
    EnginePluginContext::SetHostServices({
        [] { return std::filesystem::path("D:/projects/demo"); },
        [&store](const std::string& plugin, const std::string& key, std::string& value) {
            const auto it = store.find(plugin + "/" + key);
            if (it == store.end()) return false;
            value = it->second;
            return true;
        },
        [&store](const std::string& plugin, const std::string& key, const std::string& value) {
            store[plugin + "/" + key] = value;
        },
    });
    Check(a.GetProjectDir() == std::filesystem::path("D:/projects/demo"), "the project folder comes from the host");
    a.SetSetting("camera", "follow");
    Check(store.at("com.example.a/camera") == "follow" && a.GetSetting("camera") == "follow" &&
              b.GetSetting("camera", "track") == "track",
          "the host stores each plugin's settings under its id");
    a.SetSetting("", "ignored");
    Check(store.size() == 1, "an empty key is not stored");
    EnginePluginContext::SetHostServices({});
}

// A plugin whose nodes run in graph simulations (plugin API 1.4)
class FakeSimulation final : public cyxwiz::plugin::ISimulationProvider {
public:
    bool refuse = false;
    int begins = 0;
    int ends = 0;
    std::vector<cyxwiz::plugin::PluginSimulationGraphNode> nodes;
    std::vector<cyxwiz::plugin::PluginSimulationNodeInfo> GetSimulationNodes() override {
        return {{"Plant", "sim.view"}, {"Probe", ""}};
    }
    cyxwiz::plugin::PluginSimulationBeginResult BeginGraphRun(
        const std::vector<cyxwiz::plugin::PluginSimulationGraphNode>& graph_nodes) override {
        ++begins;
        nodes = graph_nodes;
        if (refuse) return {false, "The viewport is running the model."};
        return {};
    }
    void EndGraphRun() override { ++ends; }
};

gui::MLNode PluginNode(int id, const char* type_id, const char* name) {
    gui::MLNode node{};
    node.id = id;
    node.type = gui::NodeType::PluginCustom;
    node.name = name;
    node.extension_type_id = type_id;
    node.parameters["mjcf_path"] = "robot.xml";
    return node;
}

void CheckSimulationRegistry() {
    auto& simulation = cyxwiz::plugin::PluginSimulationRegistry::Instance();
    FakeSimulation fake;
    for (const auto& info : fake.GetSimulationNodes()) simulation.Register("com.example.sim", info, &fake);
    simulation.Register("com.example.sim", fake.GetSimulationNodes()[0], &fake);  // a duplicate is skipped
    Check(simulation.GetCount() == 2, "both simulation nodes are registered once");
    Check(simulation.IsSimulationNode("com.example.sim:Plant") && !simulation.IsSimulationNode("com.example.sim:Other"),
          "only the named node types run in a simulation");

    std::string reason;
    auto lease = simulation.BeginGraphRun(
        {PluginNode(1, "com.example.sim:Plant", "Plant"), PluginNode(2, "com.example.sim:Probe", "Probe")}, reason);
    Check(lease != nullptr, "a graph of simulation nodes begins: " + reason);
    Check(fake.begins == 1 && fake.nodes.size() == 2 && fake.nodes[0].node_type == "Plant" &&
              fake.nodes[0].node_id == 1 && fake.nodes[0].parameters.at("mjcf_path") == "robot.xml",
          "the plugin begins once, with its nodes and their settings");
    Check(lease->Panels().size() == 1 && lease->Panels()[0].second == "sim.view" &&
              lease->PluginIds().count("com.example.sim") == 1,
          "the run names the plugin and its one panel");
    Check(fake.ends == 0, "the plugin is not ended while the lease is held");
    lease.reset();
    Check(fake.ends == 1, "dropping the lease ends the plugin's run");

    lease = simulation.BeginGraphRun({PluginNode(3, "com.example.other:Thing", "Thing")}, reason);
    Check(lease == nullptr && reason.find("'Thing'") != std::string::npos && fake.begins == 1,
          "a plugin node no plugin runs in a simulation refuses the run, naming it");

    fake.refuse = true;
    lease = simulation.BeginGraphRun({PluginNode(1, "com.example.sim:Plant", "Plant")}, reason);
    Check(lease == nullptr && reason == "The viewport is running the model." && fake.ends == 1,
          "a plugin's refusal is the reason, and a refused plugin is not ended");

    lease = simulation.BeginGraphRun({}, reason);
    Check(lease != nullptr && lease->PluginIds().empty(), "a graph without plugin nodes needs no plugin");

    simulation.RemoveByPlugin("com.example.sim");
    Check(simulation.GetCount() == 0, "removing the plugin removes its simulation nodes");
}

// A plugin's environments (plugin API 1.4): copies of a point on a line.
// The action moves it by half the action; it ends at |x| >= 1 or after
// three steps. Copy i of seed s starts at 0.1 * ((s + i) % 3).
class FakeLine final : public cyxwiz::plugin::IEnvironmentBatch {
public:
    FakeLine(int count, int observation_size, int* releases)
        : count_(count), observation_size_(observation_size), releases_(releases), x_(count), steps_(count) {}
    int Count() const override { return count_; }
    int ObservationSize() const override { return observation_size_; }
    int ActionSize() const override { return 1; }
    bool Reset(uint64_t seed, float* observations) override {
        seed_ = seed;
        for (int i = 0; i < count_; ++i) Start(i, observations);
        return true;
    }
    bool Step(const float* actions, const cyxwiz::plugin::PluginEnvironmentStepBuffers& out) override {
        for (int i = 0; i < count_; ++i) {
            if (!std::isfinite(actions[i])) {
                error_ = "copy " + std::to_string(i) + " got an action that is not a number";
                return false;
            }
            x_[i] += 0.5f * actions[i];
            ++steps_[i];
            out.rewards[i] = -std::abs(x_[i]);
            out.terminated[i] = std::abs(x_[i]) >= 1.0f ? 1 : 0;
            out.truncated[i] = !out.terminated[i] && steps_[i] >= 3 ? 1 : 0;
            out.observations[i] = x_[i];
            if (out.terminated[i] || out.truncated[i]) {
                out.final_observations[i] = x_[i];
                Start(i, out.observations);
            }
        }
        return true;
    }
    std::string LastError() const override { return error_; }
    std::vector<cyxwiz::plugin::PluginEnvironmentStat> Stats() const override {
        return {{"Level", std::to_string(steps_.empty() ? 0 : steps_[0]), "steps of copy 0"}};
    }
    void Release() override {
        ++*releases_;
        delete this;
    }

private:
    void Start(int i, float* observations) {
        x_[i] = 0.1f * static_cast<float>((seed_ + static_cast<uint64_t>(i)) % 3);
        steps_[i] = 0;
        observations[i] = x_[i];
    }
    int count_;
    int observation_size_;
    int* releases_;
    uint64_t seed_ = 0;
    std::vector<float> x_;
    std::vector<int> steps_;
    std::string error_;
};

class FakeEnvironments final : public cyxwiz::plugin::IEnvironmentProvider {
public:
    int releases = 0;
    std::map<std::string, std::string> last_settings;
    std::vector<cyxwiz::plugin::PluginEnvironmentInfo> GetEnvironments() override {
        cyxwiz::plugin::PluginEnvironmentInfo line;
        line.id = "Line-v0";
        line.name = "Line";
        line.observation_size = 1;
        line.action_size = 1;
        line.action_low = {-1.0f};
        line.action_high = {1.0f};
        line.max_episode_steps = 3;
        line.settings = {{"speed", "Speed", "0.5", "How far one action moves the point"}};
        cyxwiz::plugin::PluginEnvironmentSetting wide;  // a flag in a section: it widens the observation
        wide.key = "wide";
        wide.label = "Wide";
        wide.default_value = "0";
        wide.kind = cyxwiz::plugin::PluginSettingKind::Flag;
        wide.section = "More";
        line.settings.push_back(wide);
        cyxwiz::plugin::PluginEnvironmentSetting colours;
        colours.key = "colours";
        colours.label = "Colours";
        colours.kind = cyxwiz::plugin::PluginSettingKind::Choices;
        colours.options = {{"red", "Red"}, {"blue", "Blue"}};
        colours.section = "More";
        line.settings.push_back(colours);
        auto liar = line;  // its copies report another observation size
        liar.id = "Liar-v0";
        auto blind = line;  // observes nothing: not added
        blind.id = "Blind-v0";
        blind.observation_size = 0;
        cyxwiz::plugin::PluginEnvironmentInfo coin;  // one choice is no choice: not added
        coin.id = "Coin-v0";
        coin.observation_size = 1;
        coin.discrete_actions = true;
        coin.action_size = 1;
        coin.action_count = 1;
        return {line, liar, blind, coin};
    }
    bool DescribeEnvironment(const std::string& id, const std::map<std::string, std::string>& settings,
                             cyxwiz::plugin::PluginEnvironmentInfo& info, std::string& error) override {
        for (const auto& listed : GetEnvironments()) {
            if (listed.id != id) continue;
            info = listed;
            const auto wide = settings.find("wide");
            if (wide != settings.end() && wide->second != "0" && wide->second != "1") {
                error = "Wide is 1 or 0.";
                return false;
            }
            if (wide != settings.end() && wide->second == "1") info.observation_size = 2;
            return true;
        }
        error = "No " + id;
        return false;
    }
    cyxwiz::plugin::PluginEnvironmentCreateResult CreateEnvironments(
        const std::string& id, const std::map<std::string, std::string>& settings, int count) override {
        last_settings = settings;
        const auto speed = settings.find("speed");
        if (speed != settings.end() && speed->second == "fast") return {nullptr, "Speed must be a number."};
        const auto wide = settings.find("wide");
        const bool is_wide = wide != settings.end() && wide->second == "1";
        return {new FakeLine(count, id == "Liar-v0" || is_wide ? 2 : 1, &releases), {}};
    }
};

void CheckEnvironmentRegistry() {
    using cyxwiz::plugin::PluginEnvironmentStepBuffers;
    auto& environments = cyxwiz::plugin::PluginEnvironmentRegistry::Instance();
    FakeEnvironments fake;
    auto library = std::make_shared<int>(0);
    std::weak_ptr<int> library_alive = library;
    for (const auto& info : fake.GetEnvironments()) environments.Register("com.example.env", info, &fake, library);
    environments.Register("com.example.env", fake.GetEnvironments()[0], &fake, library);  // a duplicate is skipped
    Check(environments.GetCount() == 3, "the two usable environments are registered once, beside the built-in one");
    cyxwiz::plugin::PluginEnvironmentRegistry::Environment found;
    Check(environments.Find("com.example.env:Line-v0", found) && found.info.max_episode_steps == 3 &&
              found.info.settings.size() == 3 && found.info.settings[0].default_value == "0.5",
          "the environment's info and settings are kept");
    Check(found.info.settings[1].kind == cyxwiz::plugin::PluginSettingKind::Flag && found.info.settings[1].section == "More" &&
              found.info.settings[2].options.size() == 2 && found.info.settings[2].options[1].label == "Blue",
          "a setting's kind, section and options are kept");
    {
        cyxwiz::plugin::PluginEnvironmentRegistry::Environment described;
        std::string why;
        Check(environments.Describe("com.example.env:Line-v0", {{"wide", "1"}}, described, why) &&
                  described.info.observation_size == 2,
              "the description follows the settings (a wider observation)");
        Check(!environments.Describe("com.example.env:Line-v0", {{"wide", "maybe"}}, described, why) &&
                  why == "Wide is 1 or 0.",
              "settings the plugin cannot describe are refused with its reason");
        auto wide_batch = environments.Create("com.example.env:Line-v0", {{"wide", "1"}}, 2, why);
        Check(wide_batch && wide_batch->ObservationSize() == 2, "a batch is checked against its described size: " + why);
        Check(wide_batch && !wide_batch->Stats().empty() && wide_batch->Stats()[0].label == "Level",
              "the batch's stats reach the Engine");
    }
    Check(!environments.Find("com.example.env:Blind-v0", found) && !environments.Find("com.example.env:Coin-v0", found),
          "an environment that observes nothing or has one choice is not added");

    std::string error;
    Check(!environments.Create("com.example.env:Nope", {}, 2, error) && error.find("not available") != std::string::npos,
          "an unknown environment is an error");
    Check(!environments.Create("com.example.env:Line-v0", {}, 0, error), "zero copies is an error");
    Check(!environments.Create("com.example.env:Line-v0", {{"speed", "fast"}}, 2, error) &&
              error == "Speed must be a number.",
          "the plugin's refusal is the error");
    Check(!environments.Create("com.example.env:Liar-v0", {}, 2, error) && fake.releases == 2 &&
              error.find("do not match") != std::string::npos,
          "copies that do not match their description are refused and released");

    auto batch = environments.Create("com.example.env:Line-v0", {{"speed", "0.5"}}, 3, error);
    Check(batch != nullptr, "three copies are created: " + error);
    Check(fake.last_settings.at("speed") == "0.5", "the settings reach the plugin");
    Check(batch->Count() == 3 && batch->ObservationSize() == 1 && batch->ActionSize() == 1, "the batch's sizes");
    std::vector<float> obs(3), rewards(3), final_obs(3, -9.0f);
    std::vector<uint8_t> terminated(3), truncated(3);
    const PluginEnvironmentStepBuffers out{obs.data(), rewards.data(), terminated.data(), truncated.data(),
                                           final_obs.data()};
    Check(batch->Reset(1, obs.data(), error), "reset: " + error);
    Check(obs[0] == 0.1f && obs[1] == 0.2f && obs[2] == 0.0f, "each copy starts from the seed and its index");
    const float push[3] = {2.0f, 0.0f, -2.0f};
    Check(batch->Step(push, out, error), "step: " + error);
    Check(terminated[0] == 1 && terminated[1] == 0 && terminated[2] == 1, "copies 0 and 2 leave the line");
    Check(final_obs[0] == 1.1f && final_obs[2] == -1.0f && final_obs[1] == -9.0f,
          "a copy that ended gives its last observation; the others are left alone");
    Check(obs[0] == 0.1f && obs[2] == 0.0f && obs[1] == 0.2f, "an ended copy starts again in the same step");
    const float still[3] = {0.0f, 0.0f, 0.0f};
    Check(batch->Step(still, out, error) && batch->Step(still, out, error), "two more steps");
    Check(truncated[1] == 1 && terminated[1] == 0, "copy 1 is cut at the step limit");
    const float bad[3] = {0.0f, std::nanf(""), 0.0f};
    Check(!batch->Step(bad, out, error) && error == "copy 1 got an action that is not a number",
          "a failed step says why");

    environments.RemoveByPlugin("com.example.env");
    Check(environments.GetCount() == 1, "removing the plugin removes its environments, not the built-in one");
    library.reset();
    Check(!library_alive.expired(), "a live batch keeps the plugin's library loaded");
    batch.reset();
    Check(fake.releases == 3 && library_alive.expired(), "dropping the batch releases it, then the library");
}

// The Engine's own CartPole-v1 (Gymnasium's classic control), with no plugin
void CheckBuiltinCartPole() {
    using cyxwiz::plugin::PluginEnvironmentStepBuffers;
    auto& environments = cyxwiz::plugin::PluginEnvironmentRegistry::Instance();
    const std::string key = std::string(cyxwiz::plugin::kBuiltinEnvironmentProviderId) + ":CartPole-v1";
    cyxwiz::plugin::PluginEnvironmentRegistry::Environment cartpole;
    Check(environments.Find(key, cartpole), "CartPole-v1 is offered without any plugin");
    Check(cartpole.info.discrete_actions && cartpole.info.action_count == 2 && cartpole.info.observation_size == 4 &&
              cartpole.info.max_episode_steps == 500,
          "CartPole observes 4 values and takes one of 2 actions, 500 steps an episode");
    std::string error;
    Check(!environments.Create(key, {{"speed", "1"}}, 1, error) && error == "Cart Pole has no setting 'speed'.",
          "an unknown CartPole setting is refused");
    Check(!environments.Create(key, {{"max_episode_steps", "-1"}}, 1, error), "a negative step limit is refused");
    auto batch = environments.Create(key, {{"max_episode_steps", "4"}}, 2, error);
    Check(batch != nullptr, "two CartPole copies: " + error);
    std::vector<float> obs(8), final_obs(8), rewards(2);
    std::vector<uint8_t> terminated(2), truncated(2);
    const PluginEnvironmentStepBuffers out{obs.data(), rewards.data(), terminated.data(), truncated.data(),
                                           final_obs.data()};
    Check(batch->Reset(9, obs.data(), error), "reset: " + error);
    bool in_range = true;
    for (const float v : obs) in_range = in_range && v >= -0.05f && v <= 0.05f;
    Check(in_range, "every start value is within 0.05 of rest");
    const auto first = obs;
    Check(batch->Reset(9, obs.data(), error) && obs == first, "the same seed starts the same");
    const float push[2] = {1.0f, 0.0f};
    for (int s = 0; s < 4; ++s) {
        Check(batch->Step(push, out, error), "step: " + error);
        Check(rewards[0] == 1.0f && rewards[1] == 1.0f, "+1 a step");
    }
    Check(truncated[0] == 1 && truncated[1] == 1, "both copies are cut at the four-step limit");
    const float half[2] = {0.5f, 0.0f};
    Check(!batch->Step(half, out, error) && error.find("takes 0 (push left) or 1 (push right)") != std::string::npos,
          "an action that is neither 0 nor 1 is refused");
}

}  // namespace

int main() {
    CheckCommandRegistry();
    CheckPreferencesRegistry();
    CheckPluginSettings();
    CheckSimulationRegistry();
    CheckEnvironmentRegistry();
    CheckBuiltinCartPole();
    const auto dir = PluginDir();
    Check(!dir.empty(), "plugins/examples/image_nodes not found; run from the repository root");
    auto& manager = cyxwiz::plugin::PluginManager::Instance();

    // Unload from the running state: the Plugin Manager's Unload button.
    // Whatever runs the plugin's code (a graph run) hears of it first.
    LoadAndInitialize(manager, dir);
    const uint64_t generation = cyxwiz::ExtensionNodeRegistry::Instance().Generation();
    std::vector<std::string> stopping;
    manager.SetPluginStoppingHandler([&stopping](const std::string& id) { stopping.push_back(id); });
    manager.UnloadPlugin(kPluginId);
    Check(stopping.size() == 1 && stopping[0] == kPluginId, "unloading tells the stopping handler first");
    Check(manager.GetLoadedPlugin(kPluginId) == nullptr, "the plugin is unloaded");
    Check(NodesOfPlugin() == 0, "unloading removes its nodes, " + std::to_string(NodesOfPlugin()) + " left");
    Check(!cyxwiz::plugin::PluginPanelRegistry::Instance().HasPanel(
              "com.cyxwiz.examples.image-nodes.settings"),
          "unloading removes its panel");
    Check(cyxwiz::ExtensionNodeRegistry::Instance().Generation() > generation,
          "the registry generation changes, so the editor refreshes");

    // Load again: nothing left over blocks the second registration.
    LoadAndInitialize(manager, dir);

    // Disable still cleans up as before, and tells the stopping handler too.
    manager.DisablePlugin(kPluginId);
    Check(NodesOfPlugin() == 0, "disabling removes its nodes");
    Check(stopping.size() == 2, "disabling tells the stopping handler");
    manager.SetPluginStoppingHandler(nullptr);
    manager.UnloadPlugin(kPluginId);
    Check(NodesOfPlugin() == 0, "unloading a disabled plugin leaves nothing");

    std::cout << "test_plugin_unload_registrations: all checks passed\n";
    return 0;
}
