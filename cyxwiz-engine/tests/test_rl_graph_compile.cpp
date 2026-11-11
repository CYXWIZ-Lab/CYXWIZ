// A reinforcement learning graph compiles (TOFIX195 S4.3): Environment ->
// Dense/Tanh layers -> Policy Head / Value Head -> PPO gives the RL training
// configuration (environment, both layer chains, PPO settings), and an
// incomplete or wrong graph says what is wrong (plugin not loaded, a head not
// reached, a minibatch larger than the rollout, a Softmax before the Policy
// Head, data nodes mixed in, the wrong Environment pin, a layer the networks
// cannot use). Also the Properties presentation (facts, heads, presets) and
// the SB3 reference lookup of the RL Training Dashboard.

#include "../src/core/graph_compiler.h"
#include "../src/core/graph_node_factory.h"
#include "../src/core/pipeline_runtime_capabilities.h"
#include "../src/core/rl_graph_compile.h"
#include "../src/core/rl_graph_presentation.h"
#include "../src/core/rl/ppo_reference.h"
#include "rl_graph_test_support.h"

#include <fstream>
#include <sstream>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using gui::NodeType;
using rl_graph_test::Graph;
using rl_graph_test::Ids;
using rl_graph_test::Pattern;

int failures = 0;

void Check(bool ok, const std::string& what) {
    if (ok) return;
    std::printf("FAIL: %s\n", what.c_str());
    ++failures;
}

bool HasIssue(const cyxwiz::TrainingConfiguration& config, cyxwiz::IssueLevel level, const std::string& text) {
    for (const auto& i : config.issues)
        if (i.level == level && i.message.find(text) != std::string::npos) return true;
    return false;
}

std::string Issues(const cyxwiz::TrainingConfiguration& config) {
    std::string all;
    for (const auto& i : config.issues) all += " | " + i.node_name + ": " + i.message;
    return all;
}

cyxwiz::plugin::PluginEnvironmentInfo CartPole() {
    cyxwiz::plugin::PluginEnvironmentInfo info;
    info.id = "CartPole-v1";
    info.name = "Cart Pole";
    info.observation_size = 4;
    info.discrete_actions = true;
    info.action_size = 1;
    info.action_count = 2;
    info.max_episode_steps = 500;
    info.step_seconds = 0.02;
    return info;
}

cyxwiz::plugin::PluginEnvironmentInfo Pendulum() {
    cyxwiz::plugin::PluginEnvironmentInfo info;
    info.id = "InvertedPendulum-v5";
    info.observation_size = 4;
    info.action_size = 1;
    info.action_low = {-3.0f};
    info.action_high = {3.0f};
    info.max_episode_steps = 1000;
    info.step_seconds = 0.04;
    return info;
}

cyxwiz::TrainingConfiguration Compile(const Graph& g) {
    cyxwiz::GraphCompiler compiler;
    return compiler.Compile(g.nodes, g.links, true);
}

void CheckValid() {
    Ids id{};
    Graph g = Pattern(id);
    Check(cyxwiz::HasTrainingGraphStructure(g.nodes), "the toolbar lets Train start an RL graph");
    const auto config = Compile(g);
    Check(config.is_valid && config.rl, "the PPO pattern compiles:" + Issues(config));
    if (!config.rl) return;
    const auto& rl = *config.rl;
    Check(rl.environment_key == "cyxwiz.builtin:CartPole-v1" && rl.copies == 8, "the Environment's key and copies");
    Check(rl.observation_size == 4 && rl.policy_outputs == 2 && rl.environment.discrete_actions,
          "CartPole: 4 observations, 2 logits");
    const auto types = [](const std::vector<cyxwiz::CompiledLayer>& layers) {
        std::vector<NodeType> out;
        for (const auto& l : layers) out.push_back(l.type);
        return out;
    };
    const std::vector<NodeType> mlp = {NodeType::Dense, NodeType::Tanh, NodeType::Dense, NodeType::Tanh};
    Check(types(rl.policy_layers) == mlp && types(rl.value_layers) == mlp, "both chains are Dense Tanh Dense Tanh");
    Check(rl.policy_layers.size() == 4 && rl.policy_layers[0].units == 64, "Dense keeps its units");
    Check(rl.ppo.n_steps == 2048 && rl.ppo.batch_size == 64 && rl.ppo.n_epochs == 10 && rl.ppo.gamma == 0.99f &&
              rl.ppo.loss.clip_range == 0.2f && !rl.ppo.normalize_observations && std::isnan(rl.ppo.target_return),
          "PPO's defaults are Stable-Baselines3's");
    Check(rl.ppo_node_id == id.ppo && rl.policy_head_node_id == id.phead && rl.value_head_node_id == id.vhead &&
              rl.environment_node_id == id.env,
          "the node ids are kept");

    // Tuned for CartPole-v1, as the PPO node's button writes it
    const auto preset = cyxwiz::rl_graph::TunedPpoSettings("CartPole-v1");
    Check(preset.has_value(), "CartPole-v1 has tuned settings");
    if (!preset) return;
    for (const auto& [k, v] : preset->ppo) g.At(id.ppo).parameters[k] = v;
    for (const auto& [k, v] : preset->environment) g.At(id.env).parameters[k] = v;
    g.At(id.env).parameters[std::string(cyxwiz::rl_graph::kSettingPrefix) + "max_episode_steps"] = "300";
    const auto tuned = Compile(g);
    Check(tuned.is_valid && tuned.rl && tuned.rl->ppo.n_steps == 32 && tuned.rl->ppo.batch_size == 256 &&
              tuned.rl->ppo.n_epochs == 20 && tuned.rl->ppo.learning_rate == 0.001f && tuned.rl->ppo.target_return == 475.0f,
          "the tuned CartPole settings compile:" + Issues(tuned));
    Check(tuned.rl && tuned.rl->environment_settings.at("max_episode_steps") == "300",
          "the Environment's plugin settings reach the configuration");
}

void CheckErrors() {
    Ids id{};
    {
        Graph g = Pattern(id);
        g.At(id.env).parameters["environment"] = "com.cyxwiz.simulation.mujoco:HalfCheetah-v5";
        const auto config = Compile(g);
        Check(!config.is_valid && !config.rl &&
                  HasIssue(config, cyxwiz::IssueLevel::Error,
                           "HalfCheetah-v5 comes from the plugin com.cyxwiz.simulation.mujoco, which is not loaded"),
              "an environment of a plugin that is not loaded is refused:" + Issues(config));
    }
    {
        Graph g = Pattern(id);
        g.Unlink(id.p2, id.pa2);
        const auto config = Compile(g);
        Check(!config.is_valid && HasIssue(config, cyxwiz::IssueLevel::Error, "Nothing reaches the Policy Head"),
              "a broken policy chain is refused:" + Issues(config));
    }
    {
        Graph g = Pattern(id);
        g.At(id.env).parameters["copies"] = "1";
        g.At(id.ppo).parameters["n_steps"] = "512";
        g.At(id.ppo).parameters["batch_size"] = "600";
        const auto config = Compile(g);
        Check(!config.is_valid &&
                  HasIssue(config, cyxwiz::IssueLevel::Error,
                           "Minibatch size 600 is larger than one rollout (1 copy x 512 steps = 512)"),
              "a minibatch larger than the rollout is refused:" + Issues(config));
    }
    {
        Graph g = Pattern(id);
        g.Unlink(id.pa2, id.phead);
        const int soft = g.Add(NodeType::Softmax, "Softmax");
        g.Link(id.pa2, soft);
        g.Link(soft, id.phead);
        const auto config = Compile(g);
        Check(config.is_valid && HasIssue(config, cyxwiz::IssueLevel::Warning, "A Softmax right before the Policy Head"),
              "a Softmax before the Policy Head warns but compiles:" + Issues(config));
    }
    {
        Graph g = Pattern(id);
        g.Add(NodeType::DataInput, "Data Input");
        const auto config = Compile(g);
        Check(!config.is_valid && HasIssue(config, cyxwiz::IssueLevel::Error, "a graph trains either that or a policy"),
              "a Data Input in an RL graph is refused:" + Issues(config));
    }
    {
        Graph g = Pattern(id);
        g.Unlink(id.env, id.ppo);
        g.Link(id.env, id.ppo, 0, 1);  // the Observation into PPO's Environment
        const auto config = Compile(g);
        Check(!config.is_valid && HasIssue(config, cyxwiz::IssueLevel::Error, "not its Observation"),
              "the Observation in PPO's Environment is refused:" + Issues(config));
    }
    {
        Graph g = Pattern(id);
        g.Unlink(id.v1, id.va1);
        const int conv = g.Add(NodeType::Conv2D, "Conv");
        g.Link(id.v1, conv);
        g.Link(conv, id.va1);
        const auto config = Compile(g);
        Check(!config.is_valid && HasIssue(config, cyxwiz::IssueLevel::Error, "cannot be in a policy or value network"),
              "a Conv2D in the value network is refused:" + Issues(config));
    }
    {
        Graph g = Pattern(id);
        g.Unlink(id.vhead, id.ppo);
        const auto config = Compile(g);
        Check(!config.is_valid && HasIssue(config, cyxwiz::IssueLevel::Error, "Connect a Value Head to PPO's Value"),
              "PPO without a Value Head is refused:" + Issues(config));
    }
    {
        Graph g = Pattern(id);
        g.At(id.ppo).parameters["learning_rate"] = "fast";
        const auto config = Compile(g);
        Check(!config.is_valid && HasIssue(config, cyxwiz::IssueLevel::Error, "PPO's 'learning_rate' is not a number"),
              "a setting that is not a number is refused:" + Issues(config));
    }
}

void CheckPresentation() {
    using namespace cyxwiz::rl_graph;
    const auto facts = EnvironmentFacts(CartPole());
    Check(facts.size() == 4 && facts[0].value == "4 values" && facts[1].value == "2 choices" &&
              facts[2].value == "500 steps" && facts[3].value == "0.02 s",
          "CartPole's facts");
    const auto pendulum = EnvironmentFacts(Pendulum());
    Check(pendulum.size() == 4 && pendulum[1].value == "1 value in [-3, 3]", "the pendulum's actions and range");
    const auto head = PolicyHeadFacts(CartPole(), 64);
    Check(head.size() == 3 && head[0].value == "Categorical, 2 choices" && head[1].value == "64 -> 2 logits (added)",
          "the Policy Head adds 64 -> 2 logits");
    const auto gaussian = PolicyHeadFacts(Pendulum(), 0);
    Check(gaussian[0].value == "Gaussian, 1 action" && gaussian[1].value == "1 mean (added)",
          "a continuous Policy Head adds a mean");
    Check(ValueHeadFacts(64)[0].value == "64 -> 1 (added)", "the Value Head adds 64 -> 1");

    const std::vector<ListedEnvironment> listed = {{"cyxwiz.builtin:CartPole-v1", "CyxWiz (built in)", CartPole()}};
    const auto choices = EnvironmentChoices(listed, "com.cyxwiz.simulation.mujoco:HalfCheetah-v5");
    Check(choices.size() == 2 && choices[0].label == "com.cyxwiz.simulation.mujoco:HalfCheetah-v5 · not loaded" &&
              choices[1].label == "CartPole-v1 · CyxWiz (built in)",
          "a saved environment whose plugin is not loaded stays listed, marked");
    Check(StoredSettings({{"setting.frame_skip", "5"}, {"setting.empty", ""}, {"copies", "8"}}) ==
              std::map<std::string, std::string>{{"frame_skip", "5"}},
          "the plugin settings are the setting.* parameters with a value");

    Ids id{};
    Graph g = Pattern(id);
    Check(UpstreamEnvironment(g.nodes, g.links, id.phead) && UpstreamEnvironment(g.nodes, g.links, id.phead)->id == id.env,
          "the Policy Head finds its Environment");
    Check(UpstreamEnvironment(g.nodes, g.links, id.ppo) != nullptr, "PPO finds its Environment");
    Check(UpstreamUnits(g.nodes, g.links, id.phead) == 64 && UpstreamUnits(g.nodes, g.links, id.vhead) == 64,
          "the heads see 64 features");
    Check(TunedPpoSettings("HalfCheetah-v5").has_value() &&
              TunedPpoSettings("HalfCheetah-v5")->ppo.at("normalize_observations") == "true" &&
              TunedPpoSettings("HalfCheetah-v5")->policy_head.at("log_std_init") == "-2" &&
              TunedPpoSettings("InvertedPendulum-v5").has_value() && !TunedPpoSettings("Nope-v0").has_value(),
          "tuned settings for the three gate tasks only");
}

// The recorded Stable-Baselines3 curves the dashboard draws under a run of the same settings
void CheckReference() {
    std::ifstream in(CYXWIZ_PPO_REFERENCE_FILE, std::ios::binary);
    std::ostringstream text;
    text << in.rdbuf();
    Check(!text.str().empty(), "the recorded SB3 curves are readable");
    cyxwiz::rl::PpoConfig cartpole;  // test_ppo_trainer's RL Zoo settings
    cartpole.n_steps = 32;
    cartpole.batch_size = 256;
    cartpole.n_epochs = 20;
    cartpole.learning_rate = 1e-3f;
    cartpole.gamma = 0.98f;
    cartpole.gae_lambda = 0.8f;
    const auto ref = cyxwiz::rl::FindPpoReference(text.str(), "CartPole-v1", 8, cartpole);
    Check(ref.has_value() && ref->seeds.size() == 3 && ref->label.find("3 seeds") != std::string::npos,
          "CartPole's reference: 3 seeds");
    if (ref) {
        const auto at = ref->At(70000.0);
        Check(at && *at > 400.0 && *at <= 500.0, "the seeds' mean near 70,000 steps is above 400");
        Check(!ref->At(1e9).has_value(), "no reference past the recorded steps");
    }
    auto other = cartpole;
    other.n_steps = 64;
    Check(!cyxwiz::rl::FindPpoReference(text.str(), "CartPole-v1", 8, other).has_value() &&
              !cyxwiz::rl::FindPpoReference(text.str(), "CartPole-v1", 4, cartpole).has_value(),
          "other settings or copies have no reference");
    cartpole.normalize_observations = true;
    Check(!cyxwiz::rl::FindPpoReference(text.str(), "CartPole-v1", 8, cartpole).has_value(),
          "normalised observations have no reference");
}

}  // namespace

int main() {
    cyxwiz::SetRLEnvironmentLookup([](const std::string& key, cyxwiz::plugin::PluginEnvironmentInfo& info) {
        if (key != "cyxwiz.builtin:CartPole-v1") return false;
        info = CartPole();
        return true;
    });
    CheckValid();
    CheckErrors();
    CheckPresentation();
    CheckReference();
    if (failures == 0) std::printf("test_rl_graph_compile: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
