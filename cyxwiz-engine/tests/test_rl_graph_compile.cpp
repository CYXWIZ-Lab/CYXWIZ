// A reinforcement learning graph compiles (TOFIX195 S4.3): Environment ->
// Dense/Tanh layers -> Policy Head / Value Head -> PPO gives the RL training
// configuration (environment, both layer chains, PPO settings), and an
// incomplete or wrong graph says what is wrong (plugin not loaded, a head not
// reached, a minibatch larger than the rollout, a Softmax before the Policy
// Head, data nodes mixed in, the wrong Environment pin, a layer the networks
// cannot use). Also the Properties presentation (facts, heads, presets) and
// the SB3 reference lookup of the RL Training Dashboard. SAC (TOFIX216): the
// same shape with a Q Head, its settings, its refusals (a choice of actions,
// a batch larger than the buffer, a Q branch not starting with a Dense, two
// learners) and its presentation (Q Head facts, replay memory, the preset).
// The shipped examples (examples/cyxgraph/rl) open as File > Open opens a
// pattern file and compile; the SAC ones to the stability gate's settings,
// with their recorded SB3 reference.

#include "../src/core/graph_compiler.h"
#include "../src/core/graph_node_factory.h"
#include "../src/core/pipeline_runtime_capabilities.h"
#include "../src/core/rl_graph_compile.h"
#include "../src/core/rl_graph_presentation.h"
#include "../src/core/rl/rl_reference.h"
#include "../src/gui/node_type_import_registry.h"
#include "rl_graph_test_support.h"

#include <cyxwiz/neural_provider.h>

#include <filesystem>
#include <fstream>
#include <map>
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

cyxwiz::plugin::PluginEnvironmentInfo Cheetah() {
    cyxwiz::plugin::PluginEnvironmentInfo info;
    info.id = "HalfCheetah-v5";
    info.observation_size = 17;
    info.action_size = 6;
    info.action_low.assign(6, -1.0f);
    info.action_high.assign(6, 1.0f);
    info.max_episode_steps = 1000;
    info.step_seconds = 0.05;
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
    Check(rl.learner_node_id == id.ppo && rl.policy_head_node_id == id.phead && rl.value_head_node_id == id.vhead &&
              rl.environment_node_id == id.env,
          "the node ids are kept");

    // Tuned for CartPole-v1, as the PPO node's button writes it
    const auto preset = cyxwiz::rl_graph::TunedPpoSettings("CartPole-v1");
    Check(preset.has_value(), "CartPole-v1 has tuned settings");
    if (!preset) return;
    for (const auto& [k, v] : preset->learner) g.At(id.ppo).parameters[k] = v;
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
        Check(!config.is_valid && HasIssue(config, cyxwiz::IssueLevel::Error, "cannot be in a policy, value or Q network"),
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
              TunedPpoSettings("HalfCheetah-v5")->learner.at("normalize_observations") == "true" &&
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

    // HalfCheetah's reference normalises (VecNormalize), as its tuned settings do
    cyxwiz::rl::PpoConfig cheetah;
    cheetah.n_steps = 512;
    cheetah.batch_size = 64;
    cheetah.n_epochs = 20;
    cheetah.learning_rate = 2.0633e-05f;
    cheetah.gamma = 0.98f;
    cheetah.gae_lambda = 0.92f;
    cheetah.loss.clip_range = 0.1f;
    cheetah.loss.ent_coef = 0.000401762f;
    cheetah.loss.vf_coef = 0.58096f;
    cheetah.max_grad_norm = 0.8f;
    cheetah.normalize_observations = cheetah.normalize_rewards = true;
    Check(cyxwiz::rl::FindPpoReference(text.str(), "HalfCheetah-v5", 1, cheetah).has_value(),
          "HalfCheetah's normalised run has its normalised reference");
    cheetah.normalize_rewards = false;
    Check(!cyxwiz::rl::FindPpoReference(text.str(), "HalfCheetah-v5", 1, cheetah).has_value(),
          "a run normalising only the observations has none");
}

// The SAC reference matches SB3's SAC defaults for the keys a recorded task
// leaves out (generate_sac_sb3_curves.py's layout)
void CheckSacReference() {
    const std::string text = R"({"versions": {"stable_baselines3": "2.7.1"},
        "columns": ["time/total_timesteps", "rollout/ep_rew_mean", "train/critic_loss", "train/actor_loss",
                    "train/ent_coef"],
        "tasks": [
          {"id": "InvertedPendulum-v5", "copies": 1, "sac": {},
           "runs": [{"seed": 1, "updates": [[1000, 10.0, 0.5, -5.0, 0.8], [2000, 30.0, 0.4, -9.0, 0.6]]},
                    {"seed": 2, "updates": [[1000, 20.0, 0.5, -5.0, 0.8], [2000, null, 0.4, -9.0, 0.6]]}]},
          {"id": "HalfCheetah-v5", "copies": 1, "sac": {"learning_starts": 10000},
           "runs": [{"seed": 1, "updates": [[11000, -280.0, 1.0, -20.0, 0.5]]}]}]})";
    cyxwiz::rl::SacConfig defaults;  // the SAC node's: SB3's
    const auto pendulum = cyxwiz::rl::FindSacReference(text, "InvertedPendulum-v5", 1, defaults);
    Check(pendulum && pendulum->seeds.size() == 2 && pendulum->label == "SB3 2.7.1, 2 seeds",
          "SAC: the pendulum's reference at SB3's defaults, 2 seeds");
    if (pendulum) {
        Check(pendulum->At(1000.0) == 15.0, "SAC: the seeds' mean at 1,000 steps");
        Check(!pendulum->At(2000.0).has_value(), "SAC: a seed without a return there leaves no mean");
    }
    Check(!cyxwiz::rl::FindSacReference(text, "HalfCheetah-v5", 1, defaults).has_value(),
          "SAC: HalfCheetah's reference starts learning at 10,000, not the default 100");
    auto cheetah = defaults;
    cheetah.learning_starts = 10000;
    Check(cyxwiz::rl::FindSacReference(text, "HalfCheetah-v5", 1, cheetah).has_value(),
          "SAC: RL Zoo's HalfCheetah settings have its reference");
    auto other = defaults;
    other.ent_coef_auto = false;
    Check(!cyxwiz::rl::FindSacReference(text, "InvertedPendulum-v5", 1, other).has_value(),
          "SAC: a fixed entropy weight has no reference");
    other = defaults;
    other.buffer_size = 50000;
    Check(!cyxwiz::rl::FindSacReference(text, "InvertedPendulum-v5", 1, other).has_value(),
          "SAC: another replay buffer size has no reference");
    Check(!cyxwiz::rl::FindSacReference(text, "InvertedPendulum-v5", 2, defaults).has_value(),
          "SAC: other copies have no reference");

    // The recorded runs are on the environment's defaults
    cyxwiz::plugin::PluginEnvironmentInfo env;
    cyxwiz::plugin::PluginEnvironmentSetting throws;
    throws.key = "throw_every";
    throws.default_value = "0";
    cyxwiz::plugin::PluginEnvironmentSetting ground;
    ground.key = "grounds";
    ground.default_value = "";
    env.settings = {throws, ground};
    Check(cyxwiz::rl::DefaultEnvironmentSettings(env, {}) &&
              cyxwiz::rl::DefaultEnvironmentSettings(env, {{"throw_every", "0.0"}, {"grounds", ""}}),
          "no settings, or the defaults (0.0 as 0), are the recorded environment");
    Check(!cyxwiz::rl::DefaultEnvironmentSettings(env, {{"throw_every", "1"}}) &&
              !cyxwiz::rl::DefaultEnvironmentSettings(env, {{"grounds", "rough"}}) &&
              !cyxwiz::rl::DefaultEnvironmentSettings(env, {{"unknown", "0"}}),
          "throws, a ground or an undeclared key are not (the dashboard draws no reference)");

    // The recorded file: the pendulum at the SAC node's defaults, HalfCheetah at RL Zoo's settings
    std::ifstream in(CYXWIZ_SAC_REFERENCE_FILE, std::ios::binary);
    std::ostringstream recorded;
    recorded << in.rdbuf();
    Check(!recorded.str().empty(), "SAC: the recorded SB3 curves are readable");
    const auto pendulum_file = cyxwiz::rl::FindSacReference(recorded.str(), "InvertedPendulum-v5", 1, defaults);
    Check(pendulum_file && pendulum_file->seeds.size() == 3, "SAC: the pendulum's recorded reference, 3 seeds");
    if (pendulum_file) {
        const auto at = pendulum_file->At(120000.0);
        Check(at && *at > 900.0 && *at <= 1000.0, "SAC: the pendulum's seeds' mean near 120,000 steps is above 900");
    }
    const auto cheetah_file = cyxwiz::rl::FindSacReference(recorded.str(), "HalfCheetah-v5", 1, cheetah);
    Check(cheetah_file && cheetah_file->seeds.size() == 3, "SAC: HalfCheetah's recorded reference, 3 seeds");
    if (cheetah_file) {
        const auto at = cheetah_file->At(1000000.0);
        Check(at && *at > 5000.0, "SAC: HalfCheetah's seeds' mean at 1,000,000 steps is above 5,000");
    }
}

// A pattern-format graph file (a "template" of nodes by string id and links
// by pin index), built as the editor's LoadPatternAsGraph builds it
bool LoadExample(const std::filesystem::path& path, Graph& g, std::string& error) {
    std::ifstream in(path, std::ios::binary);
    nlohmann::json doc;
    try {
        in >> doc;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
    std::map<std::string, int> ids;
    for (const auto& n : doc.at("template").at("nodes")) {
        const auto type = gui::ResolveNodeTypeImportName(n.at("type").get<std::string>());
        if (!type) {
            error = "unknown node type " + n.at("type").get<std::string>();
            return false;
        }
        const int id = g.Add(*type, n.value("name", n.at("type").get<std::string>()));
        ids[n.at("id").get<std::string>()] = id;
        if (n.contains("params"))
            for (const auto& [k, v] : n.at("params").items()) g.At(id).parameters[k] = v.get<std::string>();
        // The MuJoCo plugin is not loaded here: its tasks resolve to the test's facts
        const std::string mujoco = "com.cyxwiz.simulation.mujoco:";
        const auto environment = g.At(id).parameters.find("environment");
        if (*type == NodeType::RLEnvironment && environment != g.At(id).parameters.end() &&
            environment->second.rfind(mujoco, 0) == 0)
            environment->second = "test.continuous:" + environment->second.substr(mujoco.size());
    }
    for (const auto& l : doc.at("template").at("links")) {
        const auto from = ids.find(l.at("from").get<std::string>());
        const auto to = ids.find(l.at("to").get<std::string>());
        if (from == ids.end() || to == ids.end()) {
            error = "a link to a node the graph does not have";
            return false;
        }
        g.Link(from->second, to->second, l.value("from_pin", 0), l.value("to_pin", 0));
    }
    return true;
}

void CheckExamples() {
    int sac_examples = 0;
    std::ifstream sac_file(CYXWIZ_SAC_REFERENCE_FILE, std::ios::binary);
    std::ostringstream sac_reference;
    sac_reference << sac_file.rdbuf();
    for (const auto& entry : std::filesystem::directory_iterator(CYXWIZ_RL_EXAMPLES_DIR)) {
        if (entry.path().extension() != ".cyxgraph") continue;
        const std::string name = entry.path().filename().string();
        Graph g;
        std::string error;
        Check(LoadExample(entry.path(), g, error), name + " loads: " + error);
        const auto config = Compile(g);
        Check(config.is_valid && config.rl, name + " compiles:" + Issues(config));
        if (!config.rl || config.rl->algorithm != cyxwiz::RLAlgorithm::SAC) continue;
        ++sac_examples;
        const auto& rl = *config.rl;
        const auto& c = rl.sac;
        // The stability gate's runs: one copy, 256-256 ReLU, InvertedPendulum
        // at the defaults to 950 within 150,000 steps, HalfCheetah at RL Zoo's
        Check(rl.copies == 1 && rl.policy_layers.size() == 4 && rl.q_layers.size() == 4 &&
                  rl.policy_layers[1].type == NodeType::ReLU,
              name + ": one copy, two layers of 256 with ReLU");
        if (rl.environment.id == "InvertedPendulum-v5")
            Check(c.total_timesteps == 150000 && c.target_return == 950.0f && c.learning_starts == 100,
                  name + ": the defaults, stopping at 950 within 150,000 steps");
        else
            Check(rl.environment.id == "HalfCheetah-v5" && c.total_timesteps == 1000000 &&
                      c.learning_starts == 10000 && std::isnan(c.target_return),
                  name + ": RL Zoo's HalfCheetah settings, all 1,000,000 steps");
        const auto ref = cyxwiz::rl::FindSacReference(sac_reference.str(), rl.environment.id, rl.copies, c);
        Check(ref && ref->seeds.size() == 3, name + " has the recorded SB3 reference on the dashboard");
    }
    Check(sac_examples == 2, "the two SAC examples are checked");
}

void CheckSac() {
    using namespace cyxwiz::rl_graph;
    const std::string pendulum_key = "test.continuous:InvertedPendulum-v5";
    Ids id{};
    {
        Graph g = Pattern(id, 256, NodeType::ReLU, NodeType::SAC);
        g.At(id.env).parameters["environment"] = pendulum_key;
        g.At(id.env).parameters["copies"] = "1";
        Check(cyxwiz::HasTrainingGraphStructure(g.nodes), "the toolbar lets Train start a SAC graph");
        const auto config = Compile(g);
        Check(config.is_valid && config.rl, "the SAC pattern compiles:" + Issues(config));
        if (!config.rl) return;
        const auto& rl = *config.rl;
        const std::vector<NodeType> mlp = {NodeType::Dense, NodeType::ReLU, NodeType::Dense, NodeType::ReLU};
        std::vector<NodeType> q;
        for (const auto& l : rl.q_layers) q.push_back(l.type);
        Check(rl.algorithm == cyxwiz::RLAlgorithm::SAC && q == mlp && rl.value_layers.empty() &&
                  rl.policy_layers.size() == 4 && rl.policy_outputs == 1,
              "SAC: a policy chain, a Q chain, no value chain, 1 action");
        Check(rl.learner_node_id == id.ppo && rl.q_head_node_id == id.vhead && rl.value_head_node_id == -1,
              "the SAC and Q Head ids are kept");
        const auto& c = rl.sac;
        Check(c.total_timesteps == 1000000 && c.buffer_size == 1000000 && c.learning_starts == 100 &&
                  c.batch_size == 256 && c.learning_rate == 3e-4f && c.gamma == 0.99f && c.tau == 0.005f &&
                  c.train_freq == 1 && c.gradient_steps == 1 && c.target_update_interval == 1 && c.ent_coef_auto &&
                  c.ent_coef == 1.0f && c.target_entropy_auto && c.target_entropy == -1.0f &&
                  !c.normalize_observations && std::isnan(c.target_return),
              "SAC's defaults are Stable-Baselines3's, the target entropy minus the actions");
        // The actor and the Q chain on [observation | action], at SAC's batch
        Check(!cyxwiz::FusedNetworksExpected(rl, cyxwiz::DeviceType::CPU, 0), "SAC: no fused kernels on the CPU");
        const bool cuda_provider =
            !cyxwiz::NeuralProviderRegistry::Instance().ListServing({cyxwiz::DeviceType::CUDA, 0}).empty();
        Check(cyxwiz::FusedNetworksExpected(rl, cyxwiz::DeviceType::CUDA, 0) == cuda_provider,
              "SAC's actor and Q networks are expected fused on CUDA exactly when a provider serves CUDA");
        if (cuda_provider) {
            auto huge = rl;  // a batch past the backward's rows
            huge.sac.batch_size = 70000;
            Check(!cyxwiz::FusedNetworksExpected(huge, cyxwiz::DeviceType::CUDA, 0),
                  "a SAC batch past the backward's 65,536 rows is not expected fused");
        }

        g.At(id.ppo).parameters["ent_coef_mode"] = "fixed";
        g.At(id.ppo).parameters["ent_coef"] = "0.2";
        g.At(id.ppo).parameters["target_entropy_mode"] = "typed";
        g.At(id.ppo).parameters["target_entropy"] = "-3";
        g.At(id.ppo).parameters["stop_at_return"] = "true";
        g.At(id.ppo).parameters["target_return"] = "950";
        const auto fixed = Compile(g);
        Check(fixed.rl && !fixed.rl->sac.ent_coef_auto && fixed.rl->sac.ent_coef == 0.2f &&
                  !fixed.rl->sac.target_entropy_auto && fixed.rl->sac.target_entropy == -3.0f &&
                  fixed.rl->sac.target_return == 950.0f,
              "a fixed entropy weight, a typed target entropy and a target return are read");
    }
    {
        Graph g = Pattern(id, 256, NodeType::ReLU, NodeType::SAC);  // CartPole: a choice of actions
        const auto config = Compile(g);
        Check(!config.is_valid && HasIssue(config, cyxwiz::IssueLevel::Error, "SAC learns continuous actions"),
              "SAC on a choice of actions is refused:" + Issues(config));
    }
    {
        Graph g = Pattern(id, 256, NodeType::ReLU, NodeType::SAC);
        g.At(id.env).parameters["environment"] = pendulum_key;
        g.Unlink(id.vhead, id.ppo);
        g.Unlink(id.va2, id.vhead);
        const int value = g.Add(NodeType::ValueHead, "Value Head");
        g.Link(id.va2, value);
        g.Link(value, id.ppo, 0, 2);
        const auto config = Compile(g);
        Check(!config.is_valid && HasIssue(config, cyxwiz::IssueLevel::Error, "Connect a Q Head to SAC's Q"),
              "a Value Head into SAC's Q is refused:" + Issues(config));
    }
    {
        Graph g = Pattern(id, 256, NodeType::ReLU, NodeType::SAC);
        g.At(id.env).parameters["environment"] = pendulum_key;
        g.At(id.ppo).parameters["buffer_size"] = "100";
        const auto config = Compile(g);
        Check(!config.is_valid && HasIssue(config, cyxwiz::IssueLevel::Error,
                                           "Batch size 256 is larger than the replay buffer (100 steps)"),
              "a batch larger than the buffer is refused:" + Issues(config));
    }
    {
        Graph g = Pattern(id, 256, NodeType::ReLU, NodeType::SAC);
        g.At(id.env).parameters["environment"] = pendulum_key;
        g.Unlink(id.env, id.v1);
        const int norm = g.Add(NodeType::LayerNorm, "Layer Norm");
        g.Link(id.env, norm);
        g.Link(norm, id.v1);
        const auto config = Compile(g);
        Check(!config.is_valid && HasIssue(config, cyxwiz::IssueLevel::Error, "Start the Q layers with a Dense"),
              "a Q branch not starting with a Dense is refused:" + Issues(config));
    }
    {
        Graph g = Pattern(id, 256, NodeType::ReLU, NodeType::SAC);
        g.At(id.env).parameters["environment"] = pendulum_key;
        g.At(id.ppo).parameters["tau"] = "0";
        g.At(id.ppo).parameters["learning_starts"] = "-5";
        const auto config = Compile(g);
        Check(!config.is_valid &&
                  HasIssue(config, cyxwiz::IssueLevel::Error,
                           "The discount and the target update must each be in (0, 1]") &&
                  HasIssue(config, cyxwiz::IssueLevel::Error, "the random steps first at least 0"),
              "a target update of 0 and negative random steps are refused:" + Issues(config));
    }
    {
        Graph g = Pattern(id, 256, NodeType::ReLU, NodeType::SAC);
        g.At(id.env).parameters["environment"] = pendulum_key;
        g.Add(NodeType::PPO, "PPO");
        const auto config = Compile(g);
        Check(!config.is_valid && HasIssue(config, cyxwiz::IssueLevel::Error, "keep one PPO, SAC or TD3 node"),
              "two learners are refused:" + Issues(config));
    }

    // Presentation
    const auto q = QHeadFacts(256, 6);
    Check(q.size() == 3 && q[0].value == "256 -> 1 (added)" && q[1].value.find("6 values") != std::string::npos,
          "the Q Head adds 256 -> 1 and takes the 6 actions at the first Q layer");
    const auto squashed = PolicyHeadFacts(Pendulum(), 256, NodeType::SAC);
    Check(squashed.size() == 3 && squashed[0].value == "Gaussian squashed by tanh, 1 action" &&
              squashed[1].value == "256 -> 1 mean and 256 -> 1 log std (added)",
          "a SAC Policy Head adds a mean and a log std per action");
    Check(ReplayMemoryText(cyxwiz::rl::ReplayBytes(1000000, 17, 6)) == "about 168 MB" &&
              ReplayMemoryText(cyxwiz::rl::ReplayBytes(10000000, 17, 6)) == "about 1.7 GB" &&
              cyxwiz::rl::ReplayBytes(0, 17, 6) == 0,
          "HalfCheetah's replay buffer of 1M steps is about 168 MB");
    {
        const uint64_t gpu = 4294967296ull;  // a 4 GiB card
        const auto fits = ReplayFitText(cyxwiz::rl::ReplayBytes(1000000, 17, 6), "NVIDIA GeForce GTX 1050 Ti", gpu);
        Check(fits.text == "about 168 MB, 4 % of the NVIDIA GeForce GTX 1050 Ti's 4.3 GB" && fits.advice.empty(),
              "the buffer against the device's memory");
        const auto tight = ReplayFitText(cyxwiz::rl::ReplayBytes(20000000, 17, 6), "GPU", gpu);
        Check(tight.text == "about 3.4 GB, 78 % of the GPU's 4.3 GB" && !tight.advice.empty(),
              "a buffer that leaves the networks little room says so");
        const auto over = ReplayFitText(cyxwiz::rl::ReplayBytes(30000000, 17, 6), "", gpu);
        Check(over.text == "about 5 GB, more than the training device's 4.3 GB" &&
                  over.advice == "The replay buffer does not fit the training device: lower Buffer size.",
              "a buffer the device cannot hold");
        Check(ReplayFitText(1000, "x", 0).text == "about 1 MB on the training device" &&
                  ReplayFitText(1000, "x", 0).advice.empty(),
              "the device's memory not known");
    }
    Check(AutoTargetEntropyText(6) == "-6 (minus the 6 actions)" &&
              AutoTargetEntropyText(1) == "-1 (minus the 1 action)",
          "the automatic target entropy reads as minus the actions");
    const auto preset = TunedSacSettings("HalfCheetah-v5");
    Check(preset && preset->learner.at("learning_starts") == "10000" && preset->environment.at("copies") == "1" &&
              !TunedSacSettings("InvertedPendulum-v5") && !TunedSacSettings("CartPole-v1"),
          "SAC is tuned for HalfCheetah only (RL Zoo)");
    if (preset) {
        Graph g = Pattern(id, 256, NodeType::ReLU, NodeType::SAC);
        g.At(id.env).parameters["environment"] = pendulum_key;
        for (const auto& [k, v] : preset->learner) g.At(id.ppo).parameters[k] = v;
        const auto config = Compile(g);
        Check(config.rl && config.rl->sac.learning_starts == 10000 && config.rl->sac.buffer_size == 1000000,
              "the HalfCheetah preset's values compile:" + Issues(config));
    }
}

// TD3 (TOFIX216 TD3 U1): the node compiles to a Td3Config with SB3's names
// and defaults (RL Zoo's noise), the Q branch as SAC's, and its refusals
void CheckTd3() {
    using namespace cyxwiz::rl_graph;
    const std::string pendulum_key = "test.continuous:InvertedPendulum-v5";
    Ids id{};
    {
        Graph g = Pattern(id, 256, NodeType::ReLU, NodeType::TD3);
        g.At(id.env).parameters["environment"] = pendulum_key;
        Check(cyxwiz::HasTrainingGraphStructure(g.nodes), "the toolbar sees a TD3 graph as a training graph");
        const auto config = Compile(g);
        Check(config.is_valid && config.rl, "the TD3 graph compiles:" + Issues(config));
        if (!config.rl) return;
        const auto& rl = *config.rl;
        Check(rl.algorithm == cyxwiz::RLAlgorithm::TD3 && rl.q_layers.size() == 4 && rl.value_layers.empty() &&
                  rl.policy_layers.size() == 4 && rl.policy_outputs == 1 && rl.learner_node_id == id.ppo &&
                  rl.q_head_node_id == id.vhead,
              "TD3: a policy chain, a Q chain, no value chain, 1 action");
        Check(std::string(cyxwiz::RLAlgorithmName(rl.algorithm)) == "TD3" && cyxwiz::UsesQNetworks(rl.algorithm) &&
                  !cyxwiz::UsesQNetworks(cyxwiz::RLAlgorithm::PPO),
              "TD3 is named and trains Q networks");
        const auto& c = rl.td3;
        Check(c.total_timesteps == 1000000 && c.buffer_size == 1000000 && c.learning_starts == 100 &&
                  c.batch_size == 256 && c.learning_rate == 1e-3f && c.gamma == 0.99f && c.tau == 0.005f &&
                  c.train_freq == 1 && c.gradient_steps == 1 && c.policy_delay == 2 && c.action_noise &&
                  c.action_noise_std == 0.1f && c.target_policy_noise == 0.2f && c.target_noise_clip == 0.5f &&
                  !c.normalize_observations && std::isnan(c.target_return),
              "TD3's defaults are Stable-Baselines3's, with RL Zoo's Gaussian noise 0.1");
        Check(!cyxwiz::FusedNetworksExpected(rl, cyxwiz::DeviceType::CPU, 0), "TD3: no fused kernels on the CPU");
        const bool cuda_provider =
            !cyxwiz::NeuralProviderRegistry::Instance().ListServing({cyxwiz::DeviceType::CUDA, 0}).empty();
        Check(cyxwiz::FusedNetworksExpected(rl, cyxwiz::DeviceType::CUDA, 0) == cuda_provider,
              "TD3's 256-unit actor and Q networks are expected fused on CUDA exactly when a provider serves CUDA");

        g.At(id.ppo).parameters["action_noise"] = "none";
        g.At(id.ppo).parameters["action_noise_std"] = "0";  // moot without noise
        g.At(id.ppo).parameters["policy_delay"] = "3";
        g.At(id.ppo).parameters["target_policy_noise"] = "0.1";
        g.At(id.ppo).parameters["target_noise_clip"] = "0.3";
        g.At(id.ppo).parameters["stop_at_return"] = "true";
        g.At(id.ppo).parameters["target_return"] = "950";
        const auto typed = Compile(g);
        Check(typed.rl && !typed.rl->td3.action_noise && typed.rl->td3.policy_delay == 3 &&
                  typed.rl->td3.target_policy_noise == 0.1f && typed.rl->td3.target_noise_clip == 0.3f &&
                  typed.rl->td3.target_return == 950.0f,
              "no noise, a policy delay, the target smoothing and a target return are read:" + Issues(typed));
    }
    {
        Graph g = Pattern(id, 256, NodeType::ReLU, NodeType::TD3);  // CartPole: a choice of actions
        const auto config = Compile(g);
        Check(!config.is_valid && HasIssue(config, cyxwiz::IssueLevel::Error, "TD3 learns continuous actions"),
              "TD3 on a choice of actions is refused:" + Issues(config));
    }
    {
        Graph g = Pattern(id, 256, NodeType::ReLU, NodeType::TD3);
        g.At(id.env).parameters["environment"] = pendulum_key;
        g.Unlink(id.vhead, id.ppo);
        g.Unlink(id.va2, id.vhead);
        const int value = g.Add(NodeType::ValueHead, "Value Head");
        g.Link(id.va2, value);
        g.Link(value, id.ppo, 0, 2);
        const auto config = Compile(g);
        Check(!config.is_valid && HasIssue(config, cyxwiz::IssueLevel::Error, "Connect a Q Head to TD3's Q"),
              "a Value Head into TD3's Q is refused:" + Issues(config));
    }
    {
        Graph g = Pattern(id, 256, NodeType::ReLU, NodeType::TD3);
        g.At(id.env).parameters["environment"] = pendulum_key;
        g.At(id.ppo).parameters["policy_delay"] = "0";
        g.At(id.ppo).parameters["action_noise_std"] = "0";
        g.At(id.ppo).parameters["buffer_size"] = "100";
        const auto config = Compile(g);
        Check(!config.is_valid && HasIssue(config, cyxwiz::IssueLevel::Error, "policy every must each be at least 1") &&
                  HasIssue(config, cyxwiz::IssueLevel::Error, "The noise std must be above 0") &&
                  HasIssue(config, cyxwiz::IssueLevel::Error,
                           "Batch size 256 is larger than the replay buffer (100 steps)"),
              "a policy delay of 0, Gaussian noise of std 0 and a batch past the buffer are refused:" + Issues(config));
    }
    {
        Graph g = Pattern(id, 256, NodeType::ReLU, NodeType::TD3);
        g.At(id.env).parameters["environment"] = pendulum_key;
        g.Add(NodeType::SAC, "SAC");
        const auto config = Compile(g);
        Check(!config.is_valid && HasIssue(config, cyxwiz::IssueLevel::Error, "keep one PPO, SAC or TD3 node"),
              "TD3 beside SAC is refused:" + Issues(config));
    }

    // Presentation
    {
        Graph g = Pattern(id, 300, NodeType::ReLU, NodeType::TD3);
        Check(LearnerFed(g.nodes, g.links, id.phead) == NodeType::TD3 &&
                  LearnerFed(g.nodes, g.links, id.vhead) == NodeType::TD3 &&
                  LearnerFed(g.nodes, g.links, id.p1) == NodeType::Unknown,
              "the heads know they feed TD3");
    }
    const auto action = PolicyHeadFacts(Pendulum(), 300, NodeType::TD3);
    Check(action.size() == 4 &&
              action[0].value == "deterministic: tanh of the layer, 1 action, scaled to the action bounds" &&
              action[1].value == "300 -> 1 action (added)",
          "a TD3 Policy Head adds one tanh action per action");
    const auto q = QHeadFacts(300, 6, NodeType::TD3);
    Check(q.size() == 4 && q[2].value == "2 Q networks and a target copy of each (built by TD3)" &&
              q[3].label == "Policy",
          "the Q Head under TD3 says the policy learns from Q network 1");
    Check(ExplorationNoiseText(true, "0.1") == "Gaussian noise, std 0.1 of the action range, after the random steps" &&
              ExplorationNoiseText(false, "0.1") == "None: the policy's own actions after the random steps",
          "TD3's exploration reads as its noise");
    const auto preset = TunedTd3Settings("HalfCheetah-v5");
    Check(preset && preset->learner.at("learning_starts") == "10000" && preset->learner.at("action_noise_std") == "0.1" &&
              preset->learner.at("learning_rate") == "0.001" && preset->environment.at("copies") == "1" &&
              !TunedTd3Settings("InvertedPendulum-v5") && !TunedTd3Settings("CartPole-v1"),
          "TD3 is tuned for HalfCheetah only (RL Zoo's mujoco defaults)");
    if (preset) {
        Graph g = Pattern(id, 400, NodeType::ReLU, NodeType::TD3);
        g.At(id.env).parameters["environment"] = pendulum_key;
        for (const auto& [k, v] : preset->learner) g.At(id.ppo).parameters[k] = v;
        const auto config = Compile(g);
        Check(config.rl && config.rl->td3.learning_starts == 10000 && config.rl->td3.action_noise,
              "the HalfCheetah preset's values compile:" + Issues(config));
    }
    Check(gui::ResolveNodeTypeImportName("TD3") == NodeType::TD3, "a saved graph names the node TD3");
}

}  // namespace

int main() {
    cyxwiz::SetRLEnvironmentLookup([](const std::string& key, const std::map<std::string, std::string>&,
                                      cyxwiz::plugin::PluginEnvironmentInfo& info, std::string&) {
        if (key == "test.continuous:InvertedPendulum-v5") {
            info = Pendulum();
            return true;
        }
        if (key == "test.continuous:HalfCheetah-v5") {
            info = Cheetah();
            return true;
        }
        if (key != "cyxwiz.builtin:CartPole-v1") return false;
        info = CartPole();
        return true;
    });
    CheckValid();
    CheckErrors();
    CheckPresentation();
    CheckReference();
    CheckSacReference();
    CheckSac();
    CheckTd3();
    CheckExamples();
    if (failures == 0) std::printf("test_rl_graph_compile: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
