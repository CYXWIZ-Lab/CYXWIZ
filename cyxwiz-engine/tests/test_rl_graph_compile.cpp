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
        Check(!cyxwiz::FusedNetworksExpected(rl, cyxwiz::DeviceType::CUDA, 0), "no fused prediction for SAC");

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
        Check(!config.is_valid && HasIssue(config, cyxwiz::IssueLevel::Error, "keep one PPO or SAC node"),
              "two learners are refused:" + Issues(config));
    }

    // Presentation
    const auto q = QHeadFacts(256, 6);
    Check(q.size() == 3 && q[0].value == "256 -> 1 (added)" && q[1].value.find("6 values") != std::string::npos,
          "the Q Head adds 256 -> 1 and takes the 6 actions at the first Q layer");
    const auto squashed = PolicyHeadFacts(Pendulum(), 256, true);
    Check(squashed.size() == 3 && squashed[0].value == "Gaussian squashed by tanh, 1 action" &&
              squashed[1].value == "256 -> 1 mean and 256 -> 1 log std (added)",
          "a SAC Policy Head adds a mean and a log std per action");
    Check(ReplayMemoryText(cyxwiz::rl::ReplayBytes(1000000, 17, 6)) == "about 168 MB" &&
              ReplayMemoryText(cyxwiz::rl::ReplayBytes(10000000, 17, 6)) == "about 1.7 GB" &&
              cyxwiz::rl::ReplayBytes(0, 17, 6) == 0,
          "HalfCheetah's replay buffer of 1M steps is about 168 MB");
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

}  // namespace

int main() {
    cyxwiz::SetRLEnvironmentLookup([](const std::string& key, const std::map<std::string, std::string>&,
                                      cyxwiz::plugin::PluginEnvironmentInfo& info, std::string&) {
        if (key == "test.continuous:InvertedPendulum-v5") {
            info = Pendulum();
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
    CheckSac();
    if (failures == 0) std::printf("test_rl_graph_compile: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
