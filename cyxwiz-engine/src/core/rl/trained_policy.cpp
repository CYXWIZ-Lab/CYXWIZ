#include "trained_policy.h"

#include "../formats/cyxmodel_format.h"
#include "../graph_compiler.h"
#include "../graph_document.h"
#include "../model_builder.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace cyxwiz::rl {

namespace {

using json = nlohmann::json;

constexpr int kPolicySchema = 1;

// The graph's layers plus the head's output layer
std::unique_ptr<IExecutableModel> Network(const std::vector<CompiledLayer>& hidden, size_t inputs, size_t outputs,
                                          float output_gain, const char* what, std::string& error) {
    std::vector<CompiledLayer> layers = hidden;
    CompiledLayer head;
    head.type = gui::NodeType::Dense;
    head.name = std::string(what) + " output";
    head.units = static_cast<int>(outputs);
    head.parameters["units"] = std::to_string(outputs);
    layers.push_back(head);
    auto built = BuildNetworkFromLayers(layers, inputs);
    if (!built.ok()) {
        error = std::string("The ") + what + " network could not be built: " + built.error_message;
        return nullptr;
    }
    InitializeLikeStableBaselines3(*built.model, output_gain);
    return std::move(built.model);
}

bool ToDType(DataType type, TensorDType& out) {
    switch (type) {
        case DataType::Float32: out = TensorDType::Float32; return true;
        case DataType::Float64: out = TensorDType::Float64; return true;
        case DataType::Int32: out = TensorDType::Int32; return true;
        case DataType::Int64: out = TensorDType::Int64; return true;
        case DataType::UInt8: out = TensorDType::UInt8; return true;
        default: return false;
    }
}

DataType FromDType(TensorDType type) {
    switch (type) {
        case TensorDType::Float64: return DataType::Float64;
        case TensorDType::Int32: return DataType::Int32;
        case TensorDType::Int64: return DataType::Int64;
        case TensorDType::UInt8: return DataType::UInt8;
        default: return DataType::Float32;
    }
}

std::vector<uint8_t> Bytes(const Tensor& t) {
    std::vector<uint8_t> out(t.NumBytes());
    if (const void* data = t.ReadData(); data && !out.empty()) std::memcpy(out.data(), data, out.size());
    return out;
}

std::vector<int64_t> Shape(const Tensor& t) { return {t.Shape().begin(), t.Shape().end()}; }

Tensor FromBytes(const std::vector<uint8_t>& data, const std::vector<int64_t>& shape, TensorDType dtype) {
    return Tensor(std::vector<size_t>(shape.begin(), shape.end()), data.data(), FromDType(dtype));
}

json UpdateToJson(const PpoUpdateMetrics& m) {
    const auto number = [](float v) { return std::isfinite(v) ? json(v) : json(nullptr); };
    return {{"update", m.update},
            {"timesteps", m.timesteps},
            {"episodes", m.episodes},
            {"mean_episode_return", number(m.mean_episode_return)},
            {"mean_episode_length", number(m.mean_episode_length)},
            {"loss", number(m.loss)},
            {"policy_loss", number(m.policy_loss)},
            {"value_loss", number(m.value_loss)},
            {"entropy_loss", number(m.entropy_loss)},
            {"approx_kl", number(m.approx_kl)},
            {"clip_fraction", number(m.clip_fraction)},
            {"explained_variance", number(m.explained_variance)},
            {"std", number(m.std)},
            {"steps_per_second", number(m.steps_per_second)},
            {"reached_target", m.reached_target}};
}

PpoUpdateMetrics UpdateFromJson(const json& j) {
    const auto number = [&j](const char* key) {
        return j.contains(key) && j[key].is_number() ? j[key].get<float>() : std::numeric_limits<float>::quiet_NaN();
    };
    PpoUpdateMetrics m;
    m.update = j.value("update", 0);
    m.timesteps = j.value("timesteps", int64_t{0});
    m.episodes = j.value("episodes", 0);
    m.mean_episode_return = number("mean_episode_return");
    m.mean_episode_length = number("mean_episode_length");
    m.loss = number("loss");
    m.policy_loss = number("policy_loss");
    m.value_loss = number("value_loss");
    m.entropy_loss = number("entropy_loss");
    m.approx_kl = number("approx_kl");
    m.clip_fraction = number("clip_fraction");
    m.explained_variance = number("explained_variance");
    m.std = number("std");
    m.steps_per_second = number("steps_per_second");
    m.reached_target = j.value("reached_target", false);
    return m;
}

// Optimizer states in optimizer/: "states" (JSON: each state's type,
// learning rate, step count, hyperparameters and tensor list) and one
// "<state>__<n>" file per tensor with the weights' header. The names hold no
// '.' or '/', which the package writer would rewrite.
std::map<std::string, std::vector<uint8_t>> OptimizerFiles(const std::map<std::string, OptimizerState>& states,
                                                           std::string& error) {
    formats::CyxModelFormat format;
    std::map<std::string, std::vector<uint8_t>> files;
    json index = json::object();
    for (const auto& [key, state] : states) {
        json tensors = json::array();
        int n = 0;
        for (const auto& [name, tensor] : state.tensors) {
            TensorDType dtype;
            if (!ToDType(tensor.GetDataType(), dtype)) {
                error = "The " + key + " optimizer's state '" + name + "' has a type a .cyxmodel cannot hold.";
                return {};
            }
            const std::string file = key + "__" + std::to_string(n++);
            files[file] = format.SerializeTensorWithHeader(Bytes(tensor), Shape(tensor), dtype);
            tensors.push_back({{"name", name}, {"file", file}});
        }
        index[key] = {{"schema_version", state.schema_version},
                      {"optimizer_type", state.optimizer_type},
                      {"learning_rate", state.learning_rate},
                      {"step_count", state.step_count},
                      {"hyperparameters", state.hyperparameters},
                      {"tensors", tensors}};
    }
    const std::string text = index.dump();
    files["states"] = std::vector<uint8_t>(text.begin(), text.end());
    return files;
}

bool OptimizerStates(const std::map<std::string, std::vector<uint8_t>>& files,
                     std::map<std::string, OptimizerState>& states, std::string& error) {
    const auto index_it = files.find("states");
    if (index_it == files.end()) return true;  // saved without them
    formats::CyxModelFormat format;
    const json index = json::parse(index_it->second.begin(), index_it->second.end());
    for (const auto& [key, s] : index.items()) {
        OptimizerState state;
        state.schema_version = s.at("schema_version").get<int>();
        state.optimizer_type = s.at("optimizer_type").get<std::string>();
        state.learning_rate = s.at("learning_rate").get<double>();
        state.step_count = s.at("step_count").get<int>();
        state.hyperparameters = s.at("hyperparameters").get<std::map<std::string, double>>();
        for (const auto& t : s.at("tensors")) {
            const auto file = files.find(t.at("file").get<std::string>());
            std::vector<uint8_t> data;
            std::vector<int64_t> shape;
            TensorDType dtype;
            if (file == files.end() || !format.DeserializeTensorWithHeader(file->second, data, shape, dtype)) {
                error = "The " + key + " optimizer's state is incomplete.";
                return false;
            }
            state.tensors[t.at("name").get<std::string>()] = FromBytes(data, shape, dtype);
        }
        states[key] = std::move(state);
    }
    return true;
}

// The parameters of `model` from the package's "<prefix><name>" weights
bool SetWeights(IExecutableModel& model, const std::string& prefix,
                const std::map<std::string, std::vector<uint8_t>>& weights,
                const std::map<std::string, std::vector<int64_t>>& shapes,
                const std::map<std::string, TensorDType>& dtypes, const char* what, std::string& error) {
    auto params = model.GetParameters();
    for (auto& [name, tensor] : params) {
        const auto key = prefix + name;
        const auto data = weights.find(key);
        if (data == weights.end()) {
            error = std::string("The saved ") + what + " network has no '" + name + "': the graph's layers differ from it.";
            return false;
        }
        const auto& shape = shapes.at(key);
        if (shape != Shape(tensor)) {
            const auto text = [](const std::vector<int64_t>& s) {
                std::string out;
                for (size_t i = 0; i < s.size(); ++i) out += (i ? " x " : "") + std::to_string(s[i]);
                return out;
            };
            error = std::string("The saved ") + what + " network's '" + name + "' is " + text(shape) +
                    "; the graph builds " + text(Shape(tensor)) + ".";
            return false;
        }
        tensor = FromBytes(data->second, shape, dtypes.at(key));
    }
    model.SetParameters(params);
    return true;
}

}  // namespace

bool BuildRLNetworks(const RLTrainingConfiguration& config, std::unique_ptr<IExecutableModel>& policy,
                     std::unique_ptr<IExecutableModel>& value, std::string& error) {
    policy = Network(config.policy_layers, config.observation_size, config.policy_outputs, 0.01f, "policy", error);
    if (!policy) return false;
    value = Network(config.value_layers, config.observation_size, 1, 1.0f, "value", error);
    return value != nullptr;
}

std::unique_ptr<TrainedPolicy> MakeTrainedPolicy(const RLTrainingConfiguration& config, std::string graph_json,
                                                 PpoTrainedState state, std::vector<PpoUpdateMetrics> updates) {
    auto out = std::make_unique<TrainedPolicy>();
    out->config = config;
    out->graph_json = std::move(graph_json);
    out->policy = std::move(state.policy);
    out->value = std::move(state.value);
    out->log_std = state.log_std;
    out->normalizer = std::move(state.normalizer);
    out->optimizers = std::move(state.optimizers);
    out->updates = std::move(updates);
    if (!out->updates.empty()) {
        out->steps = out->updates.back().timesteps;
        out->mean_return = out->updates.back().mean_episode_return;
    }
    return out;
}

std::vector<float> PolicyActions(TrainedPolicy& p, std::vector<float> observations, size_t rows, bool best,
                                 uint64_t seed) {
    const auto& info = p.config.environment;
    const size_t obs_size = p.config.observation_size;
    if (p.normalizer) p.normalizer->NormalizeObservations(observations.data(), rows);
    p.policy->SetTraining(false);
    const Tensor head = p.policy->Forward(Tensor({rows, obs_size}, observations.data()));
    const size_t outputs = p.config.policy_outputs;
    std::vector<float> actions;
    if (info.discrete_actions) {
        actions.resize(rows);
        if (best) {  // the most likely choice of each row
            const float* logits = head.ReadData<float>();
            for (size_t r = 0; r < rows; ++r) {
                const float* row = logits + r * outputs;
                actions[r] = static_cast<float>(std::max_element(row, row + outputs) - row);
            }
        } else {
            const Tensor drawn = SampleCategorical(head, seed);
            const float* data = drawn.ReadData<float>();
            std::copy(data, data + rows, actions.begin());
        }
        return actions;
    }
    const Tensor chosen = best ? head : SampleGaussian(head, p.log_std, seed);
    const float* data = chosen.ReadData<float>();
    const size_t act = static_cast<size_t>(info.action_size);
    actions.assign(data, data + rows * act);
    for (size_t i = 0; i < actions.size(); ++i)
        actions[i] = std::clamp(actions[i], info.action_low[i % act], info.action_high[i % act]);
    return actions;
}

std::string PolicyMismatch(const TrainedPolicy& p, const RLTrainingConfiguration& graph) {
    if (p.config.environment_key != graph.environment_key)
        return "This policy plays " + p.config.environment.id + "; this graph's Environment is " +
               graph.environment.id + ".";
    const auto widths = [](const std::vector<CompiledLayer>& layers) {
        std::string text;
        for (const auto& l : layers)
            if (l.units > 0) text += (text.empty() ? "" : "-") + std::to_string(l.units);
        return text.empty() ? std::string("no") : text;
    };
    const auto kinds = [](const std::vector<CompiledLayer>& layers) {
        std::vector<int> out;
        for (const auto& l : layers) out.push_back(static_cast<int>(l.type));
        return out;
    };
    const bool same = widths(p.config.policy_layers) == widths(graph.policy_layers) &&
                      widths(p.config.value_layers) == widths(graph.value_layers) &&
                      kinds(p.config.policy_layers) == kinds(graph.policy_layers) &&
                      kinds(p.config.value_layers) == kinds(graph.value_layers);
    if (!same)
        return "The policy was trained on " + widths(p.config.policy_layers) + " policy and " +
               widths(p.config.value_layers) + " value layers; this graph has " + widths(graph.policy_layers) +
               " and " + widths(graph.value_layers) + ".";
    return {};
}

bool SaveTrainedPolicy(const TrainedPolicy& p, const std::filesystem::path& path, std::string& error) {
    error.clear();
    if (!p.policy || !p.value) {
        error = "There is no trained policy to save.";
        return false;
    }
    if (p.graph_json.empty()) {
        error = "The policy has no graph to save with it.";
        return false;
    }
    try {
        std::map<std::string, std::vector<uint8_t>> weights;
        std::map<std::string, std::vector<int64_t>> shapes;
        std::map<std::string, TensorDType> dtypes;
        int64_t parameters = 0;
        int layers = 0;
        const auto add = [&](const std::string& name, const Tensor& t) {
            TensorDType dtype;
            if (!ToDType(t.GetDataType(), dtype)) throw std::runtime_error("'" + name + "' has an unsupported type");
            weights[name] = Bytes(t);
            shapes[name] = Shape(t);
            dtypes[name] = dtype;
            parameters += static_cast<int64_t>(t.NumElements());
            if (t.Shape().size() == 2) ++layers;
        };
        for (const auto& [name, t] : p.policy->GetParameters()) add("policy." + name, t);
        for (const auto& [name, t] : p.value->GetParameters()) add("value." + name, t);
        const auto& info = p.config.environment;
        if (!info.discrete_actions) add("log_std", p.log_std);

        json policy_json = {
            {"schema_version", kPolicySchema},
            {"environment",
             {{"key", p.config.environment_key},
              {"id", info.id},
              {"name", info.name},
              {"settings", p.config.environment_settings},
              {"copies", p.config.copies}}},
            {"observation_size", p.config.observation_size},
            {"discrete_actions", info.discrete_actions},
            {"policy_outputs", p.config.policy_outputs},
            {"steps", p.steps},
            {"mean_return", std::isfinite(p.mean_return) ? json(p.mean_return) : json(nullptr)},
            {"networks", {{"policy", "policy."}, {"value", "value."}, {"log_std", info.discrete_actions ? "" : "log_std"}}},
        };
        if (p.normalizer) {
            const auto& s = p.normalizer->Settings();
            const auto& stats = p.normalizer->ObservationStats();
            policy_json["normalize"] = {{"observations", s.observations},
                                        {"rewards", s.rewards},
                                        {"clip_observations", s.clip_observations},
                                        {"clip_rewards", s.clip_rewards},
                                        {"gamma", s.gamma},
                                        {"epsilon", s.epsilon},
                                        {"observation_mean", stats.mean},
                                        {"observation_var", stats.var},
                                        {"observation_count", stats.count}};
        }
        json updates = json::array();
        for (const auto& u : p.updates) updates.push_back(UpdateToJson(u));
        policy_json["updates"] = updates;

        auto optimizer_files = OptimizerFiles(p.optimizers, error);
        if (!error.empty()) return false;

        ModelManifest manifest;
        manifest.model_name = path.stem().string();
        manifest.model_type = "PPO policy";
        manifest.model_family = "rl_policy";
        manifest.num_parameters = static_cast<int>(parameters);
        manifest.num_layers = layers;
        manifest.has_graph = true;
        manifest.has_rl_policy = true;
        manifest.has_optimizer_state = !p.optimizers.empty();
        manifest.custom_metadata["rl_environment"] = info.id;
        manifest.custom_metadata["rl_environment_key"] = p.config.environment_key;
        manifest.custom_metadata["rl_steps"] = std::to_string(p.steps);
        if (std::isfinite(p.mean_return)) {
            char text[32];
            std::snprintf(text, sizeof(text), "%.1f", p.mean_return);
            manifest.custom_metadata["rl_mean_return"] = text;
        }
        ExportOptions options;
        options.include_graph = true;
        options.include_optimizer_state = true;
        options.include_training_history = false;
        options.compress = false;
        options.rl_policy_json = policy_json.dump(2);
        formats::CyxModelFormat format;
        if (!format.Create(path.string(), manifest, p.graph_json, TrainingConfig{}, nullptr, weights, shapes,
                           &optimizer_files, options, &dtypes)) {
            error = "The policy could not be saved: " + format.GetLastError();
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        error = std::string("The policy could not be saved: ") + e.what();
        return false;
    }
}

std::unique_ptr<TrainedPolicy> LoadTrainedPolicy(const std::filesystem::path& path, std::string& error) {
    error.clear();
    try {
        formats::CyxModelFormat format;
        std::string policy_text;
        if (!format.ExtractRLPolicy(path.string(), policy_text)) {
            error = format.GetLastError();
            return nullptr;
        }
        const json pj = json::parse(policy_text);
        if (pj.value("schema_version", 0) > kPolicySchema) {
            error = "The policy was saved by a newer CyxWiz (policy schema " +
                    std::to_string(pj.value("schema_version", 0)) + ").";
            return nullptr;
        }
        ModelManifest manifest;
        std::string graph_json;
        TrainingConfig trained;
        std::map<std::string, std::vector<uint8_t>> weights, optimizer_files;
        std::map<std::string, std::vector<int64_t>> shapes;
        std::map<std::string, TensorDType> dtypes;
        ImportOptions import;
        import.load_optimizer_state = true;
        if (!format.Extract(path.string(), manifest, graph_json, trained, nullptr, weights, shapes, &optimizer_files,
                            import, &dtypes)) {
            error = format.GetLastError();
            return nullptr;
        }

        // The networks come from the graph, compiled as Train compiles it
        GraphDocument document;
        if (!ParseGraphDocument(graph_json, document, error)) {
            error = "The policy's graph could not be read: " + error;
            return nullptr;
        }
        GraphCompiler compiler;
        const auto compiled = compiler.Compile(document.nodes, document.links, true);
        if (!compiled.rl) {
            const auto& env = pj.at("environment");
            std::string why = compiled.error_message;
            for (const auto& issue : compiled.issues)
                if (issue.level == IssueLevel::Error) {
                    why = issue.message;
                    break;
                }
            error = "This policy plays " + env.value("id", std::string{}) + " (" + env.value("key", std::string{}) +
                    "), and its graph does not compile here: " + why;
            return nullptr;
        }
        auto out = std::make_unique<TrainedPolicy>();
        out->config = *compiled.rl;
        out->graph_json = graph_json;
        if (!BuildRLNetworks(out->config, out->policy, out->value, error)) return nullptr;
        if (!SetWeights(*out->policy, "policy.", weights, shapes, dtypes, "policy", error) ||
            !SetWeights(*out->value, "value.", weights, shapes, dtypes, "value", error))
            return nullptr;
        if (!out->config.environment.discrete_actions) {
            const auto it = weights.find("log_std");
            if (it == weights.end()) {
                error = "The saved policy has no log standard deviation.";
                return nullptr;
            }
            out->log_std = FromBytes(it->second, shapes.at("log_std"), dtypes.at("log_std"));
        }
        if (pj.contains("normalize")) {
            const auto& n = pj["normalize"];
            VecNormalizeSettings s;
            s.observations = n.value("observations", false);
            s.rewards = n.value("rewards", false);
            s.clip_observations = n.value("clip_observations", 10.0f);
            s.clip_rewards = n.value("clip_rewards", 10.0f);
            s.gamma = n.value("gamma", 0.99f);
            s.epsilon = n.value("epsilon", 1e-8f);
            RunningMeanStd stats(out->config.observation_size);
            stats.mean = n.at("observation_mean").get<std::vector<double>>();
            stats.var = n.at("observation_var").get<std::vector<double>>();
            stats.count = n.at("observation_count").get<double>();
            out->normalizer = std::make_unique<VecNormalize>(1, out->config.observation_size, s);
            if (!out->normalizer->RestoreObservationStats(stats)) {
                error = "The saved observation statistics do not match the environment's " +
                        std::to_string(out->config.observation_size) + " observation values.";
                return nullptr;
            }
        }
        if (!OptimizerStates(optimizer_files, out->optimizers, error)) return nullptr;
        for (const auto& u : pj.value("updates", json::array())) out->updates.push_back(UpdateFromJson(u));
        out->steps = pj.value("steps", int64_t{0});
        out->mean_return = pj.contains("mean_return") && pj["mean_return"].is_number()
                               ? pj["mean_return"].get<float>()
                               : std::numeric_limits<float>::quiet_NaN();
        return out;
    } catch (const std::exception& e) {
        error = "The policy could not be read: " + std::string(e.what());
        return nullptr;
    }
}

}  // namespace cyxwiz::rl
