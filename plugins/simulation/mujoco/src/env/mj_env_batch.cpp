#include "mj_env_batch.h"

#include <algorithm>
#include <cmath>
#include <thread>

namespace cyxwiz::plugin::mujoco {

namespace {

mjModel* LoadModel(const std::string& model_path, std::string& error) {
    char load_error[1000] = "";
    mjModel* model = mj_loadXML(model_path.c_str(), nullptr, load_error, sizeof(load_error));
    if (!model) error = "The model " + model_path + " did not load: " + load_error;
    return model;
}

std::string FormatSetting(double value, bool integer) {
    if (integer) return std::to_string(static_cast<long long>(value));
    std::string text = std::to_string(value);
    while (text.size() > 1 && text.back() == '0') text.pop_back();
    if (!text.empty() && text.back() == '.') text.pop_back();
    return text;
}

}  // namespace

MjEnvBatch* MjEnvBatch::Create(const MjTaskSpec& spec, const std::string& model_path,
                               const std::map<std::string, std::string>& settings, int count, std::string& error) {
    if (count < 1) {
        error = "Ask for at least one copy.";
        return nullptr;
    }
    MjTaskValues values;
    if (!ReadMjTaskValues(spec, settings, values, error)) return nullptr;

    std::unique_ptr<MjEnvBatch> batch(new MjEnvBatch());
    batch->model_ = LoadModel(model_path, error);
    if (!batch->model_) return nullptr;
    batch->task_ = spec.create(values);
    batch->count_ = count;
    batch->observation_size_ = batch->task_->ObservationSize(batch->model_);
    batch->action_size_ = static_cast<int>(batch->model_->nu);
    const int limit = static_cast<int>(values.at("max_episode_steps"));
    batch->max_episode_steps_ = limit > 0 ? limit : spec.max_episode_steps;
    for (int i = 0; i < count; ++i) {
        mjData* data = mj_makeData(batch->model_);
        if (!data) {
            error = "MuJoCo could not make the state of copy " + std::to_string(i) + " (out of memory?).";
            return nullptr;
        }
        batch->data_.push_back(data);
    }
    batch->random_.resize(count);
    batch->steps_.assign(count, 0);
    int threads = static_cast<int>(values.at("threads"));
    if (threads <= 0) threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    threads = std::min(threads, count);
    if (threads > 1) batch->pool_ = std::make_unique<MjWorkerPool>(threads);
    return batch.release();
}

MjEnvBatch::~MjEnvBatch() {
    pool_.reset();  // no worker touches the data below any more
    for (mjData* data : data_) mj_deleteData(data);
    if (model_) mj_deleteModel(model_);
}

void MjEnvBatch::StartCopy(int copy, float* observation) {
    mjData* d = data_[copy];
    mj_resetData(model_, d);
    task_->ResetState(model_, d, random_[copy]);
    steps_[copy] = 0;
    task_->Observe(model_, d, observation);
}

bool MjEnvBatch::Reset(uint64_t seed, float* observations) {
    for (int i = 0; i < count_; ++i) random_[i] = MjRandom(MjRandom::CopySeed(seed, i));
    const auto start = [this, observations](int i) {
        StartCopy(i, observations + static_cast<size_t>(i) * observation_size_);
    };
    if (pool_) pool_->Run(count_, start);
    else for (int i = 0; i < count_; ++i) start(i);
    started_ = true;
    error_.clear();
    return true;
}

bool MjEnvBatch::Step(const float* actions, const PluginEnvironmentStepBuffers& out) {
    if (!started_) {
        error_ = "Reset the environments before stepping them.";
        return false;
    }
    const size_t action_values = static_cast<size_t>(count_) * action_size_;
    for (size_t i = 0; i < action_values; ++i) {
        if (!std::isfinite(actions[i])) {
            error_ = "Copy " + std::to_string(i / action_size_) + " was given an action that is not a number.";
            return false;
        }
    }
    const auto step = [this, actions, &out](int i) {
        mjData* d = data_[i];
        const float* action = actions + static_cast<size_t>(i) * action_size_;
        for (int a = 0; a < action_size_; ++a) d->ctrl[a] = action[a];
        const MjTaskStep result = task_->Step(model_, d, action);
        ++steps_[i];
        const bool truncated = !result.terminated && max_episode_steps_ > 0 && steps_[i] >= max_episode_steps_;
        float* observation = out.observations + static_cast<size_t>(i) * observation_size_;
        out.rewards[i] = static_cast<float>(result.reward);
        out.terminated[i] = result.terminated ? 1 : 0;
        out.truncated[i] = truncated ? 1 : 0;
        task_->Observe(model_, d, observation);
        if (result.terminated || truncated) {
            std::copy(observation, observation + observation_size_,
                      out.final_observations + static_cast<size_t>(i) * observation_size_);
            StartCopy(i, observation);
        }
    };
    if (pool_) pool_->Run(count_, step);
    else for (int i = 0; i < count_; ++i) step(i);
    error_.clear();
    return true;
}

bool DescribeMjTask(const MjTaskSpec& spec, const std::string& model_path, PluginEnvironmentInfo& info,
                    std::string& error) {
    mjModel* model = LoadModel(model_path, error);
    if (!model) return false;
    MjTaskValues defaults;
    if (!ReadMjTaskValues(spec, {}, defaults, error)) {
        mj_deleteModel(model);
        return false;
    }
    const auto task = spec.create(defaults);
    info = PluginEnvironmentInfo{};
    info.id = spec.id;
    info.name = spec.name;
    info.description = spec.description;
    info.category = spec.category;
    info.observation_size = task->ObservationSize(model);
    info.action_size = static_cast<int>(model->nu);
    for (int i = 0; i < model->nu; ++i) {
        const bool limited = model->actuator_ctrllimited[i] != 0;
        info.action_low.push_back(limited ? static_cast<float>(model->actuator_ctrlrange[2 * i]) : -INFINITY);
        info.action_high.push_back(limited ? static_cast<float>(model->actuator_ctrlrange[2 * i + 1]) : INFINITY);
    }
    info.max_episode_steps = spec.max_episode_steps;
    info.step_seconds = model->opt.timestep * task->FrameSkip();
    for (const auto& setting : spec.settings)
        info.settings.push_back({setting.key, setting.label, FormatSetting(setting.default_value, setting.integer),
                                 setting.description});
    for (const auto& setting : MjBatchSettings())
        info.settings.push_back({setting.key, setting.label, FormatSetting(setting.default_value, setting.integer),
                                 setting.description});
    mj_deleteModel(model);
    return true;
}

}  // namespace cyxwiz::plugin::mujoco
