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
                               const std::map<std::string, std::string>& settings, int count,
                               std::shared_ptr<MjViewSlot> view, std::string& error) {
    if (count < 1) {
        error = "Ask for at least one copy.";
        return nullptr;
    }
    MjTaskValues values;
    if (!ReadMjTaskValues(spec, settings, values, error)) return nullptr;

    std::unique_ptr<MjEnvBatch> batch(new MjEnvBatch());
    // Through the stress scene's builder: with no scene it is the model file as it is
    batch->model_ = CompileMjScene(model_path, MjScene{}, batch->stress_.robot, error);
    if (!batch->model_) return nullptr;
    batch->task_ = spec.create(values);
    batch->task_->SetRobot(batch->stress_.robot.nq, batch->stress_.robot.nv);
    batch->count_ = count;
    batch->view_ = std::move(view);
    batch->model_path_ = model_path;
    batch->title_ = spec.name;
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
    ShowCopy(-1);
    pool_.reset();  // no worker touches the data below any more
    for (mjData* data : data_) mj_deleteData(data);
    if (model_) mj_deleteModel(model_);
}

bool MjEnvBatch::ShowCopy(int copy) {
    if (!view_) return false;
    std::lock_guard<std::mutex> lock(view_->mutex);
    if (copy < 0 || copy >= count_) {
        if (view_->owner == this) view_->owner = nullptr;
        shown_copy_ = -1;
        return copy < 0;
    }
    view_->owner = this;
    view_->model_path = model_path_;
    view_->title = title_;
    view_->copy = copy;
    view_->state.clear();
    view_->episode = 1;
    view_->step = 0;
    view_->max_steps = max_episode_steps_;
    view_->step_seconds = model_->opt.timestep * task_->FrameSkip();
    view_->episode_return = 0.0;
    view_->has_last_return = false;
    view_->action.clear();
    view_->last_ended_by_task = false;
    view_->requests.clear();
    view_->scene = stress_.scene;
    view_->stress_message.clear();
    ++view_->version;
    shown_copy_ = copy;
    return true;
}

void MjEnvBatch::Publish(const SteppedCopy* step) {
    if (shown_copy_ < 0) return;
    std::lock_guard<std::mutex> lock(view_->mutex);
    if (view_->owner != this) {  // another batch took the view
        shown_copy_ = -1;
        return;
    }
    view_->state.resize(static_cast<size_t>(mj_stateSize(model_, mjSTATE_FULLPHYSICS)));
    mj_getState(model_, data_[shown_copy_], view_->state.data(), mjSTATE_FULLPHYSICS);
    if (!step) {  // a new start
        view_->episode = 1;
        view_->episode_return = 0.0;
        view_->has_last_return = false;
        view_->action.clear();
    } else {
        view_->action.assign(step->action, step->action + action_size_);
        view_->episode_return += step->reward;
        if (step->episode_ended) {  // the copy started its next episode in Step
            view_->last_ended_by_task = step->terminated;
            view_->last_return = view_->episode_return;
            view_->has_last_return = true;
            view_->episode_return = 0.0;
            ++view_->episode;
        }
    }
    view_->step = steps_[shown_copy_];
    view_->max_steps = max_episode_steps_;
    view_->scene = stress_.scene;
    if (!stress_.message.empty()) {
        view_->stress_message = std::move(stress_.message);
        stress_.message.clear();
    }
    ++view_->version;
}

MjStressHost MjEnvBatch::StressHost() {
    return {[this] { return static_cast<const mjModel*>(model_); }, [this] { return data_[shown_copy_]; },
            [this](const MjScene& scene, std::string& error) { return Rebuild(scene, error); },
            model_->opt.timestep * task_->FrameSkip()};
}

void MjEnvBatch::ApplyStress() {
    if (shown_copy_ < 0) return;
    std::vector<MjStressRequest> requests;
    {
        std::lock_guard<std::mutex> lock(view_->mutex);
        if (view_->owner != this) return;
        requests.swap(view_->requests);
    }
    if (!requests.empty()) ApplyStressRequests(requests, stress_, StressHost());
    ApplyPushes(stress_, model_, data_[shown_copy_]);
}

bool MjEnvBatch::Rebuild(const MjScene& scene, std::string& error) {
    MjRobotSize robot;
    mjModel* model = CompileMjScene(model_path_, scene, robot, error);
    if (!model) return false;
    std::vector<mjData*> data;
    for (int i = 0; i < count_; ++i) {
        mjData* d = mj_makeData(model);
        if (!d) {
            for (mjData* made : data) mj_deleteData(made);
            mj_deleteModel(model);
            error = "MuJoCo could not make the state of copy " + std::to_string(i) + " (out of memory?).";
            return false;
        }
        CopySceneState(model_, data_[i], model, d, stress_.robot);
        data.push_back(d);
    }
    for (mjData* d : data_) mj_deleteData(d);
    mj_deleteModel(model_);
    model_ = model;
    data_ = std::move(data);
    return true;
}

void MjEnvBatch::StartCopy(int copy, float* observation) {
    mjData* d = data_[copy];
    mj_resetData(model_, d);
    task_->ResetState(model_, d, random_[copy]);
    steps_[copy] = 0;
    task_->Observe(model_, d, observation);
}

bool MjEnvBatch::Reset(uint64_t seed, float* observations) {
    if (shown_copy_ >= 0) {
        EndTimedPush(stress_);
        StartNextGround(stress_, StressHost());
    }
    for (int i = 0; i < count_; ++i) random_[i] = MjRandom(MjRandom::CopySeed(seed, i));
    const auto start = [this, observations](int i) {
        StartCopy(i, observations + static_cast<size_t>(i) * observation_size_);
    };
    if (pool_) pool_->Run(count_, start);
    else for (int i = 0; i < count_; ++i) start(i);
    started_ = true;
    error_.clear();
    Publish(nullptr);
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
    ApplyStress();
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
    if (shown_copy_ >= 0) {
        const int c = shown_copy_;
        bool ended = out.terminated[c] != 0 || out.truncated[c] != 0;
        if (stress_.cut_episode) {  // the user asked for a new episode on another ground: cut this one
            stress_.cut_episode = false;
            if (!ended) {
                float* observation = out.observations + static_cast<size_t>(c) * observation_size_;
                std::copy(observation, observation + observation_size_,
                          out.final_observations + static_cast<size_t>(c) * observation_size_);
                out.truncated[c] = 1;
                StartCopy(c, observation);
                ended = true;
            }
        }
        if (ended) {
            EndTimedPush(stress_);  // the new episode starts without it
            StartNextGround(stress_, StressHost());
        }
        const SteppedCopy stepped{actions + static_cast<size_t>(c) * action_size_, out.rewards[c], ended,
                                  out.terminated[c] != 0};
        Publish(&stepped);
    }
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
    info.view_panel_id = "mujoco_viewport";  // ShowCopy's view
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
