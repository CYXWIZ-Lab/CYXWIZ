#include "mj_env_batch.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <thread>

namespace cyxwiz::plugin::mujoco {

namespace {

std::string FormatSetting(double value, bool integer) {
    if (integer) return std::to_string(static_cast<long long>(value));
    std::string text = std::to_string(value);
    while (text.size() > 1 && text.back() == '0') text.pop_back();
    if (!text.empty() && text.back() == '.') text.pop_back();
    return text;
}

PluginEnvironmentSetting TaskSetting(const MjTaskSetting& setting) {
    PluginEnvironmentSetting s;
    s.key = setting.key;
    s.label = setting.label;
    s.default_value = FormatSetting(setting.default_value, setting.integer);
    s.description = setting.description;
    return s;
}

bool SameDraw(const MjEpisodeDraw& a, const MjEpisodeDraw& b) {
    return a.friction == b.friction && a.mass == b.mass && a.motor == b.motor;
}

constexpr size_t kFallsKept = 100;

}  // namespace

MjEnvBatch* MjEnvBatch::Create(const MjTaskSpec& spec, const std::string& model_path,
                               const std::map<std::string, std::string>& settings, int count,
                               std::shared_ptr<MjViewSlot> view, std::string& error) {
    if (count < 1) {
        error = "Ask for at least one copy.";
        return nullptr;
    }
    std::unique_ptr<MjEnvBatch> batch(new MjEnvBatch());
    if (!ReadMjRobustness(settings, batch->robust_, error)) return nullptr;
    MjTaskValues values;
    if (!ReadMjTaskValues(spec, WithoutRobustness(settings), values, error)) return nullptr;

    // Through the scene builder: with no scene it is the model file as it is
    if (!batch->builder_.Load(model_path, error)) return nullptr;
    batch->base_.reset(batch->builder_.Build(MjScene{}, error), mj_deleteModel);
    if (!batch->base_) return nullptr;
    const MjRobotSize& robot = batch->builder_.Robot();
    batch->stress_.robot = robot;
    batch->task_ = spec.create(values);
    batch->task_->SetRobot(robot.nq, robot.nv);
    batch->count_ = count;
    batch->view_ = std::move(view);
    batch->model_path_ = model_path;
    batch->title_ = spec.name;
    batch->root_ = RobotRootBody(batch->base_.get(), robot);
    batch->task_observation_size_ = batch->task_->ObservationSize(batch->base_.get());
    batch->observation_size_ = batch->task_observation_size_ + (batch->robust_.look_ahead ? kLookAheadCount : 0);
    batch->action_size_ = static_cast<int>(batch->base_->nu);
    const int limit = static_cast<int>(values.at("max_episode_steps"));
    batch->max_episode_steps_ = limit > 0 ? limit : spec.max_episode_steps;
    batch->models_.assign(count, batch->base_);
    batch->scenes_.assign(count, MjScene{});
    batch->draws_.assign(count, MjEpisodeDraw{});
    batch->profiles_.assign(count, GroundProfile(MjGround{}, batch->builder_.StartX()));
    for (int i = 0; i < count; ++i) {
        mjData* data = mj_makeData(batch->base_.get());
        if (!data) {
            error = "MuJoCo could not make the state of copy " + std::to_string(i) + " (out of memory?).";
            return nullptr;
        }
        batch->data_.push_back(data);
    }
    batch->random_.resize(count);
    batch->steps_.assign(count, 0);
    batch->returns_.assign(count, 0.0);
    batch->next_push_.assign(count, 0.0);
    batch->ended_return_.assign(count, 0.0);
    batch->fell_.assign(count, 0);
    batch->curriculum_.Start(batch->robust_, count);
    int threads = static_cast<int>(values.at("threads"));
    if (threads <= 0) threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    threads = std::min(threads, count);
    if (threads > 1) batch->pool_ = std::make_unique<MjWorkerPool>(threads);
    return batch.release();
}

MjEnvBatch::~MjEnvBatch() {
    ShowCopy(-1);
    pool_.reset();  // no worker touches the data below any more
    for (size_t i = 0; i < data_.size(); ++i) mj_deleteData(data_[i]);
}

double MjEnvBatch::StepSeconds() const { return base_->opt.timestep * task_->FrameSkip(); }

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
    view_->step_seconds = StepSeconds();
    view_->episode_return = 0.0;
    view_->has_last_return = false;
    view_->action.clear();
    view_->last_ended_by_task = false;
    view_->requests.clear();
    view_->scene = scenes_[static_cast<size_t>(copy)];
    view_->stress_message.clear();
    ++view_->version;
    shown_copy_ = copy;
    stress_.scene = scenes_[static_cast<size_t>(copy)];
    return true;
}

std::vector<PluginEnvironmentStat> MjEnvBatch::Stats() const {
    std::vector<PluginEnvironmentStat> stats;
    if (!robust_.Any()) return stats;
    char value[32], note[96];
    if (!robust_.grounds.empty()) {
        std::snprintf(value, sizeof(value), "%.2f", curriculum_.Level());
        if (robust_.curriculum_mark > 0.0)
            std::snprintf(note, sizeof(note), "%s", curriculum_.Rising() ? "curriculum, rising" : "curriculum at its top");
        else
            std::snprintf(note, sizeof(note), "drawn from %.2f to %.2f", robust_.difficulty_from, robust_.difficulty_to);
        stats.push_back({"Difficulty", value, note});
    }
    if (!falls_.empty()) {
        const double fell = static_cast<double>(std::count(falls_.begin(), falls_.end(), true));
        std::snprintf(value, sizeof(value), "%.0f %%", 100.0 * fell / static_cast<double>(falls_.size()));
        std::snprintf(note, sizeof(note), "episodes that ended early or upside down (last %zu)", falls_.size());
        stats.push_back({"Falls", value, note});
    }
    return stats;
}

void MjEnvBatch::Publish(const SteppedCopy* step) {
    if (shown_copy_ < 0) return;
    std::lock_guard<std::mutex> lock(view_->mutex);
    if (view_->owner != this) {  // another batch took the view
        shown_copy_ = -1;
        return;
    }
    const size_t c = static_cast<size_t>(shown_copy_);
    const mjModel* m = models_[c].get();
    view_->state.resize(static_cast<size_t>(mj_stateSize(m, mjSTATE_FULLPHYSICS)));
    mj_getState(m, data_[c], view_->state.data(), mjSTATE_FULLPHYSICS);
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
    view_->step = steps_[c];
    view_->max_steps = max_episode_steps_;
    view_->scene = scenes_[c];
    if (!stress_.message.empty()) {
        view_->stress_message = std::move(stress_.message);
        stress_.message.clear();
    }
    ++view_->version;
}

MjStressHost MjEnvBatch::StressHost() {
    return {[this] { return static_cast<const mjModel*>(models_[static_cast<size_t>(shown_copy_)].get()); },
            [this] { return data_[static_cast<size_t>(shown_copy_)]; },
            [this](const MjScene& scene, std::string& error) { return BuildCopy(shown_copy_, scene, error); },
            StepSeconds()};
}

void MjEnvBatch::ApplyStress() {
    if (shown_copy_ < 0) return;
    std::vector<MjStressRequest> requests;
    {
        std::lock_guard<std::mutex> lock(view_->mutex);
        if (view_->owner != this) return;
        requests.swap(view_->requests);
    }
    const size_t c = static_cast<size_t>(shown_copy_);
    stress_.scene = scenes_[c];
    if (!requests.empty()) ApplyStressRequests(requests, stress_, StressHost());
    ApplyPushes(stress_, models_[c].get(), data_[c]);
}

bool MjEnvBatch::BuildCopy(int copy, const MjScene& scene, std::string& error) {
    const size_t c = static_cast<size_t>(copy);
    const MjEpisodeDraw& draw = draws_[c];
    const bool plain = scene.Empty() && SameDraw(draw, MjEpisodeDraw{});
    std::shared_ptr<mjModel> model;
    if (plain) {
        model = base_;
    } else {
        model.reset(builder_.Build(scene, error), mj_deleteModel);
        if (!model) return false;
        if (!SameDraw(draw, MjEpisodeDraw{})) {
            ApplyFactors(draw, builder_.Robot(), model.get());
            if (draw.mass != 1.0) {  // the derived masses and inertias follow
                mjData* scratch = mj_makeData(model.get());
                if (!scratch) {
                    error = "MuJoCo could not make a state for copy " + std::to_string(copy) + " (out of memory?).";
                    return false;
                }
                mj_setConst(model.get(), scratch);
                mj_deleteData(scratch);
                model->stat = base_->stat;  // the view and the solver's scale stay the robot's
            }
        }
    }
    if (model != models_[c]) {
        mjData* d = mj_makeData(model.get());
        if (!d) {
            error = "MuJoCo could not make the state of copy " + std::to_string(copy) + " (out of memory?).";
            return false;
        }
        CopySceneState(models_[c].get(), data_[c], model.get(), d, builder_.Robot());
        mj_deleteData(data_[c]);
        data_[c] = d;
        models_[c] = std::move(model);
    }
    if (scene.ground != scenes_[c].ground) profiles_[c] = GroundProfile(scene.ground, builder_.StartX());
    scenes_[c] = scene;
    if (copy == shown_copy_) stress_.scene = scene;
    return true;
}

bool MjEnvBatch::NewEpisode(int copy) {
    const size_t c = static_cast<size_t>(copy);
    const bool stress_ground = copy == shown_copy_ && stress_.next_ground.has_value();
    if (!robust_.PerCopyModels() && !stress_ground) return false;
    MjScene scene = scenes_[c];
    if (robust_.PerCopyModels()) {
        draws_[c] = DrawEpisode(robust_, curriculum_.Level(), random_[c]);
        if (!robust_.grounds.empty()) scene.ground = draws_[c].ground;
    }
    if (stress_ground) {  // the Stress panel's ground wins for the shown copy
        scene.ground = *stress_.next_ground;
        stress_.next_ground.reset();
    }
    std::string error;
    if (!BuildCopy(copy, scene, error)) {
        if (copy == shown_copy_) stress_.message = error;
        error_ = error;
        return false;
    }
    return true;
}

void MjEnvBatch::StartCopy(int copy, float* observation) {
    const size_t c = static_cast<size_t>(copy);
    mjData* d = data_[c];
    const mjModel* m = models_[c].get();
    mj_resetData(m, d);
    task_->ResetState(m, d, random_[c]);
    steps_[c] = 0;
    returns_[c] = 0.0;
    if (robust_.push_every > 0.0) next_push_[c] = robust_.push_every * random_[c].Uniform(0.5, 1.5);
    ObserveCopy(copy, observation);
}

void MjEnvBatch::ObserveCopy(int copy, float* observation) const {
    const size_t c = static_cast<size_t>(copy);
    const mjData* d = data_[c];
    task_->Observe(models_[c].get(), d, observation);
    if (!robust_.look_ahead) return;
    // The ground's height above the floor ahead of the robot's root
    const double x = root_ >= 0 ? d->xpos[3 * root_] : 0.0;
    for (int k = 0; k < kLookAheadCount; ++k)
        observation[task_observation_size_ + k] =
            static_cast<float>(profiles_[c].At(x + kLookAheadSpacing * static_cast<double>(k + 1)));
}

bool MjEnvBatch::Reset(uint64_t seed, float* observations) {
    if (shown_copy_ >= 0) EndTimedPush(stress_);
    for (int i = 0; i < count_; ++i) random_[i] = MjRandom(MjRandom::CopySeed(seed, i));
    const auto start = [this, observations](int i) {
        StartCopy(i, observations + static_cast<size_t>(i) * observation_size_);
    };
    if (pool_) pool_->Run(count_, start);
    else for (int i = 0; i < count_; ++i) start(i);
    // Each copy's first episode: its ground and factors (one copy at a time)
    error_.clear();
    for (int i = 0; i < count_; ++i) {
        if (NewEpisode(i)) ObserveCopy(i, observations + static_cast<size_t>(i) * observation_size_);
        else if (!error_.empty()) return false;
    }
    started_ = true;
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
    const MjRobotSize& robot = builder_.Robot();
    const auto step = [this, actions, &out, &robot](int i) {
        const size_t c = static_cast<size_t>(i);
        mjData* d = data_[c];
        const mjModel* m = models_[c].get();
        // A training push: a random velocity change at the copy's own times
        if (robust_.push_every > 0.0 && d->time >= next_push_[c]) {
            MjRandom& random = random_[c];
            const std::array<double, 3> way = Direction(random.Uniform(0.0, 360.0), random.Uniform(0.0, 45.0));
            const double speed = robust_.push_speed * random.Uniform(0.5, 1.0);
            KickRobot(m, d, robot, {way[0] * speed, way[1] * speed, way[2] * speed});
            next_push_[c] = d->time + robust_.push_every * random.Uniform(0.5, 1.5);
        }
        const float* action = actions + c * action_size_;
        for (int a = 0; a < action_size_; ++a) d->ctrl[a] = action[a];
        const MjTaskStep result = task_->Step(m, d, action);
        ++steps_[c];
        returns_[c] += result.reward;
        const bool truncated = !result.terminated && max_episode_steps_ > 0 && steps_[c] >= max_episode_steps_;
        float* observation = out.observations + c * observation_size_;
        out.rewards[c] = static_cast<float>(result.reward);
        out.terminated[c] = result.terminated ? 1 : 0;
        out.truncated[c] = truncated ? 1 : 0;
        ObserveCopy(i, observation);
        if (result.terminated || truncated) {
            ended_return_[c] = returns_[c];
            fell_[c] = result.terminated || RobotFlipped(m, d, robot) ? 1 : 0;
            std::copy(observation, observation + observation_size_, out.final_observations + c * observation_size_);
            StartCopy(i, observation);
        }
    };
    if (pool_) pool_->Run(count_, step);
    else for (int i = 0; i < count_; ++i) step(i);
    error_.clear();

    bool shown_ended = false;
    if (shown_copy_ >= 0) {
        const size_t c = static_cast<size_t>(shown_copy_);
        shown_ended = out.terminated[c] != 0 || out.truncated[c] != 0;
        if (stress_.cut_episode) {  // the user asked for a new episode on another ground: cut this one
            stress_.cut_episode = false;
            if (!shown_ended) {
                float* observation = out.observations + c * observation_size_;
                std::copy(observation, observation + observation_size_, out.final_observations + c * observation_size_);
                out.truncated[c] = 1;
                StartCopy(shown_copy_, observation);
                shown_ended = true;
                fell_[c] = 2;  // cut by the user: neither a fall nor a finished episode
            }
        }
    }
    // The episodes that ended: the curriculum and the falls, then each copy's next ground (one at a time)
    for (int i = 0; i < count_; ++i) {
        const size_t c = static_cast<size_t>(i);
        if (!out.terminated[c] && !out.truncated[c]) continue;
        if (fell_[c] != 2) {
            curriculum_.Finished(ended_return_[c]);
            falls_.push_back(fell_[c] == 1);
            if (falls_.size() > kFallsKept) falls_.pop_front();
        }
        fell_[c] = 0;
        if (i == shown_copy_) EndTimedPush(stress_);  // the new episode starts without it
        if (NewEpisode(i)) ObserveCopy(i, out.observations + c * observation_size_);
    }
    if (shown_copy_ >= 0) {
        const size_t c = static_cast<size_t>(shown_copy_);
        const SteppedCopy stepped{actions + c * action_size_, out.rewards[c], shown_ended, out.terminated[c] != 0};
        Publish(&stepped);
    }
    return true;
}

bool DescribeMjTask(const MjTaskSpec& spec, const std::string& model_path, PluginEnvironmentInfo& info,
                    std::string& error) {
    char load_error[1000] = "";
    mjModel* model = mj_loadXML(model_path.c_str(), nullptr, load_error, sizeof(load_error));
    if (!model) {
        error = "The model " + model_path + " did not load: " + load_error;
        return false;
    }
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
    for (const auto& setting : spec.settings) info.settings.push_back(TaskSetting(setting));
    for (const auto& setting : MjBatchSettings()) info.settings.push_back(TaskSetting(setting));
    for (const auto& setting : RobustnessSettings()) info.settings.push_back(setting);
    mj_deleteModel(model);
    return true;
}

bool DescribeMjTaskWith(const MjTaskSpec& spec, const PluginEnvironmentInfo& listed,
                        const std::map<std::string, std::string>& settings, PluginEnvironmentInfo& info,
                        std::string& error) {
    MjRobustness robust;
    if (!ReadMjRobustness(settings, robust, error)) return false;
    MjTaskValues values;
    if (!ReadMjTaskValues(spec, WithoutRobustness(settings), values, error)) return false;
    info = listed;
    if (robust.look_ahead) info.observation_size += kLookAheadCount;
    return true;
}

}  // namespace cyxwiz::plugin::mujoco
