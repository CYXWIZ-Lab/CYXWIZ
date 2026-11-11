#include "trial_runner.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <filesystem>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace cyxwiz {

namespace {

const gui::MLNode* FindNode(const std::vector<gui::MLNode>& nodes, int id) {
    const auto it = std::find_if(nodes.begin(), nodes.end(), [id](const gui::MLNode& n) { return n.id == id; });
    return it == nodes.end() ? nullptr : &*it;
}

// What a trial takes from the changed graph's compile: the model, the
// optimizer and scheduler, and the training loop.
void TakeTrialSettings(const TrainingConfiguration& from, TrainingConfiguration& to) {
    to.layers = from.layers;
    to.graph_plan = from.graph_plan;
    to.graph_op_node_ids = from.graph_op_node_ids;
    to.multi_input_layer_node_ids = from.multi_input_layer_node_ids;
    to.metric_learning_graph = from.metric_learning_graph;
    to.bert_encoder_graph = from.bert_encoder_graph;
    to.backend_placements = from.backend_placements;
    to.compiler_placement_fingerprint = from.compiler_placement_fingerprint;

    to.batch_size = from.batch_size;
    to.epochs = from.epochs;
    to.grad_accum_steps = from.grad_accum_steps;
    to.early_stopping_patience = from.early_stopping_patience;

    to.optimizer_node_id = from.optimizer_node_id;
    to.optimizer_type = from.optimizer_type;
    to.learning_rate = from.learning_rate;
    to.momentum = from.momentum;
    to.beta1 = from.beta1;
    to.beta2 = from.beta2;
    to.epsilon = from.epsilon;
    to.rmsprop_alpha = from.rmsprop_alpha;
    to.weight_decay = from.weight_decay;
    to.weight_decay_exclude = from.weight_decay_exclude;
    to.lr_schedule = from.lr_schedule;
    to.warmup_ratio = from.warmup_ratio;
    to.min_lr_ratio = from.min_lr_ratio;
    to.grad_clip_norm = from.grad_clip_norm;
    to.scheduler = from.scheduler;
    to.scheduler_node_id = from.scheduler_node_id;
    to.regularization_node_id = from.regularization_node_id;
}

}  // namespace

bool IsTrialTarget(const TrainingConfiguration& launched, const gui::MLNode& node, const std::string& property,
                   std::string* why) {
    const auto refuse = [why](std::string reason) {
        if (why) *why = std::move(reason);
        return false;
    };
    if (property.empty()) return refuse("no property named");
    const bool layer = std::any_of(launched.layers.begin(), launched.layers.end(),
                                   [&node](const CompiledLayer& l) { return l.node_id == node.id; });
    const bool optimizer = node.id == launched.optimizer_node_id;
    const bool scheduler = node.id == launched.scheduler_node_id || node.id == launched.regularization_node_id;
    const bool loop = node.type == gui::NodeType::DataLoader && (property == "batch_size" || property == "epochs");
    if (!layer && !optimizer && !scheduler && !loop) {
        return refuse(node.name + " is not a layer, the optimizer, the scheduler or the DataLoader's batch size / "
                      "epochs of the trained model");
    }
    if (!loop && !node.parameters.count(property)) return refuse(node.name + " has no property '" + property + "'");
    return true;
}

bool TrialConfiguration(const TrialSetup& setup, const std::vector<TrialChange>& changes, TrainingConfiguration& out,
                        std::string& error) {
    out = setup.launched;
    if (changes.empty() && !setup.other_architecture) return true;

    std::vector<gui::MLNode> nodes = setup.nodes;
    for (const auto& change : changes) {
        auto it = std::find_if(nodes.begin(), nodes.end(), [&](const gui::MLNode& n) { return n.id == change.node_id; });
        if (it == nodes.end()) {
            error = "no node " + std::to_string(change.node_id) + " in the graph";
            return false;
        }
        std::string why;
        if (!IsTrialTarget(setup.launched, *it, change.property, &why)) {
            error = why;
            return false;
        }
        it->parameters[change.property] = change.value;
    }

    TrainingConfiguration compiled = GraphCompiler{}.Compile(nodes, setup.links, /*allow_unloaded_data=*/true);
    // A data finding (data not loaded here) is the launcher's, already settled;
    // any other error is the changed graph's.
    const auto graph_error = std::find_if(compiled.issues.begin(), compiled.issues.end(), [&](const ValidationIssue& i) {
        if (i.level != IssueLevel::Error) return false;
        const gui::MLNode* node = FindNode(nodes, i.node_id);
        return !(node && (node->type == gui::NodeType::DataInput || node->type == gui::NodeType::DatasetInput));
    });
    if (graph_error != compiled.issues.end()) {
        error = "the changed graph does not compile: " + graph_error->message;
        return false;
    }
    TakeTrialSettings(compiled, out);
    return true;
}

TrialOutcome RunTrial(const TrialSetup& setup, const std::vector<TrialChange>& changes,
                      const TrialCallbacks& callbacks) {
    TrialOutcome outcome;
    const auto start = std::chrono::steady_clock::now();
    const auto finish = [&](std::string error) {
        outcome.error = std::move(error);
        outcome.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return outcome;
    };
    if (!setup.recipe) return finish(setup.recipe.error.empty() ? "no training recipe" : setup.recipe.error);

    TrainingConfiguration config;
    std::string error;
    if (!TrialConfiguration(setup, changes, config, error)) return finish(error);

    // The trial's checkpoints go to a scratch folder removed after; nothing
    // is kept as the project's model.
    static std::atomic<uint64_t> next{0};
    const std::filesystem::path scratch =
        std::filesystem::temp_directory_path() / "cyxwiz_trials" /
        (std::to_string(
#ifdef _WIN32
             _getpid()
#else
             getpid()
#endif
         ) + "_" + std::to_string(++next));
    struct RemoveAfter {
        std::filesystem::path path;
        ~RemoveAfter() {
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
        }
    } remove_after{scratch};
    config.save_best_checkpoint = false;
    config.checkpoint_dir = scratch.string();
    const int epochs = std::max(1, config.epochs);
    const int batch_size = std::max(1, config.batch_size);

    std::unique_ptr<TrainingExecutor> executor;
    try {
        executor = setup.recipe.make(config, batch_size, error);
    } catch (const std::exception& e) {
        error = e.what();
    }
    if (!executor) return finish(error.empty() ? "the run could not be built" : error);

    bool cancelled = false;
    const BatchCallback on_batch = [&](int, int, int, float, float) {
        if (!cancelled && callbacks.should_cancel && callbacks.should_cancel()) {
            cancelled = true;
            executor->Stop();
        }
    };
    const EpochCallback on_epoch = [&](int epoch, float train_loss, float, float val_loss, float val_acc, float) {
        // The executor numbers epochs from 1; val_loss is -1 when validation skipped it.
        if (callbacks.on_epoch) callbacks.on_epoch({epoch, epochs, train_loss, val_loss, val_acc});
    };
    try {
        executor->Train(epochs, batch_size, on_batch, on_epoch);
    } catch (const std::exception& e) {
        return finish(std::string("training failed: ") + e.what());
    }
    if (cancelled) {
        outcome.cancelled = true;
        return finish("cancelled");
    }

    const TrainingMetrics m = executor->GetMetrics();
    if (m.terminal_status == "failed")
        return finish(m.terminal_reason.empty() ? std::string("training failed") : m.terminal_reason);
    outcome.epochs = m.last_executed_epoch;
    outcome.score.regression = UsesRegressionMetrics(executor->GetConfig());
    outcome.score.has_validation = m.has_validation_metrics;
    outcome.score.val_loss = m.val_loss;
    outcome.score.val_accuracy = m.val_accuracy;
    outcome.score.val_mae = m.val_mae;
    outcome.score.val_rmse = m.val_rmse;
    outcome.score.train_loss = m.train_loss;
    outcome.score.train_accuracy = m.train_accuracy;
    if (!m.has_validation_metrics) return finish("the graph has no validation part to score the trial on");
    outcome.ok = true;
    finish({});
    spdlog::info("Trial: {} changes, {} epochs, validation loss {:.4f}{}", changes.size(), outcome.epochs,
                 outcome.score.val_loss,
                 outcome.score.regression ? fmt::format(", MAE {:.4f}", outcome.score.val_mae)
                                          : fmt::format(", accuracy {:.2f} %", outcome.score.val_accuracy * 100.0f));
    return outcome;
}

}  // namespace cyxwiz
