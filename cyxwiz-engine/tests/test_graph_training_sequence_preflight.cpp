#include "../src/gui/graph_training_launcher.h"

#include "../src/core/arrow_dataset.h"
#include "../src/core/async_task_manager.h"
#include "../src/core/data_registry.h"

#include <arrow/api.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr const char* kDatasetName = "gui_sequence_preflight_runtime";

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

std::shared_ptr<arrow::Array> FinishStringArray(
    const std::vector<std::string>& values) {
    arrow::StringBuilder builder;
    for (const auto& value : values) {
        auto status = builder.Append(value);
        Check(status.ok(), status.ToString());
    }

    std::shared_ptr<arrow::Array> array;
    auto status = builder.Finish(&array);
    Check(status.ok(), status.ToString());
    return array;
}

std::shared_ptr<arrow::Table> MakeSequenceTable() {
    auto schema = arrow::schema({
        arrow::field("tokens", arrow::utf8()),
        arrow::field("pos_tags", arrow::utf8()),
        arrow::field("ner_tags", arrow::utf8()),
    });
    return arrow::Table::Make(schema,
                              {FinishStringArray({"London", "wins"}),
                               FinishStringArray({"NNP", "VBZ"}),
                               FinishStringArray({"B-geo", "O"})},
                              2);
}

gui::MLNode MakeDataInputNode() {
    gui::MLNode node;
    node.id = 1;
    node.type = gui::NodeType::DataInput;
    node.category = gui::NodeCategory::DataPipeline;
    node.name = "Sequence Data";
    node.parameters = {
        {"dataset_name", kDatasetName},
        {"data_loaded", "true"},
        {"file_category", "ner"},
    };
    return node;
}


bool WaitFor(const std::function<bool()>& predicate,
             std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    do {
        cyxwiz::AsyncTaskManager::Instance().ProcessCompletedCallbacks();
        if (predicate()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (std::chrono::steady_clock::now() < deadline);

    cyxwiz::AsyncTaskManager::Instance().ProcessCompletedCallbacks();
    return predicate();
}

std::string LatestFailedTaskError() {
    for (const auto& task :
         cyxwiz::AsyncTaskManager::Instance().GetRecentTasks(10)) {
        if (task.state == cyxwiz::TaskState::Failed) {
            return task.error_message;
        }
    }
    return {};
}
cyxwiz::TrainingConfiguration MakeSequenceConfig() {
    cyxwiz::TrainingConfiguration config;
    config.is_valid = true;
    config.dataset_name = kDatasetName;
    config.batch_size = 2;
    config.epochs = 1;
    config.sequence_batch.enabled = true;
    config.sequence_batch.token_column = "tokens";
    config.sequence_batch.pos_column = "pos_tags";
    config.sequence_batch.tag_column = "ner_tags";
    config.sequence_batch.create_attention_mask = true;
    config.sequence_batch.max_sequence_length = 8;
    return config;
}

} // namespace

int main() {
    auto& registry = cyxwiz::DataRegistry::Instance();
    registry.UnregisterTabularDataset(kDatasetName);
    Check(registry.RegisterArrowTable(MakeSequenceTable(), kDatasetName) !=
              nullptr,
          "sequence preflight dataset should register");

    const std::vector<gui::MLNode> nodes = {MakeDataInputNode()};
    const std::vector<gui::NodeLink> links;

    std::atomic<bool> dispatch_called{false};
    auto valid_config = MakeSequenceConfig();
    auto valid_result = gui::StartGraphTrainingFromCompiledConfig(
        nodes,
        links,
        std::move(valid_config),
        registry,
        std::weak_ptr<cyxwiz::TrainingPlotPanel>{},
        [](bool) {},
        [&](cyxwiz::TrainingConfiguration,
            const std::string&,
            const std::string&,
            int,
            int,
            std::weak_ptr<cyxwiz::TrainingPlotPanel>,
            std::function<void(bool)>) {
            dispatch_called.store(true);
            return true;
        });
    Check(valid_result.started, valid_result.error_message);
    Check(WaitFor([&] { return dispatch_called.load(); },
                  std::chrono::seconds(5)),
          "valid sequence preflight should call dispatch");

    std::atomic<bool> missing_dispatch_called{false};
    std::atomic<bool> missing_preparation_failed{false};
    auto missing_config = MakeSequenceConfig();
    missing_config.sequence_batch.tag_column = "missing_ner_tags";
    auto missing_result = gui::StartGraphTrainingFromCompiledConfig(
        nodes,
        links,
        std::move(missing_config),
        registry,
        std::weak_ptr<cyxwiz::TrainingPlotPanel>{},
        [&](bool preparing) {
            if (!preparing) {
                missing_preparation_failed.store(true);
            }
        },
        [&](cyxwiz::TrainingConfiguration,
            const std::string&,
            const std::string&,
            int,
            int,
            std::weak_ptr<cyxwiz::TrainingPlotPanel>,
            std::function<void(bool)>) {
            missing_dispatch_called.store(true);
            return true;
        });
    Check(missing_result.started, missing_result.error_message);
    Check(WaitFor([&] { return missing_preparation_failed.load(); },
                  std::chrono::seconds(5)),
          "missing sequence column should fail async preparation");
    Check(!missing_dispatch_called.load(),
          "missing sequence column should not call dispatch");

    const std::string missing_error = LatestFailedTaskError();
    Check(missing_error.find("tag column 'missing_ner_tags'") !=
              std::string::npos,
          "missing sequence column error should name the missing tag column: " +
              missing_error);
    Check(missing_error.find(kDatasetName) != std::string::npos,
          "missing sequence column error should name the dataset: " +
              missing_error);
    Check(missing_result.status_title == "Training launch queued",
          "missing sequence column should initially expose queued status: " +
              missing_result.status_title);

    registry.UnregisterTabularDataset(kDatasetName);
    cyxwiz::AsyncTaskManager::Instance().Shutdown();
    std::cout << "Graph training sequence preflight test passed\n";
    return 0;
}
