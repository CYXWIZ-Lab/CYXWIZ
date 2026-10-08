#include "core/arrow_dataset.h"
#include "core/checkpoint_manager.h"
#include "core/sequence_arrow_batcher.h"
#include "core/test_executor.h"
#include "core/test_manager.h"
#include <cyxwiz/cyxwiz.h>
#include <cyxwiz/device.h>
#include <arrow/api.h>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <thread>

namespace {
void Check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
cyxwiz::DeviceType ParseDevice(const std::string& name) {
    if (name == "cpu") return cyxwiz::DeviceType::CPU;
    if (name == "cuda") return cyxwiz::DeviceType::CUDA;
    if (name == "opencl") return cyxwiz::DeviceType::OPENCL;
    if (name == "oneapi") return cyxwiz::DeviceType::ONEAPI;
    throw std::runtime_error("Use cpu, cuda, opencl or oneapi");
}
std::shared_ptr<cyxwiz::SequentialModel> MakeModel(size_t vocabulary) {
    auto model = std::make_shared<cyxwiz::SequentialModel>();
    model->Add<cyxwiz::EmbeddingModule>(vocabulary, 4);
    model->Add<cyxwiz::TimeDistributedDenseModule>(4, vocabulary);
    return model;
}
}

int main(int argc, char** argv) {
    try {
        Check(argc == 3, "usage: test_testing_device_handoff <owner> <worker>");
        cyxwiz::Initialize();
        const auto owner = ParseDevice(argv[1]);
        const auto worker = ParseDevice(argv[2]);
        Check(cyxwiz::Device(owner, 0).ActivateExact(true).success, "owner activation");
        arrow::StringBuilder tokens, groups;
        for (auto text : {"x x y", "x y", "y x y"})
            Check(tokens.Append(text).ok(), "tokens append");
        for (auto group : {"a", "b", "c"})
            Check(groups.Append(group).ok(), "groups append");
        std::shared_ptr<arrow::Array> t, g;
        Check(tokens.Finish(&t).ok() && groups.Finish(&g).ok(), "fixture arrays");
        auto dataset = std::make_shared<cyxwiz::ArrowDataset>(arrow::Table::Make(
            arrow::schema({arrow::field("tokens", arrow::utf8()),
                           arrow::field("group", arrow::utf8())}), {t, g}), "device_handoff");
        cyxwiz::TrainingConfiguration config;
        config.sequence_batch.enabled = true;
        config.sequence_batch.create_causal_lm_targets = true;
        config.sequence_batch.create_attention_mask = true;
        config.sequence_batch.token_column = "tokens";
        config.sequence_batch.sentence_id_column = "group";
        config.sequence_batch.max_sequence_length = 4;
        config.shuffle = false;
        config.drop_last = false;
        config.train_ratio = 1.0f;
        config.val_ratio = config.test_ratio = 0.0f;
        config.forbid_native_cpu_fallback = true;
        config.loss_type = gui::NodeType::CrossEntropyLoss;
        config.loss_params["ignore_index"] = "-100";
        config.loss_params["reduction"] = "mean";
        auto batchers = cyxwiz::BuildSequenceBatcherFromArrowDataset(dataset, config, 2);
        Check(batchers.success(), "sequence fixture");
        cyxwiz::ApplySequenceBatcherBuildResultToTrainingConfig(batchers, config);

        const auto directory = std::filesystem::temp_directory_path() /
            ("cyxwiz_test_device_" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
        cyxwiz::CheckpointManager checkpoint(directory.string());
        auto initial = MakeModel(config.output_size);
        cyxwiz::TrainingMetrics saved;
        saved.current_epoch = 1;
        Check(!checkpoint.SaveCheckpoint(*initial, nullptr, saved, "best").empty(), "checkpoint save");
        auto model = MakeModel(config.output_size);
        Check(checkpoint.LoadCheckpoint(*model, nullptr, "best").has_value(), "checkpoint load");
        cyxwiz::TestExecutor reference(config, dataset, "", cyxwiz::TestDatasetScope::EntireProvidedDataset);
        reference.SetModel(model);
        reference.Test(2);
        const auto expected = reference.GetMetrics();
        Check(expected.is_complete && expected.total_target_values == 5, "reference scores five targets");

        cyxwiz::TestExecutor test(config, dataset, "", cyxwiz::TestDatasetScope::EntireProvidedDataset);
        test.SetModel(model); // Legacy API: called on the model-owning thread.
        std::exception_ptr error;
        std::thread evaluation([&] {
            try {
                Check(cyxwiz::Device(worker, 0).ActivateExact(true).success, "worker activation");
                test.Test(2);
                Check(cyxwiz::Device::GetCurrentDevice()->GetType() == owner, "testing activated model owner");
            } catch (...) { error = std::current_exception(); }
        });
        evaluation.join();
        if (error) std::rethrow_exception(error);
        const auto actual = test.GetMetrics();
        Check(actual.is_complete && !test.IsTesting(), "worker completion");
        Check(actual.total_target_values == expected.total_target_values, "same target count");
        Check(std::abs(actual.test_loss - expected.test_loss) < 1e-5f, "same loss");
        Check(std::abs(actual.test_accuracy - expected.test_accuracy) < 1e-6f, "same accuracy");

        // The UI thread may now select a different device from the stored model.
        // Explicit ownership must win over BOTH the caller and process selection.
        Check(cyxwiz::Device(worker, 0).ActivateExact(true).success, "alternate caller activation");
        cyxwiz::Device::RecordProcessDevice(worker, 0);
        const cyxwiz::ProcessDeviceSelection model_device{owner, 0};
        cyxwiz::TestExecutor explicit_owner(config, dataset, "", cyxwiz::TestDatasetScope::EntireProvidedDataset);
        explicit_owner.SetModel(model, model_device);
        explicit_owner.Test(2);
        Check(std::abs(explicit_owner.GetMetrics().test_loss - expected.test_loss) < 1e-5f,
              "explicit owner overrides caller default");
        Check(cyxwiz::Device::GetProcessDevice()->type == worker, "test does not change process selection");

        // An unavailable owner is an error before any scoring, never permission
        // to run these weights on a fallback backend. Reuse verifies stale metric cleanup.
        explicit_owner.SetModel(model, cyxwiz::ProcessDeviceSelection{owner, 1000000});
        bool rejected = false;
        try { explicit_owner.Test(2); }
        catch (const std::exception& failure) {
            const std::string message = failure.what();
            rejected = message.find("No fallback was attempted") != std::string::npos;
        }
        Check(rejected && !explicit_owner.IsTesting(), "unavailable owner rejected with cleanup");
        Check(!explicit_owner.GetMetrics().is_complete && explicit_owner.GetMetrics().total_target_values == 0,
              "failed activation cannot expose earlier success");
        explicit_owner.SetModel(model, model_device);
        explicit_owner.Test(1, [&](int, int, float) { explicit_owner.Stop(); });
        Check(!explicit_owner.GetMetrics().is_complete && !explicit_owner.IsTesting(), "cancel is not completion");

        // Exercise the same public manager admission used by Run Test in the GUI.
        Check(cyxwiz::Device(worker, 0).ActivateExact(true).success, "manager caller activation");
        auto& tasks = cyxwiz::AsyncTaskManager::Instance();
        tasks.Initialize(2);
        auto& manager = cyxwiz::TestManager::Instance();
        bool completed = false;
        const auto ui_thread = std::this_thread::get_id();
        Check(manager.StartTestingArrow(config, dataset, "",
            cyxwiz::TestDatasetScope::EntireProvidedDataset, 2, model,
            [&](const cyxwiz::TestingMetrics& metrics) {
                Check(std::this_thread::get_id() == ui_thread, "manager callback belongs to UI thread");
                Check(metrics.is_complete && std::abs(metrics.test_loss - expected.test_loss) < 1e-5f,
                      "manager preserves checkpoint scores across devices");
                completed = true;
            }, model_device), "manager admission");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (manager.IsTestingActive() || !completed) {
            Check(std::chrono::steady_clock::now() < deadline, "manager timeout");
            tasks.ProcessCompletedCallbacks();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        manager.WaitForTestingStop();
        tasks.Shutdown();
        tasks.ProcessCompletedCallbacks();
        Check(cyxwiz::Device::GetProcessDevice()->type == worker, "manager preserves process selection");
        Check(cyxwiz::Device(owner, 0).ActivateExact(false).success, "restore owner for tensor teardown");
        std::cout << "PASS: checkpoint owner=" << argv[1] << " worker=" << argv[2]
                  << " targets=" << actual.total_target_values << " loss=" << actual.test_loss << '\n';
        // Keep the tiny unique fixture for inspection; never remove caller paths.
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
