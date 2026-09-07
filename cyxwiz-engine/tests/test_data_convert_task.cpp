#include "core/data_convert_task.h"
#include "core/async_task_manager.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>

namespace {
int checks = 0;
void Check(bool ok, const char* message) {
    ++checks;
    if (!ok) throw std::runtime_error(message);
}
}

int main() try {
    namespace fs = std::filesystem;
    const auto folder = fs::temp_directory_path() / ("cyxwiz_convert_task_" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Check(fs::create_directory(folder), "unique fixture directory");
    struct Cleanup { fs::path path; ~Cleanup() { std::error_code ec; fs::remove_all(path, ec); } } cleanup{folder};
    const auto input = folder / "source.csv";
    { std::ofstream file(input); file << "id,value\n1,2\n3,4\n"; }
    cyxwiz::DataConvertOptions options;
    options.input_path = input.string();
    options.output_path = (folder / "written.parquet").string();
    const auto expected_output = options.output_path;
    auto state = std::make_shared<cyxwiz::DataConvertTaskResult>();
    auto task = cyxwiz::MakeDataConvertTask(options, state);
    Check(!task->IsCancellable(), "file publication must not advertise cancellation");
    task->RequestCancel();
    Check(!task->IsCancelRequested(), "unsupported cancellation must not relabel the write");
    options.input_path = "changed-after-queue";
    options.output_path = (folder / "wrong.parquet").string();
    // Execute the production task on a worker, without needing a desktop GUI.
    auto worker = std::async(std::launch::async, [task] { task->Execute(); });
    worker.get();
    Check(state->done.load(), "completion is published");
    Check(state->result.ok && state->result.rows_written == 2, "worker conversion result");
    Check(task->GetState() == cyxwiz::TaskState::Completed, "successful task state");
    Check(fs::exists(expected_output) && !fs::exists(options.output_path), "queued options are immutable copies");

    state = std::make_shared<cyxwiz::DataConvertTaskResult>();
    task = cyxwiz::MakeDataConvertTask(options, state);
    worker = std::async(std::launch::async, [task] { task->Execute(); });
    worker.get();
    Check(state->done.load() && !state->result.ok && !state->result.error.empty(), "failure is published");
    Check(task->GetState() == cyxwiz::TaskState::Failed, "failure must reach the task panel");
    Check(!fs::exists(options.output_path), "failed source does not create output");

    options.input_path = input.string();
    options.output_path = (folder / "detached.parquet").string();
    state = std::make_shared<cyxwiz::DataConvertTaskResult>();
    std::weak_ptr<cyxwiz::DataConvertTaskResult> observer = state;
    task = cyxwiz::MakeDataConvertTask(options, state);
    state.reset(); // Dialog destruction: the worker must own its own storage.
    worker = std::async(std::launch::async, [task] { task->Execute(); });
    worker.get();
    Check(fs::exists(options.output_path) && task->GetState() == cyxwiz::TaskState::Completed,
          "closing UI does not cancel an authorized file write");
    task.reset();
    Check(observer.expired(), "task result storage releases after its final owner");
    std::cout << "DataConvert task: " << checks << " checks passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
