#include "data_convert_task.h"
#include "async_task_manager.h"
#include <stdexcept>
#include <utility>

namespace cyxwiz {
std::shared_ptr<AsyncTask> MakeDataConvertTask(
    DataConvertOptions options, std::shared_ptr<DataConvertTaskResult> state) {
    if (!state) throw std::invalid_argument("DataConvert task requires result storage");
    const std::string name = "Convert data to " + options.output_path;
    // No UI/node pointers cross the worker boundary. Do not offer cancellation
    // until the writer supports transactional cancellation without partial files.
    return std::make_shared<LambdaTask>(name,
        [state = std::move(state), options = std::move(options)](LambdaTask& task) {
            try {
                task.ReportProgress(0.05f, "Reading and converting source data");
                state->result = DataConvertService::Convert(options);
            } catch (const std::exception& error) {
                state->result.error = std::string("Data conversion failed: ") + error.what();
            } catch (...) {
                state->result.error = "Data conversion failed with an unknown error.";
            }
            if (state->result.ok) task.MarkCompleted("Conversion complete: " + state->result.output_path);
            else task.MarkFailed(state->result.error);
            state->done.store(true);
        }, false);
}
} // namespace cyxwiz
