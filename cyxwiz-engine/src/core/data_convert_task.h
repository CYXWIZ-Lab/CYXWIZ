#pragma once

#include "data_convert_service.h"
#include <atomic>
#include <memory>

namespace cyxwiz {
class AsyncTask;

// The worker exclusively writes result until done publishes it. The UI then
// exclusively consumes it; dropping the UI reference does not cancel the write.
struct DataConvertTaskResult {
    std::atomic<bool> done{false};
    DataConvertResult result;
};

std::shared_ptr<AsyncTask> MakeDataConvertTask(
    DataConvertOptions options, std::shared_ptr<DataConvertTaskResult> result);
} // namespace cyxwiz
