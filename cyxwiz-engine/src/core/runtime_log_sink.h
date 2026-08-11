#pragma once

#include "runtime_log_store.h"

#include <spdlog/details/null_mutex.h>
#include <spdlog/sinks/base_sink.h>

#include <mutex>
#include <string>

namespace cyxwiz {

template <typename Mutex>
class RuntimeLogSink final : public spdlog::sinks::base_sink<Mutex> {
public:
    explicit RuntimeLogSink(RuntimeLogStore& store) : store_(store) {}

protected:
    void sink_it_(const spdlog::details::log_msg& message) override {
        try {
            RuntimeLogEvent event;
            event.timestamp_utc = message.time;
            event.level = ToRuntimeLevel(message.level);
            event.category = "system";
            event.source.assign(
                message.logger_name.data(), message.logger_name.size());
            event.event_name = "spdlog";
            event.thread_id = std::to_string(message.thread_id);
            event.message.assign(message.payload.data(), message.payload.size());

            if (const auto code = ExtractLegacyDiagnosticCode(event.message)) {
                event.primary_error_code = *code;
            }
            if (!message.source.empty()) {
                event.details.emplace_back(
                    "source_file",
                    message.source.filename ? message.source.filename : "");
                event.details.emplace_back(
                    "source_line", std::to_string(message.source.line));
                event.details.emplace_back(
                    "source_function",
                    message.source.funcname ? message.source.funcname : "");
            }

            store_.Append(std::move(event));
        } catch (...) {
            store_.RecordDropped();
        }
    }

    void flush_() override {}

private:
    static RuntimeLogLevel ToRuntimeLevel(spdlog::level::level_enum level) {
        switch (level) {
            case spdlog::level::trace: return RuntimeLogLevel::Trace;
            case spdlog::level::debug: return RuntimeLogLevel::Debug;
            case spdlog::level::info: return RuntimeLogLevel::Info;
            case spdlog::level::warn: return RuntimeLogLevel::Warning;
            case spdlog::level::err: return RuntimeLogLevel::Error;
            case spdlog::level::critical: return RuntimeLogLevel::Critical;
            case spdlog::level::off: return RuntimeLogLevel::Info;
            case spdlog::level::n_levels: return RuntimeLogLevel::Info;
        }
        return RuntimeLogLevel::Info;
    }

    RuntimeLogStore& store_;
};

using RuntimeLogSinkMt = RuntimeLogSink<std::mutex>;
using RuntimeLogSinkSt = RuntimeLogSink<spdlog::details::null_mutex>;

} // namespace cyxwiz
