#pragma once

#include "console.h"
#include <spdlog/sinks/base_sink.h>
#include <mutex>
#include <memory>

namespace gui {

// Custom spdlog sink that forwards log messages to the Console panel
template<typename Mutex>
class ConsoleSink : public spdlog::sinks::base_sink<Mutex> {
public:
    explicit ConsoleSink(Console* console) : console_(console) {}

protected:
    void sink_it_(const spdlog::details::log_msg& msg) override {
        if (!console_) return;

        // Format the message
        spdlog::memory_buf_t formatted;
        spdlog::sinks::base_sink<Mutex>::formatter_->format(msg, formatted);
        std::string message = fmt::to_string(formatted);

        // Remove trailing newline if present
        if (!message.empty() && message.back() == '\n') {
            message.pop_back();
        }

        // Map spdlog level to Console LogLevel
        switch (msg.level) {
            case spdlog::level::trace:
            case spdlog::level::debug:
            case spdlog::level::info:
                console_->AddInfo(message);
                break;
            case spdlog::level::warn:
                console_->AddWarning(message);
                break;
            case spdlog::level::err:
            case spdlog::level::critical:
                console_->AddError(message);
                break;
            default:
                console_->AddInfo(message);
                break;
        }
    }

    void flush_() override {
        // Nothing to flush for immediate GUI display
    }

private:
    Console* console_;
};

using ConsoleSinkMt = ConsoleSink<std::mutex>;
using ConsoleSinkSt = ConsoleSink<spdlog::details::null_mutex>;

} // namespace gui
