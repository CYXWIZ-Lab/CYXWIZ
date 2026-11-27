// The backend logs through the host program's logger once it is handed over
// (cyxwiz::UseHostLogger), also where spdlog is linked statically and the
// backend keeps its own private copy (Linux, macOS).
#include <cyxwiz/cyxwiz.h>
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/spdlog.h>

#include <cstdio>
#include <memory>
#include <sstream>
#include <string>

int main() {
    std::ostringstream lines;
    auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(lines);
    auto logger = std::make_shared<spdlog::logger>("host", sink);
    logger->set_level(spdlog::level::info);
    spdlog::set_default_logger(logger);
    cyxwiz::UseHostLogger(logger);
    cyxwiz::UseHostLogger(nullptr);  // ignored: the host's logger stays

    if (!cyxwiz::Initialize()) {
        std::fprintf(stderr, "FAIL: the backend did not initialize\n");
        return 1;
    }
    cyxwiz::Shutdown();
    logger->flush();

    const std::string text = lines.str();
    if (text.find("Shutting down CyxWiz Backend") == std::string::npos) {
        std::fprintf(stderr, "FAIL: the backend's lines did not reach the host's logger:\n%s\n", text.c_str());
        return 1;
    }
    std::printf("backend host logger: the backend's lines reach the host's logger\n");
    return 0;
}
