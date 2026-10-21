#include <catch2/catch_test_macros.hpp>
#include <cyxwiz/cyxwiz.h>
#include <atomic>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

TEST_CASE("Backend lifecycle tolerates concurrent initialize calls", "[engine]") {
    constexpr int thread_count = 4;
    std::atomic<int> initialized{0};
    const std::string expected_version =
        std::to_string(CYXWIZ_VERSION_MAJOR) + "." +
        std::to_string(CYXWIZ_VERSION_MINOR) + "." +
        std::to_string(CYXWIZ_VERSION_PATCH);
    std::atomic<int> version_matches{0};
    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    for (int i = 0; i < thread_count; ++i) {
        threads.emplace_back([&]() {
            if (cyxwiz::Initialize()) {
                initialized.fetch_add(1, std::memory_order_relaxed);
            }
            if (std::strcmp(cyxwiz::GetVersionString(), expected_version.c_str()) == 0) {
                version_matches.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    REQUIRE(initialized.load(std::memory_order_relaxed) == thread_count);
    REQUIRE(version_matches.load(std::memory_order_relaxed) == thread_count);
    cyxwiz::Shutdown();
}
