#include "../src/core/runtime_log_sink.h"
#include "../src/core/runtime_log_store.h"

#include <spdlog/logger.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

cyxwiz::RuntimeLogEvent MakeEvent(std::string message) {
    cyxwiz::RuntimeLogEvent event;
    event.category = "test";
    event.source = "runtime_log_store_test";
    event.event_name = "contract";
    event.message = std::move(message);
    return event;
}

void TestCapacityValidation() {
    bool threw = false;
    try {
        cyxwiz::RuntimeLogStore invalid(0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    Check(threw, "zero-capacity stores must be rejected");
}

void TestRetentionAndSnapshots() {
    cyxwiz::RuntimeLogStore store(3);
    Check(store.Append(MakeEvent("one")), "first append should succeed");
    Check(store.Append(MakeEvent("two")), "second append should succeed");
    Check(store.Append(MakeEvent("three")), "third append should succeed");
    Check(store.Append(MakeEvent("four")), "fourth append should succeed");

    const auto snapshot = store.Snapshot();
    Check(snapshot.events.size() == 3, "snapshot should respect capacity");
    Check(snapshot.events[0].message == "two" &&
              snapshot.events[0].sequence == 2,
          "oldest retained event should follow the evicted event");
    Check(snapshot.events[2].message == "four" &&
              snapshot.events[2].sequence == 4,
          "newest retained event should preserve insertion order");
    Check(snapshot.stats.oldest_sequence == 2 &&
              snapshot.stats.newest_sequence == 4,
          "snapshot should report the retained sequence range");
    Check(snapshot.stats.evicted_count == 1,
          "overwriting one retained slot should count one eviction");

    cyxwiz::RuntimeLogSnapshotRequest request;
    request.after_sequence = 2;
    request.through_sequence = 4;
    request.limit = 1;
    const auto limited = store.Snapshot(request);
    Check(limited.events.size() == 1 && limited.events[0].sequence == 3,
          "bounded snapshots should return the earliest matching sequence");
    Check(limited.truncated, "bounded snapshots should report truncation");
}

void TestValidationAndCounters() {
    cyxwiz::RuntimeLogStore store(4);
    auto oversized = MakeEvent(std::string(
        cyxwiz::RuntimeLogStore::kMaxMessageBytes + 1, 'x'));
    Check(!store.Append(std::move(oversized)),
          "oversized messages should be rejected");

    auto invalid_code = MakeEvent("invalid diagnostic");
    invalid_code.primary_error_code = "CW01";
    Check(!store.Append(std::move(invalid_code)),
          "ambiguous compact diagnostic codes should be rejected");

    store.RecordDropped(2);
    store.RecordSuppressed(3);
    const auto stats = store.GetStats();
    Check(stats.rejected_count == 2,
          "validation failures should be counted as rejected");
    Check(stats.dropped_count == 2,
          "explicit ingestion drops should be accounted for");
    Check(stats.suppressed_count == 3,
          "producer suppression should be accounted for separately");
}

void TestConcurrentProducers() {
    constexpr int kThreadCount = 4;
    constexpr int kEventsPerThread = 500;
    cyxwiz::RuntimeLogStore store(kThreadCount * kEventsPerThread);
    std::vector<std::thread> producers;

    for (int thread = 0; thread < kThreadCount; ++thread) {
        producers.emplace_back([thread, &store]() {
            for (int event_index = 0; event_index < kEventsPerThread;
                 ++event_index) {
                auto event = MakeEvent(
                    std::to_string(thread) + ":" +
                    std::to_string(event_index));
                Check(store.Append(std::move(event)),
                      "concurrent append should succeed");
            }
        });
    }
    for (auto& producer : producers) {
        producer.join();
    }

    cyxwiz::RuntimeLogSnapshotRequest request;
    request.limit = kThreadCount * kEventsPerThread;
    const auto snapshot = store.Snapshot(request);
    Check(snapshot.events.size() == request.limit,
          "all concurrent events should be retained");
    for (size_t index = 1; index < snapshot.events.size(); ++index) {
        Check(snapshot.events[index - 1].sequence + 1 ==
                  snapshot.events[index].sequence,
              "concurrent events should have a strict total order");
    }
    Check(snapshot.stats.dropped_count == 0 &&
              snapshot.stats.rejected_count == 0,
          "valid concurrent producers should not lose events");
}

void TestSpdlogSinkAndDiagnosticExtraction() {
    cyxwiz::RuntimeLogStore store(8);
    auto sink = std::make_shared<cyxwiz::RuntimeLogSinkMt>(store);
    spdlog::logger logger("runtime-test", sink);
    logger.set_level(spdlog::level::trace);
    logger.warn("[CW-G-0501] Kernel execution failed");

    const auto snapshot = store.Snapshot();
    Check(snapshot.events.size() == 1,
          "the spdlog adapter should append one runtime event");
    const auto& event = snapshot.events.front();
    Check(event.level == cyxwiz::RuntimeLogLevel::Warning,
          "spdlog warning should retain severity");
    Check(event.category == "system" && event.source == "runtime-test",
          "legacy logs should retain canonical category and logger source");
    Check(event.primary_error_code == "CW-G-0501",
          "legacy diagnostic prefix should be extracted once at ingestion");
    Check(event.message == "[CW-G-0501] Kernel execution failed",
          "legacy text should remain available for raw search");
    Check(event.timestamp_utc.time_since_epoch().count() != 0,
          "sink events should carry a system-clock timestamp");

    Check(cyxwiz::IsCanonicalDiagnosticCode("CW-C-0101"),
          "canonical diagnostic code should validate");
    Check(!cyxwiz::IsCanonicalDiagnosticCode("cw-c-0101") &&
              !cyxwiz::IsCanonicalDiagnosticCode("CW01"),
          "stored diagnostic codes should require canonical form");
    Check(!cyxwiz::ExtractLegacyDiagnosticCode("prefix [CW-C-0101]")
               .has_value(),
          "legacy extraction should only accept a leading code prefix");
}

} // namespace

int main() {
    TestCapacityValidation();
    TestRetentionAndSnapshots();
    TestValidationAndCounters();
    TestConcurrentProducers();
    TestSpdlogSinkAndDiagnosticExtraction();
    std::cout << "Runtime log store contracts passed\n";
    return 0;
}
