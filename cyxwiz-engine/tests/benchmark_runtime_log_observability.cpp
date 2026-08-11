#include "../src/core/runtime_log_inspector.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

cyxwiz::RuntimeLogEvent MakeTrainingEvent(size_t index) {
    cyxwiz::RuntimeLogEvent event;
    event.timestamp_utc = std::chrono::system_clock::now();
    event.level = index % 97 == 0 ? cyxwiz::RuntimeLogLevel::Warning
                                  : cyxwiz::RuntimeLogLevel::Info;
    event.category = index % 8 == 0 ? "device" : "training";
    event.source = "TrainingExecutor";
    event.event_name = index % 97 == 0
        ? "ArrayFire.HostSync"
        : "Training.BatchProgress";
    event.run_id = "benchmark-training-run";
    event.task_id = 42;
    event.backend = "arrayfire_cuda";
    event.device_id = 0;
    event.message = "batch " + std::to_string(index) + " completed";
    event.details.emplace_back("epoch", std::to_string(index / 512));
    event.details.emplace_back("batch", std::to_string(index % 512));
    return event;
}

double ToMilliseconds(std::chrono::steady_clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

double PercentileMilliseconds(
    std::vector<std::chrono::steady_clock::duration> samples,
    double percentile) {
    std::sort(samples.begin(), samples.end());
    const auto index = static_cast<size_t>(
        percentile * static_cast<double>(samples.size() - 1));
    return ToMilliseconds(samples[index]);
}

} // namespace

int main() {
    constexpr size_t kEventCount = 100000;
    constexpr size_t kQueryIterations = 250;
    constexpr uint64_t kAggregatedProducerSuppression = 32000;

    cyxwiz::RuntimeLogStore store(
        cyxwiz::RuntimeLogStore::kDefaultCapacity);
    const auto append_started = std::chrono::steady_clock::now();
    for (size_t index = 0; index < kEventCount; ++index) {
        Check(store.Append(MakeTrainingEvent(index)),
              "representative training event should be accepted");
    }
    const auto append_elapsed =
        std::chrono::steady_clock::now() - append_started;
    store.RecordSuppressed(kAggregatedProducerSuppression);

    cyxwiz::RuntimeLogInspectorRequest request;
    request.criteria.category = "training";
    request.criteria.backend = "arrayfire_cuda";
    request.criteria.text = "BATCH";
    request.display_limit = 1000;

    std::vector<std::chrono::steady_clock::duration> query_samples;
    query_samples.reserve(kQueryIterations);
    cyxwiz::RuntimeLogInspectorResult last_result;
    for (size_t iteration = 0; iteration < kQueryIterations; ++iteration) {
        const auto query_started = std::chrono::steady_clock::now();
        last_result = cyxwiz::QueryRuntimeLogInspector(store, request);
        query_samples.push_back(
            std::chrono::steady_clock::now() - query_started);
    }

    const auto stats = store.GetStats();
    Check(stats.capacity == cyxwiz::RuntimeLogStore::kDefaultCapacity &&
              stats.size == stats.capacity,
          "the representative workload must remain at the fixed store bound");
    Check(stats.evicted_count == kEventCount - stats.capacity,
          "retention eviction must account for every overwritten event");
    Check(stats.suppressed_count == kAggregatedProducerSuppression,
          "aggregated producer suppression must remain visible");
    Check(stats.dropped_count == 0 && stats.rejected_count == 0,
          "valid representative events must not be dropped or rejected");
    Check(!last_result.filter_error.has_value() &&
              last_result.query.events.size() <= request.display_limit &&
              last_result.query.matched_count >=
                  last_result.query.events.size(),
          "query output must remain valid and display-bounded");

    const double append_ms = ToMilliseconds(append_elapsed);
    const double append_us_per_event =
        append_ms * 1000.0 / static_cast<double>(kEventCount);
    const double query_p50_ms =
        PercentileMilliseconds(query_samples, 0.50);
    const double query_p95_ms =
        PercentileMilliseconds(query_samples, 0.95);
    const size_t text_bound_bytes = stats.capacity *
        cyxwiz::RuntimeLogStore::kMaxEventTextBytes;

    std::cout << std::fixed << std::setprecision(3)
              << "Runtime log observability benchmark\n"
              << "workload_events=" << kEventCount
              << " retained=" << stats.size
              << " evicted=" << stats.evicted_count << '\n'
              << "append_total_ms=" << append_ms
              << " append_mean_us_per_event=" << append_us_per_event << '\n'
              << "query_iterations=" << kQueryIterations
              << " query_p50_ms=" << query_p50_ms
              << " query_p95_ms=" << query_p95_ms << '\n'
              << "query_matched=" << last_result.query.matched_count
              << " query_displayed=" << last_result.query.events.size()
              << " query_truncated="
              << (last_result.query.truncated ? "true" : "false") << '\n'
              << "max_event_text_bytes="
              << cyxwiz::RuntimeLogStore::kMaxEventTextBytes
              << " bounded_text_mib="
              << static_cast<double>(text_bound_bytes) / (1024.0 * 1024.0)
              << '\n'
              << "producer_suppressed=" << stats.suppressed_count
              << " dropped=" << stats.dropped_count
              << " rejected=" << stats.rejected_count << '\n';
    return 0;
}
