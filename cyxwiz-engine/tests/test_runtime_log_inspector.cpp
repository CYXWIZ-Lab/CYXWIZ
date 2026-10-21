#include "../src/core/runtime_log_inspector.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

cyxwiz::RuntimeLogEvent MakeEvent(
    std::string category, cyxwiz::RuntimeLogLevel level,
    std::string message) {
    cyxwiz::RuntimeLogEvent event;
    event.timestamp_utc = std::chrono::system_clock::now();
    event.category = std::move(category);
    event.level = level;
    event.source = "inspector_test";
    event.message = std::move(message);
    return event;
}

void Populate(cyxwiz::RuntimeLogStore& store) {
    Check(store.Append(MakeEvent(
              "system", cyxwiz::RuntimeLogLevel::Info, "startup")),
          "startup append should succeed");

    auto cuda = MakeEvent(
        "device", cyxwiz::RuntimeLogLevel::Info, "CUDA device bound");
    cuda.run_id = "train-1";
    cuda.backend = "arrayfire_cuda";
    cuda.device_id = 0;
    Check(store.Append(std::move(cuda)), "device append should succeed");

    auto warning = MakeEvent(
        "training", cyxwiz::RuntimeLogLevel::Warning,
        "native CPU fallback");
    warning.run_id = "train-1";
    warning.task_id = 7;
    warning.backend = "arrayfire_cuda";
    warning.device_id = 0;
    warning.primary_error_code = "CW-G-0501";
    Check(store.Append(std::move(warning)), "warning append should succeed");

    auto error = MakeEvent(
        "training", cyxwiz::RuntimeLogLevel::Error, "training failed");
    error.run_id = "train-2";
    error.task_id = 8;
    error.primary_error_code = "CW-T-0501";
    Check(store.Append(std::move(error)), "error append should succeed");
}

void TestCriteriaCompositionAndQuery() {
    cyxwiz::RuntimeLogStore store(8);
    Populate(store);

    cyxwiz::RuntimeLogInspectorRequest request;
    request.criteria.levels = {false, false, false, true, true, true};
    request.criteria.category = "training";
    request.criteria.run_id = "train-1";
    request.criteria.text = "FALLBACK";
    const auto result = cyxwiz::QueryRuntimeLogInspector(store, request);

    Check(!result.filter_error.has_value() &&
              result.query.matched_count == 1 &&
              result.query.events.size() == 1 &&
              result.query.events[0].sequence == 3,
          "criteria should compose through the shared parser and evaluator");
    Check(result.effective_filter.find("level=warn") != std::string::npos &&
              result.effective_filter.find("category=\"training\"") !=
                  std::string::npos &&
              result.effective_filter.find("message contains \"FALLBACK\"") !=
                  std::string::npos,
          "effective filter should remain inspectable and deterministic");

    Check(result.query.facets.categories ==
              std::vector<std::string>({"device", "system", "training"}) &&
              result.query.facets.codes ==
                  std::vector<std::string>({"CW-G-0501", "CW-T-0501"}) &&
              result.query.facets.run_ids ==
                  std::vector<std::string>({"train-1", "train-2"}) &&
              result.query.facets.task_ids == std::vector<uint64_t>({7, 8}),
          "facets should describe the unfiltered retained snapshot");
}

void TestStructuredFilterErrorsAndNewestLimit() {
    cyxwiz::RuntimeLogStore store(8);
    Populate(store);

    cyxwiz::RuntimeLogInspectorRequest invalid;
    invalid.criteria.structured_filter = "unknown=value";
    const auto rejected = cyxwiz::QueryRuntimeLogInspector(store, invalid);
    Check(rejected.filter_error.has_value() && rejected.query.events.empty(),
          "invalid structured filters should remain visible as parse errors");

    cyxwiz::RuntimeLogInspectorRequest latest;
    latest.display_limit = 2;
    const auto bounded = cyxwiz::QueryRuntimeLogInspector(store, latest);
    Check(bounded.query.events.size() == 2 &&
              bounded.query.events[0].sequence == 3 &&
              bounded.query.events[1].sequence == 4 &&
              bounded.query.truncated,
          "inspector limits should retain the newest matching rows");

    latest.through_sequence = 3;
    const auto paused = cyxwiz::QueryRuntimeLogInspector(store, latest);
    Check(paused.query.high_water_sequence == 3 &&
              paused.query.events.back().sequence == 3,
          "a frozen high-water mark should produce a stable paused view");
}

void TestNoLevelsMatchesNothing() {
    cyxwiz::RuntimeLogStore store(2);
    Check(store.Append(MakeEvent(
              "system", cyxwiz::RuntimeLogLevel::Info, "startup")),
          "fixture append should succeed");
    cyxwiz::RuntimeLogInspectorRequest request;
    request.criteria.levels.fill(false);
    const auto result = cyxwiz::QueryRuntimeLogInspector(store, request);
    Check(!result.filter_error.has_value() && result.query.matched_count == 0,
          "disabling every level should return an empty valid result");
}

void TestHighVolumeBoundAndQueryParity() {
    cyxwiz::RuntimeLogStore store(4096);
    const auto started = std::chrono::steady_clock::now();
    for (size_t index = 0; index < 50000; ++index) {
        auto event = MakeEvent(
            index % 2 == 0 ? "training" : "system",
            index % 7 == 0 ? cyxwiz::RuntimeLogLevel::Warning
                           : cyxwiz::RuntimeLogLevel::Info,
            "synthetic event " + std::to_string(index));
        Check(store.Append(std::move(event)),
              "high-volume append should remain accepted");
    }

    cyxwiz::RuntimeLogInspectorRequest request;
    request.criteria.category = "training";
    request.criteria.levels = {false, false, false, true, true, true};
    request.display_limit = 1000;
    const auto inspector = cyxwiz::QueryRuntimeLogInspector(store, request);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started);
    Check(elapsed < std::chrono::seconds(2),
          "bounded append and inspector query should stay responsive");
    Check(inspector.query.store_stats.size == 4096 &&
              inspector.query.store_stats.evicted_count == 45904 &&
              inspector.query.events.size() <= 1000,
          "high-volume input should remain bounded with visible eviction");

    auto parsed = cyxwiz::ParseRuntimeLogFilter(inspector.effective_filter);
    Check(parsed.Ok(), "inspector effective filter should parse for parity");
    cyxwiz::RuntimeLogQueryRequest direct_request;
    direct_request.filter = &*parsed.filter;
    direct_request.limit = store.GetStats().capacity;
    const auto direct =
        cyxwiz::RuntimeLogQueryService(store).Query(direct_request);
    Check(inspector.query.matched_count == direct.matched_count &&
              !inspector.query.events.empty() && !direct.events.empty() &&
              inspector.query.events.back().sequence ==
                  direct.events.back().sequence,
          "GUI inspector and direct query should share matched counts and high-water sequence");
}

} // namespace

int main() {
    TestCriteriaCompositionAndQuery();
    TestStructuredFilterErrorsAndNewestLimit();
    TestNoLevelsMatchesNothing();
    TestHighVolumeBoundAndQueryParity();
    std::cout << "Runtime log inspector contracts passed\n";
    return 0;
}
