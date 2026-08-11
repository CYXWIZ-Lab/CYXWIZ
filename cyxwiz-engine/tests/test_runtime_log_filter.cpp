#include "../src/core/runtime_log_filter.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

cyxwiz::RuntimeLogFilter Parse(const std::string& expression) {
    auto result = cyxwiz::ParseRuntimeLogFilter(expression);
    if (!result.Ok()) {
        const auto& error = *result.error;
        std::cerr << "FAIL: could not parse '" << expression << "' at "
                  << error.position << ": " << error.message << "\n";
        std::exit(1);
    }
    return std::move(*result.filter);
}

cyxwiz::RuntimeLogEvent MakeEvent(
    std::string category, cyxwiz::RuntimeLogLevel level,
    std::string message = {}) {
    cyxwiz::RuntimeLogEvent event;
    event.category = std::move(category);
    event.level = level;
    event.source = "GraphCompiler";
    event.event_name = "compile.validation";
    event.run_id = "run-42";
    event.task_id = 17;
    event.thread_id = "9";
    event.backend = "arrayfire_cuda";
    event.device_id = 1;
    event.device_name = "NVIDIA_GeForce_GTX_1050_Ti";
    event.node_id = 23;
    event.dataset_name = "training.parquet";
    event.diagnostic_phase = "preflight";
    event.component = "GraphCompiler";
    event.message = std::move(message);
    return event;
}

void TestBooleanPrecedenceAndParentheses() {
    const auto precedence = Parse(
        "category=training or category=device and level>=warn");
    const auto training_info = MakeEvent(
        "training", cyxwiz::RuntimeLogLevel::Info);
    const auto device_info = MakeEvent("device", cyxwiz::RuntimeLogLevel::Info);
    const auto device_warning = MakeEvent(
        "device", cyxwiz::RuntimeLogLevel::Warning);

    Check(precedence.Matches(training_info),
          "and should bind more tightly than or");
    Check(!precedence.Matches(device_info),
          "device info should fail the warning constraint");
    Check(precedence.Matches(device_warning),
          "device warning should satisfy the right branch");

    const auto grouped = Parse(
        "(category=training or category=device) and level>=warn");
    Check(!grouped.Matches(training_info),
          "parentheses should apply warning level to both categories");
    Check(grouped.Matches(device_warning),
          "grouped expression should retain valid warning events");

    const auto negated = Parse("not (category=test or level<info)");
    Check(negated.Matches(training_info),
          "not should negate a grouped expression");
}

void TestStringsAliasesAndEscapes() {
    auto event = MakeEvent(
        "training", cyxwiz::RuntimeLogLevel::Error,
        "ArrayFire.HostSync: native \"CPU\" fallback");

    Check(Parse("message contains \"native \\\"CPU\\\" fallback\"")
              .Matches(event),
          "quoted strings should support escaped quotes");
    Check(Parse("event=compile.validation and run=run-42")
              .Matches(event),
          "event and run aliases should map to structured fields");
    Check(Parse("source=GraphCompiler and component!=TrainingExecutor")
              .Matches(event),
          "string equality and inequality should be field-aware");
    Check(Parse("category=TRAINING").Matches(event),
          "category input should normalize to lower case");
    Check(Parse("message contains arrayfire").Matches(event) &&
              Parse("message contains FALLBACK").Matches(event),
          "message containment should be case-insensitive");
}

void TestTypedComparisons() {
    const auto event = MakeEvent("training", cyxwiz::RuntimeLogLevel::Warning);
    Check(Parse("level>=WARN and level<critical").Matches(event),
          "level comparison should be ordered and case-insensitive");
    Check(Parse("task_id=17 and device_id>0 and node_id<=23")
              .Matches(event),
          "numeric fields should compare parsed integers");
    Check(!Parse("task=18 or device_id=-1").Matches(event),
          "numeric mismatches should not fall back to text comparison");
}

void TestDiagnosticCodeMatching() {
    auto event = MakeEvent(
        "training", cyxwiz::RuntimeLogLevel::Error,
        "Training failed after backend error");
    event.primary_error_code = "CW-T-0501";
    event.issue_codes = {"CW-G-0501", "CW-M-0702"};

    Check(Parse("error_code=cw-t-0501").Matches(event),
          "exact diagnostic input should normalize to canonical upper case");
    Check(Parse("code=CW-G-0501").Matches(event),
          "exact matching should inspect issue_codes");
    Check(Parse("run_id=run-42 and error_code=CW-T-0501").Matches(event),
          "run and diagnostic predicates should compose over one event");
    Check(!Parse("run_id=run-41 and error_code=CW-T-0501").Matches(event),
          "run constraints should not be ignored during code matching");
    Check(Parse("error_code matches cw-g-*").Matches(event),
          "family matching should inspect all attached codes");
    Check(!Parse("error_code matches CW-C-*").Matches(event),
          "unrelated diagnostic families should not match");
    Check(Parse("error_code!=CW-C-0101").Matches(event),
          "code inequality should require no attached exact match");
    Check(!Parse("error_code!=CW-G-0501").Matches(event),
          "code inequality should fail when any attached code matches");

    Check(cyxwiz::NormalizeDiagnosticCode("cw-c-0101") == "CW-C-0101",
          "exact code normalization should be reusable");
    Check(cyxwiz::NormalizeDiagnosticFamily("cw-p-*") == "CW-P-*",
          "family normalization should be reusable");
}

void TestParseErrors() {
    const auto unknown = cyxwiz::ParseRuntimeLogFilter("unknown=value");
    Check(!unknown.Ok() && unknown.error &&
              unknown.error->position == 0 &&
              unknown.error->message.find("unknown filter field") !=
                  std::string::npos,
          "unknown fields should return a positioned parse error");

    const auto compact = cyxwiz::ParseRuntimeLogFilter("error_code=CW01");
    Check(!compact.Ok() && compact.error &&
              compact.error->position == 11 &&
              compact.error->message.find("CW-C-0101") != std::string::npos,
          "compact diagnostic codes should identify the value and explain "
          "canonical format");

    Check(!cyxwiz::ParseRuntimeLogFilter("message matches fallback").Ok(),
          "matches should be restricted to diagnostic families");
    Check(!cyxwiz::ParseRuntimeLogFilter("level contains warn").Ok(),
          "contains should be rejected for ordered levels");
    Check(!cyxwiz::ParseRuntimeLogFilter("task_id=abc").Ok(),
          "numeric fields should reject text values");
    Check(!cyxwiz::ParseRuntimeLogFilter("(category=training").Ok(),
          "missing parentheses should be reported");
    Check(!cyxwiz::ParseRuntimeLogFilter(
               "message contains \"unterminated")
               .Ok(),
          "unterminated strings should be reported");
    Check(!cyxwiz::ParseRuntimeLogFilter("").Ok(),
          "empty input should not silently become match-all");
}

void TestBoundedQueryService() {
    cyxwiz::RuntimeLogStore store(5);
    Check(store.Append(MakeEvent("system", cyxwiz::RuntimeLogLevel::Info,
                                 "startup")),
          "query fixture append should succeed");
    Check(store.Append(MakeEvent("training", cyxwiz::RuntimeLogLevel::Info,
                                 "batch one")),
          "query fixture append should succeed");
    Check(store.Append(MakeEvent("device", cyxwiz::RuntimeLogLevel::Warning,
                                 "fallback")),
          "query fixture append should succeed");
    Check(store.Append(MakeEvent("training", cyxwiz::RuntimeLogLevel::Error,
                                 "batch two")),
          "query fixture append should succeed");
    Check(store.Append(MakeEvent("training", cyxwiz::RuntimeLogLevel::Info,
                                 "complete")),
          "query fixture append should succeed");

    const auto filter = Parse("category=training");
    cyxwiz::RuntimeLogQueryRequest request;
    request.filter = &filter;
    request.limit = 1;
    const cyxwiz::RuntimeLogQueryService service(store);
    const auto result = service.Query(request);
    Check(result.scanned_count == 5 && result.matched_count == 3,
          "query metadata should distinguish scanned and matched events");
    Check(result.events.size() == 1 && result.events[0].sequence == 2,
          "result limits should preserve earliest retained match order");
    Check(result.truncated && result.high_water_sequence == 5,
          "limited queries should expose truncation and high-water sequence");

    request.after_sequence = 2;
    request.through_sequence = 4;
    request.limit = 10;
    const auto windowed = service.Query(request);
    Check(windowed.scanned_count == 2 && windowed.matched_count == 1 &&
              windowed.events[0].sequence == 4,
          "sequence windows should be applied before filter evaluation");
    Check(windowed.high_water_sequence == 4 && !windowed.truncated,
          "explicit sequence windows should report their effective high water");

    Check(store.Append(MakeEvent("system", cyxwiz::RuntimeLogLevel::Info,
                                 "evict oldest")),
          "query fixture append should permit bounded-store eviction");
    const auto after_eviction = service.Query({});
    Check(after_eviction.store_stats.evicted_count == 1 &&
              after_eviction.store_stats.oldest_sequence == 2,
          "query metadata should expose store eviction and retained range");
}

} // namespace

int main() {
    TestBooleanPrecedenceAndParentheses();
    TestStringsAliasesAndEscapes();
    TestTypedComparisons();
    TestDiagnosticCodeMatching();
    TestParseErrors();
    TestBoundedQueryService();
    std::cout << "Runtime log filter contracts passed\n";
    return 0;
}
