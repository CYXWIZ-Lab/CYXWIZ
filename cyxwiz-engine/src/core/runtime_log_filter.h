#pragma once

#include "runtime_log_store.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace cyxwiz {

enum class RuntimeLogFilterField : uint8_t {
    Level,
    Category,
    Source,
    EventName,
    RunId,
    TaskId,
    ThreadId,
    Backend,
    DeviceId,
    DeviceName,
    NodeId,
    DatasetName,
    ErrorCode,
    DiagnosticPhase,
    Component,
    Message
};

enum class RuntimeLogFilterOperator : uint8_t {
    Equal,
    NotEqual,
    Greater,
    GreaterOrEqual,
    Less,
    LessOrEqual,
    Contains,
    Matches
};

enum class RuntimeLogFilterValueKind : uint8_t {
    String,
    SignedInteger,
    UnsignedInteger,
    Level,
    DiagnosticCode,
    DiagnosticFamily
};

struct RuntimeLogFilterValue {
    RuntimeLogFilterValueKind kind = RuntimeLogFilterValueKind::String;
    std::string text;
    int64_t signed_integer = 0;
    uint64_t unsigned_integer = 0;
    RuntimeLogLevel level = RuntimeLogLevel::Info;
};

struct RuntimeLogFilterPredicate {
    RuntimeLogFilterField field = RuntimeLogFilterField::Message;
    RuntimeLogFilterOperator operation = RuntimeLogFilterOperator::Equal;
    RuntimeLogFilterValue value;
};

struct RuntimeLogFilterExpression {
    enum class Kind : uint8_t { Predicate, And, Or, Not };

    Kind kind = Kind::Predicate;
    RuntimeLogFilterPredicate predicate;
    std::unique_ptr<RuntimeLogFilterExpression> left;
    std::unique_ptr<RuntimeLogFilterExpression> right;
};

class RuntimeLogFilter {
public:
    explicit RuntimeLogFilter(
        std::unique_ptr<RuntimeLogFilterExpression> root);
    RuntimeLogFilter(RuntimeLogFilter&&) noexcept = default;
    RuntimeLogFilter& operator=(RuntimeLogFilter&&) noexcept = default;
    RuntimeLogFilter(const RuntimeLogFilter&) = delete;
    RuntimeLogFilter& operator=(const RuntimeLogFilter&) = delete;

    bool Matches(const RuntimeLogEvent& event) const;

private:
    std::unique_ptr<RuntimeLogFilterExpression> root_;
};

struct RuntimeLogFilterParseError {
    size_t position = 0;
    std::string message;
};

struct RuntimeLogFilterParseResult {
    std::optional<RuntimeLogFilter> filter;
    std::optional<RuntimeLogFilterParseError> error;

    bool Ok() const { return filter.has_value(); }
};

RuntimeLogFilterParseResult ParseRuntimeLogFilter(const std::string& input);

std::string_view RuntimeLogFilterHelpText();

struct RuntimeLogQueryRequest {
    const RuntimeLogFilter* filter = nullptr;
    uint64_t after_sequence = 0;
    uint64_t through_sequence = std::numeric_limits<uint64_t>::max();
    size_t limit = 1000;
    bool collect_facets = false;
};

struct RuntimeLogQueryFacets {
    std::vector<std::string> categories;
    std::vector<std::string> sources;
    std::vector<std::string> codes;
    std::vector<std::string> run_ids;
    std::vector<uint64_t> task_ids;
    std::vector<int> device_ids;
    std::vector<std::string> backends;
};

struct RuntimeLogQueryResult {
    std::vector<RuntimeLogEvent> events;
    RuntimeLogStoreStats store_stats;
    size_t scanned_count = 0;
    size_t matched_count = 0;
    uint64_t high_water_sequence = 0;
    bool truncated = false;
    RuntimeLogQueryFacets facets;
};

class RuntimeLogQueryService {
public:
    explicit RuntimeLogQueryService(const RuntimeLogStore& store)
        : store_(store) {}

    RuntimeLogQueryResult Query(const RuntimeLogQueryRequest& request) const;

private:
    const RuntimeLogStore& store_;
};

} // namespace cyxwiz
