#include "runtime_log_inspector.h"

#include <algorithm>
#include <sstream>
#include <vector>

namespace cyxwiz {
namespace {

std::string EscapeFilterValue(std::string_view value) {
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (const char current : value) {
        switch (current) {
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\n': escaped += "\\n"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped.push_back(current); break;
        }
    }
    escaped.push_back('"');
    return escaped;
}

void AddClause(std::vector<std::string>& clauses, std::string clause) {
    if (!clause.empty()) clauses.push_back(std::move(clause));
}

} // namespace

std::string BuildRuntimeLogInspectorFilter(
    const RuntimeLogInspectorCriteria& criteria) {
    static constexpr std::array<const char*, 6> level_names = {
        "trace", "debug", "info", "warn", "error", "critical"};
    std::vector<std::string> clauses;

    const size_t enabled_levels = static_cast<size_t>(std::count(
        criteria.levels.begin(), criteria.levels.end(), true));
    if (enabled_levels == 0) {
        AddClause(clauses, "level=trace and level=debug");
    } else if (enabled_levels != criteria.levels.size()) {
        std::ostringstream levels;
        levels << '(';
        bool first = true;
        for (size_t index = 0; index < criteria.levels.size(); ++index) {
            if (!criteria.levels[index]) continue;
            if (!first) levels << " or ";
            levels << "level=" << level_names[index];
            first = false;
        }
        levels << ')';
        AddClause(clauses, levels.str());
    }

    const auto add_string = [&clauses](std::string_view field,
                                       const std::string& value) {
        if (!value.empty()) {
            AddClause(clauses, std::string(field) + '=' +
                                   EscapeFilterValue(value));
        }
    };
    add_string("category", criteria.category);
    add_string("source", criteria.source);
    add_string("error_code", criteria.code);
    add_string("run_id", criteria.run_id);
    add_string("backend", criteria.backend);
    if (criteria.task_id) {
        AddClause(clauses, "task_id=" + std::to_string(*criteria.task_id));
    }
    if (criteria.device_id) {
        AddClause(clauses,
                  "device_id=" + std::to_string(*criteria.device_id));
    }
    if (!criteria.text.empty()) {
        AddClause(clauses, "message contains " +
                               EscapeFilterValue(criteria.text));
    }
    if (!criteria.structured_filter.empty()) {
        AddClause(clauses, '(' + criteria.structured_filter + ')');
    }

    std::ostringstream expression;
    for (size_t index = 0; index < clauses.size(); ++index) {
        if (index != 0) expression << " and ";
        expression << clauses[index];
    }
    return expression.str();
}

RuntimeLogInspectorResult QueryRuntimeLogInspector(
    const RuntimeLogStore& store,
    const RuntimeLogInspectorRequest& request) {
    RuntimeLogInspectorResult result;
    result.effective_filter =
        BuildRuntimeLogInspectorFilter(request.criteria);

    std::optional<RuntimeLogFilter> filter;
    if (!result.effective_filter.empty()) {
        auto parsed = ParseRuntimeLogFilter(result.effective_filter);
        if (!parsed.Ok()) {
            result.filter_error = parsed.error;
            result.query.store_stats = store.GetStats();
            return result;
        }
        filter.emplace(std::move(*parsed.filter));
    }

    RuntimeLogQueryRequest query_request;
    query_request.filter = filter ? &*filter : nullptr;
    query_request.after_sequence = request.after_sequence;
    query_request.through_sequence = request.through_sequence;
    query_request.limit = store.GetStats().capacity;
    query_request.collect_facets = true;
    result.query = RuntimeLogQueryService(store).Query(query_request);

    const size_t display_limit = std::max<size_t>(1, request.display_limit);
    if (result.query.events.size() > display_limit) {
        const size_t omitted = result.query.events.size() - display_limit;
        result.query.events.erase(result.query.events.begin(),
                                  result.query.events.begin() + omitted);
        result.query.truncated = true;
    }
    return result;
}

} // namespace cyxwiz
