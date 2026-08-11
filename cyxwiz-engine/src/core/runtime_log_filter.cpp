#include "runtime_log_filter.h"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <set>
#include <string_view>
#include <utility>

namespace cyxwiz {
namespace {

std::string LowerAscii(std::string_view value) {
    std::string lowered(value);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lowered;
}

enum class TokenKind : uint8_t {
    Word,
    String,
    Equal,
    NotEqual,
    Greater,
    GreaterOrEqual,
    Less,
    LessOrEqual,
    LeftParen,
    RightParen,
    End,
    Invalid
};

struct Token {
    TokenKind kind = TokenKind::End;
    std::string text;
    size_t position = 0;
    std::string error;
};

class Lexer {
public:
    explicit Lexer(std::string_view input) : input_(input) {}

    Token Next() {
        while (position_ < input_.size() &&
               std::isspace(static_cast<unsigned char>(input_[position_]))) {
            ++position_;
        }
        if (position_ == input_.size()) {
            return {TokenKind::End, {}, position_, {}};
        }

        const size_t start = position_;
        const char current = input_[position_++];
        switch (current) {
            case '(':
                return {TokenKind::LeftParen, "(", start, {}};
            case ')':
                return {TokenKind::RightParen, ")", start, {}};
            case '=':
                if (Consume('=')) return {TokenKind::Equal, "==", start, {}};
                return {TokenKind::Equal, "=", start, {}};
            case '!':
                if (Consume('=')) return {TokenKind::NotEqual, "!=", start, {}};
                return Invalid(start, "expected '=' after '!'");
            case '>':
                if (Consume('=')) {
                    return {TokenKind::GreaterOrEqual, ">=", start, {}};
                }
                return {TokenKind::Greater, ">", start, {}};
            case '<':
                if (Consume('=')) {
                    return {TokenKind::LessOrEqual, "<=", start, {}};
                }
                return {TokenKind::Less, "<", start, {}};
            case '"':
                return ReadString(start);
            default:
                --position_;
                return ReadWord(start);
        }
    }

private:
    bool Consume(char expected) {
        if (position_ < input_.size() && input_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    Token ReadString(size_t start) {
        std::string value;
        while (position_ < input_.size()) {
            const char current = input_[position_++];
            if (current == '"') {
                return {TokenKind::String, std::move(value), start, {}};
            }
            if (current == '\\') {
                if (position_ == input_.size()) {
                    return Invalid(start, "unterminated escape sequence");
                }
                const char escaped = input_[position_++];
                switch (escaped) {
                    case '"': value.push_back('"'); break;
                    case '\\': value.push_back('\\'); break;
                    case 'n': value.push_back('\n'); break;
                    case 't': value.push_back('\t'); break;
                    default:
                        return Invalid(position_ - 1, "unsupported escape sequence");
                }
            } else {
                value.push_back(current);
            }
        }
        return Invalid(start, "unterminated quoted string");
    }

    Token ReadWord(size_t start) {
        while (position_ < input_.size()) {
            const char current = input_[position_];
            if (std::isspace(static_cast<unsigned char>(current)) ||
                current == '(' || current == ')' || current == '=' ||
                current == '!' || current == '>' || current == '<' ||
                current == '"') {
                break;
            }
            ++position_;
        }
        if (position_ == start) {
            ++position_;
            return Invalid(start, "unexpected character");
        }
        return {TokenKind::Word,
                std::string(input_.substr(start, position_ - start)), start, {}};
    }

    static Token Invalid(size_t position, std::string message) {
        return {TokenKind::Invalid, {}, position, std::move(message)};
    }

    std::string_view input_;
    size_t position_ = 0;
};

std::optional<RuntimeLogFilterField> ParseField(std::string_view name) {
    const auto field = LowerAscii(name);
    if (field == "level") return RuntimeLogFilterField::Level;
    if (field == "category") return RuntimeLogFilterField::Category;
    if (field == "source") return RuntimeLogFilterField::Source;
    if (field == "event" || field == "event_name") {
        return RuntimeLogFilterField::EventName;
    }
    if (field == "run" || field == "run_id") return RuntimeLogFilterField::RunId;
    if (field == "task" || field == "task_id") return RuntimeLogFilterField::TaskId;
    if (field == "thread" || field == "thread_id") {
        return RuntimeLogFilterField::ThreadId;
    }
    if (field == "backend") return RuntimeLogFilterField::Backend;
    if (field == "device_id") return RuntimeLogFilterField::DeviceId;
    if (field == "device" || field == "device_name") {
        return RuntimeLogFilterField::DeviceName;
    }
    if (field == "node" || field == "node_id") return RuntimeLogFilterField::NodeId;
    if (field == "dataset" || field == "dataset_name") {
        return RuntimeLogFilterField::DatasetName;
    }
    if (field == "error_code" || field == "code") {
        return RuntimeLogFilterField::ErrorCode;
    }
    if (field == "diagnostic_phase") {
        return RuntimeLogFilterField::DiagnosticPhase;
    }
    if (field == "component") return RuntimeLogFilterField::Component;
    if (field == "message") return RuntimeLogFilterField::Message;
    return std::nullopt;
}

std::optional<RuntimeLogLevel> ParseLevel(std::string_view value) {
    const auto level = LowerAscii(value);
    if (level == "trace") return RuntimeLogLevel::Trace;
    if (level == "debug") return RuntimeLogLevel::Debug;
    if (level == "info") return RuntimeLogLevel::Info;
    if (level == "warn" || level == "warning") return RuntimeLogLevel::Warning;
    if (level == "error" || level == "err") return RuntimeLogLevel::Error;
    if (level == "critical") return RuntimeLogLevel::Critical;
    return std::nullopt;
}

bool IsSignedField(RuntimeLogFilterField field) {
    return field == RuntimeLogFilterField::DeviceId ||
           field == RuntimeLogFilterField::NodeId;
}

bool IsUnsignedField(RuntimeLogFilterField field) {
    return field == RuntimeLogFilterField::TaskId;
}

bool IsRelational(RuntimeLogFilterOperator operation) {
    return operation == RuntimeLogFilterOperator::Greater ||
           operation == RuntimeLogFilterOperator::GreaterOrEqual ||
           operation == RuntimeLogFilterOperator::Less ||
           operation == RuntimeLogFilterOperator::LessOrEqual;
}

template <typename Integer>
bool ParseInteger(std::string_view text, Integer& value) {
    const auto result = std::from_chars(
        text.data(), text.data() + text.size(), value, 10);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

class Parser {
public:
    explicit Parser(std::string_view input) : lexer_(input) { Advance(); }

    RuntimeLogFilterParseResult Parse() {
        auto root = ParseOr();
        if (!root) return Failure();
        if (current_.kind != TokenKind::End) {
            SetError(current_.position, "unexpected token '" + current_.text + "'");
            return Failure();
        }

        RuntimeLogFilterParseResult result;
        result.filter.emplace(std::move(root));
        return result;
    }

private:
    std::unique_ptr<RuntimeLogFilterExpression> ParseOr() {
        auto left = ParseAnd();
        while (left && IsKeyword("or")) {
            Advance();
            auto right = ParseAnd();
            if (!right) return nullptr;
            left = Binary(RuntimeLogFilterExpression::Kind::Or,
                          std::move(left), std::move(right));
        }
        return left;
    }

    std::unique_ptr<RuntimeLogFilterExpression> ParseAnd() {
        auto left = ParseUnary();
        while (left && IsKeyword("and")) {
            Advance();
            auto right = ParseUnary();
            if (!right) return nullptr;
            left = Binary(RuntimeLogFilterExpression::Kind::And,
                          std::move(left), std::move(right));
        }
        return left;
    }

    std::unique_ptr<RuntimeLogFilterExpression> ParseUnary() {
        if (IsKeyword("not")) {
            Advance();
            auto operand = ParseUnary();
            if (!operand) return nullptr;
            auto expression = std::make_unique<RuntimeLogFilterExpression>();
            expression->kind = RuntimeLogFilterExpression::Kind::Not;
            expression->left = std::move(operand);
            return expression;
        }
        return ParsePrimary();
    }

    std::unique_ptr<RuntimeLogFilterExpression> ParsePrimary() {
        if (current_.kind == TokenKind::LeftParen) {
            const size_t open_position = current_.position;
            Advance();
            auto expression = ParseOr();
            if (!expression) return nullptr;
            if (current_.kind != TokenKind::RightParen) {
                SetError(open_position, "missing closing ')'");
                return nullptr;
            }
            Advance();
            return expression;
        }
        return ParsePredicate();
    }

    std::unique_ptr<RuntimeLogFilterExpression> ParsePredicate() {
        if (current_.kind == TokenKind::Invalid) {
            SetError(current_.position, current_.error);
            return nullptr;
        }
        if (current_.kind != TokenKind::Word) {
            SetError(current_.position, "expected a filter field");
            return nullptr;
        }

        const Token field_token = current_;
        const auto field = ParseField(field_token.text);
        if (!field) {
            SetError(field_token.position,
                     "unknown filter field '" + field_token.text + "'");
            return nullptr;
        }
        Advance();

        const auto operation = ParseOperation();
        if (!operation) return nullptr;
        if (current_.kind != TokenKind::Word &&
            current_.kind != TokenKind::String) {
            if (current_.kind == TokenKind::Invalid) {
                SetError(current_.position, current_.error);
            } else {
                SetError(current_.position, "expected a filter value");
            }
            return nullptr;
        }

        const Token value_token = current_;
        Advance();
        RuntimeLogFilterPredicate predicate;
        predicate.field = *field;
        predicate.operation = *operation;
        if (!BuildValue(predicate, value_token)) return nullptr;

        auto expression = std::make_unique<RuntimeLogFilterExpression>();
        expression->kind = RuntimeLogFilterExpression::Kind::Predicate;
        expression->predicate = std::move(predicate);
        return expression;
    }

    std::optional<RuntimeLogFilterOperator> ParseOperation() {
        RuntimeLogFilterOperator operation;
        switch (current_.kind) {
            case TokenKind::Equal: operation = RuntimeLogFilterOperator::Equal; break;
            case TokenKind::NotEqual:
                operation = RuntimeLogFilterOperator::NotEqual;
                break;
            case TokenKind::Greater: operation = RuntimeLogFilterOperator::Greater; break;
            case TokenKind::GreaterOrEqual:
                operation = RuntimeLogFilterOperator::GreaterOrEqual;
                break;
            case TokenKind::Less: operation = RuntimeLogFilterOperator::Less; break;
            case TokenKind::LessOrEqual:
                operation = RuntimeLogFilterOperator::LessOrEqual;
                break;
            case TokenKind::Word: {
                const auto keyword = LowerAscii(current_.text);
                if (keyword == "contains") {
                    operation = RuntimeLogFilterOperator::Contains;
                } else if (keyword == "matches") {
                    operation = RuntimeLogFilterOperator::Matches;
                } else {
                    SetError(current_.position, "expected a filter operator");
                    return std::nullopt;
                }
                break;
            }
            case TokenKind::Invalid:
                SetError(current_.position, current_.error);
                return std::nullopt;
            default:
                SetError(current_.position, "expected a filter operator");
                return std::nullopt;
        }
        Advance();
        return operation;
    }

    bool BuildValue(RuntimeLogFilterPredicate& predicate, const Token& token) {
        auto& value = predicate.value;
        if (predicate.field == RuntimeLogFilterField::Level) {
            if (predicate.operation == RuntimeLogFilterOperator::Contains ||
                predicate.operation == RuntimeLogFilterOperator::Matches) {
                return InvalidOperation(token.position, "level");
            }
            const auto level = ParseLevel(token.text);
            if (!level) {
                SetError(token.position,
                         "invalid level; expected trace, debug, info, warn, "
                         "error, or critical");
                return false;
            }
            value.kind = RuntimeLogFilterValueKind::Level;
            value.level = *level;
            return true;
        }

        if (IsSignedField(predicate.field)) {
            if (predicate.operation == RuntimeLogFilterOperator::Contains ||
                predicate.operation == RuntimeLogFilterOperator::Matches) {
                return InvalidOperation(token.position, "numeric field");
            }
            if (!ParseInteger(token.text, value.signed_integer)) {
                SetError(token.position, "expected a signed integer value");
                return false;
            }
            value.kind = RuntimeLogFilterValueKind::SignedInteger;
            return true;
        }

        if (IsUnsignedField(predicate.field)) {
            if (predicate.operation == RuntimeLogFilterOperator::Contains ||
                predicate.operation == RuntimeLogFilterOperator::Matches) {
                return InvalidOperation(token.position, "numeric field");
            }
            if (!ParseInteger(token.text, value.unsigned_integer)) {
                SetError(token.position, "expected a non-negative integer value");
                return false;
            }
            value.kind = RuntimeLogFilterValueKind::UnsignedInteger;
            return true;
        }

        if (predicate.field == RuntimeLogFilterField::ErrorCode) {
            if (IsRelational(predicate.operation) ||
                predicate.operation == RuntimeLogFilterOperator::Contains) {
                return InvalidOperation(token.position, "error_code");
            }
            if (predicate.operation == RuntimeLogFilterOperator::Matches) {
                const auto family = NormalizeDiagnosticFamily(token.text);
                if (!family) {
                    SetError(token.position,
                             "invalid diagnostic family; expected CW-C-* or "
                             "another CW-X-* family");
                    return false;
                }
                value.kind = RuntimeLogFilterValueKind::DiagnosticFamily;
                value.text = *family;
                return true;
            }
            const auto code = NormalizeDiagnosticCode(token.text);
            if (!code) {
                SetError(token.position,
                         "invalid diagnostic code; expected CW-C-0101 format");
                return false;
            }
            value.kind = RuntimeLogFilterValueKind::DiagnosticCode;
            value.text = *code;
            return true;
        }

        if (IsRelational(predicate.operation) ||
            predicate.operation == RuntimeLogFilterOperator::Matches) {
            return InvalidOperation(token.position, "string field");
        }
        value.kind = RuntimeLogFilterValueKind::String;
        value.text = predicate.field == RuntimeLogFilterField::Category
                         ? LowerAscii(token.text)
                         : token.text;
        if (predicate.field == RuntimeLogFilterField::Category &&
            !IsCanonicalRuntimeLogCategory(value.text)) {
            SetError(token.position,
                     "invalid category; use lower-case letters, digits, or '_'");
            return false;
        }
        return true;
    }

    bool InvalidOperation(size_t position, std::string_view field_type) {
        SetError(position,
                 "operator is not valid for " + std::string(field_type));
        return false;
    }

    bool IsKeyword(std::string_view keyword) const {
        return current_.kind == TokenKind::Word &&
               LowerAscii(current_.text) == keyword;
    }

    void Advance() { current_ = lexer_.Next(); }

    void SetError(size_t position, std::string message) {
        if (!error_) {
            error_ = RuntimeLogFilterParseError{position, std::move(message)};
        }
    }

    RuntimeLogFilterParseResult Failure() {
        RuntimeLogFilterParseResult result;
        result.error = error_.value_or(RuntimeLogFilterParseError{
            current_.position, "invalid filter expression"});
        return result;
    }

    static std::unique_ptr<RuntimeLogFilterExpression> Binary(
        RuntimeLogFilterExpression::Kind kind,
        std::unique_ptr<RuntimeLogFilterExpression> left,
        std::unique_ptr<RuntimeLogFilterExpression> right) {
        auto expression = std::make_unique<RuntimeLogFilterExpression>();
        expression->kind = kind;
        expression->left = std::move(left);
        expression->right = std::move(right);
        return expression;
    }

    Lexer lexer_;
    Token current_;
    std::optional<RuntimeLogFilterParseError> error_;
};

template <typename Value>
bool CompareOrdered(Value actual, RuntimeLogFilterOperator operation,
                    Value expected) {
    switch (operation) {
        case RuntimeLogFilterOperator::Equal: return actual == expected;
        case RuntimeLogFilterOperator::NotEqual: return actual != expected;
        case RuntimeLogFilterOperator::Greater: return actual > expected;
        case RuntimeLogFilterOperator::GreaterOrEqual: return actual >= expected;
        case RuntimeLogFilterOperator::Less: return actual < expected;
        case RuntimeLogFilterOperator::LessOrEqual: return actual <= expected;
        case RuntimeLogFilterOperator::Contains:
        case RuntimeLogFilterOperator::Matches:
            return false;
    }
    return false;
}

std::string_view StringField(const RuntimeLogEvent& event,
                             RuntimeLogFilterField field) {
    switch (field) {
        case RuntimeLogFilterField::Category: return event.category;
        case RuntimeLogFilterField::Source: return event.source;
        case RuntimeLogFilterField::EventName: return event.event_name;
        case RuntimeLogFilterField::RunId: return event.run_id;
        case RuntimeLogFilterField::ThreadId: return event.thread_id;
        case RuntimeLogFilterField::Backend: return event.backend;
        case RuntimeLogFilterField::DeviceName: return event.device_name;
        case RuntimeLogFilterField::DatasetName: return event.dataset_name;
        case RuntimeLogFilterField::DiagnosticPhase:
            return event.diagnostic_phase;
        case RuntimeLogFilterField::Component: return event.component;
        case RuntimeLogFilterField::Message: return event.message;
        case RuntimeLogFilterField::Level:
        case RuntimeLogFilterField::TaskId:
        case RuntimeLogFilterField::DeviceId:
        case RuntimeLogFilterField::NodeId:
        case RuntimeLogFilterField::ErrorCode:
            return {};
    }
    return {};
}

bool ContainsAsciiInsensitive(std::string_view actual,
                              std::string_view expected) {
    return std::search(
               actual.begin(), actual.end(), expected.begin(), expected.end(),
               [](unsigned char left, unsigned char right) {
                   return std::tolower(left) == std::tolower(right);
               }) != actual.end();
}

bool MatchString(std::string_view actual, RuntimeLogFilterOperator operation,
                 std::string_view expected, bool insensitive_contains) {
    switch (operation) {
        case RuntimeLogFilterOperator::Equal: return actual == expected;
        case RuntimeLogFilterOperator::NotEqual: return actual != expected;
        case RuntimeLogFilterOperator::Contains:
            return insensitive_contains
                ? ContainsAsciiInsensitive(actual, expected)
                : actual.find(expected) != std::string_view::npos;
        case RuntimeLogFilterOperator::Greater:
        case RuntimeLogFilterOperator::GreaterOrEqual:
        case RuntimeLogFilterOperator::Less:
        case RuntimeLogFilterOperator::LessOrEqual:
        case RuntimeLogFilterOperator::Matches:
            return false;
    }
    return false;
}

bool MatchCode(const RuntimeLogEvent& event,
               const RuntimeLogFilterPredicate& predicate) {
    const auto matches_one = [&](std::string_view code) {
        if (predicate.operation == RuntimeLogFilterOperator::Matches) {
            return code.size() >= 5 && predicate.value.text.size() == 6 &&
                   code.substr(0, 5) ==
                       std::string_view(predicate.value.text).substr(0, 5);
        }
        return code == predicate.value.text;
    };

    bool matched = !event.primary_error_code.empty() &&
                   matches_one(event.primary_error_code);
    if (!matched) {
        matched = std::any_of(
            event.issue_codes.begin(), event.issue_codes.end(), matches_one);
    }
    return predicate.operation == RuntimeLogFilterOperator::NotEqual
               ? !matched
               : matched;
}

bool EvaluatePredicate(const RuntimeLogFilterPredicate& predicate,
                       const RuntimeLogEvent& event) {
    switch (predicate.field) {
        case RuntimeLogFilterField::Level:
            return CompareOrdered(event.level, predicate.operation,
                                  predicate.value.level);
        case RuntimeLogFilterField::TaskId:
            return CompareOrdered(event.task_id, predicate.operation,
                                  predicate.value.unsigned_integer);
        case RuntimeLogFilterField::DeviceId:
            return CompareOrdered(static_cast<int64_t>(event.device_id),
                                  predicate.operation,
                                  predicate.value.signed_integer);
        case RuntimeLogFilterField::NodeId:
            return CompareOrdered(static_cast<int64_t>(event.node_id),
                                  predicate.operation,
                                  predicate.value.signed_integer);
        case RuntimeLogFilterField::ErrorCode:
            return MatchCode(event, predicate);
        default:
            return MatchString(StringField(event, predicate.field),
                               predicate.operation, predicate.value.text,
                               predicate.field == RuntimeLogFilterField::Message);
    }
}

bool EvaluateExpression(const RuntimeLogFilterExpression& expression,
                        const RuntimeLogEvent& event) {
    switch (expression.kind) {
        case RuntimeLogFilterExpression::Kind::Predicate:
            return EvaluatePredicate(expression.predicate, event);
        case RuntimeLogFilterExpression::Kind::And:
            return EvaluateExpression(*expression.left, event) &&
                   EvaluateExpression(*expression.right, event);
        case RuntimeLogFilterExpression::Kind::Or:
            return EvaluateExpression(*expression.left, event) ||
                   EvaluateExpression(*expression.right, event);
        case RuntimeLogFilterExpression::Kind::Not:
            return !EvaluateExpression(*expression.left, event);
    }
    return false;
}

} // namespace

std::string_view RuntimeLogFilterHelpText() {
    return
        "Expressions select runtime events using field conditions.\n"
        "Format: <field> <operator> <value>\n\n"
        "Boolean logic\n"
        "  not is evaluated first, then and, then or.\n"
        "  Use parentheses when combining alternatives.\n"
        "  Example: (category=training or category=device) and level>=warn\n\n"
        "Operators by field type\n"
        "  Strings: =, ==, !=, contains\n"
        "  Level/numbers: =, ==, !=, >, >=, <, <=\n"
        "  Error codes: = or != for one code; matches for a CW-X-* family\n"
        "  'message contains' ignores ASCII letter case. Other string\n"
        "  comparisons are case-sensitive. Category, level, and CW code\n"
        "  input is canonicalized.\n\n"
        "Values\n"
        "  Quote values containing spaces: message contains \"host sync\"\n"
        "  Quoted values support escaped quote (\\\"), backslash (\\\\),\n"
        "  newline (\\n), and tab (\\t).\n"
        "  Levels: trace, debug, info, warn, error, critical\n\n"
        "Fields and aliases\n"
        "  level, category, source, message, backend, component\n"
        "  event or event_name; run or run_id; task or task_id\n"
        "  thread or thread_id; device or device_name; device_id\n"
        "  node or node_id; dataset or dataset_name; diagnostic_phase\n"
        "  code or error_code\n\n"
        "Examples\n"
        "  message contains \"native CPU fallback\"\n"
        "  category=training and level>=warn\n"
        "  (category=training or category=device) and level>=warn\n"
        "  not (category=system or level<info)\n"
        "  backend=arrayfire_cuda and device_id=0\n"
        "  run_id=train-42 and task_id=17\n"
        "  source=GraphCompiler and event=compile.validation\n"
        "  dataset=training.parquet and component!=DataLoader\n"
        "  error_code=CW-T-0501\n"
        "  error_code matches CW-G-*";
}

RuntimeLogFilter::RuntimeLogFilter(
    std::unique_ptr<RuntimeLogFilterExpression> root)
    : root_(std::move(root)) {}

bool RuntimeLogFilter::Matches(const RuntimeLogEvent& event) const {
    return root_ && EvaluateExpression(*root_, event);
}

RuntimeLogFilterParseResult ParseRuntimeLogFilter(const std::string& input) {
    return Parser(input).Parse();
}

RuntimeLogQueryResult RuntimeLogQueryService::Query(
    const RuntimeLogQueryRequest& request) const {
    RuntimeLogSnapshotRequest snapshot_request;
    snapshot_request.after_sequence = request.after_sequence;
    snapshot_request.through_sequence = request.through_sequence;
    snapshot_request.limit = store_.GetStats().capacity;
    const auto snapshot = store_.Snapshot(snapshot_request);

    RuntimeLogQueryResult result;
    result.store_stats = snapshot.stats;
    result.scanned_count = snapshot.events.size();
    result.high_water_sequence =
        request.through_sequence == std::numeric_limits<uint64_t>::max()
            ? snapshot.stats.newest_sequence
            : std::min(request.through_sequence,
                       snapshot.stats.newest_sequence);
    result.events.reserve(std::min(request.limit, snapshot.events.size()));

    std::set<std::string> categories;
    std::set<std::string> sources;
    std::set<std::string> codes;
    std::set<std::string> run_ids;
    std::set<uint64_t> task_ids;
    std::set<int> device_ids;
    std::set<std::string> backends;

    for (const auto& event : snapshot.events) {
        if (request.collect_facets) {
            if (!event.category.empty()) categories.insert(event.category);
            if (!event.source.empty()) sources.insert(event.source);
            if (!event.primary_error_code.empty()) {
                codes.insert(event.primary_error_code);
            }
            codes.insert(event.issue_codes.begin(), event.issue_codes.end());
            if (!event.run_id.empty()) run_ids.insert(event.run_id);
            if (event.task_id != 0) task_ids.insert(event.task_id);
            if (event.device_id >= 0) device_ids.insert(event.device_id);
            if (!event.backend.empty()) backends.insert(event.backend);
        }
        if (request.filter && !request.filter->Matches(event)) {
            continue;
        }
        ++result.matched_count;
        if (result.events.size() < request.limit) {
            result.events.push_back(event);
        }
    }
    result.truncated = snapshot.truncated ||
                       result.matched_count > result.events.size();
    if (request.collect_facets) {
        result.facets.categories.assign(categories.begin(), categories.end());
        result.facets.sources.assign(sources.begin(), sources.end());
        result.facets.codes.assign(codes.begin(), codes.end());
        result.facets.run_ids.assign(run_ids.begin(), run_ids.end());
        result.facets.task_ids.assign(task_ids.begin(), task_ids.end());
        result.facets.device_ids.assign(device_ids.begin(), device_ids.end());
        result.facets.backends.assign(backends.begin(), backends.end());
    }
    return result;
}

} // namespace cyxwiz
