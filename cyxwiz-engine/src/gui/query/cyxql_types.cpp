#include "cyxql_types.h"
#include <sstream>
#include <iomanip>

namespace cyxwiz::query {

// ============================================================================
// Token Implementation
// ============================================================================

bool Token::isKeyword() const {
    return token_type >= TokenType::MATCH && token_type <= TokenType::FALSE_LITERAL;
}

bool Token::isOperator() const {
    return (token_type >= TokenType::EQUALS && token_type <= TokenType::IN) ||
           (token_type >= TokenType::PLUS && token_type <= TokenType::POWER);
}

bool Token::isLiteral() const {
    return token_type == TokenType::STRING ||
           token_type == TokenType::INTEGER ||
           token_type == TokenType::FLOAT ||
           token_type == TokenType::TRUE_LITERAL ||
           token_type == TokenType::FALSE_LITERAL ||
           token_type == TokenType::NULL_LITERAL;
}

std::string Token::toString() const {
    return typeToString(token_type) + "(\"" + value + "\") at " +
           std::to_string(line) + ":" + std::to_string(column);
}

std::string Token::typeToString(TokenType tok_type) {
    switch (tok_type) {
        case TokenType::MATCH: return "MATCH";
        case TokenType::WHERE: return "WHERE";
        case TokenType::RETURN: return "RETURN";
        case TokenType::CREATE: return "CREATE";
        case TokenType::DELETE: return "DELETE";
        case TokenType::SET: return "SET";
        case TokenType::WITH: return "WITH";
        case TokenType::ORDER: return "ORDER";
        case TokenType::BY: return "BY";
        case TokenType::LIMIT: return "LIMIT";
        case TokenType::SKIP: return "SKIP";
        case TokenType::AND: return "AND";
        case TokenType::OR: return "OR";
        case TokenType::NOT: return "NOT";
        case TokenType::XOR: return "XOR";
        case TokenType::COUNT: return "COUNT";
        case TokenType::SUM: return "SUM";
        case TokenType::AVG: return "AVG";
        case TokenType::MIN: return "MIN";
        case TokenType::MAX: return "MAX";
        case TokenType::COLLECT: return "COLLECT";
        case TokenType::TYPE: return "TYPE";
        case TokenType::ID: return "ID";
        case TokenType::PROPERTIES: return "PROPERTIES";
        case TokenType::KEYS: return "KEYS";
        case TokenType::EXISTS: return "EXISTS";
        case TokenType::SIZE: return "SIZE";
        case TokenType::ASC: return "ASC";
        case TokenType::DESC: return "DESC";
        case TokenType::AS: return "AS";
        case TokenType::IS: return "IS";
        case TokenType::NULL_LITERAL: return "NULL";
        case TokenType::TRUE_LITERAL: return "TRUE";
        case TokenType::FALSE_LITERAL: return "FALSE";
        case TokenType::LPAREN: return "LPAREN";
        case TokenType::RPAREN: return "RPAREN";
        case TokenType::LBRACKET: return "LBRACKET";
        case TokenType::RBRACKET: return "RBRACKET";
        case TokenType::LBRACE: return "LBRACE";
        case TokenType::RBRACE: return "RBRACE";
        case TokenType::COLON: return "COLON";
        case TokenType::COMMA: return "COMMA";
        case TokenType::DOT: return "DOT";
        case TokenType::PIPE: return "PIPE";
        case TokenType::ARROW: return "ARROW";
        case TokenType::LEFT_ARROW: return "LEFT_ARROW";
        case TokenType::DASH: return "DASH";
        case TokenType::STAR: return "STAR";
        case TokenType::EQUALS: return "EQUALS";
        case TokenType::NOT_EQUALS: return "NOT_EQUALS";
        case TokenType::LESS: return "LESS";
        case TokenType::GREATER: return "GREATER";
        case TokenType::LESS_EQ: return "LESS_EQ";
        case TokenType::GREATER_EQ: return "GREATER_EQ";
        case TokenType::CONTAINS: return "CONTAINS";
        case TokenType::STARTS_WITH: return "STARTS_WITH";
        case TokenType::ENDS_WITH: return "ENDS_WITH";
        case TokenType::IN: return "IN";
        case TokenType::PLUS: return "PLUS";
        case TokenType::MINUS: return "MINUS";
        case TokenType::MULTIPLY: return "MULTIPLY";
        case TokenType::DIVIDE: return "DIVIDE";
        case TokenType::MODULO: return "MODULO";
        case TokenType::POWER: return "POWER";
        case TokenType::IDENTIFIER: return "IDENTIFIER";
        case TokenType::STRING: return "STRING";
        case TokenType::INTEGER: return "INTEGER";
        case TokenType::FLOAT: return "FLOAT";
        case TokenType::END_OF_INPUT: return "END_OF_INPUT";
        case TokenType::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// ASTNode Implementation
// ============================================================================

ASTNodePtr ASTNode::makeIdentifier(const std::string& name, int line, int col) {
    auto node = std::make_shared<ASTNode>(ASTNodeType::Identifier, name);
    node->line = line;
    node->column = col;
    return node;
}

ASTNodePtr ASTNode::makeLiteral(const LiteralValue& val, int line, int col) {
    auto node = std::make_shared<ASTNode>(ASTNodeType::Literal);
    node->literal = val;
    node->line = line;
    node->column = col;
    return node;
}

ASTNodePtr ASTNode::makeNodePattern(const std::string& var, const std::string& label) {
    auto node = std::make_shared<ASTNode>(ASTNodeType::NodePattern);
    if (!var.empty()) {
        node->addChild(makeIdentifier(var));
    }
    if (!label.empty()) {
        auto labelNode = std::make_shared<ASTNode>(ASTNodeType::Label, label);
        node->addChild(labelNode);
    }
    return node;
}

ASTNodePtr ASTNode::makeRelPattern(const std::string& var, const std::string& type) {
    auto node = std::make_shared<ASTNode>(ASTNodeType::RelationshipPattern);
    if (!var.empty()) {
        node->addChild(makeIdentifier(var));
    }
    if (!type.empty()) {
        auto typeNode = std::make_shared<ASTNode>(ASTNodeType::RelType, type);
        node->addChild(typeNode);
    }
    return node;
}

ASTNodePtr ASTNode::makeBinaryExpr(const std::string& op, ASTNodePtr left, ASTNodePtr right) {
    auto node = std::make_shared<ASTNode>(ASTNodeType::BinaryExpr, op);
    node->addChild(left);
    node->addChild(right);
    return node;
}

ASTNodePtr ASTNode::makeFunctionCall(const std::string& name, std::vector<ASTNodePtr> args) {
    auto node = std::make_shared<ASTNode>(ASTNodeType::FunctionCall, name);
    for (auto& arg : args) {
        node->addChild(arg);
    }
    return node;
}

void ASTNode::addChild(ASTNodePtr child) {
    if (child) {
        children.push_back(child);
    }
}

void ASTNode::setProperty(const std::string& key, const std::string& value) {
    properties[key] = value;
}

std::string ASTNode::getProperty(const std::string& key, const std::string& defaultVal) const {
    auto it = properties.find(key);
    return it != properties.end() ? it->second : defaultVal;
}

bool ASTNode::hasProperty(const std::string& key) const {
    return properties.find(key) != properties.end();
}

std::string ASTNode::toString(int indent) const {
    std::stringstream ss;
    std::string pad(indent * 2, ' ');

    ss << pad << typeToString(type);
    if (!value.empty()) {
        ss << " \"" << value << "\"";
    }

    // Print literal value if present
    if (type == ASTNodeType::Literal) {
        ss << " = ";
        std::visit([&ss](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::nullptr_t>) {
                ss << "null";
            } else if constexpr (std::is_same_v<T, bool>) {
                ss << (arg ? "true" : "false");
            } else if constexpr (std::is_same_v<T, int64_t>) {
                ss << arg;
            } else if constexpr (std::is_same_v<T, double>) {
                ss << arg;
            } else if constexpr (std::is_same_v<T, std::string>) {
                ss << "\"" << arg << "\"";
            }
        }, literal);
    }

    // Print properties
    if (!properties.empty()) {
        ss << " {";
        bool first = true;
        for (const auto& [k, v] : properties) {
            if (!first) ss << ", ";
            ss << k << ": " << v;
            first = false;
        }
        ss << "}";
    }

    ss << "\n";

    // Print children
    for (const auto& child : children) {
        ss << child->toString(indent + 1);
    }

    return ss.str();
}

std::string ASTNode::typeToString(ASTNodeType node_type) {
    switch (node_type) {
        case ASTNodeType::Query: return "Query";
        case ASTNodeType::MatchClause: return "MatchClause";
        case ASTNodeType::WhereClause: return "WhereClause";
        case ASTNodeType::ReturnClause: return "ReturnClause";
        case ASTNodeType::CreateClause: return "CreateClause";
        case ASTNodeType::DeleteClause: return "DeleteClause";
        case ASTNodeType::SetClause: return "SetClause";
        case ASTNodeType::OrderByClause: return "OrderByClause";
        case ASTNodeType::LimitClause: return "LimitClause";
        case ASTNodeType::SkipClause: return "SkipClause";
        case ASTNodeType::WithClause: return "WithClause";
        case ASTNodeType::NodePattern: return "NodePattern";
        case ASTNodeType::RelationshipPattern: return "RelationshipPattern";
        case ASTNodeType::PathPattern: return "PathPattern";
        case ASTNodeType::PropertyAccess: return "PropertyAccess";
        case ASTNodeType::PropertyList: return "PropertyList";
        case ASTNodeType::PropertyAssignment: return "PropertyAssignment";
        case ASTNodeType::BinaryExpr: return "BinaryExpr";
        case ASTNodeType::UnaryExpr: return "UnaryExpr";
        case ASTNodeType::ComparisonExpr: return "ComparisonExpr";
        case ASTNodeType::LogicalExpr: return "LogicalExpr";
        case ASTNodeType::FunctionCall: return "FunctionCall";
        case ASTNodeType::Identifier: return "Identifier";
        case ASTNodeType::Label: return "Label";
        case ASTNodeType::RelType: return "RelType";
        case ASTNodeType::Literal: return "Literal";
        case ASTNodeType::ListExpr: return "ListExpr";
        case ASTNodeType::MapExpr: return "MapExpr";
        case ASTNodeType::Star: return "Star";
        case ASTNodeType::Range: return "Range";
        case ASTNodeType::Alias: return "Alias";
        case ASTNodeType::SortItem: return "SortItem";
        default: return "Unknown";
    }
}

// ============================================================================
// ResultValue Implementation
// ============================================================================

ResultValue ResultValue::makeNull() {
    ResultValue v;
    v.type = Type::Null;
    v.scalar = nullptr;
    return v;
}

ResultValue ResultValue::makeBool(bool val) {
    ResultValue v;
    v.type = Type::Bool;
    v.scalar = val;
    return v;
}

ResultValue ResultValue::makeInt(int64_t val) {
    ResultValue v;
    v.type = Type::Int;
    v.scalar = val;
    return v;
}

ResultValue ResultValue::makeFloat(double val) {
    ResultValue v;
    v.type = Type::Float;
    v.scalar = val;
    return v;
}

ResultValue ResultValue::makeString(const std::string& val) {
    ResultValue v;
    v.type = Type::String;
    v.scalar = val;
    return v;
}

ResultValue ResultValue::makeNode(int id) {
    ResultValue v;
    v.type = Type::Node;
    v.nodeId = id;
    return v;
}

ResultValue ResultValue::makeLink(int id) {
    ResultValue v;
    v.type = Type::Link;
    v.linkId = id;
    return v;
}

ResultValue ResultValue::makePath(std::vector<int> nodes, std::vector<int> links) {
    ResultValue v;
    v.type = Type::Path;
    v.pathNodeIds = std::move(nodes);
    v.pathLinkIds = std::move(links);
    return v;
}

ResultValue ResultValue::makeList(std::vector<ResultValue> items) {
    ResultValue v;
    v.type = Type::List;
    v.list = std::move(items);
    return v;
}

ResultValue ResultValue::makeMap(std::map<std::string, ResultValue> items) {
    ResultValue v;
    v.type = Type::Map;
    v.map = std::move(items);
    return v;
}

std::string ResultValue::toString() const {
    std::stringstream ss;
    switch (type) {
        case Type::Null:
            ss << "null";
            break;
        case Type::Bool:
            ss << (std::get<bool>(scalar) ? "true" : "false");
            break;
        case Type::Int:
            ss << std::get<int64_t>(scalar);
            break;
        case Type::Float:
            ss << std::get<double>(scalar);
            break;
        case Type::String:
            ss << "\"" << std::get<std::string>(scalar) << "\"";
            break;
        case Type::Node:
            ss << "Node(" << nodeId << ")";
            break;
        case Type::Link:
            ss << "Link(" << linkId << ")";
            break;
        case Type::Path:
            ss << "Path[";
            for (size_t i = 0; i < pathNodeIds.size(); i++) {
                if (i > 0) ss << "->";
                ss << "(" << pathNodeIds[i] << ")";
            }
            ss << "]";
            break;
        case Type::List:
            ss << "[";
            for (size_t i = 0; i < list.size(); i++) {
                if (i > 0) ss << ", ";
                ss << list[i].toString();
            }
            ss << "]";
            break;
        case Type::Map:
            ss << "{";
            bool first = true;
            for (const auto& [k, v] : map) {
                if (!first) ss << ", ";
                ss << k << ": " << v.toString();
                first = false;
            }
            ss << "}";
            break;
    }
    return ss.str();
}

bool ResultValue::toBool() const {
    switch (type) {
        case Type::Null: return false;
        case Type::Bool: return std::get<bool>(scalar);
        case Type::Int: return std::get<int64_t>(scalar) != 0;
        case Type::Float: return std::get<double>(scalar) != 0.0;
        case Type::String: return !std::get<std::string>(scalar).empty();
        case Type::Node: return nodeId >= 0;
        case Type::Link: return linkId >= 0;
        case Type::Path: return !pathNodeIds.empty();
        case Type::List: return !list.empty();
        case Type::Map: return !map.empty();
        default: return false;
    }
}

int64_t ResultValue::toInt() const {
    switch (type) {
        case Type::Int: return std::get<int64_t>(scalar);
        case Type::Float: return static_cast<int64_t>(std::get<double>(scalar));
        case Type::Bool: return std::get<bool>(scalar) ? 1 : 0;
        case Type::String: {
            try {
                return std::stoll(std::get<std::string>(scalar));
            } catch (...) {
                return 0;
            }
        }
        default: return 0;
    }
}

double ResultValue::toFloat() const {
    switch (type) {
        case Type::Float: return std::get<double>(scalar);
        case Type::Int: return static_cast<double>(std::get<int64_t>(scalar));
        case Type::Bool: return std::get<bool>(scalar) ? 1.0 : 0.0;
        case Type::String: {
            try {
                return std::stod(std::get<std::string>(scalar));
            } catch (...) {
                return 0.0;
            }
        }
        default: return 0.0;
    }
}

// ============================================================================
// ResultRow Implementation
// ============================================================================

ResultValue ResultRow::get(const std::string& column) const {
    auto it = values.find(column);
    return it != values.end() ? it->second : ResultValue::makeNull();
}

bool ResultRow::has(const std::string& column) const {
    return values.find(column) != values.end();
}

void ResultRow::set(const std::string& column, ResultValue value) {
    values[column] = std::move(value);
}

// ============================================================================
// QueryResult Implementation
// ============================================================================

QueryResult QueryResult::makeError(const std::string& msg, int line, int col) {
    QueryResult r;
    r.success = false;
    r.error = msg;
    r.errorLine = line;
    r.errorColumn = col;
    return r;
}

QueryResult QueryResult::makeSuccess() {
    QueryResult r;
    r.success = true;
    return r;
}

void QueryResult::addColumn(const std::string& name) {
    // Check for duplicates
    for (const auto& col : columns) {
        if (col == name) return;
    }
    columns.push_back(name);
}

void QueryResult::addRow(ResultRow row) {
    rows.push_back(std::move(row));
}

std::string QueryResult::toString() const {
    std::stringstream ss;

    if (!success) {
        ss << "Error: " << error;
        if (errorLine > 0) {
            ss << " at line " << errorLine << ", column " << errorColumn;
        }
        return ss.str();
    }

    ss << "Query executed successfully.\n";
    ss << "Rows: " << rows.size() << ", Columns: " << columns.size() << "\n";

    if (nodesCreated > 0) ss << "Nodes created: " << nodesCreated << "\n";
    if (nodesDeleted > 0) ss << "Nodes deleted: " << nodesDeleted << "\n";
    if (nodesModified > 0) ss << "Nodes modified: " << nodesModified << "\n";
    if (linksCreated > 0) ss << "Links created: " << linksCreated << "\n";
    if (linksDeleted > 0) ss << "Links deleted: " << linksDeleted << "\n";
    if (linksModified > 0) ss << "Links modified: " << linksModified << "\n";
    if (propertiesSet > 0) ss << "Properties set: " << propertiesSet << "\n";

    ss << "Execution time: " << std::fixed << std::setprecision(2)
       << executionTimeMs << " ms\n";

    return ss.str();
}

std::string QueryResult::toTable() const {
    if (!success) {
        return "Error: " + error;
    }

    if (columns.empty() || rows.empty()) {
        return "(empty result)";
    }

    std::stringstream ss;

    // Calculate column widths
    std::vector<size_t> widths(columns.size());
    for (size_t i = 0; i < columns.size(); i++) {
        widths[i] = columns[i].length();
    }
    for (const auto& row : rows) {
        for (size_t i = 0; i < columns.size(); i++) {
            auto val = row.get(columns[i]).toString();
            widths[i] = std::max(widths[i], val.length());
        }
    }

    // Print separator
    auto printSep = [&]() {
        ss << "+";
        for (size_t w : widths) {
            ss << std::string(w + 2, '-') << "+";
        }
        ss << "\n";
    };

    // Print header
    printSep();
    ss << "|";
    for (size_t i = 0; i < columns.size(); i++) {
        ss << " " << std::left << std::setw(widths[i]) << columns[i] << " |";
    }
    ss << "\n";
    printSep();

    // Print rows
    for (const auto& row : rows) {
        ss << "|";
        for (size_t i = 0; i < columns.size(); i++) {
            auto val = row.get(columns[i]).toString();
            ss << " " << std::left << std::setw(widths[i]) << val << " |";
        }
        ss << "\n";
    }
    printSep();

    ss << rows.size() << " row(s)\n";
    return ss.str();
}

// ============================================================================
// Binding Implementation
// ============================================================================

Binding Binding::forNode(int id) {
    Binding b;
    b.type = Type::Node;
    b.nodeId = id;
    return b;
}

Binding Binding::forLink(int id) {
    Binding b;
    b.type = Type::Link;
    b.linkId = id;
    return b;
}

Binding Binding::forPath(std::vector<int> nodes, std::vector<int> links) {
    Binding b;
    b.type = Type::Path;
    b.pathNodeIds = std::move(nodes);
    b.pathLinkIds = std::move(links);
    return b;
}

Binding Binding::forValue(ResultValue val) {
    Binding b;
    b.type = Type::Value;
    b.value = std::move(val);
    return b;
}

// ============================================================================
// BindingContext Implementation
// ============================================================================

void BindingContext::bind(const std::string& name, Binding binding) {
    bindings_[name] = std::move(binding);
}

bool BindingContext::has(const std::string& name) const {
    return bindings_.find(name) != bindings_.end();
}

Binding BindingContext::get(const std::string& name) const {
    auto it = bindings_.find(name);
    if (it != bindings_.end()) {
        return it->second;
    }
    return Binding();
}

void BindingContext::clear() {
    bindings_.clear();
}

BindingContext BindingContext::clone() const {
    BindingContext copy;
    copy.bindings_ = bindings_;
    return copy;
}

std::vector<std::string> BindingContext::getNames() const {
    std::vector<std::string> names;
    names.reserve(bindings_.size());
    for (const auto& [name, _] : bindings_) {
        names.push_back(name);
    }
    return names;
}

} // namespace cyxwiz::query
