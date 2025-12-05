#include "cyxql_executor.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <spdlog/spdlog.h>

namespace cyxwiz::query {

Executor::Executor(gui::NodeEditor& editor) : editor_(editor) {}

QueryResult Executor::execute(const std::string& query) {
    auto start = std::chrono::high_resolution_clock::now();

    // Parse the query
    Parser parser(query);
    auto ast = parser.parse();

    if (parser.hasError()) {
        auto result = QueryResult::makeError(parser.getError(),
                                              parser.getErrorLine(),
                                              parser.getErrorColumn());
        auto end = std::chrono::high_resolution_clock::now();
        lastExecutionTimeMs_ = std::chrono::duration<double, std::milli>(end - start).count();
        result.executionTimeMs = lastExecutionTimeMs_;
        return result;
    }

    auto result = execute(ast);

    auto end = std::chrono::high_resolution_clock::now();
    lastExecutionTimeMs_ = std::chrono::duration<double, std::milli>(end - start).count();
    result.executionTimeMs = lastExecutionTimeMs_;

    return result;
}

QueryResult Executor::execute(const ASTNodePtr& ast) {
    if (!ast || ast->type != ASTNodeType::Query) {
        return QueryResult::makeError("Invalid AST: expected Query node");
    }

    // Refresh cached graph state
    nodes_ = editor_.GetNodes();
    links_ = editor_.GetLinks();

    // Start with a single empty binding context
    std::vector<BindingContext> contexts;
    contexts.push_back(BindingContext());

    QueryResult result = QueryResult::makeSuccess();
    ASTNodePtr returnClause = nullptr;
    ASTNodePtr orderByClause = nullptr;
    ASTNodePtr limitClause = nullptr;
    ASTNodePtr skipClause = nullptr;

    // Process clauses in order
    for (const auto& clause : ast->children) {
        if (!clause) continue;

        switch (clause->type) {
            case ASTNodeType::MatchClause:
                executeMatch(clause, contexts);
                break;

            case ASTNodeType::WhereClause:
                executeWhere(clause, contexts);
                break;

            case ASTNodeType::ReturnClause:
                returnClause = clause;
                break;

            case ASTNodeType::CreateClause:
                executeCreate(clause, result);
                break;

            case ASTNodeType::DeleteClause:
                executeDelete(clause, contexts, result);
                break;

            case ASTNodeType::SetClause:
                executeSet(clause, contexts, result);
                break;

            case ASTNodeType::OrderByClause:
                orderByClause = clause;
                break;

            case ASTNodeType::LimitClause:
                limitClause = clause;
                break;

            case ASTNodeType::SkipClause:
                skipClause = clause;
                break;

            case ASTNodeType::WithClause:
                executeWith(clause, contexts);
                break;

            default:
                spdlog::warn("Unknown clause type: {}", ASTNode::typeToString(clause->type));
                break;
        }
    }

    // Apply ORDER BY before SKIP/LIMIT
    if (orderByClause) {
        executeOrderBy(orderByClause, contexts);
    }

    // Apply SKIP
    if (skipClause) {
        executeSkip(skipClause, contexts);
    }

    // Apply LIMIT
    if (limitClause) {
        executeLimit(limitClause, contexts);
    }

    // Execute RETURN clause if present
    if (returnClause) {
        result = executeReturn(returnClause, contexts);
    }

    return result;
}

void Executor::executeMatch(const ASTNodePtr& clause, std::vector<BindingContext>& contexts) {
    // Process each pattern in the MATCH clause
    for (const auto& pattern : clause->children) {
        matchPattern(pattern, contexts);
    }
}

void Executor::matchPattern(const ASTNodePtr& pattern, std::vector<BindingContext>& contexts) {
    if (!pattern) return;

    if (pattern->type == ASTNodeType::PathPattern) {
        matchPathPattern(pattern, contexts);
    } else if (pattern->type == ASTNodeType::NodePattern) {
        // Single node pattern
        std::vector<BindingContext> newContexts;

        for (auto& ctx : contexts) {
            auto candidates = getCandidateNodes(pattern);
            for (int nodeId : candidates) {
                const auto* node = findNodeById(nodeId);
                if (node) {
                    BindingContext newCtx = ctx.clone();
                    if (matchNodePattern(pattern, *node, newCtx)) {
                        newContexts.push_back(newCtx);
                    }
                }
            }
        }

        contexts = std::move(newContexts);
    }
}

void Executor::matchPathPattern(const ASTNodePtr& path, std::vector<BindingContext>& contexts) {
    if (!path || path->children.empty()) return;

    // Path: node -[rel]-> node -[rel]-> node ...
    std::vector<BindingContext> newContexts;

    for (auto& ctx : contexts) {
        // Start with first node pattern
        auto& firstNode = path->children[0];
        if (firstNode->type != ASTNodeType::NodePattern) continue;

        auto candidates = getCandidateNodes(firstNode);

        for (int startNodeId : candidates) {
            const auto* startNode = findNodeById(startNodeId);
            if (!startNode) continue;

            BindingContext startCtx = ctx.clone();
            if (!matchNodePattern(firstNode, *startNode, startCtx)) continue;

            // Now traverse the path
            std::vector<BindingContext> pathContexts;
            pathContexts.push_back(startCtx);

            // Process relationship-node pairs
            for (size_t i = 1; i + 1 < path->children.size(); i += 2) {
                auto& relPattern = path->children[i];
                auto& nextNodePattern = path->children[i + 1];

                if (relPattern->type != ASTNodeType::RelationshipPattern) continue;
                if (nextNodePattern->type != ASTNodeType::NodePattern) continue;

                std::vector<BindingContext> nextPathContexts;

                for (auto& pathCtx : pathContexts) {
                    // Get current node from context
                    int currentNodeId = -1;

                    // Find the last bound node
                    for (size_t j = i; j > 0; j -= 2) {
                        auto& prevNode = path->children[j - 1];
                        if (prevNode->type == ASTNodeType::NodePattern) {
                            // Get variable name from node pattern
                            for (const auto& child : prevNode->children) {
                                if (child->type == ASTNodeType::Identifier) {
                                    if (pathCtx.has(child->value)) {
                                        auto binding = pathCtx.get(child->value);
                                        if (binding.type == Binding::Type::Node) {
                                            currentNodeId = binding.nodeId;
                                            break;
                                        }
                                    }
                                }
                            }
                            if (currentNodeId >= 0) break;
                        }
                    }

                    if (currentNodeId < 0 && i == 1) {
                        // Use start node
                        currentNodeId = startNodeId;
                    }

                    if (currentNodeId < 0) continue;

                    // Get direction from relationship pattern
                    std::string direction = relPattern->getProperty("direction", "right");

                    // Get candidate links
                    std::vector<gui::NodeLink> candidateLinks;
                    if (direction == "right") {
                        candidateLinks = getOutgoingLinks(currentNodeId);
                    } else if (direction == "left") {
                        candidateLinks = getIncomingLinks(currentNodeId);
                    } else {
                        candidateLinks = getAllLinks(currentNodeId);
                    }

                    // Try each link
                    for (const auto& link : candidateLinks) {
                        BindingContext linkCtx = pathCtx.clone();

                        if (!matchRelPattern(relPattern, link, linkCtx)) continue;

                        // Get the other node
                        int nextNodeId = (direction == "left") ? link.from_node : link.to_node;
                        if (direction == "both") {
                            nextNodeId = (link.from_node == currentNodeId) ? link.to_node : link.from_node;
                        }

                        const auto* nextNode = findNodeById(nextNodeId);
                        if (!nextNode) continue;

                        BindingContext nextCtx = linkCtx.clone();
                        if (matchNodePattern(nextNodePattern, *nextNode, nextCtx)) {
                            nextPathContexts.push_back(nextCtx);
                        }
                    }
                }

                pathContexts = std::move(nextPathContexts);
                if (pathContexts.empty()) break;
            }

            // Add all successful path contexts
            for (auto& pathCtx : pathContexts) {
                newContexts.push_back(std::move(pathCtx));
            }
        }
    }

    contexts = std::move(newContexts);
}

bool Executor::matchNodePattern(const ASTNodePtr& pattern, const gui::MLNode& node, BindingContext& ctx) {
    std::string varName;
    std::string labelName;

    // Extract variable and label from pattern
    for (const auto& child : pattern->children) {
        if (child->type == ASTNodeType::Identifier) {
            varName = child->value;
        } else if (child->type == ASTNodeType::Label) {
            labelName = child->value;
        }
    }

    // Check label match (if specified)
    if (!labelName.empty()) {
        std::string nodeTypeName = nodeTypeToString(node.type);
        if (nodeTypeName != labelName) {
            return false;
        }
    }

    // Bind variable (if specified)
    if (!varName.empty()) {
        // Check if already bound to a different node
        if (ctx.has(varName)) {
            auto existing = ctx.get(varName);
            if (existing.type == Binding::Type::Node && existing.nodeId != node.id) {
                return false;
            }
        }
        ctx.bind(varName, Binding::forNode(node.id));
    }

    // Check property constraints (if any)
    for (const auto& child : pattern->children) {
        if (child->type == ASTNodeType::PropertyList) {
            for (const auto& prop : child->children) {
                if (prop->type == ASTNodeType::PropertyAssignment) {
                    std::string key = prop->value;
                    if (!prop->children.empty()) {
                        auto expected = evaluateExpression(prop->children[0], ctx);
                        auto actual = getNodeProperty(node, key);
                        if (compareValues(actual, expected) != 0) {
                            return false;
                        }
                    }
                }
            }
        }
    }

    return true;
}

bool Executor::matchRelPattern(const ASTNodePtr& pattern, const gui::NodeLink& link, BindingContext& ctx) {
    std::string varName;
    std::string typeName;

    for (const auto& child : pattern->children) {
        if (child->type == ASTNodeType::Identifier) {
            varName = child->value;
        } else if (child->type == ASTNodeType::RelType) {
            typeName = child->value;
        }
    }

    // Check type match
    if (!typeName.empty()) {
        std::string linkTypeName = linkTypeToString(link.type);
        if (linkTypeName != typeName) {
            return false;
        }
    }

    // Bind variable
    if (!varName.empty()) {
        if (ctx.has(varName)) {
            auto existing = ctx.get(varName);
            if (existing.type == Binding::Type::Link && existing.linkId != link.id) {
                return false;
            }
        }
        ctx.bind(varName, Binding::forLink(link.id));
    }

    return true;
}

std::vector<int> Executor::getCandidateNodes(const ASTNodePtr& pattern) {
    std::vector<int> candidates;

    // Check for label filter
    std::string labelFilter;
    for (const auto& child : pattern->children) {
        if (child->type == ASTNodeType::Label) {
            labelFilter = child->value;
            break;
        }
    }

    for (const auto& node : nodes_) {
        if (labelFilter.empty() || nodeTypeToString(node.type) == labelFilter) {
            candidates.push_back(node.id);
        }
    }

    return candidates;
}

std::vector<gui::NodeLink> Executor::getOutgoingLinks(int nodeId) {
    std::vector<gui::NodeLink> result;
    for (const auto& link : links_) {
        if (link.from_node == nodeId) {
            result.push_back(link);
        }
    }
    return result;
}

std::vector<gui::NodeLink> Executor::getIncomingLinks(int nodeId) {
    std::vector<gui::NodeLink> result;
    for (const auto& link : links_) {
        if (link.to_node == nodeId) {
            result.push_back(link);
        }
    }
    return result;
}

std::vector<gui::NodeLink> Executor::getAllLinks(int nodeId) {
    std::vector<gui::NodeLink> result;
    for (const auto& link : links_) {
        if (link.from_node == nodeId || link.to_node == nodeId) {
            result.push_back(link);
        }
    }
    return result;
}

void Executor::executeWhere(const ASTNodePtr& clause, std::vector<BindingContext>& contexts) {
    if (clause->children.empty()) return;

    auto& condition = clause->children[0];
    std::vector<BindingContext> filtered;

    for (auto& ctx : contexts) {
        auto result = evaluateExpression(condition, ctx);
        if (result.toBool()) {
            filtered.push_back(std::move(ctx));
        }
    }

    contexts = std::move(filtered);
}

QueryResult Executor::executeReturn(const ASTNodePtr& clause, const std::vector<BindingContext>& contexts) {
    QueryResult result = QueryResult::makeSuccess();

    // Check for RETURN *
    if (!clause->children.empty() && clause->children[0]->type == ASTNodeType::Star) {
        // Return all bound variables
        if (!contexts.empty()) {
            auto names = contexts[0].getNames();
            for (const auto& name : names) {
                result.addColumn(name);
            }

            for (const auto& ctx : contexts) {
                ResultRow row;
                for (const auto& name : names) {
                    auto binding = ctx.get(name);
                    switch (binding.type) {
                        case Binding::Type::Node:
                            row.set(name, ResultValue::makeNode(binding.nodeId));
                            break;
                        case Binding::Type::Link:
                            row.set(name, ResultValue::makeLink(binding.linkId));
                            break;
                        case Binding::Type::Path:
                            row.set(name, ResultValue::makePath(binding.pathNodeIds, binding.pathLinkIds));
                            break;
                        case Binding::Type::Value:
                            row.set(name, binding.value);
                            break;
                    }
                }
                result.addRow(row);
            }
        }
        return result;
    }

    // Collect column names
    std::vector<std::string> columnNames;
    for (const auto& item : clause->children) {
        if (item->type == ASTNodeType::Alias) {
            columnNames.push_back(item->value);
        } else if (item->type == ASTNodeType::Identifier) {
            columnNames.push_back(item->value);
        } else if (item->type == ASTNodeType::PropertyAccess) {
            // Build column name from property access
            std::string name;
            if (!item->children.empty() && item->children[0]->type == ASTNodeType::Identifier) {
                name = item->children[0]->value + "." + item->value;
            } else {
                name = item->value;
            }
            columnNames.push_back(name);
        } else if (item->type == ASTNodeType::FunctionCall) {
            columnNames.push_back(item->value + "(...)");
        } else {
            columnNames.push_back("expr");
        }
    }

    for (const auto& name : columnNames) {
        result.addColumn(name);
    }

    // Check for aggregation
    bool hasAggregation = false;
    for (const auto& item : clause->children) {
        if (item->type == ASTNodeType::FunctionCall) {
            std::string fn = item->value;
            if (fn == "count" || fn == "COUNT" ||
                fn == "sum" || fn == "SUM" ||
                fn == "avg" || fn == "AVG" ||
                fn == "min" || fn == "MIN" ||
                fn == "max" || fn == "MAX" ||
                fn == "collect" || fn == "COLLECT") {
                hasAggregation = true;
                break;
            }
        } else if (item->type == ASTNodeType::Alias && !item->children.empty()) {
            auto& expr = item->children[0];
            if (expr->type == ASTNodeType::FunctionCall) {
                std::string fn = expr->value;
                if (fn == "count" || fn == "COUNT" ||
                    fn == "sum" || fn == "SUM" ||
                    fn == "avg" || fn == "AVG" ||
                    fn == "min" || fn == "MIN" ||
                    fn == "max" || fn == "MAX" ||
                    fn == "collect" || fn == "COLLECT") {
                    hasAggregation = true;
                    break;
                }
            }
        }
    }

    if (hasAggregation) {
        // Aggregate all contexts into single row
        ResultRow row;
        for (size_t i = 0; i < clause->children.size(); i++) {
            auto& item = clause->children[i];
            auto& colName = columnNames[i];

            auto expr = (item->type == ASTNodeType::Alias && !item->children.empty())
                         ? item->children[0] : item;

            if (expr->type == ASTNodeType::FunctionCall) {
                std::string fn = expr->value;
                std::transform(fn.begin(), fn.end(), fn.begin(), ::tolower);

                if (fn == "count" || fn == "sum" || fn == "avg" || fn == "min" || fn == "max" || fn == "collect") {
                    std::vector<ResultValue> values;
                    for (const auto& ctx : contexts) {
                        if (!expr->children.empty()) {
                            values.push_back(evaluateExpression(expr->children[0], ctx));
                        } else {
                            values.push_back(ResultValue::makeInt(1));  // COUNT(*)
                        }
                    }
                    row.set(colName, aggregate(fn, values));
                } else {
                    // Non-aggregate function, use first context
                    if (!contexts.empty()) {
                        row.set(colName, evaluateExpression(expr, contexts[0]));
                    }
                }
            } else {
                // Non-aggregate expression, use first context
                if (!contexts.empty()) {
                    row.set(colName, evaluateExpression(expr, contexts[0]));
                }
            }
        }
        result.addRow(row);
    } else {
        // No aggregation - one row per context
        for (const auto& ctx : contexts) {
            ResultRow row;
            for (size_t i = 0; i < clause->children.size(); i++) {
                auto& item = clause->children[i];
                auto& colName = columnNames[i];

                auto expr = (item->type == ASTNodeType::Alias && !item->children.empty())
                             ? item->children[0] : item;

                row.set(colName, evaluateExpression(expr, ctx));
            }
            result.addRow(row);
        }
    }

    return result;
}

void Executor::executeCreate(const ASTNodePtr& clause, QueryResult& result) {
    // For now, just log that we would create nodes
    // Actual implementation would modify the graph
    spdlog::info("CREATE clause execution (not fully implemented)");
    result.nodesCreated = 0;  // Would increment as nodes are created
}

void Executor::executeDelete(const ASTNodePtr& clause, const std::vector<BindingContext>& contexts, QueryResult& result) {
    spdlog::info("DELETE clause execution (not fully implemented)");
    result.nodesDeleted = 0;
}

void Executor::executeSet(const ASTNodePtr& clause, std::vector<BindingContext>& contexts, QueryResult& result) {
    spdlog::info("SET clause execution (not fully implemented)");
    result.propertiesSet = 0;
}

void Executor::executeOrderBy(const ASTNodePtr& clause, std::vector<BindingContext>& contexts) {
    if (clause->children.empty()) return;

    // Sort by the first sort item for now
    auto& sortItem = clause->children[0];
    bool descending = sortItem->getProperty("order") == "desc";

    auto& sortExpr = sortItem->children.empty() ? sortItem : sortItem->children[0];

    std::stable_sort(contexts.begin(), contexts.end(),
        [this, &sortExpr, descending](const BindingContext& a, const BindingContext& b) {
            auto valA = evaluateExpression(sortExpr, a);
            auto valB = evaluateExpression(sortExpr, b);
            int cmp = compareValues(valA, valB);
            return descending ? (cmp > 0) : (cmp < 0);
        });
}

void Executor::executeLimit(const ASTNodePtr& clause, std::vector<BindingContext>& contexts) {
    if (clause->children.empty()) return;

    auto& limitExpr = clause->children[0];
    if (limitExpr->type == ASTNodeType::Literal) {
        int64_t limit = std::get<int64_t>(limitExpr->literal);
        if (limit >= 0 && static_cast<size_t>(limit) < contexts.size()) {
            contexts.resize(limit);
        }
    }
}

void Executor::executeSkip(const ASTNodePtr& clause, std::vector<BindingContext>& contexts) {
    if (clause->children.empty()) return;

    auto& skipExpr = clause->children[0];
    if (skipExpr->type == ASTNodeType::Literal) {
        int64_t skip = std::get<int64_t>(skipExpr->literal);
        if (skip > 0 && static_cast<size_t>(skip) < contexts.size()) {
            contexts.erase(contexts.begin(), contexts.begin() + skip);
        } else if (skip >= static_cast<int64_t>(contexts.size())) {
            contexts.clear();
        }
    }
}

void Executor::executeWith(const ASTNodePtr& clause, std::vector<BindingContext>& contexts) {
    // WITH acts like RETURN but continues the query
    // For now, we'll keep all bindings
    spdlog::info("WITH clause - passing through bindings");
}

ResultValue Executor::evaluateExpression(const ASTNodePtr& expr, const BindingContext& ctx) {
    if (!expr) return ResultValue::makeNull();

    switch (expr->type) {
        case ASTNodeType::Literal:
            return std::visit([](auto&& arg) -> ResultValue {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, std::nullptr_t>) {
                    return ResultValue::makeNull();
                } else if constexpr (std::is_same_v<T, bool>) {
                    return ResultValue::makeBool(arg);
                } else if constexpr (std::is_same_v<T, int64_t>) {
                    return ResultValue::makeInt(arg);
                } else if constexpr (std::is_same_v<T, double>) {
                    return ResultValue::makeFloat(arg);
                } else if constexpr (std::is_same_v<T, std::string>) {
                    return ResultValue::makeString(arg);
                }
                return ResultValue::makeNull();
            }, expr->literal);

        case ASTNodeType::Identifier: {
            if (ctx.has(expr->value)) {
                auto binding = ctx.get(expr->value);
                switch (binding.type) {
                    case Binding::Type::Node:
                        return ResultValue::makeNode(binding.nodeId);
                    case Binding::Type::Link:
                        return ResultValue::makeLink(binding.linkId);
                    case Binding::Type::Path:
                        return ResultValue::makePath(binding.pathNodeIds, binding.pathLinkIds);
                    case Binding::Type::Value:
                        return binding.value;
                }
            }
            return ResultValue::makeNull();
        }

        case ASTNodeType::PropertyAccess:
            return evaluatePropertyAccess(expr, ctx);

        case ASTNodeType::FunctionCall:
            return evaluateFunctionCall(expr, ctx);

        case ASTNodeType::BinaryExpr:
            return evaluateBinaryExpr(expr, ctx);

        case ASTNodeType::UnaryExpr:
            return evaluateUnaryExpr(expr, ctx);

        case ASTNodeType::ComparisonExpr: {
            // IS NULL / IS NOT NULL
            if (expr->value == "IS NULL") {
                auto val = evaluateExpression(expr->children[0], ctx);
                return ResultValue::makeBool(val.type == ResultValue::Type::Null);
            } else if (expr->value == "IS NOT NULL") {
                auto val = evaluateExpression(expr->children[0], ctx);
                return ResultValue::makeBool(val.type != ResultValue::Type::Null);
            }
            break;
        }

        case ASTNodeType::ListExpr: {
            std::vector<ResultValue> items;
            for (const auto& child : expr->children) {
                items.push_back(evaluateExpression(child, ctx));
            }
            return ResultValue::makeList(items);
        }

        case ASTNodeType::Star:
            return ResultValue::makeString("*");

        default:
            break;
    }

    return ResultValue::makeNull();
}

ResultValue Executor::evaluatePropertyAccess(const ASTNodePtr& expr, const BindingContext& ctx) {
    if (expr->children.empty()) return ResultValue::makeNull();

    auto& base = expr->children[0];
    std::string propName = expr->value;

    if (base->type == ASTNodeType::Identifier) {
        std::string varName = base->value;
        if (ctx.has(varName)) {
            auto binding = ctx.get(varName);
            if (binding.type == Binding::Type::Node) {
                const auto* node = findNodeById(binding.nodeId);
                if (node) {
                    return getNodeProperty(*node, propName);
                }
            } else if (binding.type == Binding::Type::Link) {
                const auto* link = findLinkById(binding.linkId);
                if (link) {
                    return getLinkProperty(*link, propName);
                }
            }
        }
    } else if (base->type == ASTNodeType::PropertyAccess) {
        // Nested property access
        auto baseVal = evaluatePropertyAccess(base, ctx);
        if (baseVal.type == ResultValue::Type::Map && baseVal.map.count(propName)) {
            return baseVal.map[propName];
        }
    }

    return ResultValue::makeNull();
}

ResultValue Executor::evaluateFunctionCall(const ASTNodePtr& expr, const BindingContext& ctx) {
    std::string fn = expr->value;
    std::transform(fn.begin(), fn.end(), fn.begin(), ::tolower);

    // Built-in functions
    if (fn == "type") {
        if (!expr->children.empty()) {
            auto arg = evaluateExpression(expr->children[0], ctx);
            if (arg.type == ResultValue::Type::Node) {
                const auto* node = findNodeById(arg.nodeId);
                if (node) {
                    return ResultValue::makeString(nodeTypeToString(node->type));
                }
            } else if (arg.type == ResultValue::Type::Link) {
                const auto* link = findLinkById(arg.linkId);
                if (link) {
                    return ResultValue::makeString(linkTypeToString(link->type));
                }
            }
        }
        return ResultValue::makeNull();
    }

    if (fn == "id") {
        if (!expr->children.empty()) {
            auto arg = evaluateExpression(expr->children[0], ctx);
            if (arg.type == ResultValue::Type::Node) {
                return ResultValue::makeInt(arg.nodeId);
            } else if (arg.type == ResultValue::Type::Link) {
                return ResultValue::makeInt(arg.linkId);
            }
        }
        return ResultValue::makeNull();
    }

    if (fn == "properties") {
        if (!expr->children.empty()) {
            auto arg = evaluateExpression(expr->children[0], ctx);
            if (arg.type == ResultValue::Type::Node) {
                const auto* node = findNodeById(arg.nodeId);
                if (node) {
                    std::map<std::string, ResultValue> props;
                    props["name"] = ResultValue::makeString(node->name);
                    props["type"] = ResultValue::makeString(nodeTypeToString(node->type));
                    for (const auto& [k, v] : node->parameters) {
                        props[k] = ResultValue::makeString(v);
                    }
                    return ResultValue::makeMap(props);
                }
            }
        }
        return ResultValue::makeNull();
    }

    if (fn == "size") {
        if (!expr->children.empty()) {
            auto arg = evaluateExpression(expr->children[0], ctx);
            if (arg.type == ResultValue::Type::String) {
                return ResultValue::makeInt(std::get<std::string>(arg.scalar).size());
            } else if (arg.type == ResultValue::Type::List) {
                return ResultValue::makeInt(arg.list.size());
            }
        }
        return ResultValue::makeInt(0);
    }

    // Aggregate functions are handled in executeReturn
    return ResultValue::makeNull();
}

ResultValue Executor::evaluateBinaryExpr(const ASTNodePtr& expr, const BindingContext& ctx) {
    if (expr->children.size() < 2) return ResultValue::makeNull();

    std::string op = expr->value;
    auto left = evaluateExpression(expr->children[0], ctx);
    auto right = evaluateExpression(expr->children[1], ctx);

    // Logical operators
    if (op == "AND") {
        return ResultValue::makeBool(left.toBool() && right.toBool());
    }
    if (op == "OR") {
        return ResultValue::makeBool(left.toBool() || right.toBool());
    }
    if (op == "XOR") {
        return ResultValue::makeBool(left.toBool() != right.toBool());
    }

    // Comparison operators
    if (op == "=" || op == "==") {
        return ResultValue::makeBool(compareValues(left, right) == 0);
    }
    if (op == "<>" || op == "!=") {
        return ResultValue::makeBool(compareValues(left, right) != 0);
    }
    if (op == "<") {
        return ResultValue::makeBool(compareValues(left, right) < 0);
    }
    if (op == ">") {
        return ResultValue::makeBool(compareValues(left, right) > 0);
    }
    if (op == "<=") {
        return ResultValue::makeBool(compareValues(left, right) <= 0);
    }
    if (op == ">=") {
        return ResultValue::makeBool(compareValues(left, right) >= 0);
    }

    // String operators
    if (op == "CONTAINS" || op == "contains") {
        if (left.type == ResultValue::Type::String && right.type == ResultValue::Type::String) {
            auto& s1 = std::get<std::string>(left.scalar);
            auto& s2 = std::get<std::string>(right.scalar);
            return ResultValue::makeBool(s1.find(s2) != std::string::npos);
        }
        return ResultValue::makeBool(false);
    }

    if (op == "STARTS WITH" || op == "starts with") {
        if (left.type == ResultValue::Type::String && right.type == ResultValue::Type::String) {
            auto& s1 = std::get<std::string>(left.scalar);
            auto& s2 = std::get<std::string>(right.scalar);
            return ResultValue::makeBool(s1.rfind(s2, 0) == 0);
        }
        return ResultValue::makeBool(false);
    }

    if (op == "ENDS WITH" || op == "ends with") {
        if (left.type == ResultValue::Type::String && right.type == ResultValue::Type::String) {
            auto& s1 = std::get<std::string>(left.scalar);
            auto& s2 = std::get<std::string>(right.scalar);
            if (s2.size() > s1.size()) return ResultValue::makeBool(false);
            return ResultValue::makeBool(s1.compare(s1.size() - s2.size(), s2.size(), s2) == 0);
        }
        return ResultValue::makeBool(false);
    }

    if (op == "IN" || op == "in") {
        if (right.type == ResultValue::Type::List) {
            for (const auto& item : right.list) {
                if (compareValues(left, item) == 0) {
                    return ResultValue::makeBool(true);
                }
            }
        }
        return ResultValue::makeBool(false);
    }

    // Arithmetic operators
    if (op == "+") {
        if (left.type == ResultValue::Type::String || right.type == ResultValue::Type::String) {
            return ResultValue::makeString(left.toString() + right.toString());
        }
        if (left.type == ResultValue::Type::Float || right.type == ResultValue::Type::Float) {
            return ResultValue::makeFloat(left.toFloat() + right.toFloat());
        }
        return ResultValue::makeInt(left.toInt() + right.toInt());
    }
    if (op == "-") {
        if (left.type == ResultValue::Type::Float || right.type == ResultValue::Type::Float) {
            return ResultValue::makeFloat(left.toFloat() - right.toFloat());
        }
        return ResultValue::makeInt(left.toInt() - right.toInt());
    }
    if (op == "*") {
        if (left.type == ResultValue::Type::Float || right.type == ResultValue::Type::Float) {
            return ResultValue::makeFloat(left.toFloat() * right.toFloat());
        }
        return ResultValue::makeInt(left.toInt() * right.toInt());
    }
    if (op == "/") {
        double r = right.toFloat();
        if (r == 0.0) return ResultValue::makeNull();
        return ResultValue::makeFloat(left.toFloat() / r);
    }
    if (op == "%") {
        int64_t r = right.toInt();
        if (r == 0) return ResultValue::makeNull();
        return ResultValue::makeInt(left.toInt() % r);
    }
    if (op == "^") {
        return ResultValue::makeFloat(std::pow(left.toFloat(), right.toFloat()));
    }

    return ResultValue::makeNull();
}

ResultValue Executor::evaluateUnaryExpr(const ASTNodePtr& expr, const BindingContext& ctx) {
    if (expr->children.empty()) return ResultValue::makeNull();

    std::string op = expr->value;
    auto val = evaluateExpression(expr->children[0], ctx);

    if (op == "NOT" || op == "not") {
        return ResultValue::makeBool(!val.toBool());
    }
    if (op == "-") {
        if (val.type == ResultValue::Type::Float) {
            return ResultValue::makeFloat(-val.toFloat());
        }
        return ResultValue::makeInt(-val.toInt());
    }

    return ResultValue::makeNull();
}

std::string Executor::nodeTypeToString(gui::NodeType type) const {
    // Map NodeType enum to string names
    static const std::map<gui::NodeType, std::string> typeNames = {
        {gui::NodeType::Dense, "Dense"},
        {gui::NodeType::Conv1D, "Conv1D"},
        {gui::NodeType::Conv2D, "Conv2D"},
        {gui::NodeType::Conv3D, "Conv3D"},
        {gui::NodeType::DepthwiseConv2D, "DepthwiseConv2D"},
        {gui::NodeType::MaxPool2D, "MaxPool2D"},
        {gui::NodeType::AvgPool2D, "AvgPool2D"},
        {gui::NodeType::GlobalMaxPool, "GlobalMaxPool"},
        {gui::NodeType::GlobalAvgPool, "GlobalAvgPool"},
        {gui::NodeType::AdaptiveAvgPool, "AdaptiveAvgPool"},
        {gui::NodeType::BatchNorm, "BatchNorm"},
        {gui::NodeType::LayerNorm, "LayerNorm"},
        {gui::NodeType::GroupNorm, "GroupNorm"},
        {gui::NodeType::InstanceNorm, "InstanceNorm"},
        {gui::NodeType::Dropout, "Dropout"},
        {gui::NodeType::Flatten, "Flatten"},
        {gui::NodeType::RNN, "RNN"},
        {gui::NodeType::LSTM, "LSTM"},
        {gui::NodeType::GRU, "GRU"},
        {gui::NodeType::Bidirectional, "Bidirectional"},
        {gui::NodeType::TimeDistributed, "TimeDistributed"},
        {gui::NodeType::Embedding, "Embedding"},
        {gui::NodeType::MultiHeadAttention, "MultiHeadAttention"},
        {gui::NodeType::SelfAttention, "SelfAttention"},
        {gui::NodeType::CrossAttention, "CrossAttention"},
        {gui::NodeType::LinearAttention, "LinearAttention"},
        {gui::NodeType::TransformerEncoder, "TransformerEncoder"},
        {gui::NodeType::TransformerDecoder, "TransformerDecoder"},
        {gui::NodeType::PositionalEncoding, "PositionalEncoding"},
        {gui::NodeType::ReLU, "ReLU"},
        {gui::NodeType::LeakyReLU, "LeakyReLU"},
        {gui::NodeType::PReLU, "PReLU"},
        {gui::NodeType::ELU, "ELU"},
        {gui::NodeType::SELU, "SELU"},
        {gui::NodeType::GELU, "GELU"},
        {gui::NodeType::Swish, "Swish"},
        {gui::NodeType::Mish, "Mish"},
        {gui::NodeType::Sigmoid, "Sigmoid"},
        {gui::NodeType::Tanh, "Tanh"},
        {gui::NodeType::Softmax, "Softmax"},
        {gui::NodeType::Reshape, "Reshape"},
        {gui::NodeType::Permute, "Permute"},
        {gui::NodeType::Squeeze, "Squeeze"},
        {gui::NodeType::Unsqueeze, "Unsqueeze"},
        {gui::NodeType::View, "View"},
        {gui::NodeType::Split, "Split"},
        {gui::NodeType::Concatenate, "Concatenate"},
        {gui::NodeType::Add, "Add"},
        {gui::NodeType::Multiply, "Multiply"},
        {gui::NodeType::Average, "Average"},
        {gui::NodeType::Output, "Output"},
        {gui::NodeType::MSELoss, "MSELoss"},
        {gui::NodeType::CrossEntropyLoss, "CrossEntropyLoss"},
        {gui::NodeType::BCELoss, "BCELoss"},
        {gui::NodeType::BCEWithLogits, "BCEWithLogits"},
        {gui::NodeType::L1Loss, "L1Loss"},
        {gui::NodeType::SmoothL1Loss, "SmoothL1Loss"},
        {gui::NodeType::HuberLoss, "HuberLoss"},
        {gui::NodeType::NLLLoss, "NLLLoss"},
        {gui::NodeType::SGD, "SGD"},
        {gui::NodeType::Adam, "Adam"},
        {gui::NodeType::AdamW, "AdamW"},
        {gui::NodeType::RMSprop, "RMSprop"},
        {gui::NodeType::Adagrad, "Adagrad"},
        {gui::NodeType::NAdam, "NAdam"},
        {gui::NodeType::StepLR, "StepLR"},
        {gui::NodeType::CosineAnnealing, "CosineAnnealing"},
        {gui::NodeType::ReduceOnPlateau, "ReduceOnPlateau"},
        {gui::NodeType::ExponentialLR, "ExponentialLR"},
        {gui::NodeType::WarmupScheduler, "WarmupScheduler"},
        {gui::NodeType::L1Regularization, "L1Regularization"},
        {gui::NodeType::L2Regularization, "L2Regularization"},
        {gui::NodeType::ElasticNet, "ElasticNet"},
        {gui::NodeType::Lambda, "Lambda"},
        {gui::NodeType::Identity, "Identity"},
        {gui::NodeType::Constant, "Constant"},
        {gui::NodeType::Parameter, "Parameter"},
        {gui::NodeType::DatasetInput, "DatasetInput"},
        {gui::NodeType::DataLoader, "DataLoader"},
        {gui::NodeType::Augmentation, "Augmentation"},
        {gui::NodeType::DataSplit, "DataSplit"},
        {gui::NodeType::TensorReshape, "TensorReshape"},
        {gui::NodeType::Normalize, "Normalize"},
        {gui::NodeType::OneHotEncode, "OneHotEncode"},
    };

    auto it = typeNames.find(type);
    return it != typeNames.end() ? it->second : "Unknown";
}

gui::NodeType Executor::stringToNodeType(const std::string& str) const {
    static const std::map<std::string, gui::NodeType> nameToType = {
        {"Dense", gui::NodeType::Dense},
        {"Conv2D", gui::NodeType::Conv2D},
        {"MaxPool2D", gui::NodeType::MaxPool2D},
        {"BatchNorm", gui::NodeType::BatchNorm},
        {"Dropout", gui::NodeType::Dropout},
        {"ReLU", gui::NodeType::ReLU},
        {"LSTM", gui::NodeType::LSTM},
        {"GRU", gui::NodeType::GRU},
        {"MultiHeadAttention", gui::NodeType::MultiHeadAttention},
        {"DatasetInput", gui::NodeType::DatasetInput},
        {"Output", gui::NodeType::Output},
        // Add more as needed
    };

    auto it = nameToType.find(str);
    return it != nameToType.end() ? it->second : gui::NodeType::Dense;
}

std::string Executor::linkTypeToString(gui::LinkType type) const {
    switch (type) {
        case gui::LinkType::TensorFlow: return "TensorFlow";
        case gui::LinkType::ResidualSkip: return "ResidualSkip";
        case gui::LinkType::DenseSkip: return "DenseSkip";
        case gui::LinkType::AttentionFlow: return "AttentionFlow";
        case gui::LinkType::GradientFlow: return "GradientFlow";
        case gui::LinkType::ParameterFlow: return "ParameterFlow";
        case gui::LinkType::LossFlow: return "LossFlow";
        default: return "Unknown";
    }
}

gui::LinkType Executor::stringToLinkType(const std::string& str) const {
    if (str == "TensorFlow") return gui::LinkType::TensorFlow;
    if (str == "ResidualSkip") return gui::LinkType::ResidualSkip;
    if (str == "DenseSkip") return gui::LinkType::DenseSkip;
    if (str == "AttentionFlow") return gui::LinkType::AttentionFlow;
    if (str == "GradientFlow") return gui::LinkType::GradientFlow;
    if (str == "ParameterFlow") return gui::LinkType::ParameterFlow;
    if (str == "LossFlow") return gui::LinkType::LossFlow;
    return gui::LinkType::TensorFlow;
}

const gui::MLNode* Executor::findNodeById(int id) const {
    for (const auto& node : nodes_) {
        if (node.id == id) return &node;
    }
    return nullptr;
}

const gui::NodeLink* Executor::findLinkById(int id) const {
    for (const auto& link : links_) {
        if (link.id == id) return &link;
    }
    return nullptr;
}

ResultValue Executor::getNodeProperty(const gui::MLNode& node, const std::string& prop) const {
    if (prop == "id") return ResultValue::makeInt(node.id);
    if (prop == "name") return ResultValue::makeString(node.name);
    if (prop == "type") return ResultValue::makeString(nodeTypeToString(node.type));

    // Check parameters
    auto it = node.parameters.find(prop);
    if (it != node.parameters.end()) {
        // Try to parse as number
        try {
            if (it->second.find('.') != std::string::npos) {
                return ResultValue::makeFloat(std::stod(it->second));
            } else {
                return ResultValue::makeInt(std::stoll(it->second));
            }
        } catch (...) {
            return ResultValue::makeString(it->second);
        }
    }

    return ResultValue::makeNull();
}

ResultValue Executor::getLinkProperty(const gui::NodeLink& link, const std::string& prop) const {
    if (prop == "id") return ResultValue::makeInt(link.id);
    if (prop == "type") return ResultValue::makeString(linkTypeToString(link.type));
    if (prop == "from_node") return ResultValue::makeInt(link.from_node);
    if (prop == "to_node") return ResultValue::makeInt(link.to_node);
    if (prop == "from_pin") return ResultValue::makeInt(link.from_pin);
    if (prop == "to_pin") return ResultValue::makeInt(link.to_pin);

    return ResultValue::makeNull();
}

int Executor::compareValues(const ResultValue& a, const ResultValue& b) const {
    // Handle null
    if (a.type == ResultValue::Type::Null && b.type == ResultValue::Type::Null) return 0;
    if (a.type == ResultValue::Type::Null) return -1;
    if (b.type == ResultValue::Type::Null) return 1;

    // Compare by type
    if (a.type == ResultValue::Type::String || b.type == ResultValue::Type::String) {
        return a.toString().compare(b.toString());
    }

    if (a.type == ResultValue::Type::Float || b.type == ResultValue::Type::Float) {
        double diff = a.toFloat() - b.toFloat();
        if (diff < 0) return -1;
        if (diff > 0) return 1;
        return 0;
    }

    if (a.type == ResultValue::Type::Int || b.type == ResultValue::Type::Int) {
        int64_t diff = a.toInt() - b.toInt();
        if (diff < 0) return -1;
        if (diff > 0) return 1;
        return 0;
    }

    if (a.type == ResultValue::Type::Bool && b.type == ResultValue::Type::Bool) {
        bool ba = std::get<bool>(a.scalar);
        bool bb = std::get<bool>(b.scalar);
        if (ba == bb) return 0;
        return ba ? 1 : -1;
    }

    if (a.type == ResultValue::Type::Node && b.type == ResultValue::Type::Node) {
        return a.nodeId - b.nodeId;
    }

    if (a.type == ResultValue::Type::Link && b.type == ResultValue::Type::Link) {
        return a.linkId - b.linkId;
    }

    return 0;
}

ResultValue Executor::aggregate(const std::string& func, const std::vector<ResultValue>& values) {
    if (func == "count") {
        return ResultValue::makeInt(values.size());
    }

    if (func == "sum") {
        double sum = 0;
        for (const auto& v : values) {
            sum += v.toFloat();
        }
        return ResultValue::makeFloat(sum);
    }

    if (func == "avg") {
        if (values.empty()) return ResultValue::makeNull();
        double sum = 0;
        for (const auto& v : values) {
            sum += v.toFloat();
        }
        return ResultValue::makeFloat(sum / values.size());
    }

    if (func == "min") {
        if (values.empty()) return ResultValue::makeNull();
        ResultValue minVal = values[0];
        for (size_t i = 1; i < values.size(); i++) {
            if (compareValues(values[i], minVal) < 0) {
                minVal = values[i];
            }
        }
        return minVal;
    }

    if (func == "max") {
        if (values.empty()) return ResultValue::makeNull();
        ResultValue maxVal = values[0];
        for (size_t i = 1; i < values.size(); i++) {
            if (compareValues(values[i], maxVal) > 0) {
                maxVal = values[i];
            }
        }
        return maxVal;
    }

    if (func == "collect") {
        return ResultValue::makeList(std::vector<ResultValue>(values.begin(), values.end()));
    }

    return ResultValue::makeNull();
}

// Convenience function
QueryResult runQuery(gui::NodeEditor& editor, const std::string& query) {
    Executor executor(editor);
    return executor.execute(query);
}

} // namespace cyxwiz::query
