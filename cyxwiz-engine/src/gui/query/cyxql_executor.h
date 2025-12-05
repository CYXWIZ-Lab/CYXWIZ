#pragma once

#include "cyxql_types.h"
#include "cyxql_parser.h"
#include "../node_editor.h"
#include <string>
#include <vector>
#include <functional>
#include <chrono>

namespace cyxwiz::query {

/**
 * CyxQL Executor - Executes parsed queries against a NodeEditor graph
 *
 * Supports:
 * - MATCH: Find nodes and paths matching patterns
 * - WHERE: Filter matches by conditions
 * - RETURN: Select what data to return
 * - CREATE: Create new nodes and relationships
 * - DELETE: Delete nodes (and their connections)
 * - SET: Modify node properties
 * - ORDER BY: Sort results
 * - LIMIT/SKIP: Pagination
 *
 * Built-in functions:
 * - type(n): Get node type as string
 * - id(n): Get node ID
 * - properties(n): Get all properties as map
 * - count(x): Count items
 * - collect(x): Aggregate into list
 */
class Executor {
public:
    explicit Executor(gui::NodeEditor& editor);

    // Execute a query string
    QueryResult execute(const std::string& query);

    // Execute a parsed AST
    QueryResult execute(const ASTNodePtr& ast);

    // Get last execution time
    double getLastExecutionTimeMs() const { return lastExecutionTimeMs_; }

private:
    // Clause execution
    void executeMatch(const ASTNodePtr& clause, std::vector<BindingContext>& contexts);
    void executeWhere(const ASTNodePtr& clause, std::vector<BindingContext>& contexts);
    QueryResult executeReturn(const ASTNodePtr& clause, const std::vector<BindingContext>& contexts);
    void executeCreate(const ASTNodePtr& clause, QueryResult& result);
    void executeDelete(const ASTNodePtr& clause, const std::vector<BindingContext>& contexts, QueryResult& result);
    void executeSet(const ASTNodePtr& clause, std::vector<BindingContext>& contexts, QueryResult& result);
    void executeOrderBy(const ASTNodePtr& clause, std::vector<BindingContext>& contexts);
    void executeLimit(const ASTNodePtr& clause, std::vector<BindingContext>& contexts);
    void executeSkip(const ASTNodePtr& clause, std::vector<BindingContext>& contexts);
    void executeWith(const ASTNodePtr& clause, std::vector<BindingContext>& contexts);

    // Pattern matching
    void matchPattern(const ASTNodePtr& pattern, std::vector<BindingContext>& contexts);
    void matchPathPattern(const ASTNodePtr& path, std::vector<BindingContext>& contexts);
    bool matchNodePattern(const ASTNodePtr& node, const gui::MLNode& graphNode, BindingContext& ctx);
    bool matchRelPattern(const ASTNodePtr& rel, const gui::NodeLink& link, BindingContext& ctx);

    // Get candidate nodes for a pattern
    std::vector<int> getCandidateNodes(const ASTNodePtr& nodePattern);

    // Get outgoing/incoming links for a node
    std::vector<gui::NodeLink> getOutgoingLinks(int nodeId);
    std::vector<gui::NodeLink> getIncomingLinks(int nodeId);
    std::vector<gui::NodeLink> getAllLinks(int nodeId);

    // Expression evaluation
    ResultValue evaluateExpression(const ASTNodePtr& expr, const BindingContext& ctx);
    ResultValue evaluatePropertyAccess(const ASTNodePtr& expr, const BindingContext& ctx);
    ResultValue evaluateFunctionCall(const ASTNodePtr& expr, const BindingContext& ctx);
    ResultValue evaluateBinaryExpr(const ASTNodePtr& expr, const BindingContext& ctx);
    ResultValue evaluateUnaryExpr(const ASTNodePtr& expr, const BindingContext& ctx);

    // Type conversion
    std::string nodeTypeToString(gui::NodeType type) const;
    gui::NodeType stringToNodeType(const std::string& str) const;
    std::string linkTypeToString(gui::LinkType type) const;
    gui::LinkType stringToLinkType(const std::string& str) const;

    // Node/link access by ID
    const gui::MLNode* findNodeById(int id) const;
    const gui::NodeLink* findLinkById(int id) const;

    // Property access
    ResultValue getNodeProperty(const gui::MLNode& node, const std::string& prop) const;
    ResultValue getLinkProperty(const gui::NodeLink& link, const std::string& prop) const;

    // Comparison helpers
    int compareValues(const ResultValue& a, const ResultValue& b) const;
    bool valueMatchesCondition(const ResultValue& val, const std::string& op, const ResultValue& other) const;

    // Aggregation
    ResultValue aggregate(const std::string& func, const std::vector<ResultValue>& values);

    // Graph modification
    int createNode(const ASTNodePtr& nodePattern);
    int createRelationship(int fromNode, int toNode, const ASTNodePtr& relPattern);
    void deleteNode(int nodeId);
    void deleteLink(int linkId);
    void setProperty(int nodeId, const std::string& prop, const ResultValue& value);

    // State
    gui::NodeEditor& editor_;
    double lastExecutionTimeMs_ = 0.0;

    // Cached graph state (refreshed each query)
    std::vector<gui::MLNode> nodes_;
    std::vector<gui::NodeLink> links_;
};

// Convenience function to run a query
QueryResult runQuery(gui::NodeEditor& editor, const std::string& query);

} // namespace cyxwiz::query
