#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <variant>

// Undefine Windows macros that conflict with our enum values
#ifdef TYPE
#undef TYPE
#endif
#ifdef SIZE
#undef SIZE
#endif
#ifdef DELETE
#undef DELETE
#endif
#ifdef IN
#undef IN
#endif
#ifdef OUT
#undef OUT
#endif
#ifdef TRUE
#undef TRUE
#endif
#ifdef FALSE
#undef FALSE
#endif
#ifdef ERROR
#undef ERROR
#endif
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace cyxwiz::query {

// ============================================================================
// Token Types for Lexer
// ============================================================================

enum class TokenType {
    // Keywords
    MATCH,          // MATCH clause
    WHERE,          // WHERE clause
    RETURN,         // RETURN clause
    CREATE,         // CREATE clause
    DELETE,         // DELETE clause
    SET,            // SET clause
    WITH,           // WITH clause (pipeline)
    ORDER,          // ORDER BY
    BY,             // ORDER BY
    LIMIT,          // LIMIT clause
    SKIP,           // SKIP clause

    // Logical operators
    AND,            // AND
    OR,             // OR
    NOT,            // NOT
    XOR,            // XOR

    // Aggregate functions
    COUNT,          // count()
    SUM,            // sum()
    AVG,            // avg()
    MIN,            // min()
    MAX,            // max()
    COLLECT,        // collect() - aggregate into list

    // Built-in functions
    TYPE,           // type() - get node/link type
    ID,             // id() - get node/link id
    PROPERTIES,     // properties() - get all properties
    KEYS,           // keys() - get property keys
    EXISTS,         // exists() - check if property exists
    SIZE,           // size() - list/string length

    // Sort order
    ASC,            // Ascending
    DESC,           // Descending

    // Alias
    AS,             // AS for aliasing

    // NULL handling
    IS,             // IS NULL, IS NOT NULL
    NULL_LITERAL,   // NULL

    // Boolean literals
    TRUE_LITERAL,   // true
    FALSE_LITERAL,  // false

    // Symbols - Grouping
    LPAREN,         // (
    RPAREN,         // )
    LBRACKET,       // [
    RBRACKET,       // ]
    LBRACE,         // {
    RBRACE,         // }

    // Symbols - Delimiters
    COLON,          // :
    COMMA,          // ,
    DOT,            // .
    PIPE,           // |

    // Symbols - Relationships
    ARROW,          // ->
    LEFT_ARROW,     // <-
    DASH,           // -
    STAR,           // * (variable length paths)

    // Comparison operators
    EQUALS,         // =
    NOT_EQUALS,     // <> or !=
    LESS,           // <
    GREATER,        // >
    LESS_EQ,        // <=
    GREATER_EQ,     // >=
    CONTAINS,       // CONTAINS (string)
    STARTS_WITH,    // STARTS WITH
    ENDS_WITH,      // ENDS WITH
    IN,             // IN (list membership)

    // Arithmetic operators
    PLUS,           // +
    MINUS,          // -
    MULTIPLY,       // *
    DIVIDE,         // /
    MODULO,         // %
    POWER,          // ^

    // Literals
    IDENTIFIER,     // Variable/label name
    STRING,         // "string" or 'string'
    INTEGER,        // 123
    FLOAT,          // 123.45

    // Special
    END_OF_INPUT,   // End of query
    ERROR           // Lexer error
};

// Token structure
struct Token {
    TokenType token_type;
    std::string value;
    int line;
    int column;

    Token() : token_type(TokenType::ERROR), line(0), column(0) {}
    Token(TokenType tok_type, std::string tok_value, int tok_line, int tok_col)
        : token_type(tok_type), value(std::move(tok_value)), line(tok_line), column(tok_col) {}

    bool is(TokenType tok) const { return token_type == tok; }
    bool isKeyword() const;
    bool isOperator() const;
    bool isLiteral() const;

    std::string toString() const;
    static std::string typeToString(TokenType tok_type);
};

// ============================================================================
// AST Node Types for Parser
// ============================================================================

enum class ASTNodeType {
    // Root
    Query,              // Complete query

    // Clauses
    MatchClause,        // MATCH pattern
    WhereClause,        // WHERE condition
    ReturnClause,       // RETURN expressions
    CreateClause,       // CREATE pattern
    DeleteClause,       // DELETE identifiers
    SetClause,          // SET property = value
    OrderByClause,      // ORDER BY expressions
    LimitClause,        // LIMIT n
    SkipClause,         // SKIP n
    WithClause,         // WITH (pipeline)

    // Pattern elements
    NodePattern,        // (identifier:Label {props})
    RelationshipPattern,// -[identifier:TYPE {props}]->
    PathPattern,        // Node-Rel-Node chain

    // Property access
    PropertyAccess,     // node.property
    PropertyList,       // {key: value, ...}
    PropertyAssignment, // property = value

    // Expressions
    BinaryExpr,         // left op right
    UnaryExpr,          // op expr
    ComparisonExpr,     // expr comp_op expr
    LogicalExpr,        // expr AND/OR expr

    // Values
    FunctionCall,       // func(args)
    Identifier,         // variable name
    Label,              // :NodeType
    RelType,            // :REL_TYPE
    Literal,            // string/number/bool/null
    ListExpr,           // [item, item, ...]
    MapExpr,            // {key: value, ...}

    // Special
    Star,               // * for all columns or variable paths
    Range,              // *1..5 for path length
    Alias,              // expr AS alias
    SortItem            // expr ASC/DESC
};

// Forward declaration
struct ASTNode;
using ASTNodePtr = std::shared_ptr<ASTNode>;

// Value types for literals
using LiteralValue = std::variant<
    std::nullptr_t,     // NULL
    bool,               // Boolean
    int64_t,            // Integer
    double,             // Float
    std::string         // String
>;

// AST Node structure
struct ASTNode {
    ASTNodeType type;
    std::string value;                          // For identifiers, operators, etc.
    std::vector<ASTNodePtr> children;           // Child nodes
    std::map<std::string, std::string> properties;  // Node/rel properties
    LiteralValue literal;                       // For literal values

    // Source location for error reporting
    int line = 0;
    int column = 0;

    ASTNode() : type(ASTNodeType::Query) {}
    explicit ASTNode(ASTNodeType t) : type(t) {}
    ASTNode(ASTNodeType t, std::string v) : type(t), value(std::move(v)) {}

    // Factory methods for common node types
    static ASTNodePtr makeIdentifier(const std::string& name, int line = 0, int col = 0);
    static ASTNodePtr makeLiteral(const LiteralValue& val, int line = 0, int col = 0);
    static ASTNodePtr makeNodePattern(const std::string& var = "", const std::string& label = "");
    static ASTNodePtr makeRelPattern(const std::string& var = "", const std::string& type = "");
    static ASTNodePtr makeBinaryExpr(const std::string& op, ASTNodePtr left, ASTNodePtr right);
    static ASTNodePtr makeFunctionCall(const std::string& name, std::vector<ASTNodePtr> args);

    // Helpers
    void addChild(ASTNodePtr child);
    void setProperty(const std::string& key, const std::string& value);
    std::string getProperty(const std::string& key, const std::string& defaultVal = "") const;
    bool hasProperty(const std::string& key) const;

    std::string toString(int indent = 0) const;
    static std::string typeToString(ASTNodeType node_type);
};

// ============================================================================
// Query Result
// ============================================================================

// Result value types
struct ResultValue {
    enum class Type { Null, Bool, Int, Float, String, Node, Link, Path, List, Map };

    Type type = Type::Null;
    LiteralValue scalar;                        // For scalar values
    int nodeId = -1;                            // For Node type
    int linkId = -1;                            // For Link type
    std::vector<int> pathNodeIds;               // For Path type
    std::vector<int> pathLinkIds;               // For Path type
    std::vector<ResultValue> list;              // For List type
    std::map<std::string, ResultValue> map;     // For Map type

    ResultValue() = default;
    static ResultValue makeNull();
    static ResultValue makeBool(bool v);
    static ResultValue makeInt(int64_t v);
    static ResultValue makeFloat(double v);
    static ResultValue makeString(const std::string& v);
    static ResultValue makeNode(int id);
    static ResultValue makeLink(int id);
    static ResultValue makePath(std::vector<int> nodes, std::vector<int> links);
    static ResultValue makeList(std::vector<ResultValue> items);
    static ResultValue makeMap(std::map<std::string, ResultValue> items);

    std::string toString() const;
    bool toBool() const;
    int64_t toInt() const;
    double toFloat() const;
};

// Result row (one match/result)
struct ResultRow {
    std::map<std::string, ResultValue> values;  // Column name -> value

    ResultValue get(const std::string& column) const;
    bool has(const std::string& column) const;
    void set(const std::string& column, ResultValue value);
};

// Complete query result
struct QueryResult {
    bool success = true;
    std::string error;                          // Error message if !success
    int errorLine = 0;                          // Error location
    int errorColumn = 0;

    // Result data
    std::vector<std::string> columns;           // Column names
    std::vector<ResultRow> rows;                // Result rows

    // Statistics for modification queries
    int nodesCreated = 0;
    int nodesDeleted = 0;
    int nodesModified = 0;
    int linksCreated = 0;
    int linksDeleted = 0;
    int linksModified = 0;
    int propertiesSet = 0;

    // Execution info
    double executionTimeMs = 0.0;

    // Factory methods
    static QueryResult makeError(const std::string& msg, int line = 0, int col = 0);
    static QueryResult makeSuccess();

    // Helpers
    void addColumn(const std::string& name);
    void addRow(ResultRow row);
    bool isEmpty() const { return rows.empty(); }
    size_t rowCount() const { return rows.size(); }
    size_t columnCount() const { return columns.size(); }

    std::string toString() const;
    std::string toTable() const;                // Format as ASCII table
};

// ============================================================================
// Binding Context (for query execution)
// ============================================================================

// Variable binding during pattern matching
struct Binding {
    enum class Type { Node, Link, Path, Value };

    Type type = Type::Value;
    int nodeId = -1;                            // For Node binding
    int linkId = -1;                            // For Link binding
    std::vector<int> pathNodeIds;               // For Path binding
    std::vector<int> pathLinkIds;               // For Path binding
    ResultValue value;                          // For Value binding

    static Binding forNode(int id);
    static Binding forLink(int id);
    static Binding forPath(std::vector<int> nodes, std::vector<int> links);
    static Binding forValue(ResultValue val);
};

// Binding context (variable -> binding map)
class BindingContext {
public:
    void bind(const std::string& name, Binding binding);
    bool has(const std::string& name) const;
    Binding get(const std::string& name) const;
    void clear();

    // Create a copy for backtracking
    BindingContext clone() const;

    // Get all bound variable names
    std::vector<std::string> getNames() const;

private:
    std::map<std::string, Binding> bindings_;
};

} // namespace cyxwiz::query
