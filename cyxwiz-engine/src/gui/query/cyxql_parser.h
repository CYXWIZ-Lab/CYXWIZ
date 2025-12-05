#pragma once

#include "cyxql_types.h"
#include "cyxql_lexer.h"
#include <string>
#include <memory>
#include <functional>

namespace cyxwiz::query {

/**
 * CyxQL Parser - Builds AST from tokens
 *
 * Grammar (simplified):
 *
 * query          := clause+
 * clause         := match_clause | where_clause | return_clause |
 *                   create_clause | delete_clause | set_clause |
 *                   order_by_clause | limit_clause | skip_clause | with_clause
 *
 * match_clause   := MATCH pattern (',' pattern)*
 * pattern        := path_pattern
 * path_pattern   := node_pattern (rel_pattern node_pattern)*
 * node_pattern   := '(' identifier? (':' label)? properties? ')'
 * rel_pattern    := ('-' | '<-') '[' identifier? (':' type)? range? properties? ']' ('->' | '-')
 * range          := '*' (int? '..' int?)?
 * properties     := '{' (property (',' property)*)? '}'
 * property       := identifier ':' expression
 *
 * where_clause   := WHERE condition
 * condition      := or_expr
 * or_expr        := and_expr (OR and_expr)*
 * and_expr       := not_expr (AND not_expr)*
 * not_expr       := NOT? comparison
 * comparison     := expression ((= | <> | < | > | <= | >=) expression)?
 *
 * return_clause  := RETURN return_item (',' return_item)*
 * return_item    := expression (AS identifier)?
 *
 * expression     := term ((+ | -) term)*
 * term           := factor ((* | / | %) factor)*
 * factor         := unary | primary
 * unary          := (- | NOT) unary | primary
 * primary        := literal | identifier | property_access | function_call |
 *                   list_expr | map_expr | '(' expression ')' | '*'
 * property_access := identifier '.' identifier
 * function_call  := identifier '(' (expression (',' expression)*)? ')'
 *
 * create_clause  := CREATE pattern (',' pattern)*
 * delete_clause  := DELETE identifier (',' identifier)*
 * set_clause     := SET set_item (',' set_item)*
 * set_item       := property_access '=' expression
 *
 * order_by_clause := ORDER BY sort_item (',' sort_item)*
 * sort_item      := expression (ASC | DESC)?
 *
 * limit_clause   := LIMIT integer
 * skip_clause    := SKIP integer
 * with_clause    := WITH return_item (',' return_item)*
 */
class Parser {
public:
    explicit Parser(const std::string& input);
    explicit Parser(Lexer& lexer);

    // Parse the input and return the AST root
    ASTNodePtr parse();

    // Error handling
    bool hasError() const { return hasError_; }
    std::string getError() const { return errorMessage_; }
    int getErrorLine() const { return errorLine_; }
    int getErrorColumn() const { return errorColumn_; }

private:
    // Token management
    Token current() const { return current_; }
    Token previous() const { return previous_; }
    Token advance();
    bool check(TokenType type) const;
    bool match(TokenType type);
    bool matchAny(std::initializer_list<TokenType> types);
    void consume(TokenType type, const std::string& message);
    bool isAtEnd() const;

    // Error handling
    void error(const std::string& message);
    void synchronize();  // Panic mode recovery

    // Query structure
    ASTNodePtr parseQuery();
    ASTNodePtr parseClause();
    ASTNodePtr parseMatchClause();
    ASTNodePtr parseWhereClause();
    ASTNodePtr parseReturnClause();
    ASTNodePtr parseCreateClause();
    ASTNodePtr parseDeleteClause();
    ASTNodePtr parseSetClause();
    ASTNodePtr parseOrderByClause();
    ASTNodePtr parseLimitClause();
    ASTNodePtr parseSkipClause();
    ASTNodePtr parseWithClause();

    // Pattern parsing
    ASTNodePtr parsePattern();
    ASTNodePtr parsePathPattern();
    ASTNodePtr parseNodePattern();
    ASTNodePtr parseRelationshipPattern();
    ASTNodePtr parseRange();
    ASTNodePtr parsePropertyList();

    // Expression parsing (precedence climbing)
    ASTNodePtr parseExpression();
    ASTNodePtr parseOrExpression();
    ASTNodePtr parseAndExpression();
    ASTNodePtr parseNotExpression();
    ASTNodePtr parseComparison();
    ASTNodePtr parseAddition();
    ASTNodePtr parseMultiplication();
    ASTNodePtr parseUnary();
    ASTNodePtr parsePrimary();
    ASTNodePtr parsePropertyAccess(ASTNodePtr base);
    ASTNodePtr parseFunctionCall(const std::string& name);
    ASTNodePtr parseListExpression();
    ASTNodePtr parseMapExpression();

    // Return items
    ASTNodePtr parseReturnItem();
    ASTNodePtr parseSortItem();
    ASTNodePtr parseSetItem();

    // State
    std::unique_ptr<Lexer> ownedLexer_;
    Lexer* lexer_;
    Token current_;
    Token previous_;

    bool hasError_ = false;
    std::string errorMessage_;
    int errorLine_ = 0;
    int errorColumn_ = 0;
};

} // namespace cyxwiz::query
