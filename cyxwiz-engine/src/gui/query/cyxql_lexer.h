#pragma once

#include "cyxql_types.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace cyxwiz::query {

/**
 * CyxQL Lexer - Tokenizes CyxQL query strings
 *
 * Supports:
 * - Keywords: MATCH, WHERE, RETURN, CREATE, DELETE, SET, etc.
 * - Operators: =, <>, <, >, <=, >=, +, -, *, /, etc.
 * - Symbols: (, ), [, ], {, }, :, ,, ., ->, <-, etc.
 * - Literals: strings ("..." or '...'), integers, floats, booleans
 * - Identifiers: variable and label names
 * - Comments: // single-line and /* multi-line
 */
class Lexer {
public:
    explicit Lexer(const std::string& input);

    // Get next token
    Token nextToken();

    // Peek at next token without consuming
    Token peekToken();

    // Get all tokens (for debugging)
    std::vector<Token> tokenize();

    // Check if at end of input
    bool isAtEnd() const { return pos_ >= input_.length(); }

    // Get current position
    int getLine() const { return line_; }
    int getColumn() const { return column_; }

    // Error info
    bool hasError() const { return hasError_; }
    std::string getError() const { return errorMessage_; }

private:
    // Core lexing methods
    Token scanToken();
    Token makeToken(TokenType type);
    Token makeToken(TokenType type, const std::string& value);
    Token errorToken(const std::string& message);

    // Character helpers
    char peek() const;
    char peekNext() const;
    char advance();
    bool match(char expected);
    void skipWhitespace();
    void skipComment();

    // Token scanners
    Token scanString(char quote);
    Token scanNumber();
    Token scanIdentifier();

    // Keyword lookup
    TokenType identifierType(const std::string& text) const;

    // State
    std::string input_;
    size_t pos_ = 0;
    size_t tokenStart_ = 0;
    int line_ = 1;
    int column_ = 1;
    int tokenLine_ = 1;
    int tokenColumn_ = 1;

    // Peek buffer
    bool hasPeeked_ = false;
    Token peekedToken_;

    // Error state
    bool hasError_ = false;
    std::string errorMessage_;

    // Keyword map (initialized in constructor)
    static const std::unordered_map<std::string, TokenType> keywords_;
};

} // namespace cyxwiz::query
