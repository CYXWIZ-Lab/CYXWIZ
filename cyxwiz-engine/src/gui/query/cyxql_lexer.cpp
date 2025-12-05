#include "cyxql_lexer.h"
#include <cctype>
#include <algorithm>

namespace cyxwiz::query {

// Static keyword map
const std::unordered_map<std::string, TokenType> Lexer::keywords_ = {
    // Main clauses
    {"MATCH", TokenType::MATCH},
    {"WHERE", TokenType::WHERE},
    {"RETURN", TokenType::RETURN},
    {"CREATE", TokenType::CREATE},
    {"DELETE", TokenType::DELETE},
    {"SET", TokenType::SET},
    {"WITH", TokenType::WITH},
    {"ORDER", TokenType::ORDER},
    {"BY", TokenType::BY},
    {"LIMIT", TokenType::LIMIT},
    {"SKIP", TokenType::SKIP},

    // Logical
    {"AND", TokenType::AND},
    {"OR", TokenType::OR},
    {"NOT", TokenType::NOT},
    {"XOR", TokenType::XOR},

    // Aggregate functions
    {"COUNT", TokenType::COUNT},
    {"SUM", TokenType::SUM},
    {"AVG", TokenType::AVG},
    {"MIN", TokenType::MIN},
    {"MAX", TokenType::MAX},
    {"COLLECT", TokenType::COLLECT},

    // Built-in functions
    {"TYPE", TokenType::TYPE},
    {"ID", TokenType::ID},
    {"PROPERTIES", TokenType::PROPERTIES},
    {"KEYS", TokenType::KEYS},
    {"EXISTS", TokenType::EXISTS},
    {"SIZE", TokenType::SIZE},

    // Sort order
    {"ASC", TokenType::ASC},
    {"DESC", TokenType::DESC},
    {"ASCENDING", TokenType::ASC},
    {"DESCENDING", TokenType::DESC},

    // Alias
    {"AS", TokenType::AS},

    // NULL handling
    {"IS", TokenType::IS},
    {"NULL", TokenType::NULL_LITERAL},

    // Boolean
    {"TRUE", TokenType::TRUE_LITERAL},
    {"FALSE", TokenType::FALSE_LITERAL},

    // String operators
    {"CONTAINS", TokenType::CONTAINS},
    {"STARTS", TokenType::STARTS_WITH},  // Will combine with WITH
    {"ENDS", TokenType::ENDS_WITH},      // Will combine with WITH
    {"IN", TokenType::IN},
};

Lexer::Lexer(const std::string& input) : input_(input) {}

Token Lexer::nextToken() {
    if (hasPeeked_) {
        hasPeeked_ = false;
        return peekedToken_;
    }
    return scanToken();
}

Token Lexer::peekToken() {
    if (!hasPeeked_) {
        peekedToken_ = scanToken();
        hasPeeked_ = true;
    }
    return peekedToken_;
}

std::vector<Token> Lexer::tokenize() {
    std::vector<Token> tokens;
    while (!isAtEnd()) {
        Token tok = nextToken();
        tokens.push_back(tok);
        if (tok.token_type == TokenType::END_OF_INPUT || tok.token_type == TokenType::ERROR) {
            break;
        }
    }
    return tokens;
}

Token Lexer::scanToken() {
    skipWhitespace();

    tokenStart_ = pos_;
    tokenLine_ = line_;
    tokenColumn_ = column_;

    if (isAtEnd()) {
        return makeToken(TokenType::END_OF_INPUT);
    }

    char c = advance();

    // Single character tokens
    switch (c) {
        case '(': return makeToken(TokenType::LPAREN);
        case ')': return makeToken(TokenType::RPAREN);
        case '[': return makeToken(TokenType::LBRACKET);
        case ']': return makeToken(TokenType::RBRACKET);
        case '{': return makeToken(TokenType::LBRACE);
        case '}': return makeToken(TokenType::RBRACE);
        case ':': return makeToken(TokenType::COLON);
        case ',': return makeToken(TokenType::COMMA);
        case '.': return makeToken(TokenType::DOT);
        case '|': return makeToken(TokenType::PIPE);
        case '+': return makeToken(TokenType::PLUS);
        case '%': return makeToken(TokenType::MODULO);
        case '^': return makeToken(TokenType::POWER);
        case '*': return makeToken(TokenType::STAR);

        // Two-character tokens
        case '-':
            if (match('>')) return makeToken(TokenType::ARROW);
            if (match('-')) {
                // Comment: skip to end of line
                while (!isAtEnd() && peek() != '\n') advance();
                return scanToken();  // Get next real token
            }
            // Check if it's a negative number
            if (std::isdigit(peek())) {
                return scanNumber();
            }
            return makeToken(TokenType::DASH);

        case '<':
            if (match('-')) return makeToken(TokenType::LEFT_ARROW);
            if (match('=')) return makeToken(TokenType::LESS_EQ);
            if (match('>')) return makeToken(TokenType::NOT_EQUALS);
            return makeToken(TokenType::LESS);

        case '>':
            if (match('=')) return makeToken(TokenType::GREATER_EQ);
            return makeToken(TokenType::GREATER);

        case '=':
            return makeToken(TokenType::EQUALS);

        case '!':
            if (match('=')) return makeToken(TokenType::NOT_EQUALS);
            return errorToken("Unexpected character '!'");

        case '/':
            if (match('/')) {
                // Single-line comment
                while (!isAtEnd() && peek() != '\n') advance();
                return scanToken();
            }
            if (match('*')) {
                // Multi-line comment
                skipComment();
                return scanToken();
            }
            return makeToken(TokenType::DIVIDE);

        // String literals
        case '"':
        case '\'':
            return scanString(c);
    }

    // Number
    if (std::isdigit(c)) {
        // Back up since we already consumed the digit
        pos_--;
        column_--;
        return scanNumber();
    }

    // Identifier or keyword
    if (std::isalpha(c) || c == '_') {
        // Back up
        pos_--;
        column_--;
        return scanIdentifier();
    }

    return errorToken("Unexpected character '" + std::string(1, c) + "'");
}

Token Lexer::makeToken(TokenType type) {
    std::string value = input_.substr(tokenStart_, pos_ - tokenStart_);
    return Token(type, value, tokenLine_, tokenColumn_);
}

Token Lexer::makeToken(TokenType type, const std::string& value) {
    return Token(type, value, tokenLine_, tokenColumn_);
}

Token Lexer::errorToken(const std::string& message) {
    hasError_ = true;
    errorMessage_ = message;
    return Token(TokenType::ERROR, message, tokenLine_, tokenColumn_);
}

char Lexer::peek() const {
    if (isAtEnd()) return '\0';
    return input_[pos_];
}

char Lexer::peekNext() const {
    if (pos_ + 1 >= input_.length()) return '\0';
    return input_[pos_ + 1];
}

char Lexer::advance() {
    char c = input_[pos_++];
    if (c == '\n') {
        line_++;
        column_ = 1;
    } else {
        column_++;
    }
    return c;
}

bool Lexer::match(char expected) {
    if (isAtEnd()) return false;
    if (input_[pos_] != expected) return false;
    advance();
    return true;
}

void Lexer::skipWhitespace() {
    while (!isAtEnd()) {
        char c = peek();
        switch (c) {
            case ' ':
            case '\t':
            case '\r':
            case '\n':
                advance();
                break;
            default:
                return;
        }
    }
}

void Lexer::skipComment() {
    // Skip /* ... */ comment
    int depth = 1;
    while (!isAtEnd() && depth > 0) {
        if (peek() == '*' && peekNext() == '/') {
            advance();
            advance();
            depth--;
        } else if (peek() == '/' && peekNext() == '*') {
            advance();
            advance();
            depth++;
        } else {
            advance();
        }
    }
}

Token Lexer::scanString(char quote) {
    std::string value;

    while (!isAtEnd() && peek() != quote) {
        if (peek() == '\n') {
            return errorToken("Unterminated string literal");
        }

        // Handle escape sequences
        if (peek() == '\\') {
            advance();
            if (isAtEnd()) {
                return errorToken("Unterminated escape sequence");
            }

            char escaped = advance();
            switch (escaped) {
                case 'n': value += '\n'; break;
                case 't': value += '\t'; break;
                case 'r': value += '\r'; break;
                case '\\': value += '\\'; break;
                case '"': value += '"'; break;
                case '\'': value += '\''; break;
                default:
                    return errorToken("Invalid escape sequence '\\" +
                                     std::string(1, escaped) + "'");
            }
        } else {
            value += advance();
        }
    }

    if (isAtEnd()) {
        return errorToken("Unterminated string literal");
    }

    // Consume closing quote
    advance();

    return makeToken(TokenType::STRING, value);
}

Token Lexer::scanNumber() {
    bool isFloat = false;
    bool hasDigit = false;
    size_t start = pos_;

    // Handle optional leading minus
    if (peek() == '-') {
        advance();
    }

    // Integer part
    while (!isAtEnd() && std::isdigit(peek())) {
        advance();
        hasDigit = true;
    }

    // Decimal part
    if (peek() == '.' && std::isdigit(peekNext())) {
        isFloat = true;
        advance();  // Consume '.'
        while (!isAtEnd() && std::isdigit(peek())) {
            advance();
        }
    }

    // Exponent part
    if (peek() == 'e' || peek() == 'E') {
        isFloat = true;
        advance();
        if (peek() == '+' || peek() == '-') {
            advance();
        }
        if (!std::isdigit(peek())) {
            return errorToken("Invalid number literal: expected digit after exponent");
        }
        while (!isAtEnd() && std::isdigit(peek())) {
            advance();
        }
    }

    if (!hasDigit) {
        return errorToken("Invalid number literal");
    }

    std::string value = input_.substr(start, pos_ - start);
    return makeToken(isFloat ? TokenType::FLOAT : TokenType::INTEGER, value);
}

Token Lexer::scanIdentifier() {
    size_t start = pos_;

    // First character: letter or underscore
    if (std::isalpha(peek()) || peek() == '_') {
        advance();
    }

    // Subsequent: letter, digit, or underscore
    while (!isAtEnd() && (std::isalnum(peek()) || peek() == '_')) {
        advance();
    }

    std::string text = input_.substr(start, pos_ - start);
    TokenType type = identifierType(text);

    // Special handling for compound keywords like "STARTS WITH" and "ENDS WITH"
    if (type == TokenType::STARTS_WITH || type == TokenType::ENDS_WITH) {
        // We got STARTS or ENDS, need to check for WITH
        skipWhitespace();
        size_t withStart = pos_;

        // Try to read "WITH"
        if (std::toupper(peek()) == 'W') {
            while (!isAtEnd() && std::isalpha(peek())) {
                advance();
            }
            std::string next = input_.substr(withStart, pos_ - withStart);
            std::string nextUpper = next;
            std::transform(nextUpper.begin(), nextUpper.end(), nextUpper.begin(), ::toupper);

            if (nextUpper == "WITH") {
                // It's "STARTS WITH" or "ENDS WITH"
                return makeToken(type, text + " " + next);
            } else {
                // Not WITH, back up
                pos_ = withStart;
            }
        }
    }

    return makeToken(type, text);
}

TokenType Lexer::identifierType(const std::string& text) const {
    // Convert to uppercase for keyword matching
    std::string upper = text;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);

    auto it = keywords_.find(upper);
    if (it != keywords_.end()) {
        return it->second;
    }

    return TokenType::IDENTIFIER;
}

} // namespace cyxwiz::query
