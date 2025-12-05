#include "cyxql_parser.h"
#include <sstream>

namespace cyxwiz::query {

Parser::Parser(const std::string& input)
    : ownedLexer_(std::make_unique<Lexer>(input))
    , lexer_(ownedLexer_.get())
{
    // Prime the parser with the first token
    advance();
}

Parser::Parser(Lexer& lexer)
    : lexer_(&lexer)
{
    advance();
}

Token Parser::advance() {
    previous_ = current_;
    current_ = lexer_->nextToken();
    return previous_;
}

bool Parser::check(TokenType type) const {
    return current_.token_type == type;
}

bool Parser::match(TokenType type) {
    if (check(type)) {
        advance();
        return true;
    }
    return false;
}

bool Parser::matchAny(std::initializer_list<TokenType> types) {
    for (TokenType type : types) {
        if (check(type)) {
            advance();
            return true;
        }
    }
    return false;
}

void Parser::consume(TokenType type, const std::string& message) {
    if (check(type)) {
        advance();
        return;
    }
    error(message);
}

bool Parser::isAtEnd() const {
    return current_.token_type == TokenType::END_OF_INPUT ||
           current_.token_type == TokenType::ERROR;
}

void Parser::error(const std::string& message) {
    if (hasError_) return;  // Only report first error

    hasError_ = true;
    errorLine_ = current_.line;
    errorColumn_ = current_.column;

    std::stringstream ss;
    ss << message << " at line " << errorLine_ << ", column " << errorColumn_;
    if (current_.token_type != TokenType::END_OF_INPUT) {
        ss << " (got '" << current_.value << "')";
    }
    errorMessage_ = ss.str();
}

void Parser::synchronize() {
    advance();

    while (!isAtEnd()) {
        // Synchronize at clause boundaries
        switch (current_.token_type) {
            case TokenType::MATCH:
            case TokenType::WHERE:
            case TokenType::RETURN:
            case TokenType::CREATE:
            case TokenType::DELETE:
            case TokenType::SET:
            case TokenType::ORDER:
            case TokenType::LIMIT:
            case TokenType::SKIP:
            case TokenType::WITH:
                return;
            default:
                break;
        }
        advance();
    }
}

ASTNodePtr Parser::parse() {
    return parseQuery();
}

ASTNodePtr Parser::parseQuery() {
    auto query = std::make_shared<ASTNode>(ASTNodeType::Query);

    while (!isAtEnd() && !hasError_) {
        auto clause = parseClause();
        if (clause) {
            query->addChild(clause);
        }
        if (hasError_) {
            synchronize();
        }
    }

    return query;
}

ASTNodePtr Parser::parseClause() {
    if (match(TokenType::MATCH)) return parseMatchClause();
    if (match(TokenType::WHERE)) return parseWhereClause();
    if (match(TokenType::RETURN)) return parseReturnClause();
    if (match(TokenType::CREATE)) return parseCreateClause();
    if (match(TokenType::DELETE)) return parseDeleteClause();
    if (match(TokenType::SET)) return parseSetClause();
    if (match(TokenType::ORDER)) return parseOrderByClause();
    if (match(TokenType::LIMIT)) return parseLimitClause();
    if (match(TokenType::SKIP)) return parseSkipClause();
    if (match(TokenType::WITH)) return parseWithClause();

    error("Expected clause (MATCH, WHERE, RETURN, CREATE, DELETE, SET, ORDER BY, LIMIT, SKIP, WITH)");
    return nullptr;
}

ASTNodePtr Parser::parseMatchClause() {
    auto match = std::make_shared<ASTNode>(ASTNodeType::MatchClause);

    // Parse first pattern
    auto pattern = parsePattern();
    if (pattern) {
        match->addChild(pattern);
    }

    // Parse additional patterns
    while (this->match(TokenType::COMMA)) {
        pattern = parsePattern();
        if (pattern) {
            match->addChild(pattern);
        }
    }

    return match;
}

ASTNodePtr Parser::parseWhereClause() {
    auto where = std::make_shared<ASTNode>(ASTNodeType::WhereClause);
    auto condition = parseOrExpression();
    where->addChild(condition);
    return where;
}

ASTNodePtr Parser::parseReturnClause() {
    auto ret = std::make_shared<ASTNode>(ASTNodeType::ReturnClause);

    // Check for RETURN *
    if (match(TokenType::STAR)) {
        auto star = std::make_shared<ASTNode>(ASTNodeType::Star);
        ret->addChild(star);
        return ret;
    }

    // Parse return items
    auto item = parseReturnItem();
    if (item) {
        ret->addChild(item);
    }

    while (match(TokenType::COMMA)) {
        item = parseReturnItem();
        if (item) {
            ret->addChild(item);
        }
    }

    return ret;
}

ASTNodePtr Parser::parseCreateClause() {
    auto create = std::make_shared<ASTNode>(ASTNodeType::CreateClause);

    auto pattern = parsePattern();
    if (pattern) {
        create->addChild(pattern);
    }

    while (match(TokenType::COMMA)) {
        pattern = parsePattern();
        if (pattern) {
            create->addChild(pattern);
        }
    }

    return create;
}

ASTNodePtr Parser::parseDeleteClause() {
    auto del = std::make_shared<ASTNode>(ASTNodeType::DeleteClause);

    // Parse identifier list
    consume(TokenType::IDENTIFIER, "Expected identifier after DELETE");
    auto id = ASTNode::makeIdentifier(previous_.value, previous_.line, previous_.column);
    del->addChild(id);

    while (match(TokenType::COMMA)) {
        consume(TokenType::IDENTIFIER, "Expected identifier");
        id = ASTNode::makeIdentifier(previous_.value, previous_.line, previous_.column);
        del->addChild(id);
    }

    return del;
}

ASTNodePtr Parser::parseSetClause() {
    auto set = std::make_shared<ASTNode>(ASTNodeType::SetClause);

    auto item = parseSetItem();
    if (item) {
        set->addChild(item);
    }

    while (match(TokenType::COMMA)) {
        item = parseSetItem();
        if (item) {
            set->addChild(item);
        }
    }

    return set;
}

ASTNodePtr Parser::parseOrderByClause() {
    consume(TokenType::BY, "Expected BY after ORDER");

    auto orderBy = std::make_shared<ASTNode>(ASTNodeType::OrderByClause);

    auto item = parseSortItem();
    if (item) {
        orderBy->addChild(item);
    }

    while (match(TokenType::COMMA)) {
        item = parseSortItem();
        if (item) {
            orderBy->addChild(item);
        }
    }

    return orderBy;
}

ASTNodePtr Parser::parseLimitClause() {
    auto limit = std::make_shared<ASTNode>(ASTNodeType::LimitClause);

    consume(TokenType::INTEGER, "Expected integer after LIMIT");
    auto val = ASTNode::makeLiteral(std::stoll(previous_.value), previous_.line, previous_.column);
    limit->addChild(val);

    return limit;
}

ASTNodePtr Parser::parseSkipClause() {
    auto skip = std::make_shared<ASTNode>(ASTNodeType::SkipClause);

    consume(TokenType::INTEGER, "Expected integer after SKIP");
    auto val = ASTNode::makeLiteral(std::stoll(previous_.value), previous_.line, previous_.column);
    skip->addChild(val);

    return skip;
}

ASTNodePtr Parser::parseWithClause() {
    auto with = std::make_shared<ASTNode>(ASTNodeType::WithClause);

    auto item = parseReturnItem();
    if (item) {
        with->addChild(item);
    }

    while (match(TokenType::COMMA)) {
        item = parseReturnItem();
        if (item) {
            with->addChild(item);
        }
    }

    return with;
}

ASTNodePtr Parser::parsePattern() {
    return parsePathPattern();
}

ASTNodePtr Parser::parsePathPattern() {
    auto path = std::make_shared<ASTNode>(ASTNodeType::PathPattern);

    // First node
    auto node = parseNodePattern();
    if (!node) {
        error("Expected node pattern");
        return nullptr;
    }
    path->addChild(node);

    // Relationship-node pairs
    while (check(TokenType::DASH) || check(TokenType::LEFT_ARROW)) {
        auto rel = parseRelationshipPattern();
        if (rel) {
            path->addChild(rel);
        }

        node = parseNodePattern();
        if (node) {
            path->addChild(node);
        }
    }

    return path;
}

ASTNodePtr Parser::parseNodePattern() {
    consume(TokenType::LPAREN, "Expected '(' to start node pattern");

    auto node = std::make_shared<ASTNode>(ASTNodeType::NodePattern);

    // Optional identifier
    if (match(TokenType::IDENTIFIER)) {
        auto id = ASTNode::makeIdentifier(previous_.value, previous_.line, previous_.column);
        node->addChild(id);
    }

    // Optional label(s)
    while (match(TokenType::COLON)) {
        consume(TokenType::IDENTIFIER, "Expected label after ':'");
        auto label = std::make_shared<ASTNode>(ASTNodeType::Label, previous_.value);
        node->addChild(label);
    }

    // Optional properties
    if (check(TokenType::LBRACE)) {
        auto props = parsePropertyList();
        if (props) {
            node->addChild(props);
        }
    }

    consume(TokenType::RPAREN, "Expected ')' to close node pattern");
    return node;
}

ASTNodePtr Parser::parseRelationshipPattern() {
    auto rel = std::make_shared<ASTNode>(ASTNodeType::RelationshipPattern);

    // Direction: left arrow or dash
    bool leftArrow = match(TokenType::LEFT_ARROW);
    if (!leftArrow) {
        consume(TokenType::DASH, "Expected '-' or '<-' for relationship");
    }

    // Check for relationship details in brackets
    if (match(TokenType::LBRACKET)) {
        // Optional identifier
        if (match(TokenType::IDENTIFIER)) {
            auto id = ASTNode::makeIdentifier(previous_.value, previous_.line, previous_.column);
            rel->addChild(id);
        }

        // Optional type
        if (match(TokenType::COLON)) {
            consume(TokenType::IDENTIFIER, "Expected relationship type after ':'");
            auto type = std::make_shared<ASTNode>(ASTNodeType::RelType, previous_.value);
            rel->addChild(type);
        }

        // Optional range for variable-length paths
        if (match(TokenType::STAR)) {
            auto range = parseRange();
            if (range) {
                rel->addChild(range);
            }
        }

        // Optional properties
        if (check(TokenType::LBRACE)) {
            auto props = parsePropertyList();
            if (props) {
                rel->addChild(props);
            }
        }

        consume(TokenType::RBRACKET, "Expected ']' to close relationship");
    }

    // Direction: dash or right arrow
    bool rightArrow = match(TokenType::ARROW);
    if (!rightArrow) {
        consume(TokenType::DASH, "Expected '-' or '->' for relationship");
    }

    // Set direction in properties
    if (leftArrow) {
        rel->setProperty("direction", "left");
    } else if (rightArrow) {
        rel->setProperty("direction", "right");
    } else {
        rel->setProperty("direction", "both");
    }

    return rel;
}

ASTNodePtr Parser::parseRange() {
    auto range = std::make_shared<ASTNode>(ASTNodeType::Range);

    // Parse optional min
    if (match(TokenType::INTEGER)) {
        range->setProperty("min", previous_.value);
    }

    // Parse optional ..
    if (match(TokenType::DOT)) {
        consume(TokenType::DOT, "Expected '..' in range");

        // Parse optional max
        if (match(TokenType::INTEGER)) {
            range->setProperty("max", previous_.value);
        }
    } else if (range->hasProperty("min")) {
        // Just a single number means exact length
        range->setProperty("max", range->getProperty("min"));
    }

    return range;
}

ASTNodePtr Parser::parsePropertyList() {
    consume(TokenType::LBRACE, "Expected '{' for properties");

    auto props = std::make_shared<ASTNode>(ASTNodeType::PropertyList);

    if (!check(TokenType::RBRACE)) {
        // First property
        consume(TokenType::IDENTIFIER, "Expected property name");
        std::string key = previous_.value;
        consume(TokenType::COLON, "Expected ':' after property name");
        auto value = parseExpression();

        auto prop = std::make_shared<ASTNode>(ASTNodeType::PropertyAssignment, key);
        prop->addChild(value);
        props->addChild(prop);

        // Additional properties
        while (match(TokenType::COMMA)) {
            consume(TokenType::IDENTIFIER, "Expected property name");
            key = previous_.value;
            consume(TokenType::COLON, "Expected ':' after property name");
            value = parseExpression();

            prop = std::make_shared<ASTNode>(ASTNodeType::PropertyAssignment, key);
            prop->addChild(value);
            props->addChild(prop);
        }
    }

    consume(TokenType::RBRACE, "Expected '}' to close properties");
    return props;
}

ASTNodePtr Parser::parseReturnItem() {
    auto expr = parseExpression();

    if (match(TokenType::AS)) {
        consume(TokenType::IDENTIFIER, "Expected alias after AS");
        auto alias = std::make_shared<ASTNode>(ASTNodeType::Alias, previous_.value);
        alias->addChild(expr);
        return alias;
    }

    return expr;
}

ASTNodePtr Parser::parseSortItem() {
    auto expr = parseExpression();

    auto sort = std::make_shared<ASTNode>(ASTNodeType::SortItem);
    sort->addChild(expr);

    if (match(TokenType::ASC)) {
        sort->setProperty("order", "asc");
    } else if (match(TokenType::DESC)) {
        sort->setProperty("order", "desc");
    } else {
        sort->setProperty("order", "asc");  // Default
    }

    return sort;
}

ASTNodePtr Parser::parseSetItem() {
    // Parse property access (e.g., n.name)
    consume(TokenType::IDENTIFIER, "Expected identifier");
    auto base = ASTNode::makeIdentifier(previous_.value, previous_.line, previous_.column);

    consume(TokenType::DOT, "Expected '.' for property access");
    consume(TokenType::IDENTIFIER, "Expected property name");
    std::string propName = previous_.value;

    consume(TokenType::EQUALS, "Expected '=' for SET");
    auto value = parseExpression();

    auto propAccess = std::make_shared<ASTNode>(ASTNodeType::PropertyAccess, propName);
    propAccess->addChild(base);

    auto assignment = std::make_shared<ASTNode>(ASTNodeType::PropertyAssignment);
    assignment->addChild(propAccess);
    assignment->addChild(value);

    return assignment;
}

ASTNodePtr Parser::parseExpression() {
    return parseOrExpression();
}

ASTNodePtr Parser::parseOrExpression() {
    auto left = parseAndExpression();

    while (match(TokenType::OR)) {
        auto right = parseAndExpression();
        left = ASTNode::makeBinaryExpr("OR", left, right);
    }

    return left;
}

ASTNodePtr Parser::parseAndExpression() {
    auto left = parseNotExpression();

    while (match(TokenType::AND)) {
        auto right = parseNotExpression();
        left = ASTNode::makeBinaryExpr("AND", left, right);
    }

    return left;
}

ASTNodePtr Parser::parseNotExpression() {
    if (match(TokenType::NOT)) {
        auto expr = parseNotExpression();
        auto notExpr = std::make_shared<ASTNode>(ASTNodeType::UnaryExpr, "NOT");
        notExpr->addChild(expr);
        return notExpr;
    }

    return parseComparison();
}

ASTNodePtr Parser::parseComparison() {
    auto left = parseAddition();

    // IS NULL / IS NOT NULL
    if (match(TokenType::IS)) {
        bool negated = match(TokenType::NOT);
        consume(TokenType::NULL_LITERAL, "Expected NULL after IS/IS NOT");

        std::string op = negated ? "IS NOT NULL" : "IS NULL";
        auto comp = std::make_shared<ASTNode>(ASTNodeType::ComparisonExpr, op);
        comp->addChild(left);
        return comp;
    }

    // Comparison operators
    if (matchAny({TokenType::EQUALS, TokenType::NOT_EQUALS,
                  TokenType::LESS, TokenType::GREATER,
                  TokenType::LESS_EQ, TokenType::GREATER_EQ})) {
        std::string op = previous_.value;
        auto right = parseAddition();
        return ASTNode::makeBinaryExpr(op, left, right);
    }

    // String operators
    if (matchAny({TokenType::CONTAINS, TokenType::STARTS_WITH, TokenType::ENDS_WITH})) {
        std::string op = previous_.value;
        auto right = parseAddition();
        return ASTNode::makeBinaryExpr(op, left, right);
    }

    // IN operator
    if (match(TokenType::IN)) {
        auto right = parseAddition();
        return ASTNode::makeBinaryExpr("IN", left, right);
    }

    return left;
}

ASTNodePtr Parser::parseAddition() {
    auto left = parseMultiplication();

    while (matchAny({TokenType::PLUS, TokenType::MINUS})) {
        std::string op = previous_.value;
        auto right = parseMultiplication();
        left = ASTNode::makeBinaryExpr(op, left, right);
    }

    return left;
}

ASTNodePtr Parser::parseMultiplication() {
    auto left = parseUnary();

    while (matchAny({TokenType::STAR, TokenType::DIVIDE, TokenType::MODULO})) {
        std::string op = previous_.value;
        auto right = parseUnary();
        left = ASTNode::makeBinaryExpr(op, left, right);
    }

    return left;
}

ASTNodePtr Parser::parseUnary() {
    if (match(TokenType::MINUS)) {
        auto expr = parseUnary();
        auto neg = std::make_shared<ASTNode>(ASTNodeType::UnaryExpr, "-");
        neg->addChild(expr);
        return neg;
    }

    return parsePrimary();
}

ASTNodePtr Parser::parsePrimary() {
    // Literals
    if (match(TokenType::INTEGER)) {
        return ASTNode::makeLiteral(std::stoll(previous_.value), previous_.line, previous_.column);
    }

    if (match(TokenType::FLOAT)) {
        return ASTNode::makeLiteral(std::stod(previous_.value), previous_.line, previous_.column);
    }

    if (match(TokenType::STRING)) {
        return ASTNode::makeLiteral(previous_.value, previous_.line, previous_.column);
    }

    if (match(TokenType::TRUE_LITERAL)) {
        return ASTNode::makeLiteral(true, previous_.line, previous_.column);
    }

    if (match(TokenType::FALSE_LITERAL)) {
        return ASTNode::makeLiteral(false, previous_.line, previous_.column);
    }

    if (match(TokenType::NULL_LITERAL)) {
        return ASTNode::makeLiteral(nullptr, previous_.line, previous_.column);
    }

    // Star (for RETURN *)
    if (match(TokenType::STAR)) {
        return std::make_shared<ASTNode>(ASTNodeType::Star);
    }

    // List expression
    if (check(TokenType::LBRACKET)) {
        return parseListExpression();
    }

    // Map expression
    if (check(TokenType::LBRACE)) {
        return parseMapExpression();
    }

    // Parenthesized expression
    if (match(TokenType::LPAREN)) {
        auto expr = parseExpression();
        consume(TokenType::RPAREN, "Expected ')' after expression");
        return expr;
    }

    // Function calls and identifiers
    if (matchAny({TokenType::IDENTIFIER,
                  TokenType::COUNT, TokenType::SUM, TokenType::AVG,
                  TokenType::MIN, TokenType::MAX, TokenType::COLLECT,
                  TokenType::TYPE, TokenType::ID, TokenType::PROPERTIES,
                  TokenType::KEYS, TokenType::EXISTS, TokenType::SIZE})) {
        std::string name = previous_.value;
        int line = previous_.line;
        int col = previous_.column;

        // Check for function call
        if (check(TokenType::LPAREN)) {
            return parseFunctionCall(name);
        }

        // Check for property access
        auto id = ASTNode::makeIdentifier(name, line, col);
        if (check(TokenType::DOT)) {
            return parsePropertyAccess(id);
        }

        return id;
    }

    error("Expected expression");
    return nullptr;
}

ASTNodePtr Parser::parsePropertyAccess(ASTNodePtr base) {
    while (match(TokenType::DOT)) {
        consume(TokenType::IDENTIFIER, "Expected property name after '.'");
        auto access = std::make_shared<ASTNode>(ASTNodeType::PropertyAccess, previous_.value);
        access->addChild(base);
        base = access;
    }
    return base;
}

ASTNodePtr Parser::parseFunctionCall(const std::string& name) {
    consume(TokenType::LPAREN, "Expected '(' for function call");

    std::vector<ASTNodePtr> args;

    if (!check(TokenType::RPAREN)) {
        // Check for DISTINCT modifier (for COUNT(DISTINCT x))
        bool distinct = match(TokenType::IDENTIFIER) && previous_.value == "DISTINCT";
        if (!distinct && previous_.token_type == TokenType::IDENTIFIER) {
            // It was a regular argument, need to back up
            // Since we can't back up, we'll treat it as the first argument
            auto arg = ASTNode::makeIdentifier(previous_.value, previous_.line, previous_.column);
            if (check(TokenType::DOT)) {
                arg = parsePropertyAccess(arg);
            }
            args.push_back(arg);
        }

        // Parse remaining arguments
        while (!check(TokenType::RPAREN) && !isAtEnd()) {
            if (!args.empty() && !distinct) {
                if (!match(TokenType::COMMA)) {
                    break;
                }
            }
            args.push_back(parseExpression());
            distinct = false;  // DISTINCT only applies to first
        }
    }

    consume(TokenType::RPAREN, "Expected ')' after function arguments");

    return ASTNode::makeFunctionCall(name, args);
}

ASTNodePtr Parser::parseListExpression() {
    consume(TokenType::LBRACKET, "Expected '[' for list");

    auto list = std::make_shared<ASTNode>(ASTNodeType::ListExpr);

    if (!check(TokenType::RBRACKET)) {
        list->addChild(parseExpression());

        while (match(TokenType::COMMA)) {
            list->addChild(parseExpression());
        }
    }

    consume(TokenType::RBRACKET, "Expected ']' to close list");
    return list;
}

ASTNodePtr Parser::parseMapExpression() {
    consume(TokenType::LBRACE, "Expected '{' for map");

    auto map = std::make_shared<ASTNode>(ASTNodeType::MapExpr);

    if (!check(TokenType::RBRACE)) {
        consume(TokenType::IDENTIFIER, "Expected key");
        std::string key = previous_.value;
        consume(TokenType::COLON, "Expected ':' after key");
        auto value = parseExpression();

        auto entry = std::make_shared<ASTNode>(ASTNodeType::PropertyAssignment, key);
        entry->addChild(value);
        map->addChild(entry);

        while (match(TokenType::COMMA)) {
            consume(TokenType::IDENTIFIER, "Expected key");
            key = previous_.value;
            consume(TokenType::COLON, "Expected ':' after key");
            value = parseExpression();

            entry = std::make_shared<ASTNode>(ASTNodeType::PropertyAssignment, key);
            entry->addChild(value);
            map->addChild(entry);
        }
    }

    consume(TokenType::RBRACE, "Expected '}' to close map");
    return map;
}

} // namespace cyxwiz::query
