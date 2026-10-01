// Python tokenizer for the Script Editor colouring (TOFIX133 P0 item 4):
// both quote kinds, triple quotes, prefixes, escapes, numbers, decorators.
#include "../src/core/python_tokenizer.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace cyxwiz::pytokens;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::vector<std::pair<Kind, std::string>> Tokens(const std::string& line) {
    std::vector<std::pair<Kind, std::string>> out;
    const char* p = line.data();
    const char* end = p + line.size();
    while (p < end) {
        const char* b = nullptr;
        const char* e = nullptr;
        Kind k;
        if (Next(p, end, b, e, k)) {
            out.emplace_back(k, std::string(b, e));
            p = e;
        } else {
            ++p;
        }
    }
    return out;
}

bool Has(const std::vector<std::pair<Kind, std::string>>& tokens, Kind kind, const std::string& text) {
    for (const auto& [k, t] : tokens)
        if (k == kind && t == text) return true;
    return false;
}
}  // namespace

int main() {
    auto t = Tokens("name = 'it''s'");
    Check(Has(t, Kind::String, "'it'") && Has(t, Kind::String, "'s'"), "single-quoted strings");

    t = Tokens(R"(x = "a \" b" + 'c \' d')");
    Check(Has(t, Kind::String, R"("a \" b")") && Has(t, Kind::String, R"('c \' d')"), "escaped quotes stay inside");

    t = Tokens("doc = '''one line ''' + \"\"\"two\"\"\"");
    Check(Has(t, Kind::String, "'''one line '''") && Has(t, Kind::String, "\"\"\"two\"\"\""), "triple quotes on one line");

    t = Tokens("s = '''starts here");
    Check(Has(t, Kind::String, "'''starts here"), "unclosed triple quote runs to the end of the line");

    t = Tokens("p = rb'\\d+' + f\"{x}\" + U'u'");
    Check(Has(t, Kind::String, "rb'\\d+'") && Has(t, Kind::String, "f\"{x}\"") && Has(t, Kind::String, "U'u'"),
          "prefixed strings");
    Check(!Has(t, Kind::Identifier, "rb") && !Has(t, Kind::Identifier, "f"), "prefix is not an identifier");

    t = Tokens("n = 0x1F + 1_000 + 3.5e-2 + .5 + 2j");
    for (const char* n : {"0x1F", "1_000", "3.5e-2", ".5", "2j"}) Check(Has(t, Kind::Number, n), std::string("number ") + n);

    t = Tokens("@torch.no_grad()");
    Check(Has(t, Kind::Decorator, "@torch.no_grad"), "decorator");

    t = Tokens("for item in items:");
    Check(Has(t, Kind::Identifier, "for") && Has(t, Kind::Identifier, "items") && Has(t, Kind::Punctuation, ":"),
          "identifiers and punctuation (keywords are identifiers the editor looks up)");

    t = Tokens("x = 'a#b'");
    Check(Has(t, Kind::String, "'a#b'"), "hash inside a single-quoted string is part of it");

    std::cout << "python tokenizer: quotes, triple quotes, prefixes, escapes, numbers, decorators. OK\n";
    return 0;
}
