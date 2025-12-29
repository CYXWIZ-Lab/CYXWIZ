#include "script_manager.h"
#include "scripting_engine.h"
#include <algorithm>
#include <cctype>
#include <spdlog/spdlog.h>
#include <imgui.h>

namespace scripting {

ScriptManager::ScriptManager() {
    InitializeKeywords();
    InitializeBuiltins();
    InitializeSnippets();
    InitializeModules();
    spdlog::info("ScriptManager initialized");
}

ScriptManager::~ScriptManager() = default;

void ScriptManager::Initialize(ScriptingEngine* engine) {
    scripting_engine_ = engine;
}

void ScriptManager::InitializeKeywords() {
    keywords_ = {
        "if", "elif", "else", "for", "while", "break", "continue", "pass",
        "try", "except", "finally", "raise", "with", "as", "assert",
        "def", "return", "yield", "class", "lambda",
        "and", "or", "not", "in", "is",
        "import", "from", "global", "nonlocal", "del",
        "async", "await", "True", "False", "None"
    };
}

void ScriptManager::InitializeBuiltins() {
    builtins_ = {
        "int", "float", "str", "bool", "list", "dict", "set", "tuple",
        "print", "input", "open", "range", "enumerate", "zip", "map",
        "filter", "sorted", "reversed", "len", "abs", "min", "max", "sum",
        "type", "isinstance", "dir", "getattr", "setattr", "hasattr"
    };
    builtin_signatures_ = {
        {"print", "print(*objects, sep=' ', end='\\n')"},
        {"len", "len(s) -> int"},
        {"range", "range(stop) or range(start, stop[, step])"},
        {"enumerate", "enumerate(iterable, start=0)"},
        {"isinstance", "isinstance(obj, classinfo) -> bool"}
    };
}

void ScriptManager::InitializeSnippets() {
    snippets_ = {
        {"def", CompletionItem::Kind::Snippet, "function"},
        {"class", CompletionItem::Kind::Snippet, "class"},
        {"for", CompletionItem::Kind::Snippet, "for loop"},
    };
    snippets_[0].insert_text = "def func():\n    pass";
    snippets_[1].insert_text = "class Name:\n    pass";
    snippets_[2].insert_text = "for i in range(n):\n    pass";
}

void ScriptManager::InitializeModules() {
    common_modules_ = {"os", "sys", "math", "numpy", "torch", "pandas"};
}

std::vector<CompletionItem> ScriptManager::GetCompletions(
    const std::string& code, size_t, const std::string& line, int col
) {
    std::vector<CompletionItem> results;
    if (col > 0 && line[col-1] == '.') {
        return GetAttributeCompletions(GetObjectBeforeDot(line, col-1));
    }
    std::string prefix = GetWordAtCursor(line, col);
    if (prefix.empty()) return results;

    auto kw = GetKeywordCompletions(prefix);
    auto bi = GetBuiltinCompletions(prefix);
    results.insert(results.end(), kw.begin(), kw.end());
    results.insert(results.end(), bi.begin(), bi.end());
    ScoreCompletions(results, prefix);
    std::sort(results.begin(), results.end(),
        [](const CompletionItem& a, const CompletionItem& b) { return a.score > b.score; });
    if (results.size() > 30) results.resize(30);
    return results;
}

std::vector<CompletionItem> ScriptManager::GetKeywordCompletions(const std::string& prefix) {
    std::vector<CompletionItem> r;
    for (const auto& k : keywords_)
        if (k.find(prefix) == 0) r.emplace_back(k, CompletionItem::Kind::Keyword, "keyword");
    return r;
}

std::vector<CompletionItem> ScriptManager::GetBuiltinCompletions(const std::string& prefix) {
    std::vector<CompletionItem> r;
    for (const auto& b : builtins_) {
        if (b.find(prefix) == 0) {
            CompletionItem item(b, CompletionItem::Kind::Builtin);
            auto it = builtin_signatures_.find(b);
            item.detail = (it != builtin_signatures_.end()) ? it->second : "builtin";
            r.push_back(item);
        }
    }
    return r;
}

std::vector<CompletionItem> ScriptManager::GetSnippetCompletions(const std::string& prefix) {
    std::vector<CompletionItem> r;
    for (const auto& s : snippets_) if (s.label.find(prefix) == 0) r.push_back(s);
    return r;
}

std::vector<CompletionItem> ScriptManager::GetModuleCompletions(const std::string& prefix) {
    std::vector<CompletionItem> r;
    for (const auto& m : common_modules_)
        if (m.find(prefix) == 0) r.emplace_back(m, CompletionItem::Kind::Module, "module");
    return r;
}

std::vector<CompletionItem> ScriptManager::GetAttributeCompletions(const std::string& obj) {
    std::vector<CompletionItem> r;
    if (obj == "np" || obj == "numpy")
        for (const char* f : {"array", "zeros", "ones", "sum", "mean"})
            r.emplace_back(f, CompletionItem::Kind::Function, "numpy");
    else if (obj == "torch")
        for (const char* f : {"tensor", "zeros", "ones", "nn", "optim"})
            r.emplace_back(f, CompletionItem::Kind::Function, "torch");
    return r;
}

std::vector<CompletionItem> ScriptManager::GetPythonCompletions(const std::string&, const std::string&) {
    return {};
}

std::string ScriptManager::GetFunctionSignature(const std::string& f) {
    auto it = builtin_signatures_.find(f);
    return (it != builtin_signatures_.end()) ? it->second : "";
}

bool ScriptManager::ShouldTriggerCompletion(char c) const {
    return c == '.' || std::isalpha(static_cast<unsigned char>(c)) || c == '_';
}

std::string ScriptManager::GetWordAtCursor(const std::string& line, int col) {
    if (col <= 0) return "";
    int s = col - 1;
    while (s >= 0 && (std::isalnum(static_cast<unsigned char>(line[s])) || line[s] == '_')) s--;
    return line.substr(s + 1, col - s - 1);
}

std::string ScriptManager::GetObjectBeforeDot(const std::string& line, int col) {
    if (col <= 0) return "";
    int e = col - 1;
    while (e >= 0 && std::isspace(static_cast<unsigned char>(line[e]))) e--;
    int s = e;
    while (s >= 0 && (std::isalnum(static_cast<unsigned char>(line[s])) || line[s] == '_')) s--;
    return line.substr(s + 1, e - s);
}

void ScriptManager::ScoreCompletions(std::vector<CompletionItem>& items, const std::string& prefix) {
    for (auto& i : items) i.score = CalculateMatchScore(i.label, prefix);
}

int ScriptManager::CalculateMatchScore(const std::string& item, const std::string& prefix) {
    if (item.find(prefix) == 0) return 100 - static_cast<int>(item.length());
    return 0;
}

const char* GetCompletionKindIcon(CompletionItem::Kind k) {
    switch (k) {
        case CompletionItem::Kind::Keyword: return "K";
        case CompletionItem::Kind::Builtin: return "B";
        case CompletionItem::Kind::Module: return "M";
        case CompletionItem::Kind::Function: return "f";
        case CompletionItem::Kind::Variable: return "v";
        default: return "?";
    }
}

unsigned int GetCompletionKindColor(CompletionItem::Kind k) {
    return IM_COL32(200, 200, 200, 255);
}

} // namespace scripting
