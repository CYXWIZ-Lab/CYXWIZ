#include "python_repl_presentation.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <unordered_set>

namespace cyxwiz::repl {
namespace {

const std::unordered_set<std::string>& Keywords() {
    static const std::unordered_set<std::string> words = {
        "False", "None", "True", "and", "as", "assert", "async", "await",
        "break", "class", "continue", "def", "del", "elif", "else", "except",
        "finally", "for", "from", "global", "if", "import", "in", "is",
        "lambda", "match", "case", "nonlocal", "not", "or", "pass", "raise",
        "return", "try", "while", "with", "yield"};
    return words;
}

const std::unordered_set<std::string>& Builtins() {
    static const std::unordered_set<std::string> words = {
        "abs", "all", "any", "bool", "dict", "dir", "enumerate", "filter",
        "float", "format", "getattr", "hasattr", "help", "id", "input", "int",
        "isinstance", "iter", "len", "list", "map", "max", "min", "next",
        "object", "open", "print", "range", "repr", "reversed", "round",
        "set", "setattr", "sorted", "str", "sum", "super", "tuple", "type",
        "zip", "self"};
    return words;
}

bool IsIdentStart(char c) {
    return std::isalpha(static_cast<unsigned char>(c)) || c == '_' ||
           (static_cast<unsigned char>(c) & 0x80);
}

bool IsIdentChar(char c) {
    return IsIdentStart(c) || std::isdigit(static_cast<unsigned char>(c));
}

void Push(std::vector<Token>& line, std::string text, TokenKind kind) {
    if (text.empty()) return;
    if (!line.empty() && line.back().kind == kind) {
        line.back().text += text;
    } else {
        line.push_back({std::move(text), kind});
    }
}

std::vector<std::string> SplitLines(const std::string& code) {
    std::vector<std::string> lines;
    std::string current;
    for (char c : code) {
        if (c == '\n') {
            lines.push_back(current);
            current.clear();
        } else if (c != '\r') {
            current += c;
        }
    }
    lines.push_back(current);
    return lines;
}

std::string LeadingIndent(const std::string& line) {
    size_t n = 0;
    while (n < line.size() && (line[n] == ' ' || line[n] == '\t')) ++n;
    return line.substr(0, n);
}

std::string TrimRight(std::string value) {
    while (!value.empty() &&
           std::isspace(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }
    return value;
}

// Removes a trailing comment (outside strings) for the ':' / '\' checks.
std::string StripComment(const std::string& line) {
    char quote = 0;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (quote) {
            if (c == '\\') { ++i; continue; }
            if (c == quote) quote = 0;
        } else if (c == '"' || c == '\'') {
            quote = c;
        } else if (c == '#') {
            return line.substr(0, i);
        }
    }
    return line;
}

}  // namespace

std::vector<std::vector<Token>> HighlightPython(const std::string& code) {
    std::vector<std::vector<Token>> result;
    std::string triple;  // active triple-quote delimiter across lines
    for (const std::string& line : SplitLines(code)) {
        std::vector<Token> tokens;
        size_t i = 0;
        if (!triple.empty()) {
            const size_t end = line.find(triple);
            if (end == std::string::npos) {
                Push(tokens, line, TokenKind::String);
                result.push_back(std::move(tokens));
                continue;
            }
            Push(tokens, line.substr(0, end + 3), TokenKind::String);
            i = end + 3;
            triple.clear();
        }
        while (i < line.size()) {
            const char c = line[i];
            if (c == '#') {
                Push(tokens, line.substr(i), TokenKind::Comment);
                break;
            }
            if (c == '@' && LeadingIndent(line).size() == i) {
                size_t j = i + 1;
                while (j < line.size() && (IsIdentChar(line[j]) || line[j] == '.')) ++j;
                Push(tokens, line.substr(i, j - i), TokenKind::Decorator);
                i = j;
                continue;
            }
            // String, with an optional prefix (f, r, b, rb, ...).
            size_t prefix = 0;
            while (i + prefix < line.size() && prefix < 2 &&
                   std::strchr("rRbBfFuU", line[i + prefix]) != nullptr) {
                ++prefix;
            }
            const size_t q = i + prefix;
            if (q < line.size() && (line[q] == '"' || line[q] == '\'') &&
                (prefix == 0 || !IsIdentChar(i > 0 ? line[i - 1] : ' '))) {
                const char quote = line[q];
                const std::string delim3(3, quote);
                if (line.compare(q, 3, delim3) == 0) {
                    const size_t end = line.find(delim3, q + 3);
                    if (end == std::string::npos) {
                        Push(tokens, line.substr(i), TokenKind::String);
                        triple = delim3;
                        i = line.size();
                    } else {
                        Push(tokens, line.substr(i, end + 3 - i), TokenKind::String);
                        i = end + 3;
                    }
                    continue;
                }
                size_t j = q + 1;
                while (j < line.size() && line[j] != quote) {
                    if (line[j] == '\\') ++j;
                    ++j;
                }
                j = std::min(j + 1, line.size());
                Push(tokens, line.substr(i, j - i), TokenKind::String);
                i = j;
                continue;
            }
            if (std::isdigit(static_cast<unsigned char>(c)) ||
                (c == '.' && i + 1 < line.size() &&
                 std::isdigit(static_cast<unsigned char>(line[i + 1])))) {
                size_t j = i;
                while (j < line.size() &&
                       (std::isalnum(static_cast<unsigned char>(line[j])) ||
                        line[j] == '.' || line[j] == '_')) {
                    ++j;
                }
                Push(tokens, line.substr(i, j - i), TokenKind::Number);
                i = j;
                continue;
            }
            if (IsIdentStart(c)) {
                size_t j = i;
                while (j < line.size() && IsIdentChar(line[j])) ++j;
                std::string word = line.substr(i, j - i);
                const bool attribute = i > 0 && line[i - 1] == '.';
                TokenKind kind = TokenKind::Plain;
                if (!attribute && Keywords().count(word)) kind = TokenKind::Keyword;
                else if (!attribute && Builtins().count(word)) kind = TokenKind::Builtin;
                Push(tokens, std::move(word), kind);
                i = j;
                continue;
            }
            Push(tokens, std::string(1, c), TokenKind::Plain);
            ++i;
        }
        result.push_back(std::move(tokens));
    }
    return result;
}

bool IsInputComplete(const std::string& code) {
    const std::vector<std::string> lines = SplitLines(code);
    int depth = 0;
    std::string triple;
    for (const std::string& line : lines) {
        char quote = 0;
        for (size_t i = 0; i < line.size(); ++i) {
            const char c = line[i];
            if (!triple.empty()) {
                if (line.compare(i, 3, triple) == 0) { triple.clear(); i += 2; }
                continue;
            }
            if (quote) {
                if (c == '\\') { ++i; continue; }
                if (c == quote) quote = 0;
                continue;
            }
            if (c == '#') break;
            if ((c == '"' || c == '\'') && line.compare(i, 3, std::string(3, c)) == 0) {
                triple = std::string(3, c);
                i += 2;
                continue;
            }
            if (c == '"' || c == '\'') { quote = c; continue; }
            if (c == '(' || c == '[' || c == '{') ++depth;
            if (c == ')' || c == ']' || c == '}') --depth;
        }
    }
    if (!triple.empty() || depth > 0) return false;

    std::string last = TrimRight(StripComment(lines.back()));
    if (!last.empty() && (last.back() == ':' || last.back() == '\\')) return false;

    // A compound statement (any line ending in ':') runs once the user adds
    // an empty line, as in the standard Python REPL.
    bool compound = false;
    for (const std::string& line : lines) {
        const std::string stripped = TrimRight(StripComment(line));
        if (!stripped.empty() && stripped.back() == ':') compound = true;
    }
    if (compound && lines.size() > 1) {
        return TrimRight(lines.back()).empty();
    }
    return true;
}

std::string NextLineIndent(const std::string& code) {
    const std::vector<std::string> lines = SplitLines(code);
    const std::string& last = lines.back();
    std::string indent = LeadingIndent(last);
    const std::string stripped = TrimRight(StripComment(last));
    if (!stripped.empty() && stripped.back() == ':') indent += "    ";
    return indent;
}

int CountLines(const std::string& code) {
    return static_cast<int>(std::count(code.begin(), code.end(), '\n')) + 1;
}

std::string LineCountLabel(const std::string& code) {
    const int lines = CountLines(code);
    return std::to_string(lines) + (lines == 1 ? " line" : " lines") + " \xC2\xB7 Python";
}

std::string CompletionWordAt(const std::string& code, size_t cursor) {
    cursor = std::min(cursor, code.size());
    size_t start = cursor;
    while (start > 0 && (IsIdentChar(code[start - 1]) || code[start - 1] == '.')) {
        --start;
    }
    return code.substr(start, cursor - start);
}

std::optional<std::string> MissingModuleInstallCommand(const std::string& error_text) {
    const std::string marker = "No module named '";
    const size_t at = error_text.find(marker);
    if (at == std::string::npos) return std::nullopt;
    const size_t start = at + marker.size();
    const size_t end = error_text.find('\'', start);
    if (end == std::string::npos || end == start) return std::nullopt;
    std::string module = error_text.substr(start, end - start);
    module = module.substr(0, module.find('.'));  // top-level package
    static const std::pair<const char*, const char*> kPackages[] = {
        {"cv2", "opencv-python"}, {"PIL", "pillow"}, {"sklearn", "scikit-learn"},
        {"yaml", "pyyaml"}, {"bs4", "beautifulsoup4"}, {"skimage", "scikit-image"},
        {"dateutil", "python-dateutil"}};
    for (const auto& [import_name, package] : kPackages) {
        if (module == import_name) {
            module = package;
            break;
        }
    }
    return "pip install " + module;
}

std::string FormatElapsed(double seconds) {
    char buffer[32];
    if (seconds < 10.0) {
        std::snprintf(buffer, sizeof(buffer), "%.1f s", seconds);
    } else if (seconds < 60.0) {
        std::snprintf(buffer, sizeof(buffer), "%.0f s", seconds);
    } else {
        const int total = static_cast<int>(seconds + 0.5);
        std::snprintf(buffer, sizeof(buffer), "%d min %02d s", total / 60, total % 60);
    }
    return buffer;
}

std::string EnvironmentLabel(const std::string& interpreter_path,
                             const std::string& project_root) {
    if (interpreter_path.empty()) return {};
    namespace fs = std::filesystem;
    const fs::path interpreter(interpreter_path);
    if (!project_root.empty()) {
        const fs::path root = fs::path(project_root).lexically_normal();
        const fs::path venv = root / "python";
        const std::string interp = interpreter.lexically_normal().generic_string();
        const std::string venv_text = venv.generic_string();
        if (interp.size() > venv_text.size() && interp.compare(0, venv_text.size(), venv_text) == 0) {
            return root.filename().string() + "/python";
        }
    }
    fs::path folder = interpreter.parent_path();
    const std::string leaf = folder.filename().string();
    if (leaf == "bin" || leaf == "Scripts") folder = folder.parent_path();
    return folder.filename().string().empty() ? folder.string() : folder.filename().string();
}

StatusView BuildStatusView(const StatusInput& in) {
    StatusView view;
    view.version_label = "Python " + (in.version.empty() ? in.linked_version : in.version);
    view.environment_label = EnvironmentLabel(in.interpreter_path, in.project_root);
    const bool project_env = !in.project_root.empty() &&
        view.environment_label == std::filesystem::path(in.project_root).filename().string() + "/python";
    view.environment_kind = project_env ? "project environment" : "interpreter";
    view.runtime_label = in.bundled_runtime ? "bundled runtime" : "system Python";

    if (!in.engine_available) {
        view.state = StatusState::Unavailable;
        view.state_label = "Unavailable";
        view.message = "Python scripting is not available in this Engine build.";
        return view;
    }
    if (!in.has_project) {
        view.state = StatusState::Unavailable;
        view.state_label = "No project";
        view.message = "Open a project to use the Python REPL.";
        return view;
    }
    if (!in.interpreter_mismatch.empty()) {
        view.state = StatusState::Unavailable;
        view.state_label = "Restart needed";
        view.message = "This project uses a different Python environment. Restart "
                       "CyxWiz to switch to it.";
        view.can_reset = true;
        return view;
    }
    if (in.running) {
        view.state = StatusState::Running;
        view.state_label = "Running \xC2\xB7 " + FormatElapsed(in.running_seconds);
        view.can_interrupt = true;
        return view;
    }
    if (in.env_setup_pending) {
        view.state = StatusState::SettingUp;
        view.state_label = "Setting up";
        view.message = "Setting up the project's Python environment. Python will be "
                       "ready in a few seconds.";
        return view;
    }
    if (in.env_setup_failed) {
        view.state = StatusState::Unavailable;
        view.state_label = "Setup failed";
        view.message = in.env_setup_message.empty()
            ? "The project's Python environment could not be created."
            : in.env_setup_message;
        view.can_run = true;  // falls back to the Engine's interpreter
        return view;
    }
    if (!in.last_init_error.empty() && !in.initialized) {
        view.state = StatusState::Unavailable;
        view.state_label = "Not available";
        view.message = in.last_init_error;
        view.can_run = true;  // retry
        return view;
    }
    view.can_run = true;
    view.can_reset = in.initialized;
    if (in.initialized) {
        view.state = StatusState::Ready;
        view.state_label = "Ready";
    } else {
        view.state = StatusState::NotStarted;
        view.state_label = "Ready";
        view.message = "Python starts when you run your first command.";
    }
    return view;
}

std::string WelcomeText(const StatusView& status) {
    std::string text = status.version_label;
    if (!status.environment_label.empty()) {
        text += " in " + status.environment_kind + " " + status.environment_label;
    }
    text += ". Type help for CyxWiz shortcuts (DuckDB, Polars, MATLAB-style functions).";
    return text;
}

}  // namespace cyxwiz::repl
