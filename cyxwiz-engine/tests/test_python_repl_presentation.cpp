#include "../src/core/python_repl_presentation.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

std::string Joined(const std::vector<cyxwiz::repl::Token>& line) {
    std::string text;
    for (const auto& token : line) text += token.text;
    return text;
}

void TestCompleteness() {
    using cyxwiz::repl::IsInputComplete;
    Check(IsInputComplete("sys.version"), "single expression runs on Enter");
    Check(IsInputComplete("x = 1  # note:"), "a comment ending in ':' is not a block");
    Check(!IsInputComplete("for epoch in range(3):"), "block header waits for a body");
    Check(!IsInputComplete("for epoch in range(3):\n    print(epoch)"),
          "block runs only after an empty line");
    Check(IsInputComplete("for epoch in range(3):\n    print(epoch)\n"),
          "empty line closes the block");
    Check(!IsInputComplete("values = [1, 2,"), "open bracket continues");
    Check(!IsInputComplete("text = \"\"\"first line"), "open triple quote continues");
    Check(IsInputComplete("text = \"\"\"a\nb\"\"\""), "closed triple quote runs");
    Check(!IsInputComplete("total = 1 + \\"), "backslash continuation");
    Check(IsInputComplete("s = '(not a bracket'"), "brackets inside strings are ignored");
}

void TestIndentAndWords() {
    using namespace cyxwiz::repl;
    Check(NextLineIndent("for i in range(3):") == "    ", "indent after ':'");
    Check(NextLineIndent("for i in x:\n    if i:") == "        ", "nested indent");
    Check(NextLineIndent("    total += i") == "    ", "keeps current indent");
    Check(LineCountLabel("a\nb\nc") == "3 lines \xC2\xB7 Python", "line count label");
    Check(LineCountLabel("a") == "1 line \xC2\xB7 Python", "singular line label");
    Check(CompletionWordAt("df = pycyxwiz.lo", 16) == "pycyxwiz.lo", "dotted word at cursor");
    Check(CompletionWordAt("print(sys.ver", 13) == "sys.ver", "word after bracket");
}

void TestHints() {
    using cyxwiz::repl::MissingModuleInstallCommand;
    Check(MissingModuleInstallCommand("ModuleNotFoundError: No module named 'pandas'") ==
              std::optional<std::string>("pip install pandas"),
          "pandas hint");
    Check(MissingModuleInstallCommand("No module named 'cv2'") ==
              std::optional<std::string>("pip install opencv-python"),
          "cv2 maps to opencv-python");
    Check(MissingModuleInstallCommand("No module named 'sklearn.linear_model'") ==
              std::optional<std::string>("pip install scikit-learn"),
          "submodule maps to top package");
    Check(!MissingModuleInstallCommand("ZeroDivisionError: division by zero"),
          "no hint for other errors");
    Check(cyxwiz::repl::FormatElapsed(1.44) == "1.4 s", "seconds with one decimal");
    Check(cyxwiz::repl::FormatElapsed(125) == "2 min 05 s", "minutes");
}

void TestHighlight() {
    using namespace cyxwiz::repl;
    const auto lines = HighlightPython(
        "for epoch in range(3):\n    print(f\"loss {epoch}\")  # log\n"
        "x = obj.print\ndoc = \"\"\"start\nstill\"\"\"");
    Check(lines.size() == 5, "one token list per line");
    Check(Joined(lines[1]) == "    print(f\"loss {epoch}\")  # log", "tokens rebuild the line");
    Check(lines[0][0].kind == TokenKind::Keyword && lines[0][0].text == "for", "keyword");
    bool saw_builtin = false, saw_string = false, saw_comment = false, saw_number = false;
    for (const auto& token : lines[0]) {
        if (token.kind == TokenKind::Builtin && token.text == "range") saw_builtin = true;
        if (token.kind == TokenKind::Number && token.text == "3") saw_number = true;
    }
    for (const auto& token : lines[1]) {
        if (token.kind == TokenKind::String) saw_string = true;
        if (token.kind == TokenKind::Comment) saw_comment = true;
    }
    Check(saw_builtin && saw_number && saw_string && saw_comment, "token kinds");
    bool attribute_plain = true;
    for (const auto& token : lines[2]) {
        if (token.text == "print" && token.kind != TokenKind::Plain) attribute_plain = false;
    }
    Check(attribute_plain, "attribute named like a builtin stays plain");
    Check(lines[4].size() == 1 && lines[4][0].kind == TokenKind::String,
          "triple-quoted string spans lines");
}

void TestStatus() {
    using namespace cyxwiz::repl;
    StatusInput input;
    input.has_project = true;
    input.initialized = true;
    input.version = "3.12.14";
    input.linked_version = "3.12";
    input.project_root = "C:/Users/chick/Documents/CyxWiz Projects/Berean";
    input.interpreter_path =
        "C:/Users/chick/Documents/CyxWiz Projects/Berean/python/Scripts/python.exe";
    input.bundled_runtime = true;
    StatusView view = BuildStatusView(input);
    Check(view.state == StatusState::Ready && view.state_label == "Ready", "ready");
    Check(view.version_label == "Python 3.12.14", "version label");
    Check(view.environment_kind == "project environment", "project environment kind");
    Check(view.environment_label == "Berean/python", "project environment label");
    Check(view.runtime_label == "bundled runtime", "bundled runtime label");
    Check(WelcomeText(view).rfind("Python 3.12.14 in project environment Berean/python.", 0) == 0,
          "welcome text");

    input.running = true;
    input.running_seconds = 3.2;
    view = BuildStatusView(input);
    Check(view.state == StatusState::Running && view.state_label == "Running \xC2\xB7 3.2 s",
          "running label");
    Check(view.can_interrupt && !view.can_run, "interrupt while running");

    input.running = false;
    input.initialized = false;
    input.version.clear();  // not started yet: only the linked version is known
    input.env_setup_pending = true;
    view = BuildStatusView(input);
    Check(view.state == StatusState::SettingUp && !view.can_run, "setting up blocks runs");
    Check(view.version_label == "Python 3.12", "linked version before start");

    input.env_setup_pending = false;
    input.interpreter_mismatch = "desired='a' active='b'";
    view = BuildStatusView(input);
    Check(view.state == StatusState::Unavailable && view.state_label == "Restart needed",
          "mismatch needs restart");

    StatusInput none;
    none.linked_version = "3.12";
    view = BuildStatusView(none);
    Check(view.state_label == "No project" && !view.can_run, "no project");

    StatusInput system;
    system.has_project = true;
    system.initialized = true;
    system.version = "3.12.4";
    system.project_root = "D:/work/demo";
    system.interpreter_path = "C:/Python312/python.exe";
    view = BuildStatusView(system);
    Check(view.environment_kind == "interpreter" && view.environment_label == "Python312",
          "system interpreter label");
    Check(view.runtime_label == "system Python", "system runtime label");
}

}  // namespace

int main() {
    TestCompleteness();
    TestIndentAndWords();
    TestHints();
    TestHighlight();
    TestStatus();
    if (failures) {
        std::cerr << failures << " Python REPL presentation check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "Python REPL presentation tests passed\n";
    return EXIT_SUCCESS;
}
