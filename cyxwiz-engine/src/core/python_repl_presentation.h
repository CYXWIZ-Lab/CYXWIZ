#pragma once

// Presentation model for the Console's Python REPL (tofix121).
// Pure data in, data out: no ImGui, no Python, no backend calls, so the
// logic behind the transcript, input and status header is testable alone.

#include <optional>
#include <string>
#include <vector>

namespace cyxwiz::repl {

// ---- Syntax colouring -------------------------------------------------------

enum class TokenKind { Plain, Keyword, Builtin, String, Number, Comment, Decorator };

struct Token {
    std::string text;
    TokenKind kind = TokenKind::Plain;
};

// One token list per source line. Triple-quoted strings may span lines.
std::vector<std::vector<Token>> HighlightPython(const std::string& code);

// ---- Input behaviour --------------------------------------------------------

// True when pressing Enter should run the input rather than start a new line:
// brackets and strings are closed, no trailing ':' or '\', and a compound
// block (a line ending in ':') has been closed with an empty last line.
bool IsInputComplete(const std::string& code);

// Indentation for the line after `code` (previous indent, plus four spaces
// after a line ending in ':').
std::string NextLineIndent(const std::string& code);

// Line count shown under the input ("7 lines · Python").
int CountLines(const std::string& code);
std::string LineCountLabel(const std::string& code);

// Identifier (with dots) ending at `cursor`, for Tab completion.
std::string CompletionWordAt(const std::string& code, size_t cursor);

// ---- Results ----------------------------------------------------------------

// "pip install <package>" for a ModuleNotFoundError message, mapping common
// import names to their package (cv2 -> opencv-python, PIL -> pillow ...).
std::optional<std::string> MissingModuleInstallCommand(const std::string& error_text);

// "0.8 s", "12 s", "2 min 05 s".
std::string FormatElapsed(double seconds);

// ---- Status header ----------------------------------------------------------

enum class StatusState { Ready, Running, SettingUp, NotStarted, Unavailable };

struct StatusInput {
    bool has_project = false;
    bool engine_available = true;
    bool initialized = false;
    bool running = false;
    double running_seconds = 0.0;
    bool env_setup_pending = false;
    bool env_setup_failed = false;
    std::string env_setup_message;
    std::string interpreter_mismatch;  // non-empty: restart required
    std::string last_init_error;
    std::string version;               // "3.12.14"; empty before start
    std::string linked_version;        // "3.12" (compile-time)
    std::string interpreter_path;      // selected interpreter
    std::string project_root;          // active project folder
    bool bundled_runtime = false;      // interpreter/venv base is the bundled one
};

struct StatusView {
    StatusState state = StatusState::NotStarted;
    std::string state_label;   // "Ready", "Running · 3.2 s", ...
    std::string version_label; // "Python 3.12.14" or "Python 3.12"
    std::string environment_kind;  // "project environment" / "interpreter"
    std::string environment_label; // "Berean/python"
    std::string runtime_label;     // "bundled runtime" / "system Python"
    std::string message;           // explanation for SettingUp/Unavailable
    bool can_run = false;
    bool can_interrupt = false;
    bool can_reset = false;
};

StatusView BuildStatusView(const StatusInput& input);

// "Berean/python" for <root>/python/..., else the interpreter's folder.
std::string EnvironmentLabel(const std::string& interpreter_path,
                             const std::string& project_root);

// Welcome line at the top of a new transcript.
std::string WelcomeText(const StatusView& status);

}  // namespace cyxwiz::repl
