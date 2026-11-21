#pragma once

#include <functional>
#include <future>
#include <map>
#include <string>
#include <vector>

namespace gui {

// Preferences > Python (TOFIX129 B9-9): the interpreter, new projects'
// environments, what each Python session starts with, the console's
// history. Every setting saves at once; another interpreter is taken with
// "Use this Python". Checking and finding interpreters run Python, so they
// run off the UI thread.
class PythonSettingsPanel {
public:
    // `runtime_details` gives the running Python's details (Runtime details)
    void Render(const std::function<std::string()>& runtime_details);
    // Read the settings again (after Restore this page's defaults)
    void Reload() { loaded_ = false; }

private:
    struct Check {
        std::string version;
        std::string error;  // empty: usable
    };
    struct Found {
        std::string path;
        std::string version;
    };
    void Load();
    void StartCheck(const std::string& path);
    void Poll();
    void RenderInterpreter(const std::function<std::string()>& runtime_details);
    void RenderNewProjects();
    void RenderSessionStart();
    void RenderConsole();

    bool loaded_ = false;
    std::string saved_path_;  // the interpreter set (EngineConfig)
    std::string path_;        // the field
    bool changed_this_run_ = false;
    std::map<std::string, Check> checks_;  // by path
    std::future<Check> checking_;
    std::string checking_path_;
    std::future<std::vector<Found>> finding_;
    std::vector<Found> found_;
    bool found_shown_ = false;
    std::string packages_;
    std::vector<std::string> packages_ignored_;
    std::string startup_script_;
    std::string details_text_;
};

}  // namespace gui
