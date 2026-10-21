#include "runtime_log_category.h"

namespace cyxwiz {
namespace {

enum class Match { Starts, Contains, Ends };

struct Rule {
    Match match;
    std::string_view text;
    std::string_view category;
};

// Checked in order; the first match wins. Prefixes are the subsystem names the
// Engine's log lines start with.
constexpr Rule kRules[] = {
    // Training and generation.
    {Match::Starts, "TrainingExecutor", "training"},
    {Match::Starts, "TrainingManager", "training"},
    {Match::Starts, "TrainingPlotPanel", "training"},
    {Match::Starts, "CheckpointManager", "training"},
    {Match::Starts, "Generation", "training"},
    {Match::Starts, "Epoch ", "training"},
    {Match::Starts, "StartTrainingFromGraph", "training"},
    {Match::Starts, "Task 'Training", "training"},
    {Match::Starts, "Task 'Prepare graph training", "training"},
    // Compute: backends, ArrayFire, the saved compute choice, routes.
    {Match::Starts, "Initializing CyxWiz Backend", "compute"},
    {Match::Starts, "Backend initialization", "compute"},
    {Match::Starts, "ArrayFire", "compute"},
    {Match::Starts, "CUDA backend", "compute"},
    {Match::Starts, "OpenCL", "compute"},
    {Match::Starts, "oneAPI", "compute"},
    {Match::Starts, "Saved compute choice", "compute"},
    {Match::Starts, "Loaded machine compute", "compute"},
    {Match::Starts, "Loaded route qualification", "compute"},
    {Match::Starts, "Compute ", "compute"},
    // Devices.
    {Match::Starts, "Device info", "device"},
    {Match::Starts, "Total devices", "device"},
    {Match::Starts, "Available compute devices", "device"},
    // Plugins.
    {Match::Starts, "PluginManager", "plugin"},
    {Match::Starts, "Plugin", "plugin"},
    {Match::Starts, "Loading plugin", "plugin"},
    {Match::Starts, "ExtensionNodeRegistry", "plugin"},
    {Match::Starts, "Extension node", "plugin"},
    // Python and scripts.
    {Match::Contains, "Python", "python"},
    {Match::Starts, "ScriptingEngine", "python"},
    {Match::Starts, "ScriptManager", "python"},
    {Match::Starts, "StartupScriptManager", "python"},
    {Match::Contains, "startup scripts", "python"},
    {Match::Contains, "virtual environment", "python"},
    {Match::Starts, "Script Editor language tools", "python"},
    {Match::Starts, "Created new script file", "python"},
    // Projects and the asset browser.
    {Match::Starts, "Project ", "project"},
    {Match::Starts, "Opening project", "project"},
    {Match::Starts, "Opened project", "project"},
    {Match::Starts, "Creating main window with project", "project"},
    {Match::Starts, "Refreshing asset browser", "project"},
    {Match::Starts, "Loaded project settings", "project"},
    {Match::Contains, "recent projects", "project"},
    {Match::Contains, "'Scanning project files'", "project"},
    // Account and network.
    {Match::Starts, "AuthClient", "account"},
    {Match::Starts, "Saved sign-in", "account"},
    {Match::Starts, "Auth session", "account"},
    {Match::Starts, "ReservationClient", "network"},
    {Match::Starts, "P2P", "network"},
    {Match::Starts, "Network components", "network"},
    // Data.
    {Match::Starts, "DataRegistry", "data"},
    {Match::Starts, "ParquetBackedDataset", "data"},
    {Match::Starts, "DuckDB", "data"},
    {Match::Starts, "DataInput", "data"},
    {Match::Starts, "Data Pipeline", "data"},
    // Graphs and node patterns.
    {Match::Starts, "GraphCompiler", "graph"},
    {Match::Starts, "NodeMetadataRegistry", "graph"},
    {Match::Starts, "Pattern Library", "graph"},
    {Match::Starts, "Loaded pattern", "graph"},
    {Match::Starts, "Loading builtin patterns", "graph"},
    {Match::Starts, "User patterns directory", "graph"},
    // The window, fonts and panels.
    {Match::Starts, "Window ", "ui"},
    {Match::Starts, "Dark title bar", "ui"},
    {Match::Starts, "Native Windows window", "ui"},
    {Match::Starts, "OpenGL", "ui"},
    {Match::Starts, "Found config file", "ui"},
    {Match::Starts, "Loaded config from", "ui"},
    {Match::Starts, "Appearance:", "ui"},
    {Match::Starts, "Display metrics", "ui"},
    {Match::Contains, "font", "ui"},
    {Match::Contains, "Font", "ui"},
    {Match::Starts, "Merged ", "ui"},
    {Match::Starts, "Loaded Inter", "ui"},
    {Match::Starts, "Loaded JetBrainsMono", "ui"},
    {Match::Ends, "initialized", "ui"},
    {Match::Contains, "Panel", "ui"},
    {Match::Contains, "panel", "ui"},
    {Match::Starts, "MainWindow", "ui"},
    {Match::Starts, "Installed custom dock", "ui"},
    {Match::Starts, "Applied tab size", "ui"},
    {Match::Starts, "Syntax highlighting", "ui"},
    {Match::Contains, "tutorial", "ui"},
    // The Engine itself.
    {Match::Starts, "Starting CyxWiz Engine", "engine"},
    {Match::Starts, "Working directory", "engine"},
    {Match::Starts, "Launch working directory", "engine"},
    {Match::Starts, "AsyncTaskManager", "engine"},
    {Match::Starts, "Task '", "engine"},
    {Match::Starts, "Cancel requested for task", "engine"},
    {Match::Starts, "Console ", "engine"},
};

bool Matches(const Rule& rule, std::string_view message) {
    switch (rule.match) {
        case Match::Starts: return message.substr(0, rule.text.size()) == rule.text;
        case Match::Contains: return message.find(rule.text) != std::string_view::npos;
        case Match::Ends:
            return message.size() >= rule.text.size() &&
                   message.substr(message.size() - rule.text.size()) == rule.text;
    }
    return false;
}

}  // namespace

// A category holds a-z, 0-9 and _ (the store refuses anything else).
std::string CanonicalCategory(std::string_view name) {
    std::string category;
    for (const unsigned char c : name.substr(0, 64)) {
        if (c >= 'A' && c <= 'Z') category.push_back(static_cast<char>(c - 'A' + 'a'));
        else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_') category.push_back(static_cast<char>(c));
        else category.push_back('_');
    }
    return category.empty() ? std::string("system") : category;
}

std::string ClassifyLogCategory(std::string_view logger_name, std::string_view message) {
    if (!logger_name.empty() && logger_name != "cyxwiz") return CanonicalCategory(logger_name);
    while (!message.empty() && (message.front() == ' ' || message.front() == '\t')) message.remove_prefix(1);
    for (const auto& rule : kRules)
        if (Matches(rule, message)) return std::string(rule.category);
    return "system";
}

}  // namespace cyxwiz
