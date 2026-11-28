// MuJoCo's warnings reach the Engine log through the plugin's context: the
// first of each kind once, repeats counted, no MUJOCO_LOG.TXT in the working
// folder, and MuJoCo's previous handlers back after removal. The plugin's own
// lines (LogInfo, LogWarn, LogError) take the same route.

#include "mj_log.h"

#include <mujoco/mujoco.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace cyxwiz::plugin;
using namespace cyxwiz::plugin::mujoco;

namespace {

int failures = 0;

void Check(bool ok, const std::string& what) {
    if (ok) return;
    ++failures;
    std::printf("FAIL: %s\n", what.c_str());
}

class LogContext final : public PluginContext {
public:
    const std::string& GetPluginId() const override { return id_; }
    const std::filesystem::path& GetPluginDir() const override { return dir_; }
    AssistantContextSnapshot GetAssistantContextSnapshot() const override { return {}; }
    void LogInfo(const std::string& message) override { infos.push_back(message); }
    void LogWarn(const std::string& message) override { warnings.push_back(message); }
    void LogError(const std::string& message) override { errors.push_back(message); }
    void LogDebug(const std::string&) override {}
    bool RegisterNodeProvider(INodeProvider*) override { return false; }
    bool RegisterPanelProvider(IPanelProvider*) override { return false; }
    bool RegisterDataProvider(IDataProvider*) override { return false; }
    bool RegisterTrainingHook(ITrainingHook*) override { return false; }
    bool RegisterAnalyticsProvider(IAnalyticsProvider*) override { return false; }
    PluginUi* GetUi() override { return nullptr; }
    bool ShowPanel(const std::string&) override { return false; }
    void RequestFullFrameRate() override {}
    std::filesystem::path GetProjectDir() const override { return {}; }
    std::string GetSetting(const std::string&, const std::string& fallback) const override { return fallback; }
    void SetSetting(const std::string&, const std::string&) override {}

    std::vector<std::string> infos, warnings, errors;

private:
    std::string id_ = "test";
    std::filesystem::path dir_;
};

int previous_calls = 0;
void PreviousWarning(const char*) { ++previous_calls; }
void PreviousError(const char*) {}

}  // namespace

int main() {
    namespace fs = std::filesystem;
    const auto original_cwd = fs::current_path();
    const auto stamp = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto work = fs::temp_directory_path() / ("cyxwiz-mujoco-log-" + stamp);
    fs::create_directories(work);
    fs::current_path(work);

    mju_user_warning = PreviousWarning;
    mju_user_error = PreviousError;
    {
        LogContext ctx;
        InstallMuJoCoLogHandlers(ctx);
        mju_warning("compiler attribute 'settotalmass' is deprecated");
        mju_warning("compiler attribute 'settotalmass' is deprecated");
        mju_warning("Nan, Inf or huge value in QACC at DOF %d. Time = %.4f.", 3, 1.25);
        mju_warning("Nan, Inf or huge value in QACC at DOF %d. Time = %.4f.", 7, 2.5);
        Check(ctx.warnings.size() == 2, "each kind of warning reaches the Engine log once (got " +
                                            std::to_string(ctx.warnings.size()) + ")");
        Check(!ctx.warnings.empty() && ctx.warnings[0].find("settotalmass") != std::string::npos,
              "the warning keeps MuJoCo's text");
        Check(previous_calls == 0, "the plugin's handler, not the previous one, gets the warnings");

        RemoveMuJoCoLogHandlers();
        Check(ctx.infos.size() == 1 && ctx.infos[0].find("2 repeats") != std::string::npos,
              "the repeats are summed up once at removal");
        Check(mju_user_warning == PreviousWarning && mju_user_error == PreviousError,
              "removal puts MuJoCo's previous handlers back");
        mju_warning("after removal");
        Check(previous_calls == 1, "after removal warnings reach the previous handler");
        Check(ctx.warnings.size() == 2, "after removal nothing reaches the plugin's context");
    }
    {
        // The plugin's own lines take the same route, formatted, each at its level.
        LogContext ctx;
        InstallMuJoCoLogHandlers(ctx);
        LogInfo("Loaded model '{}' with {} actuators", "half_cheetah", 6);
        LogWarn("Policy playback: the model {} did not load", "policy.cyxmodel");
        LogError("Failed to download '{}' (status: {})", "scene.xml", 404);
        Check(ctx.infos == std::vector<std::string>{"Loaded model 'half_cheetah' with 6 actuators"},
              "an info line reaches the plugin's context, formatted");
        Check(ctx.warnings == std::vector<std::string>{"Policy playback: the model policy.cyxmodel did not load"},
              "a warning line reaches the plugin's context");
        Check(ctx.errors == std::vector<std::string>{"Failed to download 'scene.xml' (status: 404)"},
              "an error line reaches the plugin's context");
        RemoveMuJoCoLogHandlers();
        LogInfo("after removal {}", 1);
        Check(ctx.infos.size() == 1, "after removal the plugin's lines no longer reach its context");
    }
    mju_user_warning = nullptr;
    mju_user_error = nullptr;

    Check(!fs::exists(work / "MUJOCO_LOG.TXT"), "MuJoCo wrote no MUJOCO_LOG.TXT in the working folder");
    fs::current_path(original_cwd);
    std::error_code ec;
    fs::remove_all(work, ec);

    if (failures) {
        std::printf("%d MuJoCo log check(s) failed\n", failures);
        return 1;
    }
    std::printf("MuJoCo log checks passed\n");
    return 0;
}
