#pragma once

#include "mj_env_library.h"
#include "mj_env_manager.h"
#include "mj_renderer.h"

#include <string>
#include <functional>

namespace cyxwiz::plugin::mujoco {

// Forward declaration
class MjMenagerieDownloader;

// Callback invoked when user loads an environment from the browser.
using EnvLoadCallback = std::function<bool(const std::string& mjcf_path)>;

class MjEnvBrowserPanel {
public:
    void SetLoadCallback(EnvLoadCallback cb) { load_callback_ = std::move(cb); }
    void SetDownloader(MjMenagerieDownloader* dl) { downloader_ = dl; }

    void Render(MjEnvLibrary& library, const MjEnvManager& env, bool* visible);

private:
    void RenderCategoryTabs(const MjEnvLibrary& library);
    void RenderSearchBar();
    void RenderEnvCard(const EnvInfo& info, MjEnvLibrary& library, bool is_loaded);

    EnvLoadCallback load_callback_;
    MjMenagerieDownloader* downloader_ = nullptr;
    std::string filter_text_;
    std::string selected_category_;  // Empty = "All"
    std::string loaded_env_id_;
    std::string pending_load_path_;  // Deferred load — processed next frame
    std::string pending_load_id_;
    std::string downloading_env_id_; // Currently downloading model ID
    char search_buf_[128] = {};
};

} // namespace cyxwiz::plugin::mujoco
