#pragma once

// MjEnvBrowserPanel — the Environment Library: search, source filter, a
// category list with counts and a grid of environment cards (Load, Download,
// Loaded), plus Import from URL. Drawn with the Engine's widgets
// (PluginContext::GetUi, plugin API 1.4).

#include "mj_env_library.h"
#include "mj_env_manager.h"

#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace cyxwiz::plugin {
class PluginContext;
}

namespace cyxwiz::plugin::mujoco {

// Forward declaration
class MjMenagerieDownloader;

// Callback invoked when user loads an environment from the browser: the model
// file and the library entry it belongs to.
using EnvLoadCallback = std::function<bool(const std::string& mjcf_path, const EnvInfo& info)>;

class MjEnvBrowserPanel {
public:
    void SetContext(PluginContext* ctx) { ctx_ = ctx; }
    void SetLoadCallback(EnvLoadCallback cb) { load_callback_ = std::move(cb); }
    void SetDownloader(MjMenagerieDownloader* dl) { downloader_ = dl; }
    // Why another model cannot be loaded now (a graph run steps the loaded one),
    // or empty when it can
    void SetLoadBlocker(std::function<std::string()> fn) { load_blocker_ = std::move(fn); }
    // A model loaded from elsewhere (a file the viewport opened): no card is Loaded
    void ClearLoaded() { loaded_env_id_.clear(); }

    void Render(MjEnvLibrary& library, const MjEnvManager& env, bool* visible);

private:
    // A download that finished on the downloader's thread, applied to the
    // library on the UI thread at the start of the next frame.
    struct FinishedDownload {
        bool url_import = false;
        std::string id;    // Menagerie model id, or the imported model's name
        std::string dir;   // where the files went
        std::string url;
        bool ok = false;
        std::string error;
    };

    void ApplyFinishedDownloads(MjEnvLibrary& library);
    void RenderTopRow();
    void RenderUrlImport();
    void RenderCategories(const std::vector<const EnvInfo*>& by_source);
    void RenderCards(MjEnvLibrary& library, const MjEnvManager& env, const std::vector<const EnvInfo*>& shown);
    void RenderCard(MjEnvLibrary& library, const EnvInfo& info, bool is_loaded, float height);
    void RenderFooter(const MjEnvLibrary& library);
    bool MatchesSource(const EnvInfo& info) const;
    bool MatchesSearch(const EnvInfo& info) const;

    PluginContext* ctx_ = nullptr;
    EnvLoadCallback load_callback_;
    std::function<std::string()> load_blocker_;
    MjMenagerieDownloader* downloader_ = nullptr;

    char search_buf_[128] = {};
    int source_ = 0;                 // All, Built-in, Menagerie, Imported
    std::string selected_category_;  // Empty = "All"
    std::string loaded_env_id_;
    std::string pending_load_path_;  // Deferred load — processed next frame
    EnvInfo pending_load_info_;

    std::string downloading_env_id_;                        // Menagerie model being downloaded
    std::map<std::string, std::string> download_errors_;    // id -> why its download failed
    std::mutex finished_mutex_;
    std::vector<FinishedDownload> finished_;                // filled by the downloader's thread

    // URL import
    bool import_open_ = false;
    char url_buf_[512] = {};
    bool url_importing_ = false;
    std::string url_import_status_;
    std::string url_import_error_;
};

} // namespace cyxwiz::plugin::mujoco
