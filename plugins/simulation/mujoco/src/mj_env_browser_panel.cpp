#include "mj_env_browser_panel.h"
#include "mj_menagerie_downloader.h"

#include <imgui.h>
#include <algorithm>
#include <filesystem>
#include <cstdlib>

namespace cyxwiz::plugin::mujoco {

void MjEnvBrowserPanel::Render(MjEnvLibrary& library,
                                const MjEnvManager& env,
                                bool* visible) {
    if (!visible || !*visible) return;

    // Process deferred load BEFORE any ImGui rendering.
    // LoadEnvironment switches GL contexts which corrupts ImGui's GL state mid-frame.
    if (!pending_load_path_.empty()) {
        std::string path = std::move(pending_load_path_);
        std::string id = std::move(pending_load_id_);
        pending_load_path_.clear();
        pending_load_id_.clear();
        if (load_callback_ && load_callback_(path)) {
            loaded_env_id_ = id;
        }
    }

    ImGui::SetNextWindowSize(ImVec2(580, 520), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Environment Library", visible)) {
        ImGui::End();
        return;
    }

    RenderUrlImport(library);
    ImGui::Spacing();
    RenderSearchBar();
    RenderCategoryTabs(library);
    ImGui::Separator();

    // Determine filter text (lowercase)
    std::string filter = search_buf_;
    std::transform(filter.begin(), filter.end(), filter.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    // Environment cards
    float panel_width = ImGui::GetContentRegionAvail().x;
    float card_width = 240.0f;
    int cols = std::max(1, static_cast<int>(panel_width / card_width));

    int col = 0;
    for (const auto& info : library.GetAll()) {
        // Category filter
        if (!selected_category_.empty() && info.category != selected_category_)
            continue;

        // Text search filter
        if (!filter.empty()) {
            std::string name_lower = info.name;
            std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            std::string desc_lower = info.description;
            std::transform(desc_lower.begin(), desc_lower.end(), desc_lower.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (name_lower.find(filter) == std::string::npos &&
                desc_lower.find(filter) == std::string::npos)
                continue;
        }

        if (col > 0) ImGui::SameLine();

        bool is_loaded = (env.IsLoaded() && loaded_env_id_ == info.id);
        RenderEnvCard(info, library, is_loaded);

        col++;
        if (col >= cols) col = 0;
    }

    ImGui::End();
}

void MjEnvBrowserPanel::RenderSearchBar() {
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 4.0f);
    ImGui::InputTextWithHint("##env_search", "Search environments...", search_buf_, sizeof(search_buf_));
}

void MjEnvBrowserPanel::RenderCategoryTabs(const MjEnvLibrary& library) {
    // "All" tab
    bool all_selected = selected_category_.empty();
    if (all_selected) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
    }
    if (ImGui::SmallButton("All")) {
        selected_category_.clear();
    }
    if (all_selected) ImGui::PopStyleColor();

    for (const auto& cat : library.GetCategories()) {
        ImGui::SameLine();
        bool sel = (selected_category_ == cat);
        if (sel) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
        }
        if (ImGui::SmallButton(cat.c_str())) {
            selected_category_ = (selected_category_ == cat) ? "" : cat;
        }
        if (sel) ImGui::PopStyleColor();
    }
}

void MjEnvBrowserPanel::RenderEnvCard(const EnvInfo& info,
                                       MjEnvLibrary& library,
                                       bool is_loaded) {
    ImGui::PushID(info.id.c_str());

    float card_w = 230.0f;
    ImVec2 card_start = ImGui::GetCursorScreenPos();

    // Card background
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 card_end(card_start.x + card_w, card_start.y + 140.0f);
    ImU32 bg_col = is_loaded ? IM_COL32(40, 80, 60, 255) : IM_COL32(45, 45, 50, 255);
    dl->AddRectFilled(card_start, card_end, bg_col, 6.0f);
    dl->AddRect(card_start, card_end, IM_COL32(80, 80, 90, 255), 6.0f);

    // Content inside card
    ImGui::BeginGroup();
    ImGui::SetCursorScreenPos(ImVec2(card_start.x + 10, card_start.y + 8));

    // Name
    ImGui::TextColored(ImVec4(0.9f, 0.9f, 1.0f, 1.0f), "%s", info.name.c_str());

    // Source badge + category
    ImGui::SameLine(card_w - 110);
    if (info.source == EnvSource::Menagerie) {
        if (info.requires_download) {
            ImGui::TextColored(ImVec4(0.4f, 0.7f, 1.0f, 1.0f), "[Cloud]");
        } else {
            ImGui::TextColored(ImVec4(0.3f, 0.9f, 0.5f, 1.0f), "[Downloaded]");
        }
    } else {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "[Builtin]");
    }

    // Description (truncated)
    ImGui::SetCursorScreenPos(ImVec2(card_start.x + 10, card_start.y + 30));
    ImGui::PushTextWrapPos(card_start.x + card_w - 10);
    ImGui::TextDisabled("%s", info.description.c_str());
    ImGui::PopTextWrapPos();

    // Metadata row
    ImGui::SetCursorScreenPos(ImVec2(card_start.x + 10, card_start.y + 80));
    ImGui::Text("Act: %d", info.act_dim);
    ImGui::SameLine();
    ImGui::Text("Obs: %d", info.obs_dim);
    if (info.source == EnvSource::Menagerie) {
        ImGui::SameLine();
        ImGui::TextDisabled("%s", info.category.c_str());
    } else {
        ImGui::SameLine();
        ImGui::Text("Steps: %d", info.max_steps);
    }

    // Action button area
    ImGui::SetCursorScreenPos(ImVec2(card_start.x + 10, card_start.y + 108));

    bool is_downloading = (downloader_ && downloader_->IsDownloading() && downloading_env_id_ == info.id);

    if (is_loaded) {
        ImGui::TextColored(ImVec4(0.3f, 0.9f, 0.5f, 1.0f), "Loaded");
    } else if (is_downloading) {
        // Progress bar
        float progress = downloader_->GetProgress();
        ImGui::SetNextItemWidth(card_w - 20);
        ImGui::ProgressBar(progress, ImVec2(card_w - 20, 20),
                           downloader_->GetStatusText().c_str());
    } else if (info.source == EnvSource::Menagerie && info.requires_download) {
        // Download button
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.35f, 0.6f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.2f, 0.45f, 0.75f, 1.0f));
        if (ImGui::Button("Download", ImVec2(card_w - 20, 22))) {
            if (downloader_ && !downloader_->IsDownloading()) {
                downloading_env_id_ = info.id;
                std::string dest = library.GetMenagerieModelDir(info);
                downloader_->DownloadModelAsync(info.repo_path, dest,
                    [this, id = info.id, &library](bool success, const std::string&) {
                        if (success) {
                            library.MarkDownloaded(id);
                        }
                        downloading_env_id_.clear();
                    });
            }
        }
        ImGui::PopStyleColor(2);
    } else {
        // Load button (builtin or already downloaded)
        if (ImGui::Button("Load", ImVec2(card_w - 20, 22))) {
            pending_load_path_ = library.GetAssetPath(info);
            pending_load_id_ = info.id;
        }
    }

    ImGui::EndGroup();

    // Advance cursor past card
    ImGui::SetCursorScreenPos(ImVec2(card_start.x, card_end.y + 6));
    ImGui::Dummy(ImVec2(card_w, 0));

    ImGui::PopID();
}

void MjEnvBrowserPanel::RenderUrlImport(MjEnvLibrary& library) {
    // Collapsible section for URL import
    if (!ImGui::CollapsingHeader("Import from URL")) return;

    ImGui::Indent(8.0f);

    float avail = ImGui::GetContentRegionAvail().x - 16.0f;
    float btn_w = 80.0f;

    ImGui::SetNextItemWidth(avail - btn_w - 8.0f);
    ImGui::InputTextWithHint("##url_input", "Paste MJCF URL or GitHub directory URL...",
                              url_buf_, sizeof(url_buf_));

    ImGui::SameLine();

    bool is_downloading = (downloader_ && downloader_->IsDownloading() && url_importing_);

    if (is_downloading) {
        ImGui::BeginDisabled();
        ImGui::Button("Importing...", ImVec2(btn_w, 0));
        ImGui::EndDisabled();
    } else {
        if (ImGui::Button("Import", ImVec2(btn_w, 0))) {
            std::string url = url_buf_;
            if (!url.empty() && downloader_) {
                url_importing_ = true;
                url_import_status_.clear();
                url_import_error_.clear();

                // Determine destination: ~/.cyxwiz/imported/<model_name>/
                std::string model_name = "imported_model";
                // Try to extract a name from the URL
                auto last_slash = url.rfind('/');
                if (last_slash != std::string::npos && last_slash + 1 < url.size()) {
                    std::string tail = url.substr(last_slash + 1);
                    // Remove .xml extension if present
                    auto dot = tail.rfind('.');
                    if (dot != std::string::npos)
                        model_name = tail.substr(0, dot);
                    else
                        model_name = tail;
                }

                // Build dest dir
                std::string home;
#ifdef _WIN32
                const char* userprofile = std::getenv("USERPROFILE");
                if (userprofile) home = userprofile;
#else
                const char* h = std::getenv("HOME");
                if (h) home = h;
#endif
                namespace fs = std::filesystem;
                std::string dest_dir = (fs::path(home) / ".cyxwiz" / "imported" / model_name).string();

                downloader_->DownloadFromUrlAsync(url, dest_dir,
                    [this, model_name, dest_dir, &library](bool success, const std::string& err) {
                        url_importing_ = false;
                        if (success) {
                            url_import_status_ = "Downloaded to " + dest_dir;
                            // Find the .xml file and add as custom environment
                            namespace fs = std::filesystem;
                            for (const auto& entry : fs::recursive_directory_iterator(dest_dir)) {
                                if (entry.path().extension() == ".xml") {
                                    EnvInfo info;
                                    info.id = "imported_" + model_name;
                                    info.name = model_name;
                                    info.filename = entry.path().filename().string();
                                    info.description = "Imported from URL";
                                    info.category = "Imported";
                                    info.source = EnvSource::Builtin; // Local file, no download needed
                                    info.obs_dim = 0;
                                    info.act_dim = 0;
                                    info.max_steps = 1000;
                                    info.custom_asset_path = entry.path().string();
                                    library.AddEnv(std::move(info));
                                    break;
                                }
                            }
                        } else {
                            url_import_error_ = err.empty() ? "Download failed" : err;
                        }
                    });
            }
        }
    }

    // Progress bar during download
    if (is_downloading) {
        float progress = downloader_->GetProgress();
        ImGui::ProgressBar(progress, ImVec2(avail, 18),
                           downloader_->GetStatusText().c_str());
    }

    // Status/error messages
    if (!url_import_status_.empty()) {
        ImGui::TextColored(ImVec4(0.3f, 0.9f, 0.5f, 1.0f), "%s", url_import_status_.c_str());
    }
    if (!url_import_error_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s", url_import_error_.c_str());
    }

    ImGui::Unindent(8.0f);
}

} // namespace cyxwiz::plugin::mujoco
