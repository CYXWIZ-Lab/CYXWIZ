// Preferences > Storage (TOFIX160 P7, approved mockup
// D:/bench157/storage_mockup.html): every prepared-data cache with its size,
// its limit and Clear (asked first). Measuring and clearing walk folders, so
// they run as tasks; the tab draws the last measure.

#include "toolbar.h"

#include "../../core/async_task_manager.h"
#include "../../core/data_budget.h"
#include "../../core/data_registry.h"
#include "../../core/project_manager.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_fonts.h"
#include "../ui_platform.h"
#include "../ui_tokens.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

namespace cyxwiz {

void ToolbarPanel::MeasureStorage() {
    if (storage_measuring_) return;
    storage_measuring_ = true;
    storage_stale_ = false;
    const std::string root = ProjectManager::Instance().GetProjectRoot();
    auto result = std::make_shared<std::vector<StorageCache>>();
    std::weak_ptr<int> alive = storage_alive_;
    AsyncTaskManager::Instance().RunAsync(
        "Measure the prepared-data caches",
        [root, result](LambdaTask&) { *result = MeasureStorageCaches(root, DataRegistry::Instance().FileInUse()); },
        nullptr,
        [this, alive, result](bool, const std::string&) {
            if (alive.expired()) return;
            storage_caches_ = std::move(*result);
            storage_measuring_ = false;
        },
        storage_alive_);
}

void ToolbarPanel::RenderStorageTab() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (storage_stale_) MeasureStorage();

    ImGui::Spacing();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim,
                       "Copies the Engine keeps on disk so data loads and prepares faster next time. Clearing one never "
                       "changes your files or graphs: what it held is made again when needed.");
    ImGui::TextColored(t.text_faint,
                       "When a project opens, files older than 30 days go, then the oldest until each folder is under "
                       "10 GB. A file a loaded dataset reads is never removed.");
    ImGui::PopTextWrapPos();
    ImGui::Spacing();

    if (storage_caches_.empty()) {
        ImGui::TextColored(t.text_dim, "%s", storage_measuring_ ? "Measuring..." : "No cache found.");
        return;
    }

    uint64_t total = 0;
    for (const auto& c : storage_caches_) total += c.bytes;
    {
        ui::FontScope medium(ui::Font::Medium);
        ImGui::TextColored(t.text_bright, "%s", FormatDataBytes(total).c_str());
    }
    ImGui::SameLine(0.0f, 8.0f);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "on disk in %zu caches%s", storage_caches_.size(),
                       storage_measuring_ ? " (measuring...)" : "");
    ImGui::Spacing();

    // One width for every card's size column, so the sizes line up.
    float right_w = ui::ButtonWidth("Clear...", ui::ButtonSize::Small);
    for (const auto& c : storage_caches_)
        right_w = std::max(right_w, ImGui::CalcTextSize((StorageItemsText(c.items, c.unit) + " \xC2\xB7 " + c.limit).c_str()).x);
    right_w += 8.0f;

    std::string clear_id;
    for (const auto& c : storage_caches_) {
        ImGui::PushID(c.id.c_str());
        ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_raised);
        if (ImGui::BeginChild("##cache", ImVec2(0.0f, 0.0f),
                              ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding)) {
            const std::string size = FormatDataBytes(c.bytes);
            const std::string items = StorageItemsText(c.items, c.unit) + " \xC2\xB7 " + c.limit;
            if (ImGui::BeginTable("##row", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("what", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("size", ImGuiTableColumnFlags_WidthFixed,
                                        std::min(right_w, ImGui::GetContentRegionAvail().x * 0.45f));
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                {
                    ui::FontScope medium(ui::Font::Medium);
                    ImGui::TextColored(t.text_bright, "%s", c.label.c_str());
                }
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TextColored(t.text_dim, "%s", c.detail.c_str());
                ImGui::TextColored(t.text_faint, "%s", c.path.c_str());
                ImGui::PopTextWrapPos();
                if (ui::LinkButton("Show in Explorer") && !ui::ShowInFileManager(c.path))
                    spdlog::warn("Storage: could not open {}", c.path);
                ImGui::TableSetColumnIndex(1);
                ImGui::TextColored(t.text, "%s", size.c_str());
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TextColored(t.text_faint, "%s", items.c_str());
                ImGui::PopTextWrapPos();
                if (ui::DangerButton("Clear...", c.bytes > 0 && storage_confirm_ != c.id,
                                     c.bytes > 0 ? "Clearing is being asked" : "Nothing to clear"))
                    storage_confirm_ = c.id;
                ImGui::EndTable();
            }
            if (c.in_use > 0 && storage_confirm_ != c.id) {
                ImGui::TextColored(t.text_dim, "%d %s read by a loaded dataset and stay%s.", c.in_use,
                                   c.in_use == 1 ? "file is" : "files are", c.in_use == 1 ? "s" : "");
            }
            if (storage_confirm_ == c.id) {
                ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_window);
                if (ImGui::BeginChild("##confirm", ImVec2(0.0f, 0.0f),
                                      ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding)) {
                    ImGui::TextColored(t.text_bright, "Clear %s?", c.label.c_str());
                    ImGui::PushTextWrapPos(0.0f);
                    ImGui::TextColored(t.text_dim, "%s, %s. What it held is made again the next time it is needed "
                                       "(slower once). Files a loaded dataset reads stay.",
                                       StorageItemsText(c.items, c.unit).c_str(), size.c_str());
                    ImGui::PopTextWrapPos();
                    if (ui::DangerButton("Clear##confirm")) clear_id = c.id;
                    ImGui::SameLine();
                    if (ui::SecondaryButton("Cancel##confirm")) storage_confirm_.clear();
                }
                ImGui::EndChild();
                ImGui::PopStyleColor();
            }
            if (storage_done_id_ == c.id && !storage_done_note_.empty())
                ImGui::TextColored(t.success, "%s", storage_done_note_.c_str());
        }
        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::PopID();
        ImGui::Spacing();
    }

    if (ProjectManager::Instance().GetProjectRoot().empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim,
                           "Open a project to see its caches (prepared inputs, streamed preparations, prepared "
                           "datasets, saved query answers).");
        ImGui::PopTextWrapPos();
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_faint,
                       "Sizes are measured when the tab opens and after each Clear. Tools > Monitoring > Clear Cache "
                       "still clears the prepared datasets.");
    ImGui::PopTextWrapPos();

    // Clear in a task, then measure again.
    if (!clear_id.empty()) {
        storage_confirm_.clear();
        const std::string root = ProjectManager::Instance().GetProjectRoot();
        auto freed = std::make_shared<StorageClearResult>();
        std::weak_ptr<int> alive = storage_alive_;
        AsyncTaskManager::Instance().RunAsync(
            "Clear a prepared-data cache",
            [clear_id, root, freed](LambdaTask& task) {
                *freed = ClearStorageCache(clear_id, root, DataRegistry::Instance().FileInUse());
                task.MarkCompleted(FormatDataBytes(freed->freed_bytes) + " freed");
            },
            nullptr,
            [this, alive, clear_id, freed](bool, const std::string&) {
                if (alive.expired()) return;
                storage_done_id_ = clear_id;
                storage_done_note_ = "Cleared " + FormatDataBytes(freed->freed_bytes) + " (" +
                                     StorageItemsText(freed->removed, clear_id == "prepared" ? "entry" : "file") + ")" +
                                     (freed->kept > 0 ? "; " + std::to_string(freed->kept) + " kept (in use)" : "") +
                                     ".";
                spdlog::info("Storage: {}", storage_done_note_);
                storage_stale_ = true;
            },
            storage_alive_);
    }
}

}  // namespace cyxwiz
