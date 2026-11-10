#include "studio_debugger_items_run.h"

#include "studio_debugger_colors.h"
#include "studio_debugger_widgets.h"

#include "../icons.h"
#include "../ui_tokens.h"

#include <imgui.h>

#include <algorithm>

namespace cyxwiz {

void StudioDebuggerPanel::DataLoadsEnded(const std::string& failed) {
    for (ItemsRun* run : {top_images_run_.get(), embedding_run_.get()})
        if (run) ItemsLoadsEnded(*run, failed);
}

StudioDebuggerPanel::DataItemsJob BeginItems(ItemsRun& run, const StudioDebuggerPanel::DataItemsCallback& callback,
                                             const std::string& what) {
    run.error.clear();
    const auto begun = callback ? callback() : StudioDebuggerPanel::DataItemsStart{};
    if (!begun.job) {
        if (!begun.loading.empty() || begun.compiling) {
            if (!run.waiting_for_load && !run.start_after_load) run.started = std::chrono::steady_clock::now();
            run.waiting_for_load = true;
            if (!begun.loading.empty()) run.loading = begun.loading;
            run.waiting_for_compile = begun.compiling;
        } else {
            run.waiting_for_load = false;
            run.waiting_for_compile = false;
            run.error = StudioDebuggerPanel::DataItemsNotStarted(begun, what);
        }
        return {};
    }
    const bool after_wait = run.waiting_for_load || run.start_after_load;
    run.waiting_for_load = false;
    run.waiting_for_compile = false;
    run.start_after_load = false;
    if (!after_wait) {
        run.loading.clear();
        run.started = std::chrono::steady_clock::now();
    }
    run.computing = true;
    run.stage->store(1);
    run.done->store(0);
    run.total->store(0);
    run.cancel->store(false);
    return begun.job;
}

bool RetryItems(ItemsRun& run) {
    if (run.computing) return false;
    if (run.start_after_load) return true;
    return run.waiting_for_compile;
}

void ItemsLoadsEnded(ItemsRun& run, const std::string& failed) {
    if (!run.waiting_for_load || run.waiting_for_compile) return;
    if (failed.empty()) {
        run.start_after_load = true;  // the view starts again on its next frame
    } else {
        run.waiting_for_load = false;
        run.error = "the data could not be loaded (" + failed + ")";
    }
}

void StopItems(ItemsRun& run) {
    run.cancel->store(true);
    run.waiting_for_load = false;
    run.waiting_for_compile = false;
    run.start_after_load = false;
}

void DrawItemsRunCard(const ItemsRun& run, const std::string& title, const std::vector<ItemsStep>& steps,
                      int bar_step) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_raised);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 8.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(16.0f, 14.0f));
    ImGui::BeginChild("ItemsRunCard", ImVec2(std::min(ImGui::GetContentRegionAvail().x, 760.0f), 0.0f),
                      ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
    const int seconds =
        static_cast<int>(std::chrono::duration<float>(std::chrono::steady_clock::now() - run.started).count());
    ImGui::TextColored(DebuggerBright(), "%s", title.c_str());
    const std::string elapsed = std::to_string(seconds) + " s";
    RightAlignOnLine(ImGui::CalcTextSize(elapsed.c_str()).x);
    ImGui::TextColored(DebuggerMuted(), "%s", elapsed.c_str());
    ImGui::Spacing();

    const int stage = run.waiting_for_load ? 0 : run.stage->load();
    std::vector<ItemsStep> all;
    all.push_back({"Load the data", run.waiting_for_compile ? "loaded; the graph compiles again"
                                    : run.waiting_for_load ? "loading " + run.loading
                                                           : (run.loading.empty() ? "already loaded" : run.loading + ": loaded")});
    all.insert(all.end(), steps.begin(), steps.end());
    for (int i = 0; i < static_cast<int>(all.size()); ++i) {
        const bool finished = i < stage;
        const bool active = i == stage;
        ImGui::PushID(i);
        ImGui::TextColored(finished ? t.success : (active ? t.accent_text : t.text_faint), "%s",
                           finished ? ICON_FA_CHECK : (active ? ICON_FA_SPINNER : ICON_FA_CIRCLE));
        ImGui::SameLine(0.0f, 10.0f);
        ImGui::PushTextWrapPos(ImGui::GetContentRegionMax().x - 150.0f);
        ImGui::TextColored(active ? DebuggerText() : DebuggerMuted(), "%s", all[i].title.c_str());
        ImGui::PopTextWrapPos();
        std::string detail = i == 0 ? all[i].detail : (active ? all[i].detail : (finished ? "done" : ""));
        if (active && i == bar_step)
            detail = std::to_string(run.done->load()) + " of " + std::to_string(run.total->load());
        if (!detail.empty()) {
            ImGui::SameLine();
            RightAlignOnLine(ImGui::CalcTextSize(detail.c_str()).x);
            ImGui::TextColored(DebuggerMuted(), "%s", detail.c_str());
        }
        if (active && i == bar_step && run.total->load() > 0) {
            const float fraction = static_cast<float>(run.done->load()) / static_cast<float>(run.total->load());
            const ImVec2 at = ImGui::GetCursorScreenPos();
            const float width = ImGui::GetContentRegionAvail().x - 26.0f;
            ImGui::Dummy(ImVec2(width + 26.0f, 8.0f));
            ImDrawList* dl = ImGui::GetWindowDrawList();
            dl->AddRectFilled(ImVec2(at.x + 26.0f, at.y + 1.0f), ImVec2(at.x + 26.0f + width, at.y + 7.0f),
                              ui::ToU32(t.bg_input), 3.0f);
            dl->AddRectFilled(ImVec2(at.x + 26.0f, at.y + 1.0f),
                              ImVec2(at.x + 26.0f + width * std::clamp(fraction, 0.0f, 1.0f), at.y + 7.0f),
                              ui::ToU32(t.accent), 3.0f);
        }
        ImGui::PopID();
        ImGui::Spacing();
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(DebuggerFaint(), "%s",
                       "The steps also show in Background Tasks. The rest of the debugger stays usable; Stop ends "
                       "after the current batch.");
    ImGui::PopTextWrapPos();
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
}

}  // namespace cyxwiz
