// Bounded, read-only rendering for text files that are too large for TextEditor.

#include "script_editor.h"

#include <algorithm>
#include <climits>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

#include <imgui.h>
#include <spdlog/spdlog.h>

namespace cyxwiz {

ScriptEditorPanel::~ScriptEditorPanel() {
    async_owner_alive_->store(false);
    for (auto& tab : tabs_) {
        if (tab) {
            CancelTabTasks(*tab);
        }
    }
}

void ScriptEditorPanel::CancelTabTasks(EditorTab& tab) {
    if (tab.load_task_id != 0) {
        AsyncTaskManager::Instance().Cancel(tab.load_task_id);
        tab.load_task_id = 0;
    }
    if (tab.large_page_task_id != 0) {
        AsyncTaskManager::Instance().Cancel(tab.large_page_task_id);
        tab.large_page_task_id = 0;
    }
    ++tab.large_page_generation;
}

void ScriptEditorPanel::OpenLargeFileAsync(
    std::uint64_t document_id,
    const std::string& filepath) {
    const int tab_index = FindTabIndex(document_id);
    if (tab_index < 0) {
        return;
    }

    struct IndexResult {
        LargeTextFileIndex index;
    };
    auto result = std::make_shared<IndexResult>();
    std::weak_ptr<std::atomic<bool>> owner_alive = async_owner_alive_;
    const std::string filename = std::filesystem::path(filepath).filename().string();

    const std::uint64_t task_id = AsyncTaskManager::Instance().RunAsync(
        "Indexing large file: " + filename,
        [filepath, result](LambdaTask& task) {
            std::string error;
            const bool success = LargeTextFile::BuildIndex(
                filepath,
                kLargeTextCheckpointStride,
                result->index,
                error,
                [&task]() { return task.ShouldStop(); },
                [&task](float progress, const std::string& status) {
                    task.ReportProgress(progress, status);
                });
            if (!success && !task.ShouldStop()) {
                throw std::runtime_error(error.empty() ? "Could not index large file" : error);
            }
        },
        nullptr,
        [this, owner_alive, document_id, filepath, result](
            bool success,
            const std::string& error) {
            const auto alive = owner_alive.lock();
            if (!alive || !alive->load()) {
                return;
            }

            const int current_index = FindTabIndex(document_id);
            if (current_index < 0) {
                return;
            }

            auto& tab = tabs_[current_index];
            tab->load_task_id = 0;
            tab->is_loading = false;
            if (!success) {
                tab->large_error = error.empty() ? "Indexing cancelled" : error;
                tab->load_status = tab->large_error;
                spdlog::warn("Large-file indexing stopped for '{}': {}", filepath, tab->large_error);
                return;
            }

            tab->large_index = std::move(result->index);
            tab->load_progress = 1.0f;
            tab->load_status.clear();
            tab->go_to_line = tab->large_index.line_count == 0 ? 0 : 1;
            spdlog::info(
                "Large-file index ready for '{}': {} bytes, {} lines, {} checkpoints",
                filepath,
                tab->large_index.file_size,
                tab->large_index.line_count,
                tab->large_index.checkpoint_offsets.size());

            if (tab->large_index.line_count > 0) {
                RequestLargeFilePage(document_id, 0);
            }
        });

    tabs_[tab_index]->load_task_id = task_id;
}

void ScriptEditorPanel::RequestLargeFilePage(
    std::uint64_t document_id,
    std::uint64_t first_line) {
    const int tab_index = FindTabIndex(document_id);
    if (tab_index < 0) {
        return;
    }

    auto& tab = tabs_[tab_index];
    if (!tab->is_large_file || tab->large_index.line_count == 0) {
        return;
    }

    const std::uint64_t clamped = std::min(first_line, tab->large_index.line_count - 1);
    const auto page_lines = static_cast<std::uint64_t>(kLargeTextPageLines);
    const auto half_page = page_lines / 2;
    std::uint64_t page_start = clamped > half_page ? clamped - half_page : 0;
    if (tab->large_index.line_count > page_lines &&
        page_start + page_lines > tab->large_index.line_count) {
        page_start = tab->large_index.line_count - page_lines;
    }

    const bool requested_line_is_loaded =
        !tab->large_page.lines.empty() &&
        clamped >= tab->large_page.first_line &&
        clamped < tab->large_page.first_line + tab->large_page.lines.size();
    const bool requested_line_is_pending =
        tab->large_page_loading &&
        clamped >= tab->requested_page_start &&
        clamped < tab->requested_page_start + page_lines;
    if (requested_line_is_loaded || requested_line_is_pending) {
        return;
    }

    if (tab->large_page_task_id != 0) {
        AsyncTaskManager::Instance().Cancel(tab->large_page_task_id);
    }

    tab->large_page_loading = true;
    tab->large_error.clear();
    tab->requested_page_start = page_start;
    const std::uint64_t generation = ++tab->large_page_generation;

    struct PageResult {
        LargeTextFilePage page;
    };
    auto result = std::make_shared<PageResult>();
    const auto index = tab->large_index;
    const auto path = tab->filepath;
    const auto filename = tab->filename;
    std::weak_ptr<std::atomic<bool>> owner_alive = async_owner_alive_;

    const std::uint64_t task_id = AsyncTaskManager::Instance().RunAsync(
        "Reading large-file page: " + filename,
        [path, index, page_start, result](LambdaTask& task) {
            task.ReportProgress(0.1f, "Seeking to page...");
            std::string error;
            const bool success = LargeTextFile::ReadPage(
                path,
                index,
                page_start,
                kLargeTextPageLines,
                kLargeTextMaxLineBytes,
                result->page,
                error,
                [&task]() { return task.ShouldStop(); });
            if (!success && !task.ShouldStop()) {
                throw std::runtime_error(error.empty() ? "Could not read large-file page" : error);
            }
            task.ReportProgress(1.0f, "Page ready");
        },
        nullptr,
        [this, owner_alive, document_id, generation, result](
            bool success,
            const std::string& error) {
            const auto alive = owner_alive.lock();
            if (!alive || !alive->load()) {
                return;
            }

            const int current_index = FindTabIndex(document_id);
            if (current_index < 0) {
                return;
            }

            auto& current_tab = tabs_[current_index];
            if (generation != current_tab->large_page_generation) {
                return;
            }

            current_tab->large_page_task_id = 0;
            current_tab->large_page_loading = false;
            if (!success) {
                current_tab->large_error = error.empty() ? "Page load cancelled" : error;
                return;
            }
            current_tab->large_page = std::move(result->page);
            current_tab->large_error.clear();
        });

    tab->large_page_task_id = task_id;
}

void ScriptEditorPanel::RenderLargeFileViewer(EditorTab& tab) {
    const double size_mib = static_cast<double>(tab.large_index.file_size) /
        (1024.0 * 1024.0);
    ImGui::TextDisabled(
        "Read-only large-file view | %.1f MiB | %llu lines | %zu-line cache",
        size_mib,
        static_cast<unsigned long long>(tab.large_index.line_count),
        kLargeTextPageLines);

    if (!tab.large_error.empty()) {
        ImGui::TextColored(
            ImVec4(1.0f, 0.35f, 0.35f, 1.0f),
            "%s",
            tab.large_error.c_str());
    }

    auto navigate_to = [this, &tab](std::uint64_t zero_based_line) {
        if (tab.large_index.line_count == 0) {
            return;
        }
        const auto target = std::min(zero_based_line, tab.large_index.line_count - 1);
        tab.go_to_line = target + 1;
        tab.scroll_to_line = target;
        tab.request_large_scroll = true;
        RequestLargeFilePage(tab.document_id, target);
    };

    ImGui::BeginDisabled(tab.large_index.line_count == 0);
    ImGui::SetNextItemWidth(170.0f);
    const std::uint64_t step = 1;
    const std::uint64_t fast_step = 1000;
    ImGui::InputScalar(
        "Line##large_text_line",
        ImGuiDataType_U64,
        &tab.go_to_line,
        &step,
        &fast_step);
    ImGui::SameLine();
    if (ImGui::Button("Go##large_text_go")) {
        const auto target = tab.go_to_line == 0 ? 0 : tab.go_to_line - 1;
        navigate_to(target);
    }
    ImGui::SameLine();
    if (ImGui::Button("First##large_text_first")) {
        navigate_to(0);
    }
    ImGui::SameLine();
    if (ImGui::Button("Previous page##large_text_previous")) {
        const auto current = tab.large_page.first_line;
        navigate_to(current > kLargeTextPageLines ? current - kLargeTextPageLines : 0);
    }
    ImGui::SameLine();
    if (ImGui::Button("Next page##large_text_next")) {
        navigate_to(tab.large_page.first_line + kLargeTextPageLines);
    }
    ImGui::SameLine();
    if (ImGui::Button("Last##large_text_last")) {
        navigate_to(tab.large_index.line_count - 1);
    }
    ImGui::EndDisabled();

    if (tab.large_page_loading) {
        ImGui::SameLine();
        ImGui::TextDisabled("Loading visible page...");
    }

    ImGui::Separator();
    const ImGuiWindowFlags child_flags = ImGuiWindowFlags_HorizontalScrollbar;
    ImGui::BeginChild("##large_text_content", ImVec2(0, 0), false, child_flags);

    const float line_height = ImGui::GetTextLineHeightWithSpacing();
    if (tab.request_large_scroll) {
        ImGui::SetScrollY(static_cast<float>(tab.scroll_to_line) * line_height);
        tab.request_large_scroll = false;
    }

    const auto renderable_lines = static_cast<int>(std::min<std::uint64_t>(
        tab.large_index.line_count,
        static_cast<std::uint64_t>(INT_MAX)));
    if (tab.large_index.line_count > static_cast<std::uint64_t>(INT_MAX)) {
        ImGui::TextDisabled("Viewer is limited to the first %d lines.", INT_MAX);
    }

    int digits = 1;
    for (std::uint64_t count = std::max<std::uint64_t>(1, tab.large_index.line_count);
         count >= 10;
         count /= 10) {
        ++digits;
    }
    const float gutter_width = ImGui::CalcTextSize("0").x * static_cast<float>(digits + 2);

    ImGuiListClipper clipper;
    clipper.Begin(renderable_lines, line_height);
    while (clipper.Step()) {
        if (clipper.DisplayStart < clipper.DisplayEnd) {
            RequestLargeFilePage(tab.document_id, static_cast<std::uint64_t>(clipper.DisplayStart));
            RequestLargeFilePage(
                tab.document_id,
                static_cast<std::uint64_t>(clipper.DisplayEnd - 1));
        }

        for (int line_index = clipper.DisplayStart;
             line_index < clipper.DisplayEnd;
             ++line_index) {
            const auto absolute_line = static_cast<std::uint64_t>(line_index);
            ImGui::TextDisabled(
                "%*llu",
                digits,
                static_cast<unsigned long long>(absolute_line + 1));
            ImGui::SameLine(gutter_width);

            const bool in_page =
                absolute_line >= tab.large_page.first_line &&
                absolute_line < tab.large_page.first_line + tab.large_page.lines.size();
            if (!in_page) {
                ImGui::TextDisabled("Loading...");
                continue;
            }

            const auto page_offset = static_cast<std::size_t>(
                absolute_line - tab.large_page.first_line);
            ImGui::TextUnformatted(tab.large_page.lines[page_offset].c_str());
            if (page_offset < tab.large_page.truncated_lines.size() &&
                tab.large_page.truncated_lines[page_offset]) {
                ImGui::SameLine();
                ImGui::TextDisabled("[line truncated]");
            }
        }
    }

    ImGui::EndChild();
}

} // namespace cyxwiz
