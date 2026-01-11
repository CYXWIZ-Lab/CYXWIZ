#include "cloud_upload_dialog.h"
#include "../icons.h"
#include <imgui.h>
#include <algorithm>
#include <filesystem>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#endif

namespace gui {

static std::string FormatFileSize(int64_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB"};
    int unit = 0;
    double size = static_cast<double>(bytes);
    while (size >= 1024.0 && unit < 3) {
        size /= 1024.0;
        unit++;
    }
    char buf[32];
    snprintf(buf, sizeof(buf), "%.1f %s", size, units[unit]);
    return buf;
}

CloudUploadDialog::CloudUploadDialog()
    : Panel("Upload to Cloud", false) {
}

CloudUploadDialog::~CloudUploadDialog() {
    should_cancel_.store(true);
}

void CloudUploadDialog::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(500, 400), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_CLOUD_ARROW_UP " Upload to CyxCloud###UploadDialog", &visible_)) {
        // Check connection
        if (!datastream_client_ || !datastream_client_->IsConnected()) {
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f),
                ICON_FA_TRIANGLE_EXCLAMATION " Not connected to CyxCloud Gateway");
            ImGui::End();
            return;
        }

        // Drop zone or file list
        if (pending_files_.empty()) {
            RenderDropZone();
        } else {
            RenderFileList();
        }

        ImGui::Separator();

        // Actions or progress
        if (is_uploading_.load()) {
            RenderUploadProgress();
        } else {
            RenderActions();
        }

        // Error display
        if (!last_error_.empty() && error_time_ > 0) {
            error_time_ -= ImGui::GetIO().DeltaTime;
            ImGui::Separator();
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                ICON_FA_CIRCLE_EXCLAMATION " %s", last_error_.c_str());
        }
    }
    ImGui::End();
}

void CloudUploadDialog::RenderDropZone() {
    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImVec2 zone_size = ImVec2(avail.x, avail.y - 60);

    // Draw drop zone
    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    // Dashed border
    ImU32 border_color = IM_COL32(100, 100, 100, 255);
    draw_list->AddRect(pos, ImVec2(pos.x + zone_size.x, pos.y + zone_size.y),
                       border_color, 8.0f, 0, 2.0f);

    // Center text
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + zone_size.y / 2 - 30);

    float text_width = ImGui::CalcTextSize(ICON_FA_CLOUD_ARROW_UP " Drop files here").x;
    ImGui::SetCursorPosX((avail.x - text_width) / 2);
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
        ICON_FA_CLOUD_ARROW_UP " Drop files here");

    text_width = ImGui::CalcTextSize("or").x;
    ImGui::SetCursorPosX((avail.x - text_width) / 2);
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "or");

    float btn_width = 120;
    ImGui::SetCursorPosX((avail.x - btn_width) / 2);
    if (ImGui::Button(ICON_FA_FOLDER_OPEN " Browse...", ImVec2(btn_width, 0))) {
        ShowFileBrowser();
    }

    // Handle drop payload
    if (ImGui::BeginDragDropTarget()) {
        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("CYXWIZ_FILE_PATH")) {
            const char* path = static_cast<const char*>(payload->Data);
            AddFile(path);
        }
        ImGui::EndDragDropTarget();
    }

    // Also accept external drops (via GLFW callback, if implemented)
    // Check for dropped files from the window
}

void CloudUploadDialog::RenderFileList() {
    std::lock_guard<std::mutex> lock(files_mutex_);

    ImGui::Text("Files to upload (%zu):", pending_files_.size());

    ImVec2 avail = ImGui::GetContentRegionAvail();
    float list_height = avail.y - 60;

    if (ImGui::BeginChild("##file_list", ImVec2(0, list_height), true)) {
        for (size_t i = 0; i < pending_files_.size(); i++) {
            auto& file = pending_files_[i];
            ImGui::PushID(static_cast<int>(i));

            // Checkbox
            ImGui::Checkbox("##selected", &file.selected);
            ImGui::SameLine();

            // File icon
            ImGui::Text(ICON_FA_FILE);
            ImGui::SameLine();

            // Filename
            ImGui::Text("%s", file.filename.c_str());
            ImGui::SameLine(250);

            // Size
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                "%s", FormatFileSize(file.size_bytes).c_str());
            ImGui::SameLine(330);

            // Status
            switch (file.state) {
                case PendingFile::State::Pending:
                    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                        ICON_FA_CLOCK " Pending");
                    break;
                case PendingFile::State::Uploading:
                    ImGui::TextColored(ImVec4(0.9f, 0.7f, 0.2f, 1.0f),
                        ICON_FA_SPINNER " %.0f%%", file.progress * 100);
                    break;
                case PendingFile::State::Complete:
                    ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f),
                        ICON_FA_CHECK " Done");
                    break;
                case PendingFile::State::Error:
                    ImGui::TextColored(ImVec4(0.9f, 0.3f, 0.3f, 1.0f),
                        ICON_FA_CIRCLE_XMARK " Error");
                    if (ImGui::IsItemHovered()) {
                        ImGui::BeginTooltip();
                        ImGui::Text("%s", file.error.c_str());
                        ImGui::EndTooltip();
                    }
                    break;
            }

            // Remove button (if not uploading)
            if (file.state == PendingFile::State::Pending || file.state == PendingFile::State::Error) {
                ImGui::SameLine(430);
                if (ImGui::SmallButton(ICON_FA_XMARK)) {
                    RemoveFile(i);
                    ImGui::PopID();
                    break;  // List modified, exit loop
                }
            }

            ImGui::PopID();
        }
    }
    ImGui::EndChild();

    // Add more files button
    if (ImGui::Button(ICON_FA_PLUS " Add More")) {
        ShowFileBrowser();
    }
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_TRASH " Clear All")) {
        ClearState();
    }
}

void CloudUploadDialog::RenderUploadProgress() {
    std::lock_guard<std::mutex> lock(files_mutex_);

    // Overall progress
    size_t completed = 0;
    size_t total = 0;
    for (const auto& f : pending_files_) {
        if (f.selected) {
            total++;
            if (f.state == PendingFile::State::Complete) {
                completed++;
            }
        }
    }

    float overall_progress = total > 0 ? static_cast<float>(completed) / total : 0;
    ImGui::Text("Uploading... %zu / %zu files", completed, total);
    ImGui::ProgressBar(overall_progress, ImVec2(-1, 0));

    // Current file progress
    if (current_upload_index_ < pending_files_.size()) {
        const auto& current = pending_files_[current_upload_index_];
        ImGui::Text("Current: %s", current.filename.c_str());
        ImGui::ProgressBar(current.progress, ImVec2(-1, 0));
    }

    ImGui::Separator();

    if (ImGui::Button(ICON_FA_STOP " Cancel")) {
        should_cancel_.store(true);
    }
}

void CloudUploadDialog::RenderActions() {
    // Count selected files
    size_t selected_count = 0;
    int64_t total_size = 0;
    for (const auto& f : pending_files_) {
        if (f.selected && f.state == PendingFile::State::Pending) {
            selected_count++;
            total_size += f.size_bytes;
        }
    }

    if (selected_count > 0) {
        ImGui::Text("%zu files selected (%s)", selected_count, FormatFileSize(total_size).c_str());
    }

    // Upload button
    if (ImGui::Button(ICON_FA_CLOUD_ARROW_UP " Upload", ImVec2(120, 0))) {
        if (selected_count > 0) {
            StartUpload();
        }
    }

    ImGui::SameLine();

    if (ImGui::Button("Cancel", ImVec2(80, 0))) {
        visible_ = false;
    }

    // Check if all done
    bool all_done = true;
    for (const auto& f : pending_files_) {
        if (f.selected && f.state != PendingFile::State::Complete) {
            all_done = false;
            break;
        }
    }

    if (!pending_files_.empty() && all_done) {
        ImGui::SameLine();
        if (ImGui::Button(ICON_FA_CHECK " Done", ImVec2(80, 0))) {
            if (on_complete_ && !completed_file_ids_.empty()) {
                on_complete_(completed_file_ids_);
            }
            visible_ = false;
        }
    }
}

void CloudUploadDialog::AddFile(const std::string& path) {
    std::lock_guard<std::mutex> lock(files_mutex_);

    // Check if already added
    for (const auto& f : pending_files_) {
        if (f.path == path) return;
    }

    // Get file info
    std::error_code ec;
    auto file_size = std::filesystem::file_size(path, ec);
    if (ec) {
        last_error_ = "Failed to get file size: " + path;
        error_time_ = 5.0f;
        return;
    }

    PendingFile file;
    file.path = path;
    file.filename = std::filesystem::path(path).filename().string();
    file.size_bytes = static_cast<int64_t>(file_size);
    file.selected = true;
    file.state = PendingFile::State::Pending;

    pending_files_.push_back(std::move(file));
}

void CloudUploadDialog::AddFiles(const std::vector<std::string>& paths) {
    for (const auto& path : paths) {
        AddFile(path);
    }
}

void CloudUploadDialog::RemoveFile(size_t index) {
    std::lock_guard<std::mutex> lock(files_mutex_);
    if (index < pending_files_.size()) {
        pending_files_.erase(pending_files_.begin() + index);
    }
}

void CloudUploadDialog::ClearState() {
    std::lock_guard<std::mutex> lock(files_mutex_);
    pending_files_.clear();
    completed_file_ids_.clear();
    current_upload_index_ = 0;
    last_error_.clear();
    error_time_ = 0;
}

void CloudUploadDialog::StartUpload() {
    if (!datastream_client_ || is_uploading_.load()) return;

    is_uploading_.store(true);
    should_cancel_.store(false);
    current_upload_index_ = 0;
    completed_file_ids_.clear();

    // Find first selected file
    UploadNextFile();
}

void CloudUploadDialog::UploadNextFile() {
    // Find next file to upload
    size_t next_index = current_upload_index_;
    {
        std::lock_guard<std::mutex> lock(files_mutex_);
        while (next_index < pending_files_.size()) {
            if (pending_files_[next_index].selected &&
                pending_files_[next_index].state == PendingFile::State::Pending) {
                break;
            }
            next_index++;
        }

        if (next_index >= pending_files_.size()) {
            // All done
            is_uploading_.store(false);
            return;
        }

        current_upload_index_ = next_index;
        pending_files_[next_index].state = PendingFile::State::Uploading;
    }

    // Upload in background thread
    std::thread([this, next_index]() {
        PendingFile* file = nullptr;
        {
            std::lock_guard<std::mutex> lock(files_mutex_);
            if (next_index < pending_files_.size()) {
                file = &pending_files_[next_index];
            }
        }

        if (file && !should_cancel_.load()) {
            UploadFile(*file);
        }

        if (!should_cancel_.load()) {
            current_upload_index_ = next_index + 1;
            UploadNextFile();
        } else {
            is_uploading_.store(false);
        }
    }).detach();
}

void CloudUploadDialog::UploadFile(PendingFile& file) {
    if (!datastream_client_) return;

    std::string file_id;
    bool success = datastream_client_->UploadFile(
        file.path,
        file_id,
        [&file](int64_t sent, int64_t total) {
            file.progress = static_cast<float>(sent) / static_cast<float>(total);
        }
    );

    if (success) {
        file.state = PendingFile::State::Complete;
        file.file_id = file_id;
        completed_file_ids_.push_back(file_id);
    } else {
        file.state = PendingFile::State::Error;
        file.error = datastream_client_->GetLastError();
    }
}

void CloudUploadDialog::ShowFileBrowser() {
#ifdef _WIN32
    OPENFILENAMEA ofn = {};
    char szFile[4096] = "";

    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = nullptr;
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.lpstrFilter = "All Files\0*.*\0Image Files\0*.png;*.jpg;*.jpeg\0Data Files\0*.csv;*.json;*.h5\0";
    ofn.nFilterIndex = 1;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_ALLOWMULTISELECT | OFN_EXPLORER;

    if (GetOpenFileNameA(&ofn)) {
        // Parse multi-select result
        std::vector<std::string> files;
        char* p = szFile;
        std::string directory = p;
        p += directory.length() + 1;

        if (*p == '\0') {
            // Single file selected
            files.push_back(directory);
        } else {
            // Multiple files selected
            while (*p != '\0') {
                std::string filename = p;
                files.push_back(directory + "\\" + filename);
                p += filename.length() + 1;
            }
        }

        AddFiles(files);
    }
#else
    // Linux/macOS: Use portable file dialog library or system call
    // For now, just show an input field
    show_file_browser_ = true;
#endif
}

} // namespace gui
