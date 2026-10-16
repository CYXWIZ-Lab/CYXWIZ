#include "import_dialog.h"
#include "../icons.h"
#include "../../core/file_dialogs.h"
#include "../../core/model_importer.h"
#include "../../core/model_open_presentation.h"
#include "../../core/project_manager.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstdio>
#include <filesystem>

namespace cyxwiz {

ImportDialog::ImportDialog()
    : Panel("Open Model File", false) {
}

ImportDialog::~ImportDialog() {
    for (auto& [thread, job] : workers_) {
        if (thread.joinable()) thread.join();
    }
}

void ImportDialog::Open() {
    is_open_ = true;
    // Reopened while a file is being read: that read stays on screen.
    if (job_) return;
    input_path_[0] = '\0';
    probed_path_.clear();
    probe_ = ProbeResult{};
    error_.clear();
}

void ImportDialog::Close() {
    is_open_ = false;
}

void ImportDialog::StartJob(const std::string& path, bool extract) {
    auto job = std::make_shared<Job>();
    job->path = path;
    job->extract = extract;
    error_.clear();
    job_ = job;
    workers_.emplace_back(std::thread([job]() {
        ModelImporter importer;
        if (job->extract) {
            job->graph_json = importer.ExtractGraph(job->path);
        } else {
            job->probe = importer.ProbeFile(job->path);
        }
        job->done = true;
    }), job);
}

void ImportDialog::CollectJob() {
    // Join the workers that finished (the current one and any replaced one).
    for (auto it = workers_.begin(); it != workers_.end();) {
        if (it->second->done.load()) {
            it->first.join();
            it = workers_.erase(it);
        } else {
            ++it;
        }
    }
    if (!job_ || !job_->done.load()) return;
    const std::shared_ptr<Job> job = std::move(job_);
    job_.reset();
    if (!job->extract) {
        probed_path_ = job->path;
        probe_ = job->probe;
        return;
    }
    if (!job->graph_json || job->graph_json->empty()) {
        error_ = "The graph in this file could not be read.";
        spdlog::error("Open Model File: no graph could be extracted from {}", job->path);
        return;
    }
    const bool opened = open_graph_callback_ && open_graph_callback_(probe_, *job->graph_json, job->path);
    if (opened) {
        Close();
    } else {
        error_ = "Studio could not open the graph in this file; the log has the details.";
    }
}

void ImportDialog::Render() {
    CollectJob();
    if (!is_open_) return;

    ImGui::SetNextWindowSize(ImVec2(600, 440), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));
    if (ImGui::Begin("Open a model file###open_model_file", &is_open_, ImGuiWindowFlags_NoCollapse)) {
        const float footer_h = ui::ButtonHeight(ui::ButtonSize::Regular) + ImGui::GetStyle().ItemSpacing.y * 2.0f;
        ImGui::BeginChild("##open_model_body", ImVec2(0.0f, -footer_h));
        RenderFileField();
        RenderFile();
        ImGui::EndChild();
        RenderButtons();
    }
    ImGui::End();
}

void ImportDialog::RenderFileField() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::TextColored(t.text_dim, "File");
    const float browse = ui::ButtonWidth("Browse...", ui::ButtonSize::Small);
    const float field = ImGui::GetContentRegionAvail().x - browse - ImGui::GetStyle().ItemSpacing.x;
    ImGui::SetNextItemWidth((std::max)(120.0f, field));
    ImGui::InputTextWithHint("##input_path", "A .cyxmodel, .safetensors, .onnx or .gguf file", input_path_,
                             sizeof(input_path_));
    // A typed path is read when the field is left or Enter is pressed, not at every keystroke.
    const bool typed = ImGui::IsItemDeactivatedAfterEdit();
    ImGui::SameLine();
    if (ui::SecondaryButton("Browse...", true, nullptr, ui::ButtonSize::Small)) {
        const std::string root = ProjectManager::Instance().GetProjectRoot();
        const std::string exports = root.empty() ? std::string() : (std::filesystem::path(root) / "exports").string();
        const std::string folder = std::filesystem::is_directory(exports) ? exports : root;
        const auto path = FileDialogs::OpenFile("Open a model file",
                                                {{"Model files", "cyxmodel,safetensors,onnx,gguf"},
                                                 {"CyxWiz package", "cyxmodel"},
                                                 {"Safetensors", "safetensors"},
                                                 {"ONNX", "onnx"},
                                                 {"GGUF", "gguf"},
                                                 {"All Files", "*"}},
                                                folder.empty() ? nullptr : folder.c_str());
        if (path) {
            std::snprintf(input_path_, sizeof(input_path_), "%s", path->c_str());
            StartJob(*path, false);
        }
    } else if (typed && input_path_[0] != '\0' && !job_) {
        StartJob(input_path_, false);
    }
    ImGui::Spacing();
}

void ImportDialog::RenderFile() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (job_) {
        ImGui::TextColored(t.text_dim, job_->extract ? ICON_FA_SPINNER " Opening the graph"
                                                     : ICON_FA_SPINNER " Reading the file");
        return;
    }
    if (probed_path_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim, "Choose a model file to see what it holds and open its graph.");
        ImGui::PopTextWrapPos();
        return;
    }
    const model_open::View v = model_open::BuildView(probe_);
    ImGui::PushTextWrapPos(0.0f);
    if (!v.readable) {
        ImGui::TextColored(t.error, ICON_FA_CIRCLE_XMARK " This file could not be read");
        ImGui::TextUnformatted(v.error.c_str());
        ImGui::PopTextWrapPos();
        return;
    }
    ImGui::TextColored(t.text, "%s", v.format_line.c_str());
    ImGui::PopTextWrapPos();
    ImGui::Spacing();
    std::vector<ui::KeyValue> rows;
    for (const auto& [key, value] : v.rows) rows.push_back({key, value});
    ui::FactsTable("##open_model_rows", rows);
    if (!v.layers.empty()) {
        ImGui::Spacing();
        ImGui::TextColored(t.text_dim, "Layers and shapes");
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextUnformatted(v.layers.c_str());
        ImGui::PopTextWrapPos();
    }
    ImGui::Spacing();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim, "%s", v.note.c_str());
    if (!error_.empty()) ImGui::TextColored(t.error, ICON_FA_TRIANGLE_EXCLAMATION " %s", error_.c_str());
    ImGui::PopTextWrapPos();
}

void ImportDialog::RenderButtons() {
    const model_open::View v = model_open::BuildView(probe_);
    const bool ready = !job_ && !probed_path_.empty();
    const bool can_open = ready && v.can_open_graph;
    const char* reason = job_ ? "The file is being read"
                         : probed_path_.empty() ? "Choose a model file"
                                                : v.open_disabled_reason.c_str();
    const float right = ui::ButtonWidth("Cancel", ui::ButtonSize::Regular) + ImGui::GetStyle().ItemSpacing.x +
                        ui::ButtonWidth("Open its graph", ui::ButtonSize::Regular);
    ImGui::SetCursorPosX((std::max)(ImGui::GetCursorPosX(), ImGui::GetContentRegionMax().x - right));
    if (ui::SecondaryButton("Cancel", true, nullptr, ui::ButtonSize::Regular)) Close();
    ImGui::SameLine();
    if (ui::PrimaryButton("Open its graph", can_open, reason)) StartJob(probed_path_, true);
}

} // namespace cyxwiz
