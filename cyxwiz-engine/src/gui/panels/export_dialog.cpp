#include "export_dialog.h"
#include "../icons.h"
#include "../../core/file_dialogs.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <cyxwiz/sequential.h>
#include <cyxwiz/optimizer.h>
#include "../../core/training_executor.h"
#include "../ui_buttons.h"
#include "../ui_platform.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"
#include "../../core/guarded_extension_module.h"
#include "../../core/model_export_presentation.h"
#include "../../core/project_manager.h"
#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <filesystem>

namespace cyxwiz {

namespace {
#ifdef CYXWIZ_HAS_ONNX_EXPORT
constexpr bool kOnnxBuilt = true;
#else
constexpr bool kOnnxBuilt = false;
#endif
}  // namespace

ExportDialog::ExportDialog()
    : Panel("Export Model", false) {
}

ExportDialog::~ExportDialog() {
    // Wait for export thread to finish
    if (export_thread_ && export_thread_->joinable()) {
        export_thread_->join();
    }
}

void ExportDialog::Open(ModelFormat initial_format) {
    is_open_ = true;
    // Reopened during an export: its progress stays on screen.
    if (is_exporting_ || completion_pending_.load()) return;
    show_result_ = false;
    export_progress_ = 0;
    export_total_ = 0;
    selected_format_ = model_export::IsAvailable(initial_format, kOnnxBuilt, OnnxWritable()) ? initial_format
                                                                                         : ModelFormat::CyxModel;
    include_training_history_ = true;
    include_graph_ = true;
    const std::string path =
        model_export::DefaultPath(ProjectManager::Instance().GetProjectRoot(), graph_path_, selected_format_);
    std::snprintf(output_path_, sizeof(output_path_), "%s", path.c_str());
    const std::string name = model_export::DefaultName(trained_metadata_.model_name, graph_path_);
    std::snprintf(model_name_, sizeof(model_name_), "%s", name.c_str());
    std::snprintf(author_, sizeof(author_), "%s", trained_metadata_.author.c_str());
    std::snprintf(description_, sizeof(description_), "%s", trained_metadata_.description.c_str());
}

void ExportDialog::Close() {
    is_open_ = false;
}

void ExportDialog::SetModelData(
    std::shared_ptr<IExecutableModel> model,
    const Optimizer* optimizer,
    const TrainingMetrics* metrics,
    const std::string& graph_json,
    uint64_t graph_hash,
    const ExportOptions& trained_metadata,
    std::optional<ProcessDeviceSelection> model_device
) {
    model_ = std::move(model);
    model_device_ = model_device;
    optimizer_ = optimizer;
    metrics_ = metrics;
    graph_json_ = graph_json;
    graph_hash_ = graph_hash;
    trained_metadata_ = trained_metadata;
}

bool ExportDialog::LayerStack() const {
    return !model_ || model_->AsSequentialModel() != nullptr;
}

// ONNX is written from a stack of built-in layers: not for a graph with
// branches, nor when a layer is an extension node, whose computation is the
// extension's own code (TOFIX125 P6).
bool ExportDialog::OnnxWritable() const {
    if (!model_) return true;
    const SequentialModel* stack = model_->AsSequentialModel();
    if (!stack) return false;
    for (size_t i = 0; i < stack->Size(); ++i)
        if (dynamic_cast<const GuardedExtensionModule*>(stack->GetModule(i))) return false;
    return true;
}

void ExportDialog::Render() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (completion_pending_.exchange(false)) {
        show_result_ = true;
        if (export_complete_callback_) {
            export_complete_callback_(
                last_result_,
                completed_export_options_,
                completed_graph_hash_,
                completed_graph_json_,
                completed_optimizer_available_,
                completed_metrics_available_);
        }
    }
    if (!is_open_) return;

    ImGui::SetNextWindowSize(ImVec2(560, 460), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));
    if (ImGui::Begin("Export model###export_model", &is_open_, ImGuiWindowFlags_NoCollapse)) {
        if (is_exporting_) {
            RenderProgress();
        } else if (show_result_) {
            const bool ok = last_result_.success;
            ImGui::TextColored(ok ? t.success : t.error,
                               ok ? ICON_FA_CIRCLE_CHECK " Exported" : ICON_FA_CIRCLE_XMARK " The export failed");
            ImGui::Spacing();
            if (ok) {
                std::vector<ui::KeyValue> rows;
                for (const auto& [key, value] : model_export::ResultRows(last_result_)) rows.push_back({key, value});
                ui::FactsTable("##export_result", rows);
            } else {
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TextUnformatted(last_result_.error_message.c_str());
                ImGui::PopTextWrapPos();
            }
            ImGui::Spacing();
            if (ok && ui::SecondaryButton("Show in folder", true, nullptr, ui::ButtonSize::Small))
                ui::ShowInFileManager(last_result_.output_path);
            const float right = ui::ButtonWidth("Close", ui::ButtonSize::Regular) + ImGui::GetStyle().ItemSpacing.x +
                                ui::ButtonWidth(ok ? "Export another" : "Try again", ui::ButtonSize::Regular);
            ui::SameLineRight(right);
            if (ui::SecondaryButton(ok ? "Export another" : "Try again", true, nullptr, ui::ButtonSize::Regular))
                show_result_ = false;
            ImGui::SameLine();
            if (ui::PrimaryButton("Close")) Close();
        } else if (!model_) {
            ImGui::TextColored(t.warning, ICON_FA_TRIANGLE_EXCLAMATION " No trained model");
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(t.text_dim, "Train the graph first; the trained model is what gets exported.");
            ImGui::PopTextWrapPos();
            ImGui::Spacing();
            if (ui::PrimaryButton("Close")) Close();
        } else {
            const float footer_h = ui::ButtonHeight(ui::ButtonSize::Regular) + ImGui::GetStyle().ItemSpacing.y * 2.0f;
            ImGui::BeginChild("##export_form", ImVec2(0.0f, -footer_h));
            RenderFormatSelection();
            RenderOptions();
            RenderMetadata();
            ImGui::EndChild();
            RenderButtons();
        }
    }
    ImGui::End();
}

void ExportDialog::RenderFormatSelection() {
    const ui::Tokens& t = ui::CurrentTokens();
    const auto choices = model_export::AvailableFormats(kOnnxBuilt, OnnxWritable());
    std::vector<const char*> labels;
    int selected = 0;
    for (size_t i = 0; i < choices.size(); ++i) {
        labels.push_back(choices[i].label.c_str());
        if (choices[i].format == selected_format_) selected = static_cast<int>(i);
    }
    ImGui::TextColored(t.text_dim, "Format");
    if (ui::SegmentedControl("##export_format", labels.data(), static_cast<int>(labels.size()), &selected)) {
        selected_format_ = choices[static_cast<size_t>(selected)].format;
        const std::string path = model_export::WithExtension(output_path_, selected_format_);
        std::snprintf(output_path_, sizeof(output_path_), "%s", path.c_str());
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim, "%s", choices[static_cast<size_t>(selected)].description.c_str());
    ImGui::PopTextWrapPos();
    ImGui::Spacing();
}

void ExportDialog::RenderOptions() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::TextColored(t.text_dim, "File");
    const float browse = ui::ButtonWidth("Browse...", ui::ButtonSize::Small);
    const float field = ImGui::GetContentRegionAvail().x - browse - ImGui::GetStyle().ItemSpacing.x;
    ImGui::SetNextItemWidth((std::max)(120.0f, field));
    ImGui::InputText("##output_path", output_path_, sizeof(output_path_));
    ImGui::SameLine();
    if (ui::SecondaryButton("Browse...", true, nullptr, ui::ButtonSize::Small)) {
        const std::filesystem::path current(output_path_);
        const std::string folder = current.has_parent_path() ? current.parent_path().string() : std::string();
        const std::string name = current.has_filename() ? current.filename().string()
                                                        : "model" + GetFormatExtension(selected_format_);
        const std::string extension = GetFormatExtension(selected_format_).substr(1);
        const auto path = FileDialogs::SaveFile("Export model", {{extension + " files", extension}, {"All Files", "*"}},
                                                folder.empty() ? nullptr : folder.c_str(), name.c_str());
        if (path) {
            const std::string chosen = model_export::WithExtension(*path, selected_format_);
            std::snprintf(output_path_, sizeof(output_path_), "%s", chosen.c_str());
        }
    }
    ImGui::Spacing();
    if (selected_format_ == ModelFormat::CyxModel) {
        ImGui::TextColored(t.text_dim, "Include");
        // A graph with branches is rebuilt from its graph: always included.
        if (!LayerStack()) include_graph_ = true;
        ImGui::BeginDisabled(!LayerStack());
        ImGui::Checkbox("Graph", &include_graph_);
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip(LayerStack() ? "The node graph, to open and edit it in CyxWiz."
                                           : "Always included: a graph with branches is rebuilt from its graph.");
        ImGui::SameLine();
        ImGui::Checkbox("Training history", &include_training_history_);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Loss and accuracy per epoch.");
    } else if (selected_format_ == ModelFormat::ONNX) {
        ImGui::TextColored(t.text_dim, "Opset version");
        ImGui::SetNextItemWidth(100);
        ImGui::InputInt("##opset", &opset_version_);
        opset_version_ = std::clamp(opset_version_, 1, 20);
    }
    ImGui::Spacing();
}

void ExportDialog::RenderMetadata() {
    const ui::Tokens& t = ui::CurrentTokens();
    const float width = ImGui::GetContentRegionAvail().x;
    ImGui::TextColored(t.text_dim, "Name");
    ImGui::SetNextItemWidth(width);
    ImGui::InputText("##model_name", model_name_, sizeof(model_name_));
    ImGui::TextColored(t.text_dim, "Author (optional)");
    ImGui::SetNextItemWidth(width);
    ImGui::InputText("##author", author_, sizeof(author_));
    ImGui::TextColored(t.text_dim, "Description (optional)");
    ImGui::InputTextMultiline("##description", description_, sizeof(description_), ImVec2(width, 60));
}

void ExportDialog::RenderProgress() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::TextColored(t.text, ICON_FA_SPINNER " Exporting");
    const int progress = export_progress_.load();
    const int total = export_total_.load();
    ImGui::ProgressBar(total > 0 ? static_cast<float>(progress) / total : 0.0f, ImVec2(-FLT_MIN, 0));
    std::lock_guard<std::mutex> lock(status_mutex_);
    ImGui::TextColored(t.text_dim, "%s", export_status_.c_str());
}

void ExportDialog::RenderButtons() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (model_) ImGui::TextColored(t.text_dim, "%zu layers%s", model_->LayerCount(), LayerStack() ? "" : ", with branches");
    const bool can_export = output_path_[0] != '\0' && model_ && !is_exporting_ && !completion_pending_.load();
    const float right = ui::ButtonWidth("Cancel", ui::ButtonSize::Regular) + ImGui::GetStyle().ItemSpacing.x +
                        ui::ButtonWidth("Export", ui::ButtonSize::Regular);
    if (model_) ui::SameLineRight(right);
    if (ui::SecondaryButton("Cancel", true, nullptr, ui::ButtonSize::Regular)) Close();
    ImGui::SameLine();
    if (ui::PrimaryButton("Export", can_export, "Choose a file to export to")) StartExport();
}

void ExportDialog::StartExport() {
    if (!model_) return;

    is_exporting_ = true;
    show_result_ = false;
    export_progress_ = 0;
    export_total_ = 6;

    // Keep the completed run snapshot; UI controls only change inclusion/format.
    export_options_ = trained_metadata_;
    // Build export options
    export_options_.format = selected_format_;
    // The package does not carry optimizer state yet (no optimizer state API).
    export_options_.include_optimizer_state = false;
    export_options_.include_training_history = include_training_history_;
    export_options_.include_graph = include_graph_;
    export_options_.compress = false;  // not implemented for CYXW v3
    export_options_.onnx_opset_version = opset_version_;
    export_options_.quantization = Quantization::None;
    // Fields left empty keep what the trained run recorded.
    if (model_name_[0]) export_options_.model_name = model_name_;
    if (author_[0]) export_options_.author = author_;
    if (description_[0]) export_options_.description = description_;

    // Wait for any previous export thread
    if (export_thread_ && export_thread_->joinable()) {
        export_thread_->join();
    }

    std::shared_ptr<IExecutableModel> model = model_;
    const Optimizer* optimizer = optimizer_;
    const TrainingMetrics* metrics = metrics_;
    const std::string graph_json = graph_json_;
    const std::string output_path = output_path_;
    const uint64_t graph_hash = graph_hash_;
    const ExportOptions options = export_options_;
    const auto model_device = model_device_;

    // Start export in background thread
    export_thread_ = std::make_unique<std::thread>(
        [this, model, optimizer, metrics, graph_json, output_path, graph_hash,
         options, model_device]() {
        ModelExporter exporter;

        auto progress_callback = [this](int current, int total, const std::string& status) {
            export_progress_ = current;
            export_total_ = total;
            {
                std::lock_guard<std::mutex> lock(status_mutex_);
                export_status_ = status;
            }
        };

        // A new thread starts on ArrayFire's default backend, which can
        // differ from the one holding the weights (OpenCL training with a
        // oneAPI default); reading them there fails with error 503.
        ExportResult result;
        bool device_ready = true;
        if (model_device) {
            const auto activation =
                Device(model_device->type, model_device->device_id)
                    .ActivateExact(false);
            if (!activation.success) {
                device_ready = false;
                result.success = false;
                result.error_message =
                    "Could not switch export to the device that holds the "
                    "trained model: " + activation.message;
            }
        }
        if (device_ready) {
            result = exporter.Export(
                *model,
                optimizer,
                metrics,
                graph_json,
                output_path,
                options,
                progress_callback
            );
        }

        last_result_ = result;
        completed_export_options_ = options;
        completed_graph_hash_ = graph_hash;
        completed_graph_json_ = graph_json;
        completed_optimizer_available_ = optimizer != nullptr;
        completed_metrics_available_ = metrics != nullptr;
        completion_pending_ = true;
        is_exporting_ = false;

        if (result.success) {
            spdlog::info("Model exported successfully to {}", result.output_path);
        } else {
            spdlog::error("Model export failed: {}", result.error_message);
        }
    });
}

} // namespace cyxwiz
