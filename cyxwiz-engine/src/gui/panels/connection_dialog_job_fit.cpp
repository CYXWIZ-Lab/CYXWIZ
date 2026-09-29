// Server Connection: "Will this job fit on the node?" (TOFIX118 P4 GUI).
// The Engine measures the job once (one training step on this machine,
// ProbeGraphJobMemory) and compares it with the reserved node's device using
// the node's own admission rule; the node still checks before it starts.
#include "connection_dialog.h"

#include "../icons.h"
#include "../node_editor.h"
#include "../ui_buttons.h"

#include <cyxwiz/device.h>
#include <imgui.h>

#include <algorithm>
#include <functional>

namespace cyxwiz {

namespace {

const ImVec4 kVerified(0.24f, 0.84f, 0.55f, 1.0f);
const ImVec4 kFailed(0.96f, 0.63f, 0.29f, 1.0f);
const ImVec4 kPending(0.70f, 0.65f, 1.00f, 1.0f);
const ImVec4 kNotYet(0.64f, 0.69f, 0.78f, 1.0f);

ImVec4 KindColor(RemoteStatusKind kind) {
    switch (kind) {
        case RemoteStatusKind::Ok: return kVerified;
        case RemoteStatusKind::Failed: return kFailed;
        case RemoteStatusKind::Pending: return kPending;
        case RemoteStatusKind::None: break;
    }
    return kNotYet;
}

}  // namespace

void ConnectionDialog::UpdateJobEstimate() {
    if (estimate_future_.valid() &&
        estimate_future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        estimate_result_ = estimate_future_.get();
        estimate_state_ = estimate_result_.ok ? JobEstimateState::Ready : JobEstimateState::Failed;
    }
    if (!node_editor_ || estimate_future_.valid()) return;

    // Serializing the graph every frame is wasteful; once a second is enough
    // to notice an edit.
    const double now = ImGui::GetTime();
    if (now < estimate_next_check_) return;
    estimate_next_check_ = now + 1.0;

    std::string graph_json = node_editor_->GetGraphJson();
    if (graph_json.empty() || graph_json == "{}") return;
    const std::size_t key = std::hash<std::string>{}(graph_json) ^ (static_cast<std::size_t>(reservation_batch_size_) << 1);
    if (key == estimate_key_ && estimate_state_ != JobEstimateState::NotStarted) return;

    estimate_key_ = key;
    estimate_state_ = JobEstimateState::Running;
    last_rejection_reason_.clear();  // a changed job deserves a fresh check
    if (const auto selected = Device::GetProcessDevice()) {
        estimate_device_ = Device(selected->type, selected->device_id).GetInfo().name;
    }
    GraphTrainingJobRequest request;
    request.graph_json = std::move(graph_json);
    request.batch_size_override = reservation_batch_size_;
    estimate_future_ = std::async(std::launch::async, [request]() { return ProbeGraphJobMemory(request); });
}

JobFitCard ConnectionDialog::CurrentJobFitCard() const {
    JobFitInputs inputs;
    inputs.estimate = estimate_state_;
    if (estimate_state_ == JobEstimateState::Ready) {
        inputs.training_bytes = static_cast<std::uint64_t>(std::max<long long>(0, estimate_result_.training_bytes));
    }
    inputs.estimate_backend = estimate_result_.backend;
    inputs.estimate_device = estimate_device_;
    inputs.estimate_error = estimate_result_.error;
    inputs.node_device = reserved_node_.device_type;
    inputs.node_device_bytes = static_cast<std::uint64_t>(std::max<long long>(0, reserved_node_.vram_bytes));
    inputs.rejection_reason = last_rejection_reason_;
    return BuildJobFitCard(inputs);
}

void ConnectionDialog::RenderJobFitCard() {
    const JobFitCard card = CurrentJobFitCard();
    ImGui::PushID("job_fit");
    if (ImGui::BeginChild("##card", ImVec2(0.0f, 0.0f),
                          ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY |
                              ImGuiChildFlags_AlwaysUseWindowPadding)) {
        const ImVec4 muted = ImGui::GetStyle().Colors[ImGuiCol_TextDisabled];
        ImGui::TextUnformatted("Will this job fit on the node?");
        ImGui::SameLine();
        const float status_width = ImGui::CalcTextSize(card.status.c_str()).x;
        ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), ImGui::GetContentRegionMax().x - status_width));
        ImGui::TextColored(KindColor(card.kind), "%s", card.status.c_str());
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(muted, "%s", card.note.c_str());
        ImGui::PopTextWrapPos();

        if (card.has_bars) {
            if (ImGui::BeginTable("##fit", 2, ImGuiTableFlags_SizingStretchProp)) {
                const auto row = [&](const char* key, const std::string& value) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextColored(muted, "%s", key);
                    ImGui::TableNextColumn();
                    ImGui::PushTextWrapPos(0.0f);
                    ImGui::TextUnformatted(value.c_str());
                    ImGui::PopTextWrapPos();
                };
                row("This job needs", card.needs);
                row(card.device_label.c_str(), card.has);
                ImGui::EndTable();
            }
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, KindColor(card.kind));
            ImGui::ProgressBar(static_cast<float>(card.fill), ImVec2(-1.0f, 8.0f), "");
            ImGui::PopStyleColor();
        }
        if (card.has_reason) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(kFailed, "%s", card.reason.c_str());
            ImGui::TextColored(muted, "%s", card.next.c_str());
            ImGui::PopTextWrapPos();
        }
        if (ui::LinkButton(fit_details_open_ ? "Hide##fit" : "Details##fit")) {
            fit_details_open_ = !fit_details_open_;
        }
        if (fit_details_open_ && ImGui::BeginTable("##fit_details", 2, ImGuiTableFlags_SizingStretchProp)) {
            for (const auto& [key, value] : card.details) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(muted, "%s", key.c_str());
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(value.c_str());
            }
            ImGui::EndTable();
        }
    }
    ImGui::EndChild();
    ImGui::PopID();
}

}  // namespace cyxwiz
