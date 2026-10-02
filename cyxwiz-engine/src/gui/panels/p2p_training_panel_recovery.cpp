// P2P Training panel: the failure category card and "A checkpoint can
// continue this job" (TOFIX118 P4 GUI). The words come from
// core/remote_job_presentation; resume sends the same job again, and the node
// continues from its checkpoint (P4e-3).
#include "p2p_training_panel.h"

#include "../icons.h"
#include "../ui_buttons.h"

#include <imgui.h>

#include <algorithm>
#include <ctime>

namespace cyxwiz {

std::pair<int, int> P2PTrainingPanel::GetEpochPosition() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return {static_cast<int>(current_epoch_), static_cast<int>(total_epochs_)};
}

void P2PTrainingPanel::OnFailure(const network::TrainingFailureReport& report) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    has_failure_ = true;
    failure_card_ = BuildFailureCardFromCode(report.code, report.message, current_batch_);
    AddLogEntry("ERROR", failure_card_.title + ": " + report.message);
}

void P2PTrainingPanel::RenderRecoveryCards() {
    bool show_failure = false;
    bool show_resume = false;
    RemoteFailureCard failure;
    std::string resume_summary;
    std::string job_id;
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        show_failure = has_failure_;
        failure = failure_card_;
        job_id = job_id_;
        show_resume = (has_failure_ || stopped_by_user_) && !checkpoint_history_.empty() && resume_action_;
        if (show_resume) {
            const auto& last = checkpoint_history_.back();
            RemoteCheckpointRow row;
            row.epoch = static_cast<int>(last.epoch);
            row.next_batch = last.next_batch;
            row.total_batches = last.total_batches;
            const auto time_value = std::chrono::system_clock::to_time_t(last.timestamp);
            char received[16] = {};
            std::strftime(received, sizeof(received), "%H:%M:%S", std::localtime(&time_value));
            row.received = received;
            resume_summary = ResumeSummary(row);
        }
    }
    if (!show_failure && !show_resume) return;

    const ImVec4 muted = ImGui::GetStyle().Colors[ImGuiCol_TextDisabled];
    const ImVec4 failed(0.96f, 0.63f, 0.29f, 1.0f);
    const auto card = [](const char* id) {
        return ImGui::BeginChild(id, ImVec2(0.0f, 0.0f),
                                 ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY |
                                     ImGuiChildFlags_AlwaysUseWindowPadding);
    };

    if (show_failure) {
        if (card("##failure")) {
            ImGui::TextColored(failed, ICON_FA_TRIANGLE_EXCLAMATION);
            ImGui::SameLine();
            ImGui::TextUnformatted(failure.title.c_str());
            if (!failure.when.empty()) {
                ImGui::SameLine();
                const float width = ImGui::CalcTextSize(failure.when.c_str()).x;
                ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), ImGui::GetContentRegionMax().x - width));
                ImGui::TextColored(muted, "%s", failure.when.c_str());
            }
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextUnformatted(failure.message.c_str());
            ImGui::TextColored(muted, "%s", failure.next.c_str());
            ImGui::PopTextWrapPos();
        }
        ImGui::EndChild();
    }

    bool resume = false;
    bool start_over = false;
    if (show_resume) {
        if (card("##resume")) {
            ImGui::TextUnformatted("A checkpoint can continue this job");
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(muted, "%s", resume_summary.c_str());
            ImGui::PopTextWrapPos();
            resume = ui::PrimaryButton("Resume from checkpoint");
            if (start_over_action_) {
                ImGui::SameLine();
                start_over = ui::SecondaryButton("Start over");
            }
        }
        ImGui::EndChild();
    }
    ImGui::Separator();

    // Outside the lock: these start a new send, which restarts monitoring.
    if (resume && resume_action_) resume_action_(job_id);
    if (start_over && start_over_action_) start_over_action_();
}

}  // namespace cyxwiz
