// Server Connection: the reservation card (TOFIX118 gaps 5-6, mockup approved
// 2026-09-30). Words and numbers come from core/reservation_presentation; this
// file only draws them and routes the actions.
#include "connection_dialog.h"

#include "../icons.h"
#include "../ui_buttons.h"
#include "auth/auth_client.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <ctime>

namespace cyxwiz {

namespace {

const ImVec4 kVerified(0.24f, 0.84f, 0.55f, 1.0f);
const ImVec4 kFailed(0.96f, 0.63f, 0.29f, 1.0f);
const ImVec4 kNotYet(0.64f, 0.69f, 0.78f, 1.0f);
const ImVec4 kDanger(1.00f, 0.48f, 0.45f, 1.0f);

ImVec4 Muted() { return ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]; }

ImVec4 UrgencyColor(ReservationUrgency urgency) {
    switch (urgency) {
        case ReservationUrgency::Normal: return kVerified;
        case ReservationUrgency::Soon: return kFailed;
        case ReservationUrgency::Urgent:
        case ReservationUrgency::Ended: return kDanger;
    }
    return kVerified;
}

bool BeginCard(const char* id) {
    return ImGui::BeginChild(id, ImVec2(0.0f, 0.0f),
                             ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY |
                                 ImGuiChildFlags_AlwaysUseWindowPadding);
}

// Right-aligned text on the current line.
void RightText(const ImVec4& color, const std::string& text) {
    ImGui::SameLine();
    const float width = ImGui::CalcTextSize(text.c_str()).x;
    ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), ImGui::GetContentRegionMax().x - width));
    ImGui::TextColored(color, "%s", text.c_str());
}

// Key/value rows, `pairs_per_line` pairs side by side.
void KeyValueTable(const char* id, const ReservationDetails& rows, int pairs_per_line) {
    if (rows.empty() || !ImGui::BeginTable(id, pairs_per_line * 2, ImGuiTableFlags_SizingStretchProp)) return;
    for (size_t i = 0; i < rows.size(); ++i) {
        if (i % static_cast<size_t>(pairs_per_line) == 0) ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextColored(Muted(), "%s", rows[i].first.c_str());
        ImGui::TableNextColumn();
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextUnformatted(rows[i].second.c_str());
        ImGui::PopTextWrapPos();
    }
    ImGui::EndTable();
}

}  // namespace

ReservationNodeFacts ConnectionDialog::FactsFor(const network::NodeDisplayInfo& node) {
    ReservationNodeFacts facts;
    facts.name = node.name;
    facts.device_type = node.device_type;
    facts.vram_bytes = node.vram_bytes;
    facts.reputation = node.reputation_score;
    facts.jobs_completed = node.total_jobs_completed;
    facts.region = node.region;
    facts.online = node.is_online;
    facts.price_usd_per_hour = node.price_usd_equivalent;
    facts.free_tier = node.free_tier_available;
    return facts;
}

ActiveReservationInputs ConnectionDialog::CurrentReservationInputs() const {
    ActiveReservationInputs in;
    in.node = FactsFor(reserved_node_);
    in.node_known = !reserved_node_.node_id.empty();
    in.endpoint = active_reservation_.node_endpoint;
    in.now = static_cast<long long>(std::time(nullptr));
    in.start_time = active_reservation_.start_time;
    in.end_time = active_reservation_.end_time;
    in.server_seconds_left = reservation_seconds_left_.load();
    in.server_checked_at = reservation_heartbeat_at_.load();
    in.p2p_connected = p2p_client_ && p2p_client_->IsConnected();
    in.training_running = p2p_client_ && p2p_client_->IsStreaming() && !p2p_client_->IsWaitingForNewJob();
    if (p2p_training_panel_) {
        const auto [epoch, total] = p2p_training_panel_->GetEpochPosition();
        in.epoch = epoch;
        in.total_epochs = total;
        in.current_job_id = p2p_training_panel_->GetJobId();
    }
    in.reservation_id = active_reservation_.reservation_id;
    in.node_id = reserved_node_.node_id;
    in.access_until = active_reservation_.p2p_token_expires;
    in.epochs_setting = reservation_epochs_;
    in.batch_setting = reservation_batch_size_;
    return in;
}

void ConnectionDialog::RenderReservationError() {
    if (reservation_error_.empty()) return;
    ImGui::PushID("reservation_error");
    if (BeginCard("##error")) {
        ImGui::TextColored(kFailed, ICON_FA_TRIANGLE_EXCLAMATION);
        ImGui::SameLine();
        ImGui::PushTextWrapPos(ImGui::GetContentRegionMax().x - ImGui::CalcTextSize("Dismiss").x - 16.0f);
        ImGui::TextUnformatted(reservation_error_.c_str());
        ImGui::PopTextWrapPos();
        ImGui::SameLine();
        ImGui::SetCursorPosX(ImGui::GetContentRegionMax().x - ImGui::CalcTextSize("Dismiss").x);
        if (ui::LinkButton("Dismiss")) reservation_error_.clear();
    }
    ImGui::EndChild();
    ImGui::PopID();
}

void ConnectionDialog::RenderReservationPanel() {
    const auto* node = SelectedNode();
    if (!node) return;

    ImGui::SeparatorText(ICON_FA_CLOCK " Reserve node");

    ReserveQuoteInputs in;
    in.node = FactsFor(*node);
    in.duration_minutes = reservation_duration_minutes_;
    in.reserving = reserving_;
    auto& auth = auth::AuthClient::Instance();
    in.has_account = auth.IsAuthenticated();
    const ReserveQuote quote = BuildReserveQuote(in);

    ImGui::PushID("reserve");
    if (BeginCard("##quote")) {
        ImGui::TextUnformatted(quote.title.c_str());
        ImGui::SameLine();
        ImGui::TextColored(Muted(), "%s", quote.subtitle.c_str());
        RightText(node->is_online ? kVerified : kNotYet, node->is_online ? "Online" : "Offline");

        ImGui::Spacing();
        ImGui::TextColored(Muted(), "How long");
        ImGui::SetNextItemWidth(260.0f);
        ImGui::SliderInt("##duration", &reservation_duration_minutes_, 10, 480, "");
        ImGui::SameLine();
        ImGui::TextUnformatted(quote.duration.c_str());
        ImGui::SameLine(0.0f, 28.0f);
        ImGui::SetNextItemWidth(110.0f);
        ImGui::InputInt("Epochs", &reservation_epochs_);
        reservation_epochs_ = std::clamp(reservation_epochs_, 1, 1000);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(110.0f);
        ImGui::InputInt("Batch size", &reservation_batch_size_);
        reservation_batch_size_ = std::clamp(reservation_batch_size_, 1, 512);

        ImGui::Separator();
        KeyValueTable("##money",
                      {{"Price", quote.price},
                       {"Held now", quote.hold},
                       {"Your balance after the hold", quote.balance_after},
                       {"You pay", quote.pay_rule}},
                      1);

        if (ui::PrimaryButton(quote.button.c_str(), quote.enabled, quote.disabled_reason.c_str())) {
            StartReservation();
        }
        if (!quote.enabled && !quote.disabled_reason.empty() && !reserving_) {
            ImGui::SameLine();
            ImGui::TextColored(kFailed, "%s", quote.disabled_reason.c_str());
        }
        ImGui::SameLine();
        if (ui::LinkButton(quote_details_open_ ? "Hide##quote" : "Details##quote")) {
            quote_details_open_ = !quote_details_open_;
        }
        if (quote_details_open_) KeyValueTable("##quote_details", quote.details, 2);
    }
    ImGui::EndChild();
    ImGui::PopID();
}

void ConnectionDialog::RenderActiveReservationPanel() {
    const ActiveReservationInputs in = CurrentReservationInputs();
    const ActiveReservationCard card = BuildActiveReservationCard(in);

    // The central server no longer knows the reservation (heartbeat 404):
    // it is over, whatever this clock says.
    if (reservation_gone_.load()) {
        std::string words;
        {
            std::lock_guard<std::mutex> lock(reservation_gone_mutex_);
            words = reservation_gone_message_;
        }
        reservation_gone_ = false;
        FinishReservation(ReservationEndReason::Lost, words);
        return;
    }

    // The node ends the reservation itself and tells the Engine why (gap 6);
    // give its message a moment to arrive, then close up here.
    if (card.urgency == ReservationUrgency::Ended) {
        if (expired_at_ == 0) expired_at_ = in.now;
        const bool stream_open = p2p_client_ && p2p_client_->IsStreaming();
        if (!stream_open || in.now - expired_at_ >= 10) {
            FinishReservation(ReservationEndReason::TimeRanOut, std::string());
            return;
        }
    }

    ImGui::SeparatorText(ICON_FA_BOLT " Active reservation");
    ImGui::PushID("active");
    if (BeginCard("##card")) {
        ImGui::TextUnformatted(card.title.c_str());
        ImGui::SameLine();
        ImGui::TextColored(Muted(), "%s", card.subtitle.c_str());
        RightText(card.connected ? kVerified : kNotYet, card.connection);

        ImGui::Spacing();
        const ImVec4 time_color = UrgencyColor(card.urgency);
        ImGui::BeginGroup();
        ImGui::TextColored(Muted(), "Time left");
        ImGui::SetWindowFontScale(1.6f);
        ImGui::TextColored(time_color, "%s", card.time_left.c_str());
        ImGui::SetWindowFontScale(1.0f);
        ImGui::EndGroup();
        ImGui::SameLine(0.0f, 24.0f);
        ImGui::BeginGroup();
        if (card.has_bar) {
            ImGui::TextColored(Muted(), "%s", card.started.c_str());
            RightText(Muted(), card.ends);
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, time_color);
            ImGui::ProgressBar(static_cast<float>(card.fill), ImVec2(-1.0f, 8.0f), "");
            ImGui::PopStyleColor();
        } else {
            ImGui::TextColored(Muted(), "%s", card.ends.c_str());
        }
        ImGui::TextColored(card.stale ? kFailed : Muted(), "%s", card.source.c_str());
        ImGui::EndGroup();

        if (card.urgency == ReservationUrgency::Ended) {
            ImGui::TextColored(kDanger, ICON_FA_CLOCK " The reservation has ended; closing the connection...");
        }

        KeyValueTable("##money", {{"Reserved time", card.price}}, 1);

        if (card.warn) {
            ImGui::TextColored(kFailed, ICON_FA_TRIANGLE_EXCLAMATION);
            ImGui::SameLine();
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextUnformatted(card.warning.c_str());
            ImGui::PopTextWrapPos();
        }

        // Extend (primary when time is short), Leave, Details.
        const bool ended = card.urgency == ReservationUrgency::Ended;
        const bool extend = card.warn ? ui::PrimaryButton("Extend " ICON_FA_CARET_DOWN, !ended)
                                      : ui::SecondaryButton("Extend " ICON_FA_CARET_DOWN, !ended);
        if (extend) ImGui::OpenPopup("extend_menu");
        if (ImGui::BeginPopup("extend_menu")) {
            for (const auto& option : BuildExtendOptions(in.node.price_usd_per_hour)) {
                if (ImGui::MenuItem(option.label.c_str(), option.cost.c_str())) {
                    ExtendActiveReservation(option.minutes);
                }
            }
            ImGui::Separator();
            ImGui::TextColored(Muted(), "Added to the reservation; the time is yours until it runs out.");
            ImGui::EndPopup();
        }
        ImGui::SameLine();
        if (ui::SecondaryButton("Leave node", !ended)) show_leave_confirm_ = true;
        ImGui::SameLine();
        ImGui::SetCursorPosX(ImGui::GetContentRegionMax().x - ImGui::CalcTextSize("Hide details").x);
        if (ui::LinkButton(reservation_details_open_ ? "Hide details" : "Details##active")) {
            reservation_details_open_ = !reservation_details_open_;
        }
        if (reservation_details_open_) KeyValueTable("##details", card.details, 3);
    }
    ImGui::EndChild();
    ImGui::PopID();

    RenderTrainingOnNodePanel();
}

void ConnectionDialog::RenderTrainingOnNodePanel() {
    ImGui::SeparatorText(ICON_FA_MICROCHIP " Training on the node");
    ImGui::PushID("training");
    if (BeginCard("##card")) {
        const bool connected = p2p_client_ && p2p_client_->IsConnected();
        if (!connected) {
            ImGui::TextColored(Muted(), "Connect to the reserved node to train on it.");
            if (ui::PrimaryButton(ICON_FA_LINK " Connect to node")) ConnectToReservedNode();
        } else {
            UpdateJobEstimate();
            RenderJobFitCard();
            ImGui::Spacing();

            const bool streaming = p2p_client_->IsStreaming();
            const bool waiting = p2p_client_->IsWaitingForNewJob();
            const auto fit = CurrentJobFitCard();
            if (waiting) {
                ImGui::TextColored(kVerified, ICON_FA_CIRCLE_CHECK " Job finished: ready for the next one");
                ImGui::SetNextItemWidth(110.0f);
                ImGui::InputInt("Epochs##next", &reservation_epochs_);
                reservation_epochs_ = std::clamp(reservation_epochs_, 1, 1000);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(110.0f);
                ImGui::InputInt("Batch size##next", &reservation_batch_size_);
                reservation_batch_size_ = std::clamp(reservation_batch_size_, 1, 512);
                if (ui::PrimaryButton(ICON_FA_PLAY " Start new training", fit.start_enabled, fit.start_note.c_str())) {
                    StartNewP2PTraining();
                }
            } else if (!streaming) {
                if (ui::PrimaryButton(ICON_FA_PLAY " Start training", fit.start_enabled, fit.start_note.c_str())) {
                    StartP2PTraining();
                }
            } else if (ui::SecondaryButton(ICON_FA_STOP " Stop training...")) {
                show_stop_confirm_ = true;
            }
            ImGui::SameLine();
            if (ui::SecondaryButton(ICON_FA_LINK_SLASH " Disconnect from node")) DisconnectFromNode();
            ImGui::SameLine();
            ImGui::TextColored(Muted(), "Disconnecting keeps the reservation and this card.");
        }
    }
    ImGui::EndChild();
    ImGui::PopID();
}

void ConnectionDialog::RenderLeaveNodePopup() {
    if (show_leave_confirm_) {
        ImGui::OpenPopup("Leave node");
        show_leave_confirm_ = false;
    }
    ImGui::SetNextWindowSize(ImVec2(520.0f, 0.0f), ImGuiCond_Appearing);
    if (ImGui::BeginPopupModal("Leave node", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        const LeaveSummary summary = BuildLeaveSummary(CurrentReservationInputs());
        ImGui::TextUnformatted(summary.title.c_str());
        ImGui::PushTextWrapPos(500.0f);
        ImGui::TextColored(Muted(), "%s", summary.body.c_str());
        ImGui::PopTextWrapPos();
        ImGui::Spacing();
        ImGui::TextUnformatted(summary.ends.c_str());
        ImGui::Spacing();
        if (ui::SecondaryButton("Stay", true, nullptr, ui::ButtonSize::Regular)) ImGui::CloseCurrentPopup();
        ImGui::SameLine();
        if (ui::PrimaryButton("Leave node", true, nullptr, ui::ButtonSize::Regular)) {
            ImGui::CloseCurrentPopup();
            LeaveNode();
        }
        ImGui::EndPopup();
    }
}

void ConnectionDialog::RenderStopTrainingPopup() {
    if (show_stop_confirm_) {
        ImGui::OpenPopup("Stop training");
        show_stop_confirm_ = false;
    }
    if (ImGui::BeginPopupModal("Stop training", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextUnformatted("Stop the training on the node?");
        ImGui::TextColored(Muted(), "The reservation stays; the node keeps the job's checkpoints.");
        ImGui::Spacing();
        if (ui::SecondaryButton("Keep training", true, nullptr, ui::ButtonSize::Regular)) ImGui::CloseCurrentPopup();
        ImGui::SameLine();
        if (ui::DangerButton("Stop training", true, nullptr, ui::ButtonSize::Regular)) {
            if (p2p_client_) p2p_client_->StopTraining();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

void ConnectionDialog::RenderReservationReceipt() {
    if (!has_receipt_) return;
    const ReservationReceipt receipt = BuildReservationReceipt(receipt_facts_);
    ImGui::SeparatorText(ICON_FA_CLOCK " Reservation ended");
    ImGui::PushID("receipt");
    if (BeginCard("##card")) {
        ImGui::TextUnformatted(receipt.title.c_str());
        RightText(Muted(), receipt.why);
        KeyValueTable("##facts", {{"Time used", receipt.time_used}, {"Paid", receipt.paid}, {"Jobs", receipt.jobs}},
                      3);
        if (!receipt.note.empty()) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(receipt.failed ? kFailed : Muted(), "%s", receipt.note.c_str());
            ImGui::PopTextWrapPos();
        }
        if (ui::SecondaryButton("Reserve again")) has_receipt_ = false;
        ImGui::SameLine();
        if (ui::LinkButton(receipt_details_open_ ? "Hide##receipt" : "Details##receipt")) {
            receipt_details_open_ = !receipt_details_open_;
        }
        ImGui::SameLine();
        ImGui::SetCursorPosX(ImGui::GetContentRegionMax().x - ImGui::CalcTextSize("Close").x);
        if (ui::LinkButton("Close##receipt")) has_receipt_ = false;
        if (receipt_details_open_) KeyValueTable("##details", receipt.details, 2);
    }
    ImGui::EndChild();
    ImGui::PopID();
}

void ConnectionDialog::RenderReconnectPrompt() {
    if (has_active_reservation_ || found_reservations_.empty()) return;
    const auto rows = BuildReconnectRows(found_reservations_, static_cast<long long>(std::time(nullptr)));
    if (rows.empty()) {
        found_reservations_.clear();  // all ran out
        return;
    }
    ImGui::SeparatorText(rows.size() == 1 ? ICON_FA_LINK " You have an active reservation"
                                          : ICON_FA_LINK " You have active reservations");
    ImGui::PushID("reconnect");
    if (BeginCard("##card")) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(Muted(),
                           "The reserved time is yours until it runs out. Reconnect to use it; there is nothing "
                           "to come back to after that.");
        ImGui::PopTextWrapPos();
        std::string reconnect_id;
        if (ImGui::BeginTable("##rows", 4, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg)) {
            const float actions = ui::ButtonWidth("Reconnect", ui::ButtonSize::Small) + 12.0f;
            ImGui::TableSetupColumn("Node", ImGuiTableColumnFlags_WidthStretch, 1.4f);
            ImGui::TableSetupColumn("Time left", ImGuiTableColumnFlags_WidthStretch, 0.8f);
            ImGui::TableSetupColumn("Note", ImGuiTableColumnFlags_WidthStretch, 1.2f);
            ImGui::TableSetupColumn("##actions", ImGuiTableColumnFlags_WidthFixed, actions);
            for (const auto& row : rows) {
                ImGui::PushID(row.reservation_id.c_str());
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(row.node.c_str());
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(row.time_left.c_str());
                ImGui::TableNextColumn();
                ImGui::TextColored(Muted(), "%s", row.note.c_str());
                ImGui::TableNextColumn();
                if (ui::SecondaryButton("Reconnect")) reconnect_id = row.reservation_id;
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
        // Outside the table: this changes found_reservations_.
        if (!reconnect_id.empty()) ReconnectToReservation(reconnect_id);
    }
    ImGui::EndChild();
    ImGui::PopID();
}

}  // namespace cyxwiz
