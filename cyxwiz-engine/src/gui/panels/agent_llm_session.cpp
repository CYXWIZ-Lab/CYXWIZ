#include "agent_llm_session.h"

#include <imgui.h>

#include <algorithm>
#include <exception>
#include <iterator>
#include <sstream>
#include <utility>

namespace cyxwiz {

namespace {

constexpr std::size_t kMaxTranscriptEntries = 64;
constexpr const char *kContextLabels[] = {"General", "Trace", "Training"};

const char *EntryLabel(AgentLlmSession::ContextMode mode) {
  switch (mode) {
  case AgentLlmSession::ContextMode::Trace:
    return "Explain the selected debugger trace.";
  case AgentLlmSession::ContextMode::Training:
    return "Explain the selected training stop reason.";
  case AgentLlmSession::ContextMode::General:
  default:
    return "Ask CyxWiz";
  }
}

} // namespace

AgentLlmSession::~AgentLlmSession() { JoinRequestWorker(); }

void AgentLlmSession::SetCommandHandler(CommandHandler handler) {
  command_handler_ = std::move(handler);
}

void AgentLlmSession::ResetProjectState() {
  ++project_generation_;
  std::fill(input_buffer_.begin(), input_buffer_.end(), '\0');
  transcript_.clear();
  scroll_to_bottom_ = false;

  std::lock_guard lock(response_mutex_);
  pending_response_.reset();
}

void AgentLlmSession::RenderContent(std::string_view project_root) {
  CheckRequestCompletion();

  ImGui::TextDisabled("Project: %.*s", static_cast<int>(project_root.size()),
                      project_root.data());

  int context_index = static_cast<int>(context_mode_);
  ImGui::SetNextItemWidth(150.0f);
  if (ImGui::Combo("Context##AgentLlmContext", &context_index, kContextLabels,
                   static_cast<int>(std::size(kContextLabels)))) {
    context_mode_ = static_cast<ContextMode>(context_index);
  }
  ImGui::SameLine();
  ImGui::Checkbox("Retrieval only", &retrieval_only_);
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip(
        "Search the assistant knowledge pack without model generation.");
  }

  ImGui::SameLine();
  if (request_running_.load(std::memory_order_acquire)) {
    ImGui::TextColored(ImVec4(0.35f, 0.75f, 1.0f, 1.0f),
                       "Assistant working...");
  } else {
    ImGui::TextDisabled("Provider: on demand");
  }

  ImGui::Separator();
  RenderTranscript();
  ImGui::Separator();
  RenderComposer();
}

void AgentLlmSession::RenderTranscript() {
  const float footer_height = ImGui::GetFrameHeightWithSpacing() * 2.0f +
                              ImGui::GetStyle().ItemSpacing.y;
  const float transcript_height =
      std::max(80.0f, ImGui::GetContentRegionAvail().y - footer_height);

  ImGui::BeginChild("AgentLlmTranscript", ImVec2(0.0f, transcript_height),
                    false, ImGuiWindowFlags_HorizontalScrollbar);

  if (ImGui::SmallButton("Copy conversation")) {
    const std::string transcript = BuildTranscript();
    ImGui::SetClipboardText(transcript.c_str());
  }
  ImGui::SameLine();
  if (ImGui::SmallButton("Clear")) {
    transcript_.clear();
  }

  if (transcript_.empty()) {
    ImGui::Spacing();
    ImGui::TextDisabled("Ask a question, explain a selected trace, or inspect "
                        "a training stop.");
  }

  for (std::size_t entry_index = 0; entry_index < transcript_.size();
       ++entry_index) {
    const auto &entry = transcript_[entry_index];
    ImGui::PushID(static_cast<int>(entry_index));
    ImGui::Spacing();
    switch (entry.kind) {
    case EntryKind::User:
      ImGui::TextColored(ImVec4(0.45f, 0.72f, 1.0f, 1.0f), "You");
      break;
    case EntryKind::Agent:
      ImGui::TextColored(ImVec4(0.45f, 0.82f, 0.55f, 1.0f), "Agent LLM");
      break;
    case EntryKind::Error:
      ImGui::TextColored(ImVec4(0.95f, 0.45f, 0.40f, 1.0f), "Agent LLM error");
      break;
    }

    if (!entry.backend_state.empty()) {
      ImGui::TextColored(ImVec4(0.90f, 0.62f, 0.25f, 1.0f), "Backend: %s",
                         entry.backend_state.c_str());
    }

    if (entry.retrieval_requested) {
      ImGui::TextColored(entry.retrieval_ok ? ImVec4(0.45f, 0.82f, 0.55f, 1.0f)
                                            : ImVec4(0.95f, 0.45f, 0.40f, 1.0f),
                         "Evidence: %s",
                         entry.retrieval_ok ? "retrieved" : "not found");
      ImGui::SameLine();
      ImGui::TextDisabled("|");
      ImGui::SameLine();
      if (!entry.runtime_requested) {
        ImGui::TextDisabled("Local model: not requested");
      } else {
        ImGui::TextColored(entry.runtime_ok ? ImVec4(0.45f, 0.82f, 0.55f, 1.0f)
                                            : ImVec4(0.95f, 0.45f, 0.40f, 1.0f),
                           "Local model: %s",
                           entry.runtime_ok ? "completed" : "unavailable");
      }
    }

    if (!entry.text.empty()) {
      ImGui::TextWrapped("%s", entry.text.c_str());
    }

    for (const auto &section : entry.sections) {
      ImGui::Spacing();
      ImGui::TextColored(ImVec4(0.72f, 0.78f, 0.88f, 1.0f), "%s",
                         section.title.c_str());
      if (section.content.empty()) {
        ImGui::TextDisabled("None reported.");
      } else {
        ImGui::TextWrapped("%s", section.content.c_str());
      }
    }

    if (!entry.sources.empty() &&
        ImGui::TreeNodeEx("Sources", ImGuiTreeNodeFlags_DefaultOpen,
                          "Sources (%d)",
                          static_cast<int>(entry.sources.size()))) {
      for (const auto &source : entry.sources) {
        ImGui::PushID(source.rank);
        ImGui::Bullet();
        ImGui::SameLine();
        ImGui::TextWrapped("#%d %s:%d-%d", source.rank, source.path.c_str(),
                           source.line_start, source.line_end);
        if (!source.title.empty()) {
          ImGui::TextDisabled("%s", source.title.c_str());
        }
        if (!source.snippet.empty() && ImGui::TreeNode("Retrieved excerpt")) {
          ImGui::TextWrapped("%s", source.snippet.c_str());
          ImGui::TreePop();
        }
        ImGui::PopID();
      }
      ImGui::TreePop();
    }
    ImGui::PopID();
  }

  if (scroll_to_bottom_) {
    ImGui::SetScrollHereY(1.0f);
    scroll_to_bottom_ = false;
  }
  ImGui::EndChild();
}

void AgentLlmSession::RenderComposer() {
  const bool running = request_running_.load(std::memory_order_acquire);
  const float send_width = 72.0f;
  const float input_width =
      std::max(100.0f, ImGui::GetContentRegionAvail().x - send_width -
                           ImGui::GetStyle().ItemSpacing.x);

  ImGui::BeginDisabled(running);
  ImGui::SetNextItemWidth(input_width);
  const bool focus_requested = focus_input_ && !running;
  if (focus_requested) {
    ImGui::SetKeyboardFocusHere();
  }
  const bool submit_from_enter = ImGui::InputTextWithHint(
      "##AgentLlmInput", EntryLabel(context_mode_), input_buffer_.data(),
      input_buffer_.size(), ImGuiInputTextFlags_EnterReturnsTrue);
  if (focus_requested && ImGui::IsItemActive())
    focus_input_ = false;
  ImGui::SameLine();
  const bool submit_from_button =
      ImGui::Button("Send", ImVec2(send_width, 0.0f));
  ImGui::EndDisabled();

  if (!running && (submit_from_enter || submit_from_button)) {
    SubmitRequest();
  }
}

void AgentLlmSession::SubmitRequest() {
  if (request_running_.load(std::memory_order_acquire))
    return;

  auto request =
      BuildRequest(context_mode_, retrieval_only_, input_buffer_.data());
  if (request.command_name == "ask" && request.user_text.empty()) {
    AppendEntry(EntryKind::Error, "Enter a question before sending.");
    return;
  }
  if (!command_handler_) {
    AppendEntry(EntryKind::Error,
                "Assistant command handler is not configured.");
    return;
  }

  if (request_worker_.joinable())
    request_worker_.join();

  const std::string displayed_request =
      request.user_text.empty() ? EntryLabel(context_mode_) : request.user_text;
  AppendEntry(EntryKind::User, displayed_request);
  std::fill(input_buffer_.begin(), input_buffer_.end(), '\0');

  const auto handler = command_handler_;
  const std::uint64_t request_generation = project_generation_;
  request_finished_.store(false, std::memory_order_release);
  request_running_.store(true, std::memory_order_release);
  request_worker_ = std::thread(
      [this, handler, request = std::move(request), request_generation]() {
        plugin::AssistantCommandResponse response;
        try {
          response = handler(request);
        } catch (const std::exception &exception) {
          response.error =
              std::string("Assistant provider failed: ") + exception.what();
        } catch (...) {
          response.error = "Assistant provider failed unexpectedly.";
        }

        {
          std::lock_guard lock(response_mutex_);
          pending_response_ =
              PendingResponse{request_generation, std::move(response)};
        }
        request_finished_.store(true, std::memory_order_release);
      });
}

void AgentLlmSession::CheckRequestCompletion() {
  if (!request_running_.load(std::memory_order_acquire) ||
      !request_finished_.load(std::memory_order_acquire)) {
    return;
  }

  JoinRequestWorker();

  std::optional<PendingResponse> pending;
  {
    std::lock_guard lock(response_mutex_);
    pending = std::move(pending_response_);
    pending_response_.reset();
  }
  request_finished_.store(false, std::memory_order_release);
  request_running_.store(false, std::memory_order_release);

  if (!pending || pending->project_generation != project_generation_) {
    return;
  }

  auto response = std::move(pending->response);
  if (!response.handled) {
    AppendEntry(EntryKind::Error,
                response.error.empty()
                    ? "No assistant provider plugin is loaded."
                    : response.error);
    return;
  }

  if (response.success) {
    AppendResponse(EntryKind::Agent, std::move(response));
    return;
  }

  AppendResponse(EntryKind::Error, std::move(response));
}

void AgentLlmSession::AppendEntry(EntryKind kind, std::string text) {
  AppendTranscriptEntry({kind, std::move(text)});
}

void AgentLlmSession::AppendResponse(
    EntryKind kind, plugin::AssistantCommandResponse response) {
  std::string text;
  if (kind == EntryKind::Error) {
    text = response.error.empty() ? "Assistant request failed."
                                  : std::move(response.error);
    if (!response.output.empty()) {
      text += "\n\n" + response.output;
    }
  } else if (!response.output.empty()) {
    text = std::move(response.output);
  } else if (response.sections.empty()) {
    text = "Assistant request completed.";
  }

  TranscriptEntry entry;
  entry.kind = kind;
  entry.text = std::move(text);
  entry.retrieval_requested = response.retrieval_requested;
  entry.retrieval_ok = response.retrieval_ok;
  entry.runtime_requested = response.runtime_requested;
  entry.runtime_ok = response.runtime_ok;
  entry.backend_state = std::move(response.backend_state);
  entry.sections = std::move(response.sections);
  entry.sources = std::move(response.sources);
  AppendTranscriptEntry(std::move(entry));
}

void AgentLlmSession::AppendTranscriptEntry(TranscriptEntry entry) {
  if (transcript_.size() == kMaxTranscriptEntries) {
    transcript_.erase(transcript_.begin());
  }
  transcript_.push_back(std::move(entry));
  scroll_to_bottom_ = true;
}

std::string AgentLlmSession::BuildTranscript() const {
  std::ostringstream output;
  for (const auto &entry : transcript_) {
    switch (entry.kind) {
    case EntryKind::User:
      output << "You:\n";
      break;
    case EntryKind::Agent:
      output << "Agent LLM:\n";
      break;
    case EntryKind::Error:
      output << "Agent LLM error:\n";
      break;
    }
    if (!entry.backend_state.empty()) {
      output << "Backend: " << entry.backend_state << '\n';
    }
    if (entry.retrieval_requested) {
      output << "Evidence: " << (entry.retrieval_ok ? "retrieved" : "not found")
             << '\n';
      output << "Local model: ";
      if (!entry.runtime_requested) {
        output << "not requested\n";
      } else {
        output << (entry.runtime_ok ? "completed" : "unavailable") << '\n';
      }
    }
    if (!entry.text.empty()) {
      output << entry.text << '\n';
    }
    for (const auto &section : entry.sections) {
      output << section.title << ":\n"
             << (section.content.empty() ? "None reported." : section.content)
             << "\n\n";
    }
    if (!entry.sources.empty()) {
      output << "Sources:\n";
      for (const auto &source : entry.sources) {
        output << "- #" << source.rank << ' ' << source.path << ':'
               << source.line_start << '-' << source.line_end << '\n';
      }
    }
    output << '\n';
  }
  return output.str();
}

void AgentLlmSession::JoinRequestWorker() {
  if (request_worker_.joinable())
    request_worker_.join();
}

} // namespace cyxwiz
