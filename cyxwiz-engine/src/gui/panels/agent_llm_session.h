#pragma once

#include "../../plugin/interfaces/i_assistant_provider.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace cyxwiz {

class AgentLlmSession {
public:
  using CommandHandler = std::function<plugin::AssistantCommandResponse(
      const plugin::AssistantCommandRequest &)>;

  enum class ContextMode : std::uint8_t {
    General,
    Trace,
    Training,
  };

  AgentLlmSession() = default;
  ~AgentLlmSession();

  AgentLlmSession(const AgentLlmSession &) = delete;
  AgentLlmSession &operator=(const AgentLlmSession &) = delete;

  void SetCommandHandler(CommandHandler handler);
  void ResetProjectState();
  void RequestInputFocus() { focus_input_ = true; }
  void RenderContent(std::string_view project_root);

  static plugin::AssistantCommandRequest BuildRequest(ContextMode context_mode,
                                                      bool retrieval_only,
                                                      std::string user_text) {
    const auto first = user_text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
      user_text.clear();
    } else {
      const auto last = user_text.find_last_not_of(" \t\r\n");
      user_text = user_text.substr(first, last - first + 1);
    }

    plugin::AssistantCommandRequest request;
    request.user_text = std::move(user_text);
    if (retrieval_only) {
      request.command_name = "find_source";
    } else if (context_mode == ContextMode::Trace) {
      request.command_name = "explain_trace";
    } else if (context_mode == ContextMode::Training) {
      request.command_name = "explain_training";
    } else {
      request.command_name = "ask";
    }
    return request;
  }

private:
  enum class EntryKind : std::uint8_t {
    User,
    Agent,
    Error,
  };

  struct TranscriptEntry {
    EntryKind kind = EntryKind::Agent;
    std::string text;
    bool retrieval_requested = false;
    bool retrieval_ok = false;
    bool runtime_requested = false;
    bool runtime_ok = false;
    std::string backend_state;
    std::vector<plugin::AssistantCommandSection> sections;
    std::vector<plugin::AssistantCommandSource> sources;
  };

  struct PendingResponse {
    std::uint64_t project_generation = 0;
    plugin::AssistantCommandResponse response;
  };

  void RenderTranscript();
  void RenderComposer();
  void SubmitRequest();
  void CheckRequestCompletion();
  void AppendEntry(EntryKind kind, std::string text);
  void AppendResponse(EntryKind kind,
                      plugin::AssistantCommandResponse response);
  void AppendTranscriptEntry(TranscriptEntry entry);
  std::string BuildTranscript() const;
  void JoinRequestWorker();

  CommandHandler command_handler_;
  std::array<char, 2048> input_buffer_{};
  std::vector<TranscriptEntry> transcript_;
  ContextMode context_mode_ = ContextMode::General;
  bool retrieval_only_ = false;
  bool focus_input_ = true;
  bool scroll_to_bottom_ = false;
  std::uint64_t project_generation_ = 0;

  std::thread request_worker_;
  std::atomic<bool> request_running_{false};
  std::atomic<bool> request_finished_{false};
  std::mutex response_mutex_;
  std::optional<PendingResponse> pending_response_;
};

} // namespace cyxwiz
