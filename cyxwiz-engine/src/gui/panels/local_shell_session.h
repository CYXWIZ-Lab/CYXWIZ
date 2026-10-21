#pragma once

#include "local_shell_process.h"
#include "terminal_buffer.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <thread>

struct ImGuiInputTextCallbackData;

namespace cyxwiz {

class LocalShellSession {
public:
  LocalShellSession(LocalShellKind kind, std::filesystem::path project_root);
  ~LocalShellSession();

  LocalShellSession(const LocalShellSession &) = delete;
  LocalShellSession &operator=(const LocalShellSession &) = delete;

  void RenderContent();
  void RequestInputFocus() { focus_terminal_ = true; }

private:
  void StartProcessAsync();
  void CheckStartCompletion();
  void DrainProcessOutput();
  void RenderTerminal();
  void HandleKeyboardInput();
  void SendInput(std::string_view bytes);
  void ResizeTerminal(std::uint16_t columns, std::uint16_t rows);
  static int InputCallback(ImGuiInputTextCallbackData *data);

  LocalShellProcess process_;
  std::filesystem::path project_root_;
  std::uint16_t columns_ = 80;
  std::uint16_t rows_ = 24;
  TerminalBuffer terminal_;
  std::array<char, 8> input_capture_{};
  std::string status_message_;
  std::size_t scroll_offset_ = 0;
  std::thread start_thread_;
  std::atomic<bool> start_in_progress_{false};
  std::atomic<bool> start_finished_{false};
  bool start_success_ = false;
  std::string start_error_;
  bool start_attempted_ = false;
  bool focus_terminal_ = true;
  bool terminal_focused_ = false;
};

} // namespace cyxwiz
