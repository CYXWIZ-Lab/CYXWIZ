#include "local_shell_session.h"

#include "../editor_fonts.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace cyxwiz {

namespace {

ImU32 ToImGuiColor(std::uint32_t rgb) {
  return IM_COL32((rgb >> 16) & 0xFF, (rgb >> 8) & 0xFF, rgb & 0xFF, 255);
}

std::string EncodeUtf8(unsigned int value) {
  std::string encoded;
  if (value <= 0x7F) {
    encoded.push_back(static_cast<char>(value));
  } else if (value <= 0x7FF) {
    encoded.push_back(static_cast<char>(0xC0 | (value >> 6)));
    encoded.push_back(static_cast<char>(0x80 | (value & 0x3F)));
  } else if (value <= 0xFFFF) {
    encoded.push_back(static_cast<char>(0xE0 | (value >> 12)));
    encoded.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3F)));
    encoded.push_back(static_cast<char>(0x80 | (value & 0x3F)));
  } else {
    encoded.push_back(static_cast<char>(0xF0 | (value >> 18)));
    encoded.push_back(static_cast<char>(0x80 | ((value >> 12) & 0x3F)));
    encoded.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3F)));
    encoded.push_back(static_cast<char>(0x80 | (value & 0x3F)));
  }
  return encoded;
}

std::uint8_t TerminalModifiers(const ImGuiIO &io) {
  std::uint8_t modifiers = TerminalBuffer::NoModifier;
  if (io.KeyShift)
    modifiers |= TerminalBuffer::Shift;
  if (io.KeyAlt)
    modifiers |= TerminalBuffer::Alt;
  if (io.KeyCtrl)
    modifiers |= TerminalBuffer::Control;
  return modifiers;
}

} // namespace

LocalShellSession::LocalShellSession(LocalShellKind kind,
                                     std::filesystem::path project_root)
    : process_(kind), project_root_(std::move(project_root)),
      terminal_(columns_, rows_) {}

LocalShellSession::~LocalShellSession() {
  if (start_thread_.joinable())
    start_thread_.join();
}

void LocalShellSession::RenderContent() {
  CheckStartCompletion();
  DrainProcessOutput();

  const bool starting = start_in_progress_.load(std::memory_order_acquire);
  const char *state_text = starting               ? "Starting..."
                           : process_.IsRunning() ? "Running"
                                                  : "Stopped";
  const ImVec4 state_color = starting ? ImVec4(0.92f, 0.72f, 0.30f, 1.0f)
                             : process_.IsRunning()
                                 ? ImVec4(0.45f, 0.82f, 0.55f, 1.0f)
                                 : ImVec4(0.95f, 0.45f, 0.40f, 1.0f);

  ImGui::TextColored(ImVec4(0.55f, 0.72f, 0.95f, 1.0f), "%s",
                     LocalShellProcess::DisplayName(process_.Kind()).data());
  ImGui::SameLine();
  ImGui::TextColored(state_color, "| %s", state_text);

  ImGui::SameLine();
  ImGui::BeginDisabled(starting);
  if (ImGui::SmallButton("Restart")) {
    process_.Stop();
    terminal_.Clear();
    StartProcessAsync();
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  ImGui::BeginDisabled(!process_.IsRunning());
  if (ImGui::SmallButton("Stop"))
    process_.Stop();
  ImGui::EndDisabled();
  ImGui::SameLine();
  if (ImGui::SmallButton("Copy"))
    ImGui::SetClipboardText(terminal_.PlainText().c_str());
  ImGui::SameLine();
  if (ImGui::SmallButton("Clear")) {
    terminal_.Clear();
    scroll_offset_ = 0;
  }

  if (!status_message_.empty()) {
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.95f, 0.55f, 0.35f, 1.0f), "%s",
                       status_message_.c_str());
  }

  RenderTerminal();
}

void LocalShellSession::StartProcessAsync() {
  if (start_thread_.joinable())
    start_thread_.join();
  start_attempted_ = true;
  status_message_.clear();
  start_error_.clear();
  start_success_ = false;
  start_finished_.store(false, std::memory_order_relaxed);
  start_in_progress_.store(true, std::memory_order_release);
  const auto project_root = project_root_;
  const auto columns = columns_;
  const auto rows = rows_;
  start_thread_ = std::thread([this, project_root, columns, rows]() {
    std::string error;
    start_success_ = process_.Start(project_root, error, columns, rows);
    start_error_ = std::move(error);
    start_in_progress_.store(false, std::memory_order_release);
    start_finished_.store(true, std::memory_order_release);
  });
}

void LocalShellSession::CheckStartCompletion() {
  if (!start_finished_.exchange(false, std::memory_order_acq_rel))
    return;
  if (start_thread_.joinable())
    start_thread_.join();
  if (!start_success_) {
    status_message_ = start_error_;
    return;
  }
  status_message_.clear();
  std::string resize_error;
  if (!process_.Resize(columns_, rows_, resize_error))
    status_message_ = std::move(resize_error);
  focus_terminal_ = true;
}

void LocalShellSession::DrainProcessOutput() {
  const auto output = process_.DrainOutput();
  if (!output.empty()) {
    terminal_.Feed(output);
    SendInput(terminal_.TakeOutboundData());
  }
}

void LocalShellSession::RenderTerminal() {
  ImFont *terminal_font =
      gui::GetEditorMonoFont(ImGui::GetIO().FontGlobalScale);
  if (terminal_font)
    ImGui::PushFont(terminal_font);
  ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.063f, 0.078f, 0.094f, 1.0f));
  ImGui::BeginChild("LocalShellTerminal", ImVec2(0.0f, 0.0f), true,
                    ImGuiWindowFlags_NoScrollbar |
                        ImGuiWindowFlags_NoScrollWithMouse);

  const ImVec2 origin = ImGui::GetCursorScreenPos();
  const ImVec2 available = ImGui::GetContentRegionAvail();
  const ImVec2 canvas_size(std::max(available.x, 1.0f),
                           std::max(available.y, 1.0f));
  const float cell_width = std::max(ImGui::CalcTextSize("M").x, 1.0f);
  const float cell_height =
      std::max(ImGui::GetTextLineHeightWithSpacing(), 1.0f);
  const auto columns = static_cast<std::uint16_t>(
      std::clamp(static_cast<int>(std::floor(canvas_size.x / cell_width)), 2,
                 static_cast<int>(std::numeric_limits<std::uint16_t>::max())));
  const auto rows = static_cast<std::uint16_t>(
      std::clamp(static_cast<int>(std::floor(canvas_size.y / cell_height)), 1,
                 static_cast<int>(std::numeric_limits<std::uint16_t>::max())));
  ResizeTerminal(columns, rows);

  input_capture_.fill('\0');
  if (focus_terminal_) {
    ImGui::SetWindowFocus();
    ImGui::SetKeyboardFocusHere();
  }
  ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0, 0, 0, 0));
  ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0, 0, 0, 0));
  ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, ImVec4(0, 0, 0, 0));
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
  ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
  ImGui::InputTextMultiline("##TerminalInputCapture", input_capture_.data(),
                            input_capture_.size(), canvas_size,
                            ImGuiInputTextFlags_CallbackCharFilter |
                                ImGuiInputTextFlags_NoHorizontalScroll |
                                ImGuiInputTextFlags_NoUndoRedo,
                            &LocalShellSession::InputCallback, this);
  terminal_focused_ = ImGui::IsItemActive();
  const bool hovered = ImGui::IsItemHovered();
  if (focus_terminal_ && terminal_focused_)
    focus_terminal_ = false;
  ImGui::PopStyleVar(2);
  ImGui::PopStyleColor(3);

  if (hovered && ImGui::GetIO().MouseWheel != 0.0f &&
      !terminal_.AlternateScreen()) {
    const int delta = static_cast<int>(ImGui::GetIO().MouseWheel * 3.0f);
    const std::size_t max_offset = terminal_.TotalLineCount() > rows
                                       ? terminal_.TotalLineCount() - rows
                                       : 0;
    if (delta > 0)
      scroll_offset_ = std::min(max_offset, scroll_offset_ + delta);
    else
      scroll_offset_ = scroll_offset_ > static_cast<std::size_t>(-delta)
                           ? scroll_offset_ - static_cast<std::size_t>(-delta)
                           : 0;
  }
  if (terminal_.AlternateScreen())
    scroll_offset_ = 0;

  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  draw_list->PushClipRect(
      origin, ImVec2(origin.x + canvas_size.x, origin.y + canvas_size.y), true);
  draw_list->AddRectFilled(
      origin, ImVec2(origin.x + canvas_size.x, origin.y + canvas_size.y),
      IM_COL32(16, 20, 24, 255));

  const std::size_t total_lines = terminal_.TotalLineCount();
  const std::size_t visible_rows = std::min<std::size_t>(rows, total_lines);
  const std::size_t end_line =
      total_lines > scroll_offset_ ? total_lines - scroll_offset_ : 0;
  const std::size_t first_line =
      end_line > visible_rows ? end_line - visible_rows : 0;

  for (std::size_t visible_row = 0; visible_row < visible_rows; ++visible_row) {
    const auto &line = terminal_.LineAt(first_line + visible_row);
    const float y = origin.y + static_cast<float>(visible_row) * cell_height;
    for (std::size_t column = 0; column < line.size(); ++column) {
      const auto &cell = line[column];
      const float x = origin.x + static_cast<float>(column) * cell_width;
      if (cell.background != 0x101418) {
        draw_list->AddRectFilled(ImVec2(x, y),
                                 ImVec2(x + cell_width, y + cell_height),
                                 ToImGuiColor(cell.background));
      }
      if (cell.continuation || cell.codepoint == U' ')
        continue;
      const auto display_codepoint = gui::ResolveTerminalDisplayCodepoint(
          ImGui::GetFont(), cell.codepoint);
      std::string encoded =
          EncodeUtf8(static_cast<unsigned int>(display_codepoint));
      for (std::size_t index = 0; index < cell.combining_count; ++index)
        encoded += EncodeUtf8(static_cast<unsigned int>(cell.combining[index]));
      draw_list->AddText(ImVec2(x, y), ToImGuiColor(cell.foreground),
                         encoded.c_str(), encoded.c_str() + encoded.size());
      if (cell.bold)
        draw_list->AddText(ImVec2(x + 1.0f, y), ToImGuiColor(cell.foreground),
                           encoded.c_str(), encoded.c_str() + encoded.size());
      if (cell.underline)
        draw_list->AddLine(ImVec2(x, y + cell_height - 1.0f),
                           ImVec2(x + cell_width, y + cell_height - 1.0f),
                           ToImGuiColor(cell.foreground));
      if (cell.strike)
        draw_list->AddLine(ImVec2(x, y + cell_height * 0.55f),
                           ImVec2(x + cell_width, y + cell_height * 0.55f),
                           ToImGuiColor(cell.foreground));
    }
  }

  if (terminal_.CursorVisible() && scroll_offset_ == 0 &&
      process_.IsRunning()) {
    const std::size_t cursor_line =
        terminal_.TotalLineCount() - terminal_.Rows() + terminal_.CursorRow();
    if (cursor_line >= first_line && cursor_line < end_line) {
      const float x = origin.x + terminal_.CursorColumn() * cell_width;
      const float y = origin.y + (cursor_line - first_line) * cell_height;
      draw_list->AddRectFilled(ImVec2(x, y + cell_height - 2.0f),
                               ImVec2(x + cell_width, y + cell_height),
                               IM_COL32(210, 215, 220, 220));
    }
  }
  draw_list->PopClipRect();

  if (terminal_focused_)
    HandleKeyboardInput();
  if (!start_attempted_)
    StartProcessAsync();

  ImGui::EndChild();
  ImGui::PopStyleColor();
  if (terminal_font)
    ImGui::PopFont();
}

void LocalShellSession::HandleKeyboardInput() {
  const ImGuiIO &io = ImGui::GetIO();
  if (io.KeyCtrl && io.KeyShift && ImGui::IsKeyPressed(ImGuiKey_C)) {
    ImGui::SetClipboardText(terminal_.PlainText().c_str());
    return;
  }
  if (io.KeyCtrl && io.KeyShift && ImGui::IsKeyPressed(ImGuiKey_V)) {
    if (const char *clipboard = ImGui::GetClipboardText())
      SendInput(terminal_.EncodePaste(clipboard));
    return;
  }

  if (io.KeyCtrl && !io.KeyShift && !io.KeyAlt) {
    for (int key = static_cast<int>(ImGuiKey_A);
         key <= static_cast<int>(ImGuiKey_Z); ++key) {
      if (ImGui::IsKeyPressed(static_cast<ImGuiKey>(key))) {
        const char32_t character = U'a' + (key - static_cast<int>(ImGuiKey_A));
        SendInput(
            terminal_.EncodeCharacter(character, TerminalBuffer::Control));
        return;
      }
    }
  }

  struct KeySequence {
    ImGuiKey key;
    TerminalBuffer::Key terminal_key;
  };
  for (const auto &[key, terminal_key] : {
           KeySequence{ImGuiKey_Enter, TerminalBuffer::Key::Enter},
           KeySequence{ImGuiKey_KeypadEnter, TerminalBuffer::Key::Enter},
           KeySequence{ImGuiKey_Backspace, TerminalBuffer::Key::Backspace},
           KeySequence{ImGuiKey_Tab, TerminalBuffer::Key::Tab},
           KeySequence{ImGuiKey_Escape, TerminalBuffer::Key::Escape},
           KeySequence{ImGuiKey_UpArrow, TerminalBuffer::Key::Up},
           KeySequence{ImGuiKey_DownArrow, TerminalBuffer::Key::Down},
           KeySequence{ImGuiKey_RightArrow, TerminalBuffer::Key::Right},
           KeySequence{ImGuiKey_LeftArrow, TerminalBuffer::Key::Left},
           KeySequence{ImGuiKey_Home, TerminalBuffer::Key::Home},
           KeySequence{ImGuiKey_End, TerminalBuffer::Key::End},
           KeySequence{ImGuiKey_Delete, TerminalBuffer::Key::Delete},
           KeySequence{ImGuiKey_Insert, TerminalBuffer::Key::Insert},
           KeySequence{ImGuiKey_PageUp, TerminalBuffer::Key::PageUp},
           KeySequence{ImGuiKey_PageDown, TerminalBuffer::Key::PageDown},
           KeySequence{ImGuiKey_F1, TerminalBuffer::Key::Function1},
           KeySequence{ImGuiKey_F2, TerminalBuffer::Key::Function2},
           KeySequence{ImGuiKey_F3, TerminalBuffer::Key::Function3},
           KeySequence{ImGuiKey_F4, TerminalBuffer::Key::Function4},
           KeySequence{ImGuiKey_F5, TerminalBuffer::Key::Function5},
           KeySequence{ImGuiKey_F6, TerminalBuffer::Key::Function6},
           KeySequence{ImGuiKey_F7, TerminalBuffer::Key::Function7},
           KeySequence{ImGuiKey_F8, TerminalBuffer::Key::Function8},
           KeySequence{ImGuiKey_F9, TerminalBuffer::Key::Function9},
           KeySequence{ImGuiKey_F10, TerminalBuffer::Key::Function10},
           KeySequence{ImGuiKey_F11, TerminalBuffer::Key::Function11},
           KeySequence{ImGuiKey_F12, TerminalBuffer::Key::Function12},
       }) {
    if (ImGui::IsKeyPressed(key)) {
      SendInput(terminal_.EncodeKey(terminal_key, TerminalModifiers(io)));
      return;
    }
  }
}

void LocalShellSession::SendInput(std::string_view bytes) {
  if (!process_.IsRunning() || bytes.empty())
    return;
  std::string error;
  if (!process_.Write(bytes, error))
    status_message_ = std::move(error);
  else
    status_message_.clear();
}

void LocalShellSession::ResizeTerminal(std::uint16_t columns,
                                       std::uint16_t rows) {
  if (columns == columns_ && rows == rows_)
    return;
  columns_ = columns;
  rows_ = rows;
  terminal_.Resize(columns_, rows_);
  if (process_.IsRunning()) {
    std::string error;
    if (!process_.Resize(columns_, rows_, error))
      status_message_ = std::move(error);
  }
}

int LocalShellSession::InputCallback(ImGuiInputTextCallbackData *data) {
  if (!data || data->EventFlag != ImGuiInputTextFlags_CallbackCharFilter)
    return 0;
  auto *session = static_cast<LocalShellSession *>(data->UserData);
  if (!session || data->EventChar == 0)
    return 1;
  if (data->EventChar == '\r' || data->EventChar == '\n' ||
      data->EventChar == '\t')
    return 1;
  const std::uint8_t modifiers =
      ImGui::GetIO().KeyAlt ? TerminalBuffer::Alt : TerminalBuffer::NoModifier;
  session->SendInput(session->terminal_.EncodeCharacter(
      static_cast<char32_t>(data->EventChar), modifiers));
  return 1;
}

} // namespace cyxwiz
