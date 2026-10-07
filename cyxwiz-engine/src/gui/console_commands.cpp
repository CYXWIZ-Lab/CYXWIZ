// Console "Commands" session: header, quick commands, command blocks and
// the prompt with suggestions (tofix121). Suggestions and output parsing
// live in core/console_commands_presentation; command execution stays in
// RuntimeConsoleCommandService and console.cpp (pip).

#include "console_palette.h"
#include "console.h"
#include "../core/async_task_manager.h"
#include "../core/console_commands_presentation.h"
#include "../core/runtime_console_commands.h"
#include "../core/runtime_log_presentation.h"
#include "editor_fonts.h"
#include "icons.h"
#include "ui_buttons.h"
#include "separate_windows.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <imgui.h>
#include <imgui_internal.h>
#include <sstream>
#include <unordered_set>

namespace ui = cyxwiz::ui;
namespace cmd = cyxwiz::commands;

namespace {

ImVec4 kMuted;
ImVec4 kFaint;
ImVec4 kText;
ImVec4 kBright;
ImVec4 kGreen;
ImVec4 kAmber;
ImVec4 kRed;
ImVec4 kBlue;
ImVec4 kAccent;
ImVec4 kPrompt;
ImVec4 kPanelBg;
ImVec4 kInputBg;
ImVec4 kBorder;

std::array<ImVec4, 6> kLevelColors{};

// Colours follow the active theme (gui/console_palette), refreshed
// at the start of each render.
void RefreshPalette() {
  const auto &p = ::gui::CurrentConsolePalette();
  kMuted = p.muted;
  kFaint = p.faint;
  kText = p.text;
  kBright = p.bright;
  kGreen = p.success;
  kAmber = p.warning;
  kRed = p.error;
  kBlue = p.info;
  kAccent = p.accent_text;
  kPrompt = p.accent;
  kPanelBg = p.panel;
  kInputBg = p.input;
  kBorder = p.border;
  kLevelColors = {p.muted, p.info, p.text, p.warning, p.error, p.critical};
}

constexpr std::array<const char *, 6> kLevelNames = {"Trace", "Debug", "Info",
                                                     "Warn",  "Error", "Critical"};

constexpr const char *kQuickCommands[] = {
    "help",           "show logs errors",      "show errors last 20", "show device active",
    "show training current", "show backend packs", "pip list"};

ImFont *CommandsMonoFont() { return cyxwiz::gui::GetCodeFont(); }

void Tooltip(const char *text) {
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal |
                           ImGuiHoveredFlags_AllowWhenDisabled))
    ImGui::SetTooltip("%s", text);
}

void Flow(float width, float right, bool first) {
  if (first)
    return;
  const float x = ImGui::GetItemRectMax().x + ImGui::GetStyle().ItemSpacing.x;
  if (x + width <= right)
    ImGui::SameLine();
}

std::string LocalClock(std::chrono::system_clock::time_point timestamp) {
  auto text = cyxwiz::logs::FormatLocalTime(timestamp);  // HH:MM:SS.mmm
  return text.substr(0, 8);
}

} // namespace

namespace gui {

void Console::SetCommandsNotice(std::string message) {
  command_notice_ = std::move(message);
  command_notice_until_ = ImGui::GetTime() + 2.5;
}

void Console::EnsureCommandForms() {
  if (!command_forms_.empty())
    return;
  std::vector<cmd::CommandSource> sources;
  for (const auto &descriptor : cyxwiz::RuntimeConsoleCommandService::Descriptors()) {
    sources.push_back({descriptor.name, descriptor.usage, descriptor.description,
                       descriptor.detailed_help});
    command_usages_.emplace_back(std::string(descriptor.name),
                                 std::string(descriptor.usage));
  }
  command_forms_ = cmd::BuildCommandForms(sources);
}

void Console::AddBlockLine(uint64_t block, const std::string &message, LogLevel level) {
  const uint64_t previous = current_command_block_;
  current_command_block_ = block;
  AddLog(message, level);
  current_command_block_ = previous;
}

void Console::FinishPipBlock(uint64_t block, bool success) {
  const auto found = command_blocks_.find(block);
  if (found == command_blocks_.end())
    return;
  found->second.running = false;
  found->second.success = success && !found->second.cancelled;
  found->second.task_id = 0;
}

void Console::ExecCommand(const char *command) {
  EnsureCommandForms();
  const uint64_t block = ++next_command_block_;
  {
    std::lock_guard<std::mutex> lock(log_mutex_);
    LogEntry entry;
    entry.message = command;
    entry.level = LogLevel::Info;
    entry.timestamp = std::chrono::system_clock::now();
    entry.sequence = ++next_command_sequence_;
    entry.is_command = true;
    entry.block = block;
    items_.push_back(std::move(entry));
    if (items_.size() > 1000)
      items_.pop_front();
  }
  scroll_to_bottom_.store(true, std::memory_order_relaxed);
  auto &state = command_blocks_[block];
  state.command = command;
  state.started = std::chrono::steady_clock::now();

  current_command_block_ = block;
  const auto result = command_service_->Execute(command);
  command_blocks_[block].success = result.success;
  switch (result.action) {
  case cyxwiz::RuntimeConsoleAction::Clear:
    current_command_block_ = 0;
    ClearCommandTranscript();
    return;
  case cyxwiz::RuntimeConsoleAction::ExecutePip:
    ExecutePipCommand(result.action_arguments);
    break;
  case cyxwiz::RuntimeConsoleAction::None:
    break;
  }
  AppendCommandResult(result);
  current_command_block_ = 0;

  // Blocks whose command entry scrolled out of the transcript are dropped.
  if (command_blocks_.size() > 400) {
    std::unordered_set<uint64_t> live;
    for (const auto &entry : SnapshotEntries())
      if (entry.is_command)
        live.insert(entry.block);
    for (auto it = command_blocks_.begin(); it != command_blocks_.end();) {
      if (!it->second.running && !live.count(it->first))
        it = command_blocks_.erase(it);
      else
        ++it;
    }
  }
}

void Console::RenderCommandsSession(bool request_focus) {
  RefreshPalette();
  EnsureCommandForms();
  RenderCommandsHeader();
  RenderCommandQuickRow();

  const ImGuiStyle &style = ImGui::GetStyle();
  const float footer = ImGui::GetFrameHeight() + 20.0f + ImGui::GetTextLineHeight() +
                       style.ItemSpacing.y * 3.0f + 8.0f;
  ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0, 0, 0, 0));
  ImGui::BeginChild("CommandsTranscript", ImVec2(0, -footer), ImGuiChildFlags_None,
                    ImGuiWindowFlags_HorizontalScrollbar);
  RenderCommandTranscript();
  ImGui::EndChild();
  ImGui::PopStyleColor();

  RenderCommandInput(request_focus);

  const ImGuiIO &io = ImGui::GetIO();
  if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) && io.KeyCtrl) {
    if (ImGui::IsKeyPressed(ImGuiKey_L, false))
      ClearCommandTranscript();
    if (ImGui::IsKeyPressed(ImGuiKey_C, false) && !io.WantTextInput &&
        (selected_command_block_ != 0 || selected_command_sequence_ != 0))
      CopySelectedCommand();
  }
}

void Console::RenderCommandsHeader() {
  const float right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8.0f, 6.0f));
  ImGui::AlignTextToFramePadding();
  ImGui::TextColored(kBright, "Runtime commands");
  const char *about = "query diagnostics, manage the session filter, run project pip";
  Flow(ImGui::CalcTextSize(about).x, right, false);
  ImGui::TextColored(kMuted, "%s", about);

  const std::string &filter = command_service_->ActiveFilterExpression();
  Flow(ImGui::CalcTextSize("Session filter").x, right, false);
  ImGui::TextColored(kMuted, "%s", filter.empty() ? "No session filter" : "Session filter");
  if (filter.empty()) {
    Tooltip("Set one with: filter set <expression>. It narrows later show logs queries.");
  } else {
    Flow(ImGui::CalcTextSize(filter.c_str()).x + 16.0f, right, false);
    ImGui::TextColored(kAccent, "%s", filter.c_str());
    Tooltip("Combined with every show logs query in this session.");
    Flow(ui::ButtonWidth("Clear filter", ui::ButtonSize::Small), right, false);
    if (ui::SecondaryButton("Clear filter"))
      ExecCommand("filter clear");
  }

  // Running pip command.
  for (const auto &[block, state] : command_blocks_) {
    if (!state.running || state.task_id == 0)
      continue;
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - state.started).count();
    const std::string running = std::string(ICON_FA_SPINNER) + " " + state.command +
                                " \xC2\xB7 " + cmd::FormatElapsed(seconds);
    Flow(ImGui::CalcTextSize(running.c_str()).x, right, false);
    ImGui::TextColored(kAccent, "%s", running.c_str());
    Flow(ui::ButtonWidth("Cancel pip", ui::ButtonSize::Small), right, false);
    ImGui::PushID(static_cast<int>(block));
    if (ui::DangerButton("Cancel pip")) {
      command_blocks_[block].cancelled = true;
      cyxwiz::AsyncTaskManager::Instance().Cancel(state.task_id);
      SetCommandsNotice("Cancelling pip...");
    }
    ImGui::PopID();
    break;
  }

  const std::string copy_label = std::string(ICON_FA_COPY) + " Copy all";
  const std::string clear_label = std::string(ICON_FA_TRASH_CAN) + " Clear";
  const float buttons = ui::ButtonWidth(copy_label.c_str(), ui::ButtonSize::Small) +
                        ui::ButtonWidth(clear_label.c_str(), ui::ButtonSize::Small) +
                        ImGui::GetStyle().ItemSpacing.x;
  Flow(buttons, right, false);
  if (ImGui::GetItemRectMax().y > ImGui::GetCursorScreenPos().y - 1.0f) {
    const float x = right - buttons;
    if (x > ImGui::GetCursorScreenPos().x)
      ImGui::SetCursorScreenPos(ImVec2(x, ImGui::GetCursorScreenPos().y));
  }
  if (ui::SecondaryButton(copy_label.c_str()))
    CopyCommandTranscript();
  Tooltip("Copy every command and its output.");
  ImGui::SameLine();
  if (ui::SecondaryButton(clear_label.c_str()))
    ClearCommandTranscript();
  Tooltip("Clear this transcript (Ctrl+L). Runtime logs are unchanged.");
  ImGui::PopStyleVar();
  ImGui::Separator();
}

void Console::RenderCommandQuickRow() {
  const float right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
  ImGui::AlignTextToFramePadding();
  ImGui::TextColored(kFaint, "Quick");
  ImFont *mono = CommandsMonoFont();
  if (mono)
    ImGui::PushFont(mono);
  for (const char *quick : kQuickCommands) {
    Flow(ui::ChipButtonWidth(quick), right, false);
    if (ui::ChipButton(quick)) {
      ExecCommand(quick);
      command_input_focus_pending_ = true;
    }
  }
  if (mono)
    ImGui::PopFont();
  ImGui::Separator();
}

void Console::RenderCommandTranscript() {
  const auto entries = SnapshotEntries();
  std::unordered_map<uint64_t, std::vector<const LogEntry *>> block_lines;
  std::unordered_set<uint64_t> headers;
  for (const auto &entry : entries) {
    if (entry.is_command)
      headers.insert(entry.block);
    else if (entry.block != 0)
      block_lines[entry.block].push_back(&entry);
  }

  ImFont *mono = CommandsMonoFont();
  if (mono)
    ImGui::PushFont(mono);
  ImGui::Dummy(ImVec2(0.0f, 2.0f));
  ImGui::Indent(6.0f);
  if (entries.empty()) {
    ImGui::PushFont(nullptr);
    ImGui::TextColored(kMuted, "Type a command below or pick a quick command. "
                               "help lists every command.");
    ImGui::PopFont();
  }
  for (const auto &entry : entries) {
    if (entry.is_command) {
      static const std::vector<const LogEntry *> kNone;
      const auto lines = block_lines.find(entry.block);
      RenderCommandBlock(entry, lines == block_lines.end() ? kNone : lines->second);
    } else if (entry.block == 0 || !headers.count(entry.block)) {
      RenderCommandLine(entry, false);
    }
  }
  ImGui::Unindent(6.0f);
  if (mono)
    ImGui::PopFont();

  const bool grew = entries.size() != command_last_entry_count_;
  command_last_entry_count_ = entries.size();
  if (auto_scroll_ && (grew || scroll_to_bottom_.load(std::memory_order_relaxed)))
    ImGui::SetScrollHereY(1.0f);
}

void Console::RenderCommandLine(const LogEntry &entry, bool in_block) {
  const float indent = in_block ? ImGui::CalcTextSize("> ").x + 8.0f : 0.0f;
  if (indent > 0.0f)
    ImGui::Indent(indent);
  ImGui::PushID(static_cast<int>(entry.sequence));

  if (const auto event = cmd::ParseEventLine(entry.message)) {
    const float ch = ImGui::CalcTextSize("0").x;
    const float x0 = ImGui::GetCursorPosX();
    const size_t level = cmd::EventLevelIndex(event->level);
    ImGui::TextColored(kFaint, "#%llu", static_cast<unsigned long long>(event->sequence));
    ImGui::SameLine(x0 + ch * 8.0f);
    const auto when = cmd::ParseUtcTimestamp(event->timestamp);
    ImGui::TextColored(kMuted, "%s",
                       when ? cyxwiz::logs::FormatLocalTime(*when).c_str() : event->time.c_str());
    ImGui::SameLine(x0 + ch * 22.0f);
    ImGui::TextColored(kLevelColors[level], "%s", kLevelNames[level]);
    ImGui::SameLine(x0 + ch * 31.0f);
    ImGui::TextUnformatted(event->category.c_str());
    ImGui::SameLine(x0 + ch * 42.0f);
    ImGui::TextUnformatted(event->source.empty() ? "-" : event->source.c_str());
    ImGui::SameLine(x0 + ch * 52.0f);
    if (!event->extra.empty()) {
      ImGui::TextColored(kFaint, "%s", event->extra.c_str());
      ImGui::SameLine();
    }
    ImGui::TextColored(level >= 4 ? kBright : kText, "%s", event->message.c_str());
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal))
      ImGui::SetTooltip("%s UTC\n%s", event->timestamp.c_str(), entry.message.c_str());
  } else if (const auto summary = cmd::PrettySummary(entry.message)) {
    ImGui::PushFont(nullptr);
    ImGui::TextColored(kMuted, "%s", summary->c_str());
    ImGui::PopFont();
  } else {
    ImVec4 color = kText;
    switch (entry.level) {
    case LogLevel::Warning:
      color = kAmber;
      break;
    case LogLevel::Error:
      color = kRed;
      break;
    case LogLevel::Success:
      color = kGreen;
      break;
    case LogLevel::Debug:
      color = kBlue;
      break;
    case LogLevel::Info:
      break;
    }
    const bool selected = !in_block && selected_command_sequence_ == entry.sequence;
    if (!in_block) {
      ImGui::TextColored(kFaint, "%s", LocalClock(entry.timestamp).c_str());
      ImGui::SameLine();
    }
    // Help structure: "=== Title ===" headings and "Section:" lines stand out.
    const std::string &text = entry.message;
    const bool heading = text.rfind("=== ", 0) == 0;
    const bool section = !text.empty() && text.back() == ':' && text.front() != ' ' &&
                         text.find(' ') == std::string::npos;
    if (entry.level == LogLevel::Info && heading)
      color = kBright;
    else if (entry.level == LogLevel::Info && section)
      color = kAccent;
    ImGui::PushStyleColor(ImGuiCol_Text, selected ? kBright : color);
    ImGui::PushTextWrapPos(0.0f);  // long lines wrap instead of running off
    ImGui::TextUnformatted(text.c_str(), text.c_str() + text.size());
    ImGui::PopTextWrapPos();
    ImGui::PopStyleColor();
    if (!in_block) {
      if (ImGui::IsItemClicked()) {
        selected_command_sequence_ = entry.sequence;
        selected_command_block_ = 0;
      }
      if (ImGui::BeginPopupContextItem("LineMenu")) {
        ImGui::PushFont(nullptr);
        if (ImGui::MenuItem("Copy line")) {
          ImGui::SetClipboardText(entry.message.c_str());
          SetCommandsNotice("Copied line");
        }
        ImGui::PopFont();
        ImGui::EndPopup();
      }
    }
  }
  ImGui::PopID();
  if (indent > 0.0f)
    ImGui::Unindent(indent);
}

void Console::RenderCommandBlock(const LogEntry &command,
                                 const std::vector<const LogEntry *> &lines) {
  const auto found = command_blocks_.find(command.block);
  const CommandBlockState fallback;
  const CommandBlockState &state = found == command_blocks_.end() ? fallback : found->second;
  size_t rows = 0;
  for (const auto *line : lines)
    if (cmd::ParseEventLine(line->message))
      ++rows;
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - state.started).count();
  const auto status =
      cmd::BuildBlockStatus(state.running, state.success, state.cancelled, seconds, rows);

  ImDrawList *dl = ImGui::GetWindowDrawList();
  ImDrawListSplitter splitter;
  splitter.Split(dl, 2);
  splitter.SetCurrentChannel(dl, 1);
  const ImVec2 start = ImGui::GetCursorScreenPos();
  const float full_w = ImGui::GetContentRegionAvail().x;
  ImGui::PushID(static_cast<int>(command.block));
  ImGui::BeginGroup();

  // Header: prompt, command, status, time.
  ImGui::TextColored(kPrompt, ">");
  ImGui::SameLine();
  ImGui::TextColored(kBright, "%s", command.message.c_str());
  const std::string when = LocalClock(command.timestamp);
  ImVec4 status_color = kGreen;
  std::string status_text = status.text;
  switch (status.kind) {
  case cmd::BlockStatus::Kind::Ok:
    status_text = std::string(ICON_FA_CIRCLE_CHECK) + (status.text.empty() ? "" : " ") +
                  status.text;
    break;
  case cmd::BlockStatus::Kind::Error:
    status_color = kRed;
    status_text = std::string(ICON_FA_CIRCLE_XMARK) + " " + status.text;
    break;
  case cmd::BlockStatus::Kind::Running:
    status_color = kAccent;
    status_text = std::string(ICON_FA_SPINNER) + " " + status.text;
    break;
  case cmd::BlockStatus::Kind::Cancelled:
    status_color = kAmber;
    break;
  }
  ImGui::PushFont(nullptr);
  const float status_w = ImGui::CalcTextSize(status_text.c_str()).x +
                         ImGui::CalcTextSize(when.c_str()).x + 16.0f;
  ImGui::SameLine();
  const float x = start.x + full_w - status_w - 8.0f;
  if (x > ImGui::GetCursorScreenPos().x)
    ImGui::SetCursorScreenPos(ImVec2(x, ImGui::GetCursorScreenPos().y));
  ImGui::TextColored(status_color, "%s", status_text.c_str());
  ImGui::SameLine();
  ImGui::TextColored(kFaint, "%s", when.c_str());
  ImGui::PopFont();

  for (const auto *line : lines)
    RenderCommandLine(*line, true);

  // An error names where to find the right form.
  if (status.kind == cmd::BlockStatus::Kind::Error) {
    const std::string name = cmd::CommandName(command.message);
    const auto usage = std::find_if(command_usages_.begin(), command_usages_.end(),
                                    [&name](const auto &item) { return item.first == name; });
    const bool has_usage_line = std::any_of(lines.begin(), lines.end(), [](const auto *line) {
      return line->message.rfind("Usage:", 0) == 0;
    });
    ImGui::Indent(ImGui::CalcTextSize("> ").x + 8.0f);
    ImGui::PushFont(nullptr);
    if (usage != command_usages_.end()) {
      if (!has_usage_line) {
        ImGui::TextColored(kMuted, "Usage:");
        ImGui::SameLine();
        ImGui::TextColored(kAccent, "%s", usage->second.c_str());
      }
      ImGui::TextColored(kMuted, "help %s lists every form and example.", name.c_str());
    } else {
      ImGui::TextColored(kMuted, "Unknown command. help lists every command.");
    }
    ImGui::PopFont();
    ImGui::Unindent(ImGui::CalcTextSize("> ").x + 8.0f);
  }
  ImGui::EndGroup();

  const ImVec2 min(start.x - 6.0f, start.y - 3.0f);
  const ImVec2 max(start.x + full_w - 2.0f, ImGui::GetItemRectMax().y + 3.0f);
  const bool hovered = ImGui::IsMouseHoveringRect(min, max) && ImGui::IsWindowHovered();
  splitter.SetCurrentChannel(dl, 0);
  if (status.kind == cmd::BlockStatus::Kind::Error)
    dl->AddRectFilled(min, max, ImGui::GetColorU32(::gui::CurrentConsolePalette().error_card), 6.0f);
  if (status.kind == cmd::BlockStatus::Kind::Running)
    dl->AddRectFilled(ImVec2(min.x, min.y), ImVec2(min.x + 2.0f, max.y),
                      ImGui::GetColorU32(kPrompt));
  if (selected_command_block_ == command.block)
    dl->AddRectFilled(min, max, ImGui::GetColorU32(::gui::CurrentConsolePalette().selection),
                      6.0f);
  else if (hovered)
    dl->AddRectFilled(min, max, ImGui::GetColorU32(::gui::CurrentConsolePalette().hover), 6.0f);
  splitter.Merge(dl);

  if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
    selected_command_block_ = command.block;
    selected_command_sequence_ = 0;
  }
  if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
    selected_command_block_ = command.block;
    ImGui::OpenPopup("BlockMenu");
  }
  if (ImGui::BeginPopup("BlockMenu")) {
    ImGui::PushFont(nullptr);
    if (ImGui::MenuItem("Copy command and output", "Ctrl+C")) {
      ImGui::SetClipboardText(CommandBlockText(command.block).c_str());
      SetCommandsNotice("Copied command and output");
    }
    if (ImGui::MenuItem("Copy command")) {
      ImGui::SetClipboardText(command.message.c_str());
      SetCommandsNotice("Copied command");
    }
    ImGui::Separator();
    if (ImGui::MenuItem("Run again", nullptr, false, !state.running)) {
      const std::string again = command.message;
      ImGui::PopFont();
      ImGui::EndPopup();
      ImGui::PopID();
      ExecCommand(again.c_str());
      ImGui::Dummy(ImVec2(0.0f, 6.0f));
      return;
    }
    if (ImGui::MenuItem("Edit in prompt")) {
      std::strncpy(input_buf_, command.message.c_str(), sizeof(input_buf_) - 1);
      input_buf_[sizeof(input_buf_) - 1] = '\0';
      command_input_focus_pending_ = true;
      command_cursor_to_end_ = 2;
    }
    ImGui::PopFont();
    ImGui::EndPopup();
  }
  ImGui::PopID();
  ImGui::Dummy(ImVec2(0.0f, 6.0f));
}

std::string Console::CommandBlockText(uint64_t block) const {
  std::string text;
  for (const auto &entry : SnapshotEntries()) {
    if (entry.block != block)
      continue;
    text += entry.is_command ? "> " + entry.message : "  " + entry.message;
    text += '\n';
  }
  return text;
}

void Console::CopyCommandTranscript() {
  const auto entries = SnapshotEntries();
  if (entries.empty()) {
    SetCommandsNotice("Nothing to copy");
    return;
  }
  std::string text;
  for (const auto &entry : entries) {
    if (entry.is_command)
      text += "[" + LocalClock(entry.timestamp) + "] > " + entry.message;
    else if (entry.block != 0)
      text += "  " + entry.message;
    else
      text += "[" + LocalClock(entry.timestamp) + "] " + GetLevelPrefix(entry.level) + " " +
              entry.message;
    text += '\n';
  }
  ImGui::SetClipboardText(text.c_str());
  SetCommandsNotice("Copied the whole transcript");
}

void Console::CopySelectedCommand() {
  if (selected_command_block_ != 0) {
    ImGui::SetClipboardText(CommandBlockText(selected_command_block_).c_str());
    SetCommandsNotice("Copied command and output");
    return;
  }
  for (const auto &entry : SnapshotEntries()) {
    if (entry.sequence == selected_command_sequence_) {
      ImGui::SetClipboardText(entry.message.c_str());
      SetCommandsNotice("Copied line");
      return;
    }
  }
  selected_command_sequence_ = 0;
}

void Console::RenderCommandInput(bool request_focus) {
  if (request_focus)
    command_input_focus_pending_ = true;
  const ImGuiStyle &style = ImGui::GetStyle();
  ImFont *mono = CommandsMonoFont();

  const ImGuiID input_id = ImGui::GetID("##command_input");
  const bool active = ImGui::GetActiveID() == input_id;
  if (active) {
    // Escape would revert the text: it only closes the suggestions.
    if (ImGui::IsKeyPressed(ImGuiKey_Escape, false))
      command_suggest_open_ = false;
    ImGui::SetKeyOwner(ImGuiKey_Escape, ImGui::GetID("##command_keys"),
                       ImGuiInputFlags_LockThisFrame);
    if (command_suggest_open_ && !command_matches_.empty() &&
        (ImGui::IsKeyPressed(ImGuiKey_Enter, false) ||
         ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false))) {
      command_accept_pending_ = true;  // applied in the callback, same frame
    }
  }

  // Box: prompt, input, usage hint, Run.
  const float avail_w = ImGui::GetContentRegionAvail().x;
  const ImVec2 box_min = ImGui::GetCursorScreenPos();
  const float box_h = ImGui::GetFrameHeight() + 12.0f;
  const ImVec2 box_max(box_min.x + avail_w, box_min.y + box_h);
  ImDrawList *dl = ImGui::GetWindowDrawList();
  dl->AddRectFilled(box_min, box_max, ImGui::GetColorU32(kInputBg), 8.0f);

  std::string hint;
  if (const auto usage = cmd::UsageFormFor(command_forms_, input_buf_)) {
    const auto &form = command_forms_[*usage];
    const auto described = std::find_if(
        cyxwiz::RuntimeConsoleCommandService::Descriptors().begin(),
        cyxwiz::RuntimeConsoleCommandService::Descriptors().end(),
        [&form](const auto &d) { return d.name == form.command; });
    hint = form.command + " \xC2\xB7 " +
           (described != cyxwiz::RuntimeConsoleCommandService::Descriptors().end()
                ? std::string(described->description)
                : form.description);
  }
  const char *run_label = "Run";
  const float run_w = ui::ButtonWidth(run_label, ui::ButtonSize::Small);
  const float hint_w = hint.empty() ? 0.0f : ImGui::CalcTextSize(hint.c_str()).x + 12.0f;
  const float prompt_w = 26.0f;

  ImGui::SetCursorScreenPos(ImVec2(box_min.x + 10.0f, box_min.y + 6.0f));
  if (mono)
    ImGui::PushFont(mono);
  ImGui::AlignTextToFramePadding();
  ImGui::TextColored(kPrompt, ">");
  ImGui::SameLine(0.0f, 0.0f);
  ImGui::SetCursorScreenPos(ImVec2(box_min.x + prompt_w, box_min.y + 6.0f));
  ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0, 0, 0, 0));
  ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
  const float input_w =
      std::max(120.0f, avail_w - prompt_w - hint_w - run_w - style.ItemSpacing.x * 2 - 12.0f);
  ImGui::SetNextItemWidth(input_w);
  if (command_input_focus_pending_)
    ImGui::SetKeyboardFocusHere();
  const ImGuiInputTextFlags flags =
      ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CallbackHistory |
      ImGuiInputTextFlags_CallbackCompletion | ImGuiInputTextFlags_CallbackAlways;
  command_accepted_ = false;
  const bool submitted = ImGui::InputTextWithHint(
      "##command_input", "Type a command, e.g. show logs errors", input_buf_,
      IM_ARRAYSIZE(input_buf_), flags, &Console::InputTextCallback, this);
  const bool input_active = ImGui::IsItemActive();
  const ImVec2 input_min = ImGui::GetItemRectMin();
  if (command_input_focus_pending_ && input_active)
    command_input_focus_pending_ = false;
  ImGui::PopStyleVar();
  ImGui::PopStyleColor();

  // Ghost text: the rest of the selected suggestion.
  if (command_suggest_open_ && !command_matches_.empty() && input_active) {
    const auto &form =
        command_forms_[command_matches_[std::clamp(command_match_selected_, 0,
                                                    static_cast<int>(command_matches_.size()) -
                                                        1)]];
    const size_t typed = std::strlen(input_buf_);
    if (form.form.size() > typed) {
      const float text_w = ImGui::CalcTextSize(input_buf_).x;
      dl->AddText(ImVec2(input_min.x + style.FramePadding.x + text_w,
                         input_min.y + style.FramePadding.y),
                  ImGui::GetColorU32(kFaint),
                  form.form.c_str() + typed);
    }
  }
  const float suggest_anchor_y = box_min.y;
  if (mono)
    ImGui::PopFont();

  if (!hint.empty()) {
    ImGui::SameLine();
    ImGui::TextColored(kFaint, "%s", hint.c_str());
  }
  ImGui::SameLine();
  ImGui::SetCursorScreenPos(ImVec2(box_max.x - run_w - 8.0f, box_min.y + 6.0f));
  const bool has_text = input_buf_[0] != '\0';
  const bool run_clicked = ui::PrimaryButton(run_label, has_text, "Type a command first.",
                                             ui::ButtonSize::Small);
  // Neutral border only (owner, 2026-09-27: no focus highlight on the prompt).
  dl->AddRect(box_min, box_max, ImGui::GetColorU32(kBorder), 8.0f);

  if ((submitted && !command_accepted_) || run_clicked) {
    if (input_buf_[0]) {
      const std::string command = input_buf_;
      input_buf_[0] = '\0';
      command_suggest_open_ = false;
      ExecCommand(command.c_str());
    }
    command_input_focus_pending_ = true;
  } else if (command_accepted_) {
    command_input_focus_pending_ = true;
    command_cursor_to_end_ = 2;
  }

  // Keep the suggestion list in step with the text.
  if (!command_accepted_) {
    command_matches_ = cmd::MatchForms(command_forms_, input_buf_);
    if (command_matches_.empty())
      command_suggest_open_ = false;
    else if (input_active && !command_suggest_open_ && command_last_input_ != input_buf_)
      command_suggest_open_ = true;  // typing opens it
    command_match_selected_ =
        std::clamp(command_match_selected_, 0,
                   std::max(0, static_cast<int>(command_matches_.size()) - 1));
  }

  command_last_input_ = input_buf_;

  // Hints row.
  ImGui::SetCursorScreenPos(ImVec2(box_min.x, box_max.y + 6.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(14.0f, 4.0f));
  ImGui::TextColored(kFaint, "Enter run   Up/Down history   Tab complete   "
                             "Ctrl+L clear   Esc close suggestions");
  const char *scroll_label = auto_scroll_ ? "Auto-scroll on" : "Auto-scroll off";
  ImGui::SameLine();
  float notice_x = ImGui::GetCursorScreenPos().x;
  if (!command_notice_.empty()) {
    if (ImGui::GetTime() < command_notice_until_) {
      ImGui::TextColored(kAccent, "%s", command_notice_.c_str());
      ImGui::SameLine();
      notice_x = ImGui::GetCursorScreenPos().x;
    } else {
      command_notice_.clear();
    }
  }
  const float scroll_x = box_max.x - ImGui::CalcTextSize(scroll_label).x;
  if (scroll_x > notice_x)
    ImGui::SetCursorScreenPos(ImVec2(scroll_x, ImGui::GetCursorScreenPos().y));
  ImGui::TextColored(auto_scroll_ ? kFaint : kAmber, "%s", scroll_label);
  if (ImGui::IsItemHovered()) {
    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    ImGui::SetTooltip("Click to turn auto-scroll %s", auto_scroll_ ? "off" : "on");
  }
  if (ImGui::IsItemClicked())
    auto_scroll_ = !auto_scroll_;
  ImGui::PopStyleVar();

  if (command_suggest_open_ && !command_matches_.empty()) {
    ::gui::NextWindowFollowsCurrent();
    ImGui::SetNextWindowPos(ImVec2(box_min.x, suggest_anchor_y - 4.0f), ImGuiCond_Always,
                            ImVec2(0.0f, 1.0f));
    RenderCommandSuggestions();
  }
}

void Console::RenderCommandSuggestions() {
  ImFont *mono = CommandsMonoFont();
  ImGui::PushStyleColor(ImGuiCol_WindowBg, kPanelBg);
  ImGui::PushStyleColor(ImGuiCol_Border, kBorder);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6.0f, 6.0f));
  const ImGuiWindowFlags flags =
      ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
      ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
      ImGuiWindowFlags_NoNav | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDocking;
  if (ImGui::Begin("##command_suggestions", nullptr, flags)) {
    ImGui::BringWindowToDisplayFront(ImGui::GetCurrentWindow());
    float form_w = 0.0f;
    if (mono)
      ImGui::PushFont(mono);
    for (const size_t index : command_matches_)
      form_w = std::max(form_w, ImGui::CalcTextSize(command_forms_[index].form.c_str()).x);
    if (mono)
      ImGui::PopFont();
    for (size_t i = 0; i < command_matches_.size(); ++i) {
      const auto &form = command_forms_[command_matches_[i]];
      const bool selected = static_cast<int>(i) == command_match_selected_;
      ImGui::PushID(static_cast<int>(i));
      const ImVec2 row = ImGui::GetCursorScreenPos();
      if (ImGui::Selectable("##form", selected, 0,
                            ImVec2(form_w + 320.0f, ImGui::GetTextLineHeight() + 2.0f))) {
        std::strncpy(input_buf_, form.insert_text.c_str(), sizeof(input_buf_) - 1);
        input_buf_[sizeof(input_buf_) - 1] = '\0';
        command_suggest_open_ = false;
        command_input_focus_pending_ = true;
        command_cursor_to_end_ = 2;
      }
      if (selected && command_scroll_selected_)
        ImGui::SetScrollHereY();
      ImGui::SetCursorScreenPos(ImVec2(row.x + 6.0f, row.y + 1.0f));
      if (mono)
        ImGui::PushFont(mono);
      ImGui::TextColored(selected ? kBright : kText, "%s", form.form.c_str());
      if (mono)
        ImGui::PopFont();
      ImGui::SameLine(form_w + 24.0f);
      ImGui::TextColored(kMuted, "%s", form.description.c_str());
      ImGui::PopID();
    }
    command_scroll_selected_ = false;
    ImGui::TextColored(kFaint, "Tab or Enter insert \xC2\xB7 Up/Down choose \xC2\xB7 Esc close "
                               "\xC2\xB7 %d form%s",
                       static_cast<int>(command_matches_.size()),
                       command_matches_.size() == 1 ? "" : "s");
  }
  ImGui::End();
  ImGui::PopStyleVar(2);
  ImGui::PopStyleColor(2);
}

int Console::InputTextCallback(ImGuiInputTextCallbackData *data) {
  return static_cast<Console *>(data->UserData)->HandleInputTextCallback(data);
}

int Console::HandleInputTextCallback(ImGuiInputTextCallbackData *data) {
  const auto insert = [data](const std::string &text) {
    data->DeleteChars(0, data->BufTextLen);
    data->InsertChars(0, text.c_str());
  };
  const bool list_open = command_suggest_open_ && !command_matches_.empty();

  if (data->EventFlag == ImGuiInputTextFlags_CallbackHistory) {
    if (list_open) {
      const int count = static_cast<int>(command_matches_.size());
      command_match_selected_ = (command_match_selected_ +
                                 (data->EventKey == ImGuiKey_UpArrow ? count - 1 : 1)) %
                                count;
      command_scroll_selected_ = true;
      return 0;
    }
    const auto command = data->EventKey == ImGuiKey_UpArrow
                             ? command_service_->PreviousCommand()
                             : command_service_->NextCommand();
    if (command)
      insert(*command);
    return 0;
  }

  if (data->EventFlag == ImGuiInputTextFlags_CallbackCompletion) {
    const auto matches = cmd::MatchForms(command_forms_, std::string(data->Buf, data->BufTextLen));
    if (list_open) {
      insert(command_forms_[command_matches_[command_match_selected_]].insert_text);
      command_suggest_open_ = false;
    } else if (matches.size() == 1) {
      insert(command_forms_[matches.front()].insert_text);
    } else if (!matches.empty()) {
      command_matches_ = matches;
      command_match_selected_ = 0;
      command_suggest_open_ = true;
    } else {
      SetCommandsNotice("No command starts with that");
    }
    return 0;
  }

  if (data->EventFlag == ImGuiInputTextFlags_CallbackAlways) {
    if (command_accept_pending_) {
      command_accept_pending_ = false;
      if (list_open) {
        insert(command_forms_[command_matches_[command_match_selected_]].insert_text);
        command_suggest_open_ = false;
        command_accepted_ = true;
      }
    }
    if (command_cursor_to_end_ > 0) {
      --command_cursor_to_end_;
      data->CursorPos = data->BufTextLen;
      data->SelectionStart = data->SelectionEnd = data->CursorPos;
    }
  }
  return 0;
}

} // namespace gui
