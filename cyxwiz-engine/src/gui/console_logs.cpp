// Console "Logs" session: toolbar, status line, runtime-log table and details
// drawer (tofix121). Wording and formatting live in
// core/runtime_log_presentation; the query worker, saved views and export
// dialog stay in console.cpp.

#include "console_palette.h"
#include "console.h"
#include "../core/console_commands_presentation.h"
#include "../core/runtime_log_export.h"
#include "../core/runtime_log_presentation.h"
#include "../core/runtime_log_store.h"
#include "editor_fonts.h"
#include "icons.h"
#include "ui_buttons.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <imgui.h>
#include <unordered_map>
#include <string>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace ui = cyxwiz::ui;

namespace {

// Palette shared with the REPL redesign (tofix121 mockups).
ImVec4 kMuted;
ImVec4 kFaint;
ImVec4 kText;
ImVec4 kBright;
ImVec4 kLive;
ImVec4 kAmber;
ImVec4 kRed;
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
  kLive = p.success;
  kAmber = p.warning;
  kRed = p.error;
  kAccent = p.accent_text;
  kPrompt = p.accent;
  kPanelBg = p.panel;
  kInputBg = p.input;
  kBorder = p.border;
  kLevelColors = {p.muted, p.info, p.text, p.warning, p.error, p.critical};
}

constexpr std::array<const char *, 6> kLevelIcons = {
    ICON_FA_CIRCLE, ICON_FA_CIRCLE, ICON_FA_CIRCLE_INFO,
    ICON_FA_TRIANGLE_EXCLAMATION, ICON_FA_CIRCLE_XMARK, ICON_FA_CIRCLE_XMARK};
constexpr std::array<const char *, 6> kLevelHelp = {
    "Most detailed severity. Supported by the engine, but normally filtered "
    "because Release starts at Info and Debug starts at Debug.",
    "Developer diagnostic severity. Enabled by the default Debug build.",
    "Normal runtime progress and lifecycle information.",
    "A recoverable problem or condition requiring attention.",
    "An operation failed but the process may continue.",
    "Highest severity for fatal or process-ending failures."};

void Tooltip(const char *text, ImGuiHoveredFlags flags = ImGuiHoveredFlags_DelayNormal) {
  if (ImGui::IsItemHovered(flags | ImGuiHoveredFlags_AllowWhenDisabled))
    ImGui::SetTooltip("%s", text);
}

template <size_t Size> void CopyToBuffer(char (&buffer)[Size], const std::string &value) {
  static_assert(Size > 0);
  std::strncpy(buffer, value.c_str(), Size - 1);
  buffer[Size - 1] = '\0';
}

bool IsValidSavedViewName(const std::string &name) {
  return !name.empty() && name.size() <= 63 &&
         std::all_of(name.begin(), name.end(), [](unsigned char value) {
           return std::isalnum(value) != 0 || value == '_' || value == '-';
         });
}

std::string BuildSavedViewExpression(const cyxwiz::RuntimeLogInspectorCriteria &criteria) {
  auto controls_only = criteria;
  controls_only.structured_filter.clear();
  if (cyxwiz::BuildRuntimeLogInspectorFilter(controls_only).empty())
    return criteria.structured_filter;
  return cyxwiz::BuildRuntimeLogInspectorFilter(criteria);
}

// Places the next item on the current line when it fits before `right`,
// otherwise on a new line; `first` marks the first item of a row.
void Flow(float width, float right, bool first) {
  if (first)
    return;
  const float spacing = ImGui::GetStyle().ItemSpacing.x;
  const float x = ImGui::GetItemRectMax().x + spacing;
  if (x + width <= right)
    ImGui::SameLine();
}

// Wraps `text` at `width` (current font) so a read-only multiline field shows
// the whole message without horizontal scrolling.
// The first two lines of `text` wrapped at `width`; `more` when it goes on.
struct TwoLines {
  std::string first, second;
  bool more = false;
};
TwoLines WrapTwoLines(const std::string &text, float width) {
  TwoLines lines;
  ImFont *font = ImGui::GetFont();
  const float scale = ImGui::GetFontSize() / font->FontSize;
  const char *begin = text.c_str();
  const char *end = begin + text.size();
  const auto next_line = [&](const char *&cursor) {
    const char *newline = static_cast<const char *>(std::memchr(cursor, '\n', static_cast<size_t>(end - cursor)));
    const char *limit = newline ? newline : end;
    const char *wrap = font->CalcWordWrapPositionA(scale, cursor, limit, std::max(20.0f, width));
    if (wrap == cursor) wrap = std::min(limit, cursor + 1);
    std::string line(cursor, wrap);
    cursor = wrap;
    while (cursor < end && (*cursor == ' ' || *cursor == '\n')) {
      const bool was_newline = *cursor == '\n';
      ++cursor;
      if (was_newline) break;
    }
    return line;
  };
  const char *cursor = begin;
  lines.first = next_line(cursor);
  if (cursor < end) lines.second = next_line(cursor);
  lines.more = cursor < end;
  return lines;
}

std::string WrapForWidth(const std::string &text, float width) {
  if (width <= 0.0f)
    return text;
  ImFont *font = ImGui::GetFont();
  const float size = ImGui::GetFontSize();
  std::string out;
  const char *line = text.c_str();
  const char *end = line + text.size();
  while (line < end) {
    const char *eol = static_cast<const char *>(std::memchr(line, '\n', end - line));
    const char *line_end = eol ? eol : end;
    const char *cursor = line;
    while (cursor < line_end) {
      const char *wrap = font->CalcWordWrapPositionA(size / font->FontSize, cursor,
                                                     line_end, width);
      if (wrap == cursor)
        wrap = cursor + 1;
      out.append(cursor, wrap);
      cursor = wrap;
      while (cursor < line_end && *cursor == ' ')
        ++cursor;
      if (cursor < line_end)
        out += '\n';
    }
    if (eol) {
      out += '\n';
      line = eol + 1;
    } else {
      line = end;
    }
  }
  return out;
}

} // namespace

namespace gui {

bool OpenPathWithSystem(const std::filesystem::path &path, bool reveal) {
  if (path.empty())
    return false;
  const std::string target = path.string();
#ifdef _WIN32
  HINSTANCE result;
  if (reveal) {
    const std::string params = "/select,\"" + target + "\"";
    result = ShellExecuteA(nullptr, "open", "explorer.exe", params.c_str(), nullptr,
                           SW_SHOWNORMAL);
  } else {
    result = ShellExecuteA(nullptr, "open", target.c_str(), nullptr, nullptr,
                           SW_SHOWNORMAL);
  }
  return reinterpret_cast<INT_PTR>(result) > 32;
#else
  const pid_t pid = fork();
  if (pid == 0) {
#ifdef __APPLE__
    if (reveal)
      execlp("open", "open", "-R", target.c_str(), static_cast<char *>(nullptr));
    else
      execlp("open", "open", target.c_str(), static_cast<char *>(nullptr));
#else
    const std::string open_target =
        reveal ? path.parent_path().string() : target;
    execlp("xdg-open", "xdg-open", open_target.c_str(), static_cast<char *>(nullptr));
#endif
    _exit(127);
  }
  int status = 0;
  return pid > 0 && waitpid(pid, &status, 0) == pid && WIFEXITED(status) &&
         WEXITSTATUS(status) == 0;
#endif
}

void Console::SetLogsNotice(std::string message) {
  logs_notice_ = std::move(message);
  logs_notice_until_ = ImGui::GetTime() + 2.5;
}

void Console::UpdateLogsProblemBadge() {
  const double now = ImGui::GetTime();
  if (now < logs_badge_next_check_)
    return;
  logs_badge_next_check_ = now + 0.5;

  std::optional<std::uint64_t> logs_id;
  for (const auto &session : workbench_.Sessions()) {
    if (session.kind == ConsoleSessionKind::Logs) {
      logs_id = session.id;
      break;
    }
  }
  if (!logs_id)
    return;
  auto &store = cyxwiz::RuntimeLogStore::Instance();
  const uint64_t newest = store.GetStats().newest_sequence;
  const bool logs_visible = show_window_ && workbench_.ActiveSessionId() == logs_id;
  if (logs_visible || logs_badge_checked_sequence_ == 0) {
    logs_badge_checked_sequence_ = newest;
    logs_problem_badge_ = 0;
    workbench_.SetProblemBadge(*logs_id, 0);
    return;
  }
  if (newest <= logs_badge_checked_sequence_)
    return;
  cyxwiz::RuntimeLogSnapshotRequest request;
  request.after_sequence = logs_badge_checked_sequence_;
  request.limit = store.GetStats().capacity;
  const auto snapshot = store.Snapshot(request);
  for (const auto &event : snapshot.events) {
    if (cyxwiz::logs::IsProblemLevel(event.level))
      ++logs_problem_badge_;
  }
  logs_badge_checked_sequence_ = newest;
  workbench_.SetProblemBadge(*logs_id, logs_problem_badge_);
}

void Console::RenderLogsSession(bool request_focus) {
  RefreshPalette();
  const double now = ImGui::GetTime();
  if (now >= inspector_next_refresh_time_) {
    RequestInspectorQuery(false);
    inspector_next_refresh_time_ = now + 0.1;
  }
  RenderInspectorTab(request_focus);

  const ImGuiIO &io = ImGui::GetIO();
  if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) &&
      !io.WantTextInput) {
    if (inspector_selected_sequence_ != 0 && io.KeyCtrl &&
        ImGui::IsKeyPressed(ImGuiKey_C, false)) {
      CopySelectedRuntimeLog();
    }
    if (inspector_selected_sequence_ != 0 && ImGui::IsKeyPressed(ImGuiKey_Escape, false))
      inspector_selected_sequence_ = 0;
  }
}

void Console::RenderInspectorTab(bool request_focus) {
  if (pending_scroll_selection_ != 0) {
    inspector_scroll_to_selected_ = pending_scroll_selection_;
    pending_scroll_selection_ = 0;
  }
  const auto result = SnapshotInspectorResult();
  RenderInspectorFilters(result.get(), request_focus);
  RenderLogsStatus(result.get());
  RenderRuntimeLogExportDialog();

  if (!result) {
    ImGui::TextDisabled("Loading runtime events...");
    return;
  }

  const auto &events = result->query.events;
  const bool has_selection =
      std::any_of(events.begin(), events.end(), [this](const auto &event) {
        return event.sequence == inspector_selected_sequence_;
      });
  // Details sit beside the list when there is room, so the list keeps its
  // height (TOFIX159 B3); under it otherwise.
  const float avail = ImGui::GetContentRegionAvail().y;
  const float avail_w = ImGui::GetContentRegionAvail().x;
  const bool beside = has_selection && avail_w >= 1000.0f;
  if (beside) {
    const float details_w = std::clamp(avail_w * 0.34f, 360.0f, 520.0f);
    ImGui::BeginChild("RuntimeLogInspectorBody",
                      ImVec2(avail_w - details_w - ImGui::GetStyle().ItemSpacing.x, 0), false);
    RenderInspectorTable(result.get());
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginGroup();
    RenderInspectorDetails(result.get());
    ImGui::EndGroup();
    return;
  }
  const float drawer_h =
      has_selection ? std::clamp(avail * 0.40f, 150.0f, 380.0f) : 0.0f;
  const float table_h =
      has_selection ? std::max(80.0f, avail - drawer_h - ImGui::GetStyle().ItemSpacing.y)
                    : 0.0f;

  ImGui::BeginChild("RuntimeLogInspectorBody", ImVec2(0, table_h), false);
  RenderInspectorTable(result.get());
  ImGui::EndChild();
  if (has_selection)
    RenderInspectorDetails(result.get());
}

void Console::RenderInspectorFilters(const cyxwiz::RuntimeLogInspectorResult *result,
                                     bool request_focus) {
  if (request_focus)
    log_search_focus_pending_ = true;
  bool changed = false;
  const ImGuiStyle &style = ImGui::GetStyle();

  // ---- Row 1: search, severity chips, Pause, Filters, Actions ----
  const std::string pause_label = inspector_paused_
                                      ? std::string(ICON_FA_PLAY) + " Resume"
                                      : std::string(ICON_FA_PAUSE) + " Pause";
  const size_t field_filter_count = cyxwiz::logs::CountFieldFilters(
      !inspector_criteria_.category.empty(), !inspector_criteria_.source.empty(),
      !inspector_criteria_.code.empty(), !inspector_criteria_.run_id.empty(),
      !inspector_criteria_.backend.empty(), inspector_criteria_.task_id.has_value(),
      inspector_criteria_.device_id.has_value());
  const std::string filters_label =
      std::string(ICON_FA_FILTER) + (field_filter_count == 0
                                         ? " Filters"
                                         : " Filters (" +
                                               std::to_string(field_filter_count) + ")");
  const std::string actions_label = std::string("Actions ") + ICON_FA_CHEVRON_DOWN;

  std::array<std::string, 6> counts;
  float chips_w = 0.0f;
  for (size_t i = 0; i < counts.size(); ++i) {
    counts[i] = result ? cyxwiz::logs::FormatCount(result->query.facets.level_counts[i])
                       : std::string("-");
    chips_w += ui::ToggleChipWidth(cyxwiz::logs::kLevelLabels[i], counts[i].c_str()) +
               6.0f;
  }
  const float buttons_w = ui::ButtonWidth(pause_label.c_str(), ui::ButtonSize::Small) +
                          ui::ButtonWidth(filters_label.c_str(), ui::ButtonSize::Small) +
                          ui::ButtonWidth(actions_label.c_str(), ui::ButtonSize::Small) +
                          style.ItemSpacing.x * 3.0f;
  const float row_left = ImGui::GetCursorScreenPos().x;
  const float right = row_left + ImGui::GetContentRegionAvail().x;
  float search_w = right - row_left - chips_w - buttons_w - style.ItemSpacing.x;
  if (search_w < 220.0f)
    search_w = right - row_left;  // chips and buttons move to the next line

  ImGui::PushStyleColor(ImGuiCol_FrameBg, kInputBg);
  ImGui::SetNextItemWidth(search_w);
  if (log_search_focus_pending_)
    ImGui::SetKeyboardFocusHere();
  const std::string search_hint = std::string(ICON_FA_MAGNIFYING_GLASS) + "  Search messages";
  if (ImGui::InputTextWithHint("##runtime_text", search_hint.c_str(), inspector_text_,
                               IM_ARRAYSIZE(inspector_text_))) {
    inspector_criteria_.text = inspector_text_;
    changed = true;
  }
  ImGui::PopStyleColor();
  if (log_search_focus_pending_ && ImGui::IsItemActive())
    log_search_focus_pending_ = false;
  Tooltip("Case-insensitive search in messages.");

  for (size_t i = 0; i < counts.size(); ++i) {
    const char *label = cyxwiz::logs::kLevelLabels[i];
    const float w = ui::ToggleChipWidth(label, counts[i].c_str());
    if (ImGui::GetItemRectMax().x + 6.0f + w <= right)
      ImGui::SameLine(0.0f, 6.0f);
    const std::string id = std::string("##chip_") + label;
    const bool emphasise = i >= 4 && result && result->query.facets.level_counts[i] > 0;
    if (ui::ToggleChip(id.c_str(), label, counts[i].c_str(), inspector_criteria_.levels[i],
                       ImGui::GetColorU32(kLevelColors[i]), emphasise)) {
      inspector_criteria_.levels[i] = !inspector_criteria_.levels[i];
      changed = true;
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
      ImGui::SetTooltip("%s\n%s events retained in this view. Click to %s.", kLevelHelp[i],
                        counts[i].c_str(), inspector_criteria_.levels[i] ? "hide" : "show");
    }
  }

  Flow(ui::ButtonWidth(pause_label.c_str(), ui::ButtonSize::Small), right, false);
  if (ui::SecondaryButton(pause_label.c_str())) {
    inspector_paused_ = !inspector_paused_;
    if (inspector_paused_) {
      inspector_frozen_sequence_ =
          cyxwiz::RuntimeLogStore::Instance().GetStats().newest_sequence;
    }
    changed = true;
  }
  Tooltip(inspector_paused_
              ? "Resume the live log tail from the current runtime high-water mark."
              : "Freeze this view at its current high-water mark while ingestion "
                "continues.");
  Flow(ui::ButtonWidth(filters_label.c_str(), ui::ButtonSize::Small), right, false);
  if (ui::SecondaryButton(filters_label.c_str()))
    ImGui::OpenPopup("RuntimeLogAdvancedFilters");
  Tooltip("Category, source, code, run, backend, task and device filters, and "
          "saved-view management.");
  Flow(ui::ButtonWidth(actions_label.c_str(), ui::ButtonSize::Small), right, false);
  if (ui::SecondaryButton(actions_label.c_str()))
    ImGui::OpenPopup("RuntimeLogActions");

  // ---- Row 2: structured filter, help, saved view, auto-scroll ----
  const bool has_saved_selection =
      inspector_selected_saved_filter_ >= 0 &&
      inspector_selected_saved_filter_ < static_cast<int>(inspector_saved_filters_.size());
  std::string saved_preview = "Custom";
  if (has_saved_selection) {
    const auto &saved = inspector_saved_filters_[inspector_selected_saved_filter_];
    saved_preview = saved.name;
    if (BuildSavedViewExpression(inspector_criteria_) != saved.expression)
      saved_preview += " (modified)";
  }
  const float view_w = 200.0f;
  const float auto_w = ImGui::GetFrameHeight() + style.ItemInnerSpacing.x +
                       ImGui::CalcTextSize("Auto-scroll").x;
  const float trailing = ui::ButtonWidth("?", ui::ButtonSize::Small) +
                         ImGui::CalcTextSize("View").x + view_w + auto_w +
                         style.ItemSpacing.x * 5.0f;
  float filter_w = right - row_left - trailing;
  if (filter_w < 260.0f)
    filter_w = right - row_left;
  ImGui::PushStyleColor(ImGuiCol_FrameBg, kInputBg);
  ImGui::PushStyleColor(ImGuiCol_Text, inspector_criteria_.structured_filter.empty()
                                           ? kText
                                           : kAccent);
  ImGui::SetNextItemWidth(filter_w);
  if (ImGui::InputTextWithHint("##runtime_filter",
                               "Structured filter, e.g. level >= warn and category = \"device\"",
                               inspector_filter_, IM_ARRAYSIZE(inspector_filter_))) {
    inspector_criteria_.structured_filter = inspector_filter_;
    changed = true;
  }
  ImGui::PopStyleColor(2);
  Flow(ui::ButtonWidth("?", ui::ButtonSize::Small), right, false);
  if (ui::SecondaryButton("?##StructuredFilterHelp"))
    ImGui::OpenPopup("StructuredFilterHelp");
  Tooltip("Structured filter syntax and examples.");
  Flow(ImGui::CalcTextSize("View").x + view_w + style.ItemSpacing.x, right, false);
  ImGui::AlignTextToFramePadding();
  ImGui::TextColored(kMuted, "View");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(view_w);
  if (ImGui::BeginCombo("##SavedRuntimeLogView", saved_preview.c_str())) {
    if (ImGui::Selectable("Custom", !has_saved_selection)) {
      inspector_selected_saved_filter_ = -1;
      inspector_filter_name_[0] = '\0';
      inspector_filter_status_.clear();
    }
    for (size_t index = 0; index < inspector_saved_filters_.size(); ++index) {
      const auto &saved = inspector_saved_filters_[index];
      const std::string label =
          saved.validation_error.empty() ? saved.name : saved.name + " (invalid)";
      if (ImGui::Selectable(label.c_str(),
                            inspector_selected_saved_filter_ == static_cast<int>(index))) {
        inspector_selected_saved_filter_ = static_cast<int>(index);
        inspector_criteria_ = {};
        inspector_text_[0] = '\0';
        inspector_criteria_.structured_filter = saved.expression;
        CopyToBuffer(inspector_filter_, saved.expression);
        inspector_filter_name_[0] = '\0';
        inspector_filter_status_ = "Applied saved view '" + saved.name + "'";
        inspector_filter_status_error_ = false;
        changed = true;
      }
    }
    if (inspector_saved_filters_.empty())
      ImGui::TextDisabled("No saved views yet. Save one from Filters.");
    ImGui::EndCombo();
  }
  Tooltip("Saved views apply a complete filter (search, severity, fields and "
          "structured filter). Manage them under Filters.");
  Flow(auto_w, right, false);
  ImGui::Checkbox("Auto-scroll", &logs_auto_scroll_);
  Tooltip("Follow newly received runtime events.");

  // ---- Popups ----
  ImGui::SetNextWindowSize(ImVec2(680.0f, 560.0f), ImGuiCond_Appearing);
  if (ImGui::BeginPopup("StructuredFilterHelp")) {
    ImGui::TextUnformatted("Structured filter help");
    ImGui::Separator();
    ImGui::BeginChild("StructuredFilterHelpContent", ImVec2(0, 0), false);
    const auto help = cyxwiz::RuntimeLogFilterHelpText();
    ImGui::TextUnformatted(help.data(), help.data() + help.size());
    ImGui::EndChild();
    ImGui::EndPopup();
  }

  if (ImGui::BeginPopup("RuntimeLogActions")) {
    const bool has_row = inspector_selected_sequence_ != 0;
    if (ImGui::MenuItem("Copy selected row", "Ctrl+C", false, has_row))
      CopySelectedRuntimeLog();
    Tooltip(has_row ? "Copy the selected row with all its fields."
                    : "Select a row first.");
    if (ImGui::MenuItem("Copy filtered rows", nullptr, false,
                        result && !result->query.events.empty())) {
      CopyFilteredRuntimeLogs();
    }
    Tooltip("Copy the displayed filtered rows (up to the 1,000-row display limit).");
    const bool export_running = IsRuntimeLogExportRunning();
    if (ImGui::MenuItem("Export...", nullptr, false, !export_running))
      OpenRuntimeLogExportDialog();
    Tooltip(export_running ? "An export is already running."
                           : "Export the frozen filtered rows or the selected row as "
                             "JSON Lines or readable text, with a redaction preview.");
    ImGui::Separator();
    if (ImGui::MenuItem("Clear view"))
      ClearLogView();
    Tooltip("Hide retained events through the current high-water mark. Runtime "
            "evidence is not deleted; Show retained brings it back.");
    if (ImGui::MenuItem("Show retained", nullptr, false, inspector_after_sequence_ != 0)) {
      inspector_after_sequence_ = 0;
      changed = true;
    }
    Tooltip(inspector_after_sequence_ != 0
                ? "Show retained events hidden by Clear view. Store eviction still applies."
                : "Nothing is hidden.");
    if (ImGui::MenuItem("Reset filters")) {
      inspector_criteria_ = {};
      inspector_text_[0] = '\0';
      inspector_filter_[0] = '\0';
      inspector_filter_name_[0] = '\0';
      inspector_selected_saved_filter_ = -1;
      inspector_selected_sequence_ = 0;
      inspector_filter_status_.clear();
      changed = true;
    }
    Tooltip("Reset search, severity, field and structured filters and the selection.");
    ImGui::Separator();
    const bool log_exists = !log_file_path_.empty() && std::filesystem::exists(log_file_path_);
    if (ImGui::MenuItem("Open log file", nullptr, false, log_exists)) {
      if (!OpenPathWithSystem(log_file_path_, false))
        SetLogsNotice("Could not open " + log_file_path_.string());
    }
    Tooltip(log_exists ? log_file_path_.string().c_str()
                       : "engine_log.txt was not found next to the Engine.");
    if (ImGui::MenuItem("Show log file in folder", nullptr, false, log_exists)) {
      if (!OpenPathWithSystem(log_file_path_, true))
        SetLogsNotice("Could not open the log folder");
    }
    ImGui::EndPopup();
  }

  ImGui::SetNextWindowSize(ImVec2(540.0f, 0.0f), ImGuiCond_Appearing);
  if (ImGui::BeginPopup("RuntimeLogAdvancedFilters")) {
    ImGui::TextUnformatted("Field filters");
    ImGui::TextDisabled("Each selection is combined with AND.");
    if (result &&
        ImGui::BeginTable("RuntimeLogFieldFilters", 2, ImGuiTableFlags_SizingStretchSame)) {
      const auto string_combo = [&changed](const char *label, const char *id,
                                           std::string &selected,
                                           const std::vector<std::string> &values) {
        ImGui::TableNextColumn();
        ImGui::TextDisabled("%s", label);
        ImGui::SetNextItemWidth(-1.0f);
        if (!ImGui::BeginCombo(id, selected.empty() ? "Any" : selected.c_str()))
          return;
        if (ImGui::Selectable("Any", selected.empty())) {
          selected.clear();
          changed = true;
        }
        for (const auto &value : values) {
          if (ImGui::Selectable(value.c_str(), selected == value)) {
            selected = value;
            changed = true;
          }
        }
        ImGui::EndCombo();
      };
      const auto &facets = result->query.facets;
      string_combo("Category", "##FilterCategory", inspector_criteria_.category,
                   facets.categories);
      string_combo("Source", "##FilterSource", inspector_criteria_.source, facets.sources);
      string_combo("Code", "##FilterCode", inspector_criteria_.code, facets.codes);
      string_combo("Run", "##FilterRun", inspector_criteria_.run_id, facets.run_ids);
      string_combo("Backend", "##FilterBackend", inspector_criteria_.backend,
                   facets.backends);

      ImGui::TableNextColumn();
      ImGui::TextDisabled("Task");
      ImGui::SetNextItemWidth(-1.0f);
      const std::string task_preview =
          inspector_criteria_.task_id ? std::to_string(*inspector_criteria_.task_id) : "Any";
      if (ImGui::BeginCombo("##FilterTask", task_preview.c_str())) {
        if (ImGui::Selectable("Any", !inspector_criteria_.task_id)) {
          inspector_criteria_.task_id.reset();
          changed = true;
        }
        for (const auto value : facets.task_ids) {
          const auto label = std::to_string(value);
          if (ImGui::Selectable(label.c_str(), inspector_criteria_.task_id == value)) {
            inspector_criteria_.task_id = value;
            changed = true;
          }
        }
        ImGui::EndCombo();
      }

      ImGui::TableNextColumn();
      ImGui::TextDisabled("Device");
      ImGui::SetNextItemWidth(-1.0f);
      const std::string device_preview = inspector_criteria_.device_id
                                             ? std::to_string(*inspector_criteria_.device_id)
                                             : "Any";
      if (ImGui::BeginCombo("##FilterDevice", device_preview.c_str())) {
        if (ImGui::Selectable("Any", !inspector_criteria_.device_id)) {
          inspector_criteria_.device_id.reset();
          changed = true;
        }
        for (const auto value : facets.device_ids) {
          const auto label = std::to_string(value);
          if (ImGui::Selectable(label.c_str(), inspector_criteria_.device_id == value)) {
            inspector_criteria_.device_id = value;
            changed = true;
          }
        }
        ImGui::EndCombo();
      }
      ImGui::EndTable();
    } else if (!result) {
      ImGui::TextDisabled("Log fields are still loading.");
    }
    if (field_filter_count > 0 && ui::LinkButton("Clear field filters")) {
      inspector_criteria_.category.clear();
      inspector_criteria_.source.clear();
      inspector_criteria_.code.clear();
      inspector_criteria_.run_id.clear();
      inspector_criteria_.backend.clear();
      inspector_criteria_.task_id.reset();
      inspector_criteria_.device_id.reset();
      changed = true;
    }

    ImGui::Spacing();
    ImGui::SeparatorText("Saved views");
    ImGui::TextDisabled("A saved view captures search, severity, fields and structured "
                        "filter. Pick one from View.");
    ImGui::Text("Selected: %s", saved_preview.c_str());
    ImGui::TextDisabled("New view name");
    ImGui::SetNextItemWidth(-1.0f);
    ImGui::InputTextWithHint("##SavedRuntimeLogViewName",
                             "View name (letters, digits, _ or -)", inspector_filter_name_,
                             IM_ARRAYSIZE(inspector_filter_name_));

    const auto set_status = [this](std::string message, bool error) {
      inspector_filter_status_ = std::move(message);
      inspector_filter_status_error_ = error;
    };
    const auto validate_current = [this, &set_status](std::string &expression) {
      expression = BuildSavedViewExpression(inspector_criteria_);
      if (expression.empty()) {
        set_status("Set at least one filter before saving a view.", true);
        return false;
      }
      const auto parsed = cyxwiz::ParseRuntimeLogFilter(expression);
      if (!parsed.Ok()) {
        set_status("Fix the active filter before saving this view.", true);
        return false;
      }
      return true;
    };

    if (ui::SecondaryButton("Save As")) {
      const std::string name = inspector_filter_name_;
      std::string expression;
      const auto existing =
          std::find_if(inspector_saved_filters_.begin(), inspector_saved_filters_.end(),
                       [&name](const auto &saved) { return saved.name == name; });
      if (!IsValidSavedViewName(name)) {
        set_status("View name must use letters, digits, '_' or '-' (max 63).", true);
      } else if (existing != inspector_saved_filters_.end()) {
        set_status("That view already exists. Select it and use Update.", true);
      } else if (inspector_saved_filters_.size() >= 32) {
        set_status("At most 32 saved views are allowed.", true);
      } else if (validate_current(expression)) {
        inspector_saved_filters_.push_back({name, expression, {}});
        inspector_selected_saved_filter_ =
            static_cast<int>(inspector_saved_filters_.size() - 1);
        if (PersistSavedInspectorFilters())
          set_status("Saved view '" + name + "'", false);
      }
    }
    Tooltip("Create a new view from every currently active filter.");
    ImGui::SameLine();
    if (ui::SecondaryButton("Update", has_saved_selection, "Select a saved view first.") &&
        has_saved_selection) {
      std::string expression;
      if (validate_current(expression)) {
        auto &saved = inspector_saved_filters_[inspector_selected_saved_filter_];
        saved.expression = std::move(expression);
        saved.validation_error.clear();
        if (PersistSavedInspectorFilters())
          set_status("Updated view '" + saved.name + "'", false);
      }
    }
    if (has_saved_selection)
      Tooltip("Replace the selected view with every currently active filter.");
    ImGui::SameLine();
    if (ui::DangerButton("Delete", has_saved_selection, "Select a saved view first.") &&
        has_saved_selection) {
      const std::string name = inspector_saved_filters_[inspector_selected_saved_filter_].name;
      inspector_saved_filters_.erase(inspector_saved_filters_.begin() +
                                     inspector_selected_saved_filter_);
      inspector_selected_saved_filter_ = -1;
      inspector_filter_name_[0] = '\0';
      if (PersistSavedInspectorFilters())
        set_status("Deleted view '" + name + "'", false);
    }
    if (has_saved_selection)
      Tooltip("Delete the selected saved view. The active filter remains applied.");

    if (!inspector_filter_status_.empty()) {
      ImGui::PushStyleColor(ImGuiCol_Text, inspector_filter_status_error_ ? kRed : kLive);
      ImGui::TextWrapped("%s", inspector_filter_status_.c_str());
      ImGui::PopStyleColor();
    }
    const bool still_selected =
        inspector_selected_saved_filter_ >= 0 &&
        inspector_selected_saved_filter_ < static_cast<int>(inspector_saved_filters_.size());
    if (still_selected &&
        !inspector_saved_filters_[inspector_selected_saved_filter_].validation_error.empty()) {
      ImGui::TextWrapped(
          "Saved view error: %s",
          inspector_saved_filters_[inspector_selected_saved_filter_].validation_error.c_str());
    }
    ImGui::EndPopup();
  }

  if (changed)
    RequestInspectorQuery(true);
}

void Console::RenderLogsStatus(const cyxwiz::RuntimeLogInspectorResult *result) {
  ImGui::Separator();
  if (result && result->filter_error) {
    ImGui::PushStyleColor(ImGuiCol_Text, kRed);
    ImGui::TextWrapped("%s Filter error at position %zu: %s", ICON_FA_TRIANGLE_EXCLAMATION,
                       result->filter_error->position,
                       result->filter_error->message.c_str());
    ImGui::PopStyleColor();
  }
  if (result && !result->effective_filter.empty()) {
    constexpr size_t kMaxVisible = 140;
    const std::string visible =
        result->effective_filter.size() <= kMaxVisible
            ? result->effective_filter
            : result->effective_filter.substr(0, kMaxVisible - 3) + "...";
    ImGui::TextColored(kFaint, "Active filter: %s", visible.c_str());
    Tooltip(result->effective_filter.c_str());
  }

  const float right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(14.0f, 4.0f));
  if (result) {
    const auto &query = result->query;
    cyxwiz::logs::StatusInput in;
    in.stats = query.store_stats;
    in.shown = query.events.size();
    in.matched = query.matched_count;
    in.high_water = query.high_water_sequence;
    in.truncated = query.truncated;
    in.paused = inspector_paused_;
    in.hidden_through = inspector_after_sequence_;
    const auto view = cyxwiz::logs::BuildStatusView(in);

    bool first = true;
    auto span = [&](const std::string &text, const ImVec4 &color) {
      Flow(ImGui::CalcTextSize(text.c_str()).x, right, first);
      first = false;
      ImGui::TextColored(color, "%s", text.c_str());
    };
    span(std::string(view.live ? ICON_FA_CIRCLE : ICON_FA_PAUSE) + " " + view.state,
         view.live ? kLive : kAmber);
    span(view.showing, kMuted);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
      ImGui::SetTooltip(
          "Scanned %zu retained events.\nQuery worker: requested %llu, executed %llu, "
          "coalesced %llu, stale results discarded %llu. The queue holds at most one "
          "pending request.",
          query.scanned_count,
          static_cast<unsigned long long>(
              inspector_query_requests_.load(std::memory_order_relaxed)),
          static_cast<unsigned long long>(
              inspector_query_executions_.load(std::memory_order_relaxed)),
          static_cast<unsigned long long>(
              inspector_query_coalesced_.load(std::memory_order_relaxed)),
          static_cast<unsigned long long>(
              inspector_query_stale_.load(std::memory_order_relaxed)));
    }
    span(view.retained, kMuted);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
      ImGui::SetTooltip("Oldest retained #%s, newest #%s.",
                        cyxwiz::logs::FormatCount(query.store_stats.oldest_sequence).c_str(),
                        cyxwiz::logs::FormatCount(query.store_stats.newest_sequence).c_str());
    }
    span(view.evicted, kMuted);
    Tooltip("Oldest events removed because the in-memory log holds a fixed number of "
            "events. engine_log.txt keeps everything.");
    span(view.losses, view.has_losses ? kAmber : kMuted);
    Tooltip("Dropped: lost while the log was busy. Rejected: malformed events. "
            "Suppressed: repeats collapsed by rate limiting.");
    span(view.high_water, kMuted);
    if (!view.hidden.empty()) {
      span(view.hidden, kAmber);
      Flow(ui::ButtonWidth("Show retained", ui::ButtonSize::Small), right, false);
      if (ui::LinkButton("Show retained")) {
        inspector_after_sequence_ = 0;
        RequestInspectorQuery(true);
      }
    }
    if (!view.truncation.empty())
      span(view.truncation, kAmber);
    if (!inspector_hidden_columns_.empty() && !query.events.empty()) {
      span(inspector_hidden_columns_ + " hidden: empty in these rows", kFaint);
      Tooltip("A column shows again as soon as a shown row has a value for it.");
    }
  } else {
    ImGui::TextColored(kMuted, "Loading runtime events...");
  }
  if (!logs_notice_.empty()) {
    if (ImGui::GetTime() < logs_notice_until_) {
      Flow(ImGui::CalcTextSize(logs_notice_.c_str()).x, right, false);
      ImGui::TextColored(kAccent, "%s", logs_notice_.c_str());
    } else {
      logs_notice_.clear();
    }
  }
  ImGui::PopStyleVar();
  RenderRuntimeLogExportStatus();
  ImGui::Separator();
}

void Console::RenderInspectorTable(const cyxwiz::RuntimeLogInspectorResult *result) {
  const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
                                ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY |
                                ImGuiTableFlags_SizingFixedFit |
                                ImGuiTableFlags_NoSavedSettings;
  if (result->query.events.empty()) {
    ImGui::Spacing();
    const char *empty =
        result->query.scanned_count == 0 && inspector_after_sequence_ != 0
            ? "No new events since the view was cleared. Show retained brings back "
              "earlier events."
        : result->query.store_stats.size > 0 ? "No events match the current filters."
                                             : "No runtime events yet.";
    ImGui::TextColored(kMuted, "%s", empty);
    return;
  }
  const float char_w = ImGui::CalcTextSize("0").x;
  const auto &events = result->query.events;
  // Columns that say nothing in these rows are left out (TOFIX159 B3): a
  // column whose rows are all empty, or (Source) all the Engine's own logger.
  bool any_source = false, any_code = false, any_run = false, any_device = false;
  for (const auto &event : events) {
    any_source = any_source || (!event.source.empty() && event.source != "cyxwiz");
    any_code = any_code || !event.primary_error_code.empty();
    any_run = any_run || !event.run_id.empty();
    any_device = any_device || !cyxwiz::logs::DeviceLabel(event).empty();
  }
  std::string hidden;
  for (const auto &[shown, name] : {std::pair{any_source, "Source"}, std::pair{any_code, "Code"},
                                    std::pair{any_run, "Run"}, std::pair{any_device, "Device"}})
    if (!shown) hidden += (hidden.empty() ? "" : ", ") + std::string(name);
  inspector_hidden_columns_ = hidden;

  enum Column { kTime, kLevel, kCategory, kSource, kCode, kRun, kDevice, kMessage };
  std::vector<Column> columns = {kTime, kLevel, kCategory};
  if (any_source) columns.push_back(kSource);
  if (any_code) columns.push_back(kCode);
  if (any_run) columns.push_back(kRun);
  if (any_device) columns.push_back(kDevice);
  columns.push_back(kMessage);
  const auto column_width = [&](Column column) {
    switch (column) {
    case kTime: return ImGui::CalcTextSize("00:00:00.000").x + 6.0f;
    case kLevel: return char_w * 11.0f;
    case kCategory: return char_w * 10.0f;
    case kSource: return char_w * 10.0f;
    case kCode: return char_w * 11.0f;
    case kRun: return char_w * 17.0f;
    case kDevice: return char_w * 11.0f;
    default: return 0.0f;
    }
  };
  const ImGuiStyle &style = ImGui::GetStyle();
  float fixed = 0.0f;
  for (Column column : columns) fixed += column_width(column) + style.CellPadding.x * 2.0f + 1.0f;
  const float message_w =
      std::max(80.0f, ImGui::GetContentRegionAvail().x - fixed - style.ScrollbarSize - style.CellPadding.x * 2.0f);

  // One table ID per column set: ImGui keeps column widths by index, so when
  // Source / Run / Device appear (a training run starts) the old stretched
  // Message width would land on Source and push Message off the right edge.
  std::string table_id = "RuntimeLogTable";
  for (Column column : columns) table_id += static_cast<char>('a' + column);
  if (!ImGui::BeginTable(table_id.c_str(), static_cast<int>(columns.size()), flags))
    return;
  ImGui::TableSetupScrollFreeze(0, 1);
  for (Column column : columns) {
    static const char *kNames[] = {"Time", "Level", "Category", "Source", "Code", "Run", "Device", "Message"};
    if (column == kMessage)
      ImGui::TableSetupColumn(kNames[column], ImGuiTableColumnFlags_WidthStretch);
    else
      ImGui::TableSetupColumn(kNames[column], ImGuiTableColumnFlags_WidthFixed, column_width(column));
  }
  ImGui::TableHeadersRow();

  // A message wraps to at most two lines, so rows have their own heights; the
  // rows outside the view are one spacer row above and one below.
  const float line_h = ImGui::GetTextLineHeight();
  const float row_pad = style.CellPadding.y * 2.0f;
  std::vector<float> tops(events.size() + 1, 0.0f);
  for (size_t i = 0; i < events.size(); ++i) {
    auto &cached = inspector_row_lines_[events[i].sequence];
    if (cached.first != message_w) {
      const auto lines = WrapTwoLines(events[i].message, message_w);
      cached = {message_w, lines.second.empty() ? 1 : 2};
    }
    tops[i + 1] = tops[i] + line_h * static_cast<float>(cached.second) + row_pad;
  }
  if (inspector_row_lines_.size() > events.size() * 2 + 512) inspector_row_lines_.clear();

  const auto dash = [] { ImGui::TextColored(kFaint, "-"); };
  if (inspector_scroll_to_selected_ != 0) {
    for (size_t i = 0; i < events.size(); ++i)
      if (events[i].sequence == inspector_scroll_to_selected_)
        ImGui::SetScrollY(std::max(0.0f, tops[i] - ImGui::GetWindowHeight() * 0.4f));
    inspector_scroll_to_selected_ = 0;
  }
  const float view_top = ImGui::GetScrollY() - line_h * 2.0f;
  const float view_bottom = ImGui::GetScrollY() + ImGui::GetWindowHeight() + line_h * 2.0f;
  const size_t first = static_cast<size_t>(
      std::upper_bound(tops.begin(), tops.end(), std::max(0.0f, view_top)) - tops.begin()) - 1;
  size_t last = first;
  while (last < events.size() && tops[last] < view_bottom) ++last;
  if (first > 0)
    ImGui::TableNextRow(ImGuiTableRowFlags_None, tops[first]);

  for (size_t row = first; row < last; ++row) {
    const auto &event = events[row];
    const size_t level = cyxwiz::logs::LevelIndex(event.level);
    const bool problem = cyxwiz::logs::IsProblemLevel(event.level);
    const bool selected = inspector_selected_sequence_ == event.sequence;
    const float height = tops[row + 1] - tops[row];
    ImGui::TableNextRow(ImGuiTableRowFlags_None, height);
    ImGui::TableSetColumnIndex(0);
    ImGui::PushID(static_cast<int>(event.sequence));
    if (ImGui::Selectable("##row", selected,
                          ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowDoubleClick |
                              ImGuiSelectableFlags_AllowOverlap,
                          ImVec2(0.0f, height - row_pad))) {
      if (ImGui::IsMouseDoubleClicked(0)) {
        inspector_selected_sequence_ = event.sequence;
        ImGui::SetClipboardText(cyxwiz::logs::FormatRow(event).c_str());
        SetLogsNotice("Copied row #" + cyxwiz::logs::FormatCount(event.sequence));
      } else {
        inspector_selected_sequence_ = selected ? 0 : event.sequence;
        if (!selected)
          pending_scroll_selection_ = event.sequence;
      }
    }
    if (selected) {
      const ImVec2 min = ImGui::GetItemRectMin();
      const ImVec2 max = ImGui::GetItemRectMax();
      ImGui::GetWindowDrawList()->AddRectFilled(min, ImVec2(min.x + 2.0f, max.y),
                                                ImGui::GetColorU32(kPrompt));
    }
    if (ImGui::BeginPopupContextItem("RuntimeLogContext")) {
      inspector_selected_sequence_ = event.sequence;
      if (ImGui::MenuItem("Copy message")) {
        ImGui::SetClipboardText(event.message.c_str());
        SetLogsNotice("Copied message");
      }
      if (ImGui::MenuItem("Copy row")) {
        ImGui::SetClipboardText(cyxwiz::logs::FormatRow(event).c_str());
        SetLogsNotice("Copied row #" + cyxwiz::logs::FormatCount(event.sequence));
      }
      ImGui::EndPopup();
    }
    ImGui::SameLine(0.0f, 0.0f);
    const auto time = cyxwiz::logs::FormatLocalTime(event.timestamp_utc);
    ImGui::TextColored(selected ? kText : kMuted, "%s", time.c_str());
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
      ImGui::SetTooltip("%s UTC\n%s local",
                        cyxwiz::logs::FormatUtcTimestamp(event.timestamp_utc).c_str(),
                        cyxwiz::logs::FormatLocalTimestamp(event.timestamp_utc).c_str());
    }
    ImGui::PopID();

    for (size_t c = 1; c < columns.size(); ++c) {
      ImGui::TableSetColumnIndex(static_cast<int>(c));
      switch (columns[c]) {
      case kLevel:
        ImGui::TextColored(kLevelColors[level], "%s %s", kLevelIcons[level], cyxwiz::logs::kLevelLabels[level]);
        break;
      case kCategory:
        ImGui::TextUnformatted(event.category.c_str());
        break;
      case kSource:
        if (event.source.empty()) dash();
        else ImGui::TextUnformatted(event.source.c_str());
        break;
      case kCode:
        if (event.primary_error_code.empty()) dash();
        else ImGui::TextColored(problem ? kRed : kText, "%s", event.primary_error_code.c_str());
        break;
      case kRun:
        if (event.run_id.empty()) dash();
        else {
          ImGui::TextUnformatted(event.run_id.c_str());
          Tooltip(event.run_id.c_str());
        }
        break;
      case kDevice: {
        const auto device = cyxwiz::logs::DeviceLabel(event);
        if (device.empty()) dash();
        else {
          ImGui::TextUnformatted(device.c_str());
          if (!event.device_name.empty()) Tooltip(event.device_name.c_str());
        }
        break;
      }
      case kMessage: {
        const auto lines = WrapTwoLines(cyxwiz::gui::DrawableText(ImGui::GetFont(), event.message), message_w);
        const ImVec4 colour = problem || selected ? kBright : kText;
        ImGui::TextColored(colour, "%s", lines.first.c_str());
        if (!lines.second.empty())
          ImGui::TextColored(colour, "%s%s", lines.second.c_str(), lines.more ? " ..." : "");
        if (lines.more)
          Tooltip("The message goes on. Select the row to read all of it.");
        break;
      }
      default:
        break;
      }
    }
  }
  if (last < events.size())
    ImGui::TableNextRow(ImGuiTableRowFlags_None, tops[events.size()] - tops[last]);
  if (logs_auto_scroll_ && !inspector_paused_ &&
      inspector_last_rendered_high_water_ != result->query.high_water_sequence) {
    ImGui::SetScrollY(ImGui::GetScrollMaxY() + tops[events.size()]);
  }
  inspector_last_rendered_high_water_ = result->query.high_water_sequence;
  ImGui::EndTable();
}

void Console::RenderInspectorDetails(const cyxwiz::RuntimeLogInspectorResult *result) {
  const auto selected =
      std::find_if(result->query.events.begin(), result->query.events.end(),
                   [this](const auto &event) {
                     return event.sequence == inspector_selected_sequence_;
                   });
  if (selected == result->query.events.end())
    return;
  const auto &event = *selected;
  const size_t level = cyxwiz::logs::LevelIndex(event.level);

  ImGui::PushStyleColor(ImGuiCol_ChildBg, kPanelBg);
  ImGui::PushStyleColor(ImGuiCol_Border, kBorder);
  ImGui::BeginChild("RuntimeLogDetails", ImVec2(0, 0), ImGuiChildFlags_AlwaysUseWindowPadding);

  // Header: level, title and code on one line; the actions on the next, so
  // nothing is pushed out when the details sit in a narrow pane (TOFIX159 B3).
  ImGui::AlignTextToFramePadding();
  ImGui::TextColored(kLevelColors[level], "%s %s", kLevelIcons[level],
                     cyxwiz::logs::kLevelLabels[level]);
  ImGui::SameLine();
  const std::string title = cyxwiz::logs::DetailsTitle(event);
  const ImVec2 title_at = ImGui::GetCursorScreenPos();
  ImGui::PushClipRect(title_at,
                      ImVec2(title_at.x + ImGui::GetContentRegionAvail().x, title_at.y + 40.0f), true);
  ImGui::TextColored(kBright, "%s", title.c_str());
  if (!event.primary_error_code.empty()) {
    ImGui::SameLine();
    ImGui::TextColored(kRed, "%s", event.primary_error_code.c_str());
  }
  ImGui::PopClipRect();
  if (ui::SecondaryButton("Copy message")) {
    ImGui::SetClipboardText(event.message.c_str());
    SetLogsNotice("Copied message");
  }
  ImGui::SameLine();
  if (ui::SecondaryButton("Copy row")) {
    ImGui::SetClipboardText(cyxwiz::logs::FormatRow(event).c_str());
    SetLogsNotice("Copied row #" + cyxwiz::logs::FormatCount(event.sequence));
  }
  Tooltip("The row with every field, as one line.");
  ImGui::SameLine();
  if (ImGui::GetContentRegionAvail().x < ui::ButtonWidth("Copy as JSON", ui::ButtonSize::Small))
    ImGui::NewLine();
  if (ui::SecondaryButton("Copy as JSON")) {
    cyxwiz::RuntimeLogRedactionOptions keep_all;
    keep_all.secrets = keep_all.paths = keep_all.dataset_names = keep_all.query_text = keep_all.python_output = false;
    ImGui::SetClipboardText(cyxwiz::RuntimeLogExportService::EventToJson(event, keep_all).dump(2).c_str());
    SetLogsNotice("Copied row #" + cyxwiz::logs::FormatCount(event.sequence) + " as JSON");
  }
  Tooltip("The event with every field, as JSON.");
  ImGui::SameLine();
  const float close_w = ImGui::CalcTextSize("Close").x + 4.0f;
  if (ImGui::GetContentRegionAvail().x > close_w)
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - close_w);
  if (ui::LinkButton("Close"))
    inspector_selected_sequence_ = 0;
  Tooltip("Close details (Esc).");

  // Full message: wrapped, selectable. A JSON ending (a generation sample) is
  // shown formatted under it unless Show raw is on (TOFIX159 B3).
  const auto json = cyxwiz::commands::FindJsonPayload(event.message);
  const float field_w = ImGui::GetContentRegionAvail().x;
  const std::string &shown_message = json && !inspector_raw_json_ ? json->before : event.message;
  std::string wrapped = WrapForWidth(cyxwiz::gui::DrawableText(ImGui::GetFont(), shown_message), field_w - 20.0f);
  const int lines = 1 + static_cast<int>(std::count(wrapped.begin(), wrapped.end(), '\n'));
  const float msg_h = std::min(8, std::max(1, lines)) * ImGui::GetTextLineHeight() +
                      ImGui::GetStyle().FramePadding.y * 2.0f + 2.0f;
  ImGui::PushStyleColor(ImGuiCol_FrameBg, kInputBg);
  ImGui::InputTextMultiline("##RuntimeLogMessage", wrapped.data(), wrapped.size() + 1,
                            ImVec2(-1.0f, msg_h), ImGuiInputTextFlags_ReadOnly);
  ImGui::PopStyleColor();
  Tooltip("Select text to copy part of the message, or use Copy message.");
  if (json) {
    ImGui::TextColored(kMuted, "JSON \xC2\xB7 %zu %s", json->entries, json->entries == 1 ? "entry" : "entries");
    ImGui::SameLine();
    ImGui::Checkbox("Show raw", &inspector_raw_json_);
    if (!inspector_raw_json_) {
      std::string pretty = json->pretty;
      const int json_lines = 1 + static_cast<int>(std::count(pretty.begin(), pretty.end(), '\n'));
      ImGui::PushStyleColor(ImGuiCol_FrameBg, kInputBg);
      ImGui::InputTextMultiline("##RuntimeLogJson", pretty.data(), pretty.size() + 1,
                                ImVec2(-1.0f, std::min(14, std::max(3, json_lines)) * ImGui::GetTextLineHeight() +
                                                  ImGui::GetStyle().FramePadding.y * 2.0f + 2.0f),
                                ImGuiInputTextFlags_ReadOnly);
      ImGui::PopStyleColor();
    }
  }

  // Fields, two pairs per line; click a value to copy it.
  const auto fields = cyxwiz::logs::DetailFields(event);
  if (ImGui::BeginTable("RuntimeLogFields", 4, ImGuiTableFlags_SizingStretchProp)) {
    const float key_w = ImGui::CalcTextSize("Issue codes__").x;
    ImGui::TableSetupColumn("k1", ImGuiTableColumnFlags_WidthFixed, key_w);
    ImGui::TableSetupColumn("v1", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("k2", ImGuiTableColumnFlags_WidthFixed, key_w);
    ImGui::TableSetupColumn("v2", ImGuiTableColumnFlags_WidthStretch);
    for (size_t i = 0; i < fields.size(); ++i) {
      if (i % 2 == 0)
        ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::TextColored(kMuted, "%s", fields[i].first.c_str());
      ImGui::TableNextColumn();
      ImGui::PushID(static_cast<int>(i));
      ImGui::PushTextWrapPos(0.0f);
      ImGui::TextUnformatted(fields[i].second.c_str());
      ImGui::PopTextWrapPos();
      if (ImGui::IsItemHovered()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        ImGui::SetTooltip("Click to copy");
      }
      if (ImGui::IsItemClicked()) {
        ImGui::SetClipboardText(fields[i].second.c_str());
        SetLogsNotice("Copied " + fields[i].first);
      }
      ImGui::PopID();
    }
    ImGui::EndTable();
  }
  ImGui::EndChild();
  ImGui::PopStyleColor(2);
}

void Console::CopyFilteredRuntimeLogs() {
  const auto result = SnapshotInspectorResult();
  if (!result || result->query.events.empty()) {
    SetLogsNotice("Nothing to copy");
    return;
  }
  std::string output;
  for (const auto &event : result->query.events) {
    output += cyxwiz::logs::FormatRow(event);
    output += '\n';
  }
  ImGui::SetClipboardText(output.c_str());
  SetLogsNotice("Copied " + cyxwiz::logs::FormatCount(result->query.events.size()) +
                " rows");
}

void Console::CopySelectedRuntimeLog() {
  const auto result = SnapshotInspectorResult();
  if (!result)
    return;
  const auto selected =
      std::find_if(result->query.events.begin(), result->query.events.end(),
                   [this](const auto &event) {
                     return event.sequence == inspector_selected_sequence_;
                   });
  if (selected == result->query.events.end()) {
    inspector_selected_sequence_ = 0;
    SetLogsNotice("Select a row first");
    return;
  }
  ImGui::SetClipboardText(cyxwiz::logs::FormatRow(*selected).c_str());
  SetLogsNotice("Copied row #" + cyxwiz::logs::FormatCount(selected->sequence));
}

} // namespace gui
