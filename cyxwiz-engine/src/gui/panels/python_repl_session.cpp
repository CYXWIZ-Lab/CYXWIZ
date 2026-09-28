#include "../console_palette.h"
#include "python_repl_session.h"
#include "../../core/engine_config.h"
#include "../../core/project_manager.h"
#include "../../scripting/scripting_engine.h"
#include "../editor_fonts.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <imgui.h>
#include <imgui_internal.h>
#include <iterator>
#include <sstream>

namespace cyxwiz {

namespace {

constexpr size_t kMaxEntries = 500;
constexpr size_t kMaxFieldBytes = 256 * 1024;
constexpr size_t kMaxHistory = 100;
constexpr const char *kInputLabel = "##repl_input";

// Palette (tofix121 mockup).
ImVec4 kText;
ImVec4 kMuted;
ImVec4 kFaint;
ImVec4 kSeparator;
ImVec4 kPrompt;
ImVec4 kContinuation;
ImVec4 kAccent;
ImVec4 kKeyword;
ImVec4 kBuiltin;
ImVec4 kString;
ImVec4 kNumber;
ImVec4 kDecorator;
ImVec4 kError;
ImVec4 kWarning;
ImVec4 kReady;
ImVec4 kInputBg;
ImVec4 kPanelBg;
ImVec4 kBarBg;
ImVec4 kBorder;
ImVec4 kInnerBorder;

// Colours follow the active theme (gui/console_palette), refreshed
// at the start of each render.
void RefreshPalette() {
  const auto &p = ::gui::CurrentConsolePalette();
  kText = p.text;
  kMuted = p.muted;
  kFaint = p.faint;
  kSeparator = p.border;
  kPrompt = p.accent;
  kContinuation = p.continuation;
  kAccent = p.accent_text;
  kKeyword = p.keyword;
  kBuiltin = p.builtin;
  kString = p.string;
  kNumber = p.number;
  kDecorator = p.decorator;
  kError = (p.light ? ImVec4(0.72f, 0.36f, 0.05f, 1.0f) : ImVec4(0.96f, 0.63f, 0.29f, 1.0f));
  kWarning = p.warning;
  kReady = p.success;
  kInputBg = p.input;
  kPanelBg = p.panel;
  kBarBg = p.bar;
  kBorder = p.border;
  kInnerBorder = p.inner_border;
}


ImVec4 TokenColor(repl::TokenKind kind) {
  switch (kind) {
  case repl::TokenKind::Keyword:
    return kKeyword;
  case repl::TokenKind::Builtin:
    return kBuiltin;
  case repl::TokenKind::String:
    return kString;
  case repl::TokenKind::Number:
    return kNumber;
  case repl::TokenKind::Comment:
    return kFaint;
  case repl::TokenKind::Decorator:
    return kDecorator;
  case repl::TokenKind::Plain:
    break;
  }
  return kText;
}

ImVec4 StateColor(repl::StatusState state) {
  switch (state) {
  case repl::StatusState::Ready:
  case repl::StatusState::NotStarted:
    return kReady;
  case repl::StatusState::Running:
  case repl::StatusState::SettingUp:
    return kAccent;
  case repl::StatusState::Unavailable:
    break;
  }
  return kError;
}

std::string TrimRight(std::string value) {
  while (!value.empty() &&
         (value.back() == ' ' || value.back() == '\t' || value.back() == '\n' ||
          value.back() == '\r')) {
    value.pop_back();
  }
  return value;
}

std::string Trim(std::string value) {
  value = TrimRight(std::move(value));
  const auto first = value.find_first_not_of(" \t\r\n");
  return first == std::string::npos ? std::string() : value.substr(first);
}

// Leading blank lines dropped, trailing whitespace of every line kept apart
// from the last: the command as the user meant it.
std::string NormalizeSubmission(const std::string &text) {
  std::string value = TrimRight(text);
  size_t start = 0;
  while (start < value.size()) {
    const size_t eol = value.find('\n', start);
    if (eol == std::string::npos)
      break;
    if (value.find_first_not_of(" \t\r", start) < eol)
      break;
    start = eol + 1;
  }
  return value.substr(start);
}

void AppendCapped(std::string &field, const std::string &text) {
  if (field.size() >= kMaxFieldBytes)
    return;
  field.append(text, 0, kMaxFieldBytes - field.size());
  if (field.size() >= kMaxFieldBytes)
    field += "\n... (output truncated; showing the first 256 KB)\n";
}

std::vector<std::string> SplitLines(const std::string &text) {
  std::vector<std::string> lines;
  std::string line;
  std::istringstream stream(text);
  while (std::getline(stream, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    lines.push_back(line);
  }
  return lines;
}

std::string CommonPrefix(const std::vector<std::string> &items) {
  if (items.empty())
    return {};
  std::string prefix = items.front();
  for (const auto &item : items) {
    size_t n = 0;
    while (n < prefix.size() && n < item.size() && prefix[n] == item[n])
      ++n;
    prefix.resize(n);
  }
  return prefix;
}

bool StartsWith(const std::string &value, const std::string &prefix) {
  return value.compare(0, prefix.size(), prefix) == 0;
}

// Interpreter is the bundled one, or a venv whose base (pyvenv.cfg "home")
// is the bundled runtime's folder.
bool UsesBundledRuntime(const std::string &interpreter,
                        const std::string &bundled) {
  namespace fs = std::filesystem;
  if (interpreter.empty() || bundled.empty())
    return false;
  std::error_code ec;
  if (fs::equivalent(interpreter, bundled, ec))
    return true;
  const fs::path cfg =
      fs::path(interpreter).parent_path().parent_path() / "pyvenv.cfg";
  std::ifstream in(cfg);
  std::string line;
  while (std::getline(in, line)) {
    const auto eq = line.find('=');
    if (eq == std::string::npos || Trim(line.substr(0, eq)) != "home")
      continue;
    const std::string home = Trim(line.substr(eq + 1));
    return fs::equivalent(home, fs::path(bundled).parent_path(), ec);
  }
  return false;
}

// Enter runs the input when it is complete. Trailing spaces of an indented
// empty last line count as an empty line (the block is closed).
bool ReadyToRun(const std::string &text) {
  std::string candidate = text;
  while (!candidate.empty() && (candidate.back() == ' ' || candidate.back() == '\t'))
    candidate.pop_back();
  return !Trim(candidate).empty() && repl::IsInputComplete(candidate);
}

// Code views use the Engine-wide code text size (Preferences > Appearance).
ImFont *ReplMonoFont() { return gui::GetCodeFont(); }

// Small key cap used in the hints row.
void KeyCap(const char *label) {
  ImDrawList *dl = ImGui::GetWindowDrawList();
  const ImVec2 size = ImGui::CalcTextSize(label);
  const ImVec2 pad(5.0f, 1.0f);
  const ImVec2 pos = ImGui::GetCursorScreenPos();
  const ImVec2 max(pos.x + size.x + pad.x * 2, pos.y + size.y + pad.y * 2);
  dl->AddRectFilled(pos, max, ImGui::GetColorU32(kInputBg),
                    4.0f);
  dl->AddRect(pos, max, ImGui::GetColorU32(kBorder),
              4.0f);
  dl->AddText(ImVec2(pos.x + pad.x, pos.y + pad.y),
              ImGui::GetColorU32(kMuted), label);
  ImGui::Dummy(ImVec2(max.x - pos.x, max.y - pos.y));
}

void Hint(const char *key, const char *text, const char *key2 = nullptr) {
  KeyCap(key);
  if (key2) {
    ImGui::SameLine(0.0f, 2.0f);
    KeyCap(key2);
  }
  ImGui::SameLine(0.0f, 4.0f);
  ImGui::TextColored(kFaint, "%s", text);
}

void DrawTokens(ImDrawList *dl, ImVec2 pos, const std::vector<repl::Token> &line) {
  ImFont *font = ImGui::GetFont();
  const float size = ImGui::GetFontSize();
  for (const auto &token : line) {
    const char *begin = token.text.c_str();
    const char *end = begin + token.text.size();
    dl->AddText(font, size, pos, ImGui::GetColorU32(TokenColor(token.kind)), begin,
                end);
    pos.x += font->CalcTextSizeA(size, FLT_MAX, 0.0f, begin, end).x;
  }
}

const char *kHelpText = R"(CyxWiz Python REPL Help
=======================

COMMANDS:
  clear       - Clear output window
  help()      - Show this help message

DUCKDB (SQL Analytics):
  sql(query)       - Run SQL query on in-memory database
  read_csv(path)   - Load CSV file
  read_parquet(p)  - Load Parquet file
  read_json(path)  - Load JSON file
  db               - DuckDB connection object

  Examples:
    sql("SELECT 1 + 1 AS result")
    sql("SELECT * FROM 'data.csv' LIMIT 10")
    read_csv('data.csv').filter('age > 30')

POLARS (Fast DataFrames):
  pl               - Polars module
  df(data)         - Create DataFrame
  col('name')      - Column expression
  pl_csv(path)     - Read CSV file
  pl_parquet(p)    - Read Parquet file
  scan_csv(path)   - Lazy CSV reader
  scan_parquet(p)  - Lazy Parquet reader

  Examples:
    data = df({'a': [1, 2, 3], 'b': [4, 5, 6]})
    data.filter(col('a') > 1)
    pl_csv('data.csv').head(10)

MATLAB-STYLE FUNCTIONS:
  Linear Algebra:  eye, zeros, ones, svd, eig, qr, chol, lu, det,
                   rank, trace, norm, cond, inv, transpose, solve
  Signal:          fft, ifft, conv, spectrogram, lowpass, highpass
  Statistics:      kmeans, dbscan, gmm, pca, tsne
  Time Series:     acf, pacf, decompose, stationarity, arima

  Examples:
    I = eye(3)              # 3x3 identity matrix
    pm(I)                   # Print matrix nicely
    U, S, V = svd([[1,2],[3,4]])

Type any Python code to execute.
)";

} // namespace

PythonReplSession::PythonReplSession() : input_buffer_(1024, '\0') {}

PythonReplSession::~PythonReplSession() {
  if (command_executing_) {
    StopAsyncCommand();
  }
}

void PythonReplSession::SetScriptingEngine(
    std::shared_ptr<scripting::ScriptingEngine> engine) {
  scripting_engine_ = std::move(engine);
  RefreshInterpreterInfo(true);
}

void PythonReplSession::ResetProjectState() {
  const bool had_entries = !entries_.empty();
  if (command_executing_)
    StopAsyncCommand();
  command_executing_ = false;
  running_entry_ = -1;
  entries_.clear();
  command_history_.clear();
  history_position_ = -1;
  history_draft_.clear();
  completion_open_ = false;
  completion_items_.clear();
  completion_all_.clear();
  selected_entry_ = -1;
  SetInputText("");
  scroll_to_bottom_ = true;
  focus_input_ = true;
  RefreshInterpreterInfo(true);
  if (had_entries)
    AddSystem("Python REPL project state reset.");
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

void PythonReplSession::RefreshInterpreterInfo(bool force) {
  const auto now = std::chrono::steady_clock::now();
  if (!force && now - interpreter_refreshed_ < std::chrono::seconds(2))
    return;
  interpreter_refreshed_ = now;
  if (!scripting_engine_) {
    interpreter_ = {};
    bundled_runtime_ = false;
    last_init_error_.clear();
    return;
  }
  interpreter_ = scripting_engine_->GetInterpreterInfo();
  last_init_error_ = scripting_engine_->GetLastInitError();
  bundled_runtime_ = UsesBundledRuntime(
      interpreter_.interpreter_path,
      core::EngineConfig::Instance().GetBundledPythonPath());
}

repl::StatusView PythonReplSession::BuildStatus() const {
  repl::StatusInput in;
#ifdef CYXWIZ_HAS_PYTHON
  in.engine_available = scripting_engine_ != nullptr;
#else
  in.engine_available = false;
#endif
#ifdef CYXWIZ_PYTHON_LINKED_MINOR
  in.linked_version = "3." + std::to_string(CYXWIZ_PYTHON_LINKED_MINOR);
#endif
  const auto &pm = ProjectManager::Instance();
  in.has_project = pm.HasActiveProject();
  in.project_root = pm.GetProjectRoot();
  if (in.has_project) {
    in.env_setup_pending = pm.IsPythonEnvSetupPending(in.project_root);
    const auto env = pm.GetPythonEnvSetupStatus();
    if (env.state == ProjectManager::PythonEnvSetupState::Failed &&
        (env.project_root.empty() || env.project_root == in.project_root)) {
      in.env_setup_failed = true;
      in.env_setup_message = env.message;
    }
  }
  in.initialized = interpreter_.initialized;
  in.version = interpreter_.version;
  in.interpreter_path = interpreter_.interpreter_path;
  in.interpreter_mismatch = interpreter_.mismatch;
  in.last_init_error = last_init_error_;
  in.bundled_runtime = bundled_runtime_;
  in.running = command_executing_;
  if (in.running) {
    in.running_seconds = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - command_started_)
                             .count();
  }
  return repl::BuildStatusView(in);
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

void PythonReplSession::RenderContent() {
  RefreshPalette();
  CheckAsyncCompletion();
  RefreshInterpreterInfo(false);
  status_ = BuildStatus();

  if (notice_timer_ > 0.0f) {
    notice_timer_ -= ImGui::GetIO().DeltaTime;
    if (notice_timer_ <= 0.0f)
      notice_.clear();
  }

  HandleShortcuts();
  RenderHeader();

  // Footer (input box + hints) height, so the transcript takes the rest.
  ImFont *mono = ReplMonoFont();
  const float mono_line =
      mono ? mono->FontSize : ImGui::GetTextLineHeight();
  const ImGuiStyle &style = ImGui::GetStyle();
  const std::string current(input_buffer_.data());
  const int lines = std::clamp(
      1 + static_cast<int>(std::count(current.begin(), current.end(), '\n')), 3, 8);
  const float pad_y = 10.0f;
  const float avail = ImGui::GetContentRegionAvail().y;
  const float min_input = mono_line * 2.0f + pad_y * 2.0f;
  const float max_input = std::max(min_input, avail * 0.6f);
  input_height_ = std::clamp(lines * mono_line + pad_y * 2.0f + input_extra_height_,
                             min_input, max_input);
  const float bar_h = ImGui::GetFrameHeight() + 8.0f;
  const float handle_h = 12.0f;
  const float hints_h =
      (ImGui::GetTextLineHeight() + 6.0f) * static_cast<float>(hint_rows_);
  const float footer_h = handle_h + input_height_ + bar_h + hints_h +
                         style.ItemSpacing.y * 3.0f + 10.0f;
  const float transcript_h = std::max(60.0f, avail - footer_h);

  ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0, 0, 0, 0));
  ImGui::BeginChild("##repl_transcript", ImVec2(0.0f, transcript_h),
                    ImGuiChildFlags_None, ImGuiWindowFlags_HorizontalScrollbar);
  RenderTranscript();
  ImGui::EndChild();
  ImGui::PopStyleColor();

  RenderInput();
}

void PythonReplSession::HandleShortcuts() {
  if (!ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows))
    return;
  const ImGuiIO &io = ImGui::GetIO();
  if (!io.KeyCtrl)
    return;
  if (ImGui::IsKeyPressed(ImGuiKey_L, false)) {
    ClearTranscript();
    return;
  }
  if (ImGui::IsKeyPressed(ImGuiKey_C, false)) {
    ImGuiInputTextState *state = ImGui::GetInputTextState(input_id_);
    const bool input_selection = state && state->HasSelection();
    if (command_executing_ && !input_selection) {
      StopAsyncCommand();
    } else if (!input_active_ && selected_entry_ >= 0 &&
               selected_entry_ < static_cast<int>(entries_.size())) {
      ImGui::SetClipboardText(EntryText(entries_[selected_entry_]).c_str());
      notice_ = "Copied entry";
      notice_timer_ = 2.0f;
    }
  }
}

void PythonReplSession::RenderHeader() {
  const repl::StatusView &view = status_;
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8.0f, 4.0f));

  // Left: state | version, environment | runtime
  const ImVec4 state_color = StateColor(view.state);
  const char *state_icon =
      view.state == repl::StatusState::Running ||
              view.state == repl::StatusState::SettingUp
          ? ICON_FA_SPINNER
          : ICON_FA_CIRCLE;
  ImGui::AlignTextToFramePadding();
  ImGui::TextColored(state_color, "%s %s", state_icon, view.state_label.c_str());
  if (!view.message.empty() && ImGui::IsItemHovered())
    ImGui::SetTooltip("%s", view.message.c_str());

  if (view.version_label != "Python ") {
    ImGui::SameLine();
    ImGui::TextColored(kSeparator, "|");
    ImGui::SameLine();
    ImGui::TextColored(::gui::CurrentConsolePalette().bright, "%s",
                       view.version_label.c_str());
  }
  if (!view.environment_label.empty()) {
    ImGui::SameLine();
    ImGui::TextColored(kMuted, "%s", view.environment_kind.c_str());
    ImGui::SameLine();
    ImGui::TextColored(kAccent, "%s", view.environment_label.c_str());
    if (ImGui::IsItemHovered()) {
      ImGui::BeginTooltip();
      ImGui::Text("Interpreter: %s", interpreter_.interpreter_path.c_str());
      const std::string &root = ProjectManager::Instance().GetProjectRoot();
      if (!root.empty())
        ImGui::Text("Project: %s", root.c_str());
      if (!interpreter_.source.empty())
        ImGui::Text("Selected from: %s settings", interpreter_.source.c_str());
      ImGui::EndTooltip();
    }
    ImGui::SameLine();
    ImGui::TextColored(kSeparator, "|");
    ImGui::SameLine();
    ImGui::TextColored(kMuted, "%s", view.runtime_label.c_str());
  }

  // Right: Interrupt (while running), Restart, Copy all, Clear.
  const std::string interrupt_label = std::string(ICON_FA_STOP) + " Interrupt";
  const std::string restart_label = std::string(ICON_FA_ROTATE_RIGHT) + " Restart";
  const std::string copy_label = std::string(ICON_FA_COPY) + " Copy all";
  const std::string clear_label = std::string(ICON_FA_TRASH_CAN) + " Clear";
  const float spacing = ImGui::GetStyle().ItemSpacing.x;
  float right_w = ui::ButtonWidth(restart_label.c_str(), ui::ButtonSize::Small) +
                  ui::ButtonWidth(copy_label.c_str(), ui::ButtonSize::Small) +
                  ui::ButtonWidth(clear_label.c_str(), ui::ButtonSize::Small) +
                  spacing * 2.0f;
  if (view.can_interrupt)
    right_w += ui::ButtonWidth(interrupt_label.c_str(), ui::ButtonSize::Small) + spacing;
  ImGui::SameLine();
  const float x = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - right_w;
  if (x > ImGui::GetCursorPosX())
    ImGui::SetCursorPosX(x);

  if (view.can_interrupt) {
    if (ui::DangerButton(interrupt_label.c_str()))
      StopAsyncCommand();
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("Stop the running command (Ctrl+C)");
    ImGui::SameLine();
  }
  if (ui::SecondaryButton(restart_label.c_str(), view.can_reset && !command_executing_,
                          command_executing_ ? "Stop the running command first."
                                             : "Python has not started yet.")) {
    ResetSession();
  }
  if (ImGui::IsItemHovered() && view.can_reset && !command_executing_)
    ImGui::SetTooltip("Clear all variables and imports. The interpreter keeps "
                      "running inside the Engine.");
  ImGui::SameLine();
  if (ui::SecondaryButton(copy_label.c_str()))
    CopyAll();
  ImGui::SameLine();
  if (ui::SecondaryButton(clear_label.c_str()))
    ClearTranscript();
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Clear the output (Ctrl+L). Variables are kept.");

  // Second line only when something needs attention.
  if (!view.message.empty() && view.state != repl::StatusState::Ready &&
      view.state != repl::StatusState::NotStarted) {
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(view.state == repl::StatusState::SettingUp ? kAccent : kError,
                       "%s %s",
                       view.state == repl::StatusState::SettingUp
                           ? ICON_FA_SPINNER
                           : ICON_FA_TRIANGLE_EXCLAMATION,
                       view.message.c_str());
    ImGui::PopTextWrapPos();
  }
  ImGui::PopStyleVar();
  ImGui::Separator();
}

void PythonReplSession::RenderTranscript() {
  ImFont *mono = ReplMonoFont();

  ImGui::Dummy(ImVec2(0.0f, 4.0f));
  ImGui::Indent(6.0f);
  {
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(kFaint, "%s", repl::WelcomeText(status_).c_str());
    ImGui::PopTextWrapPos();
  }
  ImGui::Dummy(ImVec2(0.0f, 4.0f));

  if (mono)
    ImGui::PushFont(mono);
  for (int i = 0; i < static_cast<int>(entries_.size()); ++i) {
    ImGui::PushID(i);
    RenderEntry(i, entries_[i]);
    ImGui::PopID();
  }
  if (mono)
    ImGui::PopFont();
  ImGui::Unindent(6.0f);
  ImGui::Dummy(ImVec2(0.0f, 6.0f));

  // Follow new output when auto-scroll is on, or when the user was already at
  // the bottom.
  const bool at_bottom = ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 4.0f;
  if (scroll_to_bottom_ && (auto_scroll_ || at_bottom))
    ImGui::SetScrollHereY(1.0f);
  scroll_to_bottom_ = false;
}

void PythonReplSession::RenderEntry(int index, const Entry &entry) {
  ImDrawList *dl = ImGui::GetWindowDrawList();
  const float line_h = ImGui::GetTextLineHeight();
  const float prompt_w = ImGui::CalcTextSize(">>> ").x;
  const float output_indent = prompt_w;

  ImDrawListSplitter splitter;
  splitter.Split(dl, 2);
  splitter.SetCurrentChannel(dl, 1);

  const ImVec2 start = ImGui::GetCursorScreenPos();
  ImGui::BeginGroup();

  auto text_block = [&](const std::string &text, const ImVec4 &color,
                        float indent) {
    if (text.empty())
      return;
    if (indent > 0.0f)
      ImGui::Indent(indent);
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    std::string body = text;
    while (!body.empty() && (body.back() == '\n' || body.back() == '\r'))
      body.pop_back();
    ImGui::TextUnformatted(body.c_str(), body.c_str() + body.size());
    ImGui::PopStyleColor();
    if (indent > 0.0f)
      ImGui::Unindent(indent);
  };

  const bool is_error = entry.failed && !entry.cancelled;
  switch (entry.kind) {
  case Entry::Kind::System:
    text_block(entry.output, kMuted, 0.0f);
    break;

  case Entry::Kind::Command: {
    for (size_t l = 0; l < entry.code_lines.size(); ++l) {
      const ImVec2 pos = ImGui::GetCursorScreenPos();
      if (ImGui::IsRectVisible(pos, ImVec2(pos.x + 10.0f, pos.y + line_h))) {
        dl->AddText(pos, ImGui::GetColorU32(l == 0 ? kPrompt : kContinuation),
                    l == 0 ? ">>>" : "...");
        DrawTokens(dl, ImVec2(pos.x + prompt_w, pos.y), entry.code_lines[l]);
      }
      float width = prompt_w;
      for (const auto &token : entry.code_lines[l])
        width += ImGui::CalcTextSize(token.text.c_str()).x;
      ImGui::Dummy(ImVec2(width, line_h));
    }
    text_block(entry.output, kText, output_indent);
    text_block(entry.warnings, kWarning, output_indent);
    if (entry.running) {
      ImGui::Indent(output_indent);
      const double secs = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - command_started_)
                              .count();
      ImGui::TextColored(kAccent, "%s Running \xC2\xB7 %s", ICON_FA_SPINNER,
                         repl::FormatElapsed(secs).c_str());
      ImGui::Unindent(output_indent);
    } else if (entry.cancelled) {
      text_block(entry.error.empty() ? "Interrupted" : entry.error, kWarning,
                 output_indent);
    } else if (is_error) {
      text_block(entry.traceback.empty() ? entry.error : entry.traceback, kError,
                 output_indent);
    }
    break;
  }

  case Entry::Kind::Script: {
    ImGui::PushFont(nullptr);
    std::string title;
    if (entry.running)
      title = " running...";
    else if (entry.cancelled)
      title = " cancelled after " + repl::FormatElapsed(entry.seconds);
    else if (entry.failed)
      title = " failed after " + repl::FormatElapsed(entry.seconds);
    else
      title = " finished in " + repl::FormatElapsed(entry.seconds);
    ImGui::TextColored(kMuted, "%s Script", ICON_FA_FILE_CODE);
    ImGui::SameLine(0.0f, 6.0f);
    ImGui::TextColored(kText, "%s", entry.source.c_str());
    ImGui::SameLine(0.0f, 0.0f);
    ImGui::TextColored(entry.failed ? kError : kMuted, "%s", title.c_str());
    ImGui::PopFont();
    text_block(entry.output, kText, 0.0f);
    text_block(entry.warnings, kWarning, 0.0f);
    text_block(entry.error, entry.cancelled ? kWarning : kError, 0.0f);
    break;
  }
  }

  if (!entry.hint.empty()) {
    ImGui::Indent(entry.kind == Entry::Kind::Command ? output_indent : 0.0f);
    ImGui::PushFont(nullptr);
    ImGui::TextColored(kMuted, "Install packages into the project environment:");
    ImGui::PopFont();
    ImGui::SameLine();
    ImGui::TextColored(kAccent, "%s", entry.hint.c_str());
    if (ImGui::IsItemHovered()) {
      ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
      ImGui::SetTooltip("Click to copy");
    }
    if (ImGui::IsItemClicked()) {
      ImGui::SetClipboardText(entry.hint.c_str());
      notice_ = "Copied: " + entry.hint;
      notice_timer_ = 2.5f;
    }
    ImGui::Unindent(entry.kind == Entry::Kind::Command ? output_indent : 0.0f);
  }
  ImGui::EndGroup();

  // Background, selection and hover for the whole entry.
  const float full_w = ImGui::GetContentRegionAvail().x;
  ImVec2 min(start.x - 8.0f, start.y - 3.0f);
  ImVec2 max(start.x + full_w - 2.0f, ImGui::GetItemRectMax().y + 3.0f);
  const bool hovered = ImGui::IsMouseHoveringRect(min, max) &&
                       ImGui::IsWindowHovered(ImGuiHoveredFlags_None);
  splitter.SetCurrentChannel(dl, 0);
  if (is_error && entry.kind == Entry::Kind::Command)
    dl->AddRectFilled(min, max, ImGui::GetColorU32(ImVec4(kError.x, kError.y, kError.z, 0.07f)),
                      6.0f);
  if (entry.kind == Entry::Kind::Script)
    dl->AddRectFilled(ImVec2(start.x - 8.0f, min.y), ImVec2(start.x - 6.0f, max.y),
                      ImGui::GetColorU32(kBorder));
  if (selected_entry_ == index)
    dl->AddRectFilled(min, max, ImGui::GetColorU32(::gui::CurrentConsolePalette().selection),
                      6.0f);
  else if (hovered)
    dl->AddRectFilled(min, max, ImGui::GetColorU32(::gui::CurrentConsolePalette().hover),
                      6.0f);
  splitter.Merge(dl);

  if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
    selected_entry_ = index;
  if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
    selected_entry_ = index;
    ImGui::OpenPopup("##entry_menu");
  }
  if (ImGui::BeginPopup("##entry_menu")) {
    ImGui::PushFont(nullptr);
    if (ImGui::MenuItem("Copy entry", "Ctrl+C"))
      ImGui::SetClipboardText(EntryText(entry).c_str());
    if (!entry.code.empty() && ImGui::MenuItem("Copy code"))
      ImGui::SetClipboardText(entry.code.c_str());
    if (!entry.output.empty() && ImGui::MenuItem("Copy output"))
      ImGui::SetClipboardText(entry.output.c_str());
    if (!entry.traceback.empty() && ImGui::MenuItem("Copy traceback"))
      ImGui::SetClipboardText(entry.traceback.c_str());
    if (!entry.code.empty()) {
      ImGui::Separator();
      if (ImGui::MenuItem("Edit in input")) {
        SetInputText(entry.code);
        focus_input_ = true;
      }
    }
    ImGui::PopFont();
    ImGui::EndPopup();
  }
  ImGui::Dummy(ImVec2(0.0f, 6.0f));
}

void PythonReplSession::RenderInput() {
  ImGuiIO &io = ImGui::GetIO();
  ImGuiStyle &style = ImGui::GetStyle();
  ImFont *mono = ReplMonoFont();
  const float pad_y = 10.0f;
  const float avail_w = ImGui::GetContentRegionAvail().x;

  // Drag handle: resize the input.
  {
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##repl_resize", ImVec2(avail_w, 12.0f));
    const bool active = ImGui::IsItemActive();
    if (ImGui::IsItemHovered() || active)
      ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
    if (ImGui::IsItemHovered() && !active)
      ImGui::SetTooltip("Drag to resize the input");
    if (active)
      input_extra_height_ -= io.MouseDelta.y;
    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
      input_extra_height_ = 0.0f;
    input_extra_height_ = std::clamp(input_extra_height_, -400.0f, 800.0f);
    const float cx = pos.x + avail_w * 0.5f;
    ImGui::GetWindowDrawList()->AddRectFilled(
        ImVec2(cx - 22.0f, pos.y + 4.0f), ImVec2(cx + 22.0f, pos.y + 8.0f),
        ImGui::GetColorU32(active ? kAccent : kBorder),
        2.0f);
  }

  if (mono)
    ImGui::PushFont(mono);
  const float line_h = ImGui::GetTextLineHeight();
  const float bar_h = ImGui::GetFrameHeight() + 8.0f;
  const float input_h = input_height_;
  const float gutter_w = ImGui::CalcTextSize(">>>").x + 22.0f;

  ImDrawList *dl = ImGui::GetWindowDrawList();
  const ImVec2 box_min = ImGui::GetCursorScreenPos();
  const ImVec2 box_max(box_min.x + avail_w, box_min.y + input_h + bar_h);
  dl->AddRectFilled(box_min, box_max, ImGui::GetColorU32(kInputBg), 8.0f);

  // Key handling before the widget. The widget owns Enter and the arrows
  // while active (it re-claims them every frame), so their side effects are
  // undone in the callback: an inserted newline is removed (swallow_enter_)
  // and the caret is put back when Up/Down move the completion selection.
  // Escape is locked here: ImGui would revert every edit since activation.
  input_id_ = ImGui::GetID(kInputLabel);
  const ImGuiID key_owner = ImGui::GetID("##repl_keys");
  const bool active = ImGui::GetActiveID() == input_id_;
  if (active) {
    if (ImGui::IsKeyPressed(ImGuiKey_Escape, false))
      completion_open_ = false;
    ImGui::SetKeyOwner(ImGuiKey_Escape, key_owner, ImGuiInputFlags_LockThisFrame);

    const bool enter = ImGui::IsKeyPressed(ImGuiKey_Enter, true) ||
                       ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, true);
    const std::string current(input_buffer_.data());
    const size_t cursor = static_cast<size_t>(
        std::clamp(cursor_pos_, 0, static_cast<int>(current.size())));
    const bool at_end = current.find_first_not_of(" \t\r\n", cursor) == std::string::npos;
    if (enter) {
      const bool can_submit = status_.can_run && !command_executing_;
      auto blocked = [&]() {
        notice_ = command_executing_ ? "A command is running. Press Ctrl+C to interrupt it."
                                     : (status_.message.empty() ? "Python is not ready."
                                                                : status_.message);
        notice_timer_ = 3.0f;
      };
      if (io.KeyCtrl) {
        // Ctrl+Enter validates the widget (no newline) and drops focus.
        completion_open_ = false;
        focus_input_ = true;
        if (can_submit)
          submit_requested_ = true;
        else
          blocked();
      } else if (completion_open_ && !io.KeyShift) {
        swallow_enter_ = true;
        completion_accept_ = true;
      } else if (!io.KeyShift && at_end && ReadyToRun(current)) {
        swallow_enter_ = true;
        if (can_submit)
          submit_requested_ = true;
        else
          blocked();
      } else {
        pending_enter_ = true;  // widget inserts the newline; we indent
      }
    }
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_UpArrow, true)) {
      pending_history_up_ = true;
    } else if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_DownArrow, true)) {
      pending_history_down_ = true;
    } else if (completion_open_) {
      const int count = std::max(1, static_cast<int>(completion_items_.size()));
      if (ImGui::IsKeyPressed(ImGuiKey_UpArrow, true)) {
        completion_selected_ = (completion_selected_ + count - 1) % count;
        completion_scroll_ = true;
        restore_cursor_ = true;
      } else if (ImGui::IsKeyPressed(ImGuiKey_DownArrow, true)) {
        completion_selected_ = (completion_selected_ + 1) % count;
        completion_scroll_ = true;
        restore_cursor_ = true;
      }
    }
  }

  // The widget draws selection and handles editing; its own text and caret
  // are transparent and the coloured text is drawn on top.
  ImGui::SetCursorScreenPos(ImVec2(box_min.x + gutter_w, box_min.y));
  ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0, 0, 0, 0));
  ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0, 0, 0, 0));
  ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0, 0, 0, 0));
  ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, ImVec4(0.357f, 0.239f, 0.961f, 0.35f));
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f, pad_y));
  ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 0.0f);
  if (focus_input_) {
    ImGui::SetKeyboardFocusHere();
    focus_input_ = false;
  }
  const ImGuiInputTextFlags flags = ImGuiInputTextFlags_CallbackAlways |
                                    ImGuiInputTextFlags_CallbackCompletion |
                                    ImGuiInputTextFlags_CallbackResize;
  ImGui::InputTextMultiline(kInputLabel, input_buffer_.data(), input_buffer_.size(),
                            ImVec2(avail_w - gutter_w - 1.0f, input_h), flags,
                            &PythonReplSession::InputTextCallback, this);
  input_active_ = ImGui::IsItemActive();
  ImGui::PopStyleVar(2);
  ImGui::PopStyleColor(4);

  // Locate the widget's child window for its scroll and clip rectangle.
  ImGuiWindow *parent = ImGui::GetCurrentWindow();
  char child_name[512];
  ImFormatString(child_name, sizeof(child_name), "%s/%s_%08X", parent->Name, kInputLabel,
                 input_id_);
  ImGuiWindow *child = ImGui::FindWindowByName(child_name);
  ImGuiInputTextState *state = ImGui::GetInputTextState(input_id_);
  const float scroll_x = (input_active_ && state) ? state->Scroll.x : 0.0f;
  const float scroll_y = child ? child->Scroll.y : 0.0f;
  const ImVec2 text_origin(box_min.x + gutter_w + 10.0f - scroll_x,
                           box_min.y + pad_y - scroll_y);
  const ImVec4 clip(box_min.x + gutter_w + 2.0f, box_min.y + 2.0f, box_max.x - 2.0f,
                    box_min.y + input_h - 2.0f);

  const std::string text(input_buffer_.data());
  if (text != highlight_text_) {
    highlight_text_ = text;
    highlight_lines_ = repl::HighlightPython(text);
  }
  ImDrawList *text_dl = child ? child->DrawList : dl;
  text_dl->PushClipRect(ImVec2(clip.x, clip.y), ImVec2(clip.z, clip.w), false);
  const int line_count = 1 + static_cast<int>(std::count(text.begin(), text.end(), '\n'));
  for (int l = 0; l < static_cast<int>(highlight_lines_.size()); ++l) {
    const float y = text_origin.y + l * line_h;
    if (y + line_h < clip.y || y > clip.w)
      continue;
    DrawTokens(text_dl, ImVec2(text_origin.x, y), highlight_lines_[l]);
  }
  if (text.empty() && !input_active_) {
    text_dl->AddText(text_origin, ImGui::GetColorU32(kFaint),
                     "Type Python code. Enter runs it when complete.");
  }
  // Caret.
  float caret_x = text_origin.x;
  float caret_y = text_origin.y;
  {
    const int cursor = std::clamp((input_active_ && state) ? state->GetCursorPos()
                                                           : cursor_pos_,
                                  0, static_cast<int>(text.size()));
    const size_t line_start =
        text.rfind('\n', static_cast<size_t>(cursor > 0 ? cursor - 1 : 0));
    const size_t begin =
        (cursor == 0 || line_start == std::string::npos) ? 0 : line_start + 1;
    const int line_index =
        static_cast<int>(std::count(text.begin(), text.begin() + cursor, '\n'));
    caret_x = text_origin.x +
              ImGui::CalcTextSize(text.c_str() + begin, text.c_str() + cursor).x;
    caret_y = text_origin.y + line_index * line_h;
    const bool blink_on =
        !io.ConfigInputTextCursorBlink || state == nullptr ||
        state->CursorAnim <= 0.0f || std::fmod(state->CursorAnim, 1.20f) <= 0.80f;
    if (input_active_ && blink_on) {
      text_dl->AddRectFilled(ImVec2(caret_x, caret_y + 1.0f),
                             ImVec2(caret_x + 2.0f, caret_y + line_h - 1.0f),
                             ImGui::GetColorU32(kAccent));
    }
  }
  text_dl->PopClipRect();

  // Gutter: >>> on the first line, ... on the rest.
  dl->PushClipRect(ImVec2(box_min.x, box_min.y + 2.0f),
                   ImVec2(box_min.x + gutter_w, box_min.y + input_h - 2.0f), true);
  for (int l = 0; l < line_count; ++l) {
    const float y = text_origin.y + l * line_h;
    if (y + line_h < box_min.y || y > box_min.y + input_h)
      continue;
    const char *prompt = l == 0 ? ">>>" : "...";
    const float w = ImGui::CalcTextSize(prompt).x;
    dl->AddText(ImVec2(box_min.x + gutter_w - 10.0f - w, y),
                ImGui::GetColorU32(l == 0 ? kPrompt : kContinuation), prompt);
  }
  dl->PopClipRect();
  dl->AddLine(ImVec2(box_min.x + gutter_w, box_min.y + 1.0f),
              ImVec2(box_min.x + gutter_w, box_min.y + input_h),
              ImGui::GetColorU32(kInnerBorder));
  completion_anchor_x_ = caret_x;
  completion_anchor_y_ = caret_y;
  if (mono)
    ImGui::PopFont();

  // Bottom bar: line count, Clear input, Run.
  const float bar_y = box_min.y + input_h;
  dl->AddRectFilled(ImVec2(box_min.x + 1.0f, bar_y), ImVec2(box_max.x - 1.0f, box_max.y - 1.0f),
                    ImGui::GetColorU32(kBarBg), 8.0f, ImDrawFlags_RoundCornersBottom);
  dl->AddLine(ImVec2(box_min.x + 1.0f, bar_y), ImVec2(box_max.x - 1.0f, bar_y),
              ImGui::GetColorU32(kInnerBorder));
  ImGui::SetCursorScreenPos(ImVec2(box_min.x + 10.0f, bar_y + 4.0f));
  ImGui::AlignTextToFramePadding();
  ImGui::TextColored(kFaint, "%s", repl::LineCountLabel(text).c_str());
  if (!notice_.empty()) {
    ImGui::SameLine(0.0f, 16.0f);
    ImGui::TextColored(kAccent, "%s", notice_.c_str());
  }

  const std::string run_label =
      std::string(ICON_FA_PLAY) + " Run  Ctrl+Enter";
  const char *clear_label = "Clear input";
  const float run_w = ui::ButtonWidth(run_label.c_str(), ui::ButtonSize::Small);
  const float clear_w = ui::ButtonWidth(clear_label, ui::ButtonSize::Small);
  ImGui::SameLine();
  ImGui::SetCursorScreenPos(
      ImVec2(box_max.x - 10.0f - run_w - style.ItemSpacing.x - clear_w, bar_y + 4.0f));
  if (ui::SecondaryButton(clear_label, !text.empty(), "The input is empty.")) {
    SetInputText("");
    completion_open_ = false;
    focus_input_ = true;
  }
  ImGui::SameLine();
  const bool can_run = status_.can_run && !command_executing_ && !Trim(text).empty();
  const char *run_reason = command_executing_
                               ? "A command is running."
                               : (!status_.can_run ? (status_.message.empty()
                                                          ? "Python is not ready."
                                                          : status_.message.c_str())
                                                   : "Type some Python code first.");
  if (ui::PrimaryButton(run_label.c_str(), can_run, run_reason, ui::ButtonSize::Small)) {
    submitted_text_ = text;
    has_submission_ = true;
    SetInputText("");
    focus_input_ = true;
  }

  // Border last, so it sits over the widget. Neutral only (owner, 2026-09-27:
  // no focus highlight on the input).
  dl->AddRect(box_min, box_max, ImGui::GetColorU32(kBorder), 8.0f);

  // Hints row.
  // Hints wrap onto more rows in a narrow panel instead of running off it.
  ImGui::SetCursorScreenPos(ImVec2(box_min.x, box_max.y + 7.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(14.0f, 4.0f));
  struct HintItem {
    const char *key;
    const char *text;
    const char *key2;
  };
  const HintItem hints[] = {{"Enter", "run when complete", nullptr},
                            {"Shift+Enter", "new line", nullptr},
                            {"Ctrl+Enter", "run now", nullptr},
                            {"Ctrl+Up", "history", "Ctrl+Down"},
                            {"Tab", "complete", nullptr},
                            {"Ctrl+L", "clear", nullptr},
                            {"Ctrl+C", "interrupt", nullptr}};
  const char *scroll_label = auto_scroll_ ? "Auto-scroll on" : "Auto-scroll off";
  const float scroll_w = ImGui::CalcTextSize(scroll_label).x;
  int rows = 1;
  float row_x = box_min.x;
  for (size_t h = 0; h < std::size(hints); ++h) {
    float w = ImGui::CalcTextSize(hints[h].key).x + 14.0f +
              ImGui::CalcTextSize(hints[h].text).x;
    if (hints[h].key2)
      w += ImGui::CalcTextSize(hints[h].key2).x + 12.0f;
    if (h > 0) {
      if (row_x + 14.0f + w > box_max.x - scroll_w - 14.0f) {
        ++rows;
        row_x = box_min.x;
        ImGui::SetCursorScreenPos(ImVec2(box_min.x, ImGui::GetCursorScreenPos().y));
      } else {
        ImGui::SameLine();
        row_x += 14.0f;
      }
    }
    Hint(hints[h].key, hints[h].text, hints[h].key2);
    row_x += w;
  }
  hint_rows_ = rows;
  ImGui::SameLine();
  const float scroll_x_pos = box_max.x - scroll_w;
  if (scroll_x_pos > ImGui::GetCursorScreenPos().x)
    ImGui::SetCursorScreenPos(ImVec2(scroll_x_pos, ImGui::GetCursorScreenPos().y));
  ImGui::TextColored(auto_scroll_ ? kFaint : kWarning, "%s", scroll_label);
  if (ImGui::IsItemHovered()) {
    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    ImGui::SetTooltip("Click to turn auto-scroll %s", auto_scroll_ ? "off" : "on");
  }
  if (ImGui::IsItemClicked())
    auto_scroll_ = !auto_scroll_;
  ImGui::PopStyleVar();

  RenderCompletionPopup();

  if (has_submission_) {
    has_submission_ = false;
    Submit();
  }
}

void PythonReplSession::RenderCompletionPopup() {
  if (!completion_open_ || completion_items_.empty())
    return;
  ImFont *mono = ReplMonoFont();
  ImGui::SetNextWindowPos(ImVec2(completion_anchor_x_ - 6.0f, completion_anchor_y_ - 2.0f),
                          ImGuiCond_Always, ImVec2(0.0f, 1.0f));
  const int visible = std::min<int>(8, static_cast<int>(completion_items_.size()));
  const float row_h = (mono ? mono->FontSize : ImGui::GetTextLineHeight()) +
                      ImGui::GetStyle().ItemSpacing.y;
  ImGui::SetNextWindowSizeConstraints(ImVec2(220.0f, 0.0f), ImVec2(520.0f, FLT_MAX));
  ImGui::PushStyleColor(ImGuiCol_WindowBg, kPanelBg);
  ImGui::PushStyleColor(ImGuiCol_Border, kBorder);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 6.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(4.0f, 4.0f));
  const ImGuiWindowFlags flags =
      ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
      ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
      ImGuiWindowFlags_NoNav | ImGuiWindowFlags_AlwaysAutoResize |
      ImGuiWindowFlags_NoDocking;
  if (ImGui::Begin("##repl_completion", nullptr, flags)) {
    ImGui::BringWindowToDisplayFront(ImGui::GetCurrentWindow());
    if (mono)
      ImGui::PushFont(mono);
    ImGui::BeginChild("##items", ImVec2(0.0f, visible * row_h),
                      ImGuiChildFlags_AutoResizeX, ImGuiWindowFlags_NoNav);
    for (int i = 0; i < static_cast<int>(completion_items_.size()); ++i) {
      const bool selected = i == completion_selected_;
      ImGui::PushID(i);
      if (ImGui::Selectable(completion_items_[i].c_str(), selected)) {
        // Input lost focus to this click: edit the buffer directly.
        std::string current(input_buffer_.data());
        const size_t cursor = std::min<size_t>(cursor_pos_, current.size());
        const std::string word = repl::CompletionWordAt(current, cursor);
        const size_t start = cursor - word.size();
        current.replace(start, word.size(), completion_items_[i]);
        SetInputText(current);
        pending_cursor_ = static_cast<int>(start + completion_items_[i].size());
        completion_open_ = false;
        focus_input_ = true;
      }
      if (selected && completion_scroll_)
        ImGui::SetScrollHereY();
      ImGui::PopID();
    }
    completion_scroll_ = false;
    ImGui::EndChild();
    if (mono)
      ImGui::PopFont();
    ImGui::TextColored(kFaint, "Tab/Enter insert \xC2\xB7 Esc close \xC2\xB7 %d matches",
                       static_cast<int>(completion_items_.size()));
  }
  ImGui::End();
  ImGui::PopStyleVar(2);
  ImGui::PopStyleColor(2);
}

// ---------------------------------------------------------------------------
// Input callback
// ---------------------------------------------------------------------------

int PythonReplSession::InputTextCallback(ImGuiInputTextCallbackData *data) {
  return static_cast<PythonReplSession *>(data->UserData)->HandleInputTextCallback(data);
}

void PythonReplSession::ApplyCompletion(ImGuiInputTextCallbackData *data,
                                        const std::string &completion) {
  const std::string text(data->Buf, data->BufTextLen);
  const std::string word = repl::CompletionWordAt(text, data->CursorPos);
  const int start = data->CursorPos - static_cast<int>(word.size());
  data->DeleteChars(start, static_cast<int>(word.size()));
  data->InsertChars(start, completion.c_str());
}

int PythonReplSession::HandleInputTextCallback(ImGuiInputTextCallbackData *data) {
  if (data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
    input_buffer_.resize(static_cast<size_t>(data->BufSize));
    data->Buf = input_buffer_.data();
    return 0;
  }

  if (data->EventFlag == ImGuiInputTextFlags_CallbackCompletion) {
    if (completion_open_ && !completion_items_.empty()) {
      ApplyCompletion(data, completion_items_[completion_selected_]);
      completion_open_ = false;
      return 0;
    }
    const std::string text(data->Buf, data->BufTextLen);
    const std::string word = repl::CompletionWordAt(text, data->CursorPos);
    if (word.empty()) {
      data->InsertChars(data->CursorPos, "    ");  // indent
      return 0;
    }
    if (!scripting_engine_ || command_executing_) {
      notice_ = command_executing_ ? "Completions are unavailable while a command runs"
                                   : "Python is not available";
      notice_timer_ = 2.5f;
      return 0;
    }
    std::vector<std::string> items = scripting_engine_->CompleteSync(word);
    if (word.find('.') == std::string::npos) {
      // CyxWiz names worth offering before they are imported or defined.
      static const char *const kExtras[] = {"import", "pycyxwiz", "math", "random",
                                            "json",   "numpy",    "help", "clear"};
      for (const char *extra : kExtras) {
        const std::string name(extra);
        if (StartsWith(name, word) && name != word &&
            std::find(items.begin(), items.end(), name) == items.end())
          items.push_back(name);
      }
    }
    if (items.empty()) {
      notice_ = interpreter_.initialized
                    ? "No completions for '" + word + "'"
                    : "Completions are available after Python starts (run a command)";
      notice_timer_ = 2.5f;
      return 0;
    }
    if (items.size() == 1) {
      ApplyCompletion(data, items.front());
      return 0;
    }
    std::sort(items.begin(), items.end());
    const std::string prefix = CommonPrefix(items);
    if (prefix.size() > word.size())
      ApplyCompletion(data, prefix);
    completion_all_ = std::move(items);
    completion_items_ = completion_all_;
    completion_selected_ = 0;
    completion_open_ = true;
    completion_word_ = prefix.size() > word.size() ? prefix : word;
    return 0;
  }

  if (data->EventFlag != ImGuiInputTextFlags_CallbackAlways)
    return 0;

  if (pending_cursor_ >= 0) {
    data->CursorPos = std::min(pending_cursor_, data->BufTextLen);
    data->SelectionStart = data->SelectionEnd = data->CursorPos;
    pending_cursor_ = -1;
  }

  if (restore_cursor_) {
    restore_cursor_ = false;
    data->CursorPos = std::min(cursor_pos_, data->BufTextLen);
    data->SelectionStart = data->SelectionEnd = data->CursorPos;
  }

  if (swallow_enter_) {
    swallow_enter_ = false;
    if (data->CursorPos > 0 && data->Buf[data->CursorPos - 1] == '\n')
      data->DeleteChars(data->CursorPos - 1, 1);
  }

  if (completion_accept_) {
    completion_accept_ = false;
    if (completion_open_ && !completion_items_.empty())
      ApplyCompletion(data, completion_items_[completion_selected_]);
    completion_open_ = false;
  }

  if (pending_enter_) {
    pending_enter_ = false;
    if (data->CursorPos > 0 && data->Buf[data->CursorPos - 1] == '\n') {
      const std::string before(data->Buf, data->CursorPos - 1);
      const std::string indent = repl::NextLineIndent(before);
      if (!indent.empty())
        data->InsertChars(data->CursorPos, indent.c_str());
    }
  }

  if (pending_history_up_ || pending_history_down_) {
    const int direction = pending_history_up_ ? -1 : 1;
    pending_history_up_ = pending_history_down_ = false;
    std::string recalled;
    if (RecallHistory(direction, std::string(data->Buf, data->BufTextLen), recalled)) {
      data->DeleteChars(0, data->BufTextLen);
      data->InsertChars(0, recalled.c_str());
      completion_open_ = false;
    }
  }

  if (submit_requested_) {
    submit_requested_ = false;
    submitted_text_.assign(data->Buf, data->BufTextLen);
    has_submission_ = true;
    data->DeleteChars(0, data->BufTextLen);
    completion_open_ = false;
  }

  // Keep the completion list filtered by the word being typed.
  if (completion_open_) {
    const std::string text(data->Buf, data->BufTextLen);
    const std::string word = repl::CompletionWordAt(text, data->CursorPos);
    if (word != completion_word_) {
      completion_word_ = word;
      completion_items_.clear();
      if (!word.empty()) {
        for (const auto &item : completion_all_)
          if (StartsWith(item, word))
            completion_items_.push_back(item);
      }
      completion_selected_ = 0;
      completion_open_ = !completion_items_.empty();
    }
  }

  cursor_pos_ = data->CursorPos;
  return 0;
}

void PythonReplSession::SetInputText(const std::string &text) {
  if (input_buffer_.size() < text.size() + 1)
    input_buffer_.resize(text.size() + 1024, '\0');
  std::fill(input_buffer_.begin(), input_buffer_.end(), '\0');
  std::copy(text.begin(), text.end(), input_buffer_.begin());
  pending_cursor_ = static_cast<int>(text.size());
  // An active widget keeps its own copy; drop focus so it reloads this text.
  if (ImGui::GetCurrentContext() && input_id_ != 0 &&
      ImGui::GetActiveID() == input_id_)
    ImGui::ClearActiveID();
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

void PythonReplSession::Submit() {
  const std::string command = NormalizeSubmission(submitted_text_);
  submitted_text_.clear();
  history_position_ = -1;
  history_draft_.clear();
  if (Trim(command).empty())
    return;
  ExecuteCommand(command);
  focus_input_ = true;
}

PythonReplSession::Entry &PythonReplSession::AddEntry(Entry entry) {
  entries_.push_back(std::move(entry));
  if (entries_.size() > kMaxEntries) {
    const int drop = static_cast<int>(entries_.size() - kMaxEntries);
    entries_.erase(entries_.begin(), entries_.begin() + drop);
    running_entry_ = running_entry_ >= drop ? running_entry_ - drop : -1;
    selected_entry_ = selected_entry_ >= drop ? selected_entry_ - drop : -1;
  }
  scroll_to_bottom_ = true;
  return entries_.back();
}

void PythonReplSession::AddSystem(const std::string &text) {
  Entry entry;
  entry.kind = Entry::Kind::System;
  entry.output = text;
  AddEntry(std::move(entry));
}

void PythonReplSession::ExecuteCommand(const std::string &command) {
  const std::string trimmed = Trim(command);
  if (trimmed.empty())
    return;
  AddToHistory(command);

  if (trimmed == "clear") {
    ClearTranscript();
    return;
  }

  Entry entry;
  entry.kind = Entry::Kind::Command;
  entry.code = command;
  entry.code_lines = repl::HighlightPython(command);

  if (trimmed == "help" || trimmed == "help()") {
    entry.output = kHelpText;
    AddEntry(std::move(entry));
    return;
  }
  if (trimmed.front() == '/') {
    entry.failed = true;
    entry.error = "Slash commands are not Python. Use the Agent LLM Console "
                  "session for assistant requests.";
    AddEntry(std::move(entry));
    return;
  }
  if (!scripting_engine_) {
    entry.failed = true;
    entry.error = "Error: Scripting engine not initialized";
    AddEntry(std::move(entry));
    return;
  }

  entry.running = true;
  AddEntry(std::move(entry));
  running_entry_ = static_cast<int>(entries_.size()) - 1;
  StartAsyncCommand(command);
}

void PythonReplSession::ClearTranscript() {
  entries_.clear();
  selected_entry_ = -1;
  running_entry_ = -1;
  if (command_executing_) {
    // Keep showing the command that is still running.
    Entry entry;
    entry.kind = Entry::Kind::System;
    entry.output = "Output cleared. A command is still running.";
    AddEntry(std::move(entry));
  }
}

std::string PythonReplSession::EntryText(const Entry &entry) {
  std::string text;
  auto add = [&text](const std::string &part) {
    if (part.empty())
      return;
    text += part;
    if (text.back() != '\n')
      text += '\n';
  };
  switch (entry.kind) {
  case Entry::Kind::System:
    add(entry.output);
    break;
  case Entry::Kind::Command: {
    const auto lines = SplitLines(entry.code);
    for (size_t i = 0; i < lines.size(); ++i)
      text += (i == 0 ? ">>> " : "... ") + lines[i] + "\n";
    add(entry.output);
    add(entry.warnings);
    if (entry.cancelled)
      add(entry.error.empty() ? "Interrupted" : entry.error);
    else if (entry.failed)
      add(entry.traceback.empty() ? entry.error : entry.traceback);
    break;
  }
  case Entry::Kind::Script: {
    std::string title = "Script " + entry.source;
    if (entry.running)
      title += " running...";
    else if (entry.cancelled)
      title += " cancelled after " + repl::FormatElapsed(entry.seconds);
    else if (entry.failed)
      title += " failed after " + repl::FormatElapsed(entry.seconds);
    else
      title += " finished in " + repl::FormatElapsed(entry.seconds);
    add(title);
    add(entry.output);
    add(entry.warnings);
    add(entry.error);
    break;
  }
  }
  if (!entry.hint.empty())
    add("Install packages into the project environment: " + entry.hint);
  return text;
}

void PythonReplSession::CopyAll() {
  std::string text = repl::WelcomeText(status_) + "\n\n";
  for (const auto &entry : entries_) {
    text += EntryText(entry);
    text += '\n';
  }
  ImGui::SetClipboardText(text.c_str());
  notice_ = "Copied the whole transcript";
  notice_timer_ = 2.0f;
}

void PythonReplSession::ResetSession() {
  if (!scripting_engine_)
    return;
  std::string error;
  if (scripting_engine_->ResetSession(&error)) {
    AddSystem("Session restarted: variables and imports cleared.");
  } else {
    Entry entry;
    entry.kind = Entry::Kind::System;
    entry.output = "Could not restart the session: " + error;
    AddEntry(std::move(entry));
  }
  RefreshInterpreterInfo(true);
}

// ---------------------------------------------------------------------------
// Script output (Script Editor runs)
// ---------------------------------------------------------------------------

void PythonReplSession::AppendScriptOutput(const std::string &source,
                                           const std::string &text, bool is_error) {
  Entry *target = nullptr;
  for (auto it = entries_.rbegin(); it != entries_.rend(); ++it) {
    if (it->kind == Entry::Kind::Script && it->running && it->source == source) {
      target = &*it;
      break;
    }
  }
  if (!target) {
    Entry entry;
    entry.kind = Entry::Kind::Script;
    entry.source = source;
    entry.running = true;
    target = &AddEntry(std::move(entry));
  }
  if (!text.empty()) {
    AppendCapped(is_error ? target->error : target->output, text);
    if (is_error && target->hint.empty()) {
      if (auto hint = repl::MissingModuleInstallCommand(text))
        target->hint = *hint;
    }
  }
  scroll_to_bottom_ = true;
}

void PythonReplSession::EndScriptOutput(const std::string &source, bool success,
                                        bool cancelled, double seconds) {
  for (auto it = entries_.rbegin(); it != entries_.rend(); ++it) {
    if (it->kind == Entry::Kind::Script && it->running && it->source == source) {
      it->running = false;
      it->failed = !success;
      it->cancelled = cancelled;
      it->seconds = seconds;
      if (success && !it->error.empty()) {
        // stderr from a successful run is warnings.
        it->warnings = std::move(it->error);
        it->error.clear();
      }
      if (cancelled && it->error.empty())
        it->error = "Cancelled by user";
      break;
    }
  }
  scroll_to_bottom_ = true;
  RefreshInterpreterInfo(true);
}

// ---------------------------------------------------------------------------
// History
// ---------------------------------------------------------------------------

void PythonReplSession::AddToHistory(const std::string &command) {
  if (command.empty())
    return;
  history_position_ = -1;
  if (!command_history_.empty() && command_history_.back() == command)
    return;
  command_history_.push_back(command);
  if (command_history_.size() > kMaxHistory)
    command_history_.erase(command_history_.begin());
}

bool PythonReplSession::RecallHistory(int direction, const std::string &current,
                                      std::string &out) {
  const int size = static_cast<int>(command_history_.size());
  if (size == 0)
    return false;
  if (direction < 0) {
    if (history_position_ >= size - 1)
      return false;
    if (history_position_ == -1)
      history_draft_ = current;
    ++history_position_;
    out = command_history_[size - 1 - history_position_];
    return true;
  }
  if (history_position_ < 0)
    return false;
  --history_position_;
  out = history_position_ < 0 ? history_draft_
                              : command_history_[size - 1 - history_position_];
  return true;
}

// ---------------------------------------------------------------------------
// Async execution
// ---------------------------------------------------------------------------

void PythonReplSession::StartAsyncCommand(const std::string &command) {
  if (command_executing_ || !scripting_engine_)
    return;
  command_executing_ = true;
  command_started_ = std::chrono::steady_clock::now();
  scripting_engine_->ExecuteCommandAsync(command);
}

void PythonReplSession::CheckAsyncCompletion() {
  if (!command_executing_ || !scripting_engine_)
    return;
  if (scripting_engine_->IsCommandRunning())
    return;

  auto result_opt = scripting_engine_->GetCommandResult();
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - command_started_)
          .count();
  if (running_entry_ >= 0 && running_entry_ < static_cast<int>(entries_.size())) {
    Entry &entry = entries_[running_entry_];
    entry.running = false;
    entry.seconds = seconds;
    if (result_opt) {
      const auto &result = *result_opt;
      AppendCapped(entry.output, result.output);
      AppendCapped(entry.warnings, result.stderr_output);
      if (!result.success) {
        entry.failed = true;
        if (result.timeout_exceeded) {
          entry.error = result.error_message.empty() ? "Command interrupted (timeout)"
                                                     : result.error_message;
        } else if (result.was_cancelled || result.exception_type == "KeyboardInterrupt") {
          entry.cancelled = true;
          entry.error = "Interrupted";
        } else {
          entry.error = result.error_message.empty() ? "Error" : result.error_message;
          entry.traceback = result.traceback;
          if (auto hint = repl::MissingModuleInstallCommand(entry.error))
            entry.hint = *hint;
        }
      }
    }
    scroll_to_bottom_ = true;
  }

  command_executing_ = false;
  running_entry_ = -1;
  focus_input_ = true;
  RefreshInterpreterInfo(true);
}

void PythonReplSession::StopAsyncCommand() {
  if (!command_executing_ || !scripting_engine_)
    return;
  scripting_engine_->StopCommand();
  notice_ = "Interrupting... Python stops at its next line.";
  notice_timer_ = 4.0f;
}

} // namespace cyxwiz
