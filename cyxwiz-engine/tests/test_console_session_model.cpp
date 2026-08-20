#include "../src/gui/console_session_model.h"
#include "../src/gui/console_workbench.h"
#include "../src/gui/editor_fonts.h"
#include "../src/gui/panels/agent_llm_session.h"
#include "../src/gui/panels/local_shell_process.h"
#include "../src/gui/panels/terminal_buffer.h"

#include <imgui.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

namespace {

void Check(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

void TestProfiles() {
  const auto &profiles = gui::ConsoleSessionModel::Profiles();
  Check(profiles.size() == 8, "profile registry should expose eight entries");
  Check(profiles[0].kind == gui::ConsoleSessionKind::PythonRepl &&
            profiles[1].kind == gui::ConsoleSessionKind::AgentLlm &&
            profiles[2].kind == gui::ConsoleSessionKind::Commands &&
            profiles[3].kind == gui::ConsoleSessionKind::Logs &&
            profiles[4].kind == gui::ConsoleSessionKind::PowerShell &&
            profiles[5].kind == gui::ConsoleSessionKind::GitBash &&
            profiles[6].kind == gui::ConsoleSessionKind::CommandPrompt &&
            profiles[7].kind == gui::ConsoleSessionKind::SystemShell,
        "profile registry should preserve the approved plus-menu order");
  Check(profiles[0].singleton && profiles[1].singleton &&
            profiles[2].singleton && profiles[3].singleton,
        "shared-authority sessions should be singletons");
  Check(!profiles[4].singleton && !profiles[5].singleton &&
            !profiles[6].singleton && !profiles[7].singleton,
        "shell profiles should allow independent instances");
  Check(profiles[0].requires_project && profiles[1].requires_project &&
            !profiles[2].requires_project && !profiles[3].requires_project &&
            profiles[4].requires_project && profiles[5].requires_project &&
            profiles[6].requires_project && profiles[7].requires_project,
        "project requirements should match the ticket contract");
}

void TestImplementedSessionAvailability() {
  Check(
      gui::ConsoleWorkbench::IsImplemented(
          gui::ConsoleSessionKind::PythonRepl) &&
          gui::ConsoleWorkbench::IsImplemented(
              gui::ConsoleSessionKind::AgentLlm) &&
          gui::ConsoleWorkbench::IsImplemented(gui::ConsoleSessionKind::Logs) &&
          gui::ConsoleWorkbench::IsImplemented(
              gui::ConsoleSessionKind::Commands),
      "Console should expose every platform-neutral session authority");
#ifdef _WIN32
  Check(gui::ConsoleWorkbench::IsImplemented(
            gui::ConsoleSessionKind::CommandPrompt) &&
            gui::ConsoleWorkbench::IsImplemented(
                gui::ConsoleSessionKind::PowerShell) &&
            gui::ConsoleWorkbench::IsImplemented(
                gui::ConsoleSessionKind::GitBash) &&
            !gui::ConsoleWorkbench::IsImplemented(
                gui::ConsoleSessionKind::SystemShell) &&
            !gui::ConsoleWorkbench::IsVisible(
                gui::ConsoleSessionKind::SystemShell),
        "Windows should expose its three native shell profiles");
#else
  Check(gui::ConsoleWorkbench::IsImplemented(
            gui::ConsoleSessionKind::SystemShell) &&
            gui::ConsoleWorkbench::IsVisible(
                gui::ConsoleSessionKind::SystemShell) &&
            !gui::ConsoleWorkbench::IsVisible(
                gui::ConsoleSessionKind::CommandPrompt) &&
            !gui::ConsoleWorkbench::IsVisible(
                gui::ConsoleSessionKind::PowerShell) &&
            !gui::ConsoleWorkbench::IsVisible(gui::ConsoleSessionKind::GitBash),
        "POSIX should expose one platform-native system shell profile");
#endif
}

void TestSessionActivationDefersInputFocus() {
  gui::ConsoleWorkbench workbench;
  Check(workbench.ConsumeFocusRequest(),
        "the initial Logs session should request primary-input focus");
  Check(!workbench.ConsumeFocusRequest(),
        "a focus request should be consumed exactly once");

  const auto commands =
      workbench.ActivateSession(gui::ConsoleSessionKind::Commands);
  Check(static_cast<bool>(commands),
        "Commands should activate through the shared workbench");
  Check(!workbench.ConsumeFocusRequest(),
        "popup-frame focus should be deferred past ImGui focus restoration");
  Check(workbench.ConsumeFocusRequest(),
        "the activated session should receive focus on the next frame");
  Check(!workbench.ConsumeFocusRequest(),
        "deferred focus should remain a one-shot handoff");
}

void TestProgrammaticNavigationKeepsNewSessionActive() {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  auto &io = ImGui::GetIO();
  io.DisplaySize = ImVec2(800.0f, 600.0f);
  io.DeltaTime = 1.0f / 60.0f;
  unsigned char *font_pixels = nullptr;
  int font_width = 0;
  int font_height = 0;
  io.Fonts->GetTexDataAsRGBA32(&font_pixels, &font_width, &font_height);

  gui::ConsoleWorkbench workbench;
  workbench.SetProjectRoot("D:/projects/Project A");

  ImGui::NewFrame();
  ImGui::SetNextWindowSize(ImVec2(700.0f, 400.0f));
  ImGui::Begin("Console navigation test");
  workbench.RenderCommandBar();
#ifdef _WIN32
  const auto shell =
      workbench.ActivateSession(gui::ConsoleSessionKind::GitBash);
#else
  const auto shell =
      workbench.ActivateSession(gui::ConsoleSessionKind::SystemShell);
#endif
  ImGui::End();
  ImGui::Render();
  Check(shell && workbench.ActiveSessionId() == shell.session_id,
        "new shell should become the authoritative active session");

  ImGui::NewFrame();
  ImGui::SetNextWindowSize(ImVec2(700.0f, 400.0f));
  ImGui::Begin("Console navigation test");
  workbench.RenderCommandBar();
  ImGui::End();
  ImGui::Render();
  Check(workbench.ActiveSessionId() == shell.session_id,
        "stale ImGui tab visibility must not reactivate the previous session");

  ImGui::DestroyContext();
}

void TestBundledTerminalFontCoversCliGlyphs() {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  const auto font_path = std::filesystem::current_path() / "cyxwiz-engine" /
                         "resources" / "fonts" / "JetBrainsMono-Regular.ttf";
  Check(std::filesystem::is_regular_file(font_path),
        "the bundled terminal font should be available to the test");
  auto &io = ImGui::GetIO();
#ifdef _WIN32
  const std::filesystem::path symbol_font = "C:/Windows/Fonts/seguisym.ttf";
  Check(std::filesystem::is_regular_file(symbol_font),
        "the Windows terminal symbol fallback should be available");
  const auto symbol_font_string = symbol_font.string();
  const char *symbol_font_path = symbol_font_string.c_str();
#else
  const char *symbol_font_path = nullptr;
#endif
  ImFont *font = cyxwiz::gui::AddTerminalCapableMonoFont(
      io.Fonts, font_path.string().c_str(), symbol_font_path, 14.0f, nullptr);
  Check(font && io.Fonts->Build(),
        "the bundled terminal font atlas should build");
  for (const ImWchar glyph :
       {static_cast<ImWchar>(0x203A), static_cast<ImWchar>(0x2500),
        static_cast<ImWchar>(0x256D)}) {
    Check(font->FindGlyphNoFallback(glyph) != nullptr,
          "the terminal font should contain Codex CLI presentation glyph U+" +
              std::to_string(static_cast<unsigned int>(glyph)));
  }
  const auto spinner =
      cyxwiz::gui::ResolveTerminalDisplayCodepoint(font, 0x280B);
#ifdef _WIN32
  Check(spinner == 0x280B,
        "Windows should render the exact Codex Braille spinner glyph");
#else
  Check(spinner < 0x80,
        "platforms without a symbol font should use a readable ASCII spinner");
#endif
  ImGui::DestroyContext();
}

#ifdef _WIN32
std::string WaitForShellOutput(cyxwiz::LocalShellProcess &process,
                               std::string_view expected) {
  std::string output;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(8);
  while (std::chrono::steady_clock::now() < deadline) {
    output += process.DrainOutput();
    if (output.find(expected) != std::string::npos)
      return output;
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
  return output;
}

void TestLocalShellProcessesUseProjectDirectory() {
  const auto project_root = std::filesystem::current_path();
  const auto expected = project_root.string();
  auto git_expected = expected;
  std::replace(git_expected.begin(), git_expected.end(), '\\', '/');

  struct ShellCase {
    cyxwiz::LocalShellKind kind;
    std::string command;
    std::string expected;
  };
  for (const auto &shell_case :
       {ShellCase{cyxwiz::LocalShellKind::CommandPrompt, "cd", expected},
        ShellCase{cyxwiz::LocalShellKind::PowerShell, "pwd", expected},
        ShellCase{cyxwiz::LocalShellKind::GitBash, "pwd -W", git_expected}}) {
    cyxwiz::LocalShellProcess process(shell_case.kind);
    std::string error;
    Check(process.Start(project_root, error),
          "local shell should start in the active project: " + error);
    Check(process.Send(shell_case.command, error),
          "local shell should accept input: " + error);
    const auto output = WaitForShellOutput(process, shell_case.expected);
    auto comparable_output = output;
    auto comparable_expected = shell_case.expected;
    std::replace(comparable_output.begin(), comparable_output.end(), '/', '\\');
    std::replace(comparable_expected.begin(), comparable_expected.end(), '/',
                 '\\');
    std::transform(comparable_output.begin(), comparable_output.end(),
                   comparable_output.begin(), [](unsigned char value) {
                     return static_cast<char>(std::tolower(value));
                   });
    std::transform(comparable_expected.begin(), comparable_expected.end(),
                   comparable_expected.begin(), [](unsigned char value) {
                     return static_cast<char>(std::tolower(value));
                   });
    Check(comparable_output.find(comparable_expected) != std::string::npos,
          std::string(cyxwiz::LocalShellProcess::DisplayName(shell_case.kind)) +
              " should report the active project directory; expected=" +
              shell_case.expected + "; output=" + output);
    process.Stop();
    Check(!process.IsRunning(), "local shell should stop deterministically");
  }
}

void TestPowerShellHasInteractiveTerminalSemantics() {
  cyxwiz::LocalShellProcess process(cyxwiz::LocalShellKind::PowerShell);
  std::string error;
  Check(process.Start(std::filesystem::current_path(), error, 96, 28),
        "PowerShell ConPTY should start: " + error);
  Check(process.Send(
            "Write-Output ('TTY=' + (-not [Console]::IsInputRedirected) + "
            "',' + (-not [Console]::IsOutputRedirected))",
            error),
        "PowerShell should accept a TTY probe: " + error);
  auto output = WaitForShellOutput(process, "TTY=True,True");
  Check(output.find("TTY=True,True") != std::string::npos,
        "PowerShell must see terminal handles instead of redirected pipes; "
        "output=" +
            output);

  Check(process.Resize(111, 31, error),
        "ConPTY should accept terminal resize: " + error);
  Check(process.Send("Write-Output ('WIDTH=' + [Console]::WindowWidth)", error),
        "PowerShell should accept a post-resize probe: " + error);
  output = WaitForShellOutput(process, "WIDTH=111");
  Check(output.find("WIDTH=111") != std::string::npos,
        "resizing the panel must update the pseudo-terminal width; output=" +
            output);
  process.Stop();
}
#else
std::string WaitForShellOutput(cyxwiz::LocalShellProcess &process,
                               std::string_view expected) {
  std::string output;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(8);
  while (std::chrono::steady_clock::now() < deadline) {
    output += process.DrainOutput();
    if (output.find(expected) != std::string::npos)
      return output;
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
  return output;
}

void TestSystemShellHasInteractiveTerminalSemantics() {
  const auto project_root = std::filesystem::current_path();
  cyxwiz::LocalShellProcess process(cyxwiz::LocalShellKind::SystemShell);
  std::string error;
  Check(process.Start(project_root, error, 96, 28),
        "POSIX PTY shell should start: " + error);
  Check(process.Send("printf 'ROOT=%s;TERM=%s\\n' \"$PWD\" \"$TERM\"", error),
        "POSIX PTY shell should accept input: " + error);
  const auto expected =
      "ROOT=" + project_root.string() + ";TERM=xterm-256color";
  auto output = WaitForShellOutput(process, expected);
  Check(output.find(expected) != std::string::npos,
        "POSIX shell should inherit the project root and terminal identity; "
        "output=" +
            output);

  Check(process.Resize(111, 31, error),
        "POSIX PTY should accept terminal resize: " + error);
  Check(process.Send("printf 'WIDTH='; stty size | awk '{print $2}'", error),
        "POSIX shell should accept a post-resize probe: " + error);
  output = WaitForShellOutput(process, "WIDTH=111");
  Check(output.find("WIDTH=111") != std::string::npos,
        "resizing the panel must update the POSIX PTY width; output=" + output);
  process.Stop();
  Check(!process.IsRunning(), "POSIX shell should stop deterministically");
}
#endif

void TestTerminalBufferInterpretsAnsiState() {
  cyxwiz::TerminalBuffer terminal(12, 3);
  terminal.Feed("hello\rY\x1b[31m!\x1b[0m");
  Check(terminal.LineAt(0)[0].codepoint == U'Y' &&
            terminal.LineAt(0)[1].codepoint == U'!',
        "carriage return and cursor overwrite should update terminal cells");
  Check(terminal.LineAt(0)[1].foreground == 0xCD3131,
        "ANSI foreground colors should be retained by the terminal buffer");

  terminal.Feed("\x1b[?1049halternate");
  Check(terminal.AlternateScreen() &&
            terminal.PlainText().find("alternate") != std::string::npos,
        "interactive TUIs should receive an isolated alternate screen");
  terminal.Feed("\x1b[?1049l");
  Check(!terminal.AlternateScreen() && terminal.LineAt(0)[0].codepoint == U'Y',
        "leaving the alternate screen should restore shell content");

  terminal.Feed("\x1b[?2004h");
  Check(terminal.EncodePaste("pasted") == "\x1b[200~pasted\x1b[201~",
        "terminal should encode paste using the mode requested by a CLI");

  terminal.Feed("\x1b[?1h");
  Check(terminal.EncodeKey(cyxwiz::TerminalBuffer::Key::Up) == "\x1bOA",
        "terminal key encoding should follow application cursor mode");

  terminal.Feed("\x1b[6n");
  Check(!terminal.TakeOutboundData().empty(),
        "terminal device-status queries should produce a child reply");
}

void TestAgentRequestMapping() {
  using Mode = cyxwiz::AgentLlmSession::ContextMode;
  const auto general = cyxwiz::AgentLlmSession::BuildRequest(
      Mode::General, false, "  explain this  ");
  const auto trace = cyxwiz::AgentLlmSession::BuildRequest(Mode::Trace, false,
                                                           "trace question");
  const auto training = cyxwiz::AgentLlmSession::BuildRequest(
      Mode::Training, false, "training question");
  const auto retrieval = cyxwiz::AgentLlmSession::BuildRequest(
      Mode::Training, true, "source query");

  Check(general.command_name == "ask" && general.user_text == "explain this",
        "general input should map to a trimmed ask request");
  Check(trace.command_name == "explain_trace",
        "trace context should map to the trace provider action");
  Check(training.command_name == "explain_training",
        "training context should map to the training provider action");
  Check(
      retrieval.command_name == "find_source",
      "retrieval-only should override context with the provider search action");
}

void TestEnsureDoesNotStealFocus() {
  gui::ConsoleSessionModel model;
  const auto commands =
      model.CreateOrActivate(gui::ConsoleSessionKind::Commands);
  const auto python = model.Ensure(gui::ConsoleSessionKind::PythonRepl,
                                   "D:/projects/Project A");
  Check(commands && python && python.created,
        "ensure should create the requested project session");
  Check(model.ActiveSessionId() == commands.session_id,
        "ensuring a background-output session must not steal focus");
  Check(model.SetUnread(*python.session_id) &&
            model.Find(*python.session_id)->unread,
        "an ensured background-output session should support unread state");
}

void TestInitialAndSingletonState() {
  gui::ConsoleSessionModel model;
  Check(model.Sessions().size() == 1,
        "model should begin with one lightweight Logs session");
  Check(model.ActiveSession() &&
            model.ActiveSession()->kind == gui::ConsoleSessionKind::Logs,
        "Logs should be active initially");

  const auto existing_logs =
      model.CreateOrActivate(gui::ConsoleSessionKind::Logs);
  Check(existing_logs && !existing_logs.created && model.Sessions().size() == 1,
        "creating singleton Logs should activate the existing session");

  const auto commands =
      model.CreateOrActivate(gui::ConsoleSessionKind::Commands);
  Check(commands && commands.created && model.Sessions().size() == 2,
        "Commands should be created once");
  const auto existing_commands =
      model.CreateOrActivate(gui::ConsoleSessionKind::Commands);
  Check(existing_commands && !existing_commands.created &&
            existing_commands.session_id == commands.session_id,
        "creating singleton Commands should reuse its identity");
}

void TestProjectRequirementsAndIsolation() {
  gui::ConsoleSessionModel model;
  for (const auto kind :
       {gui::ConsoleSessionKind::PythonRepl, gui::ConsoleSessionKind::AgentLlm,
        gui::ConsoleSessionKind::CommandPrompt,
        gui::ConsoleSessionKind::PowerShell, gui::ConsoleSessionKind::GitBash,
        gui::ConsoleSessionKind::SystemShell}) {
    const auto result = model.CreateOrActivate(kind);
    Check(!result &&
              result.error == gui::ConsoleSessionCreateError::ProjectRequired,
          "project-scoped profile should reject an empty project root");
  }

  const std::string project_a = "D:/projects/Project A";
  const std::string project_b = "D:/projects/Project B";
  const auto python =
      model.CreateOrActivate(gui::ConsoleSessionKind::PythonRepl, project_a);
  Check(python && python.created,
        "Python REPL should be created for an active project");
  const auto same_python =
      model.CreateOrActivate(gui::ConsoleSessionKind::PythonRepl, project_a);
  Check(same_python && !same_python.created &&
            same_python.session_id == python.session_id,
        "Python REPL should be reused within one project");
  const auto wrong_project =
      model.CreateOrActivate(gui::ConsoleSessionKind::PythonRepl, project_b);
  Check(!wrong_project && wrong_project.error ==
                              gui::ConsoleSessionCreateError::ProjectMismatch,
        "singleton state must not leak across project roots");
}

void TestMultipleShellsAndUnreadState() {
  gui::ConsoleSessionModel model;
  const std::string project_root = "D:/projects/Project A";
  const auto cmd1 = model.CreateOrActivate(
      gui::ConsoleSessionKind::CommandPrompt, project_root);
  const auto cmd2 = model.CreateOrActivate(
      gui::ConsoleSessionKind::CommandPrompt, project_root);
  const auto ps1 =
      model.CreateOrActivate(gui::ConsoleSessionKind::PowerShell, project_root);
  const auto ps2 =
      model.CreateOrActivate(gui::ConsoleSessionKind::PowerShell, project_root);
  const auto bash1 =
      model.CreateOrActivate(gui::ConsoleSessionKind::GitBash, project_root);
  const auto bash2 =
      model.CreateOrActivate(gui::ConsoleSessionKind::GitBash, project_root);
  const auto shell1 = model.CreateOrActivate(
      gui::ConsoleSessionKind::SystemShell, project_root);
  const auto shell2 = model.CreateOrActivate(
      gui::ConsoleSessionKind::SystemShell, project_root);
  Check(cmd1 && cmd2 && ps1 && ps2 && bash1 && bash2 && shell1 && shell2 &&
            cmd1.session_id != cmd2.session_id &&
            ps1.session_id != ps2.session_id &&
            bash1.session_id != bash2.session_id &&
            shell1.session_id != shell2.session_id,
        "shell profiles should create independent identities");
  Check(model.Find(*cmd1.session_id)->title == "Command Prompt 1" &&
            model.Find(*cmd2.session_id)->title == "Command Prompt 2" &&
            model.Find(*ps1.session_id)->title == "PowerShell 1" &&
            model.Find(*ps2.session_id)->title == "PowerShell 2" &&
            model.Find(*bash1.session_id)->title == "Git Bash 1" &&
            model.Find(*bash2.session_id)->title == "Git Bash 2" &&
            model.Find(*shell1.session_id)->title == "Shell 1" &&
            model.Find(*shell2.session_id)->title == "Shell 2",
        "shell titles should be stable and numbered by profile");

  Check(model.SetUnread(*cmd1.session_id),
        "existing session should accept unread state");
  Check(model.Find(*cmd1.session_id)->unread,
        "unread state should be observable");
  Check(model.Activate(*cmd1.session_id) &&
            !model.Find(*cmd1.session_id)->unread,
        "activating a session should clear its unread state");
}

void TestProjectClosePreservesGlobalSessions() {
  gui::ConsoleSessionModel model;
  const std::string project_root = "D:/projects/Project A";
  const auto commands =
      model.CreateOrActivate(gui::ConsoleSessionKind::Commands);
  const auto python =
      model.CreateOrActivate(gui::ConsoleSessionKind::PythonRepl, project_root);
  const auto agent =
      model.CreateOrActivate(gui::ConsoleSessionKind::AgentLlm, project_root);
  const auto shell =
      model.CreateOrActivate(gui::ConsoleSessionKind::PowerShell, project_root);
  Check(commands && python && agent && shell,
        "fixture sessions should be created");

  Check(model.CloseProjectScopedSessions(project_root) == 3,
        "project close should remove every matching scoped session");
  Check(model.Sessions().size() == 2,
        "project close should retain only Logs and Commands");
  for (const auto &session : model.Sessions()) {
    Check(session.kind == gui::ConsoleSessionKind::Logs ||
              session.kind == gui::ConsoleSessionKind::Commands,
          "project close should preserve only global session kinds");
  }
  Check(model.ActiveSession() != nullptr,
        "project close should select a deterministic fallback session");
}

} // namespace

int main() {
  TestProfiles();
  TestImplementedSessionAvailability();
  TestSessionActivationDefersInputFocus();
  TestProgrammaticNavigationKeepsNewSessionActive();
  TestBundledTerminalFontCoversCliGlyphs();
  TestAgentRequestMapping();
#ifdef _WIN32
  TestPowerShellHasInteractiveTerminalSemantics();
  TestLocalShellProcessesUseProjectDirectory();
#else
  TestSystemShellHasInteractiveTerminalSemantics();
#endif
  TestTerminalBufferInterpretsAnsiState();
  TestEnsureDoesNotStealFocus();
  TestInitialAndSingletonState();
  TestProjectRequirementsAndIsolation();
  TestMultipleShellsAndUnreadState();
  TestProjectClosePreservesGlobalSessions();
  std::cout << "Console session model tests passed\n";
  return 0;
}
