#include "local_shell_process.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <csignal>
#ifdef __APPLE__
#include <util.h>
#else
#include <pty.h>
#endif
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace cyxwiz {

namespace {

constexpr std::uint32_t kNoExitCode = std::numeric_limits<std::uint32_t>::max();
constexpr std::size_t kMaxPendingOutputBytes = 1024 * 1024;

#ifdef _WIN32

void CloseHandleIfOpen(HANDLE &handle) {
  if (!handle || handle == INVALID_HANDLE_VALUE)
    return;
  CloseHandle(handle);
  handle = nullptr;
}

std::string WindowsErrorMessage(DWORD error_code) {
  LPSTR message = nullptr;
  const DWORD size = FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr, error_code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      reinterpret_cast<LPSTR>(&message), 0, nullptr);
  std::string result = size && message
                           ? std::string(message, size)
                           : "Windows error " + std::to_string(error_code);
  if (message)
    LocalFree(message);
  while (!result.empty() && (result.back() == '\r' || result.back() == '\n' ||
                             result.back() == ' ')) {
    result.pop_back();
  }
  return result;
}

std::filesystem::path SystemExecutable(LocalShellKind kind) {
  if (kind == LocalShellKind::GitBash) {
    std::array<wchar_t, 32768> git_path{};
    const DWORD git_length = SearchPathW(nullptr, L"git.exe", nullptr,
                                         static_cast<DWORD>(git_path.size()),
                                         git_path.data(), nullptr);
    if (git_length > 0 && git_length < git_path.size()) {
      auto ancestor = std::filesystem::path(git_path.data()).parent_path();
      while (!ancestor.empty()) {
        for (const auto &relative :
             {std::filesystem::path("bin/bash.exe"),
              std::filesystem::path("usr/bin/bash.exe")}) {
          const auto candidate = ancestor / relative;
          if (std::filesystem::is_regular_file(candidate))
            return candidate;
        }
        const auto parent = ancestor.parent_path();
        if (parent == ancestor)
          break;
        ancestor = parent;
      }
    }

    for (const auto &candidate :
         {std::filesystem::path("C:/Program Files/Git/bin/bash.exe"),
          std::filesystem::path("C:/Program Files (x86)/Git/bin/bash.exe")}) {
      if (std::filesystem::is_regular_file(candidate))
        return candidate;
    }
    return {};
  }

  std::array<wchar_t, MAX_PATH> system_directory{};
  const UINT length = GetSystemDirectoryW(
      system_directory.data(), static_cast<UINT>(system_directory.size()));
  if (length == 0 || length >= system_directory.size())
    return {};

  std::filesystem::path executable(system_directory.data());
  if (kind == LocalShellKind::CommandPrompt) {
    return executable / "cmd.exe";
  }
  return executable / "WindowsPowerShell" / "v1.0" / "powershell.exe";
}

std::wstring CommandLine(LocalShellKind kind,
                         const std::filesystem::path &executable) {
  std::wstring command = L"\"" + executable.wstring() + L"\"";
  if (kind == LocalShellKind::CommandPrompt) {
    command += L" /D /Q";
  } else if (kind == LocalShellKind::GitBash) {
    command += L" --login -i";
  } else {
    command += L" -NoLogo -NoProfile -NoExit";
  }
  return command;
}

#else

void CloseFileDescriptor(int &descriptor) {
  if (descriptor < 0)
    return;
  close(descriptor);
  descriptor = -1;
}

std::string PosixErrorMessage(int error_code) {
  return std::strerror(error_code);
}

std::filesystem::path SystemShellExecutable() {
  if (const char *configured_shell = std::getenv("SHELL")) {
    const std::filesystem::path candidate(configured_shell);
    if (candidate.is_absolute() && access(candidate.c_str(), X_OK) == 0)
      return candidate;
  }
  for (const auto &candidate :
       {std::filesystem::path("/bin/zsh"), std::filesystem::path("/bin/bash"),
        std::filesystem::path("/bin/sh")}) {
    if (access(candidate.c_str(), X_OK) == 0)
      return candidate;
  }
  return {};
}

#endif

} // namespace

struct LocalShellProcess::ProcessState {
  std::atomic<bool> running{false};
  std::atomic<std::uint32_t> exit_code{kNoExitCode};
  std::mutex output_mutex;
  std::mutex input_mutex;
  std::string pending_output;
  std::thread reader;

#ifdef _WIN32
  HPCON pseudo_console = nullptr;
  HANDLE process = nullptr;
  HANDLE process_thread = nullptr;
  HANDLE stdout_read = nullptr;
  HANDLE stdin_write = nullptr;
#else
  int pty_master = -1;
  pid_t child_pid = -1;
#endif
};

LocalShellProcess::LocalShellProcess(LocalShellKind kind)
    : kind_(kind), state_(std::make_unique<ProcessState>()) {}

LocalShellProcess::~LocalShellProcess() { Stop(); }

std::string_view LocalShellProcess::DisplayName(LocalShellKind kind) {
  switch (kind) {
  case LocalShellKind::CommandPrompt:
    return "Command Prompt";
  case LocalShellKind::GitBash:
    return "Git Bash";
  case LocalShellKind::SystemShell:
    return "Shell";
  case LocalShellKind::PowerShell:
  default:
    return "PowerShell";
  }
}

bool LocalShellProcess::Start(const std::filesystem::path &project_root,
                              std::string &error, std::uint16_t columns,
                              std::uint16_t rows) {
  Stop();
  error.clear();

  std::error_code filesystem_error;
  if (!std::filesystem::is_directory(project_root, filesystem_error)) {
    error = "Project directory does not exist: " + project_root.string();
    return false;
  }

#ifdef _WIN32
  const auto executable = SystemExecutable(kind_);
  if (executable.empty() || !std::filesystem::is_regular_file(executable)) {
    error = std::string(DisplayName(kind_)) + " executable was not found.";
    return false;
  }

  HANDLE stdout_read = nullptr;
  HANDLE stdout_write = nullptr;
  HANDLE stdin_read = nullptr;
  HANDLE stdin_write = nullptr;
  HPCON pseudo_console = nullptr;
  PPROC_THREAD_ATTRIBUTE_LIST attributes = nullptr;

  const auto close_temporary_handles = [&]() {
    CloseHandleIfOpen(stdout_read);
    CloseHandleIfOpen(stdout_write);
    CloseHandleIfOpen(stdin_read);
    CloseHandleIfOpen(stdin_write);
    if (attributes) {
      DeleteProcThreadAttributeList(attributes);
      HeapFree(GetProcessHeap(), 0, attributes);
      attributes = nullptr;
    }
    if (pseudo_console) {
      ClosePseudoConsole(pseudo_console);
      pseudo_console = nullptr;
    }
  };

  if (!CreatePipe(&stdout_read, &stdout_write, nullptr, 0) ||
      !CreatePipe(&stdin_read, &stdin_write, nullptr, 0)) {
    error =
        "Failed to create shell pipes: " + WindowsErrorMessage(GetLastError());
    close_temporary_handles();
    return false;
  }

  const COORD size{static_cast<SHORT>(std::max<std::uint16_t>(columns, 2)),
                   static_cast<SHORT>(std::max<std::uint16_t>(rows, 1))};
  HRESULT result =
      CreatePseudoConsole(size, stdin_read, stdout_write, 0, &pseudo_console);
  if (FAILED(result)) {
    error = "Failed to create Windows pseudo-terminal (ConPTY): " +
            WindowsErrorMessage(HRESULT_CODE(result));
    close_temporary_handles();
    return false;
  }
  // ConPTY duplicates its pipe ends. Releasing the host copies here matches
  // the Windows Terminal sample and lets broken-pipe detection work reliably.
  CloseHandleIfOpen(stdout_write);
  CloseHandleIfOpen(stdin_read);

  SIZE_T attribute_bytes = 0;
  InitializeProcThreadAttributeList(nullptr, 1, 0, &attribute_bytes);
  attributes = static_cast<PPROC_THREAD_ATTRIBUTE_LIST>(
      HeapAlloc(GetProcessHeap(), 0, attribute_bytes));
  if (!attributes ||
      !InitializeProcThreadAttributeList(attributes, 1, 0, &attribute_bytes) ||
      !UpdateProcThreadAttribute(
          attributes, 0, PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE, pseudo_console,
          sizeof(pseudo_console), nullptr, nullptr)) {
    error = "Failed to configure ConPTY process attributes: " +
            WindowsErrorMessage(GetLastError());
    close_temporary_handles();
    return false;
  }

  STARTUPINFOEXW startup{};
  startup.StartupInfo.cb = sizeof(startup);
  startup.lpAttributeList = attributes;

  PROCESS_INFORMATION process{};
  auto command_line = CommandLine(kind_, executable);
  std::vector<wchar_t> mutable_command(command_line.begin(),
                                       command_line.end());
  mutable_command.push_back(L'\0');
  const auto working_directory = project_root.wstring();

  const BOOL created = CreateProcessW(
      nullptr, mutable_command.data(), nullptr, nullptr, FALSE,
      EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT, nullptr,
      working_directory.c_str(), &startup.StartupInfo, &process);

  DeleteProcThreadAttributeList(attributes);
  HeapFree(GetProcessHeap(), 0, attributes);
  attributes = nullptr;
  if (!created) {
    error = "Failed to start " + std::string(DisplayName(kind_)) + ": " +
            WindowsErrorMessage(GetLastError());
    close_temporary_handles();
    return false;
  }

  state_->pseudo_console = pseudo_console;
  pseudo_console = nullptr;
  state_->process = process.hProcess;
  state_->process_thread = process.hThread;
  state_->stdout_read = stdout_read;
  state_->stdin_write = stdin_write;
  state_->exit_code.store(kNoExitCode, std::memory_order_release);
  state_->running.store(true, std::memory_order_release);
  state_->reader = std::thread([this]() { ReaderLoop(); });
  return true;
#else
  if (kind_ != LocalShellKind::SystemShell) {
    error =
        std::string(DisplayName(kind_)) + " is not available on this platform.";
    return false;
  }
  const auto executable = SystemShellExecutable();
  if (executable.empty()) {
    error = "No executable login shell was found.";
    return false;
  }

  winsize terminal_size{};
  terminal_size.ws_col = std::max<std::uint16_t>(columns, 2);
  terminal_size.ws_row = std::max<std::uint16_t>(rows, 1);
  int pty_master = -1;
  const auto executable_string = executable.string();
  const auto executable_name = executable.filename().string();
  const pid_t child_pid =
      forkpty(&pty_master, nullptr, nullptr, &terminal_size);
  if (child_pid < 0) {
    error =
        "Failed to create POSIX pseudo-terminal: " + PosixErrorMessage(errno);
    CloseFileDescriptor(pty_master);
    return false;
  }
  if (child_pid == 0) {
    if (chdir(project_root.c_str()) != 0)
      _exit(126);
    setenv("TERM", "xterm-256color", 1);
    setenv("COLORTERM", "truecolor", 1);
    setenv("TERM_PROGRAM", "CyxWiz", 1);
    execl(executable_string.c_str(), executable_name.c_str(), "-l",
          static_cast<char *>(nullptr));
    _exit(127);
  }

  state_->pty_master = pty_master;
  state_->child_pid = child_pid;
  state_->exit_code.store(kNoExitCode, std::memory_order_release);
  state_->running.store(true, std::memory_order_release);
  state_->reader = std::thread([this]() { ReaderLoop(); });
  return true;
#endif
}

void LocalShellProcess::Stop() {
#ifdef _WIN32
  if (state_->process) {
    if (state_->running.load(std::memory_order_acquire) &&
        state_->stdin_write) {
      std::string ignored_error;
      (void)Send("exit", ignored_error);
    }
    CloseHandleIfOpen(state_->stdin_write);

    DWORD wait_result = WaitForSingleObject(state_->process, 750);
    if (wait_result == WAIT_TIMEOUT) {
      TerminateProcess(state_->process, 1);
      WaitForSingleObject(state_->process, 2000);
    }
  }

  state_->running.store(false, std::memory_order_release);
  if (state_->pseudo_console) {
    ClosePseudoConsole(state_->pseudo_console);
    state_->pseudo_console = nullptr;
  }
  if (state_->reader.joinable()) {
    CancelSynchronousIo(
        reinterpret_cast<HANDLE>(state_->reader.native_handle()));
    state_->reader.join();
  }
  CloseHandleIfOpen(state_->stdout_read);
  CloseHandleIfOpen(state_->process_thread);
  CloseHandleIfOpen(state_->process);
#else
  const pid_t child_pid = state_->child_pid;
  if (child_pid > 0 && state_->running.load(std::memory_order_acquire)) {
    if (state_->pty_master >= 0) {
      std::string ignored_error;
      (void)Write("exit\r", ignored_error);
    }
    for (int attempt = 0;
         attempt < 20 && state_->running.load(std::memory_order_acquire);
         ++attempt) {
      std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    if (state_->running.load(std::memory_order_acquire))
      kill(-child_pid, SIGHUP);
    for (int attempt = 0;
         attempt < 20 && state_->running.load(std::memory_order_acquire);
         ++attempt) {
      std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    if (state_->running.load(std::memory_order_acquire))
      kill(-child_pid, SIGKILL);
  }
  if (state_->reader.joinable())
    state_->reader.join();
  CloseFileDescriptor(state_->pty_master);
  state_->child_pid = -1;
  state_->running.store(false, std::memory_order_release);
#endif
}

bool LocalShellProcess::Send(std::string_view command, std::string &error) {
  std::string line(command);
#ifdef _WIN32
  line += "\r\n";
#else
  line += "\r";
#endif
  return Write(line, error);
}

bool LocalShellProcess::Write(std::string_view input, std::string &error) {
  error.clear();
  if (input.empty())
    return true;
  if (!IsRunning()) {
    error = std::string(DisplayName(kind_)) + " is not running.";
    return false;
  }

#ifdef _WIN32
  std::lock_guard lock(state_->input_mutex);
  std::size_t written_total = 0;
  while (written_total < input.size()) {
    DWORD written = 0;
    const BOOL ok = WriteFile(state_->stdin_write, input.data() + written_total,
                              static_cast<DWORD>(input.size() - written_total),
                              &written, nullptr);
    if (!ok || written == 0) {
      error = "Failed to write to " + std::string(DisplayName(kind_)) + ": " +
              WindowsErrorMessage(GetLastError());
      return false;
    }
    written_total += written;
  }
  return true;
#else
  std::lock_guard lock(state_->input_mutex);
  std::size_t written_total = 0;
  while (written_total < input.size()) {
    const ssize_t written =
        write(state_->pty_master, input.data() + written_total,
              input.size() - written_total);
    if (written < 0 && errno == EINTR)
      continue;
    if (written <= 0) {
      error = "Failed to write to " + std::string(DisplayName(kind_)) + ": " +
              PosixErrorMessage(errno);
      return false;
    }
    written_total += static_cast<std::size_t>(written);
  }
  return true;
#endif
}

bool LocalShellProcess::Resize(std::uint16_t columns, std::uint16_t rows,
                               std::string &error) {
  error.clear();
#ifdef _WIN32
  if (!state_->pseudo_console) {
    error = std::string(DisplayName(kind_)) + " is not running.";
    return false;
  }
  const COORD size{static_cast<SHORT>(std::max<std::uint16_t>(columns, 2)),
                   static_cast<SHORT>(std::max<std::uint16_t>(rows, 1))};
  const HRESULT result = ResizePseudoConsole(state_->pseudo_console, size);
  if (FAILED(result)) {
    error =
        "Failed to resize ConPTY: " + WindowsErrorMessage(HRESULT_CODE(result));
    return false;
  }
  return true;
#else
  if (state_->pty_master < 0) {
    error = std::string(DisplayName(kind_)) + " is not running.";
    return false;
  }
  winsize terminal_size{};
  terminal_size.ws_col = std::max<std::uint16_t>(columns, 2);
  terminal_size.ws_row = std::max<std::uint16_t>(rows, 1);
  if (ioctl(state_->pty_master, TIOCSWINSZ, &terminal_size) != 0) {
    error =
        "Failed to resize POSIX pseudo-terminal: " + PosixErrorMessage(errno);
    return false;
  }
  return true;
#endif
}

bool LocalShellProcess::IsRunning() const {
  return state_->running.load(std::memory_order_acquire);
}

std::optional<std::uint32_t> LocalShellProcess::ExitCode() const {
  const auto exit_code = state_->exit_code.load(std::memory_order_acquire);
  return exit_code == kNoExitCode ? std::nullopt
                                  : std::optional<std::uint32_t>(exit_code);
}

std::string LocalShellProcess::DrainOutput() {
  std::lock_guard lock(state_->output_mutex);
  std::string output = std::move(state_->pending_output);
  state_->pending_output.clear();
  return output;
}

void LocalShellProcess::ReaderLoop() {
#ifdef _WIN32
  std::array<char, 4096> buffer{};
  for (;;) {
    DWORD bytes_read = 0;
    if (!ReadFile(state_->stdout_read, buffer.data(),
                  static_cast<DWORD>(buffer.size()), &bytes_read, nullptr) ||
        bytes_read == 0) {
      break;
    }

    AppendPending(std::string_view(buffer.data(), bytes_read));
  }

  DWORD exit_code = 0;
  if (state_->process) {
    WaitForSingleObject(state_->process, INFINITE);
    if (!GetExitCodeProcess(state_->process, &exit_code))
      exit_code = 1;
  }
  state_->exit_code.store(exit_code, std::memory_order_release);
  state_->running.store(false, std::memory_order_release);
#else
  std::array<char, 4096> buffer{};
  for (;;) {
    const ssize_t bytes_read =
        read(state_->pty_master, buffer.data(), buffer.size());
    if (bytes_read > 0) {
      AppendPending(std::string_view(buffer.data(),
                                     static_cast<std::size_t>(bytes_read)));
      continue;
    }
    if (bytes_read < 0 && errno == EINTR)
      continue;
    break;
  }

  int process_status = 0;
  pid_t waited = -1;
  do {
    waited = waitpid(state_->child_pid, &process_status, 0);
  } while (waited < 0 && errno == EINTR);
  std::uint32_t exit_code = 1;
  if (waited > 0) {
    if (WIFEXITED(process_status))
      exit_code = static_cast<std::uint32_t>(WEXITSTATUS(process_status));
    else if (WIFSIGNALED(process_status))
      exit_code = static_cast<std::uint32_t>(128 + WTERMSIG(process_status));
  }
  state_->exit_code.store(exit_code, std::memory_order_release);
  state_->running.store(false, std::memory_order_release);
#endif
}

void LocalShellProcess::AppendPending(std::string_view text) {
  if (text.empty())
    return;
  std::lock_guard lock(state_->output_mutex);
  state_->pending_output.append(text);
  if (state_->pending_output.size() > kMaxPendingOutputBytes) {
    const auto excess = state_->pending_output.size() - kMaxPendingOutputBytes;
    const auto line_boundary = state_->pending_output.find('\n', excess);
    state_->pending_output.erase(
        0, line_boundary == std::string::npos ? excess : line_boundary + 1);
  }
}

} // namespace cyxwiz
