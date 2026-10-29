#include "child_process.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;
#endif

namespace cyxwiz::core {

#ifdef _WIN32

namespace {

std::wstring Widen(const std::string& text) {
    if (text.empty()) return {};
    const int size = MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
    std::wstring wide(static_cast<size_t>(size), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), wide.data(), size);
    return wide;
}

// One argument as CommandLineToArgvW reads it back: quoted when it has a
// space, tab or quote, with the backslashes before a quote doubled.
std::wstring QuoteArgument(const std::wstring& arg) {
    if (!arg.empty() && arg.find_first_of(L" \t\"") == std::wstring::npos) return arg;
    std::wstring quoted = L"\"";
    size_t backslashes = 0;
    for (wchar_t c : arg) {
        if (c == L'\\') {
            ++backslashes;
            continue;
        }
        if (c == L'"') quoted.append(backslashes * 2 + 1, L'\\');
        else quoted.append(backslashes, L'\\');
        backslashes = 0;
        quoted += c;
    }
    quoted.append(backslashes * 2, L'\\');
    quoted += L'"';
    return quoted;
}

}  // namespace

int RunHidden(const std::vector<std::string>& args, std::string* output) {
    if (args.empty()) return -1;
    std::wstring command;
    for (const auto& arg : args) command += (command.empty() ? L"" : L" ") + QuoteArgument(Widen(arg));

    SECURITY_ATTRIBUTES inherit{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE null_device = CreateFileW(L"NUL", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                     &inherit, OPEN_EXISTING, 0, nullptr);
    HANDLE read_pipe = nullptr, write_pipe = nullptr;
    if (output && (!CreatePipe(&read_pipe, &write_pipe, &inherit, 0) ||
                   !SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0))) {
        if (read_pipe) CloseHandle(read_pipe);
        if (write_pipe) CloseHandle(write_pipe);
        if (null_device != INVALID_HANDLE_VALUE) CloseHandle(null_device);
        return -1;
    }

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = null_device;
    startup.hStdOutput = output ? write_pipe : null_device;
    startup.hStdError = null_device;
    PROCESS_INFORMATION process{};
    const BOOL started = CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr,
                                        nullptr, &startup, &process);
    if (write_pipe) CloseHandle(write_pipe);
    if (null_device != INVALID_HANDLE_VALUE) CloseHandle(null_device);
    if (!started) {
        if (read_pipe) CloseHandle(read_pipe);
        return -1;
    }

    if (output) {
        output->clear();
        char buffer[4096];
        DWORD read = 0;
        while (ReadFile(read_pipe, buffer, sizeof(buffer), &read, nullptr) && read > 0) output->append(buffer, read);
        CloseHandle(read_pipe);
    }
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD exit_code = 1;
    GetExitCodeProcess(process.hProcess, &exit_code);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return static_cast<int>(exit_code);
}

#else

int RunHidden(const std::vector<std::string>& args, std::string* output) {
    if (args.empty()) return -1;
    int pipe_fds[2] = {-1, -1};
    if (output && pipe(pipe_fds) != 0) return -1;

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    if (output) {
        posix_spawn_file_actions_adddup2(&actions, pipe_fds[1], STDOUT_FILENO);
        posix_spawn_file_actions_addclose(&actions, pipe_fds[0]);
        posix_spawn_file_actions_addclose(&actions, pipe_fds[1]);
    } else {
        posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
    }
    posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);

    std::vector<char*> argv;
    for (const auto& arg : args) argv.push_back(const_cast<char*>(arg.c_str()));
    argv.push_back(nullptr);
    pid_t pid = 0;
    const int spawned = posix_spawnp(&pid, argv[0], &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    if (output) close(pipe_fds[1]);
    if (spawned != 0) {
        if (output) close(pipe_fds[0]);
        return -1;
    }

    if (output) {
        output->clear();
        char buffer[4096];
        for (ssize_t n; (n = read(pipe_fds[0], buffer, sizeof(buffer))) != 0;) {
            if (n < 0) {
                if (errno == EINTR) continue;
                break;
            }
            output->append(buffer, static_cast<size_t>(n));
        }
        close(pipe_fds[0]);
    }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

#endif

}  // namespace cyxwiz::core
