#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace cyxwiz {

enum class LocalShellKind : std::uint8_t {
  CommandPrompt,
  PowerShell,
  GitBash,
  SystemShell,
};

class LocalShellProcess {
public:
  explicit LocalShellProcess(LocalShellKind kind);
  ~LocalShellProcess();

  LocalShellProcess(const LocalShellProcess &) = delete;
  LocalShellProcess &operator=(const LocalShellProcess &) = delete;

  bool Start(const std::filesystem::path &project_root, std::string &error,
             std::uint16_t columns = 80, std::uint16_t rows = 24);
  void Stop();
  bool Write(std::string_view input, std::string &error);
  bool Send(std::string_view command, std::string &error);
  bool Resize(std::uint16_t columns, std::uint16_t rows, std::string &error);

  bool IsRunning() const;
  std::optional<std::uint32_t> ExitCode() const;
  std::string DrainOutput();

  LocalShellKind Kind() const { return kind_; }
  static std::string_view DisplayName(LocalShellKind kind);

private:
  struct ProcessState;

  void ReaderLoop();
  void AppendPending(std::string_view text);

  LocalShellKind kind_;
  std::unique_ptr<ProcessState> state_;
};

} // namespace cyxwiz
