#include "installer_progress_channel.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <fstream>
#include <random>
#include <system_error>

namespace cyxwiz::installer {
namespace {

constexpr std::uintmax_t kMaximumProgressBytes = 16 * 1024;

} // namespace

std::string CreateInstallerProgressToken() {
  const auto time = static_cast<std::uint64_t>(
      std::chrono::steady_clock::now().time_since_epoch().count());
  std::random_device random;
  const auto entropy = (static_cast<std::uint64_t>(random()) << 32U) |
                       static_cast<std::uint64_t>(random());
  constexpr char digits[] = "0123456789abcdef";
  std::string token(32, '0');
  for (std::size_t index = 0; index < 16; ++index) {
    token[index] = digits[(time >> ((15 - index) * 4U)) & 0x0fU];
    token[index + 16] = digits[(entropy >> ((15 - index) * 4U)) & 0x0fU];
  }
  return token;
}

bool IsInstallerProgressToken(const std::string &token) {
  if (token.size() != 32)
    return false;
  for (const char value : token) {
    if (!((value >= '0' && value <= '9') ||
          (value >= 'a' && value <= 'f'))) {
      return false;
    }
  }
  return true;
}

std::filesystem::path InstallerProgressPath(
    const std::filesystem::path &runtime_root, const std::string &token) {
  if (!runtime_root.is_absolute() || !IsInstallerProgressToken(token))
    return {};
  return runtime_root / "staging" / "progress" / (token + ".json");
}

bool PublishInstallerProgress(const std::filesystem::path &path,
                              const InstallerHelperProgress &progress) {
  if (path.empty() || !path.is_absolute() || progress.message.size() > 4096)
    return false;
  std::error_code error;
  std::filesystem::create_directories(path.parent_path(), error);
  if (error)
    return false;
  const auto temporary = path.string() + ".next";
  nlohmann::json value = {
      {"schema_version", 1},
      {"stage", progress.stage},
      {"completed_bytes", progress.completed_bytes},
      {"total_bytes", progress.total_bytes},
      {"component_index", progress.component_index},
      {"component_count", progress.component_count},
      {"message", progress.message},
  };
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output)
      return false;
    output << value.dump();
    if (!output)
      return false;
  }
#ifdef _WIN32
  std::filesystem::remove(path, error);
  error.clear();
#endif
  std::filesystem::rename(temporary, path, error);
  if (error) {
    std::filesystem::remove(temporary, error);
    return false;
  }
  return true;
}

std::optional<InstallerHelperProgress>
ReadInstallerProgress(const std::filesystem::path &path) {
  std::error_code error;
  if (!std::filesystem::is_regular_file(path, error) || error ||
      std::filesystem::file_size(path, error) > kMaximumProgressBytes || error) {
    return std::nullopt;
  }
  try {
    std::ifstream input(path, std::ios::binary);
    const auto value = nlohmann::json::parse(input);
    if (value.at("schema_version").get<int>() != 1)
      return std::nullopt;
    InstallerHelperProgress progress;
    progress.stage = value.at("stage").get<std::string>();
    progress.completed_bytes =
        value.at("completed_bytes").get<std::uint64_t>();
    progress.total_bytes = value.at("total_bytes").get<std::uint64_t>();
    progress.component_index =
        value.at("component_index").get<std::size_t>();
    progress.component_count =
        value.at("component_count").get<std::size_t>();
    progress.message = value.at("message").get<std::string>();
    if (progress.stage.size() > 64 || progress.message.size() > 4096)
      return std::nullopt;
    return progress;
  } catch (...) {
    return std::nullopt;
  }
}

} // namespace cyxwiz::installer
