#include "backend_pack_platform.h"
#include "installer_external_session.h"

#include <array>
#include <cerrno>
#include <cstring>
#include <set>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
// clang-format off
#include <windows.h>
#include <psapi.h>
// clang-format on
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#include <unistd.h>
#else
#include <link.h>
#include <unistd.h>
#endif

namespace cyxwiz::installer {

std::vector<std::filesystem::path> LoadedInstallerModules(std::string &error) {
  error.clear();
  std::vector<std::filesystem::path> paths;
#ifdef _WIN32
  std::array<HMODULE, 1024> modules{};
  DWORD needed = 0;
  if (!::EnumProcessModules(::GetCurrentProcess(), modules.data(),
                            static_cast<DWORD>(sizeof(modules)), &needed) ||
      needed > sizeof(modules)) {
    error = "Cannot inspect the manager's loaded dependency closure";
    return {};
  }
  for (std::size_t i = 0; i < needed / sizeof(HMODULE); ++i) {
    std::array<wchar_t, 32768> path{};
    const auto length = ::GetModuleFileNameW(modules[i], path.data(),
                                             static_cast<DWORD>(path.size()));
    if (!length || length >= path.size()) {
      error = "Cannot resolve an exact loaded manager dependency";
      return {};
    }
    paths.emplace_back(std::wstring(path.data(), length));
  }
#elif defined(__APPLE__)
  const auto count = ::_dyld_image_count();
  for (std::uint32_t i = 0; i < count; ++i) {
    const char *path = ::_dyld_get_image_name(i);
    if (path && *path)
      paths.emplace_back(path);
  }
#else
  ::dl_iterate_phdr(
      [](dl_phdr_info *info, std::size_t, void *context) {
        if (info->dlpi_name && *info->dlpi_name && info->dlpi_name[0] == '/')
          static_cast<std::vector<std::filesystem::path> *>(context)
              ->emplace_back(info->dlpi_name);
        return 0;
      },
      &paths);
#endif
  return paths;
}

bool RelaunchExternalInstallerSession(const std::filesystem::path &directory,
                                      const std::vector<std::string> &arguments,
                                      std::string &error) {
  const auto executable =
      directory / runtime::CurrentInstallerManagerExecutableName();
#ifdef _WIN32
  const auto quote = [](const std::wstring &value) {
    std::wstring output = L"\"";
    std::size_t slashes = 0;
    for (const auto character : value) {
      if (character == L'\\') {
        ++slashes;
        continue;
      }
      output.append(character == L'"' ? slashes * 2 + 1 : slashes, L'\\');
      slashes = 0;
      output += character;
    }
    output.append(slashes * 2, L'\\');
    return output + L'"';
  };
  std::wstring command = quote(executable.native());
  for (std::size_t i = 1; i < arguments.size(); ++i) {
    const std::u8string utf8(arguments[i].begin(), arguments[i].end());
    const auto value = std::filesystem::path(utf8).wstring();
    command += L" " + quote(value);
  }
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  PROCESS_INFORMATION process{};
  if (!::CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr,
                        FALSE, 0, nullptr, directory.c_str(), &startup,
                        &process)) {
    error = "Cannot open the external installer; Win32 error " +
            std::to_string(::GetLastError());
    return false;
  }
  ::CloseHandle(process.hThread);
  ::CloseHandle(process.hProcess);
  return true;
#else
  // Replace this process, rather than leaving a detached child or a waiter that
  // still has product libraries mapped. Keep only session-local loader paths.
  std::set<std::string> directories{directory.string()};
  for (const auto &entry :
       std::filesystem::recursive_directory_iterator(directory)) {
    if (entry.is_regular_file())
      directories.insert(entry.path().parent_path().string());
  }
  std::string library_path;
  for (const auto &path : directories) {
    if (!library_path.empty())
      library_path += ':';
    library_path += path;
  }
#ifdef __APPLE__
  const char *variable = "DYLD_LIBRARY_PATH";
#else
  const char *variable = "LD_LIBRARY_PATH";
#endif
  if (::setenv(variable, library_path.c_str(), 1) != 0 ||
      ::chdir(directory.c_str()) != 0) {
    error = "Cannot configure the external installer: " +
            std::string(std::strerror(errno));
    return false;
  }
  auto forwarded = arguments;
  forwarded[0] = executable.string();
  std::vector<char *> values;
  for (auto &value : forwarded)
    values.push_back(value.data());
  values.push_back(nullptr);
  ::execv(executable.c_str(), values.data());
  error = "Cannot open the external installer: " +
          std::string(std::strerror(errno));
  return false;
#endif
}

} // namespace cyxwiz::installer
