#include "installer_external_session.h"

#include "backend_pack_path.h"
#include "backend_pack_platform.h"

#include <archive.h>

#include <chrono>
#include <set>
#include <stdexcept>

namespace cyxwiz::installer {

bool InstallerPathWithin(const std::filesystem::path &path,
                         const std::filesystem::path &directory) {
  std::error_code error;
  const auto canonical_path = std::filesystem::weakly_canonical(path, error);
  if (error || !canonical_path.is_absolute())
    return true; // Fail closed.
  const auto canonical_directory =
      std::filesystem::weakly_canonical(directory, error);
  if (error || !canonical_directory.is_absolute())
    return true;
  auto parent = canonical_directory.begin();
  auto child = canonical_path.begin();
  for (; parent != canonical_directory.end(); ++parent, ++child) {
    if (child == canonical_path.end())
      return false;
#ifdef _WIN32
    if (runtime::FoldBackendPackPath(parent->string()) !=
        runtime::FoldBackendPackPath(child->string()))
      return false;
#else
    if (*parent != *child)
      return false;
#endif
  }
  return true;
}

bool StageExternalInstallerSession(
    const std::filesystem::path &executable_directory,
    const std::filesystem::path &metadata_root,
    const std::vector<std::filesystem::path> &loaded_modules,
    const std::filesystem::path &destination, std::string &error) {
  try {
    if (!destination.is_absolute() ||
        InstallerPathWithin(destination, executable_directory) ||
        !std::filesystem::create_directory(destination)) {
      error = "Cannot create an exclusive external installer session";
      return false;
    }
#ifndef _WIN32
    std::filesystem::permissions(destination,
                                 std::filesystem::perms::owner_all);
#endif
    std::uintmax_t bytes = 0;
    std::set<std::string> copied;
    const auto copy = [&](const std::filesystem::path &source,
                          const std::filesystem::path &relative) {
      if (!runtime::IsCanonicalBackendPackRelativePath(
              relative.generic_string()))
        throw std::runtime_error("Unsafe installer session member");
#ifdef _WIN32
      // The loader reports import-table casing (VCRUNTIME140.dll) while the
      // canonical relative path has on-disk casing; both name one NTFS file.
      if (!copied.insert(runtime::FoldBackendPackPath(relative.generic_string()))
               .second)
        return;
#else
      if (!copied.insert(relative.generic_string()).second)
        return;
#endif
      const auto exact = std::filesystem::canonical(source);
      if (!std::filesystem::is_regular_file(exact))
        throw std::runtime_error(
            "Installer session member is not a regular file");
      const auto size = std::filesystem::file_size(exact);
      if (size > 64 * 1024 * 1024 || copied.size() > 256 ||
          bytes + size > 256 * 1024 * 1024)
        throw std::runtime_error(
            "Installer session exceeds its bounded file budget");
      bytes += size;
      const auto target = destination / relative;
      std::filesystem::create_directories(target.parent_path());
      std::filesystem::copy_file(exact, target);
    };
    for (const auto &module : loaded_modules) {
      if (InstallerPathWithin(module, executable_directory)) {
        const auto relative =
            std::filesystem::relative(module, executable_directory);
        copy(module, relative);
        // Preserve a loader-visible symlink name as a regular copied file too.
        const auto lexical = module.lexically_relative(executable_directory);
        if (lexical != relative && !lexical.empty() && *lexical.begin() != "..")
          copy(module, lexical);
      }
    }
    for (const auto name :
         {runtime::CurrentInstallerManagerExecutableName(),
          runtime::CurrentBackendPackInstallerExecutableName(),
          runtime::CurrentRuntimeBootstrapperExecutableName(),
          runtime::CurrentProductRemovalFinalizerExecutableName()})
      copy(executable_directory / name, std::filesystem::path(name));
    for (const auto *relative :
         {"resources/cyxwiz.png", "resources/fonts/Inter-Regular.ttf",
          "resources/fonts/Inter-Bold.ttf",
          "resources/fonts/fa-solid-900.ttf"}) {
      if (std::filesystem::is_regular_file(executable_directory / relative))
        copy(executable_directory / relative, relative);
    }
    // Preserve authenticated metadata for Install again. Delivery still
    // verifies signatures normally; archives and activation state are
    // deliberately excluded.
    if (std::filesystem::is_regular_file(metadata_root /
                                         "trust/trusted-keys.json"))
      copy(metadata_root / "trust/trusted-keys.json",
           "runtime/trust/trusted-keys.json");
    if (std::filesystem::is_regular_file(metadata_root /
                                         "catalogs/current.json"))
      copy(metadata_root / "catalogs/current.json",
           "runtime/catalogs/current.json");
    const auto manifests = metadata_root / "catalogs/manifests";
    if (std::filesystem::is_directory(manifests)) {
      for (const auto &entry : std::filesystem::directory_iterator(manifests)) {
        if (entry.path().extension() == ".json") {
          if (!entry.is_regular_file() || entry.is_symlink())
            throw std::runtime_error("Redirected installer manifest");
          copy(entry.path(),
               std::filesystem::path("runtime/catalogs/manifests") /
                   entry.path().filename());
        }
      }
    }
    return true;
  } catch (const std::exception &exception) {
    error = "Cannot stage the external installer session: " +
            std::string(exception.what());
    return false; // Keep bounded staging for diagnostics; never delete a broad
                  // path.
  }
}

ExternalInstallerSession PrepareExternalInstallerSession(
    const std::filesystem::path &executable_directory,
    const std::filesystem::path &runtime_root,
    const std::filesystem::path &metadata_root,
    const std::vector<std::string> &arguments, std::string &error) {
  if (!InstallerPathWithin(executable_directory, runtime_root.parent_path()))
    return ExternalInstallerSession::Ready;
  try {
    // The helper also uses LibArchive. Retain this shared runtime in the
    // manager's loader closure so its transitive dependencies are staged too,
    // even before the first extraction. No new dependency is introduced.
    if (archive_version_number() <= 0) {
      error = "The installer archive runtime is unavailable";
      return ExternalInstallerSession::Failed;
    }
    const auto modules = LoadedInstallerModules(error);
    if (!error.empty())
      return ExternalInstallerSession::Failed;
    const auto destination =
        std::filesystem::temp_directory_path() /
        ("cyxwiz-manager-" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    if (InstallerPathWithin(destination, runtime_root.parent_path()) ||
        !StageExternalInstallerSession(executable_directory, metadata_root,
                                       modules, destination, error))
      return ExternalInstallerSession::Failed;
    auto forwarded = arguments;
    // Replace rather than duplicate --metadata-root; keep the user's source
    // when it is already outside the installation (including offline test
    // archives).
    if (InstallerPathWithin(metadata_root, runtime_root.parent_path())) {
      bool replaced = false;
      for (std::size_t i = 1; i + 1 < forwarded.size(); ++i) {
        if (forwarded[i] == "--metadata-root") {
          forwarded[i + 1] = (destination / "runtime").string();
          replaced = true;
          break;
        }
      }
      if (!replaced) {
        forwarded.push_back("--metadata-root");
        forwarded.push_back((destination / "runtime").string());
      }
    }
    return RelaunchExternalInstallerSession(destination, forwarded, error)
               ? ExternalInstallerSession::Relaunched
               : ExternalInstallerSession::Failed;
  } catch (const std::exception &exception) {
    error = exception.what();
    return ExternalInstallerSession::Failed;
  }
}

} // namespace cyxwiz::installer
