#include "plugin_loader.h"
#include "security/plugin_signature.h"
#include "../core/user_folders.h"
#include "../../../redist/bootstrapper/backend_pack_path.h"
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <functional>
#include <memory>
#include <set>
#include <system_error>

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <Windows.h>
#else
    #include <dlfcn.h>
#endif

namespace cyxwiz::plugin {

namespace {

// Per-user folder of the library copies the Engine loads (shadow copies);
// empty when no user folder is known.
std::filesystem::path LibraryCacheRoot() {
    return UserFile(UserCacheDir(), "plugins");
}

// .dll, .dylib, and .so with or without a version (libfoo.so.1).
bool IsSharedLibraryName(const std::filesystem::path& path) {
    const auto name = path.filename().string();
    const auto ext = path.extension().string();
    return ext == ".dll" || ext == ".DLL" || ext == ".dylib" || name.find(".so") != std::string::npos;
}

std::string SafeFolderName(const std::string& text) {
    std::string out;
    for (char c : text) out += (std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '-') ? c : '_';
    return out.empty() ? "plugin" : out;
}

// The shared libraries in the library's folder (the library first).
std::vector<std::filesystem::path> LibraryFolderFiles(const std::filesystem::path& library) {
    std::vector<std::filesystem::path> files{library};
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(library.parent_path(), ec)) {
        if (entry.is_regular_file(ec) && IsSharedLibraryName(entry.path()) && entry.path() != library)
            files.push_back(entry.path());
    }
    return files;
}

// The library's content (SHA-512, first 16 bytes, hex): the start of its
// copy's key, and what a reused copy is compared with. Empty when unreadable.
std::string LibraryContentHex(const std::filesystem::path& library) {
    const auto hash = security::PluginSignature::HashFile(library);
    return hash.empty() ? std::string{}
                        : security::PluginSignature::ToHex(hash.data(), std::min<size_t>(hash.size(), 16));
}

// The library's content and, for the libraries beside it, name, size and
// time: a rebuild of any of them is a new copy.
std::string LibraryContentKey(const std::string& library_hex, const std::vector<std::filesystem::path>& files) {
    std::string key = library_hex;
    uint64_t others = 1469598103934665603ull;  // FNV-1a over the others
    const auto mix = [&others](const std::string& text) {
        for (unsigned char c : text) others = (others ^ c) * 1099511628211ull;
    };
    std::error_code ec;
    for (size_t i = 1; i < files.size(); ++i) {
        mix(files[i].filename().string());
        mix(std::to_string(std::filesystem::file_size(files[i], ec)));
        mix(std::to_string(static_cast<long long>(
            std::filesystem::last_write_time(files[i], ec).time_since_epoch().count())));
    }
    if (files.size() > 1) key += "-" + security::PluginSignature::ToHex(
                                       reinterpret_cast<const uint8_t*>(&others), sizeof(others));
    return key;
}

// Copies `files` into `folder` (<cache>/<plugin id>/<content key>/). When
// the folder exists (this build was copied before and may be loaded), a file
// `keep` accepts stays and the others are copied again; a file in use is the
// same bytes. Otherwise the files are staged, then renamed into place: a
// reader never sees half a copy. False with `error` when a copy fails.
bool PlaceLibraryCopies(const std::filesystem::path& folder, const std::vector<std::filesystem::path>& files,
                        const std::function<bool(const std::filesystem::path& file,
                                                 const std::filesystem::path& target)>& keep,
                        std::string& error) {
    std::error_code ec;
    if (std::filesystem::is_directory(folder, ec)) {
        for (const auto& file : files) {
            const auto target = folder / file.filename();
            if (keep(file, target)) continue;
            std::filesystem::copy_file(file, target, std::filesystem::copy_options::overwrite_existing, ec);
            if (ec) {
                error = "cannot copy " + file.string() + ": " + ec.message();
                return false;
            }
        }
        return true;
    }
    const auto staging = folder.parent_path() / (folder.filename().string() + ".staging-" +
                                                  std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::remove_all(staging, ec);
    std::filesystem::create_directories(staging, ec);
    if (ec) {
        error = "cannot create " + staging.string() + ": " + ec.message();
        return false;
    }
    for (const auto& file : files) {
        std::filesystem::copy_file(file, staging / file.filename(),
                                   std::filesystem::copy_options::overwrite_existing, ec);
        if (ec) {
            error = "cannot copy " + file.string() + ": " + ec.message();
            std::filesystem::remove_all(staging, ec);
            return false;
        }
    }
    std::filesystem::rename(staging, folder, ec);
    if (ec) {
        std::error_code ignored;
        std::filesystem::remove_all(staging, ignored);
        // Not copied by another Engine meanwhile
        if (!std::filesystem::is_regular_file(folder / files.front().filename(), ignored)) {
            error = "cannot place the copy in " + folder.string() + ": " + ec.message();
            return false;
        }
    }
    return true;
}

// Copies the plugin's library and the shared libraries beside it into
// <cache>/<plugin id>/<content key>/ and returns the copy of the library.
// The Engine loads the copy, so the build output is never locked: a
// rebuild can replace it while the Engine runs, and Unload + Install loads
// the new build while models made by the old one keep using theirs.
std::filesystem::path ShadowCopyLibrary(const std::string& plugin_id, const std::filesystem::path& library,
                                        std::string& error) {
    const auto files = LibraryFolderFiles(library);
    const std::string library_hex = LibraryContentHex(library);
    if (library_hex.empty()) {
        error = "cannot read " + library.string();
        return {};
    }
    const auto cache_root = LibraryCacheRoot();
    if (cache_root.empty()) {
        error = "no user cache folder to copy the library into";
        return {};
    }
    const auto folder = cache_root / SafeFolderName(plugin_id) / LibraryContentKey(library_hex, files);
    // A reused copy must still hold the content of the build: a copy changed
    // in place keeps its size, so every copy (the library and the libraries
    // beside it) is hashed, not compared by size.
    const auto keep = [&](const std::filesystem::path& file, const std::filesystem::path& target) {
        std::error_code size_error;
        const bool same_size =
            std::filesystem::file_size(target, size_error) == std::filesystem::file_size(file, size_error) &&
            !size_error;
        return same_size && LibraryContentHex(target) == (file == library ? library_hex : LibraryContentHex(file));
    };
    if (!PlaceLibraryCopies(folder, files, keep, error)) return {};
    return folder / library.filename();
}

// ---------------------------------------------------------------------------
// Installed add-ons: the copies that load are checked against the signed
// inventory and held against writes until they are loaded.
// ---------------------------------------------------------------------------

const PackComponent* FindPackComponent(const std::vector<PackComponent>& components,
                                       const std::filesystem::path& relative) {
    const std::string wanted = runtime::FoldBackendPackPath(relative.lexically_normal().generic_string());
    for (const auto& component : components) {
        if (runtime::FoldBackendPackPath(component.path) == wanted) return &component;
    }
    return nullptr;
}

// Open handles that let others read but not change or delete the copies,
// until they are loaded (a mapped library is protected by the system then).
class CopyLock {
public:
    CopyLock() = default;
    CopyLock(const CopyLock&) = delete;
    CopyLock& operator=(const CopyLock&) = delete;
    ~CopyLock() {
#ifdef _WIN32
        for (HANDLE handle : handles_) ::CloseHandle(handle);
#endif
    }

    bool Hold(const std::filesystem::path& file, std::string& error) {
#ifdef _WIN32
        HANDLE handle = ::CreateFileW(file.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle == INVALID_HANDLE_VALUE) {
            error = "cannot hold " + file.filename().string() + " against changes (Windows error " +
                    std::to_string(::GetLastError()) + ")";
            return false;
        }
        handles_.push_back(handle);
        return true;
#else
        // Packs are Windows-only for now; elsewhere the copy is read, not locked.
        std::error_code ec;
        if (!std::filesystem::is_regular_file(file, ec)) {
            error = "cannot open " + file.filename().string();
            return false;
        }
        return true;
#endif
    }

private:
#ifdef _WIN32
    std::vector<HANDLE> handles_;
#endif
};

// The library and the shared libraries beside it in the pack, from the
// signed inventory (the folder holds exactly those files: checked before).
std::vector<const PackComponent*> PackLibraryComponents(const PackLoadCheck& pack, const PackComponent& library) {
    std::vector<const PackComponent*> result{&library};
    const auto folder = std::filesystem::path(library.path).parent_path();
    for (const auto& component : pack.components) {
        const std::filesystem::path path(component.path);
        if (&component != &library && path.parent_path() == folder && IsSharedLibraryName(path))
            result.push_back(&component);
    }
    return result;
}

// The copy's key from the signed digests: no hashing of the originals.
std::string PackContentKey(const std::vector<const PackComponent*>& libraries) {
    std::string key = "pack-" + libraries.front()->sha256.substr(0, 32);
    uint64_t others = 1469598103934665603ull;  // FNV-1a over the others' names and digests
    for (size_t i = 1; i < libraries.size(); ++i) {
        for (unsigned char c : libraries[i]->path + ":" + libraries[i]->sha256)
            others = (others ^ c) * 1099511628211ull;
    }
    if (libraries.size() > 1)
        key += "-" + security::PluginSignature::ToHex(reinterpret_cast<const uint8_t*>(&others), sizeof(others));
    return key;
}

// Removes every entry of `folder` but `keep` (file names), then lists it again:
// the folder the library loads from (its dependencies are searched there
// first) holds nothing unverified. False, with the entries, when one stays.
bool RemoveUnlistedCopies(const std::filesystem::path& folder, const std::set<std::string>& keep,
                          std::vector<PackFileIssue>& issues, std::string& error) {
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<std::filesystem::path> unlisted;
        std::error_code ec;
        for (const auto& entry : std::filesystem::directory_iterator(folder, ec)) {
            if (!keep.count(entry.path().filename().string())) unlisted.push_back(entry.path());
        }
        if (ec) {
            error = "cannot list " + folder.string() + ": " + ec.message();
            return false;
        }
        if (unlisted.empty()) return true;
        for (const auto& path : unlisted) {
            std::error_code remove_error;
            std::filesystem::remove_all(path, remove_error);
            if (pass == 1 || remove_error) {
                issues.push_back({path.filename().string(), PackFileProblem::NotListed});
                error = path.filename().string() + " in the library copies is not part of the signed package" +
                        (remove_error ? " and cannot be removed: " + remove_error.message() : std::string{});
            }
        }
        if (!issues.empty()) return false;
    }
    return true;
}

// Copies an add-on's libraries into the cache, then holds every copy against
// writes and checks it against its signed size and SHA-256. A copy that
// differs (changed in the cache) is copied again once; a second difference
// refuses the load. Any other file in that folder is removed (a library the
// plugin loads later would be found there first); one that cannot be removed
// refuses the load. On success `lock` holds the copies.
std::filesystem::path ShadowCopyPackLibrary(const std::string& plugin_id, const std::filesystem::path& plugin_dir,
                                            const PackLoadCheck& pack, const PackComponent& library,
                                            std::unique_ptr<CopyLock>& lock, std::vector<PackFileIssue>& issues,
                                            std::vector<std::filesystem::path>& copies, std::string& error) {
    const auto libraries = PackLibraryComponents(pack, library);
    const auto cache_root = LibraryCacheRoot();
    if (cache_root.empty()) {
        error = "no user cache folder to copy the library into";
        return {};
    }
    const auto folder = cache_root / SafeFolderName(plugin_id) / PackContentKey(libraries);
    std::vector<std::filesystem::path> files;
    for (const auto* component : libraries) files.push_back(plugin_dir / std::filesystem::path(component->path));
    const auto keep = [](const std::filesystem::path& file, const std::filesystem::path& target) {
        std::error_code size_error;
        return std::filesystem::file_size(target, size_error) == std::filesystem::file_size(file, size_error) &&
               !size_error;
    };
    for (int attempt = 0; attempt < 2; ++attempt) {
        if (!PlaceLibraryCopies(folder, files, keep, error)) return {};
        lock = std::make_unique<CopyLock>();
        issues.clear();
        for (const auto* component : libraries) {
            const auto copy = folder / std::filesystem::path(component->path).filename();
            PackFileProblem problem = PackFileProblem::ContentDiffers;
            std::string file_error;
            if (!lock->Hold(copy, file_error) || !pack.checks.check_file(copy, *component, problem, file_error)) {
                issues.push_back({component->path, problem});
                error = file_error;
            }
        }
        if (issues.empty()) {
            std::set<std::string> names;
            for (const auto& file : files) names.insert(file.filename().string());
            if (!RemoveUnlistedCopies(folder, names, issues, error)) return {};
            copies.clear();
            for (const auto& file : files) copies.push_back(folder / file.filename());
            return copies.front();
        }
        // Changed in the cache: start again from the checked pack folder.
        lock.reset();
        std::error_code ec;
        std::filesystem::remove_all(folder, ec);
        if (ec) {
            error = "a copy in " + folder.string() + " differs from the signed package and cannot be removed: " +
                    ec.message();
            return {};
        }
    }
    error = "its library copies still differ from the signed package after copying them again: " + error;
    return {};
}

// plugin.json -> manifest. In the Engine, not in plugin_types.h, so the
// SDK headers need no JSON library (TOFIX125 P5 step 5.3).
PluginManifest ManifestFromJson(const nlohmann::json& j) {
    PluginManifest m;
    m.id = j.value("id", "");
    m.name = j.value("name", m.id);
    m.description = j.value("description", "");
    m.version = PluginVersion::Parse(j.value("version", "0.0.0"));
    m.api_version = PluginVersion::Parse(j.value("api_version", "1.0.0"));
    m.author = j.value("author", "");
    m.license = j.value("license", "");
    m.homepage = j.value("homepage", "");
    m.repository = j.value("repository", "");

    // Permissions
    if (j.contains("permissions") && j["permissions"].is_array()) {
        m.permissions = ParsePermissions(j["permissions"].get<std::vector<std::string>>());
    }

    // Capabilities
    if (j.contains("capabilities") && j["capabilities"].is_array()) {
        for (const auto& cap : j["capabilities"]) {
            std::string s = cap.get<std::string>();
            if (s == "ProvidesNodes")       m.capabilities |= static_cast<uint32_t>(PluginCapability::ProvidesNodes);
            else if (s == "ProvidesPanels")  m.capabilities |= static_cast<uint32_t>(PluginCapability::ProvidesPanels);
            else if (s == "ProvidesData")    m.capabilities |= static_cast<uint32_t>(PluginCapability::ProvidesData);
            else if (s == "ProvidesTraining") m.capabilities |= static_cast<uint32_t>(PluginCapability::ProvidesTraining);
            else if (s == "ProvidesAnalytics") m.capabilities |= static_cast<uint32_t>(PluginCapability::ProvidesAnalytics);
            else if (s == "RequiresPython")  m.capabilities |= static_cast<uint32_t>(PluginCapability::RequiresPython);
            else if (s == "RequiresGPU")     m.capabilities |= static_cast<uint32_t>(PluginCapability::RequiresGPU);
            else if (s == "RequiresNetwork") m.capabilities |= static_cast<uint32_t>(PluginCapability::RequiresNetwork);
            else if (s == "ProvidesTensorNodes") m.capabilities |= static_cast<uint32_t>(PluginCapability::ProvidesTensorNodes);
        }
    }

    // Dependencies
    if (j.contains("dependencies") && j["dependencies"].is_array()) {
        for (const auto& dep : j["dependencies"]) {
            m.dependencies.push_back({
                dep.value("id", ""),
                dep.value("version", ">=0.0.0")
            });
        }
    }

    // Platform library
#ifdef _WIN32
    if (j.contains("platforms") && j["platforms"].contains("windows")) {
        m.library = j["platforms"]["windows"].value("library", "");
    }
#elif defined(__APPLE__)
    if (j.contains("platforms") && j["platforms"].contains("macos")) {
        m.library = j["platforms"]["macos"].value("library", "");
    }
#else
    if (j.contains("platforms") && j["platforms"].contains("linux")) {
        m.library = j["platforms"]["linux"].value("library", "");
    }
#endif
    // Fallback: top-level library field
    if (m.library.empty()) {
        m.library = j.value("library", "");
    }

    // Signature
    m.signature = j.value("signature", "");

    return m;
}

}  // namespace

// ============================================================================
// LoadedPlugin destructor
// ============================================================================

LoadedPlugin::~LoadedPlugin() {
    // Destroy plugin instance before freeing DLL
    if (instance && destroy_func) {
        try {
            destroy_func(instance);
        } catch (...) {
            spdlog::error("DestroyPlugin threw in destructor for {}", manifest.id);
        }
        instance = nullptr;
    }
    // Free DLL handle directly (do NOT call PluginLoader::Unload to avoid double-free)
    if (dll_handle) {
#ifdef _WIN32
        ::FreeLibrary(static_cast<HMODULE>(dll_handle));
#else
        ::dlclose(dll_handle);
#endif
        dll_handle = nullptr;
    }
}

// ============================================================================
// Platform-specific DLL operations
// ============================================================================

void* PluginLoader::LoadDLL(const std::filesystem::path& path, std::string& error_out) {
#ifdef _WIN32
    // Use AddDllDirectory + LOAD_LIBRARY_SEARCH_USER_DIRS for thread safety
    // (SetDllDirectoryW is process-global and not thread-safe)
    auto dir = path.parent_path().wstring();
    DLL_DIRECTORY_COOKIE cookie = ::AddDllDirectory(dir.c_str());

    // A missing DLL or function fails the call instead of showing Windows'
    // "Entry Point Not Found" box, which stops the Engine until it is closed.
    DWORD previous_mode = 0;
    ::SetThreadErrorMode(SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX, &previous_mode);
    void* handle = ::LoadLibraryExW(
        path.wstring().c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
    const DWORD load_error = ::GetLastError();
    ::SetThreadErrorMode(previous_mode, nullptr);

    if (cookie) {
        ::RemoveDllDirectory(cookie);
    }

    if (!handle) {
        ::SetLastError(load_error);
        error_out = GetLastDLLError();
        if (load_error == ERROR_PROC_NOT_FOUND) {
            error_out += " (the plugin uses a function this Engine does not have: it was built "
                         "against a newer CyxWiz SDK; rebuild it against this Engine's SDK)";
        }
    }
    return handle;
#else
    // RTLD_NOW: a function the Engine does not have fails here, with its name,
    // instead of stopping the Engine when the plugin first calls it.
    void* handle = ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        error_out = ::dlerror();
    }
    return handle;
#endif
}

void* PluginLoader::LoadPackDLL(const std::vector<std::filesystem::path>& copies,
                                std::shared_ptr<void>& dependencies, std::string& error_out) {
#ifdef _WIN32
    // The libraries beside the library by full path, in as many passes as
    // their own imports of each other need; then the library, whose import
    // of each is satisfied by the module of that name already loaded. Only
    // the default folders (the Engine's, System32) are ever searched.
    DWORD previous_mode = 0;
    ::SetThreadErrorMode(SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX, &previous_mode);
    auto loaded = std::make_shared<std::vector<HMODULE>>();
    std::vector<std::filesystem::path> pending(copies.begin() + 1, copies.end());
    bool progress = true;
    while (!pending.empty() && progress) {
        progress = false;
        for (auto it = pending.begin(); it != pending.end();) {
            if (HMODULE module = ::LoadLibraryExW(it->wstring().c_str(), nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)) {
                loaded->push_back(module);
                it = pending.erase(it);
                progress = true;
            } else {
                ++it;
            }
        }
    }
    HMODULE library = nullptr;
    if (pending.empty()) {
        library = ::LoadLibraryExW(copies.front().wstring().c_str(), nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!library) error_out = GetLastDLLError();
    } else {
        error_out = "cannot load " + pending.front().filename().string() + ": " + GetLastDLLError();
    }
    ::SetThreadErrorMode(previous_mode, nullptr);
    // Freed after the library, in reverse order.
    dependencies = std::shared_ptr<void>(loaded.get(), [loaded](void*) {
        for (auto it = loaded->rbegin(); it != loaded->rend(); ++it) ::FreeLibrary(*it);
    });
    if (!library) dependencies.reset();
    return library;
#else
    dependencies.reset();
    return LoadDLL(copies.front(), error_out);  // packs are Windows-only for now
#endif
}

void PluginLoader::UnloadDLL(void* handle) {
    if (!handle) return;
#ifdef _WIN32
    ::FreeLibrary(static_cast<HMODULE>(handle));
#else
    ::dlclose(handle);
#endif
}

void* PluginLoader::GetSymbol(void* handle, const char* name) {
    if (!handle) return nullptr;
#ifdef _WIN32
    return reinterpret_cast<void*>(::GetProcAddress(static_cast<HMODULE>(handle), name));
#else
    return ::dlsym(handle, name);
#endif
}

bool PluginLoader::CheckBuildFingerprint(void* handle, std::string& error_out) {
    const std::string engine = CurrentBuildFingerprint();
    const auto fingerprint = reinterpret_cast<BuildFingerprintFunc>(GetSymbol(handle, "CyxwizPluginBuildFingerprint"));
    if (!fingerprint) {
        error_out = MissingFingerprintMessage(engine);
        return false;
    }
    const char* text = fingerprint();
    const auto differences = CompareBuildFingerprints(engine, text ? text : "");
    if (!differences.empty()) {
        error_out = DescribeFingerprintMismatch(differences);
        return false;
    }
    return true;
}

std::string PluginLoader::GetLastDLLError() {
#ifdef _WIN32
    DWORD err = ::GetLastError();
    if (err == 0) return "";

    LPSTR buf = nullptr;
    FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<LPSTR>(&buf), 0, nullptr
    );
    std::string msg = buf ? buf : "Unknown error";
    if (buf) LocalFree(buf);
    // Trim trailing whitespace
    while (!msg.empty() && (msg.back() == '\n' || msg.back() == '\r' || msg.back() == ' '))
        msg.pop_back();
    return msg;
#else
    const char* err = ::dlerror();
    return err ? err : "Unknown error";
#endif
}

// ============================================================================
// Manifest parsing
// ============================================================================

bool PluginLoader::ParseManifest(
    const std::filesystem::path& plugin_dir,
    PluginManifest& manifest_out,
    std::string& error_out
) {
    auto manifest_path = plugin_dir / "plugin.json";
    if (!std::filesystem::exists(manifest_path)) {
        error_out = "plugin.json not found in " + plugin_dir.string();
        return false;
    }

    try {
        std::ifstream f(manifest_path);
        if (!f.is_open()) {
            error_out = "Cannot open " + manifest_path.string();
            return false;
        }

        nlohmann::json j = nlohmann::json::parse(f);
        manifest_out = ManifestFromJson(j);
        manifest_out.plugin_dir = plugin_dir;

        if (manifest_out.id.empty()) {
            error_out = "Manifest missing 'id' field";
            return false;
        }

        return true;
    } catch (const nlohmann::json::exception& e) {
        error_out = "JSON parse error: " + std::string(e.what());
        return false;
    }
}

// ============================================================================
// API version validation
// ============================================================================

bool PluginLoader::ValidateApiVersion(const PluginVersion& plugin_api, std::string& error_out) {
    PluginVersion engine_api{PLUGIN_API_VERSION_MAJOR, PLUGIN_API_VERSION_MINOR, PLUGIN_API_VERSION_PATCH};

    if (!plugin_api.IsCompatibleWith(engine_api)) {
        error_out = "API version mismatch: plugin requires " + plugin_api.ToString() +
                    ", engine provides " + engine_api.ToString();
        return false;
    }
    return true;
}

// ============================================================================
// Load plugin from directory (with manifest)
// ============================================================================

std::unique_ptr<LoadedPlugin> PluginLoader::LoadFromDirectory(
    const std::filesystem::path& plugin_dir,
    std::string& error_out,
    const PackLoadCheck* pack,
    PackLoadResult* pack_result
) {
    auto loaded = std::make_unique<LoadedPlugin>();
    loaded->plugin_dir = plugin_dir;
    const auto check_started = std::chrono::steady_clock::now();
    const auto refuse_pack = [&](std::vector<PackFileIssue> issues, const std::string& reason) {
        error_out = reason;
        if (pack_result) {
            pack_result->refused = true;
            pack_result->issues = std::move(issues);
        }
        return nullptr;
    };

    // 0. An installed add-on: every file of its folder against the signed
    // inventory (the exact file set, sizes, SHA-256) before plugin.json is read.
    // plugin.json is held against writes from before its check until it is
    // parsed, so the bytes parsed are the bytes checked.
    PackFolderCheck folder_check;
    CopyLock manifest_lock;
    if (pack) {
        if (!pack->checks.check_folder || !pack->checks.check_file)
            return refuse_pack({}, "its files cannot be checked against its signed package");
        std::string hold_error;
        if (!manifest_lock.Hold(plugin_dir / "plugin.json", hold_error))
            return refuse_pack({{"plugin.json", PackFileProblem::Unreadable}}, hold_error);
        folder_check = pack->checks.check_folder(plugin_dir, pack->components);
        if (!folder_check.ok()) {
            return refuse_pack(folder_check.issues,
                               folder_check.issues.empty() ? folder_check.error
                                                           : PackRefusalReason(folder_check.issues));
        }
    }

    // 1. Parse manifest
    if (!ParseManifest(plugin_dir, loaded->manifest, error_out)) {
        loaded->state = PluginState::Failed;
        loaded->error_message = error_out;
        return nullptr;
    }

    spdlog::debug("Loading plugin: {} v{} from {}",
                 loaded->manifest.name, loaded->manifest.version.ToString(), plugin_dir.string());

    // 2. Validate API version
    if (!ValidateApiVersion(loaded->manifest.api_version, error_out)) {
        spdlog::warn("Plugin {} skipped: {}", loaded->manifest.id, error_out);
        loaded->state = PluginState::Failed;
        loaded->error_message = error_out;
        return nullptr;
    }

    // 2b. An add-on loads its native library, one of its signed files (a
    // Python-only add-on is not supported).
    const PackComponent* pack_library = nullptr;
    if (pack) {
        pack_library = loaded->manifest.library.empty()
            ? nullptr
            : FindPackComponent(pack->components, std::filesystem::path(loaded->manifest.library));
        if (!pack_library) {
            return refuse_pack({{loaded->manifest.library, PackFileProblem::NotListed}},
                               "plugin.json names no library of its signed package");
        }
    }

    // 3. Resolve library path
    std::filesystem::path dll_path;
    if (!loaded->manifest.library.empty()) {
        dll_path = plugin_dir / loaded->manifest.library;
    } else {
        // Auto-detect: look for DLL/SO in plugin directory
#ifdef _WIN32
        for (const auto& entry : std::filesystem::directory_iterator(plugin_dir)) {
            if (entry.path().extension() == ".dll") {
                dll_path = entry.path();
                break;
            }
        }
#elif defined(__APPLE__)
        for (const auto& entry : std::filesystem::directory_iterator(plugin_dir)) {
            if (entry.path().extension() == ".dylib") {
                dll_path = entry.path();
                break;
            }
        }
#else
        for (const auto& entry : std::filesystem::directory_iterator(plugin_dir)) {
            if (entry.path().extension() == ".so") {
                dll_path = entry.path();
                break;
            }
        }
#endif
    }

    if (dll_path.empty() || !std::filesystem::exists(dll_path)) {
        // Check if this is a Python-only plugin (no native library needed)
        if (loaded->manifest.capabilities & static_cast<uint32_t>(PluginCapability::RequiresPython)) {
            spdlog::info("Plugin {} is Python-only (no native library)", loaded->manifest.id);
            loaded->state = PluginState::Loaded;
            return loaded;
        }

        error_out = "Plugin library not found: " + dll_path.string();
        loaded->state = PluginState::Failed;
        loaded->error_message = error_out;
        return nullptr;
    }

    loaded->dll_path = dll_path;

    // 3a. Load a private copy (see ShadowCopyLibrary); the signature is
    // checked on the copy, the bytes that run. An add-on's copies are checked
    // against its signed inventory and held against writes until loaded.
    std::unique_ptr<CopyLock> copy_lock;
    std::vector<std::filesystem::path> pack_copies;
    std::filesystem::path load_path;
    if (pack) {
        std::vector<PackFileIssue> copy_issues;
        load_path = ShadowCopyPackLibrary(loaded->manifest.id, plugin_dir, *pack, *pack_library, copy_lock,
                                          copy_issues, pack_copies, error_out);
        if (load_path.empty()) return refuse_pack(std::move(copy_issues), error_out);
    } else {
        load_path = ShadowCopyLibrary(loaded->manifest.id, dll_path, error_out);
    }
    if (load_path.empty()) {
        error_out = "Cannot copy the plugin library for loading: " + error_out;
        loaded->state = PluginState::Failed;
        loaded->error_message = error_out;
        return nullptr;
    }
    loaded->loaded_from = load_path;
    if (pack) {
        // The signed inventory supersedes the manifest's own signature field.
        loaded->origin = LoadedPlugin::Origin::SignedPack;
        loaded->pack_id = pack->pack_id;
        loaded->package_version = pack->package_version;
        loaded->pack_files_checked = folder_check.files_checked;
        loaded->pack_bytes_checked = folder_check.bytes_checked;
        loaded->pack_check_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - check_started).count();
        loaded->pack_checked_at = std::chrono::system_clock::now();
    }

    // 3b. Verify Ed25519 signature (a developer plugin)
    if (!pack) {
        auto sig_result = security::PluginSignature::Verify(load_path, loaded->manifest.signature);
        switch (sig_result.status) {
            case security::SignatureStatus::Valid:
                spdlog::info("Plugin {}: signature valid", loaded->manifest.id);
                break;
            case security::SignatureStatus::Invalid:
                spdlog::error("Plugin {}: INVALID signature — {}", loaded->manifest.id, sig_result.message);
                error_out = "Invalid signature: " + sig_result.message;
                loaded->state = PluginState::Failed;
                loaded->error_message = error_out;
                return nullptr;
            case security::SignatureStatus::Missing:
                spdlog::warn("Plugin {}: unsigned (no signature in manifest)", loaded->manifest.id);
                break;
            case security::SignatureStatus::FileError:
                spdlog::warn("Plugin {}: could not verify signature — {}", loaded->manifest.id, sig_result.message);
                break;
        }
    }

    // 4. Load DLL
    loaded->dll_handle = pack ? LoadPackDLL(pack_copies, loaded->pack_dependencies, error_out)
                              : LoadDLL(load_path, error_out);
    // The copies stay held while the plugin is loaded: a library it loads
    // later (delay-loaded) is read from them then.
    loaded->copy_lock = std::shared_ptr<void>(std::move(copy_lock));
    if (!loaded->dll_handle) {
        error_out = "Failed to load DLL: " + error_out;
        loaded->state = PluginState::Failed;
        loaded->error_message = error_out;
        return nullptr;
    }

    // 4b. The plugin must be built the way this Engine was (TOFIX125 P5,
    // decision D4): checked before any other plugin code runs.
    if (!CheckBuildFingerprint(loaded->dll_handle, error_out)) {
        spdlog::error("Plugin {} refused: {}", loaded->manifest.id, error_out);
        UnloadDLL(loaded->dll_handle);
        loaded->dll_handle = nullptr;
        loaded->state = PluginState::Failed;
        loaded->error_message = error_out;
        return nullptr;
    }

    // 5. Resolve entry points
    loaded->create_func = reinterpret_cast<CreatePluginFunc>(
        GetSymbol(loaded->dll_handle, "CreatePlugin"));
    loaded->destroy_func = reinterpret_cast<DestroyPluginFunc>(
        GetSymbol(loaded->dll_handle, "DestroyPlugin"));

    if (!loaded->create_func || !loaded->destroy_func) {
        error_out = "Missing required exports: CreatePlugin and/or DestroyPlugin";
        UnloadDLL(loaded->dll_handle);
        loaded->dll_handle = nullptr;
        loaded->state = PluginState::Failed;
        loaded->error_message = error_out;
        return nullptr;
    }

    // 6. Create plugin instance
    try {
        loaded->instance = loaded->create_func();
    } catch (const std::exception& e) {
        error_out = "CreatePlugin() threw: " + std::string(e.what());
        UnloadDLL(loaded->dll_handle);
        loaded->dll_handle = nullptr;
        loaded->state = PluginState::Failed;
        loaded->error_message = error_out;
        return nullptr;
    } catch (...) {
        error_out = "CreatePlugin() threw unknown exception";
        UnloadDLL(loaded->dll_handle);
        loaded->dll_handle = nullptr;
        loaded->state = PluginState::Failed;
        loaded->error_message = error_out;
        return nullptr;
    }

    if (!loaded->instance) {
        error_out = "CreatePlugin() returned nullptr";
        UnloadDLL(loaded->dll_handle);
        loaded->dll_handle = nullptr;
        loaded->state = PluginState::Failed;
        loaded->error_message = error_out;
        return nullptr;
    }

    loaded->state = PluginState::Loaded;
    if (pack) {
        spdlog::info("Plugin {}: signed add-on verified ({})", loaded->manifest.id, pack->pack_id);
        spdlog::debug("Plugin {}: {} files, {:.1f} MB checked in {:.0f} ms", loaded->manifest.id,
                      loaded->pack_files_checked, static_cast<double>(loaded->pack_bytes_checked) / 1e6,
                      loaded->pack_check_ms);
    }
    spdlog::debug("Plugin loaded: {} v{}", loaded->manifest.id, loaded->manifest.version.ToString());
    return loaded;
}

// ============================================================================
// Load plugin from DLL directly (no manifest)
// ============================================================================

std::unique_ptr<LoadedPlugin> PluginLoader::LoadFromDLL(
    const std::filesystem::path& dll_path,
    std::string& error_out
) {
    auto loaded = std::make_unique<LoadedPlugin>();
    loaded->dll_path = dll_path;
    loaded->plugin_dir = dll_path.parent_path();

    // Try to load manifest from same directory
    auto manifest_path = loaded->plugin_dir / "plugin.json";
    if (std::filesystem::exists(manifest_path)) {
        ParseManifest(loaded->plugin_dir, loaded->manifest, error_out);
    }

    // Load DLL
    loaded->dll_handle = LoadDLL(dll_path, error_out);
    if (!loaded->dll_handle) {
        loaded->state = PluginState::Failed;
        loaded->error_message = error_out;
        return nullptr;
    }

    if (!CheckBuildFingerprint(loaded->dll_handle, error_out)) {
        spdlog::error("Plugin {} refused: {}", dll_path.string(), error_out);
        UnloadDLL(loaded->dll_handle);
        loaded->dll_handle = nullptr;
        loaded->state = PluginState::Failed;
        loaded->error_message = error_out;
        return nullptr;
    }

    // Resolve symbols
    loaded->create_func = reinterpret_cast<CreatePluginFunc>(
        GetSymbol(loaded->dll_handle, "CreatePlugin"));
    loaded->destroy_func = reinterpret_cast<DestroyPluginFunc>(
        GetSymbol(loaded->dll_handle, "DestroyPlugin"));

    if (!loaded->create_func || !loaded->destroy_func) {
        error_out = "Missing required exports";
        UnloadDLL(loaded->dll_handle);
        loaded->dll_handle = nullptr;
        loaded->state = PluginState::Failed;
        loaded->error_message = error_out;
        return nullptr;
    }

    // Create instance
    try {
        loaded->instance = loaded->create_func();
    } catch (...) {
        error_out = "CreatePlugin() failed";
        UnloadDLL(loaded->dll_handle);
        loaded->dll_handle = nullptr;
        loaded->state = PluginState::Failed;
        loaded->error_message = error_out;
        return nullptr;
    }

    if (!loaded->instance) {
        error_out = "CreatePlugin() returned nullptr";
        UnloadDLL(loaded->dll_handle);
        loaded->dll_handle = nullptr;
        loaded->state = PluginState::Failed;
        loaded->error_message = error_out;
        return nullptr;
    }

    // Use manifest from plugin instance if we didn't find plugin.json
    if (loaded->manifest.id.empty() && loaded->instance) {
        loaded->manifest = loaded->instance->GetManifest();
    }

    loaded->state = PluginState::Loaded;
    return loaded;
}

// ============================================================================
// Unload plugin
// ============================================================================

void PluginLoader::Unload(LoadedPlugin& plugin) {
    // Destroy instance first
    if (plugin.instance && plugin.destroy_func) {
        try {
            plugin.destroy_func(plugin.instance);
        } catch (...) {
            spdlog::error("DestroyPlugin threw for {}", plugin.manifest.id);
        }
        plugin.instance = nullptr;
    }

    // Free DLL
    if (plugin.dll_handle) {
        UnloadDLL(plugin.dll_handle);
        plugin.dll_handle = nullptr;
    }
    plugin.pack_dependencies.reset();
    plugin.copy_lock.reset();

    plugin.state = PluginState::Unloaded;
}

void PluginLoader::PruneLibraryCache() {
    // A copy another Engine has loaded cannot be removed on Windows (it is
    // skipped); elsewhere the running process keeps its mapping.
    std::error_code ec;
    const auto root = LibraryCacheRoot();
    size_t removed = 0;
    for (const auto& plugin : std::filesystem::directory_iterator(root, ec)) {
        std::error_code inner;
        std::vector<std::filesystem::path> builds;
        for (const auto& build : std::filesystem::directory_iterator(plugin.path(), inner)) builds.push_back(build.path());
        for (const auto& build : builds) {
            // Moved aside first, so the build's own name is never left half
            // removed: an Engine that loads this build again copies it
            // afresh. A library still loaded keeps its folder (".prune") until
            // a later start-up.
            const std::string name = build.filename().string();
            const bool aside = name.find(".prune") != std::string::npos || name.find(".staging-") != std::string::npos;
            const auto doomed = aside ? build : build.parent_path() / (name + ".prune");
            std::error_code remove_error;
            if (name.find(".staging-") != std::string::npos) {
                // Another Engine may be copying into it right now.
                const auto age = std::filesystem::file_time_type::clock::now() -
                                 std::filesystem::last_write_time(build, remove_error);
                if (remove_error || age < std::chrono::hours(1)) continue;
            }
            if (!aside) std::filesystem::rename(build, doomed, remove_error);
            if (remove_error) continue;
            std::filesystem::remove_all(doomed, remove_error);
            if (!remove_error) ++removed;
        }
        std::filesystem::remove(plugin.path(), inner);  // only when empty
    }
    if (removed > 0) spdlog::debug("PluginLoader: removed {} unused plugin library copies from {}", removed, root.string());
}

} // namespace cyxwiz::plugin
