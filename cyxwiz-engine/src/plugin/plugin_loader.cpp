#include "plugin_loader.h"
#include "security/plugin_signature.h"
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <system_error>

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <Windows.h>
#else
    #include <dlfcn.h>
#endif

namespace cyxwiz::plugin {

namespace {

std::filesystem::path EnvPath(const char* name) {
    const char* value = std::getenv(name);
    return value && *value ? std::filesystem::path(value) : std::filesystem::path{};
}

// Per-user folder of the library copies the Engine loads (shadow copies).
std::filesystem::path LibraryCacheRoot() {
#ifdef _WIN32
    auto base = EnvPath("LOCALAPPDATA");
    if (!base.empty()) return base / "CyxWiz" / "cache" / "plugins";
#elif defined(__APPLE__)
    const auto home = EnvPath("HOME");
    if (!home.empty()) return home / "Library" / "Caches" / "CyxWiz" / "plugins";
#else
    auto base = EnvPath("XDG_CACHE_HOME");
    if (base.empty() && !EnvPath("HOME").empty()) base = EnvPath("HOME") / ".cache";
    if (!base.empty()) return base / "cyxwiz" / "plugins";
#endif
    std::error_code ec;
    const auto temp = std::filesystem::temp_directory_path(ec);
    return (ec ? std::filesystem::path(".") : temp) / "cyxwiz" / "plugins";
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

// The library's content (SHA-512, first 16 bytes) and, for the libraries
// beside it, name, size and time: a rebuild of any of them is a new copy.
std::string LibraryContentKey(const std::vector<std::filesystem::path>& files) {
    const auto hash = security::PluginSignature::HashFile(files.front());
    std::string key = security::PluginSignature::ToHex(hash.data(), std::min<size_t>(hash.size(), 16));
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

// Copies the plugin's library and the shared libraries beside it into
// <cache>/<plugin id>/<content key>/ and returns the copy of the library.
// The Engine loads the copy, so the build output is never locked: a
// rebuild can replace it while the Engine runs, and Unload + Install loads
// the new build while models made by the old one keep using theirs.
std::filesystem::path ShadowCopyLibrary(const std::string& plugin_id, const std::filesystem::path& library,
                                        std::string& error) {
    const auto files = LibraryFolderFiles(library);
    if (security::PluginSignature::HashFile(library).empty()) {
        error = "cannot read " + library.string();
        return {};
    }
    const auto folder = LibraryCacheRoot() / SafeFolderName(plugin_id) / LibraryContentKey(files);
    const auto copy = folder / library.filename();
    std::error_code ec;
    if (std::filesystem::is_directory(folder, ec)) {
        // This build was copied before (and may be loaded): fill in what is
        // missing; a file in use is the same bytes.
        for (const auto& file : files) {
            const auto target = folder / file.filename();
            std::error_code size_error;
            if (std::filesystem::file_size(target, size_error) == std::filesystem::file_size(file, size_error) &&
                !size_error)
                continue;
            std::filesystem::copy_file(file, target, std::filesystem::copy_options::overwrite_existing, ec);
            if (ec) {
                error = "cannot copy " + file.string() + ": " + ec.message();
                return {};
            }
        }
        return copy;
    }
    // Staged, then renamed into place: a reader never sees half a copy.
    const auto staging = folder.parent_path() / (folder.filename().string() + ".staging-" +
                                                  std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::remove_all(staging, ec);
    std::filesystem::create_directories(staging, ec);
    if (ec) {
        error = "cannot create " + staging.string() + ": " + ec.message();
        return {};
    }
    for (const auto& file : files) {
        std::filesystem::copy_file(file, staging / file.filename(),
                                   std::filesystem::copy_options::overwrite_existing, ec);
        if (ec) {
            error = "cannot copy " + file.string() + ": " + ec.message();
            std::filesystem::remove_all(staging, ec);
            return {};
        }
    }
    std::filesystem::rename(staging, folder, ec);
    if (ec) {
        std::error_code ignored;
        std::filesystem::remove_all(staging, ignored);
        if (!std::filesystem::is_regular_file(copy, ignored)) {  // not copied by another Engine meanwhile
            error = "cannot place the copy in " + folder.string() + ": " + ec.message();
            return {};
        }
    }
    return copy;
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
    std::string& error_out
) {
    auto loaded = std::make_unique<LoadedPlugin>();
    loaded->plugin_dir = plugin_dir;

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
    // checked on the copy, the bytes that run.
    std::filesystem::path load_path = ShadowCopyLibrary(loaded->manifest.id, dll_path, error_out);
    if (load_path.empty()) {
        error_out = "Cannot copy the plugin library for loading: " + error_out;
        loaded->state = PluginState::Failed;
        loaded->error_message = error_out;
        return nullptr;
    }
    loaded->loaded_from = load_path;

    // 3b. Verify Ed25519 signature
    {
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
    loaded->dll_handle = LoadDLL(load_path, error_out);
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
