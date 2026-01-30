#include "plugin_loader.h"
#include "security/plugin_signature.h"
#include <spdlog/spdlog.h>
#include <fstream>

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <Windows.h>
#else
    #include <dlfcn.h>
#endif

namespace cyxwiz::plugin {

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

    void* handle = ::LoadLibraryExW(
        path.wstring().c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);

    if (cookie) {
        ::RemoveDllDirectory(cookie);
    }

    if (!handle) {
        error_out = GetLastDLLError();
    }
    return handle;
#else
    void* handle = ::dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
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
        manifest_out = PluginManifest::FromJson(j);
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

    spdlog::info("Loading plugin: {} v{} from {}",
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

    // 3b. Verify Ed25519 signature
    {
        auto sig_result = security::PluginSignature::Verify(dll_path, loaded->manifest.signature);
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
    loaded->dll_handle = LoadDLL(dll_path, error_out);
    if (!loaded->dll_handle) {
        error_out = "Failed to load DLL: " + error_out;
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
    spdlog::info("Plugin loaded: {} v{}", loaded->manifest.id, loaded->manifest.version.ToString());
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

} // namespace cyxwiz::plugin
