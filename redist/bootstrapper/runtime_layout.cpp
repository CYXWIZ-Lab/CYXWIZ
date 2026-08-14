#include "runtime_layout.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <ctime>
#include <cwctype>
#include <fstream>
#include <iomanip>
#include <set>
#include <utility>

#include <nlohmann/json.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace cyxwiz::runtime {
namespace {

using Json = nlohmann::json;
constexpr std::uintmax_t kMaximumStateBytes = 1024 * 1024;

bool IsIdentifier(const std::string& value) {
    if (value.empty() || value.size() > 128 ||
        !std::isalnum(static_cast<unsigned char>(value.front()))) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](unsigned char character) {
        return std::islower(character) || std::isdigit(character) ||
               character == '.' || character == '_' || character == '-';
    });
}

bool HasExactKeys(
    const Json& value,
    std::initializer_list<const char*> expected) {
    if (!value.is_object() || value.size() != expected.size()) {
        return false;
    }
    return std::all_of(expected.begin(), expected.end(), [&](const char* key) {
        return value.contains(key);
    });
}

std::wstring FoldCase(std::wstring value) {
    std::transform(value.begin(), value.end(), value.begin(), [](wchar_t character) {
        return static_cast<wchar_t>(std::towlower(character));
    });
    return value;
}

bool IsWithin(
    const std::filesystem::path& canonical_root,
    const std::filesystem::path& candidate) {
    const auto root = FoldCase(canonical_root.native());
    const auto child = FoldCase(candidate.native());
    if (child == root) {
        return true;
    }
    if (child.size() <= root.size() || child.compare(0, root.size(), root) != 0) {
        return false;
    }
    const wchar_t separator = child[root.size()];
    return separator == L'\\' || separator == L'/';
}

bool ResolveContainedDirectory(
    const std::filesystem::path& canonical_root,
    const std::filesystem::path& candidate,
    const std::string& label,
    std::filesystem::path& output,
    std::string& error) {
    std::error_code filesystem_error;
    if (!std::filesystem::is_directory(candidate, filesystem_error) || filesystem_error) {
        error = label + " is missing: " + candidate.string();
        return false;
    }
    output = std::filesystem::canonical(candidate, filesystem_error);
    if (filesystem_error || !IsWithin(canonical_root, output)) {
        error = label + " resolves outside the runtime root";
        return false;
    }
    return true;
}

bool ResolveContainedFile(
    const std::filesystem::path& canonical_root,
    const std::filesystem::path& candidate,
    const std::string& label,
    std::filesystem::path& output,
    std::string& error) {
    std::error_code filesystem_error;
    if (!std::filesystem::is_regular_file(candidate, filesystem_error) || filesystem_error) {
        error = label + " is missing: " + candidate.string();
        return false;
    }
    output = std::filesystem::canonical(candidate, filesystem_error);
    if (filesystem_error || !IsWithin(canonical_root, output)) {
        error = label + " resolves outside the runtime root";
        return false;
    }
    return true;
}

bool ReadState(const std::filesystem::path& path, Json& output, std::string& error) {
    std::error_code filesystem_error;
    const auto size = std::filesystem::file_size(path, filesystem_error);
    if (filesystem_error) {
        error = "active-runtime.json is missing: " + path.string();
        return false;
    }
    if (size == 0 || size > kMaximumStateBytes) {
        error = "active-runtime.json has an invalid size";
        return false;
    }
    try {
        std::ifstream stream(path, std::ios::binary);
        if (!stream) {
            error = "active-runtime.json cannot be opened";
            return false;
        }
        output = Json::parse(stream, nullptr, true, true);
    } catch (const std::exception& exception) {
        error = std::string("active-runtime.json is invalid JSON: ") + exception.what();
        return false;
    }
    return true;
}

bool ReadIdentifier(
    const Json& object,
    const char* key,
    std::string& output,
    std::string& error);

bool ParseState(
    const Json& state,
    ActiveRuntimeState& output,
    std::string& error) {
    output = {};
    if (!HasExactKeys(
            state,
            {"schema_version", "runtime_set_id", "generation", "base_pack_id", "packs"})) {
        error = "active-runtime.json has unknown or missing fields";
        return false;
    }
    if (!state["schema_version"].is_number_unsigned() ||
        state["schema_version"].get<std::uint64_t>() != 1) {
        error = "schema_version must be 1";
        return false;
    }
    if (!ReadIdentifier(state, "runtime_set_id", output.runtime_set_id, error) ||
        !ReadIdentifier(state, "base_pack_id", output.base_pack_id, error)) {
        return false;
    }
    if (!state["generation"].is_number_unsigned()) {
        error = "generation must be a positive integer";
        return false;
    }
    output.generation = state["generation"].get<std::uint64_t>();
    if (output.generation == 0) {
        error = "generation must be a positive integer";
        return false;
    }
    if (!state["packs"].is_array()) {
        error = "packs must be an array";
        return false;
    }

    static constexpr std::array<const char*, 3> kBackends = {
        "cuda", "opencl", "oneapi"};
    std::set<std::string> seen_backends;
    for (std::size_t index = 0; index < state["packs"].size(); ++index) {
        const auto& entry = state["packs"][index];
        if (!HasExactKeys(entry, {"backend", "pack_id"})) {
            error = "packs[" + std::to_string(index) +
                    "] has unknown or missing fields";
            return false;
        }
        ActivePackState pack;
        if (!ReadIdentifier(entry, "backend", pack.backend, error) ||
            !ReadIdentifier(entry, "pack_id", pack.pack_id, error)) {
            return false;
        }
        if (std::find(kBackends.begin(), kBackends.end(), pack.backend) ==
            kBackends.end()) {
            error = "packs[" + std::to_string(index) +
                    "].backend is unsupported";
            return false;
        }
        if (!seen_backends.insert(pack.backend).second) {
            error = "duplicate active backend: " + pack.backend;
            return false;
        }
        output.packs.push_back(std::move(pack));
    }
    std::sort(
        output.packs.begin(), output.packs.end(),
        [](const ActivePackState& left, const ActivePackState& right) {
            return left.backend < right.backend;
        });
    return true;
}

Json StateDocument(const ActiveRuntimeState& state) {
    Json packs = Json::array();
    for (const auto& pack : state.packs) {
        packs.push_back({{"backend", pack.backend}, {"pack_id", pack.pack_id}});
    }
    return {
        {"schema_version", std::uint64_t{1}},
        {"runtime_set_id", state.runtime_set_id},
        {"generation", state.generation},
        {"base_pack_id", state.base_pack_id},
        {"packs", std::move(packs)}};
}

bool ReadIdentifier(
    const Json& object,
    const char* key,
    std::string& output,
    std::string& error) {
    if (!object.at(key).is_string()) {
        error = std::string(key) + " must be a string";
        return false;
    }
    output = object.at(key).get<std::string>();
    if (!IsIdentifier(output)) {
        error = std::string(key) + " is not a safe identifier";
        return false;
    }
    return true;
}

void AddDirectoryIfPresent(
    const std::filesystem::path& canonical_root,
    const std::filesystem::path& candidate,
    std::vector<std::filesystem::path>& directories) {
    std::error_code filesystem_error;
    if (!std::filesystem::is_directory(candidate, filesystem_error) || filesystem_error) {
        return;
    }
    const auto canonical = std::filesystem::canonical(candidate, filesystem_error);
    if (!filesystem_error && IsWithin(canonical_root, canonical)) {
        directories.push_back(canonical);
    }
}

}  // namespace

bool LoadActiveRuntimeState(
    const std::filesystem::path& path,
    ActiveRuntimeState& output,
    std::string& error) {
    error.clear();
    Json state;
    return ReadState(path, state, error) && ParseState(state, output, error);
}

bool SaveActiveRuntimeStateAtomic(
    const std::filesystem::path& path,
    const ActiveRuntimeState& state,
    std::string& error) {
    error.clear();
    ActiveRuntimeState normalized;
    if (!ParseState(StateDocument(state), normalized, error)) return false;

    std::error_code filesystem_error;
    std::filesystem::create_directories(path.parent_path(), filesystem_error);
    if (filesystem_error) {
        error = "cannot create runtime-state directory: " +
                filesystem_error.message();
        return false;
    }
    auto temporary = path;
    temporary += ".tmp." + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    try {
        std::ofstream stream(
            temporary, std::ios::binary | std::ios::trunc);
        if (!stream) {
            error = "cannot create temporary runtime state";
            return false;
        }
        stream << StateDocument(normalized).dump(2) << '\n';
        stream.flush();
        if (!stream) {
            error = "cannot write temporary runtime state";
            stream.close();
            std::filesystem::remove(temporary, filesystem_error);
            return false;
        }
        stream.close();
#ifdef _WIN32
        if (std::filesystem::exists(path, filesystem_error)) {
            if (!::ReplaceFileW(
                    path.c_str(), temporary.c_str(), nullptr,
                    REPLACEFILE_WRITE_THROUGH, nullptr, nullptr)) {
                error = "cannot atomically replace runtime state; Win32 error " +
                        std::to_string(::GetLastError());
                std::filesystem::remove(temporary, filesystem_error);
                return false;
            }
        } else if (!::MoveFileExW(
                       temporary.c_str(), path.c_str(),
                       MOVEFILE_WRITE_THROUGH | MOVEFILE_REPLACE_EXISTING)) {
            error = "cannot atomically publish runtime state; Win32 error " +
                    std::to_string(::GetLastError());
            std::filesystem::remove(temporary, filesystem_error);
            return false;
        }
#else
        std::filesystem::rename(temporary, path, filesystem_error);
        if (filesystem_error) {
            error = "cannot atomically publish runtime state: " +
                    filesystem_error.message();
            std::filesystem::remove(temporary, filesystem_error);
            return false;
        }
#endif
    } catch (const std::exception& exception) {
        error = std::string("runtime-state publication failed: ") +
                exception.what();
        std::filesystem::remove(temporary, filesystem_error);
        return false;
    }
    return true;
}

bool ResolveRuntimeState(
    const std::filesystem::path& runtime_root,
    const ActiveRuntimeState& state,
    ActiveRuntime& output,
    std::string& error) {
    output = {};
    error.clear();
    std::error_code filesystem_error;
    if (!std::filesystem::is_directory(runtime_root, filesystem_error) ||
        filesystem_error) {
        error = "runtime root is missing: " + runtime_root.string();
        return false;
    }
    const auto canonical_root =
        std::filesystem::canonical(runtime_root, filesystem_error);
    if (filesystem_error) {
        error = "runtime root cannot be resolved";
        return false;
    }

    ActiveRuntimeState normalized;
    if (!ParseState(StateDocument(state), normalized, error)) return false;
    output.runtime_set_id = normalized.runtime_set_id;
    output.generation = normalized.generation;
    output.base_pack_id = normalized.base_pack_id;
    output.runtime_root = canonical_root;
    if (!ResolveContainedDirectory(
            canonical_root,
            canonical_root / "base" / output.base_pack_id,
            "active base pack", output.base_directory, error)) {
        return false;
    }
    if (!ResolveContainedFile(
            canonical_root, output.base_directory / "cyxwiz-engine.exe",
            "active base Engine", output.engine_executable, error)) {
        return false;
    }
    output.dll_directories.push_back(output.base_directory);
    AddDirectoryIfPresent(
        canonical_root, output.base_directory / "arrayfire" / "bin",
        output.dll_directories);
    AddDirectoryIfPresent(
        canonical_root, output.base_directory / "python",
        output.dll_directories);

    for (const auto& state_pack : normalized.packs) {
        ActivePack pack;
        pack.backend = state_pack.backend;
        pack.pack_id = state_pack.pack_id;
        if (!ResolveContainedDirectory(
                canonical_root,
                canonical_root / "packs" / pack.backend / pack.pack_id,
                "active " + pack.backend + " pack", pack.directory, error)) {
            return false;
        }
        std::filesystem::path runtime_directory;
        if (!ResolveContainedDirectory(
                canonical_root, pack.directory / "runtime",
                "active " + pack.backend + " runtime",
                runtime_directory, error)) {
            return false;
        }
        std::filesystem::path plugin;
        if (!ResolveContainedFile(
                canonical_root,
                runtime_directory / ("af" + pack.backend + ".dll"),
                "active " + pack.backend + " plugin", plugin, error)) {
            return false;
        }
        output.packs.push_back(std::move(pack));
    }
    for (const auto& pack : output.packs) {
        output.dll_directories.push_back(pack.directory / "runtime");
    }
    return true;
}

bool ResolveActiveRuntime(
    const std::filesystem::path& runtime_root,
    ActiveRuntime& output,
    std::string& error) {
    ActiveRuntimeState state;
    const auto path = runtime_root / "active-runtime.json";
    return LoadActiveRuntimeState(path, state, error) &&
           ResolveRuntimeState(runtime_root, state, output, error);
}

void AppendBootstrapDiagnostic(
    const std::filesystem::path& runtime_root,
    const std::string& message) {
    try {
        std::ofstream stream(runtime_root / "bootstrapper.log", std::ios::app);
        if (!stream) {
            return;
        }
        const auto now = std::chrono::system_clock::now();
        const auto time = std::chrono::system_clock::to_time_t(now);
        std::tm utc{};
        gmtime_s(&utc, &time);
        stream << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ") << " " << message << '\n';
    } catch (...) {
        // Diagnostics must never obscure the original launch failure.
    }
}

}  // namespace cyxwiz::runtime
