#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace cyxwiz::runtime {

struct ActivePack {
    std::string backend;
    std::string pack_id;
    std::filesystem::path directory;
};

struct ActiveRuntime {
    std::string runtime_set_id;
    std::uint64_t generation = 0;
    std::string base_pack_id;
    std::filesystem::path runtime_root;
    std::filesystem::path base_directory;
    std::filesystem::path engine_executable;
    std::vector<ActivePack> packs;
    std::vector<std::filesystem::path> dll_directories;
};

bool ResolveActiveRuntime(
    const std::filesystem::path& runtime_root,
    ActiveRuntime& output,
    std::string& error);

void AppendBootstrapDiagnostic(
    const std::filesystem::path& runtime_root,
    const std::string& message);

}  // namespace cyxwiz::runtime
