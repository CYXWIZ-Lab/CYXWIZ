#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace cyxwiz::runtime {

struct VerifiedPackComponent {
    std::string relative_path;
    std::uint64_t size = 0;
    std::string sha256;
};

struct VerifiedBackendPackPayload {
    std::string runtime_set_id;
    std::string companion_base_id;
    std::string backend;
    std::string pack_id;
    std::filesystem::path source_directory;
    std::vector<VerifiedPackComponent> components;
};

}  // namespace cyxwiz::runtime
