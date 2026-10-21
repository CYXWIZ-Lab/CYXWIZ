#pragma once

#include <cctype>
#include <cstdlib>
#include <string>
#include <filesystem>

namespace cyxwiz::servernode {

// Where remote jobs put their fetched dataset files: CYXWIZ_NODE_DATA_DIR if
// set (put it on a data drive), else the temp folder.
inline std::filesystem::path NodeDataRoot() {
    if (const char* configured = std::getenv("CYXWIZ_NODE_DATA_DIR"); configured && *configured) {
        return configured;
    }
    return std::filesystem::temp_directory_path() / "cyxwiz-node";
}

// A job's resume checkpoints (TOFIX118 P4e-3): kept apart from its fetched
// datasets so they survive the job's cleanup; removed once the job succeeds.
// A job re-sent with the same id (node restart, Engine reconnect) resumes
// from them.
inline std::filesystem::path NodeJobCheckpointDir(const std::string& job_id) {
    std::string folder;
    for (const char c : job_id) folder.push_back(std::isalnum(static_cast<unsigned char>(c)) ? c : '_');
    return NodeDataRoot() / "checkpoints" / (folder.empty() ? std::string("job") : folder);
}

}  // namespace cyxwiz::servernode
