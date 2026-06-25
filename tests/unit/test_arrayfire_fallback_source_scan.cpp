#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

struct RawFallbackHit {
    std::string path;
    size_t line_number;
    std::string needle;
    std::string line;
};

struct ArrayFireCatchHit {
    std::string path;
    size_t line_number;
    std::string line;
};

fs::path FindRepoRoot() {
    auto dir = fs::current_path();
    while (!dir.empty()) {
        if (fs::exists(dir / "cyxwiz-backend" / "src" / "algorithms") &&
            fs::exists(dir / "tests" / "CMakeLists.txt")) {
            return dir;
        }
        const auto parent = dir.parent_path();
        if (parent == dir) {
            break;
        }
        dir = parent;
    }
    return fs::current_path();
}

bool IsSourceFile(const fs::path& path) {
    const std::string ext = path.extension().string();
    return ext == ".cpp" || ext == ".h" || ext == ".hpp";
}

bool IsAllowedRawFallbackHit(const RawFallbackHit& hit) {
    if (hit.path == "cyxwiz-backend/src/algorithms/arrayfire_backend_utils.cpp" &&
        hit.line.find("Training continues") != std::string::npos) {
        return true;
    }
    if (hit.path == "cyxwiz-backend/src/algorithms/layers/layer_recurrent_utils.cpp" &&
        (hit.line.find("falling back to CPU") != std::string::npos ||
         hit.line.find("using CPU directly") != std::string::npos)) {
        return true;
    }
    if (hit.path == "cyxwiz-backend/src/algorithms/distributed/process_group.cpp" &&
        hit.line.find("NCCL backend requested") != std::string::npos) {
        return true;
    }
    if (hit.path == "cyxwiz-backend/src/algorithms/layers/linear.cpp" &&
        hit.line.find("LinearLayer: GPU check failed") != std::string::npos) {
        return true;
    }
    return false;
}

std::string FormatHits(const std::vector<RawFallbackHit>& hits) {
    std::ostringstream out;
    for (const auto& hit : hits) {
        out << "\n" << hit.path << ":" << hit.line_number
            << ": matched \"" << hit.needle << "\": " << hit.line;
    }
    return out.str();
}

std::string FormatCatchHits(const std::vector<ArrayFireCatchHit>& hits) {
    std::ostringstream out;
    for (const auto& hit : hits) {
        out << "\n" << hit.path << ":" << hit.line_number
            << ": " << hit.line;
    }
    return out.str();
}

std::vector<std::string> ReadLines(const fs::path& path) {
    std::ifstream in(path);
    REQUIRE(in.is_open());

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        lines.push_back(line);
    }
    return lines;
}

bool WindowContains(
    const std::vector<std::string>& lines,
    size_t start_index,
    const std::string& needle,
    size_t max_lines = 35) {
    const size_t end_index = std::min(lines.size(), start_index + max_lines);
    for (size_t i = start_index; i < end_index; ++i) {
        if (lines[i].find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

bool UsesSharedFallbackPolicy(
    const std::vector<std::string>& lines,
    size_t catch_index) {
    return WindowContains(lines, catch_index, "FallbackOnce") ||
           WindowContains(lines, catch_index, "BackendFailureOnce") ||
           WindowContains(lines, catch_index, "BackendWarningOnce") ||
           WindowContains(lines, catch_index,
                          "ClassifyArrayFireBackendFallbackReason") ||
           WindowContains(lines, catch_index,
                          "BuildArrayFireBackendFallbackMessage") ||
           WindowContains(lines, catch_index,
                          "BuildRecurrentFormalParameterOverflowFallbackMessage");
}

bool IsAllowedArrayFireCatchWithoutFallbackPolicy(
    const std::string& relative_path,
    const std::vector<std::string>& lines,
    size_t catch_index) {
    if (WindowContains(lines, catch_index, "GPU check failed", 8)) {
        return true;
    }
    if (relative_path == "cyxwiz-backend/src/algorithms/optimizers/optimizer_utils.cpp" &&
        WindowContains(lines, catch_index, "s_use_gpu = false", 8)) {
        return true;
    }
    if (relative_path == "cyxwiz-backend/src/algorithms/layers/layer_recurrent_utils.cpp" &&
        WindowContains(lines, catch_index, "return true", 8)) {
        return true;
    }
    if (relative_path == "cyxwiz-backend/src/algorithms/layers/gru.cpp" &&
        WindowContains(lines, catch_index,
                       "Disabled legacy AF reference path", 12)) {
        return true;
    }
    return false;
}

} // namespace

TEST_CASE("ArrayFire fallback raw warning strings stay behind the shared policy",
          "[arrayfire][fallback][source_scan]") {
    const fs::path repo_root = FindRepoRoot();
    const fs::path algorithms_root =
        repo_root / "cyxwiz-backend" / "src" / "algorithms";
    REQUIRE(fs::exists(algorithms_root));

    const std::vector<std::string> raw_needles = {
        "falling back to CPU",
        "using CPU",
        "GPU initialization failed",
        "ArrayFire init failed",
        "ArrayFire GRULayer::Forward failed",
    };

    std::vector<RawFallbackHit> unexpected_hits;
    for (const auto& entry : fs::recursive_directory_iterator(algorithms_root)) {
        if (!entry.is_regular_file() || !IsSourceFile(entry.path())) {
            continue;
        }

        std::ifstream in(entry.path());
        REQUIRE(in.is_open());

        const std::string relative_path =
            fs::relative(entry.path(), repo_root).generic_string();
        std::string line;
        size_t line_number = 0;
        while (std::getline(in, line)) {
            ++line_number;
            for (const auto& needle : raw_needles) {
                if (line.find(needle) == std::string::npos) {
                    continue;
                }
                RawFallbackHit hit{relative_path, line_number, needle, line};
                if (!IsAllowedRawFallbackHit(hit)) {
                    unexpected_hits.push_back(std::move(hit));
                }
            }
        }
    }

    std::sort(unexpected_hits.begin(), unexpected_hits.end(),
              [](const RawFallbackHit& lhs, const RawFallbackHit& rhs) {
                  if (lhs.path != rhs.path) {
                      return lhs.path < rhs.path;
                  }
                  return lhs.line_number < rhs.line_number;
              });
    INFO("Unexpected raw fallback strings:" + FormatHits(unexpected_hits));
    REQUIRE(unexpected_hits.empty());
}

TEST_CASE("ArrayFire exception handlers route operation fallbacks through shared policy",
          "[arrayfire][fallback][source_scan]") {
    const fs::path repo_root = FindRepoRoot();
    const fs::path algorithms_root =
        repo_root / "cyxwiz-backend" / "src" / "algorithms";
    REQUIRE(fs::exists(algorithms_root));

    std::vector<ArrayFireCatchHit> unexpected_handlers;
    for (const auto& entry : fs::recursive_directory_iterator(algorithms_root)) {
        if (!entry.is_regular_file() || !IsSourceFile(entry.path())) {
            continue;
        }

        const std::vector<std::string> lines = ReadLines(entry.path());
        const std::string relative_path =
            fs::relative(entry.path(), repo_root).generic_string();
        for (size_t i = 0; i < lines.size(); ++i) {
            if (lines[i].find("catch (const af::exception") ==
                std::string::npos) {
                continue;
            }
            if (UsesSharedFallbackPolicy(lines, i) ||
                IsAllowedArrayFireCatchWithoutFallbackPolicy(
                    relative_path, lines, i)) {
                continue;
            }
            unexpected_handlers.push_back(
                ArrayFireCatchHit{relative_path, i + 1, lines[i]});
        }
    }

    std::sort(unexpected_handlers.begin(), unexpected_handlers.end(),
              [](const ArrayFireCatchHit& lhs,
                 const ArrayFireCatchHit& rhs) {
                  if (lhs.path != rhs.path) {
                      return lhs.path < rhs.path;
                  }
                  return lhs.line_number < rhs.line_number;
              });
    INFO("Unexpected ArrayFire exception handlers:" +
         FormatCatchHits(unexpected_handlers));
    REQUIRE(unexpected_handlers.empty());
}
