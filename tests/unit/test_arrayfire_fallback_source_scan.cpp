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
    if (hit.path == "cyxwiz-backend/src/algorithms/distributed/process_group.cpp" &&
        hit.line.find("NCCL backend requested") != std::string::npos) {
        return true;
    }
    if (hit.path == "cyxwiz-backend/src/algorithms/layers/linear.cpp" &&
        hit.line.find("LinearLayer: GPU check failed") != std::string::npos) {
        return true;
    }
    if (hit.path == "cyxwiz-backend/src/algorithms/layers/layer_recurrent_utils.cpp" &&
        hit.line.find("native CPU recurrent path") != std::string::npos) {
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
           WindowContains(lines, catch_index, "ArrayFireFallback") ||
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
    if (relative_path ==
            "cyxwiz-engine/src/core/execution_device_context.h" &&
        WindowContains(lines, catch_index, "context.valid = false", 8) &&
        WindowContains(lines, catch_index,
                       "context.effective_backend = \"query_failed\"", 8)) {
        return true;
    }
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

std::vector<fs::path> ArrayFireFallbackHandlerScanFiles(
    const fs::path& repo_root) {
    std::vector<fs::path> files;
    const fs::path algorithms_root =
        repo_root / "cyxwiz-backend" / "src" / "algorithms";
    for (const auto& entry : fs::recursive_directory_iterator(algorithms_root)) {
        if (entry.is_regular_file() && IsSourceFile(entry.path())) {
            files.push_back(entry.path());
        }
    }

    const fs::path core_root =
        repo_root / "cyxwiz-backend" / "src" / "core";
    const std::vector<std::string> core_tensor_files = {
        "tensor.cpp",
        "tensor_broadcast.cpp",
        "tensor_comparison.cpp",
        "tensor_concat.cpp",
        "tensor_elementwise.cpp",
        "tensor_indexing.cpp",
        "tensor_linalg.cpp",
        "tensor_logical.cpp",
        "tensor_reductions.cpp",
        "tensor_shape.cpp",
    };
    for (const std::string& name : core_tensor_files) {
        const fs::path path = core_root / name;
        if (fs::exists(path)) {
            files.push_back(path);
        }
    }

    const fs::path engine_core_root =
        repo_root / "cyxwiz-engine" / "src" / "core";
    const std::vector<std::string> engine_training_files = {
        "arrow_dataset_batcher.cpp",
        "checkpoint_manager.cpp",
        "classification_decision.cpp",
        "dataset_batcher.cpp",
        "execution_device_context.cpp",
        "execution_device_context.h",
        "execution_device_preferences.h",
        "execution_placement_plan.h",
        "graph_compiler.cpp",
        "graph_executable_model.cpp",
        "image_dataset_batcher.cpp",
        "parquet_arrow_batcher.cpp",
        "sequence_model_input.h",
        "training_executor.cpp",
        "training_executor.h",
        "training_manager.cpp",
    };
    for (const std::string& name : engine_training_files) {
        const fs::path path = engine_core_root / name;
        if (fs::exists(path)) {
            files.push_back(path);
        }
    }

    std::sort(files.begin(), files.end());
    return files;
}

} // namespace

TEST_CASE("ArrayFire fallback raw warning strings stay behind the shared policy",
          "[arrayfire][fallback][source_scan]") {
    const fs::path repo_root = FindRepoRoot();
    const auto scan_files = ArrayFireFallbackHandlerScanFiles(repo_root);
    REQUIRE_FALSE(scan_files.empty());

    const std::vector<std::string> raw_needles = {
        "falling back to CPU",
        "falling back to native CPU",
        "using native CPU fallback",
        "using CPU",
        "GPU initialization failed",
        "ArrayFire init failed",
        "ArrayFire GRULayer::Forward failed",
    };

    std::vector<RawFallbackHit> unexpected_hits;
    for (const auto& path : scan_files) {
        std::ifstream in(path);
        REQUIRE(in.is_open());

        const std::string relative_path =
            fs::relative(path, repo_root).generic_string();
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
    const auto scan_files = ArrayFireFallbackHandlerScanFiles(repo_root);
    REQUIRE_FALSE(scan_files.empty());

    std::vector<ArrayFireCatchHit> unexpected_handlers;
    for (const auto& path : scan_files) {
        const std::vector<std::string> lines = ReadLines(path);
        const std::string relative_path =
            fs::relative(path, repo_root).generic_string();
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

TEST_CASE("ArrayFire backend availability decisions are not cached process-wide",
          "[arrayfire][fallback][source_scan]") {
    const fs::path repo_root = FindRepoRoot();
    const auto scan_files = ArrayFireFallbackHandlerScanFiles(repo_root);
    REQUIRE_FALSE(scan_files.empty());

    const std::vector<std::string> stale_cache_needles = {
        "s_gpu_checked",
        "s_use_gpu",
        "GPU availability check (cached)",
    };

    std::vector<RawFallbackHit> stale_hits;
    for (const auto& path : scan_files) {
        std::ifstream in(path);
        REQUIRE(in.is_open());

        const std::string relative_path =
            fs::relative(path, repo_root).generic_string();
        std::string line;
        size_t line_number = 0;
        while (std::getline(in, line)) {
            ++line_number;
            for (const auto& needle : stale_cache_needles) {
                if (line.find(needle) != std::string::npos) {
                    stale_hits.push_back(
                        RawFallbackHit{relative_path, line_number, needle, line});
                }
            }
        }
    }

    std::sort(stale_hits.begin(), stale_hits.end(),
              [](const RawFallbackHit& lhs, const RawFallbackHit& rhs) {
                  if (lhs.path != rhs.path) {
                      return lhs.path < rhs.path;
                  }
                  return lhs.line_number < rhs.line_number;
              });
    INFO("Stale process-wide backend availability cache hits:" +
         FormatHits(stale_hits));
    REQUIRE(stale_hits.empty());
}

TEST_CASE("ArrayFire training hot path uses semantic and explicit host Tensor accessors",
          "[arrayfire][residency][source_scan]") {
    const fs::path repo_root = FindRepoRoot();
    const std::vector<std::string> relative_paths = {
        "cyxwiz-backend/src/algorithms/activations/relu.cpp",
        "cyxwiz-backend/src/algorithms/activations/sigmoid.cpp",
        "cyxwiz-backend/src/algorithms/activations/tanh.cpp",
        "cyxwiz-backend/src/algorithms/layers/dense.cpp",
        "cyxwiz-backend/src/algorithms/sequential/regularization_shape_modules.cpp",
        "cyxwiz-backend/src/algorithms/optimizers/adam_family.cpp",
        "cyxwiz-engine/src/core/classification_decision.cpp",
        "cyxwiz-engine/src/core/training_executor.cpp",
    };
    const std::vector<std::string> implicit_access_needles = {
        ".GetArray()",
        ".SetFromArray(",
        ".Data<",
    };

    std::vector<RawFallbackHit> implicit_access_hits;
    for (const auto& relative_path : relative_paths) {
        const fs::path path = repo_root / relative_path;
        std::ifstream in(path);
        REQUIRE(in.is_open());

        std::string line;
        size_t line_number = 0;
        while (std::getline(in, line)) {
            ++line_number;
            for (const auto& needle : implicit_access_needles) {
                if (line.find(needle) != std::string::npos) {
                    implicit_access_hits.push_back(
                        RawFallbackHit{
                            relative_path, line_number, needle, line});
                }
            }
        }
    }

    INFO("Implicit Tensor access in semantic training hot path:" +
         FormatHits(implicit_access_hits));
    REQUIRE(implicit_access_hits.empty());
}
