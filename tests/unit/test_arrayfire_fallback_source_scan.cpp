#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
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

struct LegacyTensorDataOwner {
    const char* path;
    size_t expected_line_count;
    const char* disposition;
    const char* owner;
};

const std::vector<LegacyTensorDataOwner>& LegacyTensorDataInventory() {
    // This is a bounded migration inventory, not a permanent allow-list.
    // "compatibility_compute" rows must either become ArrayFire-first or gain
    // an explicit observed fallback boundary. The other dispositions are
    // host-owned APIs that still need conversion to ReadData/MutableData.
    static const std::vector<LegacyTensorDataOwner> inventory = {
        {"cyxwiz-backend/src/algorithms/data/training_data_factories.cpp", 2, "explicit_host_boundary", "dataset adapter"},
        {"cyxwiz-backend/src/algorithms/distributed/cpu_backend.cpp", 5, "selected_native_cpu_backend", "CPU process group"},
        {"cyxwiz-backend/src/algorithms/distributed/ddp.cpp", 2, "transport_boundary", "DDP bucket transport"},
        {"cyxwiz-backend/src/algorithms/distributed/distributed_trainer.cpp", 10, "compatibility_compute", "legacy distributed trainer"},
        {"cyxwiz-backend/src/algorithms/distributed/distributed_trainer_checkpoint.cpp", 2, "explicit_host_boundary", "distributed checkpoint serialization"},
        {"cyxwiz-backend/src/algorithms/distributed/nccl_backend.cpp", 6, "transport_boundary", "NCCL transport"},
        {"cyxwiz-backend/src/algorithms/distributed/process_group.cpp", 1, "transport_boundary", "process-group transport"},
        {"cyxwiz-backend/src/algorithms/feature_importance.cpp", 3, "compatibility_compute", "feature importance"},
        {"cyxwiz-backend/src/algorithms/layers/batch_norm.cpp", 15, "compatibility_compute", "BatchNorm layer"},
        {"cyxwiz-backend/src/algorithms/layers/conv_transpose2d.cpp", 10, "compatibility_compute", "ConvTranspose2D layer"},
        {"cyxwiz-backend/src/algorithms/layers/conv1d.cpp", 11, "compatibility_compute", "Conv1D layer"},
        {"cyxwiz-backend/src/algorithms/layers/conv2d.cpp", 10, "compatibility_compute", "Conv2D layer"},
        {"cyxwiz-backend/src/algorithms/layers/gru.cpp", 20, "compatibility_compute", "GRU layer"},
        {"cyxwiz-backend/src/algorithms/layers/gru_backward.cpp", 7, "compatibility_compute", "GRU backward"},
        {"cyxwiz-backend/src/algorithms/layers/linear.cpp", 21, "compatibility_compute", "Linear layer"},
        {"cyxwiz-backend/src/algorithms/layers/lstm.cpp", 35, "compatibility_compute", "LSTM layer"},
        {"cyxwiz-backend/src/algorithms/layers/lstm_backward.cpp", 11, "compatibility_compute", "LSTM backward"},
        {"cyxwiz-backend/src/algorithms/layers/lstm_direction_helpers.cpp", 26, "compatibility_compute", "LSTM direction kernels"},
        {"cyxwiz-backend/src/algorithms/layers/normalization.cpp", 39, "compatibility_compute", "normalization layers"},
        {"cyxwiz-backend/src/algorithms/layers/transformer_layers.cpp", 56, "compatibility_compute", "transformer layers"},
        {"cyxwiz-backend/src/algorithms/layers/upsampling.cpp", 8, "compatibility_compute", "upsampling layers"},
        {"cyxwiz-backend/src/algorithms/losses/classification_losses.cpp", 14, "compatibility_compute", "classification losses"},
        {"cyxwiz-backend/src/algorithms/losses/loss_utils.cpp", 14, "compatibility_compute", "remaining non-regression loss compatibility kernels"},
        {"cyxwiz-backend/src/algorithms/losses/metric_learning_losses.cpp", 24, "compatibility_compute", "metric-learning losses"},
        {"cyxwiz-backend/src/algorithms/losses/probability_losses.cpp", 15, "compatibility_compute", "probability losses"},
        {"cyxwiz-backend/src/algorithms/model_interpretability.cpp", 10, "compatibility_compute", "model interpretability"},
        {"cyxwiz-backend/src/algorithms/sequential/feedforward_modules.cpp", 11, "compatibility_compute", "feed-forward modules"},
        {"cyxwiz-backend/src/algorithms/sequential/model_io.cpp", 2, "explicit_host_boundary", "model serialization"},
        {"cyxwiz-backend/src/algorithms/sequential/normalization_modules.cpp", 21, "compatibility_compute", "normalization modules"},
        {"cyxwiz-backend/src/algorithms/sequential/recurrent_modules.cpp", 15, "compatibility_compute", "recurrent modules"},
        {"cyxwiz-engine/src/core/checkpoint_payload_io.cpp", 2, "explicit_host_boundary", "checkpoint serialization"},
        {"cyxwiz-engine/src/core/dataset_batcher.cpp", 4, "explicit_host_boundary", "dataset ingress"},
        {"cyxwiz-engine/src/core/graph_executable_model.cpp", 8, "compatibility_compute", "graph executable model"},
        {"cyxwiz-engine/src/core/image_dataset_batcher.cpp", 1, "explicit_host_boundary", "image transform ingress"},
        {"cyxwiz-engine/src/core/language_model_generation.cpp", 1, "explicit_host_boundary", "token selection output"},
        {"cyxwiz-engine/src/core/language_model_training.cpp", 1, "explicit_host_boundary", "validation token output"},
        {"cyxwiz-engine/src/core/metric_learning_inference_outputs.h", 3, "explicit_host_boundary", "inference output materialization"},
        {"cyxwiz-engine/src/core/metric_learning_losses.h", 1, "explicit_host_boundary", "metric-learning label convention validation"},
        {"cyxwiz-engine/src/core/metric_learning_metrics.h", 7, "compatibility_compute", "metric-learning metrics"},
        {"cyxwiz-engine/src/core/model_exporter.cpp", 3, "explicit_host_boundary", "model export"},
        {"cyxwiz-engine/src/core/model_importer.cpp", 1, "explicit_host_boundary", "model import"},
        {"cyxwiz-engine/src/core/sequence_model_input.h", 11, "explicit_host_boundary", "sequence input canonicalization"},
        {"cyxwiz-engine/src/core/sequence_tag_metrics.h", 3, "compatibility_compute", "sequence metrics"},
        {"cyxwiz-engine/src/core/smoke_run_executor.cpp", 2, "compatibility_compute", "smoke-run metrics"},
        {"cyxwiz-engine/src/core/test_executor.cpp", 3, "compatibility_compute", "test evaluation"},
    };
    return inventory;
}

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

std::vector<fs::path> LegacyTensorDataScanFiles(const fs::path& repo_root) {
    std::vector<fs::path> files;
    const std::vector<fs::path> roots = {
        repo_root / "cyxwiz-backend" / "src" / "algorithms",
        repo_root / "cyxwiz-engine" / "src" / "core",
    };
    for (const auto& root : roots) {
        for (const auto& entry : fs::recursive_directory_iterator(root)) {
            if (entry.is_regular_file() && IsSourceFile(entry.path())) {
                files.push_back(entry.path());
            }
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

size_t CountLegacyTensorDataLines(const fs::path& path) {
    const std::vector<std::string> needles = {
        ".Data<",
        ".Data()",
        "->Data<",
        "->Data()",
    };
    size_t count = 0;
    for (const auto& line : ReadLines(path)) {
        if (std::any_of(
                needles.begin(), needles.end(),
                [&line](const std::string& needle) {
                    return line.find(needle) != std::string::npos;
                })) {
            ++count;
        }
    }
    return count;
}

bool FunctionBodyContains(
    const std::vector<std::string>& lines,
    const std::string& function_name,
    const std::string& needle) {
    bool found_signature = false;
    bool found_body = false;
    int brace_depth = 0;
    for (const auto& line : lines) {
        if (!found_signature) {
            if (line.find(function_name + "(") == std::string::npos) {
                continue;
            }
            found_signature = true;
        }
        if (line.find(needle) != std::string::npos) {
            return true;
        }
        for (const char ch : line) {
            if (ch == '{') {
                found_body = true;
                ++brace_depth;
            } else if (ch == '}') {
                --brace_depth;
            }
        }
        if (found_body && brace_depth == 0) {
            return false;
        }
    }
    return false;
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
        "cyxwiz-engine/src/core/sequence_training_step.h",
        "cyxwiz-engine/src/core/training_executor.cpp",
    };
    const std::vector<std::string> implicit_access_needles = {
        ".GetArray()",
        ".SetFromArray(",
        ".Data<",
        ".Data()",
        "->Data<",
        "->Data()",
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

TEST_CASE("Legacy Tensor Data compatibility access has exact reviewed ownership",
          "[arrayfire][fallback][tensor_host_access][source_scan]") {
    const fs::path repo_root = FindRepoRoot();
    const auto& inventory = LegacyTensorDataInventory();
    REQUIRE_FALSE(inventory.empty());

    const std::set<std::string> valid_dispositions = {
        "compatibility_compute",
        "explicit_host_boundary",
        "selected_native_cpu_backend",
        "transport_boundary",
    };
    std::map<std::string, size_t> expected;
    for (const auto& row : inventory) {
        INFO("Legacy Tensor Data inventory row: " << row.path);
        REQUIRE(row.expected_line_count > 0);
        REQUIRE_FALSE(std::string(row.owner).empty());
        REQUIRE(valid_dispositions.count(row.disposition) == 1);
        REQUIRE(expected.emplace(row.path, row.expected_line_count).second);
    }

    std::map<std::string, size_t> actual;
    for (const auto& path : LegacyTensorDataScanFiles(repo_root)) {
        const size_t count = CountLegacyTensorDataLines(path);
        if (count == 0) {
            continue;
        }
        actual.emplace(fs::relative(path, repo_root).generic_string(), count);
    }

    INFO("Every legacy Data-access file must have a reviewed owner and exact "
         "line count; update the inventory only after reviewing the new path.");
    REQUIRE(actual == expected);
}

TEST_CASE("Tensor-facing fallback helpers enforce strict policy before native compute",
          "[arrayfire][fallback][policy][source_scan]") {
    const fs::path repo_root = FindRepoRoot();
    const std::vector<std::pair<std::string, std::string>> guarded_helpers = {
        {"cyxwiz-backend/src/algorithms/activation.cpp",
         "LogActivationFallbackOnce"},
        {"cyxwiz-backend/src/algorithms/layers/pooling.cpp",
         "LogPoolingFallbackOnce"},
        {"cyxwiz-backend/src/algorithms/layers/multi_head_attention.cpp",
         "LogAttentionInitializationFallbackOnce"},
        {"cyxwiz-backend/src/algorithms/linear_algebra_tensor.cpp",
         "LogLinearAlgebraTensorFallbackOnce"},
    };

    for (const auto& [relative_path, helper_name] : guarded_helpers) {
        const auto lines = ReadLines(repo_root / relative_path);
        INFO(relative_path << "::" << helper_name);
        REQUIRE(FunctionBodyContains(
            lines,
            helper_name,
            "ThrowIfArrayFireNativeCpuFallbackForbidden"));
    }
}

TEST_CASE("Host-vector model evaluation has one native execution owner",
          "[arrayfire][classification_metrics][ownership][source_scan]") {
    const fs::path repo_root = FindRepoRoot();
    const std::vector<std::string> relative_paths = {
        "cyxwiz-backend/src/algorithms/evaluation/classification_metrics.cpp",
        "cyxwiz-backend/src/algorithms/evaluation/cross_validation.cpp",
        "cyxwiz-backend/src/algorithms/evaluation/roc_pr_curves.cpp",
        "cyxwiz-backend/src/algorithms/model_evaluation.cpp",
    };
    const std::vector<std::string> forbidden_needles = {
        "CYXWIZ_HAS_ARRAYFIRE",
        "arrayfire.h",
        "af::",
        "ArrayFireHostSyncCategory",
        "LogEvaluationFallbackOnce",
    };

    std::vector<RawFallbackHit> unexpected_hits;
    for (const auto& relative_path : relative_paths) {
        const auto lines = ReadLines(repo_root / relative_path);
        for (size_t line_index = 0; line_index < lines.size(); ++line_index) {
            for (const auto& needle : forbidden_needles) {
                if (lines[line_index].find(needle) != std::string::npos) {
                    unexpected_hits.push_back(RawFallbackHit{
                        relative_path,
                        line_index + 1,
                        needle,
                        lines[line_index]});
                }
            }
        }
    }

    INFO("Host-vector evaluation must not upload to a selected device or hide "
         "a size-dependent native fallback:" + FormatHits(unexpected_hits));
    REQUIRE(unexpected_hits.empty());

    const auto forwarding_header = ReadLines(
        repo_root / "cyxwiz-engine/src/core/model_evaluation.h");
    REQUIRE(std::any_of(
        forwarding_header.begin(), forwarding_header.end(),
        [](const std::string& line) {
            return line.find("#include <cyxwiz/model_evaluation.h>") !=
                   std::string::npos;
        }));
    for (const auto& line : forwarding_header) {
        CHECK(line.find("struct BinaryMetrics") == std::string::npos);
        CHECK(line.find("class ModelEvaluation") == std::string::npos);
    }
}
