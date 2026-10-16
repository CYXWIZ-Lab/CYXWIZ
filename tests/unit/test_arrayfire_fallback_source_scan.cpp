#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
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
    // an explicit observed fallback boundary. The remaining transport row
    // requires dedicated device-pointer ownership rather than a host accessor.
    static const std::vector<LegacyTensorDataOwner> inventory = {
        {"cyxwiz-backend/src/algorithms/distributed/distributed_trainer.cpp", 10, "compatibility_compute", "legacy distributed trainer"},
        {"cyxwiz-backend/src/algorithms/distributed/nccl_backend.cpp", 6, "transport_boundary", "NCCL transport"},
        {"cyxwiz-backend/src/algorithms/feature_importance.cpp", 3, "compatibility_compute", "feature importance"},
        {"cyxwiz-backend/src/algorithms/providers/nvidia_cublas_provider.cpp", 8, "transport_boundary", "NVIDIA neural provider host/device boundary copies (reviewed 2026-09-23, contract 0.6.0: rnn/lstm/gru forward outputs + optional stacked final states, lstm/gru backward dx and per-layer weight/bias gradients; explicit v1 boundary copies)"},
        {"cyxwiz-backend/src/algorithms/providers/opencl_cell_provider.cpp", 8, "transport_boundary", "OpenCL neural provider host/device boundary copies (reviewed 2026-09-23, contract 0.6.0 mirror of the NVIDIA provider: forward outputs + optional stacked final states, backward dx and per-layer gradients)"},
        {"cyxwiz-backend/src/algorithms/model_interpretability.cpp", 10, "compatibility_compute", "model interpretability"},
        {"cyxwiz-engine/src/core/sequence_tag_metrics.h", 2, "transport_boundary", "BIO span matching reads tag ids"},
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

bool IsAllowedArrayFireCatch(
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
    // Device-only layers (one ArrayFire path, no host loop): the handler
    // reports the error ("... failed on the ArrayFire device: ...", directly
    // or through a Throw...DeviceError helper) instead of falling back.
    // A provider classifies the error to decline a contract.
    if (WindowContains(lines, catch_index, "ClassifyArrayFireBackendFallbackReason", 4)) {
        return true;
    }
    if (WindowContains(lines, catch_index, "DeviceError(", 3) ||
        WindowContains(lines, catch_index, "failed on the ArrayFire device", 4)) {
        return true;
    }
    if (relative_path == "cyxwiz-backend/src/algorithms/optimizers/optimizer_utils.cpp" &&
        WindowContains(lines, catch_index, "s_use_gpu = false", 8)) {
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

TEST_CASE("ArrayFire exception handlers report the device error",
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
            if (IsAllowedArrayFireCatch(relative_path, lines, i)) {
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
        "cyxwiz-backend/src/algorithms/layers/gru.cpp",
        "cyxwiz-backend/src/algorithms/layers/gru_backward.cpp",
        "cyxwiz-backend/src/algorithms/layers/layer_recurrent_utils.cpp",
        "cyxwiz-backend/src/algorithms/layers/lstm.cpp",
        "cyxwiz-backend/src/algorithms/layers/lstm_backward.cpp",
        "cyxwiz-backend/src/algorithms/layers/lstm_initialization.cpp",
        "cyxwiz-backend/src/algorithms/layers/pixel_shuffle.cpp",
        "cyxwiz-backend/src/algorithms/layers/rnn.cpp",
        "cyxwiz-backend/src/algorithms/layers/upsampling.cpp",
        "cyxwiz-backend/src/algorithms/layers/multi_head_attention_arrayfire.cpp",
        "cyxwiz-backend/src/algorithms/layers/layer_norm_arrayfire.cpp",
        "cyxwiz-backend/src/algorithms/layers/transformer_layers.cpp",
        "cyxwiz-backend/src/algorithms/sequential/recurrent_modules.cpp",
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

// TOFIX140 F4: core Tensor ops run on ArrayFire at every rank; there is no
// native fallback to guard, so none may come back.
TEST_CASE("Core Tensor ops have no native fallback",
          "[arrayfire][fallback][policy][source_scan]") {
    const fs::path repo_root = FindRepoRoot();
    const std::vector<std::string> relative_paths = {
        "cyxwiz-backend/src/core/tensor_arithmetic.cpp",
        "cyxwiz-backend/src/core/tensor_broadcast.cpp",
        "cyxwiz-backend/src/core/tensor_comparison.cpp",
        "cyxwiz-backend/src/core/tensor_concat.cpp",
        "cyxwiz-backend/src/core/tensor_elementwise.cpp",
        "cyxwiz-backend/src/core/tensor_factory.cpp",
        "cyxwiz-backend/src/core/tensor_indexing.cpp",
        "cyxwiz-backend/src/core/tensor_linalg.cpp",
        "cyxwiz-backend/src/core/tensor_logical.cpp",
        "cyxwiz-backend/src/core/tensor_math_utils.h",
        "cyxwiz-backend/src/core/tensor_reductions.cpp",
        "cyxwiz-backend/src/core/tensor_shape.cpp",
    };
    const std::vector<std::string> forbidden_needles = {
        "RecordArrayFireFallback",
        "AllowNativeCpuFallback",
        "memcpy",
        "Native",
        "ranks up to 4",
    };
    std::vector<RawFallbackHit> hits;
    for (const auto& relative_path : relative_paths) {
        const auto lines = ReadLines(repo_root / relative_path);
        REQUIRE_FALSE(lines.empty());
        for (size_t line_index = 0; line_index < lines.size(); ++line_index) {
            for (const auto& needle : forbidden_needles) {
                if (lines[line_index].find(needle) != std::string::npos) {
                    hits.push_back(RawFallbackHit{relative_path, line_index + 1, needle, lines[line_index]});
                }
            }
        }
    }
    INFO("Core Tensor ops must stay device-only:" + FormatHits(hits));
    REQUIRE(hits.empty());
}

TEST_CASE("ROC / PR curves run on the device without a fallback",
          "[arrayfire][roc_pr][device_only][source_scan]") {
    // TOFIX140: the sorted cumulative counts behind ROC / PR run on the
    // selected ArrayFire device for every input size; there is no native
    // path or size switch to fall back to.
    const fs::path repo_root = FindRepoRoot();
    const auto lines = ReadLines(repo_root / "cyxwiz-backend/src/algorithms/evaluation/roc_pr_curves.cpp");
    REQUIRE_FALSE(lines.empty());
    const std::vector<std::string> forbidden_needles = {
        "CYXWIZ_HAS_ARRAYFIRE",
        "ArrayFireHostSyncCategory",
        "LogEvaluationFallbackOnce",
        "ArgSort",
    };
    std::vector<RawFallbackHit> hits;
    bool sorts_on_device = false;
    for (size_t line_index = 0; line_index < lines.size(); ++line_index) {
        // The device sort: SortWithIndex (ArrayFire sort-by-key, arrayfire_indexed.h).
        if (lines[line_index].find("SortWithIndex(") != std::string::npos) sorts_on_device = true;
        for (const auto& needle : forbidden_needles) {
            if (lines[line_index].find(needle) != std::string::npos) {
                hits.push_back(RawFallbackHit{"roc_pr_curves.cpp", line_index + 1, needle, lines[line_index]});
            }
        }
    }
    INFO("ROC / PR must stay device-only:" + FormatHits(hits));
    REQUIRE(hits.empty());
    REQUIRE(sorts_on_device);
}

TEST_CASE("Host-vector model evaluation has one native execution owner",
          "[arrayfire][classification_metrics][ownership][source_scan]") {
    const fs::path repo_root = FindRepoRoot();
    const std::vector<std::string> relative_paths = {
        "cyxwiz-backend/src/algorithms/evaluation/classification_metrics.cpp",
        "cyxwiz-backend/src/algorithms/evaluation/cross_validation.cpp",
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

// Every call of `function` (e.g. "af::sort") in the backend and engine sources,
// outside arrayfire_indexed.h: "path:line" and the call's top-level arguments.
struct ScannedCall {
    std::string where;
    std::vector<std::string> arguments;
};

std::vector<ScannedCall> ScanArrayFireCalls(const std::string& function) {
    const fs::path repo_root = FindRepoRoot();
    const std::string needle = function + "(";
    std::vector<ScannedCall> calls;
    for (const char* root : {"cyxwiz-backend/src", "cyxwiz-engine/src"}) {
        for (const auto& entry : fs::recursive_directory_iterator(repo_root / root)) {
            if (!entry.is_regular_file() || !IsSourceFile(entry.path())) continue;
            const std::string relative = fs::relative(entry.path(), repo_root).generic_string();
            if (relative == "cyxwiz-backend/src/algorithms/arrayfire_indexed.h") continue;
            std::ifstream in(entry.path(), std::ios::binary);
            const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            for (size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + 1)) {
                ScannedCall call;
                const auto line = std::count(text.begin(), text.begin() + static_cast<std::ptrdiff_t>(at), '\n') + 1;
                call.where = relative + ":" + std::to_string(line);
                int depth = 0;
                std::string argument;
                for (size_t i = at + needle.size() - 1; i < text.size(); ++i) {
                    const char c = text[i];
                    if (c == '(' && ++depth == 1) continue;
                    if (c == ')' && --depth == 0) break;
                    if (c == ',' && depth == 1) {
                        call.arguments.push_back(argument);
                        argument.clear();
                        continue;
                    }
                    if (c != ' ' && c != '\n' && c != '\r') argument += c;
                }
                call.arguments.push_back(argument);
                calls.push_back(call);
            }
        }
    }
    return calls;
}

TEST_CASE("Argsorts go through SortWithIndex, not ArrayFire's sort-with-index",
          "[arrayfire][source_scan]") {
    // af::sort(sorted, indices, values, dim, ascending) once did not return on
    // the oneAPI backend (Iris Xe, 2026-10-14; see arrayfire_indexed.h);
    // SortWithIndex uses sort-by-key.
    // Flags every five-argument af::sort call outside arrayfire_indexed.h.
    std::string list;
    for (const auto& call : ScanArrayFireCalls("af::sort")) {
        if (call.arguments.size() == 5) list += "\n  " + call.where;
    }
    INFO("af::sort with indices (use SortWithIndex):" + list);
    REQUIRE(list.empty());
}

TEST_CASE("Indexed min / max along a non-zero dimension go through MinWithIndex / MaxWithIndex",
          "[arrayfire][source_scan]") {
    // af::max(value, index, in, dim) and af::min with dim != 0 reduce only part
    // of each row on the oneAPI backend (Iris Xe, 2026-10-14); the helpers
    // reduce along dim 0. Flags every four-argument af::max / af::min call whose
    // dimension is not the literal 0, outside arrayfire_indexed.h.
    std::string list;
    for (const char* function : {"af::max", "af::min"}) {
        for (const auto& call : ScanArrayFireCalls(function)) {
            if (call.arguments.size() == 4 && call.arguments[3] != "0") list += "\n  " + call.where;
        }
    }
    INFO("indexed af::max / af::min along dim != 0 (use MaxWithIndex / MinWithIndex):" + list);
    REQUIRE(list.empty());
}
