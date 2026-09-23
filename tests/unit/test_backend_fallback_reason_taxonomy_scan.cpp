#include <catch2/catch_test_macros.hpp>

#include <cyxwiz/backend_fallback_reason.h>
#include <cyxwiz/backend_placement_observation.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

struct TaxonomyHit {
    std::string path;
    size_t line_number;
    std::string needle;
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

bool IsAllowedTaxonomyHit(const std::string& relative_path,
                          const std::string& line) {
    // RouteFailureCategory is the route-qualification schema's own failure
    // taxonomy; it coincidentally shares the "unsupported_operation" spelling
    // but is a different domain with its own persisted schema. It must not be
    // coupled to BackendFallbackReason.
    return relative_path ==
               "cyxwiz-engine/src/core/route_qualification_snapshot.cpp" &&
           line.find("RouteFailureCategory") != std::string::npos;
}

std::string FormatHits(const std::vector<TaxonomyHit>& hits) {
    std::ostringstream out;
    for (const auto& hit : hits) {
        out << "\n" << hit.path << ":" << hit.line_number
            << ": matched " << hit.needle << ": " << hit.line;
    }
    return out.str();
}

const std::vector<cyxwiz::BackendFallbackReason>& AllReasons() {
    static const std::vector<cyxwiz::BackendFallbackReason> reasons = {
        cyxwiz::BackendFallbackReason::BackendUnavailable,
        cyxwiz::BackendFallbackReason::GpuBackendException,
        cyxwiz::BackendFallbackReason::ArrayFireJitCompileFailure,
        cyxwiz::BackendFallbackReason::CudaJitParamOverflow,
        cyxwiz::BackendFallbackReason::GpuOutOfMemory,
        cyxwiz::BackendFallbackReason::UnsupportedDtype,
        cyxwiz::BackendFallbackReason::UnsupportedShape,
        cyxwiz::BackendFallbackReason::UnsupportedOperation,
        cyxwiz::BackendFallbackReason::BackendCompileTimeout,
        cyxwiz::BackendFallbackReason::BackendInternalError,
        cyxwiz::BackendFallbackReason::NvidiaProviderUnavailable,
        cyxwiz::BackendFallbackReason::NvidiaProviderUnsupportedContract,
        cyxwiz::BackendFallbackReason::NvidiaProviderWorkspaceExhausted,
        cyxwiz::BackendFallbackReason::NvidiaProviderExecutionFailed,
        cyxwiz::BackendFallbackReason::OpenclProviderUnavailable,
        cyxwiz::BackendFallbackReason::OpenclProviderUnsupportedContract,
        cyxwiz::BackendFallbackReason::OpenclProviderWorkspaceExhausted,
        cyxwiz::BackendFallbackReason::OpenclProviderExecutionFailed,
        cyxwiz::BackendFallbackReason::OpenclProviderBelowRetentionFloor,
    };
    return reasons;
}

} // namespace

TEST_CASE("Backend fallback reason names pin the persisted taxonomy contract",
          "[arrayfire][fallback][taxonomy]") {
    using cyxwiz::BackendFallbackReason;
    using cyxwiz::BackendFallbackReasonName;

    // These strings are persisted in placement caches and support bundles;
    // renaming one silently invalidates prior evidence. Change requires a
    // schema/migration decision, not a refactor.
    CHECK(std::string(BackendFallbackReasonName(
              BackendFallbackReason::BackendUnavailable)) ==
          "backend_unavailable");
    CHECK(std::string(BackendFallbackReasonName(
              BackendFallbackReason::GpuBackendException)) ==
          "gpu_backend_exception");
    CHECK(std::string(BackendFallbackReasonName(
              BackendFallbackReason::ArrayFireJitCompileFailure)) ==
          "arrayfire_jit_compile_failure");
    CHECK(std::string(BackendFallbackReasonName(
              BackendFallbackReason::CudaJitParamOverflow)) ==
          "cuda_jit_param_overflow");
    CHECK(std::string(BackendFallbackReasonName(
              BackendFallbackReason::GpuOutOfMemory)) ==
          "gpu_out_of_memory");
    CHECK(std::string(BackendFallbackReasonName(
              BackendFallbackReason::UnsupportedDtype)) ==
          "unsupported_dtype");
    CHECK(std::string(BackendFallbackReasonName(
              BackendFallbackReason::UnsupportedShape)) ==
          "unsupported_shape");
    CHECK(std::string(BackendFallbackReasonName(
              BackendFallbackReason::UnsupportedOperation)) ==
          "unsupported_operation");
    CHECK(std::string(BackendFallbackReasonName(
              BackendFallbackReason::BackendCompileTimeout)) ==
          "backend_compile_timeout");
    CHECK(std::string(BackendFallbackReasonName(
              BackendFallbackReason::BackendInternalError)) ==
          "backend_internal_error");
    CHECK(std::string(BackendFallbackReasonName(
              BackendFallbackReason::NvidiaProviderUnavailable)) ==
          "nvidia_provider_unavailable");
    CHECK(std::string(BackendFallbackReasonName(
              BackendFallbackReason::NvidiaProviderUnsupportedContract)) ==
          "nvidia_provider_unsupported_contract");
    CHECK(std::string(BackendFallbackReasonName(
              BackendFallbackReason::NvidiaProviderWorkspaceExhausted)) ==
          "nvidia_provider_workspace_exhausted");
    CHECK(std::string(BackendFallbackReasonName(
              BackendFallbackReason::NvidiaProviderExecutionFailed)) ==
          "nvidia_provider_execution_failed");
    CHECK(std::string(BackendFallbackReasonName(
              BackendFallbackReason::OpenclProviderUnavailable)) ==
          "opencl_provider_unavailable");
    CHECK(std::string(BackendFallbackReasonName(
              BackendFallbackReason::OpenclProviderUnsupportedContract)) ==
          "opencl_provider_unsupported_contract");
    CHECK(std::string(BackendFallbackReasonName(
              BackendFallbackReason::OpenclProviderWorkspaceExhausted)) ==
          "opencl_provider_workspace_exhausted");
    CHECK(std::string(BackendFallbackReasonName(
              BackendFallbackReason::OpenclProviderExecutionFailed)) ==
          "opencl_provider_execution_failed");
    CHECK(std::string(BackendFallbackReasonName(
              BackendFallbackReason::OpenclProviderBelowRetentionFloor)) ==
          "opencl_provider_below_retention_floor");

    // The observation-facing namespace must be the same pointers/values as
    // the authority, not a re-typed copy.
    CHECK(std::string(
              cyxwiz::BackendPlacementObservationReason::CudaJitParamOverflow) ==
          BackendFallbackReasonName(BackendFallbackReason::CudaJitParamOverflow));
    CHECK(std::string(
              cyxwiz::BackendPlacementObservationReason::GpuOutOfMemory) ==
          BackendFallbackReasonName(BackendFallbackReason::GpuOutOfMemory));
}

TEST_CASE("Recurrent ArrayFire exception handlers enforce the strict "
          "fallback policy gate",
          "[arrayfire][fallback][taxonomy][source_scan]") {
    // tofix67 slice 5: forbid_native_cpu_fallback runs must fail closed on
    // the recurrent path instead of silently training on CPU. The catch
    // blocks in both recurrent layers must route through the shared gate.
    const fs::path repo_root = FindRepoRoot();
    for (const char* relative_path :
         {"cyxwiz-backend/src/algorithms/layers/lstm.cpp",
          "cyxwiz-backend/src/algorithms/layers/gru.cpp"}) {
        std::ifstream in(repo_root / relative_path);
        REQUIRE(in.is_open());
        std::string line;
        bool has_gate = false;
        while (std::getline(in, line)) {
            if (line.find("ThrowIfArrayFireNativeCpuFallbackForbidden") !=
                std::string::npos) {
                has_gate = true;
                break;
            }
        }
        INFO(relative_path
             << " must call ThrowIfArrayFireNativeCpuFallbackForbidden in "
                "its ArrayFire fallback handler");
        CHECK(has_gate);
    }
}

TEST_CASE("Backend fallback reason strings have exactly one defining header",
          "[arrayfire][fallback][taxonomy][source_scan]") {
    const fs::path repo_root = FindRepoRoot();
    const std::string authority =
        "cyxwiz-backend/include/cyxwiz/backend_fallback_reason.h";

    const std::vector<fs::path> scan_roots = {
        repo_root / "cyxwiz-backend" / "src",
        repo_root / "cyxwiz-backend" / "include",
        repo_root / "cyxwiz-engine" / "src",
    };

    std::vector<std::string> needles;
    for (const auto reason : AllReasons()) {
        needles.push_back(
            "\"" + std::string(cyxwiz::BackendFallbackReasonName(reason)) +
            "\"");
    }

    std::vector<TaxonomyHit> unexpected_hits;
    for (const auto& root : scan_roots) {
        REQUIRE(fs::exists(root));
        for (const auto& entry : fs::recursive_directory_iterator(root)) {
            if (!entry.is_regular_file() || !IsSourceFile(entry.path())) {
                continue;
            }
            const std::string relative_path =
                fs::relative(entry.path(), repo_root).generic_string();
            if (relative_path == authority) {
                continue;
            }
            std::ifstream in(entry.path());
            REQUIRE(in.is_open());
            std::string line;
            size_t line_number = 0;
            while (std::getline(in, line)) {
                ++line_number;
                for (const auto& needle : needles) {
                    if (line.find(needle) != std::string::npos &&
                        !IsAllowedTaxonomyHit(relative_path, line)) {
                        unexpected_hits.push_back(TaxonomyHit{
                            relative_path, line_number, needle, line});
                    }
                }
            }
        }
    }

    INFO("Reason-code string literals outside the authority header "
         "(derive them via BackendFallbackReasonName instead):" +
         FormatHits(unexpected_hits));
    REQUIRE(unexpected_hits.empty());
}
