#include <catch2/catch_test_macros.hpp>

#include "algorithms/arrayfire_host_materialization.h"

#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace {

namespace fs = std::filesystem;

struct DirectHostHit {
    std::string path;
    size_t line = 0;
    std::string text;
};

fs::path FindRepoRoot() {
    auto dir = fs::current_path();
    while (!dir.empty()) {
        if (fs::exists(dir / "cyxwiz-backend" / "src") &&
            fs::exists(dir / "cyxwiz-engine" / "src") &&
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

bool IsProductionSource(const fs::path& path) {
    const std::string extension = path.extension().string();
    return extension == ".cpp" || extension == ".h" ||
           extension == ".hpp";
}

std::string FormatHits(const std::vector<DirectHostHit>& hits) {
    std::ostringstream out;
    for (const auto& hit : hits) {
        out << '\n' << hit.path << ':' << hit.line << ": " << hit.text;
    }
    return out.str();
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
int g_observer_calls = 0;
cyxwiz::ArrayFireHostSyncEvent g_observed_event;

void CaptureHostSync(const cyxwiz::ArrayFireHostSyncEvent& event) {
    ++g_observer_calls;
    g_observed_event = event;
}
#endif

} // namespace

TEST_CASE("Production ArrayFire host calls stay behind attributed owners",
          "[arrayfire][host_sync][source_scan]") {
    const fs::path repo_root = FindRepoRoot();
    const std::vector<fs::path> roots = {
        repo_root / "cyxwiz-backend" / "src",
        repo_root / "cyxwiz-engine" / "src",
    };
    const std::vector<std::string> allowed = {
        "cyxwiz-backend/src/algorithms/arrayfire_host_materialization.cpp",
        "cyxwiz-backend/src/core/tensor.cpp",
    };
    const std::regex direct_host("(\\.|->)host\\s*\\(");

    std::vector<DirectHostHit> unexpected;
    size_t shared_boundary_hits = 0;
    size_t tensor_boundary_hits = 0;
    for (const auto& root : roots) {
        REQUIRE(fs::exists(root));
        for (const auto& entry : fs::recursive_directory_iterator(root)) {
            if (!entry.is_regular_file() ||
                !IsProductionSource(entry.path())) {
                continue;
            }
            const std::string relative =
                fs::relative(entry.path(), repo_root).generic_string();
            std::ifstream input(entry.path());
            REQUIRE(input.is_open());
            std::string line;
            size_t line_number = 0;
            while (std::getline(input, line)) {
                ++line_number;
                if (!std::regex_search(line, direct_host)) {
                    continue;
                }
                if (relative == allowed[0]) {
                    ++shared_boundary_hits;
                } else if (relative == allowed[1]) {
                    ++tensor_boundary_hits;
                } else {
                    unexpected.push_back({relative, line_number, line});
                }
            }
        }
    }

    INFO("Unattributed direct af::array::host calls:" +
         FormatHits(unexpected));
    CHECK(shared_boundary_hits == 1);
    CHECK(tensor_boundary_hits > 0);
    REQUIRE(unexpected.empty());
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
TEST_CASE("Shared ArrayFire host boundary reports complete attribution",
          "[arrayfire][host_sync][materialization]") {
    CHECK(std::string(cyxwiz::ArrayFireHostSyncCategoryName(
              cyxwiz::ArrayFireHostSyncCategory::AlgorithmCpuPath)) ==
          "algorithm_cpu_path");
    CHECK(std::string(cyxwiz::ArrayFireHostSyncCategoryName(
              cyxwiz::ArrayFireHostSyncCategory::OutputMaterialization)) ==
          "output_materialization");
    CHECK(std::string(cyxwiz::ArrayFireHostSyncCategoryName(
              cyxwiz::ArrayFireHostSyncCategory::LossInputValidation)) ==
          "loss_input_validation");
    g_observer_calls = 0;
    g_observed_event = {};
    std::vector<float> output(6, 0.0f);
    af::array source = af::iota(af::dim4(2, 3), af::dim4(1), f32);

    {
        const cyxwiz::ScopedArrayFireHostSyncObserver observer(
            &CaptureHostSync);
        cyxwiz::MaterializeArrayFireToHost(
            source,
            output.data(),
            cyxwiz::ArrayFireHostSyncCategory::OutputMaterialization,
            "UnitTest::MaterializeArray",
            "arrayfire_column_major",
            "test_output_boundary",
            "case=complete_attribution");
    }

    REQUIRE(g_observer_calls == 1);
    CHECK(g_observed_event.operation_name == "af::array::host");
    CHECK(g_observed_event.reason_code == "test_output_boundary");
    CHECK(g_observed_event.attribution_category ==
          "output_materialization");
    CHECK(g_observed_event.attribution_operation ==
          "UnitTest::MaterializeArray");
    CHECK(g_observed_event.tensor_shape == std::vector<size_t>{2, 3});
    CHECK(g_observed_event.tensor_dtype == "float32");
    CHECK(g_observed_event.tensor_layout == "arrayfire_column_major");
    CHECK(g_observed_event.context == "case=complete_attribution");
    CHECK(g_observed_event.bytes == 6 * sizeof(float));
    CHECK_FALSE(g_observed_event.selected_backend.empty());
    CHECK(output[0] == 0.0f);
    CHECK(output[5] == 5.0f);

    CHECK_THROWS_AS(
        cyxwiz::MaterializeArrayFireToHost(
            source,
            output.data(),
            cyxwiz::ArrayFireHostSyncCategory::Unknown,
            "UnitTest::MissingCategory",
            "arrayfire_native"),
        std::invalid_argument);
}
#endif
