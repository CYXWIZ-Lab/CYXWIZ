/**
 * ONNX Loader Unit Tests
 *
 * Tests for ONNXLoader class in model_loader.cpp
 * Verifies loading, inference, and resource management for ONNX models.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>

#include "../src/model_loader.h"
#include <cyxwiz/tensor.h>
#include <cyxwiz/sequential.h>

#include <filesystem>
#include <fstream>
#include <random>
#include <cstring>

#ifdef CYXWIZ_HAS_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace fs = std::filesystem;

namespace {

#ifndef CYXWIZ_ONNX_TEST_MODEL_PATH
#define CYXWIZ_ONNX_TEST_MODEL_PATH "mnist.onnx"
#endif

const std::string TEST_MODEL_PATH = CYXWIZ_ONNX_TEST_MODEL_PATH;

bool LoadFixtureModel(cyxwiz::servernode::ONNXLoader& loader) {
    loader.SetForceCPU(true);
    return loader.Load(TEST_MODEL_PATH);
}

// Helper: Create random input tensor
cyxwiz::Tensor CreateRandomInput(const std::vector<size_t>& shape) {
    cyxwiz::Tensor tensor(shape, cyxwiz::DataType::Float32);
    float* data = static_cast<float*>(tensor.Data());

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < tensor.NumElements(); ++i) {
        data[i] = dist(rng);
    }
    return tensor;
}

std::vector<size_t> ConcreteShape(const cyxwiz::servernode::TensorSpec& spec,
                                  size_t requested_batch = 1) {
    std::vector<size_t> shape;
    shape.reserve(spec.shape.size());

    for (size_t i = 0; i < spec.shape.size(); ++i) {
        const int64_t dim = spec.shape[i];
        if (dim > 0) {
            shape.push_back(static_cast<size_t>(dim));
        } else {
            shape.push_back(i == 0 ? requested_batch : 1);
        }
    }
    return shape;
}

cyxwiz::Tensor CreateInputForSpec(const cyxwiz::servernode::TensorSpec& spec,
                                  size_t requested_batch = 1) {
    return CreateRandomInput(ConcreteShape(spec, requested_batch));
}

const cyxwiz::Tensor* FindOutput(
    const std::unordered_map<std::string, cyxwiz::Tensor>& outputs,
    const std::string& preferred_name) {
    auto it = outputs.find(preferred_name);
    if (it != outputs.end()) {
        return &it->second;
    }
    if (!outputs.empty()) {
        return &outputs.begin()->second;
    }
    return nullptr;
}

} // anonymous namespace

// ============================================================================
// Test Cases
// ============================================================================

#ifdef CYXWIZ_HAS_ONNX

TEST_CASE("ONNXLoader - Basic Loading", "[onnx][loader]") {
    using namespace cyxwiz::servernode;

    SECTION("Load valid ONNX model") {
        INFO("Fixture path: " << TEST_MODEL_PATH);
        REQUIRE(fs::exists(TEST_MODEL_PATH));

        ONNXLoader loader;
        REQUIRE_FALSE(loader.IsLoaded());

        bool loaded = LoadFixtureModel(loader);
        REQUIRE(loaded);
        REQUIRE(loader.IsLoaded());
        REQUIRE(loader.GetFormat() == "onnx");

        loader.Unload();
    }

    SECTION("Reject invalid file path") {
        ONNXLoader loader;
        bool loaded = loader.Load("nonexistent_model.onnx");
        REQUIRE_FALSE(loaded);
        REQUIRE_FALSE(loader.IsLoaded());
    }

    SECTION("Reject non-ONNX file") {
        auto bad_file = fs::temp_directory_path() / "cyxwiz_not_onnx.txt";
        std::ofstream file(bad_file);
        file << "This is not an ONNX model";
        file.close();

        ONNXLoader loader;
        loader.SetForceCPU(true);
        bool loaded = loader.Load(bad_file.string());
        REQUIRE_FALSE(loaded);
        REQUIRE_FALSE(loader.IsLoaded());

        fs::remove(bad_file);
    }
}

TEST_CASE("ONNXLoader - I/O Specs Extraction", "[onnx][loader]") {
    using namespace cyxwiz::servernode;

    INFO("Fixture path: " << TEST_MODEL_PATH);
    REQUIRE(fs::exists(TEST_MODEL_PATH));

    ONNXLoader loader;
    REQUIRE(LoadFixtureModel(loader));

    SECTION("Extract input specs correctly") {
        auto input_specs = loader.GetInputSpecs();
        REQUIRE_FALSE(input_specs.empty());
        REQUIRE_FALSE(input_specs[0].name.empty());
        REQUIRE(input_specs[0].dtype == "float32");
        REQUIRE_FALSE(input_specs[0].shape.empty());
    }

    SECTION("Extract output specs correctly") {
        auto output_specs = loader.GetOutputSpecs();
        REQUIRE_FALSE(output_specs.empty());
        REQUIRE_FALSE(output_specs[0].name.empty());
        REQUIRE(output_specs[0].dtype == "float32");
        REQUIRE_FALSE(output_specs[0].shape.empty());
    }

    SECTION("Expose concrete or dynamic input dimensions") {
        auto input_specs = loader.GetInputSpecs();
        REQUIRE_FALSE(input_specs.empty());
        for (auto dim : input_specs[0].shape) {
            REQUIRE(dim != 0);
        }
    }

    loader.Unload();
}

TEST_CASE("ONNXLoader - Inference", "[onnx][inference]") {
    using namespace cyxwiz::servernode;

    INFO("Fixture path: " << TEST_MODEL_PATH);
    REQUIRE(fs::exists(TEST_MODEL_PATH));

    ONNXLoader loader;
    REQUIRE(LoadFixtureModel(loader));

    SECTION("Run inference with valid input") {
        auto input_specs = loader.GetInputSpecs();
        auto output_specs = loader.GetOutputSpecs();
        REQUIRE_FALSE(input_specs.empty());
        REQUIRE_FALSE(output_specs.empty());

        auto input = CreateInputForSpec(input_specs[0]);

        std::unordered_map<std::string, cyxwiz::Tensor> inputs;
        inputs[input_specs[0].name] = std::move(input);

        std::unordered_map<std::string, cyxwiz::Tensor> outputs;
        bool success = loader.Infer(inputs, outputs);

        REQUIRE(success);
        REQUIRE_FALSE(outputs.empty());

        const cyxwiz::Tensor* output = FindOutput(outputs, output_specs[0].name);
        REQUIRE(output != nullptr);
        REQUIRE_FALSE(output->Shape().empty());
        REQUIRE(output->NumElements() > 0);
    }

    SECTION("Handle fixture-compatible batch inference") {
        auto input_specs = loader.GetInputSpecs();
        auto output_specs = loader.GetOutputSpecs();
        REQUIRE_FALSE(input_specs.empty());
        REQUIRE_FALSE(output_specs.empty());

        const size_t requested_batch =
            !input_specs[0].shape.empty() && input_specs[0].shape[0] <= 0 ? 4 : 1;
        auto input = CreateInputForSpec(input_specs[0], requested_batch);

        std::unordered_map<std::string, cyxwiz::Tensor> inputs;
        inputs[input_specs[0].name] = std::move(input);

        std::unordered_map<std::string, cyxwiz::Tensor> outputs;
        bool success = loader.Infer(inputs, outputs);

        REQUIRE(success);

        const cyxwiz::Tensor* output = FindOutput(outputs, output_specs[0].name);
        REQUIRE(output != nullptr);
        REQUIRE_FALSE(output->Shape().empty());
        REQUIRE(output->NumElements() > 0);
    }

    loader.Unload();
}

TEST_CASE("ONNXLoader - Resource Management", "[onnx][memory]") {
    using namespace cyxwiz::servernode;

    INFO("Fixture path: " << TEST_MODEL_PATH);
    REQUIRE(fs::exists(TEST_MODEL_PATH));

    SECTION("Unload releases resources") {
        ONNXLoader loader;
        REQUIRE(LoadFixtureModel(loader));
        REQUIRE(loader.IsLoaded());
        REQUIRE(loader.GetMemoryUsage() > 0);

        loader.Unload();
        REQUIRE_FALSE(loader.IsLoaded());
        REQUIRE(loader.GetMemoryUsage() == 0);
        REQUIRE(loader.GetInputSpecs().empty());
        REQUIRE(loader.GetOutputSpecs().empty());
    }

    SECTION("IsLoaded returns correct state") {
        ONNXLoader loader;

        // Initially not loaded
        REQUIRE_FALSE(loader.IsLoaded());

        // After loading
        REQUIRE(LoadFixtureModel(loader));
        REQUIRE(loader.IsLoaded());

        // After unloading
        loader.Unload();
        REQUIRE_FALSE(loader.IsLoaded());
    }

    SECTION("Memory usage tracking") {
        ONNXLoader loader;
        REQUIRE(loader.GetMemoryUsage() == 0);

        REQUIRE(LoadFixtureModel(loader));
        uint64_t mem = loader.GetMemoryUsage();
        REQUIRE(mem > 0);  // Should have some memory usage

        loader.Unload();
        REQUIRE(loader.GetMemoryUsage() == 0);
    }
}

TEST_CASE("ONNXLoader - Multiple Load/Unload Cycles", "[onnx][lifecycle]") {
    using namespace cyxwiz::servernode;

    INFO("Fixture path: " << TEST_MODEL_PATH);
    REQUIRE(fs::exists(TEST_MODEL_PATH));

    ONNXLoader loader;

    for (int i = 0; i < 3; ++i) {
        REQUIRE(LoadFixtureModel(loader));
        REQUIRE(loader.IsLoaded());

        // Run inference
        auto input_specs = loader.GetInputSpecs();
        REQUIRE_FALSE(input_specs.empty());
        auto input = CreateInputForSpec(input_specs[0]);

        std::unordered_map<std::string, cyxwiz::Tensor> inputs;
        inputs[input_specs[0].name] = std::move(input);
        std::unordered_map<std::string, cyxwiz::Tensor> outputs;

        REQUIRE(loader.Infer(inputs, outputs));
        REQUIRE_FALSE(outputs.empty());

        loader.Unload();
        REQUIRE_FALSE(loader.IsLoaded());
    }
}

#else  // !CYXWIZ_HAS_ONNX

TEST_CASE("ONNXLoader - ONNX Not Compiled", "[onnx][disabled]") {
    using namespace cyxwiz::servernode;

    SECTION("Load returns false when ONNX not available") {
        ONNXLoader loader;
        bool loaded = loader.Load("any_model.onnx");
        REQUIRE_FALSE(loaded);
    }
}

#endif  // CYXWIZ_HAS_ONNX

// ============================================================================
// ModelLoaderFactory Tests
// ============================================================================

TEST_CASE("ModelLoaderFactory - ONNX Support", "[factory][onnx]") {
    using namespace cyxwiz::servernode;

    SECTION("Factory creates ONNXLoader") {
        auto loader = ModelLoaderFactory::Create("onnx");
        REQUIRE(loader != nullptr);
        REQUIRE(loader->GetFormat() == "onnx");
    }

    SECTION("ONNX format is supported") {
        REQUIRE(ModelLoaderFactory::IsFormatSupported("onnx"));
        REQUIRE(ModelLoaderFactory::IsFormatSupported("ONNX"));
    }

    SECTION("Supported formats includes ONNX") {
        auto formats = ModelLoaderFactory::GetSupportedFormats();
        bool has_onnx = false;
        for (const auto& fmt : formats) {
            if (fmt == "onnx") {
                has_onnx = true;
                break;
            }
        }
        REQUIRE(has_onnx);
    }
}
