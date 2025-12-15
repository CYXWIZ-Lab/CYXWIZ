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

#ifdef CYXWIZ_HAS_ONNX_EXPORT
#include <onnx/onnx_pb.h>
#endif

namespace fs = std::filesystem;

namespace {

// Test data directory
const std::string TEST_DATA_DIR = "test_onnx_data";

// Helper: Create a simple MLP ONNX model programmatically
#ifdef CYXWIZ_HAS_ONNX_EXPORT
bool CreateTestONNXModel(const std::string& path, int in_features, int hidden, int out_features) {
    try {
        onnx::ModelProto model;
        model.set_ir_version(8);
        model.set_producer_name("CyxWiz-Test");
        model.set_producer_version("1.0");

        auto* opset = model.add_opset_import();
        opset->set_domain("");
        opset->set_version(17);

        auto* graph = model.mutable_graph();
        graph->set_name("test_mlp");

        // Input
        auto* input = graph->add_input();
        input->set_name("input");
        auto* input_type = input->mutable_type()->mutable_tensor_type();
        input_type->set_elem_type(onnx::TensorProto::FLOAT);
        auto* input_shape = input_type->mutable_shape();
        input_shape->add_dim()->set_dim_param("batch");
        input_shape->add_dim()->set_dim_value(in_features);

        // Random number generator for weights
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

        // Layer 1: Linear (in_features -> hidden)
        {
            // Weight initializer
            auto* w1 = graph->add_initializer();
            w1->set_name("fc1.weight");
            w1->set_data_type(onnx::TensorProto::FLOAT);
            w1->add_dims(hidden);
            w1->add_dims(in_features);
            for (int i = 0; i < hidden * in_features; ++i) {
                w1->add_float_data(dist(rng));
            }

            // Bias initializer
            auto* b1 = graph->add_initializer();
            b1->set_name("fc1.bias");
            b1->set_data_type(onnx::TensorProto::FLOAT);
            b1->add_dims(hidden);
            for (int i = 0; i < hidden; ++i) {
                b1->add_float_data(dist(rng));
            }

            // Gemm node
            auto* gemm1 = graph->add_node();
            gemm1->set_name("Gemm_0");
            gemm1->set_op_type("Gemm");
            gemm1->add_input("input");
            gemm1->add_input("fc1.weight");
            gemm1->add_input("fc1.bias");
            gemm1->add_output("gemm1_out");

            auto* alpha = gemm1->add_attribute();
            alpha->set_name("alpha");
            alpha->set_f(1.0f);
            alpha->set_type(onnx::AttributeProto::FLOAT);

            auto* beta = gemm1->add_attribute();
            beta->set_name("beta");
            beta->set_f(1.0f);
            beta->set_type(onnx::AttributeProto::FLOAT);

            auto* transB = gemm1->add_attribute();
            transB->set_name("transB");
            transB->set_i(1);
            transB->set_type(onnx::AttributeProto::INT);
        }

        // ReLU
        {
            auto* relu = graph->add_node();
            relu->set_name("Relu_0");
            relu->set_op_type("Relu");
            relu->add_input("gemm1_out");
            relu->add_output("relu_out");
        }

        // Layer 2: Linear (hidden -> out_features)
        {
            auto* w2 = graph->add_initializer();
            w2->set_name("fc2.weight");
            w2->set_data_type(onnx::TensorProto::FLOAT);
            w2->add_dims(out_features);
            w2->add_dims(hidden);
            for (int i = 0; i < out_features * hidden; ++i) {
                w2->add_float_data(dist(rng));
            }

            auto* b2 = graph->add_initializer();
            b2->set_name("fc2.bias");
            b2->set_data_type(onnx::TensorProto::FLOAT);
            b2->add_dims(out_features);
            for (int i = 0; i < out_features; ++i) {
                b2->add_float_data(dist(rng));
            }

            auto* gemm2 = graph->add_node();
            gemm2->set_name("Gemm_1");
            gemm2->set_op_type("Gemm");
            gemm2->add_input("relu_out");
            gemm2->add_input("fc2.weight");
            gemm2->add_input("fc2.bias");
            gemm2->add_output("output");

            auto* alpha = gemm2->add_attribute();
            alpha->set_name("alpha");
            alpha->set_f(1.0f);
            alpha->set_type(onnx::AttributeProto::FLOAT);

            auto* beta = gemm2->add_attribute();
            beta->set_name("beta");
            beta->set_f(1.0f);
            beta->set_type(onnx::AttributeProto::FLOAT);

            auto* transB = gemm2->add_attribute();
            transB->set_name("transB");
            transB->set_i(1);
            transB->set_type(onnx::AttributeProto::INT);
        }

        // Output
        auto* output = graph->add_output();
        output->set_name("output");
        auto* output_type = output->mutable_type()->mutable_tensor_type();
        output_type->set_elem_type(onnx::TensorProto::FLOAT);
        auto* output_shape = output_type->mutable_shape();
        output_shape->add_dim()->set_dim_param("batch");
        output_shape->add_dim()->set_dim_value(out_features);

        // Write to file
        std::ofstream file(path, std::ios::binary);
        if (!file) return false;
        return model.SerializeToOstream(&file);

    } catch (...) {
        return false;
    }
}
#endif

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

// Helper: Create test data directory
void SetupTestDataDir() {
    if (!fs::exists(TEST_DATA_DIR)) {
        fs::create_directory(TEST_DATA_DIR);
    }
}

// Helper: Cleanup test data directory
void CleanupTestDataDir() {
    if (fs::exists(TEST_DATA_DIR)) {
        fs::remove_all(TEST_DATA_DIR);
    }
}

} // anonymous namespace

// ============================================================================
// Test Cases
// ============================================================================

#ifdef CYXWIZ_HAS_ONNX

TEST_CASE("ONNXLoader - Basic Loading", "[onnx][loader]") {
    using namespace cyxwiz::servernode;

    SetupTestDataDir();

    SECTION("Load valid ONNX model") {
#ifdef CYXWIZ_HAS_ONNX_EXPORT
        std::string model_path = TEST_DATA_DIR + "/test_mlp.onnx";
        REQUIRE(CreateTestONNXModel(model_path, 784, 128, 10));

        ONNXLoader loader;
        REQUIRE_FALSE(loader.IsLoaded());

        bool loaded = loader.Load(model_path);
        REQUIRE(loaded);
        REQUIRE(loader.IsLoaded());
        REQUIRE(loader.GetFormat() == "onnx");

        loader.Unload();
        fs::remove(model_path);
#else
        WARN("ONNX export not available - skipping model creation test");
#endif
    }

    SECTION("Reject invalid file path") {
        ONNXLoader loader;
        bool loaded = loader.Load("nonexistent_model.onnx");
        REQUIRE_FALSE(loaded);
        REQUIRE_FALSE(loader.IsLoaded());
    }

    SECTION("Reject non-ONNX file") {
        std::string bad_file = TEST_DATA_DIR + "/not_onnx.txt";
        std::ofstream file(bad_file);
        file << "This is not an ONNX model";
        file.close();

        ONNXLoader loader;
        bool loaded = loader.Load(bad_file);
        REQUIRE_FALSE(loaded);
        REQUIRE_FALSE(loader.IsLoaded());

        fs::remove(bad_file);
    }

    CleanupTestDataDir();
}

TEST_CASE("ONNXLoader - I/O Specs Extraction", "[onnx][loader]") {
    using namespace cyxwiz::servernode;

#ifdef CYXWIZ_HAS_ONNX_EXPORT
    SetupTestDataDir();
    std::string model_path = TEST_DATA_DIR + "/specs_test.onnx";
    REQUIRE(CreateTestONNXModel(model_path, 784, 128, 10));

    ONNXLoader loader;
    REQUIRE(loader.Load(model_path));

    SECTION("Extract input specs correctly") {
        auto input_specs = loader.GetInputSpecs();
        REQUIRE(input_specs.size() == 1);
        REQUIRE(input_specs[0].name == "input");
        REQUIRE(input_specs[0].dtype == "float32");

        // Shape should have 2 dimensions (batch, features)
        REQUIRE(input_specs[0].shape.size() == 2);
        REQUIRE(input_specs[0].shape[1] == 784);  // in_features
    }

    SECTION("Extract output specs correctly") {
        auto output_specs = loader.GetOutputSpecs();
        REQUIRE(output_specs.size() == 1);
        REQUIRE(output_specs[0].name == "output");
        REQUIRE(output_specs[0].dtype == "float32");

        REQUIRE(output_specs[0].shape.size() == 2);
        REQUIRE(output_specs[0].shape[1] == 10);  // out_features
    }

    SECTION("Handle dynamic batch dimensions") {
        auto input_specs = loader.GetInputSpecs();
        // First dimension should be dynamic (-1)
        REQUIRE(input_specs[0].shape[0] == -1);
    }

    loader.Unload();
    fs::remove(model_path);
    CleanupTestDataDir();
#else
    WARN("ONNX export not available - skipping I/O specs tests");
#endif
}

TEST_CASE("ONNXLoader - Inference", "[onnx][inference]") {
    using namespace cyxwiz::servernode;

#ifdef CYXWIZ_HAS_ONNX_EXPORT
    SetupTestDataDir();
    std::string model_path = TEST_DATA_DIR + "/infer_test.onnx";
    REQUIRE(CreateTestONNXModel(model_path, 784, 128, 10));

    ONNXLoader loader;
    REQUIRE(loader.Load(model_path));

    SECTION("Run inference with valid input") {
        // Create input tensor [1, 784]
        auto input = CreateRandomInput({1, 784});

        std::unordered_map<std::string, cyxwiz::Tensor> inputs;
        inputs["input"] = std::move(input);

        std::unordered_map<std::string, cyxwiz::Tensor> outputs;
        bool success = loader.Infer(inputs, outputs);

        REQUIRE(success);
        REQUIRE(outputs.size() >= 1);

        // Check output shape
        auto it = outputs.find("output");
        REQUIRE(it != outputs.end());
        auto shape = it->second.Shape();
        REQUIRE(shape.size() == 2);
        REQUIRE(shape[0] == 1);   // batch size
        REQUIRE(shape[1] == 10);  // output features
    }

    SECTION("Handle batch inference") {
        // Create batch input [4, 784]
        auto input = CreateRandomInput({4, 784});

        std::unordered_map<std::string, cyxwiz::Tensor> inputs;
        inputs["input"] = std::move(input);

        std::unordered_map<std::string, cyxwiz::Tensor> outputs;
        bool success = loader.Infer(inputs, outputs);

        REQUIRE(success);

        auto it = outputs.find("output");
        REQUIRE(it != outputs.end());
        auto shape = it->second.Shape();
        REQUIRE(shape[0] == 4);   // batch size
        REQUIRE(shape[1] == 10);  // output features
    }

    loader.Unload();
    fs::remove(model_path);
    CleanupTestDataDir();
#else
    WARN("ONNX export not available - skipping inference tests");
#endif
}

TEST_CASE("ONNXLoader - Resource Management", "[onnx][memory]") {
    using namespace cyxwiz::servernode;

#ifdef CYXWIZ_HAS_ONNX_EXPORT
    SetupTestDataDir();
    std::string model_path = TEST_DATA_DIR + "/resource_test.onnx";
    REQUIRE(CreateTestONNXModel(model_path, 784, 128, 10));

    SECTION("Unload releases resources") {
        ONNXLoader loader;
        REQUIRE(loader.Load(model_path));
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
        REQUIRE(loader.Load(model_path));
        REQUIRE(loader.IsLoaded());

        // After unloading
        loader.Unload();
        REQUIRE_FALSE(loader.IsLoaded());
    }

    SECTION("Memory usage tracking") {
        ONNXLoader loader;
        REQUIRE(loader.GetMemoryUsage() == 0);

        REQUIRE(loader.Load(model_path));
        uint64_t mem = loader.GetMemoryUsage();
        REQUIRE(mem > 0);  // Should have some memory usage

        loader.Unload();
        REQUIRE(loader.GetMemoryUsage() == 0);
    }

    fs::remove(model_path);
    CleanupTestDataDir();
#else
    WARN("ONNX export not available - skipping resource management tests");
#endif
}

TEST_CASE("ONNXLoader - Multiple Load/Unload Cycles", "[onnx][lifecycle]") {
    using namespace cyxwiz::servernode;

#ifdef CYXWIZ_HAS_ONNX_EXPORT
    SetupTestDataDir();
    std::string model_path = TEST_DATA_DIR + "/lifecycle_test.onnx";
    REQUIRE(CreateTestONNXModel(model_path, 100, 50, 10));

    ONNXLoader loader;

    for (int i = 0; i < 3; ++i) {
        REQUIRE(loader.Load(model_path));
        REQUIRE(loader.IsLoaded());

        // Run inference
        auto input = CreateRandomInput({1, 100});
        std::unordered_map<std::string, cyxwiz::Tensor> inputs;
        inputs["input"] = std::move(input);
        std::unordered_map<std::string, cyxwiz::Tensor> outputs;

        REQUIRE(loader.Infer(inputs, outputs));

        loader.Unload();
        REQUIRE_FALSE(loader.IsLoaded());
    }

    fs::remove(model_path);
    CleanupTestDataDir();
#else
    WARN("ONNX export not available - skipping lifecycle tests");
#endif
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
