/**
 * MNIST ONNX Model Test
 *
 * Tests the mnist.onnx model using our ONNXLoader.
 * Cross-platform: Windows, Linux, macOS
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <random>
#include <chrono>

#ifdef CYXWIZ_HAS_ONNX
#include <onnxruntime_cxx_api.h>
#endif

// Simple tensor class for testing
struct TestTensor {
    std::vector<float> data;
    std::vector<int64_t> shape;

    size_t NumElements() const {
        size_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
};

// Load MNIST test images (simplified - just creates random data for testing)
std::vector<TestTensor> CreateTestImages(int num_samples) {
    std::vector<TestTensor> images;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < num_samples; ++i) {
        TestTensor img;
        img.shape = {1, 1, 28, 28};  // [batch, channel, height, width]
        img.data.resize(1 * 1 * 28 * 28);
        for (auto& v : img.data) {
            v = dist(rng);
        }
        images.push_back(std::move(img));
    }
    return images;
}

#ifdef CYXWIZ_HAS_ONNX
int RunONNXTest(const std::string& model_path, int num_samples, bool force_cpu, bool disable_graph_opt = false) {
    std::cout << "============================================\n";
    std::cout << "MNIST ONNX Test\n";
    std::cout << "============================================\n";
    std::cout << "Model: " << model_path << "\n";
    std::cout << "Samples: " << num_samples << "\n";
    if (disable_graph_opt) {
        std::cout << "Graph Optimization: DISABLED (avoids FusedConv)\n";
    }
    std::cout << "\n";

    // Check file exists
    std::ifstream file(model_path);
    if (!file.good()) {
        std::cerr << "ERROR: Model file not found: " << model_path << "\n";
        return 1;
    }
    file.close();

    try {
        // Initialize ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MNISTTest");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);

        // Disable graph optimizations if requested (avoids FusedConv which can fail on older GPUs)
        if (disable_graph_opt) {
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
        }

        // Try CUDA first, fall back to CPU
        bool use_cuda = false;
        if (!force_cpu) {
            try {
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = 0;

                // Use exhaustive algorithm search for better compatibility with older GPUs
                // This helps with Pascal architecture (GTX 10xx series, compute 6.x)
                // Options: OrtCudnnConvAlgoSearchExhaustive (0), OrtCudnnConvAlgoSearchHeuristic (1),
                //          OrtCudnnConvAlgoSearchDefault (2)
                cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;

                // Use default memory arena (more conservative memory usage)
                cuda_options.arena_extend_strategy = 0;  // kNextPowerOfTwo

                // Disable memory pattern optimization for compatibility
                cuda_options.do_copy_in_default_stream = 1;

                session_options.AppendExecutionProvider_CUDA(cuda_options);
                use_cuda = true;
                std::cout << "Execution Provider: CUDA (device 0)\n";
                std::cout << "  cuDNN Conv Algo: Heuristic (for Pascal GPU compatibility)\n";
            } catch (const Ort::Exception& e) {
                std::cout << "CUDA not available, using CPU\n";
                std::cout << "  Reason: " << e.what() << "\n";
            }
        } else {
            std::cout << "Execution Provider: CPU (forced)\n";
        }

        // Load model
        std::cout << "Loading model...\n";
        auto start = std::chrono::high_resolution_clock::now();

#ifdef _WIN32
        std::wstring wide_path(model_path.begin(), model_path.end());
        Ort::Session session(env, wide_path.c_str(), session_options);
#else
        Ort::Session session(env, model_path.c_str(), session_options);
#endif

        auto load_time = std::chrono::high_resolution_clock::now() - start;
        std::cout << "Model loaded in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(load_time).count()
                  << " ms\n";

        // Get input/output info
        Ort::AllocatorWithDefaultOptions allocator;

        auto input_name = session.GetInputNameAllocated(0, allocator);
        auto output_name = session.GetOutputNameAllocated(0, allocator);

        auto input_info = session.GetInputTypeInfo(0);
        auto input_tensor_info = input_info.GetTensorTypeAndShapeInfo();
        auto input_shape = input_tensor_info.GetShape();

        std::cout << "\nModel Info:\n";
        std::cout << "  Input: " << input_name.get() << " [";
        for (size_t i = 0; i < input_shape.size(); ++i) {
            std::cout << input_shape[i];
            if (i < input_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
        std::cout << "  Output: " << output_name.get() << "\n\n";

        // Create test data
        std::cout << "Running inference on " << num_samples << " samples...\n";
        auto images = CreateTestImages(num_samples);

        const char* input_names[] = {input_name.get()};
        const char* output_names[] = {output_name.get()};

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<double> latencies;

        for (int i = 0; i < num_samples; ++i) {
            auto& img = images[i];

            // Handle dynamic batch dimension
            std::vector<int64_t> actual_shape = input_shape;
            if (actual_shape[0] == -1) actual_shape[0] = 1;  // Set batch to 1

            size_t num_elements = 1;
            for (auto d : actual_shape) num_elements *= d;

            // Resize data if needed
            if (img.data.size() != num_elements) {
                img.data.resize(num_elements, 0.5f);
            }

            auto input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                img.data.data(),
                img.data.size(),
                actual_shape.data(),
                actual_shape.size()
            );

            auto infer_start = std::chrono::high_resolution_clock::now();

            auto outputs = session.Run(
                Ort::RunOptions{nullptr},
                input_names, &input_tensor, 1,
                output_names, 1
            );

            auto infer_time = std::chrono::high_resolution_clock::now() - infer_start;
            double ms = std::chrono::duration<double, std::milli>(infer_time).count();
            latencies.push_back(ms);

            // Get prediction
            auto& output = outputs[0];
            auto output_info = output.GetTensorTypeAndShapeInfo();
            auto output_shape = output_info.GetShape();
            const float* logits = output.GetTensorData<float>();

            int predicted = 0;
            float max_val = logits[0];
            size_t num_classes = output_shape.back();
            for (size_t j = 1; j < num_classes; ++j) {
                if (logits[j] > max_val) {
                    max_val = logits[j];
                    predicted = static_cast<int>(j);
                }
            }

            // Progress
            if ((i + 1) % 10 == 0 || i == num_samples - 1) {
                std::cout << "  Progress: " << (i + 1) << "/" << num_samples
                          << " | Last prediction: " << predicted << "\n";
            }
        }

        // Calculate stats
        double total_time = 0;
        for (auto t : latencies) total_time += t;
        double avg_latency = total_time / num_samples;

        std::cout << "\n============================================\n";
        std::cout << "Results:\n";
        std::cout << "  Samples: " << num_samples << "\n";
        std::cout << "  Avg Latency: " << avg_latency << " ms\n";
        std::cout << "  Total Time: " << total_time << " ms\n";
        std::cout << "  Provider: " << (use_cuda ? "CUDA" : "CPU") << "\n";
        std::cout << "============================================\n";

        return 0;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
#endif

int main(int argc, char* argv[]) {
#ifdef CYXWIZ_HAS_ONNX
    std::string model_path = "mnist.onnx";
    int num_samples = 50;
    bool force_cpu = false;
    bool disable_graph_opt = false;

    // Parse args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--samples" && i + 1 < argc) {
            num_samples = std::stoi(argv[++i]);
        } else if (arg == "--cpu") {
            force_cpu = true;
        } else if (arg == "--no-opt") {
            disable_graph_opt = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --model PATH    Path to ONNX model (default: mnist.onnx)\n";
            std::cout << "  --samples N     Number of test samples (default: 50)\n";
            std::cout << "  --cpu           Force CPU execution (skip CUDA)\n";
            std::cout << "  --no-opt        Disable graph optimizations (avoids FusedConv)\n";
            std::cout << "  --help          Show this help\n";
            return 0;
        }
    }

    return RunONNXTest(model_path, num_samples, force_cpu, disable_graph_opt);
#else
    std::cerr << "ERROR: ONNX support not compiled.\n";
    std::cerr << "Rebuild with CYXWIZ_ENABLE_ONNX=ON\n";
    return 1;
#endif
}
