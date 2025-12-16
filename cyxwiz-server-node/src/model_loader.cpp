#include "model_loader.h"
#include <cyxwiz/sequential.h>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <fmt/ranges.h>  // For fmt::join
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <regex>

#ifdef CYXWIZ_HAS_ONNX
#include <onnxruntime_cxx_api.h>
#ifdef CYXWIZ_PLATFORM_WINDOWS
#include <codecvt>
#include <locale>
#endif
#endif

#ifdef CYXWIZ_HAS_GGUF
#include <llama.h>
#endif

#ifdef CYXWIZ_HAS_PYTORCH
#include <torch/script.h>
#include <torch/cuda.h>
#endif

namespace cyxwiz {
namespace servernode {

// ============================================================================
// ModelLoaderFactory Implementation
// ============================================================================

std::unique_ptr<ModelLoader> ModelLoaderFactory::Create(const std::string& format) {
    // Convert format to lowercase
    std::string fmt = format;
    std::transform(fmt.begin(), fmt.end(), fmt.begin(), ::tolower);

    if (fmt == "onnx") {
        return std::make_unique<ONNXLoader>();
    } else if (fmt == "gguf") {
        return std::make_unique<GGUFLoader>();
    } else if (fmt == "pytorch" || fmt == "pt" || fmt == "pth") {
        return std::make_unique<PyTorchLoader>();
    } else if (fmt == "cyxmodel" || fmt == "cyxwiz") {
        return std::make_unique<CyxWizLoader>();
    }

    spdlog::error("Unsupported model format: {}", format);
    return nullptr;
}

bool ModelLoaderFactory::IsFormatSupported(const std::string& format) {
    std::string fmt = format;
    std::transform(fmt.begin(), fmt.end(), fmt.begin(), ::tolower);

    return fmt == "onnx" || fmt == "gguf" ||
           fmt == "pytorch" || fmt == "pt" || fmt == "pth" ||
           fmt == "cyxmodel" || fmt == "cyxwiz";
}

std::vector<std::string> ModelLoaderFactory::GetSupportedFormats() {
    return {"onnx", "gguf", "pytorch", "cyxmodel"};
}

// ============================================================================
// ONNXLoader Implementation
// ============================================================================

class ONNXLoader::Impl {
public:
    bool loaded = false;
    std::string model_path;
    std::vector<TensorSpec> input_specs;
    std::vector<TensorSpec> output_specs;
    uint64_t memory_usage = 0;
    bool force_cpu = false;  // User can force CPU execution

#ifdef CYXWIZ_HAS_ONNX
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::SessionOptions> session_options;
    Ort::AllocatorWithDefaultOptions allocator;

    // Cached I/O names for inference
    std::vector<std::string> input_names_str;
    std::vector<std::string> output_names_str;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    bool using_cuda = false;
    int cuda_device_id = 0;
    bool cuda_inference_failed = false;  // Track if CUDA inference failed (for fallback)

    // Helper to convert ONNX element type to string
    static std::string OnnxTypeToString(ONNXTensorElementDataType type) {
        switch (type) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float32";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return "float64";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return "int8";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return "int16";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "int32";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return "uint8";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return "uint16";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return "uint32";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: return "uint64";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return "bool";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: return "string";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return "bfloat16";
            default: return "unknown";
        }
    }

    // Extract tensor specs from session
    void ExtractIOSpecs() {
        if (!session) return;

        input_specs.clear();
        output_specs.clear();
        input_names_str.clear();
        output_names_str.clear();
        input_names.clear();
        output_names.clear();

        // Get input info
        size_t num_inputs = session->GetInputCount();
        for (size_t i = 0; i < num_inputs; ++i) {
            auto name_ptr = session->GetInputNameAllocated(i, allocator);
            std::string name = name_ptr.get();
            input_names_str.push_back(name);

            auto type_info = session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            auto elem_type = tensor_info.GetElementType();

            TensorSpec spec;
            spec.name = name;
            spec.shape = shape;
            spec.dtype = OnnxTypeToString(elem_type);
            input_specs.push_back(spec);

            spdlog::debug("ONNX input[{}]: {} shape=[{}] dtype={}",
                         i, name,
                         fmt::join(shape, ","),
                         spec.dtype);
        }

        // Get output info
        size_t num_outputs = session->GetOutputCount();
        for (size_t i = 0; i < num_outputs; ++i) {
            auto name_ptr = session->GetOutputNameAllocated(i, allocator);
            std::string name = name_ptr.get();
            output_names_str.push_back(name);

            auto type_info = session->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            auto elem_type = tensor_info.GetElementType();

            TensorSpec spec;
            spec.name = name;
            spec.shape = shape;
            spec.dtype = OnnxTypeToString(elem_type);
            output_specs.push_back(spec);

            spdlog::debug("ONNX output[{}]: {} shape=[{}] dtype={}",
                         i, name,
                         fmt::join(shape, ","),
                         spec.dtype);
        }

        // Cache const char* pointers for Run()
        for (const auto& name : input_names_str) {
            input_names.push_back(name.c_str());
        }
        for (const auto& name : output_names_str) {
            output_names.push_back(name.c_str());
        }
    }
#endif
};

ONNXLoader::ONNXLoader() : impl_(std::make_unique<Impl>()) {
    spdlog::debug("ONNXLoader created");
}

ONNXLoader::~ONNXLoader() {
    Unload();
}

bool ONNXLoader::Load(const std::string& model_path) {
    spdlog::info("Loading ONNX model from: {}", model_path);

#ifndef CYXWIZ_HAS_ONNX
    spdlog::error("ONNX Runtime support not compiled. Rebuild with CYXWIZ_ENABLE_ONNX=ON");
    return false;
#else
    try {
        // Check file exists
        if (!std::filesystem::exists(model_path)) {
            spdlog::error("ONNX model file not found: {}", model_path);
            return false;
        }

        impl_->model_path = model_path;

        // Create ONNX Runtime environment
        impl_->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "CyxWiz-ONNX");

        // Configure session options
        impl_->session_options = std::make_unique<Ort::SessionOptions>();
        impl_->session_options->SetIntraOpNumThreads(4);
        impl_->session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Try to use CUDA execution provider first (unless force_cpu is set)
        impl_->using_cuda = false;

        if (!impl_->force_cpu) {
            try {
                // Try CUDA EP
                // Note: Using Heuristic algorithm search for better compatibility with older GPUs
                // (Pascal architecture / GTX 10xx series / compute capability 6.x)
                OrtCUDAProviderOptions cuda_options{};
                cuda_options.device_id = impl_->cuda_device_id;
                cuda_options.arena_extend_strategy = 0;  // kNextPowerOfTwo
                cuda_options.gpu_mem_limit = SIZE_MAX;   // Use all available GPU memory
                cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
                cuda_options.do_copy_in_default_stream = 1;

                impl_->session_options->AppendExecutionProvider_CUDA(cuda_options);
                impl_->using_cuda = true;
                spdlog::info("ONNX: CUDA execution provider configured (device {})", impl_->cuda_device_id);
            } catch (const Ort::Exception& e) {
                spdlog::warn("ONNX: CUDA execution provider not available: {}", e.what());
                spdlog::info("ONNX: Falling back to CPU execution provider");
                // Reset session options to remove failed CUDA EP
                impl_->session_options = std::make_unique<Ort::SessionOptions>();
                impl_->session_options->SetIntraOpNumThreads(4);
                impl_->session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            }
        } else {
            spdlog::info("ONNX: Using CPU execution provider (forced)");
        }

        // Create session
#ifdef CYXWIZ_PLATFORM_WINDOWS
        // Windows: convert path to wide string for ONNX Runtime
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
        std::wstring wide_path = converter.from_bytes(model_path);
        impl_->session = std::make_unique<Ort::Session>(
            *impl_->env, wide_path.c_str(), *impl_->session_options);
#else
        impl_->session = std::make_unique<Ort::Session>(
            *impl_->env, model_path.c_str(), *impl_->session_options);
#endif

        // Extract input/output specifications
        impl_->ExtractIOSpecs();

        // Estimate memory usage (rough estimate based on model file size + overhead)
        auto file_size = std::filesystem::file_size(model_path);
        impl_->memory_usage = file_size * 2;  // Assume 2x for runtime overhead

        impl_->loaded = true;

        std::string device_str = impl_->using_cuda ?
            fmt::format("CUDA (device {})", impl_->cuda_device_id) : "CPU";
        spdlog::info("ONNX model loaded successfully on {}", device_str);
        spdlog::info("  Inputs: {}, Outputs: {}",
                     impl_->input_specs.size(), impl_->output_specs.size());

        return true;

    } catch (const Ort::Exception& e) {
        spdlog::error("ONNX Runtime error: {}", e.what());
        impl_->loaded = false;
        return false;
    } catch (const std::exception& e) {
        spdlog::error("Failed to load ONNX model: {}", e.what());
        impl_->loaded = false;
        return false;
    }
#endif
}

bool ONNXLoader::Infer(
    const std::unordered_map<std::string, cyxwiz::Tensor>& inputs,
    std::unordered_map<std::string, cyxwiz::Tensor>& outputs) {

    if (!impl_->loaded) {
        spdlog::error("ONNX model not loaded");
        return false;
    }

#ifndef CYXWIZ_HAS_ONNX
    spdlog::error("ONNX Runtime support not compiled");
    return false;
#else
    try {
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        // Prepare input tensors
        std::vector<Ort::Value> input_tensors;

        for (size_t i = 0; i < impl_->input_specs.size(); ++i) {
            const auto& spec = impl_->input_specs[i];

            // Find matching input from user-provided map
            auto it = inputs.find(spec.name);
            if (it == inputs.end()) {
                // Try "input" as fallback for single-input models
                if (impl_->input_specs.size() == 1) {
                    it = inputs.find("input");
                }
                if (it == inputs.end() && !inputs.empty()) {
                    it = inputs.begin();  // Use first available
                }
                if (it == inputs.end()) {
                    spdlog::error("ONNX: Input '{}' not provided", spec.name);
                    return false;
                }
            }

            const cyxwiz::Tensor& tensor = it->second;

            // Get shape (use concrete shape from tensor, replace -1 in spec)
            std::vector<int64_t> shape;
            auto tensor_shape = tensor.Shape();
            for (size_t j = 0; j < spec.shape.size(); ++j) {
                if (spec.shape[j] == -1 && j < tensor_shape.size()) {
                    shape.push_back(static_cast<int64_t>(tensor_shape[j]));
                } else if (j < tensor_shape.size()) {
                    shape.push_back(static_cast<int64_t>(tensor_shape[j]));
                } else {
                    shape.push_back(spec.shape[j]);
                }
            }

            // Calculate total elements
            size_t num_elements = 1;
            for (auto dim : shape) {
                num_elements *= static_cast<size_t>(dim);
            }

            // Create ONNX tensor from CyxWiz tensor data
            // NOTE: This assumes float32 data - extend for other types as needed
            // const_cast is safe here - ONNX Runtime doesn't modify input data
            float* data_ptr = const_cast<float*>(static_cast<const float*>(tensor.Data()));

            auto ort_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                data_ptr,
                num_elements,
                shape.data(),
                shape.size()
            );

            input_tensors.push_back(std::move(ort_tensor));
        }

        // Run inference
        auto output_tensors = impl_->session->Run(
            Ort::RunOptions{nullptr},
            impl_->input_names.data(),
            input_tensors.data(),
            input_tensors.size(),
            impl_->output_names.data(),
            impl_->output_names.size()
        );

        // Convert outputs to CyxWiz tensors
        outputs.clear();
        for (size_t i = 0; i < output_tensors.size(); ++i) {
            const auto& ort_output = output_tensors[i];
            const auto& spec = impl_->output_specs[i];

            // Get output shape
            auto type_info = ort_output.GetTensorTypeAndShapeInfo();
            auto ort_shape = type_info.GetShape();

            std::vector<size_t> shape;
            for (auto dim : ort_shape) {
                shape.push_back(static_cast<size_t>(dim));
            }

            // Get output data pointer (use const version since ort_output is const)
            const float* data = ort_output.GetTensorData<float>();

            // Calculate total elements
            size_t num_elements = 1;
            for (auto dim : shape) {
                num_elements *= dim;
            }

            // Create CyxWiz tensor and copy data
            cyxwiz::Tensor output_tensor(shape, cyxwiz::DataType::Float32);
            std::memcpy(output_tensor.Data(), data, num_elements * sizeof(float));

            outputs[spec.name] = std::move(output_tensor);
        }

        // Also add "output" alias for single-output models
        if (outputs.size() == 1 && outputs.find("output") == outputs.end()) {
            outputs["output"] = outputs.begin()->second;
        }

        return true;

    } catch (const Ort::Exception& e) {
        spdlog::error("ONNX Runtime inference error: {}", e.what());

        // If CUDA inference failed and we haven't tried CPU fallback yet
        if (impl_->using_cuda && !impl_->cuda_inference_failed) {
            spdlog::warn("CUDA inference failed, attempting CPU fallback...");
            impl_->cuda_inference_failed = true;

            try {
                // Reload model with CPU only
                impl_->session.reset();
                impl_->session_options.reset();

                impl_->session_options = std::make_unique<Ort::SessionOptions>();
                impl_->session_options->SetIntraOpNumThreads(4);
                impl_->session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

                // Create new session with CPU only
#ifdef CYXWIZ_PLATFORM_WINDOWS
                std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
                std::wstring wide_path = converter.from_bytes(impl_->model_path);
                impl_->session = std::make_unique<Ort::Session>(
                    *impl_->env, wide_path.c_str(), *impl_->session_options);
#else
                impl_->session = std::make_unique<Ort::Session>(
                    *impl_->env, impl_->model_path.c_str(), *impl_->session_options);
#endif

                impl_->using_cuda = false;
                impl_->ExtractIOSpecs();
                spdlog::info("ONNX: Reloaded model with CPU execution provider");

                // Retry inference with CPU
                return Infer(inputs, outputs);

            } catch (const Ort::Exception& fallback_error) {
                spdlog::error("CPU fallback also failed: {}", fallback_error.what());
                return false;
            }
        }

        return false;
    } catch (const std::exception& e) {
        spdlog::error("ONNX inference exception: {}", e.what());
        return false;
    }
#endif
}

std::vector<TensorSpec> ONNXLoader::GetInputSpecs() const {
    return impl_->input_specs;
}

std::vector<TensorSpec> ONNXLoader::GetOutputSpecs() const {
    return impl_->output_specs;
}

uint64_t ONNXLoader::GetMemoryUsage() const {
    return impl_->memory_usage;
}

void ONNXLoader::Unload() {
    if (impl_->loaded) {
        spdlog::info("Unloading ONNX model");
#ifdef CYXWIZ_HAS_ONNX
        impl_->session.reset();
        impl_->session_options.reset();
        impl_->env.reset();
        impl_->input_names_str.clear();
        impl_->output_names_str.clear();
        impl_->input_names.clear();
        impl_->output_names.clear();
        impl_->using_cuda = false;
#endif
        impl_->loaded = false;
        impl_->input_specs.clear();
        impl_->output_specs.clear();
        impl_->memory_usage = 0;
    }
}

bool ONNXLoader::IsLoaded() const {
    return impl_->loaded;
}

void ONNXLoader::SetForceCPU(bool force) {
    impl_->force_cpu = force;
    if (force) {
        spdlog::info("ONNXLoader: CPU execution forced");
    }
}

bool ONNXLoader::IsUsingCUDA() const {
#ifdef CYXWIZ_HAS_ONNX
    return impl_->using_cuda;
#else
    return false;
#endif
}

// ============================================================================
// GGUFLoader Implementation
// ============================================================================

class GGUFLoader::Impl {
public:
    bool loaded = false;
    std::string model_path;
    std::vector<TensorSpec> input_specs;
    std::vector<TensorSpec> output_specs;
    uint64_t memory_usage = 0;

#ifdef CYXWIZ_HAS_GGUF
    // llama.cpp context
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    llama_sampler* sampler = nullptr;

    // Configuration (set before Load)
    int n_ctx = 2048;           // Context window size
    int n_gpu_layers = 0;       // GPU offloading (0 = CPU only)
    int n_threads = 4;          // CPU threads

    // Sampling parameters
    float temperature = 0.8f;
    int max_tokens = 256;
    float top_p = 0.95f;
    int top_k = 40;
    float repeat_penalty = 1.1f;

    // Model info (populated after load)
    int vocab_size = 0;
    int embedding_dim = 0;
    bool is_embedding_model = false;

    // Detect model type from metadata
    bool DetectModelType() {
        if (!model) return false;

        const llama_vocab* vocab = llama_model_get_vocab(model);
        vocab_size = llama_n_vocab(vocab);
        embedding_dim = llama_model_n_embd(model);

        // Try to detect embedding model from metadata
        // Embedding models typically have specific architecture names
        char buf[256] = {0};
        int32_t len = llama_model_meta_val_str(model, "general.architecture", buf, sizeof(buf));
        if (len > 0) {
            std::string arch(buf);
            std::transform(arch.begin(), arch.end(), arch.begin(), ::tolower);
            is_embedding_model = (arch.find("bert") != std::string::npos ||
                                  arch.find("nomic") != std::string::npos ||
                                  arch.find("embed") != std::string::npos ||
                                  arch.find("e5") != std::string::npos);
        }

        return true;
    }

    // Setup sampler chain for text generation
    void SetupSampler() {
        if (sampler) {
            llama_sampler_free(sampler);
            sampler = nullptr;
        }

        // Create sampler chain
        sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());

        // Add samplers in order: penalties -> top-k -> top-p -> temp -> dist
        llama_sampler_chain_add(sampler, llama_sampler_init_penalties(
            64,                   // penalty_last_n
            repeat_penalty,       // penalty_repeat
            0.0f,                 // penalty_freq
            0.0f                  // penalty_present
        ));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }

    // Reset sampler with new parameters
    void UpdateSampler() {
        SetupSampler();
    }
#endif
};

GGUFLoader::GGUFLoader() : impl_(std::make_unique<Impl>()) {
    spdlog::debug("GGUFLoader created");
}

GGUFLoader::~GGUFLoader() {
    Unload();
}

bool GGUFLoader::Load(const std::string& model_path) {
    spdlog::info("Loading GGUF model from: {}", model_path);

#ifndef CYXWIZ_HAS_GGUF
    spdlog::error("llama.cpp support not compiled. Rebuild with CYXWIZ_ENABLE_GGUF=ON");
    return false;
#else
    try {
        // Check file exists
        if (!std::filesystem::exists(model_path)) {
            spdlog::error("GGUF model file not found: {}", model_path);
            return false;
        }

        impl_->model_path = model_path;

        // Initialize llama backend (safe to call multiple times)
        llama_backend_init();

        // Configure model loading
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = impl_->n_gpu_layers;

        // Load model
        impl_->model = llama_model_load_from_file(model_path.c_str(), model_params);
        if (!impl_->model) {
            spdlog::error("Failed to load GGUF model: {}", model_path);
            return false;
        }

        // Detect model type (text generation vs embedding)
        impl_->DetectModelType();

        // Configure context
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = impl_->n_ctx;
        ctx_params.n_threads = impl_->n_threads;
        ctx_params.n_threads_batch = impl_->n_threads;

        // Create context
        impl_->ctx = llama_init_from_model(impl_->model, ctx_params);
        if (!impl_->ctx) {
            spdlog::error("Failed to create llama context");
            llama_model_free(impl_->model);
            impl_->model = nullptr;
            return false;
        }

        // Setup sampler for text generation
        if (!impl_->is_embedding_model) {
            impl_->SetupSampler();
        }

        // Setup input/output specs based on model type
        impl_->input_specs.clear();
        impl_->output_specs.clear();

        if (impl_->is_embedding_model) {
            // Embedding model: text in -> embedding vector out
            TensorSpec input_spec;
            input_spec.name = "text";
            input_spec.shape = {-1};  // Variable length string
            input_spec.dtype = "string";
            impl_->input_specs.push_back(input_spec);

            TensorSpec output_spec;
            output_spec.name = "embedding";
            output_spec.shape = {-1, impl_->embedding_dim};
            output_spec.dtype = "float32";
            impl_->output_specs.push_back(output_spec);
        } else {
            // Text generation model: text/tokens in -> text/tokens out
            TensorSpec input_spec;
            input_spec.name = "prompt";
            input_spec.shape = {-1};
            input_spec.dtype = "string";
            impl_->input_specs.push_back(input_spec);

            TensorSpec output_spec;
            output_spec.name = "completion";
            output_spec.shape = {-1};
            output_spec.dtype = "string";
            impl_->output_specs.push_back(output_spec);
        }

        // Get memory usage from model
        impl_->memory_usage = llama_model_size(impl_->model);

        impl_->loaded = true;

        spdlog::info("GGUF model loaded successfully");
        spdlog::info("  Type: {}", impl_->is_embedding_model ? "Embedding" : "Text Generation");
        spdlog::info("  Vocab size: {}", impl_->vocab_size);
        spdlog::info("  Embedding dim: {}", impl_->embedding_dim);
        spdlog::info("  Context size: {}", impl_->n_ctx);
        spdlog::info("  GPU layers: {}", impl_->n_gpu_layers);
        spdlog::info("  Memory: {:.2f} GB", impl_->memory_usage / (1024.0 * 1024.0 * 1024.0));

        return true;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load GGUF model: {}", e.what());
        impl_->loaded = false;
        return false;
    }
#endif
}

bool GGUFLoader::Infer(
    const std::unordered_map<std::string, cyxwiz::Tensor>& inputs,
    std::unordered_map<std::string, cyxwiz::Tensor>& outputs) {

    if (!impl_->loaded) {
        spdlog::error("GGUF model not loaded");
        return false;
    }

#ifndef CYXWIZ_HAS_GGUF
    spdlog::error("llama.cpp support not compiled");
    return false;
#else
    try {
        // Determine inference mode from input keys
        // Mode 1: "prompt" or "text" key -> text generation
        // Mode 2: "text" key + embedding model -> embeddings
        // Mode 3: "input_ids" key -> raw token generation

        if (inputs.count("text") && impl_->is_embedding_model) {
            return InferEmbeddings(inputs, outputs);
        } else if (inputs.count("prompt") || inputs.count("text")) {
            return InferTextGeneration(inputs, outputs);
        } else if (inputs.count("input_ids")) {
            return InferTokens(inputs, outputs);
        } else {
            spdlog::error("GGUF: Unknown input format. Expected 'prompt', 'text', or 'input_ids'");
            return false;
        }

    } catch (const std::exception& e) {
        spdlog::error("GGUF inference error: {}", e.what());
        return false;
    }
#endif
}

// Text generation inference
bool GGUFLoader::InferTextGeneration(
    const std::unordered_map<std::string, cyxwiz::Tensor>& inputs,
    std::unordered_map<std::string, cyxwiz::Tensor>& outputs) {

#ifndef CYXWIZ_HAS_GGUF
    return false;
#else
    // Get prompt text from input tensor
    std::string prompt;
    if (inputs.count("prompt")) {
        const auto& tensor = inputs.at("prompt");
        const char* data = static_cast<const char*>(tensor.Data());
        prompt = std::string(data, tensor.NumElements());
    } else if (inputs.count("text")) {
        const auto& tensor = inputs.at("text");
        const char* data = static_cast<const char*>(tensor.Data());
        prompt = std::string(data, tensor.NumElements());
    }

    if (prompt.empty()) {
        spdlog::error("GGUF: Empty prompt");
        return false;
    }

    spdlog::debug("GGUF: Generating text for prompt ({} chars)", prompt.length());

    // Get vocab for tokenization
    const llama_vocab* vocab = llama_model_get_vocab(impl_->model);

    // Tokenize input
    std::vector<llama_token> tokens(impl_->n_ctx);
    int n_tokens = llama_tokenize(
        vocab,
        prompt.c_str(),
        static_cast<int32_t>(prompt.length()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        true,  // add_special (BOS)
        false  // parse_special
    );

    if (n_tokens < 0) {
        spdlog::error("GGUF: Tokenization failed");
        return false;
    }
    tokens.resize(n_tokens);

    spdlog::debug("GGUF: Prompt tokenized to {} tokens", n_tokens);

    // Clear memory for fresh generation
    llama_memory_clear(llama_get_memory(impl_->ctx), true);

    // Decode prompt tokens
    llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
    if (llama_decode(impl_->ctx, batch) != 0) {
        spdlog::error("GGUF: Failed to decode prompt");
        return false;
    }

    // Generate completion tokens
    std::vector<llama_token> generated_tokens;
    generated_tokens.reserve(impl_->max_tokens);

    for (int i = 0; i < impl_->max_tokens; ++i) {
        // Sample next token
        llama_token new_token = llama_sampler_sample(impl_->sampler, impl_->ctx, -1);

        // Check for end of generation
        if (llama_token_is_eog(vocab, new_token)) {
            break;
        }

        generated_tokens.push_back(new_token);

        // Decode new token
        batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(impl_->ctx, batch) != 0) {
            spdlog::error("GGUF: Failed to decode generated token");
            break;
        }
    }

    spdlog::debug("GGUF: Generated {} tokens", generated_tokens.size());

    // Convert tokens back to text
    std::string completion;
    for (llama_token token : generated_tokens) {
        char buf[256];
        int len = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, false);
        if (len > 0) {
            completion.append(buf, len);
        }
    }

    // Create output tensor with completion text
    std::vector<size_t> shape = {completion.length()};
    cyxwiz::Tensor output_tensor(shape, cyxwiz::DataType::UInt8);
    std::memcpy(output_tensor.Data(), completion.data(), completion.length());

    outputs["completion"] = std::move(output_tensor);
    outputs["output"] = outputs["completion"];  // Alias for compatibility

    spdlog::debug("GGUF: Generated completion ({} chars)", completion.length());
    return true;
#endif
}

// Embeddings inference
bool GGUFLoader::InferEmbeddings(
    const std::unordered_map<std::string, cyxwiz::Tensor>& inputs,
    std::unordered_map<std::string, cyxwiz::Tensor>& outputs) {

#ifndef CYXWIZ_HAS_GGUF
    return false;
#else
    // Get text input
    std::string text;
    const auto& tensor = inputs.at("text");
    const char* data = static_cast<const char*>(tensor.Data());
    text = std::string(data, tensor.NumElements());

    if (text.empty()) {
        spdlog::error("GGUF: Empty text for embedding");
        return false;
    }

    // Get vocab for tokenization
    const llama_vocab* vocab = llama_model_get_vocab(impl_->model);

    // Tokenize
    std::vector<llama_token> tokens(impl_->n_ctx);
    int n_tokens = llama_tokenize(
        vocab,
        text.c_str(),
        static_cast<int32_t>(text.length()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        true,
        false
    );

    if (n_tokens < 0) {
        spdlog::error("GGUF: Tokenization failed for embedding");
        return false;
    }
    tokens.resize(n_tokens);

    // Clear memory and decode
    llama_memory_clear(llama_get_memory(impl_->ctx), true);
    llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));

    if (llama_decode(impl_->ctx, batch) != 0) {
        spdlog::error("GGUF: Failed to decode for embedding");
        return false;
    }

    // Get embeddings
    float* embeddings = llama_get_embeddings(impl_->ctx);
    if (!embeddings) {
        spdlog::error("GGUF: Failed to get embeddings (model may not support embeddings)");
        return false;
    }

    // Create output tensor
    std::vector<size_t> shape = {1, static_cast<size_t>(impl_->embedding_dim)};
    cyxwiz::Tensor output_tensor(shape, cyxwiz::DataType::Float32);
    std::memcpy(output_tensor.Data(), embeddings, impl_->embedding_dim * sizeof(float));

    outputs["embedding"] = std::move(output_tensor);
    outputs["output"] = outputs["embedding"];  // Alias

    return true;
#endif
}

// Token-based inference (raw token IDs)
bool GGUFLoader::InferTokens(
    const std::unordered_map<std::string, cyxwiz::Tensor>& inputs,
    std::unordered_map<std::string, cyxwiz::Tensor>& outputs) {

#ifndef CYXWIZ_HAS_GGUF
    return false;
#else
    // Get input token IDs
    const auto& tensor = inputs.at("input_ids");
    const int32_t* token_data = static_cast<const int32_t*>(tensor.Data());
    size_t n_tokens = tensor.NumElements();

    std::vector<llama_token> tokens(token_data, token_data + n_tokens);

    // Get vocab for EOG check
    const llama_vocab* vocab = llama_model_get_vocab(impl_->model);

    // Clear memory
    llama_memory_clear(llama_get_memory(impl_->ctx), true);

    // Decode input tokens
    llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
    if (llama_decode(impl_->ctx, batch) != 0) {
        spdlog::error("GGUF: Failed to decode input tokens");
        return false;
    }

    // Generate completion tokens
    std::vector<llama_token> generated_tokens;
    generated_tokens.reserve(impl_->max_tokens);

    for (int i = 0; i < impl_->max_tokens; ++i) {
        llama_token new_token = llama_sampler_sample(impl_->sampler, impl_->ctx, -1);

        if (llama_token_is_eog(vocab, new_token)) {
            break;
        }

        generated_tokens.push_back(new_token);

        batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(impl_->ctx, batch) != 0) {
            break;
        }
    }

    // Create output tensor with token IDs
    std::vector<size_t> shape = {generated_tokens.size()};
    cyxwiz::Tensor output_tensor(shape, cyxwiz::DataType::Int32);
    std::memcpy(output_tensor.Data(), generated_tokens.data(), generated_tokens.size() * sizeof(int32_t));

    outputs["output_ids"] = std::move(output_tensor);
    outputs["output"] = outputs["output_ids"];

    return true;
#endif
}

std::vector<TensorSpec> GGUFLoader::GetInputSpecs() const {
    return impl_->input_specs;
}

std::vector<TensorSpec> GGUFLoader::GetOutputSpecs() const {
    return impl_->output_specs;
}

uint64_t GGUFLoader::GetMemoryUsage() const {
    return impl_->memory_usage;
}

void GGUFLoader::Unload() {
    if (impl_->loaded) {
        spdlog::info("Unloading GGUF model");
#ifdef CYXWIZ_HAS_GGUF
        if (impl_->sampler) {
            llama_sampler_free(impl_->sampler);
            impl_->sampler = nullptr;
        }
        if (impl_->ctx) {
            llama_free(impl_->ctx);
            impl_->ctx = nullptr;
        }
        if (impl_->model) {
            llama_model_free(impl_->model);
            impl_->model = nullptr;
        }
        // Note: llama_backend_free() is global, don't call per-model
#endif
        impl_->loaded = false;
        impl_->input_specs.clear();
        impl_->output_specs.clear();
        impl_->memory_usage = 0;
    }
}

bool GGUFLoader::IsLoaded() const {
    return impl_->loaded;
}

// ========== Configuration Methods ==========

void GGUFLoader::SetContextSize(int n_ctx) {
#ifdef CYXWIZ_HAS_GGUF
    impl_->n_ctx = n_ctx;
#endif
}

void GGUFLoader::SetGPULayers(int n_gpu_layers) {
#ifdef CYXWIZ_HAS_GGUF
    impl_->n_gpu_layers = n_gpu_layers;
#endif
}

void GGUFLoader::SetThreads(int n_threads) {
#ifdef CYXWIZ_HAS_GGUF
    impl_->n_threads = n_threads;
#endif
}

void GGUFLoader::SetTemperature(float temp) {
#ifdef CYXWIZ_HAS_GGUF
    impl_->temperature = temp;
    if (impl_->loaded && impl_->sampler) {
        impl_->UpdateSampler();
    }
#endif
}

void GGUFLoader::SetMaxTokens(int max_tokens) {
#ifdef CYXWIZ_HAS_GGUF
    impl_->max_tokens = max_tokens;
#endif
}

void GGUFLoader::SetTopP(float top_p) {
#ifdef CYXWIZ_HAS_GGUF
    impl_->top_p = top_p;
    if (impl_->loaded && impl_->sampler) {
        impl_->UpdateSampler();
    }
#endif
}

void GGUFLoader::SetTopK(int top_k) {
#ifdef CYXWIZ_HAS_GGUF
    impl_->top_k = top_k;
    if (impl_->loaded && impl_->sampler) {
        impl_->UpdateSampler();
    }
#endif
}

void GGUFLoader::SetRepeatPenalty(float penalty) {
#ifdef CYXWIZ_HAS_GGUF
    impl_->repeat_penalty = penalty;
    if (impl_->loaded && impl_->sampler) {
        impl_->UpdateSampler();
    }
#endif
}

// ========== Model Information Methods ==========

bool GGUFLoader::IsEmbeddingModel() const {
#ifdef CYXWIZ_HAS_GGUF
    return impl_->is_embedding_model;
#else
    return false;
#endif
}

int GGUFLoader::GetVocabSize() const {
#ifdef CYXWIZ_HAS_GGUF
    return impl_->vocab_size;
#else
    return 0;
#endif
}

int GGUFLoader::GetEmbeddingDim() const {
#ifdef CYXWIZ_HAS_GGUF
    return impl_->embedding_dim;
#else
    return 0;
#endif
}

int GGUFLoader::GetContextSize() const {
#ifdef CYXWIZ_HAS_GGUF
    return impl_->n_ctx;
#else
    return 0;
#endif
}

// ============================================================================
// PyTorchLoader Implementation (LibTorch)
// ============================================================================

class PyTorchLoader::Impl {
public:
    bool loaded = false;
    std::string model_path;
    std::vector<TensorSpec> input_specs;
    std::vector<TensorSpec> output_specs;
    uint64_t memory_usage = 0;

#ifdef CYXWIZ_HAS_PYTORCH
    torch::jit::script::Module module;
    torch::Device device = torch::kCPU;
    bool has_cuda = false;
#endif
};

PyTorchLoader::PyTorchLoader() : impl_(std::make_unique<Impl>()) {
    spdlog::debug("PyTorchLoader created");
}

PyTorchLoader::~PyTorchLoader() {
    Unload();
}

bool PyTorchLoader::Load(const std::string& model_path) {
    spdlog::info("Loading PyTorch model from: {}", model_path);

#ifdef CYXWIZ_HAS_PYTORCH
    try {
        // Check for CUDA availability
        if (torch::cuda::is_available()) {
            impl_->device = torch::Device(torch::kCUDA, 0);
            impl_->has_cuda = true;
            spdlog::info("PyTorch: Using CUDA device");
        } else {
            impl_->device = torch::kCPU;
            impl_->has_cuda = false;
            spdlog::info("PyTorch: Using CPU device");
        }

        // Load TorchScript model
        impl_->module = torch::jit::load(model_path, impl_->device);
        impl_->module.eval();  // Set to evaluation mode
        impl_->model_path = model_path;
        impl_->loaded = true;

        // TorchScript doesn't have standard input/output metadata
        // Use default specs that can be overridden
        TensorSpec input_spec;
        input_spec.name = "input";
        input_spec.shape = {-1};  // Dynamic batch
        input_spec.dtype = "float32";
        impl_->input_specs.push_back(input_spec);

        TensorSpec output_spec;
        output_spec.name = "output";
        output_spec.shape = {-1};
        output_spec.dtype = "float32";
        impl_->output_specs.push_back(output_spec);

        // Estimate memory usage from model parameters
        impl_->memory_usage = 0;
        for (const auto& param : impl_->module.parameters()) {
            impl_->memory_usage += param.numel() * param.element_size();
        }

        spdlog::info("PyTorch model loaded successfully, estimated memory: {} MB",
                     impl_->memory_usage / (1024 * 1024));
        return true;

    } catch (const c10::Error& e) {
        spdlog::error("Failed to load PyTorch model (c10::Error): {}", e.what());
        impl_->loaded = false;
        return false;
    } catch (const std::exception& e) {
        spdlog::error("Failed to load PyTorch model: {}", e.what());
        impl_->loaded = false;
        return false;
    }
#else
    spdlog::error("PyTorch support not compiled - install LibTorch and rebuild with CYXWIZ_ENABLE_PYTORCH=ON");
    return false;
#endif
}

bool PyTorchLoader::Infer(
    const std::unordered_map<std::string, cyxwiz::Tensor>& inputs,
    std::unordered_map<std::string, cyxwiz::Tensor>& outputs) {

    if (!impl_->loaded) {
        spdlog::error("Model not loaded");
        return false;
    }

#ifdef CYXWIZ_HAS_PYTORCH
    try {
        // Convert inputs to torch::Tensor
        std::vector<torch::jit::IValue> torch_inputs;

        for (const auto& [name, tensor] : inputs) {
            // Get shape as int64_t vector
            std::vector<int64_t> shape;
            for (size_t dim : tensor.Shape()) {
                shape.push_back(static_cast<int64_t>(dim));
            }

            // Create torch tensor from data
            torch::Tensor t;
            switch (tensor.DType()) {
                case cyxwiz::DataType::Float32:
                    t = torch::from_blob(
                        const_cast<void*>(tensor.Data()),
                        shape,
                        torch::kFloat32
                    ).clone().to(impl_->device);
                    break;
                case cyxwiz::DataType::Float64:
                    t = torch::from_blob(
                        const_cast<void*>(tensor.Data()),
                        shape,
                        torch::kFloat64
                    ).clone().to(impl_->device);
                    break;
                case cyxwiz::DataType::Int32:
                    t = torch::from_blob(
                        const_cast<void*>(tensor.Data()),
                        shape,
                        torch::kInt32
                    ).clone().to(impl_->device);
                    break;
                case cyxwiz::DataType::Int64:
                    t = torch::from_blob(
                        const_cast<void*>(tensor.Data()),
                        shape,
                        torch::kInt64
                    ).clone().to(impl_->device);
                    break;
                default:
                    // Default to float32
                    t = torch::from_blob(
                        const_cast<void*>(tensor.Data()),
                        shape,
                        torch::kFloat32
                    ).clone().to(impl_->device);
            }
            torch_inputs.push_back(t);
        }

        // Run inference with no gradient
        torch::NoGradGuard no_grad;
        auto result = impl_->module.forward(torch_inputs);

        // Convert output to cyxwiz::Tensor
        torch::Tensor output_tensor;
        if (result.isTensor()) {
            output_tensor = result.toTensor().to(torch::kCPU).contiguous();
        } else if (result.isTuple()) {
            // Take first element of tuple
            output_tensor = result.toTuple()->elements()[0].toTensor().to(torch::kCPU).contiguous();
        } else {
            spdlog::error("PyTorch model returned unsupported output type");
            return false;
        }

        // Build output tensor
        std::vector<size_t> output_shape;
        for (int64_t dim : output_tensor.sizes()) {
            output_shape.push_back(static_cast<size_t>(dim));
        }

        cyxwiz::DataType output_dtype = cyxwiz::DataType::Float32;
        if (output_tensor.scalar_type() == torch::kFloat64) {
            output_dtype = cyxwiz::DataType::Float64;
        } else if (output_tensor.scalar_type() == torch::kInt32) {
            output_dtype = cyxwiz::DataType::Int32;
        } else if (output_tensor.scalar_type() == torch::kInt64) {
            output_dtype = cyxwiz::DataType::Int64;
        }

        cyxwiz::Tensor out_tensor(output_shape, output_dtype);
        std::memcpy(out_tensor.Data(), output_tensor.data_ptr(),
                    output_tensor.numel() * output_tensor.element_size());

        outputs["output"] = std::move(out_tensor);

        spdlog::debug("PyTorch inference completed successfully");
        return true;

    } catch (const c10::Error& e) {
        spdlog::error("PyTorch inference failed (c10::Error): {}", e.what());
        return false;
    } catch (const std::exception& e) {
        spdlog::error("PyTorch inference failed: {}", e.what());
        return false;
    }
#else
    spdlog::error("PyTorch support not compiled");
    return false;
#endif
}

std::vector<TensorSpec> PyTorchLoader::GetInputSpecs() const {
    return impl_->input_specs;
}

std::vector<TensorSpec> PyTorchLoader::GetOutputSpecs() const {
    return impl_->output_specs;
}

uint64_t PyTorchLoader::GetMemoryUsage() const {
    return impl_->memory_usage;
}

void PyTorchLoader::Unload() {
    if (impl_->loaded) {
        spdlog::info("Unloading PyTorch model");
#ifdef CYXWIZ_HAS_PYTORCH
        impl_->module = torch::jit::script::Module();  // Reset module
#endif
        impl_->loaded = false;
        impl_->input_specs.clear();
        impl_->output_specs.clear();
        impl_->memory_usage = 0;
    }
}

bool PyTorchLoader::IsLoaded() const {
    return impl_->loaded;
}

// ============================================================================
// CyxWizLoader Implementation (Native .cyxmodel format)
// ============================================================================

using json = nlohmann::json;

class CyxWizLoader::Impl {
public:
    bool loaded = false;
    std::string model_path;
    std::unique_ptr<cyxwiz::SequentialModel> model;
    std::vector<TensorSpec> input_specs;
    std::vector<TensorSpec> output_specs;
    uint64_t memory_usage = 0;
    json metadata;

    // Parse module name like "Linear(784 -> 512)" or "ReLU" or "Dropout(p=0.2)"
    bool ParseModuleName(const std::string& name,
                         cyxwiz::ModuleType& type,
                         std::map<std::string, std::string>& params) {
        // Linear layer: "Linear(in -> out)" format
        std::regex linear_regex(R"(Linear\((\d+)\s*->\s*(\d+)\))");
        std::smatch match;
        if (std::regex_search(name, match, linear_regex)) {
            type = cyxwiz::ModuleType::Linear;
            params["in_features"] = match[1].str();
            params["out_features"] = match[2].str();
            return true;
        }

        // Dropout: "Dropout(p=0.5)" format
        std::regex dropout_regex(R"(Dropout\(p=([0-9.]+)\))");
        if (std::regex_search(name, match, dropout_regex)) {
            type = cyxwiz::ModuleType::Dropout;
            params["p"] = match[1].str();
            return true;
        }

        // Simple activations (no parameters)
        if (name == "ReLU") {
            type = cyxwiz::ModuleType::ReLU;
            return true;
        }
        if (name == "Sigmoid") {
            type = cyxwiz::ModuleType::Sigmoid;
            return true;
        }
        if (name == "Tanh") {
            type = cyxwiz::ModuleType::Tanh;
            return true;
        }
        if (name == "Softmax" || name.find("Softmax") != std::string::npos) {
            type = cyxwiz::ModuleType::Softmax;
            return true;
        }
        if (name == "GELU") {
            type = cyxwiz::ModuleType::GELU;
            return true;
        }
        if (name == "Swish" || name == "SiLU") {
            type = cyxwiz::ModuleType::Swish;
            return true;
        }
        if (name == "Mish") {
            type = cyxwiz::ModuleType::Mish;
            return true;
        }

        // LeakyReLU: "LeakyReLU(slope=0.1)" format
        std::regex leaky_regex(R"(LeakyReLU\(slope=([0-9.]+)\))");
        if (std::regex_search(name, match, leaky_regex)) {
            type = cyxwiz::ModuleType::LeakyReLU;
            params["negative_slope"] = match[1].str();
            return true;
        }

        // ELU: "ELU(alpha=1.0)" format
        std::regex elu_regex(R"(ELU\(alpha=([0-9.]+)\))");
        if (std::regex_search(name, match, elu_regex)) {
            type = cyxwiz::ModuleType::ELU;
            params["alpha"] = match[1].str();
            return true;
        }

        // Flatten: "Flatten" or "Flatten(start_dim=1)"
        if (name.find("Flatten") != std::string::npos) {
            type = cyxwiz::ModuleType::Flatten;
            std::regex flatten_regex(R"(Flatten\(start_dim=(\d+)\))");
            if (std::regex_search(name, match, flatten_regex)) {
                params["start_dim"] = match[1].str();
            }
            return true;
        }

        spdlog::warn("CyxWizLoader: Unknown module type: {}", name);
        return false;
    }

    // Build architecture from JSON metadata
    bool BuildArchitectureFromMetadata() {
        if (!metadata.contains("modules")) {
            spdlog::error("CyxWizLoader: No modules in metadata");
            return false;
        }

        model = std::make_unique<cyxwiz::SequentialModel>();

        for (const auto& mod : metadata["modules"]) {
            std::string name = mod.value("name", "");
            if (name.empty()) continue;

            cyxwiz::ModuleType type;
            std::map<std::string, std::string> params;

            if (!ParseModuleName(name, type, params)) {
                spdlog::error("CyxWizLoader: Failed to parse module: {}", name);
                return false;
            }

            auto module = cyxwiz::CreateModule(type, params);
            if (!module) {
                spdlog::error("CyxWizLoader: Failed to create module: {}", name);
                return false;
            }

            model->AddModule(std::move(module));
            spdlog::debug("CyxWizLoader: Added module: {}", name);
        }

        return model->Size() > 0;
    }

    // Read .cyxmodel header and extract JSON metadata
    bool ReadModelHeader(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            spdlog::error("CyxWizLoader: Failed to open file: {}", path);
            return false;
        }

        // Read magic number
        uint32_t magic;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != 0x43595857) {  // "CYXW"
            spdlog::error("CyxWizLoader: Invalid magic number");
            return false;
        }

        // Read version
        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != 2) {
            spdlog::error("CyxWizLoader: Unsupported format version: {}", version);
            return false;
        }

        // Read JSON length
        uint64_t json_len;
        file.read(reinterpret_cast<char*>(&json_len), sizeof(json_len));

        // Read JSON data
        std::string json_str(json_len, '\0');
        file.read(json_str.data(), json_len);

        // Parse JSON
        try {
            metadata = json::parse(json_str);
        } catch (const std::exception& e) {
            spdlog::error("CyxWizLoader: Failed to parse JSON: {}", e.what());
            return false;
        }

        return true;
    }

    // Infer input/output specs from the model architecture
    void InferSpecs() {
        input_specs.clear();
        output_specs.clear();

        if (!metadata.contains("modules") || metadata["modules"].empty()) {
            return;
        }

        // Find first Linear layer for input spec
        for (const auto& mod : metadata["modules"]) {
            std::string name = mod.value("name", "");
            std::regex linear_regex(R"(Linear\((\d+)\s*->\s*(\d+)\))");
            std::smatch match;
            if (std::regex_search(name, match, linear_regex)) {
                TensorSpec input_spec;
                input_spec.name = "input";
                input_spec.shape = {-1, std::stoll(match[1].str())};  // [batch, in_features]
                input_spec.dtype = "float32";
                input_specs.push_back(input_spec);
                break;
            }
        }

        // Find last Linear layer for output spec
        for (auto it = metadata["modules"].rbegin(); it != metadata["modules"].rend(); ++it) {
            std::string name = (*it).value("name", "");
            std::regex linear_regex(R"(Linear\((\d+)\s*->\s*(\d+)\))");
            std::smatch match;
            if (std::regex_search(name, match, linear_regex)) {
                TensorSpec output_spec;
                output_spec.name = "output";
                output_spec.shape = {-1, std::stoll(match[2].str())};  // [batch, out_features]
                output_spec.dtype = "float32";
                output_specs.push_back(output_spec);
                break;
            }
        }
    }
};

CyxWizLoader::CyxWizLoader() : impl_(std::make_unique<Impl>()) {
    spdlog::debug("CyxWizLoader created");
}

CyxWizLoader::~CyxWizLoader() {
    Unload();
}

bool CyxWizLoader::Load(const std::string& model_path) {
    spdlog::info("Loading CyxWiz model from: {}", model_path);

    try {
        // Check path exists
        if (!std::filesystem::exists(model_path)) {
            spdlog::error("CyxWizLoader: Path not found: {}", model_path);
            return false;
        }

        impl_->model_path = model_path;

        // Check if it's directory format or binary format
        if (std::filesystem::is_directory(model_path)) {
            // Directory format - load from metadata.json and weights folder
            return LoadDirectoryFormat(model_path);
        } else {
            // Binary format - load from single .cyxmodel file
            return LoadBinaryFormat(model_path);
        }

    } catch (const std::exception& e) {
        spdlog::error("CyxWizLoader: Exception during load: {}", e.what());
        impl_->loaded = false;
        return false;
    }
}

bool CyxWizLoader::LoadBinaryFormat(const std::string& path) {
    spdlog::info("Loading binary format from: {}", path);

    // Read header and extract metadata
    if (!impl_->ReadModelHeader(path)) {
        return false;
    }

    // Build model architecture from metadata
    if (!impl_->BuildArchitectureFromMetadata()) {
        spdlog::error("CyxWizLoader: Failed to build architecture");
        return false;
    }

    // Load weights using SequentialModel::Load()
    if (!impl_->model->Load(path)) {
        spdlog::error("CyxWizLoader: Failed to load weights");
        return false;
    }

    // Set to inference mode
    impl_->model->SetTraining(false);

    // Infer input/output specs
    impl_->InferSpecs();

    // Estimate memory usage
    auto params = impl_->model->GetParameters();
    uint64_t param_bytes = 0;
    for (const auto& [name, tensor] : params) {
        param_bytes += tensor.NumBytes();
    }
    impl_->memory_usage = param_bytes;

    impl_->loaded = true;
    spdlog::info("CyxWizLoader: Binary model loaded ({} modules, {} bytes)",
                 impl_->model->Size(), impl_->memory_usage);
    return true;
}

bool CyxWizLoader::LoadDirectoryFormat(const std::string& dir_path) {
    spdlog::info("Loading directory format from: {}", dir_path);

    namespace fs = std::filesystem;

    // Check for required files - support both naming conventions
    fs::path manifest_path = fs::path(dir_path) / "manifest.json";
    fs::path metadata_path = fs::path(dir_path) / "metadata.json";
    fs::path config_path = fs::path(dir_path) / "config.json";
    fs::path graph_path = fs::path(dir_path) / "graph.cyxgraph";
    fs::path weights_dir = fs::path(dir_path) / "weights";

    // Find the metadata file (try manifest.json first, then metadata.json)
    fs::path actual_metadata_path;
    if (fs::exists(manifest_path)) {
        actual_metadata_path = manifest_path;
    } else if (fs::exists(metadata_path)) {
        actual_metadata_path = metadata_path;
    } else {
        spdlog::error("CyxWizLoader: Neither manifest.json nor metadata.json found in {}", dir_path);
        return false;
    }

    spdlog::info("CyxWizLoader: Using metadata from: {}", actual_metadata_path.filename().string());

    // Read metadata/manifest
    std::ifstream meta_file(actual_metadata_path);
    if (!meta_file) {
        spdlog::error("CyxWizLoader: Failed to open {}", actual_metadata_path.string());
        return false;
    }

    try {
        impl_->metadata = json::parse(meta_file);
    } catch (const std::exception& e) {
        spdlog::error("CyxWizLoader: Failed to parse {}: {}", actual_metadata_path.filename().string(), e.what());
        return false;
    }

    // If metadata doesn't have modules, try to build from graph.cyxgraph
    if (!impl_->metadata.contains("modules") || impl_->metadata["modules"].empty()) {
        if (fs::exists(graph_path)) {
            spdlog::info("CyxWizLoader: Building modules from graph.cyxgraph");
            std::ifstream graph_file(graph_path);
            if (graph_file) {
                try {
                    json graph = json::parse(graph_file);
                    if (graph.contains("nodes")) {
                        // Convert graph nodes to modules format
                        json modules = json::array();
                        for (const auto& node : graph["nodes"]) {
                            int node_type = node.value("type", -1);
                            std::string name = node.value("name", "");

                            // Skip non-layer nodes (Input=77, Output=50, Loss=52, Optimizer=60)
                            if (node_type == 77 || node_type == 50 || node_type >= 52) {
                                continue;
                            }

                            json module;
                            module["name"] = name;
                            if (node.contains("parameters")) {
                                module["parameters"] = node["parameters"];
                            }
                            modules.push_back(module);
                        }
                        impl_->metadata["modules"] = modules;
                        spdlog::info("CyxWizLoader: Loaded {} modules from graph", modules.size());
                    }
                } catch (const std::exception& e) {
                    spdlog::warn("CyxWizLoader: Failed to parse graph.cyxgraph: {}", e.what());
                }
            }
        }
    }

    // Build architecture from metadata
    if (!impl_->BuildArchitectureFromMetadata()) {
        spdlog::error("CyxWizLoader: Failed to build architecture from directory");
        return false;
    }

    // Load weights from directory
    if (fs::exists(weights_dir) && fs::is_directory(weights_dir)) {
        if (!LoadWeightsFromDirectory(weights_dir.string())) {
            spdlog::error("CyxWizLoader: Failed to load weights from directory");
            return false;
        }
    } else {
        spdlog::warn("CyxWizLoader: weights directory not found, model has no pretrained weights");
    }

    // Set to inference mode
    impl_->model->SetTraining(false);

    // Infer input/output specs
    impl_->InferSpecs();

    // Estimate memory usage
    auto params = impl_->model->GetParameters();
    uint64_t param_bytes = 0;
    for (const auto& [name, tensor] : params) {
        param_bytes += tensor.NumBytes();
    }
    impl_->memory_usage = param_bytes;

    impl_->loaded = true;
    spdlog::info("CyxWizLoader: Directory model loaded ({} modules, {} bytes)",
                 impl_->model->Size(), impl_->memory_usage);
    return true;
}

bool CyxWizLoader::LoadWeightsFromDirectory(const std::string& weights_dir) {
    namespace fs = std::filesystem;

    if (!impl_->model) {
        spdlog::error("CyxWizLoader: Model not initialized");
        return false;
    }

    // First, try to read the weights manifest if it exists
    fs::path manifest_path = fs::path(weights_dir) / "manifest.json";
    json weights_manifest;
    std::map<std::string, json> tensor_info;  // Maps tensor name to its info

    if (fs::exists(manifest_path)) {
        std::ifstream manifest_file(manifest_path);
        if (manifest_file) {
            try {
                weights_manifest = json::parse(manifest_file);
                if (weights_manifest.contains("tensors")) {
                    for (const auto& t : weights_manifest["tensors"]) {
                        std::string tensor_name = t.value("name", "");
                        if (!tensor_name.empty()) {
                            tensor_info[tensor_name] = t;
                        }
                    }
                    spdlog::info("CyxWizLoader: Found weights manifest with {} tensors", tensor_info.size());
                }
            } catch (...) {
                spdlog::warn("CyxWizLoader: Failed to parse weights manifest");
            }
        }
    }

    // Get model parameters (expected parameters)
    auto expected_params = impl_->model->GetParameters();

    // Build new parameter map
    std::map<std::string, cyxwiz::Tensor> new_params;

    // Build mapping from model layer indices to weight file indices
    // Model has layer0, layer1, layer2 but weights might be layer0, layer3, layer6
    std::vector<std::string> weight_layer_names;
    for (const auto& [name, _] : tensor_info) {
        // Extract layer index from names like "layer0.weight", "layer3.bias"
        if (name.find("weight") != std::string::npos) {
            size_t pos = name.find('.');
            if (pos != std::string::npos) {
                weight_layer_names.push_back(name.substr(0, pos));
            }
        }
    }
    std::sort(weight_layer_names.begin(), weight_layer_names.end());

    // Map model layer indices to weight file indices
    int linear_layer_idx = 0;
    for (const auto& [name, tensor] : expected_params) {
        // Try multiple file naming conventions
        std::vector<fs::path> candidates;

        // Original name with underscore (e.g., "layer0_weight.bin")
        std::string underscore_name = name;
        std::replace(underscore_name.begin(), underscore_name.end(), '.', '_');
        candidates.push_back(fs::path(weights_dir) / (underscore_name + ".bin"));

        // Original name with dot (e.g., "layer0.weight.bin")
        candidates.push_back(fs::path(weights_dir) / (name + ".bin"));

        // Try mapping model layer index to weight file index
        // e.g., model's "layer0.weight" might map to file "layer0_weight.bin" or "layer3_weight.bin"
        size_t dot_pos = name.find('.');
        if (dot_pos != std::string::npos) {
            std::string param_type = name.substr(dot_pos + 1);  // "weight" or "bias"
            std::string layer_part = name.substr(0, dot_pos);   // "layer0"

            // Extract our layer index
            int our_idx = 0;
            if (layer_part.size() > 5) {  // "layer" + digit
                try {
                    our_idx = std::stoi(layer_part.substr(5));
                } catch (...) {}
            }

            // Try to find corresponding weight file layer
            if (our_idx < (int)weight_layer_names.size()) {
                std::string weight_layer = weight_layer_names[our_idx];
                std::string mapped_name = weight_layer + "_" + param_type;
                candidates.insert(candidates.begin(), fs::path(weights_dir) / (mapped_name + ".bin"));
            }
        }

        // Find the first existing file
        fs::path weight_file;
        for (const auto& candidate : candidates) {
            if (fs::exists(candidate)) {
                weight_file = candidate;
                break;
            }
        }

        if (weight_file.empty()) {
            spdlog::error("CyxWizLoader: Weight file not found for {}", name);
            spdlog::error("  Tried: {}", candidates[0].string());
            return false;
        }

        // Read weight data
        std::ifstream file(weight_file, std::ios::binary);
        if (!file) {
            spdlog::error("CyxWizLoader: Failed to open weight file: {}", weight_file.string());
            return false;
        }

        // Get file size
        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        // Create tensor from data with the expected shape
        auto shape = tensor.Shape();
        cyxwiz::Tensor new_tensor(shape, cyxwiz::DataType::Float32);
        size_t expected_bytes = new_tensor.NumBytes();

        // Calculate expected header size: ndims(4) + shape(ndims*8) + dtype(4)
        size_t expected_header_size = 4 + shape.size() * 8 + 4;
        size_t expected_total = expected_bytes + expected_header_size;

        // Check if file has header (size matches with header)
        bool has_header = (file_size == expected_total);

        if (has_header) {
            // Skip header: ndims (uint32_t), shape (ndims * int64_t), dtype (uint32_t)
            uint32_t ndims;
            file.read(reinterpret_cast<char*>(&ndims), sizeof(ndims));
            for (uint32_t i = 0; i < ndims; ++i) {
                int64_t dim;
                file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
            }
            uint32_t dtype;
            file.read(reinterpret_cast<char*>(&dtype), sizeof(dtype));
            spdlog::debug("CyxWizLoader: Skipped {} byte header for {}", expected_header_size, name);
        } else if (file_size != expected_bytes) {
            spdlog::error("CyxWizLoader: Weight size mismatch for {}: expected {} bytes (or {} with header), got {} bytes",
                         name, expected_bytes, expected_total, file_size);
            return false;
        }

        // Read the tensor data
        std::vector<char> data(expected_bytes);
        file.read(data.data(), expected_bytes);

        std::memcpy(new_tensor.Data(), data.data(), expected_bytes);

        // Add to parameter map
        new_params[name] = std::move(new_tensor);

        spdlog::debug("CyxWizLoader: Loaded weight {} from {} ({} bytes)",
                     name, weight_file.filename().string(), file_size);
    }

    // Set all parameters at once
    impl_->model->SetParameters(new_params);

    spdlog::info("CyxWizLoader: Loaded {} weight tensors from directory", new_params.size());
    return true;
}

bool CyxWizLoader::Infer(
    const std::unordered_map<std::string, cyxwiz::Tensor>& inputs,
    std::unordered_map<std::string, cyxwiz::Tensor>& outputs) {

    if (!impl_->loaded || !impl_->model) {
        spdlog::error("CyxWizLoader: Model not loaded");
        return false;
    }

    try {
        // Get input tensor (expect "input" key)
        auto it = inputs.find("input");
        if (it == inputs.end()) {
            // Try to use first input if "input" not found
            if (inputs.empty()) {
                spdlog::error("CyxWizLoader: No input tensors provided");
                return false;
            }
            it = inputs.begin();
        }

        // Run forward pass
        cyxwiz::Tensor output = impl_->model->Forward(it->second);

        // Store output
        outputs["output"] = std::move(output);

        return true;

    } catch (const std::exception& e) {
        spdlog::error("CyxWizLoader: Inference exception: {}", e.what());
        return false;
    }
}

std::vector<TensorSpec> CyxWizLoader::GetInputSpecs() const {
    return impl_->input_specs;
}

std::vector<TensorSpec> CyxWizLoader::GetOutputSpecs() const {
    return impl_->output_specs;
}

uint64_t CyxWizLoader::GetMemoryUsage() const {
    return impl_->memory_usage;
}

void CyxWizLoader::Unload() {
    if (impl_->loaded) {
        spdlog::info("CyxWizLoader: Unloading model");
        impl_->model.reset();
        impl_->loaded = false;
        impl_->input_specs.clear();
        impl_->output_specs.clear();
        impl_->memory_usage = 0;
        impl_->metadata.clear();
    }
}

bool CyxWizLoader::IsLoaded() const {
    return impl_->loaded;
}

} // namespace servernode
} // namespace cyxwiz
