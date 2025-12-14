#include "model_loader.h"
#include <cyxwiz/sequential.h>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <regex>

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
// ONNXLoader Implementation (Stub)
// ============================================================================

class ONNXLoader::Impl {
public:
    bool loaded = false;
    std::string model_path;
    std::vector<TensorSpec> input_specs;
    std::vector<TensorSpec> output_specs;
    uint64_t memory_usage = 0;

    // TODO: Add ONNX Runtime session and related members
    // Ort::Session* session = nullptr;
    // Ort::Env* env = nullptr;
};

ONNXLoader::ONNXLoader() : impl_(std::make_unique<Impl>()) {
    spdlog::debug("ONNXLoader created");
}

ONNXLoader::~ONNXLoader() {
    Unload();
}

bool ONNXLoader::Load(const std::string& model_path) {
    spdlog::info("Loading ONNX model from: {}", model_path);

    try {
        // TODO: Implement actual ONNX Runtime loading
        // For now, just a stub
        impl_->model_path = model_path;
        impl_->loaded = true;

        // Mock input/output specs
        TensorSpec input_spec;
        input_spec.name = "input";
        input_spec.shape = {-1, 3, 224, 224};  // -1 = batch size (dynamic)
        input_spec.dtype = "float32";
        impl_->input_specs.push_back(input_spec);

        TensorSpec output_spec;
        output_spec.name = "output";
        output_spec.shape = {-1, 1000};
        output_spec.dtype = "float32";
        impl_->output_specs.push_back(output_spec);

        impl_->memory_usage = 100 * 1024 * 1024;  // Mock: 100 MB

        spdlog::info("ONNX model loaded successfully (stub)");
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load ONNX model: {}", e.what());
        impl_->loaded = false;
        return false;
    }
}

bool ONNXLoader::Infer(
    const std::unordered_map<std::string, cyxwiz::Tensor>& inputs,
    std::unordered_map<std::string, cyxwiz::Tensor>& outputs) {

    if (!impl_->loaded) {
        spdlog::error("Model not loaded");
        return false;
    }

    // TODO: Implement actual inference with ONNX Runtime
    spdlog::debug("Running ONNX inference (stub)");
    return true;
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
        // TODO: Clean up ONNX Runtime resources
        impl_->loaded = false;
        impl_->input_specs.clear();
        impl_->output_specs.clear();
        impl_->memory_usage = 0;
    }
}

bool ONNXLoader::IsLoaded() const {
    return impl_->loaded;
}

// ============================================================================
// GGUFLoader Implementation (Stub)
// ============================================================================

class GGUFLoader::Impl {
public:
    bool loaded = false;
    std::string model_path;
    std::vector<TensorSpec> input_specs;
    std::vector<TensorSpec> output_specs;
    uint64_t memory_usage = 0;

    // TODO: Add llama.cpp context and related members
    // llama_context* ctx = nullptr;
    // llama_model* model = nullptr;
};

GGUFLoader::GGUFLoader() : impl_(std::make_unique<Impl>()) {
    spdlog::debug("GGUFLoader created");
}

GGUFLoader::~GGUFLoader() {
    Unload();
}

bool GGUFLoader::Load(const std::string& model_path) {
    spdlog::info("Loading GGUF model from: {}", model_path);

    try {
        // TODO: Implement actual llama.cpp loading
        impl_->model_path = model_path;
        impl_->loaded = true;

        // Mock input/output specs for text generation
        TensorSpec input_spec;
        input_spec.name = "input_ids";
        input_spec.shape = {-1};  // Variable length sequence
        input_spec.dtype = "int64";
        impl_->input_specs.push_back(input_spec);

        TensorSpec output_spec;
        output_spec.name = "logits";
        output_spec.shape = {-1, 32000};  // Vocab size
        output_spec.dtype = "float32";
        impl_->output_specs.push_back(output_spec);

        impl_->memory_usage = 4 * 1024 * 1024 * 1024ULL;  // Mock: 4 GB

        spdlog::info("GGUF model loaded successfully (stub)");
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load GGUF model: {}", e.what());
        impl_->loaded = false;
        return false;
    }
}

bool GGUFLoader::Infer(
    const std::unordered_map<std::string, cyxwiz::Tensor>& inputs,
    std::unordered_map<std::string, cyxwiz::Tensor>& outputs) {

    if (!impl_->loaded) {
        spdlog::error("Model not loaded");
        return false;
    }

    // TODO: Implement actual inference with llama.cpp
    spdlog::debug("Running GGUF inference (stub)");
    return true;
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
        // TODO: Clean up llama.cpp resources
        impl_->loaded = false;
        impl_->input_specs.clear();
        impl_->output_specs.clear();
        impl_->memory_usage = 0;
    }
}

bool GGUFLoader::IsLoaded() const {
    return impl_->loaded;
}

// ============================================================================
// PyTorchLoader Implementation (Stub)
// ============================================================================

class PyTorchLoader::Impl {
public:
    bool loaded = false;
    std::string model_path;
    std::vector<TensorSpec> input_specs;
    std::vector<TensorSpec> output_specs;
    uint64_t memory_usage = 0;

    // TODO: Add LibTorch module and related members
    // torch::jit::script::Module* module = nullptr;
};

PyTorchLoader::PyTorchLoader() : impl_(std::make_unique<Impl>()) {
    spdlog::debug("PyTorchLoader created");
}

PyTorchLoader::~PyTorchLoader() {
    Unload();
}

bool PyTorchLoader::Load(const std::string& model_path) {
    spdlog::info("Loading PyTorch model from: {}", model_path);

    try {
        // TODO: Implement actual LibTorch loading
        impl_->model_path = model_path;
        impl_->loaded = true;

        // Mock input/output specs
        TensorSpec input_spec;
        input_spec.name = "input";
        input_spec.shape = {-1, 3, 224, 224};
        input_spec.dtype = "float32";
        impl_->input_specs.push_back(input_spec);

        TensorSpec output_spec;
        output_spec.name = "output";
        output_spec.shape = {-1, 1000};
        output_spec.dtype = "float32";
        impl_->output_specs.push_back(output_spec);

        impl_->memory_usage = 150 * 1024 * 1024;  // Mock: 150 MB

        spdlog::info("PyTorch model loaded successfully (stub)");
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load PyTorch model: {}", e.what());
        impl_->loaded = false;
        return false;
    }
}

bool PyTorchLoader::Infer(
    const std::unordered_map<std::string, cyxwiz::Tensor>& inputs,
    std::unordered_map<std::string, cyxwiz::Tensor>& outputs) {

    if (!impl_->loaded) {
        spdlog::error("Model not loaded");
        return false;
    }

    // TODO: Implement actual inference with LibTorch
    spdlog::debug("Running PyTorch inference (stub)");
    return true;
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
        // TODO: Clean up LibTorch resources
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
