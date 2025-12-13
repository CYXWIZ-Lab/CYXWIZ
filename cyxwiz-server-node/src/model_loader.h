#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <cyxwiz/tensor.h>

namespace cyxwiz {
namespace servernode {

// Tensor specification for model inputs/outputs
struct TensorSpec {
    std::string name;
    std::vector<int64_t> shape;  // -1 for dynamic dimensions
    std::string dtype;           // "float32", "int64", etc.
};

// Abstract base class for model loaders
class ModelLoader {
public:
    virtual ~ModelLoader() = default;

    // Load model from file
    virtual bool Load(const std::string& model_path) = 0;

    // Run inference
    virtual bool Infer(
        const std::unordered_map<std::string, cyxwiz::Tensor>& inputs,
        std::unordered_map<std::string, cyxwiz::Tensor>& outputs
    ) = 0;

    // Get input tensor specifications
    virtual std::vector<TensorSpec> GetInputSpecs() const = 0;

    // Get output tensor specifications
    virtual std::vector<TensorSpec> GetOutputSpecs() const = 0;

    // Get memory usage in bytes
    virtual uint64_t GetMemoryUsage() const = 0;

    // Unload model and free resources
    virtual void Unload() = 0;

    // Check if model is loaded
    virtual bool IsLoaded() const = 0;

    // Get model format name
    virtual std::string GetFormat() const = 0;
};

// Factory for creating model loaders
class ModelLoaderFactory {
public:
    // Create a model loader for the specified format
    static std::unique_ptr<ModelLoader> Create(const std::string& format);

    // Check if format is supported
    static bool IsFormatSupported(const std::string& format);

    // Get list of supported formats
    static std::vector<std::string> GetSupportedFormats();
};

// ============================================================================
// Concrete Implementations (to be implemented in separate files)
// ============================================================================

// ONNX Runtime loader
class ONNXLoader : public ModelLoader {
public:
    ONNXLoader();
    ~ONNXLoader() override;

    bool Load(const std::string& model_path) override;
    bool Infer(
        const std::unordered_map<std::string, cyxwiz::Tensor>& inputs,
        std::unordered_map<std::string, cyxwiz::Tensor>& outputs
    ) override;
    std::vector<TensorSpec> GetInputSpecs() const override;
    std::vector<TensorSpec> GetOutputSpecs() const override;
    uint64_t GetMemoryUsage() const override;
    void Unload() override;
    bool IsLoaded() const override;
    std::string GetFormat() const override { return "onnx"; }

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// GGUF loader (for LLMs via llama.cpp)
class GGUFLoader : public ModelLoader {
public:
    GGUFLoader();
    ~GGUFLoader() override;

    bool Load(const std::string& model_path) override;
    bool Infer(
        const std::unordered_map<std::string, cyxwiz::Tensor>& inputs,
        std::unordered_map<std::string, cyxwiz::Tensor>& outputs
    ) override;
    std::vector<TensorSpec> GetInputSpecs() const override;
    std::vector<TensorSpec> GetOutputSpecs() const override;
    uint64_t GetMemoryUsage() const override;
    void Unload() override;
    bool IsLoaded() const override;
    std::string GetFormat() const override { return "gguf"; }

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// PyTorch/LibTorch loader
class PyTorchLoader : public ModelLoader {
public:
    PyTorchLoader();
    ~PyTorchLoader() override;

    bool Load(const std::string& model_path) override;
    bool Infer(
        const std::unordered_map<std::string, cyxwiz::Tensor>& inputs,
        std::unordered_map<std::string, cyxwiz::Tensor>& outputs
    ) override;
    std::vector<TensorSpec> GetInputSpecs() const override;
    std::vector<TensorSpec> GetOutputSpecs() const override;
    uint64_t GetMemoryUsage() const override;
    void Unload() override;
    bool IsLoaded() const override;
    std::string GetFormat() const override { return "pytorch"; }

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// CyxWiz native model loader (.cyxmodel format)
class CyxWizLoader : public ModelLoader {
public:
    CyxWizLoader();
    ~CyxWizLoader() override;

    bool Load(const std::string& model_path) override;
    bool Infer(
        const std::unordered_map<std::string, cyxwiz::Tensor>& inputs,
        std::unordered_map<std::string, cyxwiz::Tensor>& outputs
    ) override;
    std::vector<TensorSpec> GetInputSpecs() const override;
    std::vector<TensorSpec> GetOutputSpecs() const override;
    uint64_t GetMemoryUsage() const override;
    void Unload() override;
    bool IsLoaded() const override;
    std::string GetFormat() const override { return "cyxmodel"; }

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace servernode
} // namespace cyxwiz
