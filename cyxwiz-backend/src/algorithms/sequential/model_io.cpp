#include <cyxwiz/sequential.h>

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {
// ============================================================================
// Serialization Implementation
// ============================================================================

using json = nlohmann::json;
namespace fs = std::filesystem;

// Helper: Get current timestamp as string
static std::string GetCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

// Helper: Convert DataType to string
static std::string DataTypeToString(DataType dtype) {
    switch (dtype) {
        case DataType::Float32: return "float32";
        case DataType::Float64: return "float64";
        case DataType::Int32: return "int32";
        case DataType::Int64: return "int64";
        case DataType::UInt8: return "uint8";
        default: return "float32";
    }
}

// Helper: Write tensor to binary stream
static void WriteTensor(std::ostream& os, const Tensor& tensor) {
    // Write shape
    auto shape = tensor.Shape();
    size_t ndims = shape.size();
    os.write(reinterpret_cast<const char*>(&ndims), sizeof(ndims));
    os.write(reinterpret_cast<const char*>(shape.data()), ndims * sizeof(size_t));

    // Write dtype
    DataType dtype = tensor.GetDataType();
    os.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));

    // Write data
    size_t num_bytes = tensor.NumBytes();
    os.write(reinterpret_cast<const char*>(&num_bytes), sizeof(num_bytes));
    os.write(reinterpret_cast<const char*>(tensor.Data()), num_bytes);
}

// Helper: Read tensor from binary stream
static Tensor ReadTensor(std::istream& is) {
    // Read shape
    size_t ndims;
    is.read(reinterpret_cast<char*>(&ndims), sizeof(ndims));
    std::vector<size_t> shape(ndims);
    is.read(reinterpret_cast<char*>(shape.data()), ndims * sizeof(size_t));

    // Read dtype
    DataType dtype;
    is.read(reinterpret_cast<char*>(&dtype), sizeof(dtype));

    // Read data
    size_t num_bytes;
    is.read(reinterpret_cast<char*>(&num_bytes), sizeof(num_bytes));

    Tensor tensor(shape, dtype);
    is.read(reinterpret_cast<char*>(tensor.Data()), num_bytes);

    return tensor;
}

bool SequentialModel::Save(const std::string& path) const {
    try {
        // Ensure path has .cyxmodel extension
        std::string file_path = path;
        if (file_path.size() < 9 || file_path.substr(file_path.size() - 9) != ".cyxmodel") {
            file_path += ".cyxmodel";
        }

        fs::path model_path(file_path);

        // Create directory if needed
        if (model_path.has_parent_path()) {
            fs::create_directories(model_path.parent_path());
        }

        // Prepare metadata JSON
        json meta;
        meta["metadata"]["name"] = model_name_;
        meta["metadata"]["description"] = model_description_;
        meta["metadata"]["created_at"] = GetCurrentTimestamp();
        meta["metadata"]["framework"] = "CyxWiz";
        meta["metadata"]["format_version"] = "2.0";

        // Save module info
        meta["modules"] = json::array();
        for (size_t i = 0; i < modules_.size(); ++i) {
            json module_info;
            module_info["index"] = i;
            module_info["name"] = modules_[i]->GetName();
            module_info["has_parameters"] = modules_[i]->HasParameters();
            module_info["trainable"] = modules_[i]->IsTrainable();

            if (modules_[i]->HasParameters()) {
                auto params = modules_[i]->GetParameters();
                json param_names = json::array();
                for (const auto& [name, tensor] : params) {
                    json param_info;
                    param_info["name"] = name;
                    param_info["shape"] = tensor.Shape();
                    param_info["dtype"] = DataTypeToString(tensor.GetDataType());
                    param_names.push_back(param_info);
                }
                module_info["parameters"] = param_names;
            }

            meta["modules"].push_back(module_info);
        }

        // Serialize JSON to string
        std::string json_str = meta.dump();

        // Open single .cyxmodel file
        std::ofstream file(file_path, std::ios::binary);
        if (!file) {
            spdlog::error("SequentialModel::Save: Failed to create file: {}", file_path);
            return false;
        }

        // Write header
        // Magic number: "CYXW" (4 bytes)
        const uint32_t magic = 0x43595857;
        file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));

        // Version: 2 for single-file format (4 bytes)
        const uint32_t version = 2;
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));

        // JSON length (8 bytes)
        uint64_t json_len = json_str.size();
        file.write(reinterpret_cast<const char*>(&json_len), sizeof(json_len));

        // JSON data
        file.write(json_str.c_str(), json_len);

        // Number of modules (8 bytes)
        size_t num_modules = modules_.size();
        file.write(reinterpret_cast<const char*>(&num_modules), sizeof(num_modules));

        // Write each module's parameters
        for (const auto& module : modules_) {
            auto params = module->GetParameters();
            size_t num_params = params.size();
            file.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));

            for (const auto& [name, tensor] : params) {
                // Write parameter name length and name
                size_t name_len = name.size();
                file.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
                file.write(name.c_str(), name_len);

                // Write tensor
                WriteTensor(file, tensor);
            }
        }

        file.close();
        spdlog::info("SequentialModel::Save: Saved model to {}", file_path);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("SequentialModel::Save: Exception: {}", e.what());
        return false;
    }
}

bool SequentialModel::Load(const std::string& path) {
    try {
        // Determine file path - add .cyxmodel if not present
        std::string file_path = path;
        if (file_path.size() < 9 || file_path.substr(file_path.size() - 9) != ".cyxmodel") {
            file_path += ".cyxmodel";
        }

        // Open the .cyxmodel file
        std::ifstream file(file_path, std::ios::binary);
        if (!file) {
            spdlog::error("SequentialModel::Load: Failed to open file: {}", file_path);
            return false;
        }

        // Read and verify magic number
        uint32_t magic;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != 0x43595857) {
            spdlog::error("SequentialModel::Load: Invalid magic number (not a CyxWiz model file)");
            return false;
        }

        // Read version
        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (version != 2) {
            spdlog::error("SequentialModel::Load: Unsupported format version: {} (expected 2)", version);
            return false;
        }

        // Read JSON length and data
        uint64_t json_len;
        file.read(reinterpret_cast<char*>(&json_len), sizeof(json_len));

        std::string json_str(json_len, '\0');
        file.read(json_str.data(), json_len);

        // Parse JSON metadata
        json meta = json::parse(json_str);
        if (meta.contains("metadata")) {
            model_name_ = meta["metadata"].value("name", "");
            model_description_ = meta["metadata"].value("description", "");
        }

        // Read number of modules
        size_t num_modules;
        file.read(reinterpret_cast<char*>(&num_modules), sizeof(num_modules));

        if (num_modules != modules_.size()) {
            spdlog::error("SequentialModel::Load: Module count mismatch. Expected {}, got {}",
                         modules_.size(), num_modules);
            return false;
        }

        // Load each module's parameters
        for (size_t i = 0; i < num_modules; ++i) {
            size_t num_params;
            file.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));

            std::map<std::string, Tensor> params;
            for (size_t j = 0; j < num_params; ++j) {
                // Read parameter name
                size_t name_len;
                file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
                std::string name(name_len, '\0');
                file.read(name.data(), name_len);

                // Read tensor
                Tensor tensor = ReadTensor(file);
                params[name] = std::move(tensor);
            }

            modules_[i]->SetParameters(params);
        }

        file.close();
        spdlog::info("SequentialModel::Load: Loaded model from {} ({} modules)", file_path, num_modules);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("SequentialModel::Load: Exception: {}", e.what());
        return false;
    }
}
}  // namespace cyxwiz

