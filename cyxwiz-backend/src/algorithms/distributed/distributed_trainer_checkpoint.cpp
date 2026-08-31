#include "cyxwiz/distributed/distributed_trainer.h"

#include <spdlog/spdlog.h>

#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {
// ========== Checkpointing ==========

void DistributedTrainer::SaveCheckpoint(const std::string& path) {
    // Only master saves by default
    if (!IsMaster()) {
        return;
    }

    spdlog::info("Saving checkpoint to: {}", path);

    // Get model parameters
    auto params = model_->GetParameters();

    // Simple binary format: num_params, then for each: name_len, name, data_size, data
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        spdlog::error("Failed to open checkpoint file: {}", path);
        return;
    }

    size_t num_params = params.size();
    file.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));

    for (const auto& [name, tensor] : params) {
        size_t name_len = name.size();
        file.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        file.write(name.c_str(), name_len);

        size_t data_size = tensor.NumElements();
        file.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));
        file.write(reinterpret_cast<const char*>(tensor.ReadData<float>()),
                   data_size * sizeof(float));
    }

    spdlog::info("Checkpoint saved: {} parameters", num_params);
}

void DistributedTrainer::LoadCheckpoint(const std::string& path) {
    spdlog::info("Loading checkpoint from: {}", path);

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        spdlog::error("Failed to open checkpoint file: {}", path);
        return;
    }

    size_t num_params;
    file.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));

    std::map<std::string, Tensor> params;

    for (size_t i = 0; i < num_params; ++i) {
        size_t name_len;
        file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));

        std::string name(name_len, '\0');
        file.read(&name[0], name_len);

        size_t data_size;
        file.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));

        std::vector<size_t> shape = {data_size};
        Tensor tensor(shape);
        file.read(reinterpret_cast<char*>(tensor.MutableData<float>()),
                  data_size * sizeof(float));

        params[name] = std::move(tensor);
    }

    model_->SetParameters(params);

    // Sync parameters across all ranks
    if (ddp_) {
        ddp_->BroadcastParameters(0);
    }

    spdlog::info("Checkpoint loaded: {} parameters", num_params);
}
}  // namespace cyxwiz
