#include "fasttext_model_store.h"

#include <filesystem>
#include <map>
#include <mutex>

namespace cyxwiz {

namespace {

std::mutex& StoreMutex() {
    static std::mutex mutex;
    return mutex;
}

std::map<std::string, std::shared_ptr<const FastTextModel>>& Store() {
    static std::map<std::string, std::shared_ptr<const FastTextModel>> store;
    return store;
}

// One key per file however its path is written.
std::string Key(const std::string& path) {
    std::error_code error;
    const auto canonical = std::filesystem::weakly_canonical(std::filesystem::path(path), error);
    return (error ? std::filesystem::path(path).lexically_normal() : canonical).generic_string();
}

}  // namespace

void KeepFastTextModel(const std::string& vectors_path, std::shared_ptr<const FastTextModel> model) {
    if (vectors_path.empty() || !model) return;
    std::lock_guard<std::mutex> lock(StoreMutex());
    Store()[Key(vectors_path)] = std::move(model);
}

std::shared_ptr<const FastTextModel> FindFastTextModel(const std::string& vectors_path) {
    if (vectors_path.empty()) return nullptr;
    std::lock_guard<std::mutex> lock(StoreMutex());
    const auto it = Store().find(Key(vectors_path));
    return it == Store().end() ? nullptr : it->second;
}

}  // namespace cyxwiz
