#include "model_converter.h"
#include "formats/cyxmodel_archive.h"
#include "formats/cyxmodel_format.h"

#include <spdlog/spdlog.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace cyxwiz {
namespace fs = std::filesystem;
std::string ModelConverter::last_error_;

namespace {
void ValidatePackage(const formats::CyxModelAssets& assets) {
    formats::CyxModelFormat format;
    ModelManifest manifest;
    TrainingConfig config;
    std::string graph;
    std::map<std::string,std::vector<uint8_t>> weights;
    std::map<std::string,std::vector<int64_t>> shapes;
    if (!format.Extract(assets,manifest,graph,config,nullptr,weights,shapes,nullptr,{}))
        throw std::runtime_error(format.GetLastError());

}
}

bool ModelConverter::IsBinaryFormat(const std::string& path) {
    if (!fs::is_regular_file(path)) return false;
    std::ifstream input(path,std::ios::binary);
    unsigned char bytes[4]{};
    return input.read(reinterpret_cast<char*>(bytes),4) &&
        bytes[0]==0x57 && bytes[1]==0x58 && bytes[2]==0x59 && bytes[3]==0x43;
}

bool ModelConverter::IsDirectoryFormat(const std::string& path) {
    return fs::is_directory(path) && fs::is_regular_file(fs::path(path)/"manifest.json");
}

bool ModelConverter::BinaryToDirectory(const std::string& input, const std::string& output,
                                       ProgressCallback progress) {
    last_error_.clear();
    try {
        if (!formats::CyxModelArchive::IsV3(input))
            throw std::runtime_error("Faithful conversion requires a CYXW v3 package. Legacy CYXW v2 has no complete package inventory; re-export the original trained model or convert its original directory package.");
        if (progress) progress(0,2,"Validating binary model package...");
        const auto assets=formats::CyxModelArchive::Read(input);
        ValidatePackage(assets);
        if (progress) progress(1,2,"Writing directory package...");
        formats::CyxModelArchive::WriteDirectory(output,assets);
        if (progress) progress(2,2,"Conversion complete");
        spdlog::info("ModelConverter: CYXW v3 to directory, {} assets: {} -> {}",assets.size(),input,output);
        return true;
    } catch(const std::exception& e) {
        last_error_=e.what(); spdlog::error("ModelConverter: {}",last_error_); return false;
    }
}

bool ModelConverter::DirectoryToBinary(const std::string& input, const std::string& output,
                                       ProgressCallback progress) {
    last_error_.clear();
    try {
        if (!IsDirectoryFormat(input)) throw std::runtime_error("Select a model package folder containing manifest.json");
        if (progress) progress(0,2,"Validating directory model package...");
        const auto assets=formats::CyxModelArchive::Read(input);
        ValidatePackage(assets);
        if (progress) progress(1,2,"Writing CYXW v3 binary package...");
        formats::CyxModelArchive::WriteBinary(output,assets);
        if (progress) progress(2,2,"Conversion complete");
        spdlog::info("ModelConverter: directory to CYXW v3, {} assets: {} -> {}",assets.size(),input,output);
        return true;
    } catch(const std::exception& e) {
        last_error_=e.what(); spdlog::error("ModelConverter: {}",last_error_); return false;
    }
}
} // namespace cyxwiz
