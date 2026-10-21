#include "cyxmodel_archive.h"

#include <cyxwiz/utilities.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <system_error>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace cyxwiz::formats {
namespace {
namespace fs = std::filesystem;
constexpr uint32_t kMagic = 0x43595857;
constexpr uint32_t kVersion = 3;

void Require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error("CyxModel: " + message);
}

uint64_t ReadInteger(std::istream& input, size_t bytes) {
    uint64_t value = 0;
    for (size_t i = 0; i < bytes; ++i) {
        const int next = input.get();
        Require(next != std::char_traits<char>::eof(), "truncated integer field");
        value |= static_cast<uint64_t>(static_cast<uint8_t>(next)) << (8 * i);
    }
    return value;
}

void WriteInteger(std::ostream& output, uint64_t value, size_t bytes) {
    for (size_t i = 0; i < bytes; ++i) output.put(static_cast<char>((value >> (8 * i)) & 255));
}

std::string PortablePath(const std::string& name, const CyxModelArchiveLimits& limits) {
    Require(!name.empty() && name.size() <= limits.path_bytes, "invalid asset path length");
    Require(name.front() != '/' && name.back() != '/', "asset must be a relative file path");
    (void)nlohmann::json(name).dump(); // Strict UTF-8 validation of bounded asset names.
    std::string folded = name;
    for (char& c : folded) {
        const auto byte = static_cast<unsigned char>(c);
        Require(byte >= 32 && byte != 127 && std::string("\\:<>\"|?*").find(c) == std::string::npos,
                "unsafe asset path: " + name);
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + ('a' - 'A'));
    }
    size_t begin = 0;
    while (begin < folded.size()) {
        const auto end = folded.find('/', begin);
        const auto part = folded.substr(begin, end == std::string::npos ? end : end - begin);
        Require(!part.empty() && part != "." && part != ".." && part.back() != '.' && part.back() != ' ',
                "unsafe asset path component: " + name);
        const auto stem = part.substr(0, part.find('.'));
        const bool device = stem == "con" || stem == "prn" || stem == "aux" || stem == "nul" ||
            (stem.size() == 4 && (stem.substr(0,3) == "com" || stem.substr(0,3) == "lpt") &&
             stem[3] >= '1' && stem[3] <= '9');
        Require(!device, "reserved asset path: " + name);
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return folded;
}

void ValidateInventory(const CyxModelAssets& assets, const CyxModelArchiveLimits& limits) {
    Require(!assets.empty() && assets.size() <= limits.entries, "asset count exceeds limit or is empty");
    uint64_t total = 0;
    std::set<std::string> names;
    for (const auto& [name, data] : assets) {
        Require(data.size() <= limits.payload_bytes - total, "combined payload exceeds byte limit");
        total += data.size();
        Require(names.insert(PortablePath(name, limits)).second, "colliding asset paths");
    }
    for (const auto& name : names) {
        for (auto slash = name.find('/'); slash != std::string::npos; slash = name.find('/', slash + 1))
            Require(names.count(name.substr(0, slash)) == 0, "asset file/directory collision");
    }
}

std::string Digest(const std::string& name, const std::vector<uint8_t>& data) {
    std::string bound = name;
    bound.push_back('\0');
    bound.append(data.begin(), data.end());
    const auto hash = Utilities::HashText(bound, "sha256");
    Require(hash.success && hash.sha256_hash.size() == 64, "SHA-256 calculation failed");
    return hash.sha256_hash;
}

void ReadBytes(std::istream& input, char* output, size_t size) {
    Require(size <= static_cast<size_t>(std::numeric_limits<std::streamsize>::max()), "asset exceeds stream limit");
    if (size) input.read(output, static_cast<std::streamsize>(size));
    Require(static_cast<bool>(input), "truncated asset");
}

struct StagingDirectory {
    fs::path path;
    explicit StagingDirectory(const fs::path& destination) {
        const auto parent = fs::absolute(destination).parent_path();
        fs::create_directories(parent);
        for (int attempt = 0; attempt < 8; ++attempt) {
            const auto ids = Utilities::GenerateUUIDs(1);
            Require(ids.size() == 1, "could not allocate staging identifier");
            path = parent / (".cyxmodel-stage-" + ids.front());
            if (fs::create_directory(path)) return;
        }
        path.clear();
        throw std::runtime_error("CyxModel: could not allocate staging directory");
    }
    ~StagingDirectory() {
        if (!path.empty()) { std::error_code error; fs::remove_all(path, error); }
    }
    StagingDirectory(const StagingDirectory&) = delete;
    StagingDirectory& operator=(const StagingDirectory&) = delete;
};

void PublishFile(const fs::path& staged, const fs::path& destination) {
    Require(!fs::is_directory(destination) && !fs::is_symlink(fs::symlink_status(destination)),
            "binary destination is a directory or symlink; choose a file path");
#ifdef _WIN32
    if (!MoveFileExW(staged.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), "Publish CyxModel");
#else
    fs::rename(staged, destination);
#endif
}
} // namespace

bool CyxModelArchive::IsV3(const fs::path& path) {
    if (!fs::is_regular_file(path)) return false;
    std::ifstream input(path, std::ios::binary);
    try { return ReadInteger(input,4) == kMagic && ReadInteger(input,4) == kVersion; }
    catch (const std::exception&) { return false; }
}

CyxModelAssets CyxModelArchive::Read(const fs::path& path, const CyxModelArchiveLimits& limits) {
    CyxModelAssets assets;
    uint64_t total = 0;
    Require(!fs::is_symlink(fs::symlink_status(path)), "source symlink is not supported");
    if (fs::is_directory(path)) {
        for (const auto& entry : fs::recursive_directory_iterator(path)) {
            Require(!entry.is_symlink(), "package contains a symlink");
            if (entry.is_directory()) continue;
            Require(entry.is_regular_file(), "package contains a non-regular file");
            const auto name = fs::relative(entry.path(), path).generic_string();
            PortablePath(name, limits);
            Require(assets.size() < limits.entries, "asset count exceeds limit");
            const auto size = entry.file_size();
            Require(size <= limits.payload_bytes - total && size <= std::numeric_limits<size_t>::max(),
                    "combined payload exceeds byte limit");
            total += size;
            std::vector<uint8_t> data(static_cast<size_t>(size));
            std::ifstream input(entry.path(), std::ios::binary);
            ReadBytes(input, reinterpret_cast<char*>(data.data()), data.size());
            Require(input.peek() == std::char_traits<char>::eof(), "asset changed while reading");
            assets.emplace(name, std::move(data));
        }
    } else {
        Require(fs::is_regular_file(path), "source is not a model file or package folder");
        std::ifstream input(path, std::ios::binary);
        Require(ReadInteger(input,4) == kMagic, "invalid binary magic");
        Require(ReadInteger(input,4) == kVersion, "unsupported package version (requires CYXW v3)");
        Require(ReadInteger(input,4) == 0, "unsupported binary flags/compression");
        const auto count = ReadInteger(input,4);
        Require(count > 0 && count <= limits.entries, "asset count exceeds limit or is empty");
        const auto size = fs::file_size(path);
        uint64_t consumed = 16;
        for (uint64_t i = 0; i < count; ++i) {
            Require(size >= consumed && size - consumed >= 76, "truncated entry header");
            const auto name_size = ReadInteger(input,4);
            const auto data_size = ReadInteger(input,8);
            std::string expected(64,'\0');
            ReadBytes(input, expected.data(), expected.size());
            consumed += 76;
            Require(name_size > 0 && name_size <= limits.path_bytes && name_size <= size - consumed,
                    "invalid asset path length");
            consumed += name_size;
            Require(data_size <= size - consumed && data_size <= limits.payload_bytes - total &&
                    data_size <= std::numeric_limits<size_t>::max(), "truncated asset or payload exceeds byte limit");
            consumed += data_size;
            total += data_size;
            std::string name(static_cast<size_t>(name_size),'\0');
            ReadBytes(input,name.data(),name.size());
            PortablePath(name, limits);
            Require(assets.count(name) == 0, "duplicate asset path");
            std::vector<uint8_t> data(static_cast<size_t>(data_size));
            ReadBytes(input,reinterpret_cast<char*>(data.data()),data.size());
            Require(Digest(name,data) == expected, "asset SHA-256 mismatch: " + name);
            assets.emplace(name,std::move(data));
        }
        Require(consumed == size && input.peek() == std::char_traits<char>::eof(), "trailing archive bytes");
    }
    ValidateInventory(assets,limits);
    return assets;
}

void CyxModelArchive::WriteBinary(const fs::path& path, const CyxModelAssets& assets,
                                const CyxModelArchiveLimits& limits) {
    ValidateInventory(assets,limits);
    Require(!fs::is_directory(path), "binary destination is a directory; choose a new file path");
    StagingDirectory stage(path);
    const auto staged = stage.path / "package";
    std::ofstream output(staged,std::ios::binary);
    Require(static_cast<bool>(output), "could not open staged binary output");
    WriteInteger(output,kMagic,4); WriteInteger(output,kVersion,4);
    WriteInteger(output,0,4); WriteInteger(output,assets.size(),4);
    for (const auto& [name,data] : assets) {
        WriteInteger(output,name.size(),4); WriteInteger(output,data.size(),8);
        const auto digest = Digest(name,data);
        output.write(digest.data(),digest.size()); output.write(name.data(),name.size());
        if (!data.empty()) output.write(reinterpret_cast<const char*>(data.data()),data.size());
        Require(static_cast<bool>(output), "failed writing binary asset: " + name);
    }
    output.flush(); Require(static_cast<bool>(output), "failed flushing binary output");
    output.close(); Require(!output.fail(), "failed closing binary output");
    PublishFile(staged,fs::absolute(path));
}

void CyxModelArchive::WriteDirectory(const fs::path& path, const CyxModelAssets& assets,
                                   const CyxModelArchiveLimits& limits) {
    ValidateInventory(assets,limits);
    Require(!fs::exists(path), "directory destination exists; choose a new package path");
    StagingDirectory stage(path);
    for (const auto& [name,data] : assets) {
        const auto target = stage.path / fs::path(std::u8string(name.begin(),name.end()));
        fs::create_directories(target.parent_path());
        Require(!fs::exists(target), "asset path collision on destination filesystem: " + name);
        std::ofstream output(target,std::ios::binary);
        if (!data.empty()) output.write(reinterpret_cast<const char*>(data.data()),data.size());
        output.flush(); Require(static_cast<bool>(output), "failed writing directory asset: " + name);
        output.close(); Require(!output.fail(), "failed closing directory asset: " + name);
    }
    fs::rename(stage.path,fs::absolute(path));
    stage.path.clear();
}
} // namespace cyxwiz::formats
