#include "backend_pack_path.h"

#include <algorithm>
#include <cctype>

namespace cyxwiz::runtime {

bool IsCanonicalBackendPackRelativePath(std::string_view value) {
    if (value.empty() || value.front() == '/' || value.back() == '/' ||
        value.find('\\') != std::string_view::npos ||
        value.find(':') != std::string_view::npos) {
        return false;
    }
    std::size_t begin = 0;
    while (begin < value.size()) {
        const auto end = value.find('/', begin);
        const auto part = value.substr(
            begin,
            (end == std::string_view::npos ? value.size() : end) - begin);
        if (part.empty() || part == "." || part == "..") return false;
        begin = end == std::string_view::npos ? value.size() : end + 1;
    }
    return true;
}

std::string FoldBackendPackPath(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return value;
}

std::filesystem::path BackendPackNativeRelativePath(std::string_view value) {
    const std::u8string utf8(
        reinterpret_cast<const char8_t*>(value.data()), value.size());
    return std::filesystem::path(utf8);
}

}  // namespace cyxwiz::runtime
