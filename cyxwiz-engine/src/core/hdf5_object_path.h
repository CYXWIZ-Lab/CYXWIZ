#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace cyxwiz {

inline bool ValidateHdf5ObjectPath(const std::string& path,
                                   std::string& error,
                                   bool allow_root = false) {
    error.clear();
    if (allow_root && path == "/") return true;
    if (path.size() < 2 || path.size() > 4096 || path.front() != '/' ||
        path.back() == '/' || path.find('\0') != std::string::npos) {
        error = "Select an absolute HDF5 object path of at most 4096 bytes";
        return false;
    }
    for (std::size_t begin = 1; begin < path.size();) {
        auto end = path.find('/', begin);
        if (end == std::string::npos) end = path.size();
        const std::string_view component(path.data() + begin, end - begin);
        if (component.empty() || component == "." || component == "..") {
            error = "Invalid HDF5 dataset path: '" + path + "'";
            return false;
        }
        begin = end + 1;
    }
    return true;
}

} // namespace cyxwiz
