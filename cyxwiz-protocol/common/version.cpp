#include "version.h"

namespace cyxwiz {
namespace protocol {

constexpr int VERSION_MAJOR = 0;
constexpr int VERSION_MINOR = 1;
constexpr int VERSION_PATCH = 0;
constexpr const char* VERSION_BUILD = "alpha";

std::string GetVersionString() {
    return std::to_string(VERSION_MAJOR) + "." +
           std::to_string(VERSION_MINOR) + "." +
           std::to_string(VERSION_PATCH) + "-" +
           VERSION_BUILD;
}

Version GetVersion() {
    Version version;
    version.set_major(VERSION_MAJOR);
    version.set_minor(VERSION_MINOR);
    version.set_patch(VERSION_PATCH);
    version.set_build(VERSION_BUILD);
    return version;
}

} // namespace protocol
} // namespace cyxwiz
